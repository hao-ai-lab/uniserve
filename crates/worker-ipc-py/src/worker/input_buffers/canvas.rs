//! Canvas readout packing and resident sampler input ownership.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::{InputBuffers, copy, host_tensor, numerical, prefix, with_host};
use crate::worker::host::with_context;
use crate::worker::tensor_buffers::TensorBuffers;

impl InputBuffers {
    pub(super) fn bind_canvas(slf: &Bound<'_, Self>, slots: Py<PyAny>) -> PyResult<()> {
        let py = slf.py();
        let (device, rows) = {
            let owner = slf.borrow();
            owner.open()?;
            if owner.steps.is_none() {
                return Err(PyValueError::new_err(
                    "canvas state requires canvas input buffers",
                ));
            }
            (
                owner.device.clone_ref(py),
                owner
                    .max_rows
                    .min(slots.getattr(py, "request_pool_size")?.extract(py)?),
            )
        };
        let history = slots.getattr(py, "history_depth")?.extract::<usize>(py)?;
        let fields = sampler_fields(slots.bind(py), rows, history)?;
        let backing = Py::new(
            py,
            TensorBuffers::allocate(py, &fields, device.bind(py), false, None)?,
        )?;
        let mut owner = slf.borrow_mut();
        owner.canvas_backing = Some(backing);
        owner.canvas_slots = slots;
        Ok(())
    }

    pub(super) fn canvas_readout(
        slf: &Bound<'_, Self>,
        rows: &Bound<'_, PyTuple>,
        attention: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let mut slots = Vec::new();
        let mut groups = Vec::new();
        let mut row_candidates = Vec::with_capacity(rows.len());
        let mut offset = 0;
        for row in rows {
            let tokens = row.getattr("slot_tokens")?.extract::<Vec<usize>>()?;
            let offsets = row.getattr("candidate_offsets")?.extract::<Vec<usize>>()?;
            let candidates = row.getattr("candidate_ids")?.extract::<Vec<i64>>()?;
            row_candidates.push(candidates.len());
            for (index, token) in tokens.into_iter().enumerate() {
                slots.push((offset + token) as i64);
                groups.push(candidates[offsets[index]..offsets[index + 1]].to_vec());
            }
            offset += row.getattr("query_tokens")?.extract::<usize>()?;
        }
        if slots.len() > slf.borrow().max_tokens {
            return Err(PyValueError::new_err(
                "canvas readout slots exceed input-buffer capacity",
            ));
        }

        // The padded matrix and its real-element selection share one copy.
        // Padding repeats a candidate only to give every slot equal width;
        // the selection returns each original candidate once, in row order.
        let width = groups.iter().map(Vec::len).max().unwrap_or(0);
        let matrix = groups.len() * width;
        let size = matrix + row_candidates.iter().sum::<usize>();
        let mut values = Vec::with_capacity(size);
        for group in &groups {
            values.extend_from_slice(group);
            values.resize(values.len() + width - group.len(), group[0]);
        }
        for (index, group) in groups.iter().enumerate() {
            values.extend((index * width..index * width + group.len()).map(|value| value as i64));
        }
        Self::reserve_candidates(slf, values.len())?;
        Ok(numerical(py)?
            .call_method1(
                "_readout",
                (
                    slf,
                    rows,
                    attention,
                    host_tensor(py, &slots)?,
                    host_tensor(py, &values)?,
                    width,
                    PyTuple::new(py, row_candidates)?,
                ),
            )?
            .unbind())
    }

    fn reserve_candidates(slf: &Bound<'_, Self>, size: usize) -> PyResult<()> {
        let py = slf.py();
        let (current, device) = {
            let owner = slf.borrow();
            (
                owner
                    .candidate_storage
                    .call_method0(py, "numel")?
                    .extract::<usize>(py)?,
                owner.device.clone_ref(py),
            )
        };
        if current >= size {
            return Ok(());
        }

        // Readout executes outside graphs on this lane's stream. The caching
        // allocator orders replaced backing after prior consumers. Warmup may
        // run under inference mode, but serving must be able to write it later.
        let torch = py.import("torch")?;
        let scope = torch.call_method1("inference_mode", (false,))?;
        let storage = with_context(&scope, || {
            let options = PyDict::new(py);
            options.set_item("dtype", torch.getattr("int64")?)?;
            options.set_item("device", device)?;
            torch.call_method("empty", (size,), Some(&options))
        })?;
        slf.borrow_mut().candidate_storage = storage.unbind();
        Ok(())
    }

    pub(super) fn canvas_steps(
        slf: &Bound<'_, Self>,
        rows: &Bound<'_, PyTuple>,
        attention: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let (state, ring, backing, destination, capacity) = {
            let owner = slf.borrow();
            let backing = owner
                .canvas_backing
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("canvas steps require bound sampler state"))?;
            let ring = owner
                .steps
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("canvas steps require step storage"))?;
            (
                owner.canvas_slots.clone_ref(py),
                ring.clone_ref(py),
                backing.clone_ref(py),
                owner.column(py, "step_columns")?,
                owner.max_rows,
            )
        };
        let length = state.getattr(py, "canvas_length")?.extract::<usize>(py)?;
        let mut sampling = Vec::with_capacity(rows.len());
        let mut history = None;
        let mut first = true;
        let size = 3 * capacity + rows.len();
        let (slot, host) = ring.borrow(py).acquire(py)?;
        with_host::<i64>(host.bind(py), size, |values| {
            for (index, row) in rows.iter().enumerate() {
                let constants = row.getattr("sampling")?;
                let depth = constants.getattr("stability")?.extract::<usize>()?;
                if history.is_some_and(|value| value != depth)
                    || row.getattr("canvas_length")?.extract::<usize>()? != length
                {
                    return Err(PyValueError::new_err(
                        "canvas steps of one call share the resident canvas length and one stability threshold",
                    ));
                }
                history = Some(depth);
                sampling.push(constants);
                values[index].set(row.getattr("request_pool_idx")?.extract()?);
                values[capacity + index].set(row.getattr("seed")?.extract()?);
                values[2 * capacity + index].set(row.getattr("block")?.extract()?);
                let step = row.getattr("step")?.extract::<i64>()?;
                values[3 * capacity + index].set(step);
                first &= step == 0;
            }
            Ok(())
        })?;
        copy(
            &prefix(&destination.call_method1("view", (-1,))?, size)?,
            &prefix(&host.bind(py).call_method1("view", (-1,))?, size)?,
        )?;
        ring.borrow(py).record_copy(py, slot)?;
        let fields = sampler_fields(state.bind(py), rows.len(), history.unwrap_or(0))?;
        let views = backing.borrow(py).view(py, &fields)?;
        Ok(numerical(py)?
            .call_method1(
                "_steps",
                (
                    slf,
                    rows,
                    attention,
                    views,
                    PyTuple::new(py, sampling)?,
                    first,
                ),
            )?
            .unbind())
    }
}

fn sampler_fields<'py>(
    state: &Bound<'py, PyAny>,
    rows: usize,
    history: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let py = state.py();
    let options = PyDict::new(py);
    options.set_item("max_rows", rows)?;
    options.set_item("canvas_length", state.getattr("canvas_length")?)?;
    options.set_item("hidden_size", state.getattr("hidden_size")?)?;
    options.set_item("history_depth", history)?;
    options.set_item(
        "dtype",
        state
            .getattr("banks")?
            .get_item("self_conditioning")?
            .getattr("dtype")?,
    )?;
    numerical(py)?.call_method("sampler_buffers", (), Some(&options))
}
