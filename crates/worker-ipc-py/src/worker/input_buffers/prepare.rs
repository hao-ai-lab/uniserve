//! Select ordinary or resident decode inputs and retain output row controls.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use uniserve_worker_ipc::{CallKind, ForwardMode, MediaCall};

use super::{InputBuffers, copy, fill, numerical, prefix};
use crate::worker::block_tables::pages_to_py;
use crate::worker::decode_state::DecodeState;
use crate::worker::model_inputs::{CanvasStepRow, InputBatch, InputRow, Row, TokenRow};

impl InputBuffers {
    pub(in crate::worker) fn validate(&self, rows: &Bound<'_, PyTuple>) -> PyResult<()> {
        self.open()?;
        if rows.is_empty() || rows.len() > self.max_rows {
            return Err(PyValueError::new_err(
                "row count exceeds input-buffer capacity",
            ));
        }
        let mut tokens = 0;
        for row in rows {
            let row = Row::borrow(&row)?;
            let matches = match (&row, self.kind) {
                (Row::Token(_), CallKind::Forward(mode)) => mode != ForwardMode::TokenDenoising,
                (
                    Row::Canvas(_) | Row::CanvasStep(_),
                    CallKind::Forward(ForwardMode::TokenDenoising),
                ) => true,
                (Row::Diffusion(_), CallKind::Media(MediaCall::Denoising)) => true,
                (
                    Row::Vision(_),
                    CallKind::Media(MediaCall::VisionEncoding | MediaCall::LatentEncoding),
                ) => true,
                (Row::Decode(_), CallKind::Media(MediaCall::ImageDecoding)) => true,
                _ => false,
            };
            if !matches {
                return Err(PyTypeError::new_err(
                    "input rows do not match this computation",
                ));
            }
            if !self.table_widths.is_empty() {
                tokens += row.query_tokens(rows.py())?;
            }
        }
        if tokens > self.max_tokens {
            return Err(PyValueError::new_err(
                "query tokens exceed input-buffer capacity",
            ));
        }
        if self.max_text_tokens != 0 && tokens > self.max_text_tokens {
            return Err(PyValueError::new_err(
                "text token count exceeds input-buffer capacity",
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::worker) fn prepare(
        slf: &Bound<'_, Self>,
        rows: &Bound<'_, PyTuple>,
        mode: CallKind,
        attention: Option<&Bound<'_, PyAny>>,
        cache: Option<&Bound<'_, PyAny>>,
        tables: Option<&Bound<'_, PyAny>>,
        states: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        slf.borrow().validate(rows)?;
        let mut slots = Vec::with_capacity(rows.len());
        for row in rows {
            let row = row.cast::<InputRow>()?.borrow();
            let kind = row.kind;
            if kind != mode && !matches!((kind, mode), (CallKind::Forward(_), CallKind::Forward(_)))
            {
                return Err(PyValueError::new_err(
                    "one input call requires homogeneous computations",
                ));
            }
            slots.push(i64::from(row.request_pool_idx));
        }
        let (ring, requests, kind) = {
            let owner = slf.borrow();
            (
                owner.requests.clone_ref(py),
                prefix(&owner.column(py, "request_pool_indices")?, rows.len())?,
                owner.kind,
            )
        };
        let (slot, host) = ring.borrow(py).acquire(py)?;
        fill(host.bind(py), &slots)?;
        copy(&requests, &prefix(host.bind(py), slots.len())?)?;
        ring.borrow(py).record_copy(py, slot)?;

        let functions = numerical(py)?;
        let mut selections = Vec::new();
        let mut finish = py.None();
        let inputs = match kind {
            CallKind::Forward(ForwardMode::TokenDenoising) => {
                let steps = rows
                    .iter()
                    .map(|row| row.is_instance_of::<CanvasStepRow>())
                    .collect::<Vec<_>>();
                if steps.iter().any(|value| *value) && !steps.iter().all(|value| *value) {
                    return Err(PyValueError::new_err(
                        "one canvas call contains readout rows or canvas steps",
                    ));
                }
                let attention = Self::attention(slf, rows, attention, cache, tables)?;
                if steps[0] {
                    Self::canvas_steps(slf, rows, attention.bind(py))?
                } else {
                    Self::canvas_readout(slf, rows, attention.bind(py))?
                }
            }
            CallKind::Forward(_) => {
                selections = rows
                    .iter()
                    .map(|row| {
                        row.cast::<TokenRow>()?.borrow().selection.ok_or_else(|| {
                            PyValueError::new_err("text input requires an output selection")
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                finish = Self::finish_column(slf, rows)?;
                Self::tokens(slf, rows, attention, cache, tables, states)?
            }
            CallKind::Media(MediaCall::Denoising) => {
                let attention = Self::attention(slf, rows, attention, cache, tables)?;
                functions
                    .call_method1("_images", (slf, rows, attention))?
                    .unbind()
            }
            CallKind::Media(MediaCall::VisionEncoding | MediaCall::LatentEncoding) => {
                functions.call_method1("_vision", (slf, rows))?.unbind()
            }
            CallKind::Media(MediaCall::ImageDecoding) => {
                functions.call_method1("_decode", (slf, rows))?.unbind()
            }
            _ => unreachable!("construction selects a buffered computation"),
        };
        Ok(Py::new(
            py,
            InputBatch::build(
                py,
                mode,
                inputs,
                requests.unbind(),
                selections,
                if finish.is_none(py) {
                    None
                } else {
                    Some(finish)
                },
            )?,
        )?
        .into_any())
    }

    fn finish_column(slf: &Bound<'_, Self>, rows: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let mut values = Vec::with_capacity(rows.len());
        for row in rows {
            let row = row.cast::<TokenRow>()?.borrow();
            if row.decode_predicate.is_none() || !row.decode_predicate_tagged {
                return Ok(py.None());
            }
            values.push(u8::from(row.decode_force_finish));
        }
        let (ring, destination) = {
            let owner = slf.borrow();
            (
                owner
                    .finish
                    .as_ref()
                    .ok_or_else(|| PyValueError::new_err("token inputs require finish storage"))?
                    .clone_ref(py),
                prefix(&owner.column(py, "decode_force_finish")?, rows.len())?,
            )
        };
        let (slot, host) = ring.borrow(py).acquire(py)?;
        // torch.bool has one byte per value; its byte view exposes the same
        // CPU allocation without converting or allocating another column.
        let bytes = host
            .bind(py)
            .call_method1("view", (py.import("torch")?.getattr("uint8")?,))?;
        fill(&bytes, &values)?;
        copy(&destination, &prefix(host.bind(py), values.len())?)?;
        ring.borrow(py).record_copy(py, slot)?;
        Ok(destination.unbind())
    }

    fn tokens(
        slf: &Bound<'_, Self>,
        rows: &Bound<'_, PyTuple>,
        attention: Option<&Bound<'_, PyAny>>,
        cache: Option<&Bound<'_, PyAny>>,
        tables: Option<&Bound<'_, PyAny>>,
        states: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let indexed = rows
            .iter()
            .map(|row| Ok(row.cast::<TokenRow>()?.borrow().request_indexed_decode))
            .collect::<PyResult<Vec<_>>>()?;
        let mut resolved = rows.clone();
        if indexed.iter().any(|value| *value) {
            let invalid =
                || PyValueError::new_err("indexed decode requires valid resident request slots");
            let states = states.ok_or_else(invalid)?;
            let resident = states.cast::<DecodeState>()?.borrow();
            let capacity = resident.request_pool_size as i64;
            for (row, indexed) in rows.iter().zip(&indexed) {
                if !indexed {
                    continue;
                }
                let row = row.cast::<TokenRow>()?.borrow();
                let attention = row.as_super();
                let input = attention.as_super();
                let slot = i64::from(input.request_pool_idx);
                if input.kind != CallKind::Forward(ForwardMode::Decode)
                    || slot <= 0
                    || slot > capacity
                    || row.selection.is_none()
                    || row.token_ids.is_some()
                    || attention.positions.is_some()
                    || row.token_embeddings.is_some()
                    || row.token_embedding_mask.is_some()
                {
                    return Err(invalid());
                }
            }

            if attention.is_none()
                && indexed.iter().all(|value| *value)
                && let (Some(cache), Some(tables)) = (cache, tables)
            {
                let device = slf.borrow().device.clone_ref(py);
                if device.bind(py).getattr("type")?.extract::<String>()? == "cuda"
                    && resident.device.bind(py).eq(device.bind(py))?
                    && tables
                        .getattr("unit_tables")?
                        .getattr("device")?
                        .eq(device.bind(py))?
                {
                    let pages = Self::pages(rows, Some(cache))?;
                    let capacities = &slf.borrow().table_widths;
                    let mut widths = Vec::with_capacity(pages.len());
                    for (number, table) in pages.iter().enumerate() {
                        let width = table.width();
                        let capacity = capacities.get(number).copied().unwrap_or(0);
                        if width > capacity {
                            return Err(PyValueError::new_err(
                                "block tables exceed input-buffer capacity",
                            ));
                        }
                        widths.push(width.next_power_of_two().min(capacity));
                    }
                    return Ok(numerical(py)?
                        .call_method1(
                            "_indexed",
                            (
                                slf,
                                rows,
                                pages_to_py(py, pages)?,
                                tables,
                                states,
                                PyTuple::new(py, widths)?,
                            ),
                        )?
                        .unbind());
                }
            }
            resolved = numerical(py)?
                .call_method1("_resident_rows", (rows, states))?
                .cast_into::<PyTuple>()?;
        }
        let attention = Self::attention(slf, &resolved, attention, cache, tables)?;
        Ok(numerical(py)?
            .call_method1("_text", (slf, resolved, attention))?
            .unbind())
    }
}
