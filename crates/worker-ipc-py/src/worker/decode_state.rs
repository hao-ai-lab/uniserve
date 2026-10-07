//! Persistent decode tensors and request-slot update selection.

use std::collections::HashSet;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};

use super::tensor_buffers::TensorBuffers;

/// Slot zero is inactive padding. Verified cache lengths may be borrowed
/// from BlockTables; every other field lives until this owner and its views retire.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct DecodeState {
    #[pyo3(get)]
    pub(super) request_pool_size: usize,
    #[pyo3(get)]
    pub(super) vocab_size: usize,
    #[pyo3(get)]
    pub(super) continuation_width: usize,
    #[pyo3(get)]
    pub(super) device: Py<PyAny>,
    #[pyo3(get)]
    logits_dtype: Py<PyAny>,
    backing: Py<TensorBuffers>,
    fused: bool,
}

#[pymethods]
impl DecodeState {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (*, request_pool_size, vocab_size, continuation_width, device, logits_dtype=None, valid_cache_lengths=None))]
    pub(super) fn new(
        py: Python<'_>,
        request_pool_size: usize,
        vocab_size: usize,
        continuation_width: usize,
        device: &Bound<'_, PyAny>,
        logits_dtype: Option<&Bound<'_, PyAny>>,
        valid_cache_lengths: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let torch = py.import("torch")?;
        let dtype = match logits_dtype {
            Some(dtype) => dtype.clone(),
            None => torch.getattr("float32")?,
        };
        let fields = Self::buffers(
            py,
            request_pool_size,
            vocab_size,
            continuation_width,
            &dtype,
        )?;
        let device = torch.call_method1("device", (device,))?;
        let backing = TensorBuffers::allocate(py, &fields, &device, false, None)?;
        let rows = request_pool_size + 1;
        let lengths = if let Some(lengths) = valid_cache_lengths {
            if lengths.getattr("shape")?.extract::<Vec<usize>>()? != [rows]
                || !lengths.getattr("dtype")?.eq(torch.getattr("int32")?)?
                || !lengths.getattr("device")?.eq(&device)?
            {
                return Err(PyValueError::new_err(
                    "runtime cache-length storage is incompatible",
                ));
            }
            lengths.clone()
        } else {
            let options = PyDict::new(py);
            options.set_item("dtype", torch.getattr("int32")?)?;
            options.set_item("device", &device)?;
            torch.call_method("zeros", (rows,), Some(&options))?
        };
        backing
            .tensors
            .bind(py)
            .set_item("valid_cache_lengths", lengths)?;

        // Prompt logits are first written by prompt scoring. Other fields
        // start at the values assigned when a request slot is reset.
        for (name, value) in [
            ("logical_lengths", 0),
            ("sampling_positions", 0),
            ("future_input_tokens", 1),
            ("penalty_counts", 0),
            ("predicates", 0),
            ("_ones_int32", 1),
            ("_ones_int64", 1),
        ] {
            backing
                .backing(py, name)?
                .call_method1(py, "fill_", (value,))?;
        }
        let fused = py
            .import("uniserve_kernels.triton")?
            .call_method1("launchable", (&device,))?
            .is_truthy()?;
        let state = Self {
            request_pool_size,
            vocab_size,
            continuation_width,
            device: device.unbind(),
            logits_dtype: dtype.unbind(),
            backing: Py::new(py, backing)?,
            fused,
        };
        if fused {
            numerical(py)?.call_method1(
                "warmup",
                (
                    state.tensors(py),
                    request_pool_size,
                    continuation_width,
                    vocab_size,
                ),
            )?;
        }
        Ok(state)
    }

    /// Describe resident capacity without allocating it. BlockTables charges
    /// the shared verified-length column separately during startup sizing.
    #[staticmethod]
    #[pyo3(signature = (*, request_pool_size, vocab_size, continuation_width, logits_dtype))]
    pub(super) fn buffers<'py>(
        py: Python<'py>,
        request_pool_size: usize,
        vocab_size: usize,
        continuation_width: usize,
        logits_dtype: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if request_pool_size == 0 || vocab_size == 0 || continuation_width == 0 {
            return Err(PyValueError::new_err(
                "runtime-state dimensions must be positive",
            ));
        }
        if !logits_dtype.getattr("is_floating_point")?.is_truthy()? {
            return Err(PyValueError::new_err(
                "runtime prompt-logit dtype must be floating point",
            ));
        }
        let rows = request_pool_size + 1;
        let torch = py.import("torch")?;
        let int32 = torch.getattr("int32")?;
        let int64 = torch.getattr("int64")?;
        let boolean = torch.getattr("bool")?;
        let config = py.import("uniserve.tensors")?.getattr("BufferConfig")?;
        let fields = PyDict::new(py);
        for (name, shape, dtype) in [
            ("logical_lengths", vec![rows], &int32),
            ("sampling_positions", vec![rows], &int64),
            (
                "future_input_tokens",
                vec![rows, continuation_width],
                &int64,
            ),
            ("penalty_counts", vec![rows, vocab_size], &int32),
            ("prompt_logits", vec![rows, vocab_size], logits_dtype),
            ("predicates", vec![rows], &boolean),
            ("_ones_int32", vec![request_pool_size], &int32),
            ("_ones_int64", vec![request_pool_size], &int64),
        ] {
            fields.set_item(name, config.call1((PyTuple::new(py, shape)?, dtype))?)?;
        }
        Ok(fields)
    }

    #[getter]
    pub(super) fn valid_cache_lengths(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.backing.borrow(py).backing(py, "valid_cache_lengths")
    }

    #[getter]
    pub(super) fn logical_lengths(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.backing.borrow(py).backing(py, "logical_lengths")
    }

    #[getter]
    pub(super) fn sampling_positions(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.backing.borrow(py).backing(py, "sampling_positions")
    }

    #[getter]
    pub(super) fn future_input_tokens(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.backing.borrow(py).backing(py, "future_input_tokens")
    }

    #[getter]
    pub(super) fn penalty_counts(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.backing.borrow(py).backing(py, "penalty_counts")
    }

    #[getter]
    pub(super) fn prompt_logits(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.backing.borrow(py).backing(py, "prompt_logits")
    }

    #[getter]
    pub(super) fn predicates(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.backing.borrow(py).backing(py, "predicates")
    }

    /// Reset real slots, coalescing host columns into one copy and kernel.
    /// Device-valued columns use indexed updates and never require scalar readback.
    #[pyo3(signature = (request_pool_indices, *, valid_cache_lengths=None, logical_lengths=None, sampling_positions=None))]
    pub(super) fn reset(
        &self,
        py: Python<'_>,
        request_pool_indices: &Bound<'_, PyAny>,
        valid_cache_lengths: Option<&Bound<'_, PyAny>>,
        logical_lengths: Option<&Bound<'_, PyAny>>,
        sampling_positions: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let host = host_values(request_pool_indices)?;
        if let Some(host) = &host {
            self.check_slots(host)?;
            if host.is_empty() {
                return Ok(());
            }
            if self.fused {
                let columns = [valid_cache_lengths, logical_lengths, sampling_positions]
                    .map(|column| host_column(column, host.len()));
                let [valid, logical, sampling] = columns;
                if let [Some(valid), Some(logical), Some(sampling)] = [valid?, logical?, sampling?]
                {
                    self.reset_host(py, host, [Some(&valid), Some(&logical), Some(&sampling)])?;
                    return Ok(());
                }
            }
        }

        let indices = if let Some(host) = host {
            self.host_tensor(py, &host)?
        } else {
            let indices = numerical(py)?.call_method1(
                "device_indices",
                (request_pool_indices, &self.device, self.request_pool_size),
            )?;
            // A CPU state can receive device indices. Its required copy has
            // completed here, so check the copied values before touching rows.
            if let Some(host) = host_values(&indices)? {
                self.check_slots(&host)?;
            }
            indices
        };
        numerical(py)?.call_method1(
            "reset_indexed",
            (
                self.tensors(py),
                indices,
                valid_cache_lengths,
                logical_lengths,
                sampling_positions,
            ),
        )?;
        Ok(())
    }

    /// Set a verified extent on the caller's stream, borrowing device scalars.
    pub(super) fn set_cache_length(
        &self,
        py: Python<'_>,
        slot: i64,
        length: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.check_slots(&[slot])?;
        let target = self
            .valid_cache_lengths(py)?
            .bind(py)
            .get_item(PySlice::new(py, slot as isize, slot as isize + 1, 1))?;
        numerical(py)?.call_method1("copy_scalar", (target, length))?;
        Ok(())
    }

    pub(super) fn set_prompt_logits(
        &self,
        py: Python<'_>,
        slot: i64,
        logits: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.check_slots(&[slot])?;
        let target = self.prompt_logits(py)?.bind(py).get_item(slot)?;
        let options = PyDict::new(py);
        options.set_item("dtype", &self.logits_dtype)?;
        target.call_method1("copy_", (logits.call_method("to", (), Some(&options))?,))?;
        Ok(())
    }

    /// Commit a decode batch or one explicit prefill/verification result.
    /// Caller-provided device slots correspond to the validated host slot list.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (slots, *, tokens, predicates, valid, active, penalty_bases, device_slots=None, logical_position=None, sampling_position=None))]
    pub(super) fn apply_tokens(
        &self,
        py: Python<'_>,
        slots: Vec<i64>,
        tokens: &Bound<'_, PyAny>,
        predicates: &Bound<'_, PyAny>,
        valid: &Bound<'_, PyAny>,
        active: &Bound<'_, PyAny>,
        penalty_bases: Vec<Option<Py<PyAny>>>,
        device_slots: Option<&Bound<'_, PyAny>>,
        logical_position: Option<&Bound<'_, PyAny>>,
        sampling_position: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        self.check_slots(&slots)?;
        if slots.len() != penalty_bases.len() {
            return Err(PyValueError::new_err(
                "decode penalty rows do not align with request slots",
            ));
        }
        let numerical = numerical(py)?;
        let selected = if let Some(indices) = device_slots {
            if logical_position.is_some() || sampling_position.is_some() {
                return Err(PyValueError::new_err(
                    "batched decode does not replace explicit coordinates",
                ));
            }
            let count = slots.len();
            let indices = indices.call_method1("reshape", (-1,))?;
            if count == 0
                || indices.call_method0("numel")?.extract::<usize>()? != count
                || !indices.getattr("device")?.eq(self.device.bind(py))?
                || !indices
                    .getattr("dtype")?
                    .eq(py.import("torch")?.getattr("int64")?)?
            {
                return Err(PyValueError::new_err(
                    "decode runtime-state indices are not aligned",
                ));
            }
            let tokens = tokens.call_method1("reshape", (-1,))?;
            let predicates = predicates.call_method1("reshape", (-1,))?;
            if tokens.call_method0("numel")?.extract::<usize>()? != count
                || predicates.call_method0("numel")?.extract::<usize>()? != count
            {
                return Err(PyValueError::new_err(
                    "decode runtime-state values are not aligned",
                ));
            }
            numerical.call_method1(
                if self.fused {
                    "commit_tokens_cuda"
                } else {
                    "commit_tokens_torch"
                },
                (
                    self.tensors(py),
                    indices,
                    &tokens,
                    predicates,
                    count,
                    self.continuation_width,
                    self.request_pool_size,
                ),
            )?;
            tokens
        } else {
            let (Some(logical), Some(sampling)) = (logical_position, sampling_position) else {
                return Err(PyValueError::new_err(
                    "explicit token export requires one complete request row",
                ));
            };
            if slots.len() != 1 {
                return Err(PyValueError::new_err(
                    "explicit token export requires one complete request row",
                ));
            }
            numerical.call_method1(
                "commit_explicit",
                (
                    self.tensors(py),
                    slots[0],
                    tokens,
                    predicates,
                    logical,
                    sampling,
                ),
            )?
        };
        let penalties: Vec<_> = penalty_bases
            .iter()
            .enumerate()
            .filter_map(|(index, counts)| counts.as_ref().map(|counts| (index, counts)))
            .collect();
        if !penalties.is_empty() {
            numerical.call_method1(
                "commit_penalties",
                (selected, valid, active, PyTuple::new(py, penalties)?),
            )?;
        }
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.device)?;
        visit.call(&self.logits_dtype)?;
        visit.call(&self.backing)
    }
}

impl DecodeState {
    /// Retire native request slots without building Python request indices.
    pub(super) fn reset_slots(&self, py: Python<'_>, slots: &[usize]) -> PyResult<()> {
        if slots.is_empty() {
            return Ok(());
        }
        let slots: Vec<_> = slots.iter().map(|&slot| slot as i64).collect();
        if self.fused {
            self.reset_host(py, &slots, [None; 3])
        } else {
            numerical(py)?.call_method1(
                "reset_indexed",
                (
                    self.tensors(py),
                    self.host_tensor(py, &slots)?,
                    py.None(),
                    py.None(),
                    py.None(),
                ),
            )?;
            Ok(())
        }
    }

    fn reset_host(
        &self,
        py: Python<'_>,
        slots: &[i64],
        columns: [Option<&[i64]>; 3],
    ) -> PyResult<()> {
        let count = slots.len();
        let mut values = vec![0; 4 * count];
        values[..count].copy_from_slice(slots);
        for (index, column) in columns.into_iter().enumerate() {
            if let Some(column) = column {
                values[(index + 1) * count..(index + 2) * count].copy_from_slice(column);
            }
        }
        numerical(py)?.call_method1(
            "reset_rows",
            (
                self.tensors(py),
                self.host_tensor(py, &values)?,
                count,
                self.continuation_width,
                self.vocab_size,
            ),
        )?;
        Ok(())
    }

    fn tensors<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        self.backing.borrow(py).tensors.bind(py).clone()
    }

    fn check_slots(&self, slots: &[i64]) -> PyResult<()> {
        if slots
            .iter()
            .any(|&slot| slot < 1 || slot as usize > self.request_pool_size)
        {
            return Err(PyValueError::new_err(
                "request-pool index is outside runtime-state capacity",
            ));
        }
        if slots.len() > 1 && slots.iter().collect::<HashSet<_>>().len() != slots.len() {
            return Err(PyValueError::new_err(
                "runtime-state mutation repeats a request-pool index",
            ));
        }
        Ok(())
    }

    fn host_tensor<'py>(&self, py: Python<'py>, values: &[i64]) -> PyResult<Bound<'py, PyAny>> {
        let options = PyDict::new(py);
        options.set_item("dtype", py.import("torch")?.getattr("int64")?)?;
        options.set_item("device", &self.device)?;
        py.import("uniserve.runtime.device")?.call_method(
            "async_tensor_h2d",
            (PyTuple::new(py, values)?,),
            Some(&options),
        )
    }
}

/// A device tensor has no host values until its owner explicitly copies it.
fn host_values(value: &Bound<'_, PyAny>) -> PyResult<Option<Vec<i64>>> {
    let tensor = value.py().import("torch")?.getattr("Tensor")?;
    if value.is_instance(&tensor)? {
        if value
            .getattr("device")?
            .getattr("type")?
            .extract::<String>()?
            != "cpu"
        {
            return Ok(None);
        }
        value
            .call_method1("reshape", (-1,))?
            .call_method0("tolist")?
            .extract()
            .map(Some)
    } else {
        value.extract().map(Some)
    }
}

fn host_column(value: Option<&Bound<'_, PyAny>>, count: usize) -> PyResult<Option<Vec<i64>>> {
    let Some(value) = value else {
        return Ok(Some(vec![0; count]));
    };
    let host = host_values(value)?;
    if host.as_ref().is_some_and(|host| host.len() != count) {
        return Err(PyValueError::new_err(
            "runtime-state reset columns are not aligned",
        ));
    }
    Ok(host)
}

fn numerical(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("uniserve_worker.storage.decode_state")
}
