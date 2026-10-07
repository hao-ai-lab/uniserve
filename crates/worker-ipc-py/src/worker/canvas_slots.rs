//! Resident block-diffusion state shared by disjoint request slots.

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::tensor_buffers::TensorBuffers;
use crate::batches::CanvasSampling;

/// Slot zero is inactive padding. Executions borrow separate contiguous
/// workspaces and commit only the real slots they own until completion.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct CanvasSlots {
    #[pyo3(get)]
    pub(super) request_pool_size: usize,
    #[pyo3(get)]
    tokens: Py<PyAny>,
    #[pyo3(get)]
    pub(super) canvas_length: usize,
    #[pyo3(get)]
    pub(super) vocab_size: usize,
    #[pyo3(get)]
    pub(super) hidden_size: usize,
    #[pyo3(get)]
    pub(super) served: Py<CanvasSampling>,
    #[pyo3(get)]
    pub(super) history_depth: usize,
    #[pyo3(get)]
    pub(super) constants: Py<PyAny>,
    #[pyo3(get)]
    device: Py<PyAny>,
    #[pyo3(get)]
    pub(super) banks: Py<PyDict>,
    backing: Py<TensorBuffers>,
}

#[pymethods]
impl CanvasSlots {
    /// Bound transient FP32 logits and BF16 sampling weights to 512 MiB,
    /// retaining at least one whole canvas when a single row exceeds it.
    #[staticmethod]
    #[pyo3(signature = (*, canvas_length, vocab_size, max_rows))]
    pub(super) fn step_rows(
        canvas_length: usize,
        vocab_size: usize,
        max_rows: usize,
    ) -> PyResult<usize> {
        if canvas_length == 0 || vocab_size == 0 {
            return Err(PyValueError::new_err("canvas dimensions must be positive"));
        }
        let rows = (512 << 20) / canvas_length / vocab_size / 6;
        Ok(rows.min(max_rows).max(1))
    }

    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (*, request_pool_size, tokens, vocab_size, hidden_size, sampling, dtype, device))]
    fn new(
        py: Python<'_>,
        request_pool_size: usize,
        tokens: Py<PyAny>,
        vocab_size: usize,
        hidden_size: usize,
        sampling: Py<CanvasSampling>,
        dtype: &Bound<'_, PyAny>,
        device: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let served = sampling.borrow(py).inner;
        let canvas_length: usize = tokens.bind(py).getattr("length")?.extract()?;
        if served.canvas_length as usize != canvas_length {
            return Err(PyValueError::new_err(
                "the served canvas length is not the model's canvas length",
            ));
        }
        let history_depth = served.stability_threshold as usize;
        let fields = Self::buffers(
            py,
            request_pool_size,
            canvas_length,
            hidden_size,
            history_depth,
            dtype,
        )?;
        let options = PyDict::new(py);
        options.set_item("steps", served.max_steps)?;
        options.set_item("entropy_bound", served.entropy_bound)?;
        options.set_item("t_min", served.t_min)?;
        options.set_item("t_max", served.t_max)?;
        options.set_item("confidence", served.confidence_threshold)?;
        options.set_item("stability", served.stability_threshold)?;
        options.set_item("eos_ids", tokens.bind(py).getattr("eos_token_ids")?)?;
        options.set_item("pad_id", tokens.bind(py).getattr("pad_token_id")?)?;
        let constants = py
            .import("uniserve.diffusion.canvas")?
            .getattr("CanvasSampling")?
            .call((), Some(&options))?
            .unbind();

        let device = py.import("torch")?.call_method1("device", (device,))?;
        let backing = TensorBuffers::allocate(py, &fields, &device, false, None)?;
        let banks = PyDict::new(py);
        for name in ["canvas", "history", "self_conditioning", "live"] {
            let tensor = backing.backing(py, name)?;
            tensor.call_method0(py, "zero_")?;
            if name != "live" {
                banks.set_item(name, tensor)?;
            }
        }
        Ok(Self {
            request_pool_size,
            tokens,
            canvas_length,
            vocab_size,
            hidden_size,
            served: sampling,
            history_depth,
            constants,
            device: device.unbind(),
            banks: banks.unbind(),
            backing: Py::new(py, backing)?,
        })
    }

    /// Borrow one uint8 continuation flag per slot, including the sentinel.
    #[getter]
    pub(super) fn live(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.backing.borrow(py).backing(py, "live")
    }

    #[staticmethod]
    #[pyo3(signature = (denoiser, *, request_pool_size, sampling, device))]
    pub(super) fn for_denoiser(
        py: Python<'_>,
        denoiser: &Bound<'_, PyAny>,
        request_pool_size: usize,
        sampling: Py<CanvasSampling>,
        device: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let embedding = denoiser.getattr("backbone")?.getattr("embedding")?;
        Self::new(
            py,
            request_pool_size,
            denoiser.getattr("canvas")?.unbind(),
            denoiser
                .getattr("lm_head")?
                .getattr("vocab")?
                .getattr("size")?
                .extract()?,
            embedding.getattr("embedding_dim")?.extract()?,
            sampling,
            &embedding.getattr("weight")?.getattr("dtype")?,
            device,
        )
    }

    /// Charge the same resident banks allocated by for_denoiser at startup.
    #[staticmethod]
    #[pyo3(signature = (denoiser, *, request_pool_size, history_depth))]
    fn denoiser_bytes(
        py: Python<'_>,
        denoiser: &Bound<'_, PyAny>,
        request_pool_size: usize,
        history_depth: usize,
    ) -> PyResult<usize> {
        let embedding = denoiser.getattr("backbone")?.getattr("embedding")?;
        let fields = Self::buffers(
            py,
            request_pool_size,
            denoiser.getattr("canvas")?.getattr("length")?.extract()?,
            embedding.getattr("embedding_dim")?.extract()?,
            history_depth,
            &embedding.getattr("weight")?.getattr("dtype")?,
        )?;
        fields
            .values()
            .iter()
            .map(|field| field.getattr("nbytes")?.extract::<usize>())
            .sum()
    }

    /// Shape declarations include the inactive row; history depth may be zero.
    #[staticmethod]
    #[pyo3(signature = (*, request_pool_size, canvas_length, hidden_size, history_depth, dtype))]
    fn buffers<'py>(
        py: Python<'py>,
        request_pool_size: usize,
        canvas_length: usize,
        hidden_size: usize,
        history_depth: usize,
        dtype: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if request_pool_size == 0 || canvas_length == 0 || hidden_size == 0 {
            return Err(PyValueError::new_err(
                "canvas state dimensions must be positive",
            ));
        }
        let count = request_pool_size + 1;
        let torch = py.import("torch")?;
        let int64 = torch.getattr("int64")?;
        let uint8 = torch.getattr("uint8")?;
        let config = py.import("uniserve.tensors")?.getattr("BufferConfig")?;
        let fields = PyDict::new(py);
        for (name, shape, dtype) in [
            ("canvas", vec![count, canvas_length], &int64),
            ("history", vec![count, history_depth, canvas_length], &int64),
            (
                "self_conditioning",
                vec![count, canvas_length, hidden_size],
                dtype,
            ),
            ("live", vec![count], &uint8),
        ] {
            fields.set_item(name, config.call1((PyTuple::new(py, shape)?, dtype))?)?;
        }
        Ok(fields)
    }

    /// Copy selected state into caller-owned contiguous numerical views.
    fn gather(
        &self,
        py: Python<'_>,
        slots: &Bound<'_, PyAny>,
        views: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        py.import("uniserve_worker.storage.canvas_slots")?
            .call_method1("gather", (&self.banks, slots, views))?;
        Ok(())
    }

    /// Commit active rows; the numerical kernel masks out padding slot zero.
    #[pyo3(signature = (slots, views, *, live))]
    fn commit(
        &self,
        py: Python<'_>,
        slots: &Bound<'_, PyAny>,
        views: &Bound<'_, PyAny>,
        live: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        py.import("uniserve_worker.storage.canvas_slots")?
            .call_method1("commit", (&self.banks, self.live(py)?, slots, views, live))?;
        Ok(())
    }

    pub(super) fn close(&self, py: Python<'_>) {
        self.banks.bind(py).clear();
        self.backing.borrow_mut(py).close(py);
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.tokens)?;
        visit.call(&self.served)?;
        visit.call(&self.constants)?;
        visit.call(&self.device)?;
        visit.call(&self.banks)?;
        visit.call(&self.backing)
    }
}
