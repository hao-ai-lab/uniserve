//! Encoder input buffers and graph capture on the bound numerical context.

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};

use super::ModelRunner;
use crate::worker::cuda_graph::CUDAGraphRunner;
use crate::worker::execution::close_all;
use crate::worker::execution_context::ExecutionContext;
use crate::worker::graph_storage::GraphStorage;
use crate::worker::host::with_context;
use crate::worker::host_buffers::{HostBuffers, fill};
use crate::worker::input_buffers::InputBuffers;

/// Text encoders reuse one device column and a fenced host ring. Vision
/// encoders borrow image graphs from Execution and supply numerical layouts.
#[pyclass(extends = ModelRunner, subclass, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct EncoderRunner {
    packed: bool,
    tokens: Option<(Py<PyAny>, Py<HostBuffers>)>,
}

#[pymethods]
impl EncoderRunner {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (name, call, device, kinds, stream, context, inputs=None, *, storage, devices, exact_graphs=false, cache=None, predicates=None, rank=0, share=None))]
    fn new(
        py: Python<'_>,
        name: String,
        call: Py<PyAny>,
        device: Py<PyAny>,
        kinds: &Bound<'_, PyAny>,
        stream: Py<PyAny>,
        context: Py<ExecutionContext>,
        inputs: Option<Py<InputBuffers>>,
        storage: Py<GraphStorage>,
        devices: &Bound<'_, PyAny>,
        exact_graphs: bool,
        cache: Option<Py<PyAny>>,
        predicates: Option<Py<PyAny>>,
        rank: usize,
        share: Option<PyRef<'_, ModelRunner>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let model = call.bind(py).getattr("module")?;
        let packed = model.is_instance(&py.import("uniserve.model")?.getattr("PatchEncoder")?)?
            && !model.getattr("max_patches")?.is_none();
        let base = ModelRunner::new(
            py,
            name,
            call,
            device,
            kinds,
            stream,
            context,
            inputs,
            storage,
            devices,
            exact_graphs,
            cache,
            predicates,
            rank,
            share,
        )?;
        Ok(PyClassInitializer::from(base).add_subclass(Self {
            packed,
            tokens: None,
        }))
    }

    #[getter]
    fn packs_images(slf: PyRef<'_, Self>) -> bool {
        slf.packed
            && !slf
                .as_super()
                .execution
                .borrow(slf.py())
                .pools
                .bind(slf.py())
                .is_empty()
    }

    /// Copy flat token ids into fixed device storage on the current stream.
    /// Capacity is fixed for this runner; the returned view lasts until its
    /// next token preparation. Host sources wait only when their ring wraps.
    #[pyo3(signature = (tokens, *, capacity))]
    fn prepare_tokens(
        slf: &Bound<'_, Self>,
        tokens: Vec<i64>,
        capacity: usize,
    ) -> PyResult<Py<PyAny>> {
        Self::prepare(slf, &tokens, capacity)
    }

    fn capture_packed(slf: PyRef<'_, Self>, inputs: Py<PyAny>) -> PyResult<CUDAGraphRunner> {
        let py = slf.py();
        let base = slf.as_super();
        let context = base.execution.borrow(py).context.extract(py)?;
        let pools = base.execution.borrow(py).pools.clone_ref(py);
        let call = py
            .import("functools")?
            .call_method1(
                "partial",
                (wrap_pyfunction!(encode_packed, py)?, &base.model),
            )?
            .unbind();
        CUDAGraphRunner::capture(
            py,
            context,
            inputs,
            call,
            Some(pools.bind(py).as_any()),
            None,
            true,
            None,
        )
    }

    fn close(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let base = ModelRunner::close(slf.as_any().cast()?);
        let tokens = slf.borrow_mut().tokens.take();
        let inputs = tokens.map_or(Ok(()), |(_, host)| host.borrow(py).close(py));
        close_all(py, [base, inputs])
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some((tensor, host)) = &self.tokens {
            visit.call(tensor)?;
            visit.call(host)?;
        }
        Ok(())
    }
}

impl EncoderRunner {
    pub(in crate::worker) fn prepare(
        slf: &Bound<'_, Self>,
        tokens: &[i64],
        capacity: usize,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if tokens.is_empty() || tokens.len() > capacity {
            return Err(crate::worker::error::input_error(
                py,
                "text encoder input exceeds its token capacity",
            ));
        }
        let torch = py.import("torch")?;
        with_context(&torch.call_method0("inference_mode")?, || {
            let buffers = slf
                .borrow()
                .tokens
                .as_ref()
                .map(|(tensor, host)| (tensor.clone_ref(py), host.clone_ref(py)));
            let (tensor, host) = match buffers {
                Some(buffers) => buffers,
                None => {
                    let device = slf.borrow().as_super().device.clone_ref(py);
                    let dtype = torch.getattr("int64")?;
                    let options = PyDict::new(py);
                    options.set_item("device", &device)?;
                    options.set_item("dtype", &dtype)?;
                    let tensor = torch
                        .call_method("empty", (capacity,), Some(&options))?
                        .unbind();
                    let shape = PyTuple::new(py, [capacity])?;
                    let host = Py::new(
                        py,
                        HostBuffers::new(py, shape.as_any(), &dtype, 2, device.bind(py))?,
                    )?;
                    slf.borrow_mut().tokens = Some((tensor.clone_ref(py), host.clone_ref(py)));
                    (tensor, host)
                }
            };
            let (slot, source) = host.borrow(py).acquire(py)?;
            fill(source.bind(py), tokens)?;
            let rows = PySlice::new(py, 0, tokens.len() as isize, 1);
            let target = tensor.bind(py).get_item(&rows)?;
            let options = PyDict::new(py);
            options.set_item("non_blocking", true)?;
            target.call_method("copy_", (source.bind(py).get_item(rows)?,), Some(&options))?;
            host.borrow(py).record_copy(py, slot)?;
            Ok(target.call_method1("view", (1, -1))?.unbind())
        })
    }
}

#[pyfunction]
fn encode_packed(model: &Bound<'_, PyAny>, inputs: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    Ok(model.call_method1("encode_packed", inputs)?.unbind())
}
