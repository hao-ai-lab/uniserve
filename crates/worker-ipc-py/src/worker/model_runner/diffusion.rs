//! Denoiser resources shared by prepared layouts and request sequences.

mod layouts;
mod sequence;
pub(in crate::worker) use layouts::DenoisingBuffers;
pub(in crate::worker) use sequence::DenoisingSequence;

use indexmap::IndexMap;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyType};

use super::ModelRunner;
use crate::worker::execution::close_all;
use crate::worker::execution_context::ExecutionContext;
use crate::worker::graph_storage::GraphStorage;
use crate::worker::host::with_context;
use crate::worker::input_buffers::InputBuffers;
use crate::worker::latent::LatentPool;
use crate::worker::tensor_buffers::TensorBuffers;

/// Layouts share sample and workspace backing on one ordered execution stream.
/// Request inputs borrow that backing; the runner retires it after its graphs.
#[pyclass(extends = ModelRunner, subclass, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct DiffusionRunner {
    bank: Py<PyDict>,
    slots: usize,
    #[pyo3(get, name = "_slot_index")]
    slot_index: Option<Py<PyAny>>,
    #[pyo3(get)]
    pool: Option<Py<LatentPool>>,
    #[pyo3(get)]
    samples: Option<Py<PyAny>>,
    workspace: Option<Py<TensorBuffers>>,
    state_buffers: Option<Py<TensorBuffers>>,
    maximum: Option<Py<PyAny>>,
    pub(in crate::worker) layouts: Py<PyDict>,
    // Recent page tables are bounded by request capacity. PyTorch's pinned
    // allocator records retirement events on the copy stream when freed,
    // so these sources must be released before that stream is destroyed.
    indices: IndexMap<(usize, Vec<i64>), Py<PyAny>>,
}

#[pymethods]
impl DiffusionRunner {
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
        let slot_index = if !base.cuda_stream.is_none(py)
            && !base.execution.borrow(py).pools.bind(py).is_empty()
        {
            let torch = py.import("torch")?;
            let options = PyDict::new(py);
            options.set_item("dtype", torch.getattr("int64")?)?;
            options.set_item("device", &base.device)?;
            Some(torch.call_method("zeros", (1,), Some(&options))?.unbind())
        } else {
            None
        };

        Ok(PyClassInitializer::from(base).add_subclass(Self {
            bank: PyDict::new(py).unbind(),
            slots: 0,
            slot_index,
            pool: None,
            samples: None,
            workspace: None,
            state_buffers: None,
            maximum: None,
            layouts: PyDict::new(py).unbind(),
            indices: IndexMap::new(),
        }))
    }

    /// Allocate the maximum layout's shared storage before preparing layouts.
    #[classmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (name, call, maximum, *, device, stream, storage, devices=Vec::new(), bank=None, slots=0, pool, pages, attention=None))]
    pub(in crate::worker) fn for_layouts<'py>(
        cls: &Bound<'py, PyType>,
        name: String,
        call: &Bound<'py, PyAny>,
        maximum: Py<PyAny>,
        device: &Bound<'py, PyAny>,
        stream: &Bound<'py, PyAny>,
        storage: Py<GraphStorage>,
        devices: Vec<Py<PyAny>>,
        bank: Option<&Bound<'py, PyAny>>,
        slots: usize,
        pool: Py<LatentPool>,
        pages: usize,
        attention: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = cls.py();
        let options = PyDict::new(py);
        options.set_item(
            "attention",
            attention.unwrap_or("auto".into_pyobject(py)?.into_any().unbind()),
        )?;
        options.set_item("stream", stream)?;
        options.set_item("groups", call.getattr("groups")?)?;
        options.set_item("derive_host_lengths", false)?;
        let context = py
            .get_type::<ExecutionContext>()
            .call((call.getattr("module")?,), Some(&options))?;
        let kinds = py
            .import("uniserve_worker.protocol.call")?
            .getattr("MediaCall")?;
        let kinds = (
            kinds.getattr("LATENT_PREPARATION")?,
            kinds.getattr("DENOISING")?,
        );
        let options = PyDict::new(py);
        options.set_item("storage", &storage)?;
        options.set_item("devices", PyTuple::new(py, devices)?)?;
        let runner = cls
            .call((name, call, device, kinds, stream, context), Some(&options))?
            .cast_into::<Self>()?;
        let prepared = Self::allocate(&runner, maximum, bank, slots, pool, pages, &storage);
        if let Err(error) = prepared {
            if let Err(cleanup) = Self::close(&runner) {
                let _ = error.value(py).call_method1(
                    "add_note",
                    (format!("Resource cleanup also failed: {cleanup}"),),
                );
            }
            return Err(error);
        }
        Ok(runner)
    }

    #[getter]
    pub(in crate::worker) fn captures(slf: PyRef<'_, Self>) -> bool {
        !slf.as_super().cuda_stream.is_none(slf.py())
            && !slf
                .as_super()
                .execution
                .borrow(slf.py())
                .pools
                .bind(slf.py())
                .is_empty()
    }

    /// Prepare constants and compact views. The maximum must be prepared first.
    #[pyo3(signature = (layout, *, pages))]
    pub(in crate::worker) fn prepare<'py>(
        slf: &Bound<'py, Self>,
        layout: &Bound<'py, PyAny>,
        pages: usize,
    ) -> PyResult<Bound<'py, DenoisingBuffers>> {
        layouts::prepare(slf, layout, pages)
    }

    /// Release an uncaptured layout after its queued readers finish.
    pub(in crate::worker) fn retire(
        slf: &Bound<'_, Self>,
        layout: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        layouts::retire(slf, layout)
    }

    pub(in crate::worker) fn layout<'py>(
        slf: &Bound<'py, Self>,
        layout: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, DenoisingBuffers>> {
        layouts::layout(slf, layout)
    }

    /// Bind numerical steps to the request's slot and leading sample pages.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (layout, inputs, schedules, *, state, slot, pages))]
    fn bind(
        slf: &Bound<'_, Self>,
        layout: Py<PyAny>,
        inputs: &Bound<'_, PyAny>,
        schedules: Py<PyAny>,
        state: Py<PyAny>,
        slot: usize,
        pages: Vec<i64>,
    ) -> PyResult<DenoisingSequence> {
        sequence::bind(slf, layout, inputs, schedules, state, slot, pages)
    }

    pub(in crate::worker) fn binds(
        slf: &Bound<'_, Self>,
        sequence: PyRef<'_, DenoisingSequence>,
    ) -> PyResult<bool> {
        let owner = slf.borrow();
        Ok(owner
            .samples
            .as_ref()
            .is_some_and(|samples| samples.is(&sequence.samples))
            && owner
                .layouts
                .bind(slf.py())
                .contains(sequence.layout.bind(slf.py()))?)
    }

    /// Warm all layouts before any capture; persistent scratch shares graph pools.
    pub(in crate::worker) fn warmup(
        slf: &Bound<'_, Self>,
        sequence: &Bound<'_, DenoisingSequence>,
    ) -> PyResult<()> {
        layouts::warm(slf, sequence)
    }

    pub(in crate::worker) fn capture(
        slf: &Bound<'_, Self>,
        sequence: &Bound<'_, DenoisingSequence>,
    ) -> PyResult<()> {
        layouts::capture(slf, sequence)
    }

    /// Read the committed bank and write the successor; request progress is committed separately.
    pub(in crate::worker) fn step(
        slf: &Bound<'_, Self>,
        sequence: &Bound<'_, DenoisingSequence>,
        index: usize,
        bank: i64,
    ) -> PyResult<(Py<PyAny>, &'static str)> {
        layouts::step(slf, sequence, index, bank)
    }

    fn close(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let closed = ModelRunner::close(slf.as_any().cast::<ModelRunner>()?);
        let (layouts, workspace, state) = {
            let mut owner = slf.borrow_mut();
            let layouts = owner.layouts.bind(py).values();
            owner.layouts.bind(py).clear();
            owner.bank.bind(py).clear();
            owner.indices.clear();
            owner.pool = None;
            owner.samples = None;
            owner.slot_index = None;
            owner.maximum = None;
            (layouts, owner.workspace.take(), owner.state_buffers.take())
        };
        let buffers = close_all(
            py,
            layouts.iter().map(|layout| {
                layout.cast::<DenoisingBuffers>()?.borrow().close(py);
                Ok(())
            }),
        );
        for backing in workspace.into_iter().chain(state) {
            backing.borrow_mut(py).close(py);
        }
        close_all(py, [closed, buffers])
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.bank)?;
        visit.call(&self.slot_index)?;
        visit.call(&self.pool)?;
        visit.call(&self.samples)?;
        visit.call(&self.workspace)?;
        visit.call(&self.state_buffers)?;
        visit.call(&self.maximum)?;
        visit.call(&self.layouts)?;
        for source in self.indices.values() {
            visit.call(source)?;
        }
        Ok(())
    }
}

impl DiffusionRunner {
    #[allow(clippy::too_many_arguments)]
    fn allocate(
        slf: &Bound<'_, Self>,
        maximum: Py<PyAny>,
        bank: Option<&Bound<'_, PyAny>>,
        slots: usize,
        pool: Py<LatentPool>,
        pages: usize,
        storage: &Py<GraphStorage>,
    ) -> PyResult<()> {
        let py = slf.py();
        let bank = match bank {
            Some(bank) => py
                .get_type::<PyDict>()
                .call1((bank,))?
                .cast_into::<PyDict>()?,
            None => PyDict::new(py),
        };
        for (name, tensor) in bank.iter() {
            let shape: Vec<usize> = tensor.getattr("shape")?.extract()?;
            if shape.first() != Some(&slots)
                || !tensor.call_method0("is_contiguous")?.is_truthy()?
            {
                return Err(PyValueError::new_err(format!(
                    "bank {name} requires {slots} contiguous slot rows"
                )));
            }
        }
        let (device, stream, model, execution) = {
            let owner = slf.borrow();
            let base = owner.as_super();
            (
                base.device.clone_ref(py),
                base.cuda_stream.clone_ref(py),
                base.model.clone_ref(py),
                base.execution.clone_ref(py),
            )
        };
        if !pool.bind(py).getattr("device")?.eq(device.bind(py))?
            || pages == 0
            || pages >= pool.borrow(py).inner.num_pages()
        {
            return Err(PyValueError::new_err(
                "a denoiser runner requires layout pages of a pool on its device",
            ));
        }
        if !stream.is_none(py) {
            let current = py
                .import("torch.cuda")?
                .call_method1("current_stream", (&device,))?;
            stream.call_method1(py, "wait", (current,))?;
        }
        let captures = Self::captures(slf.borrow());
        {
            let mut owner = slf.borrow_mut();
            owner.bank = bank.unbind();
            owner.slots = slots;
            owner.pool = Some(pool.clone_ref(py));
            owner.maximum = Some(maximum.clone_ref(py));
        }

        with_context(
            &storage.bind(py).call_method1("allocate", (&execution,))?,
            || {
                let configs =
                    buffer_configs(model.bind(py), "workspace_buffers", maximum.bind(py))?;
                slf.borrow_mut().workspace = Some(Py::new(
                    py,
                    TensorBuffers::allocate(py, &configs, device.bind(py), false, None)?,
                )?);
                let options = PyDict::new(py);
                options.set_item("dtype", pool.bind(py).getattr("dtype")?)?;
                options.set_item("device", &device)?;
                let shape = (
                    pages * pool.borrow(py).inner.page_units(),
                    pool.borrow(py).latent_width,
                );
                let samples = py
                    .import("torch")?
                    .call_method("empty", (shape,), Some(&options))?;
                slf.borrow_mut().samples = Some(samples.unbind());
                if captures {
                    let configs = PyDict::new(py);
                    let config = py.import("uniserve.tensors")?.getattr("BufferConfig")?;
                    for (name, tensor) in slf.borrow().bank.bind(py).iter() {
                        let shape: Vec<usize> = tensor.getattr("shape")?.extract()?;
                        configs.set_item(
                            name,
                            config.call1((
                                PyTuple::new(py, &shape[1..])?,
                                tensor.getattr("dtype")?,
                            ))?,
                        )?;
                    }
                    slf.borrow_mut().state_buffers = Some(Py::new(
                        py,
                        TensorBuffers::allocate(
                            py,
                            configs.as_any(),
                            device.bind(py),
                            false,
                            None,
                        )?,
                    )?);
                }
                Ok(())
            },
        )?;
        storage.borrow(py).check(py)
    }
}

fn buffer_configs<'py>(
    model: &Bound<'py, PyAny>,
    name: &str,
    layout: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    match model.getattr(name) {
        Ok(query) => query.call1((layout,)),
        Err(error) if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(model.py()) => {
            Ok(PyDict::new(model.py()).into_any())
        }
        Err(error) => Err(error),
    }
}
