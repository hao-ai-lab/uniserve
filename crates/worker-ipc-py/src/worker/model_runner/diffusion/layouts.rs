//! Standalone denoising layout lifetime, preparation and graph dispatch.

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};

use super::{DenoisingSequence, DiffusionRunner, buffer_configs};
use crate::worker::cuda_graph::{CUDAGraphError, CUDAGraphRunner};
use crate::worker::execution::{GraphBucket, on_stream};
use crate::worker::execution_context::ExecutionContext;
use crate::worker::host::with_context;
use crate::worker::tensor_buffers::TensorBuffers;

/// Numerical backing for one layout. Graph input views borrow the execution's
/// shared state buffers; constants remain live until queued readers retire.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct DenoisingBuffers {
    #[pyo3(get)]
    backing: Py<TensorBuffers>,
    #[pyo3(get)]
    constants: Py<PyAny>,
    #[pyo3(get)]
    workspace: Py<PyAny>,
    #[pyo3(get)]
    pub(super) pages: usize,
    #[pyo3(get)]
    rows: Py<PyAny>,
    #[pyo3(get)]
    state: Py<PyAny>,
    #[pyo3(get)]
    gathers: Py<PyAny>,
    signature: Option<Py<PyAny>>,
    warmed: bool,
}

#[pymethods]
impl DenoisingBuffers {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.backing)?;
        for value in [
            &self.constants,
            &self.workspace,
            &self.rows,
            &self.state,
            &self.gathers,
        ] {
            visit.call(value)?;
        }
        visit.call(&self.signature)
    }
}

impl DenoisingBuffers {
    fn new(
        py: Python<'_>,
        backing: Py<TensorBuffers>,
        constants: Py<PyAny>,
        workspace: Py<PyAny>,
        pages: usize,
        rows: Py<PyAny>,
    ) -> Self {
        Self {
            backing,
            constants,
            workspace,
            pages,
            rows,
            state: PyDict::new(py).into_any().unbind(),
            gathers: PyTuple::empty(py).into_any().unbind(),
            signature: None,
            warmed: false,
        }
    }

    pub(super) fn close(&self, py: Python<'_>) {
        self.backing.borrow_mut(py).close(py);
    }
}

pub(super) fn layout<'py>(
    runner: &Bound<'py, DiffusionRunner>,
    key: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, DenoisingBuffers>> {
    runner
        .borrow()
        .layouts
        .bind(runner.py())
        .get_item(key)?
        .ok_or_else(|| PyKeyError::new_err(key.clone().unbind()))?
        .cast_into()
        .map_err(Into::into)
}

pub(super) fn prepare<'py>(
    runner: &Bound<'py, DiffusionRunner>,
    key: &Bound<'py, PyAny>,
    pages: usize,
) -> PyResult<Bound<'py, DenoisingBuffers>> {
    let py = runner.py();
    let (layouts, maximum, execution) = {
        let owner = runner.borrow();
        (
            owner.layouts.clone_ref(py),
            owner.maximum.as_ref().map(|value| value.clone_ref(py)),
            owner.as_super().execution.clone_ref(py),
        )
    };
    let layouts = layouts.bind(py);
    if let Some(value) = layouts.get_item(key)? {
        return Ok(value.cast_into()?);
    }
    let maximum =
        maximum.ok_or_else(|| PyRuntimeError::new_err("denoising buffers are not configured"))?;
    let first = layouts.is_empty();
    if first && !maximum.bind(py).eq(key)? {
        return Err(PyValueError::new_err(
            "a denoiser runner prepares its maximum first",
        ));
    }

    let allocate = || allocate_layout(runner, key, pages, first);
    // Later layouts allocate outside graph pools: their constants must not
    // occupy the free blocks that captured intermediates reuse.
    let buffers = if execution.borrow(py).buckets.bind(py).is_empty() {
        let scope = execution
            .bind(py)
            .getattr("storage")?
            .call_method1("allocate", (&execution,))?;
        with_context(&scope, allocate)?
    } else {
        allocate()?
    };
    layouts.set_item(key, &buffers)?;
    Ok(buffers)
}

pub(super) fn retire(runner: &Bound<'_, DiffusionRunner>, key: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = runner.py();
    let (maximum, execution, layouts) = {
        let owner = runner.borrow();
        (
            owner.maximum.as_ref().map(|value| value.clone_ref(py)),
            owner.as_super().execution.clone_ref(py),
            owner.layouts.clone_ref(py),
        )
    };
    let maximum = maximum
        .map(|value| value.bind(py).eq(key))
        .transpose()?
        .unwrap_or(false);
    let bucket = execution.borrow(py).buckets.bind(py).get_item(key)?;
    let captured = bucket
        .map(|value| -> PyResult<bool> {
            Ok(value
                .cast::<GraphBucket>()?
                .borrow()
                .get(py, None)
                .is_some())
        })
        .transpose()?
        .unwrap_or(false);
    if maximum || captured {
        return Err(PyValueError::new_err(
            "a captured or maximum layout stays prepared",
        ));
    }
    let Some(value) = layouts.bind(py).get_item(key)? else {
        return Ok(());
    };

    // Constants may still have queued readers when a transient layout retires.
    let stream = execution.borrow(py).context.bind(py).getattr("stream")?;
    if !stream.is_none() {
        stream.call_method0("synchronize")?;
    }
    layouts.bind(py).del_item(key)?;
    value.cast::<DenoisingBuffers>()?.borrow().close(py);
    Ok(())
}

fn require_bound(
    runner: &Bound<'_, DiffusionRunner>,
    sequence: &Bound<'_, DenoisingSequence>,
) -> PyResult<()> {
    if !DiffusionRunner::binds(runner, sequence.borrow())? {
        return Err(PyValueError::new_err(
            "denoising inputs were bound by another runner or their layout has retired",
        ));
    }
    Ok(())
}

pub(super) fn warm(
    runner: &Bound<'_, DiffusionRunner>,
    sequence: &Bound<'_, DenoisingSequence>,
) -> PyResult<()> {
    let py = runner.py();
    require_bound(runner, sequence)?;
    let buffers = layout(runner, sequence.borrow().layout.bind(py))?;
    if buffers.borrow().warmed {
        return Ok(());
    }
    let (execution, device) = {
        let owner = runner.borrow();
        (
            owner.as_super().execution.clone_ref(py),
            owner.as_super().device.clone_ref(py),
        )
    };
    if !execution.borrow(py).buckets.bind(py).is_empty() {
        return Err(PyRuntimeError::new_err(
            "warm every denoiser layout before capturing any",
        ));
    }
    let context = execution.borrow(py).context.clone_ref(py);
    with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
        on_stream(context.bind(py), device.bind(py), || {
            let scope = execution
                .bind(py)
                .getattr("storage")?
                .call_method1("allocate", (&execution,))?;
            with_context(&scope, || {
                runner
                    .call_method1("_warm_step", (&buffers, sequence))
                    .map(drop)
            })
        })
    })?;
    if device.bind(py).getattr("type")?.extract::<String>()? == "cuda" {
        let stream = context.bind(py).getattr("stream")?;
        let stream = if stream.is_none() {
            py.import("torch.cuda")?
                .call_method1("current_stream", (&device,))?
        } else {
            stream
        };
        stream.call_method0("synchronize")?;
    }
    buffers.borrow_mut().warmed = true;
    Ok(())
}

fn bucket<'py>(
    runner: &Bound<'py, DiffusionRunner>,
    sequence: &Bound<'py, DenoisingSequence>,
    buffers: &Bound<'py, DenoisingBuffers>,
) -> PyResult<Bound<'py, GraphBucket>> {
    let py = runner.py();
    let execution = runner.borrow().as_super().execution.clone_ref(py);
    let key = sequence.borrow().layout.clone_ref(py);
    let buckets = execution.borrow(py).buckets.clone_ref(py);
    if let Some(bucket) = buckets.bind(py).get_item(&key)? {
        check_signature(buffers, &sequence.borrow())?;
        return Ok(bucket.cast_into()?);
    }

    let (state, gathers) = state_views(runner, &sequence.borrow())?;
    {
        let mut backing = buffers.borrow_mut();
        backing.state = state;
        backing.gathers = gathers;
        backing.signature = Some(sequence.borrow().signature.clone_ref(py));
    }
    let bucket = Bound::new(py, GraphBucket::new(None))?;
    buckets.bind(py).set_item(key, &bucket)?;
    Ok(bucket)
}

fn check_signature(
    buffers: &Bound<'_, DenoisingBuffers>,
    sequence: &DenoisingSequence,
) -> PyResult<()> {
    if let Some(signature) = &buffers.borrow().signature
        && !signature
            .bind(buffers.py())
            .eq(sequence.signature.bind(buffers.py()))?
    {
        return Err(PyValueError::new_err(
            "denoising inputs differ from their layout's captured steps",
        ));
    }
    Ok(())
}

fn copy_indices(
    runner: &Bound<'_, DiffusionRunner>,
    buffers: &Bound<'_, DenoisingBuffers>,
    sequence: &DenoisingSequence,
    bank: i64,
    captured: bool,
) -> PyResult<()> {
    let py = runner.py();
    let key = (sequence.slot, sequence.pages.clone());
    let cached = runner.borrow_mut().indices.shift_remove(&key);
    let source = match cached {
        Some(source) => source,
        None => {
            let owner = runner.borrow();
            let pool = owner
                .pool
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("denoising runner is closed"))?
                .borrow(py);
            let pool_pages = pool.inner.num_pages() as i64;
            let cuda = owner
                .as_super()
                .device
                .bind(py)
                .getattr("type")?
                .extract::<String>()?
                == "cuda";
            let mut indices = Vec::with_capacity(4 * sequence.pages.len() + 1);
            for bank in [0, 1, 1, 0] {
                indices.extend(sequence.pages.iter().map(|&page| bank * pool_pages + page));
            }
            indices.push((sequence.slot - 1) as i64);
            let torch = py.import("torch")?;
            let options = PyDict::new(py);
            options.set_item("dtype", torch.getattr("int64")?)?;
            options.set_item("device", "cpu")?;
            options.set_item("pin_memory", cuda)?;
            torch
                .call_method("tensor", (indices,), Some(&options))?
                .unbind()
        }
    };
    {
        let mut owner = runner.borrow_mut();
        if owner.indices.len() == owner.slots {
            owner.indices.shift_remove_index(0);
        }
        owner.indices.insert(key, source.clone_ref(py));
    }

    let count = sequence.pages.len();
    let rows = source
        .bind(py)
        .call_method1("narrow", (0, bank as usize * 2 * count, 2 * count))?
        .call_method1("view", (2, count))?;
    let options = PyDict::new(py);
    options.set_item("non_blocking", true)?;
    buffers
        .borrow()
        .rows
        .bind(py)
        .call_method("copy_", (rows,), Some(&options))?;
    if captured {
        let slot = source.bind(py).call_method1("narrow", (0, 4 * count, 1))?;
        runner
            .borrow()
            .slot_index
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("denoising graph has no slot indices"))?
            .bind(py)
            .call_method("copy_", (slot,), Some(&options))?;
    }
    Ok(())
}

pub(super) fn capture(
    runner: &Bound<'_, DiffusionRunner>,
    sequence: &Bound<'_, DenoisingSequence>,
) -> PyResult<()> {
    let py = runner.py();
    require_bound(runner, sequence)?;
    if !DiffusionRunner::captures(runner.borrow()) {
        return Err(PyRuntimeError::new_err(
            "denoising graph capture requires a stream",
        ));
    }
    let (execution, device) = {
        let owner = runner.borrow();
        (
            owner.as_super().execution.clone_ref(py),
            owner.as_super().device.clone_ref(py),
        )
    };
    if execution.borrow(py).sealed {
        return Err(CUDAGraphError::new_err(
            "denoising capture is outside startup preparation",
        ));
    }
    let buffers = layout(runner, sequence.borrow().layout.bind(py))?;
    if !buffers.borrow().warmed {
        return Err(PyRuntimeError::new_err(
            "warm a layout before capturing its steps",
        ));
    }
    let bucket = bucket(runner, sequence, &buffers)?;
    if bucket.borrow().get(py, None).is_some() {
        return Ok(());
    }

    let context = execution.borrow(py).context.clone_ref(py);
    let graph = with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
        on_stream(context.bind(py), device.bind(py), || {
            // Startup uses scratch pages committed in bank one. Replays supply
            // the request's committed bank, page table and state slot.
            copy_indices(runner, &buffers, &sequence.borrow(), 1, true)?;
            let scope = execution
                .bind(py)
                .getattr("storage")?
                .call_method1("allocate", (&execution,))?;
            let temporal = with_context(&scope, || {
                let sequence = sequence.borrow();
                py.import("uniserve_worker.model_executor.cuda_graph")?
                    .call_method1(
                        "clone_inputs",
                        ((&sequence.schedules, sequence.temporal.bind(py).get_item(0)?),),
                    )
            })?;
            let forward = runner.call_method1("_graph_forward", (&buffers, sequence, &temporal))?;
            let pools = execution.borrow(py).pools.clone_ref(py);
            let graph = CUDAGraphRunner::capture(
                py,
                context.extract::<Py<ExecutionContext>>(py)?,
                temporal.unbind(),
                forward.get_item(0)?.unbind(),
                Some(pools.bind(py).as_any()),
                Some(forward.get_item(1)?.unbind()),
                false,
                None,
            )?;
            Bound::new(py, graph)
        })
    })?;
    bucket
        .borrow_mut()
        .graphs
        .insert(None, graph.into_any().unbind());
    Ok(())
}

pub(super) fn step(
    runner: &Bound<'_, DiffusionRunner>,
    sequence: &Bound<'_, DenoisingSequence>,
    index: usize,
    bank: i64,
) -> PyResult<(Py<PyAny>, &'static str)> {
    let py = runner.py();
    require_bound(runner, sequence)?;
    if index >= sequence.borrow().outputs.len() || !(0..=1).contains(&bank) {
        return Err(PyValueError::new_err(
            "denoising requires a valid solver step and sample bank",
        ));
    }
    let (execution, device) = {
        let owner = runner.borrow();
        (
            owner.as_super().execution.clone_ref(py),
            owner.as_super().device.clone_ref(py),
        )
    };
    let key = sequence.borrow().layout.clone_ref(py);
    let buffers = layout(runner, key.bind(py))?;
    let bucket = execution.borrow(py).buckets.bind(py).get_item(&key)?;
    let graph = match bucket {
        Some(bucket) => {
            check_signature(&buffers, &sequence.borrow())?;
            bucket.cast::<GraphBucket>()?.borrow().get(py, None)
        }
        None => None,
    };
    let context = execution.borrow(py).context.clone_ref(py);
    with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
        on_stream(context.bind(py), device.bind(py), || {
            copy_indices(runner, &buffers, &sequence.borrow(), bank, graph.is_some())?;
            match graph {
                Some(graph) => {
                    let inputs = sequence.borrow();
                    let temporal = (&inputs.schedules, inputs.temporal.bind(py).get_item(index)?);
                    graph
                        .bind(py)
                        .cast::<CUDAGraphRunner>()?
                        .borrow()
                        .replay(py, Some(temporal.into_pyobject(py)?.into_any().unbind()))?;
                    Ok((
                        inputs.outputs[index].bind(py).copy()?.into_any().unbind(),
                        "graph_replay",
                    ))
                }
                None => Ok((
                    runner
                        .call_method1("_eager_step", (&buffers, sequence, index))?
                        .unbind(),
                    "eager",
                )),
            }
        })
    })
}

fn allocate_layout<'py>(
    runner: &Bound<'py, DiffusionRunner>,
    layout: &Bound<'py, PyAny>,
    pages: usize,
    first: bool,
) -> PyResult<Bound<'py, DenoisingBuffers>> {
    let py = runner.py();
    let (workspace, model, device, context) = {
        let owner = runner.borrow();
        let pool = owner
            .pool
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("denoising runner has no sample pool"))?
            .borrow(py);
        let samples = owner
            .samples
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("denoising runner has no sample backing"))?;
        let rows: usize = samples.bind(py).getattr("shape")?.get_item(0)?.extract()?;
        // An empty sequence shard consumes no sample pages.
        if pages
            .checked_mul(pool.inner.page_units())
            .is_none_or(|size| size > rows)
        {
            return Err(PyValueError::new_err(
                "a layout's samples exceed the runner's pages",
            ));
        }
        let workspace = owner
            .workspace
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("denoising workspace is closed"))?;
        let base = owner.as_super();
        (
            workspace.clone_ref(py),
            base.model.clone_ref(py),
            base.device.clone_ref(py),
            base.execution.borrow(py).context.clone_ref(py),
        )
    };
    let requirements = buffer_configs(model.bind(py), "constant_buffers", layout)?;
    let backing = Py::new(
        py,
        TensorBuffers::allocate(py, &requirements, device.bind(py), false, None)?,
    )?;
    let prepared = (|| {
        let options = PyDict::new(py);
        let torch = py.import("torch")?;
        options.set_item("dtype", torch.getattr("int64")?)?;
        options.set_item("device", &device)?;
        let rows = torch
            .call_method("empty", ((2, pages),), Some(&options))?
            .unbind();
        let constants = backing.borrow(py).view(py, &requirements)?;
        let workspace_views = workspace.borrow(py).view(
            py,
            &buffer_configs(model.bind(py), "workspace_buffers", layout)?,
        )?;
        if first {
            let options = PyDict::new(py);
            options.set_item("constants", &backing)?;
            options.set_item("workspace", &workspace)?;
            context
                .bind(py)
                .call_method("prepare", (layout,), Some(&options))?;
        } else if requirements.is_truthy()? {
            let options = PyDict::new(py);
            options.set_item("out", &constants)?;
            with_context(&context.bind(py).call_method0("activate")?, || {
                model
                    .bind(py)
                    .call_method("prepare_constants", (layout,), Some(&options))
            })?;
        }
        Bound::new(
            py,
            DenoisingBuffers::new(
                py,
                backing.clone_ref(py),
                constants,
                workspace_views,
                pages,
                rows,
            ),
        )
    })();
    if prepared.is_err() {
        backing.borrow_mut(py).close(py);
    }
    prepared
}

fn state_views(
    runner: &Bound<'_, DiffusionRunner>,
    sequence: &DenoisingSequence,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let py = runner.py();
    let (bank, slots, backing) = {
        let owner = runner.borrow();
        let backing = owner
            .state_buffers
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("denoising state buffers are closed"))?;
        (owner.bank.clone_ref(py), owner.slots, backing.clone_ref(py))
    };
    let configs = PyDict::new(py);
    let config = py.import("uniserve.tensors")?.getattr("BufferConfig")?;
    for (name, (_, shape)) in &sequence.spans {
        let tensor = bank
            .bind(py)
            .get_item(name)?
            .ok_or_else(|| PyKeyError::new_err(name.clone()))?;
        configs.set_item(
            name,
            config.call1((PyTuple::new(py, shape)?, tensor.getattr("dtype")?))?,
        )?;
    }
    let views = backing.borrow(py).view(py, configs.as_any())?;
    let mut gathers = Vec::new();
    for (name, (start, _)) in &sequence.spans {
        let buffer = views.bind(py).get_item(name)?;
        let size: isize = buffer.call_method0("numel")?.extract()?;
        if size == 0 {
            continue;
        }
        let rows = bank
            .bind(py)
            .get_item(name)?
            .ok_or_else(|| PyKeyError::new_err(name.clone()))?
            .call_method1("view", (slots, -1))?;
        let rows = rows.get_item((PySlice::full(py), PySlice::new(py, *start, start + size, 1)))?;
        gathers.push((rows, buffer.call_method1("view", (1, -1))?));
    }
    Ok((views, PyTuple::new(py, gathers)?.into_any().unbind()))
}
