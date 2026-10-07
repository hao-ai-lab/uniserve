//! Canvas graph buckets, readout tails and per-execution sampler workspace.

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::{ModelRunner, joining_experts};
use crate::worker::canvas_slots::CanvasSlots;
use crate::worker::cuda_graph::{CUDAGraphError, CUDAGraphRunner, batch as graph_batch};
use crate::worker::execution::{Execution, GraphBucket};
use crate::worker::execution_context::ExecutionContext;
use crate::worker::graph_storage::GraphStorage;
use crate::worker::input_buffers::InputBuffers;
use crate::worker::model_inputs::InputBatch;
use crate::worker::model_results::ExecutionOutput;

// After four rows, each step adds at most half the previous count. Include the
// actual capacity too, so padding never exceeds one third of a bucket's rows.
const ROW_BUCKETS: [usize; 14] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128];
// Independent readout slots are chunked at the largest tail bucket.
pub(in crate::worker) const SLOT_BUCKETS: [usize; 3] = [64, 128, 256];

/// Canvas passes and their independent slot readouts share one execution lane.
/// Request state is borrowed; writable sampler scratch belongs to this runner.
#[pyclass(extends = ModelRunner, subclass, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct CanvasRunner {
    #[pyo3(get)]
    pipeline: Py<PyAny>,
    #[pyo3(get)]
    canvas_length: usize,
    #[pyo3(get)]
    canvas_slots: Option<Py<CanvasSlots>>,
    #[pyo3(get)]
    sampler_workspace: Option<Py<PyAny>>,
    #[pyo3(get)]
    step_rows: usize,
    #[pyo3(get, set)]
    pool_rows: Option<usize>,
    #[pyo3(get)]
    local_tail: bool,
    readout_state: Option<Py<PyTuple>>,
}

#[pymethods]
impl CanvasRunner {
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
        let mesh = model.getattr("backbone")?.getattr("mesh")?;
        let axes = mesh.getattr("axes")?;
        let pipeline = mesh
            .call_method1(
                "get_group",
                (if axes.contains("pp")? {
                    "pp".into_pyobject(py)?.into_any()
                } else {
                    PyTuple::empty(py).into_any()
                },),
            )?
            .unbind();
        let local_tail = !backend(py)?
            .call_method1("tail_exchanges", (&model, context.borrow(py).experts(py)))?
            .extract::<bool>()?;
        let canvas_length = model.getattr("canvas")?.getattr("length")?.extract()?;
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
            pipeline,
            canvas_length,
            canvas_slots: None,
            sampler_workspace: None,
            step_rows: 0,
            pool_rows: None,
            local_tail,
            readout_state: None,
        }))
    }

    #[getter]
    fn max_canvases(slf: PyRef<'_, Self>) -> PyResult<usize> {
        slf.capacity(slf.py(), slf.as_super())
    }

    #[getter]
    fn canvas_rows(slf: PyRef<'_, Self>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(slf.py(), slf.row_buckets(slf.py(), slf.as_super())?)?.unbind())
    }

    #[getter]
    fn readout_lengths(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.length_buckets())?.unbind())
    }

    fn capture_plan(slf: PyRef<'_, Self>) -> PyResult<Py<PyTuple>> {
        let py = slf.py();
        let constants = slf
            .canvas_slots
            .as_ref()
            .map(|slots| slots.borrow(py).constants.clone_ref(py));
        Ok((
            slf.as_super().capture_plan(py)?,
            PyTuple::new(py, slf.row_buckets(py, slf.as_super())?)?,
            slf.readout_lengths(py)?,
            constants,
        )
            .into_pyobject(py)?
            .unbind())
    }

    /// Bind request banks while keeping concurrent lanes' sampler scratch apart.
    fn bind_canvas_slots(slf: &Bound<'_, Self>, slots: Py<CanvasSlots>) -> PyResult<()> {
        let py = slf.py();
        let (buffers, device, maximum) = {
            let owner = slf.borrow();
            let base = owner.as_super();
            (
                input_buffers(base, py)?,
                base.device.clone_ref(py),
                owner.capacity(py, base)?,
            )
        };
        InputBuffers::bind_canvas(buffers.bind(py), slots.clone_ref(py))?;
        let state = slots.borrow(py);
        let rows = CanvasSlots::step_rows(
            state.canvas_length,
            state.vocab_size,
            maximum.min(state.request_pool_size),
        )?;
        let options = PyDict::new(py);
        options.set_item(
            "dtype",
            state
                .banks
                .bind(py)
                .get_item("self_conditioning")?
                .ok_or_else(|| PyValueError::new_err("canvas state has no self conditioning"))?
                .getattr("dtype")?,
        )?;
        options.set_item("device", device)?;
        let workspace = py
            .import("uniserve.diffusion.canvas")?
            .getattr("CanvasWorkspace")?
            .call_method(
                "empty",
                (
                    rows,
                    state.canvas_length,
                    state.vocab_size,
                    state.hidden_size,
                ),
                Some(&options),
            )?
            .unbind();
        drop(state);
        let mut owner = slf.borrow_mut();
        owner.canvas_slots = Some(slots);
        owner.sampler_workspace = Some(workspace);
        owner.step_rows = rows;
        Ok(())
    }

    fn close_graphs(slf: &Bound<'_, Self>) -> PyResult<()> {
        let execution = slf.borrow().as_super().execution.clone_ref(slf.py());
        let result = Execution::close_graphs(execution.bind(slf.py()));
        slf.borrow_mut().readout_state = None;
        result
    }

    fn close(slf: &Bound<'_, Self>) -> PyResult<()> {
        let result = ModelRunner::close(slf.as_any().cast()?);
        slf.borrow_mut().sampler_workspace = None;
        result
    }

    fn graph_tokens(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        let length = if key.get_item(0)?.extract::<&str>()? == "canvas" {
            key.get_item(2)?.extract()?
        } else {
            self.canvas_length
        };
        Ok(key.get_item(1)?.extract::<usize>()? * length)
    }

    fn expert_tokens(slf: PyRef<'_, Self>, batch: PyRef<'_, InputBatch>) -> PyResult<usize> {
        let py = slf.py();
        let tokens = batch
            .query_tokens
            .ok_or_else(|| PyValueError::new_err("expert execution requires host query lengths"))?;
        let inputs = batch.inputs.bind(py);
        let readout = py
            .import("uniserve_worker.model_executor.input_batch")?
            .getattr("ReadoutInput")?;
        if !slf.local_tail && inputs.is_instance(&readout)? {
            Ok(tokens.max(
                inputs
                    .getattr("slot_tokens")?
                    .call_method0("numel")?
                    .extract()?,
            ))
        } else {
            Ok(tokens)
        }
    }

    /// Pad each homogeneous call to the smallest resident row and length bucket.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (batch, *, eligible))]
    fn select_graph_shape(
        slf: PyRef<'_, Self>,
        batch: Py<InputBatch>,
        eligible: bool,
    ) -> PyResult<Option<(Py<PyTuple>, Py<InputBatch>, bool)>> {
        let py = slf.py();
        let base = slf.as_super();
        if !eligible || base.execution.borrow(py).pools.bind(py).is_empty() {
            return Ok(None);
        }
        let value = batch.borrow(py);
        let inputs = value.inputs.bind(py);
        let step = inputs.is_instance(
            &py.import("uniserve_worker.model_executor.input_batch")?
                .getattr("CanvasStepInput")?,
        )?;
        let lengths: Vec<usize> = inputs
            .getattr("attention")?
            .getattr("queries")?
            .getattr("host")?
            .extract()?;
        let maximum = lengths
            .into_iter()
            .max()
            .ok_or_else(|| PyValueError::new_err("canvas graph requires query lengths"))?;
        let length = if step {
            slf.canvas_length
        } else {
            slf.length_buckets()
                .into_iter()
                .find(|&length| length >= maximum)
                .unwrap_or(slf.canvas_length)
        };
        let buckets = slf.row_buckets(py, base)?;
        let rows = buckets
            .iter()
            .copied()
            .find(|&rows| rows >= value.row_count);
        let rows = rows.filter(|_| maximum <= length).ok_or_else(|| {
            CUDAGraphError::new_err(format!(
                "{} has no canvas graph for {} canvases of up to {maximum} tokens; \
                 capacity is {} canvases of {length} tokens",
                base.name,
                value.row_count,
                buckets.last().copied().unwrap_or(0),
            ))
        })?;

        let buffers = input_buffers(base, py)?;
        let widths = PyTuple::new(py, &buffers.borrow(py).table_widths)?;
        drop(value);
        let padded: Py<InputBatch> = backend(py)?
            .call_method1(
                if step { "_pad_steps" } else { "_pad_readout" },
                (batch, rows, length, widths, buffers),
            )?
            .extract()?;
        let key = if step {
            let values = padded.borrow(py).inputs.bind(py).clone();
            (
                "canvas_step",
                rows,
                values.getattr("sampling")?.get_item(0)?,
                values.getattr("first")?,
            )
                .into_pyobject(py)?
                .unbind()
        } else {
            ("canvas", rows, length).into_pyobject(py)?.unbind()
        };
        Ok(Some((key, padded, true)))
    }

    /// Attend graphs stop before the readout tail; eager warmup still joins all
    /// expert layers so expert-only ranks complete the same startup sequence.
    fn capture_graph(
        slf: &Bound<'_, Self>,
        key: &Bound<'_, PyAny>,
        execution: Py<InputBatch>,
        forward: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let base = slf.as_any().cast::<ModelRunner>()?;
        if key.get_item(0)?.extract::<&str>()? != "canvas" {
            return ModelRunner::capture_graph(base, key, execution, forward);
        }
        let (owner, cache, pipeline, local) = {
            let runner = slf.borrow();
            (
                runner.as_super().execution.clone_ref(py),
                runner.as_super().cache.clone_ref(py),
                runner.pipeline.clone_ref(py),
                runner.local_tail,
            )
        };
        let (context, pools) = {
            let owner = owner.borrow(py);
            (
                owner.context.extract::<Py<ExecutionContext>>(py)?,
                owner.pools.clone_ref(py),
            )
        };
        let partial = py.import("functools")?.getattr("partial")?;
        let attend = partial
            .call1((wrap_pyfunction!(attend, py)?, slf))?
            .unbind();
        let joined = joining_experts(py, attend.clone_ref(py), context.clone_ref(py))?;
        let warmup = partial
            .call1((owner.bind(py).getattr("warm_experts")?, joined))?
            .unbind();
        let graph = Py::new(
            py,
            graph_batch::capture_hidden(
                py,
                context,
                execution.clone_ref(py),
                attend,
                Some(pools.bind(py).as_any()),
                Some(cache),
                Some(warmup),
            )?,
        )?;
        if local && last_stage(pipeline.bind(py))? {
            let missing = SLOT_BUCKETS.iter().try_fold(false, |missing, &slots| {
                Ok::<_, PyErr>(
                    missing
                        || !owner
                            .borrow(py)
                            .buckets
                            .bind(py)
                            .contains(("canvas_tail", slots))?,
                )
            })?;
            if missing {
                let captured = (|| {
                    let state = graph_batch::replay_hidden(graph.borrow(py), execution.bind(py))?;
                    capture_tails(slf, state.bind(py))
                })();
                if let Err(error) = captured {
                    graph.borrow(py).close(py)?;
                    return Err(error);
                }
            }
        }
        Ok(graph.into_any())
    }

    #[pyo3(signature = (key, execution, batch, *, borrow))]
    fn replay_graph(
        slf: &Bound<'_, Self>,
        key: &Bound<'_, PyAny>,
        execution: Py<InputBatch>,
        batch: Py<InputBatch>,
        borrow: bool,
    ) -> PyResult<Py<ExecutionOutput>> {
        let py = slf.py();
        let base = slf.as_any().cast::<ModelRunner>()?;
        if key.get_item(0)?.extract::<&str>()? != "canvas" {
            return base
                .borrow()
                .replay_graph(py, key, execution, batch.borrow(py), borrow);
        }
        let graph: Py<CUDAGraphRunner> = base.borrow().batch_graph(py, key)?.extract(py)?;
        let state = graph_batch::replay_hidden(graph.borrow(py), execution.bind(py))?;
        let context: Py<ExecutionContext> =
            base.borrow().execution.borrow(py).context.extract(py)?;
        let inputs = batch.borrow(py).inputs.clone_ref(py);
        ExecutionContext::with_active(context.bind(py), || {
            if slf.borrow().local_tail {
                replay_tails(slf, state.bind(py), inputs.bind(py))
            } else {
                Ok(slf.call_method1("readout", (state, inputs))?.extract()?)
            }
        })
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.pipeline)?;
        visit.call(&self.canvas_slots)?;
        visit.call(&self.sampler_workspace)?;
        visit.call(&self.readout_state)
    }
}

impl CanvasRunner {
    /// Readout buckets may need two sequences per canvas. These are numerical
    /// input rows; they consume no additional scheduler request slots.
    pub(in crate::worker) fn input_rows(canvases: usize) -> usize {
        2 * canvases
    }

    fn capacity(&self, py: Python<'_>, base: &ModelRunner) -> PyResult<usize> {
        let inputs = input_buffers(base, py)?;
        let inputs = inputs.borrow(py);
        let rows = if base.execution.borrow(py).pools.bind(py).is_empty() {
            inputs.max_rows
        } else {
            inputs.max_rows / 2
        };
        Ok(rows
            .min(self.pool_rows.unwrap_or(rows))
            .min(inputs.max_tokens / self.canvas_length))
    }

    fn row_buckets(&self, py: Python<'_>, base: &ModelRunner) -> PyResult<Vec<usize>> {
        let mut limit = self.capacity(py, base)?;
        if let Some(slots) = &self.canvas_slots {
            limit = limit.min(slots.borrow(py).request_pool_size);
        }
        let mut rows = ROW_BUCKETS
            .into_iter()
            .filter(|&rows| rows < limit)
            .collect::<Vec<_>>();
        if limit > 0 {
            rows.push(limit);
        }
        Ok(rows)
    }

    fn length_buckets(&self) -> Vec<usize> {
        let mut length = 16;
        let mut lengths = Vec::new();
        while length < self.canvas_length {
            lengths.push(length);
            length *= 2;
        }
        lengths.push(self.canvas_length);
        lengths
    }
}

#[pyfunction]
fn attend(slf: &Bound<'_, CanvasRunner>, batch: PyRef<'_, InputBatch>) -> PyResult<Py<PyTuple>> {
    let py = slf.py();
    let (model, context) = {
        let owner = slf.borrow();
        let base = owner.as_super();
        (
            base.model.clone_ref(py),
            base.execution.borrow(py).context.clone_ref(py),
        )
    };
    let experts = context.bind(py).getattr("experts")?;
    if !experts.is_none() {
        experts.call_method0("reset_layers")?;
    }
    let state: Py<PyTuple> = model
        .call_method1(py, "attend", (&batch.inputs,))?
        .extract(py)?;
    let backing = slf
        .borrow()
        .readout_state
        .as_ref()
        .map(|state| state.clone_ref(py));
    let Some(backing) = backing else {
        if py
            .import("torch.cuda")?
            .call_method0("is_current_stream_capturing")?
            .extract::<bool>()?
        {
            slf.borrow_mut().readout_state = Some(state.clone_ref(py));
        }
        return Ok(state);
    };
    if state.bind(py).len() != backing.bind(py).len() {
        return Err(CUDAGraphError::new_err(
            "canvas attend output count changed between graph buckets",
        ));
    }
    let values = state
        .bind(py)
        .iter()
        .zip(backing.bind(py))
        .map(|(value, target)| super::copy_graph_output(&value, &target))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, values)?.unbind())
}

fn capture_tails(slf: &Bound<'_, CanvasRunner>, state: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = slf.py();
    let (execution, device) = {
        let owner = slf.borrow();
        (
            owner.as_super().execution.clone_ref(py),
            owner.as_super().device.clone_ref(py),
        )
    };
    let (context, pools, buckets) = {
        let owner = execution.borrow(py);
        (
            owner.context.extract::<Py<ExecutionContext>>(py)?,
            owner.pools.clone_ref(py),
            owner.buckets.clone_ref(py),
        )
    };
    for slots in SLOT_BUCKETS {
        let key = ("canvas_tail", slots);
        if buckets.bind(py).contains(key)? {
            continue;
        }
        // Ordinary allocation protects these rows from other graphs' transient
        // reuse of their shared graph pool.
        let inputs = backend(py)?
            .call_method1("tail_inputs", (state, slots, &device))?
            .unbind();
        let graph = Py::new(
            py,
            CUDAGraphRunner::capture(
                py,
                context.clone_ref(py),
                inputs,
                slf.getattr("_tail")?.unbind(),
                Some(pools.bind(py).as_any()),
                None,
                true,
                None,
            )?,
        )?;
        let bucket = GraphBucket::new(Some([(None, graph.into_any())].into()));
        buckets.bind(py).set_item(key, Py::new(py, bucket)?)?;
    }
    Ok(())
}

fn replay_tails(
    slf: &Bound<'_, CanvasRunner>,
    state: &Bound<'_, PyAny>,
    inputs: &Bound<'_, PyAny>,
) -> PyResult<Py<ExecutionOutput>> {
    let py = slf.py();
    let (pipeline, execution, name) = {
        let owner = slf.borrow();
        (
            owner.pipeline.clone_ref(py),
            owner.as_super().execution.clone_ref(py),
            owner.as_super().name.clone(),
        )
    };
    if !last_stage(pipeline.bind(py))? {
        return Ok(slf
            .call_method1("_broadcast_readout", (py.None(), state, inputs))?
            .extract()?);
    }
    let total: usize = inputs
        .getattr("slot_tokens")?
        .call_method0("numel")?
        .extract()?;
    let mut parts = Vec::new();
    for start in (0..total).step_by(SLOT_BUCKETS[2]) {
        let live = (total - start).min(SLOT_BUCKETS[2]);
        let slots = SLOT_BUCKETS
            .into_iter()
            .find(|&slots| slots >= live)
            .unwrap_or(SLOT_BUCKETS[2]);
        let bucket = execution
            .borrow(py)
            .buckets
            .bind(py)
            .get_item(("canvas_tail", slots))?
            .ok_or_else(|| {
                CUDAGraphError::new_err(format!(
                    "{name} has no readout tail graph of {slots} slots"
                ))
            })?;
        let tail = bucket.get_item(py.None())?.cast_into::<CUDAGraphRunner>()?;
        let static_inputs = tail.borrow().inputs.borrow(py).value.clone_ref(py);
        backend(py)?.call_method1("gather_tail", (state, &static_inputs, inputs, start, live))?;
        let normalized = tail.borrow().replay(py, None)?;
        parts
            .push(backend(py)?.call_method1("tail_candidates", (normalized, inputs, start, live))?);
    }
    let values = if parts.len() == 1 {
        parts.remove(0)
    } else {
        py.import("torch")?
            .call_method1("cat", (PyTuple::new(py, parts)?,))?
    };
    Ok(slf
        .call_method1("_broadcast_readout", (values, state, inputs))?
        .extract()?)
}

fn input_buffers(base: &ModelRunner, py: Python<'_>) -> PyResult<Py<InputBuffers>> {
    base.input_buffers
        .as_ref()
        .map(|buffers| buffers.clone_ref(py))
        .ok_or_else(|| CUDAGraphError::new_err("canvas graphs require input buffers"))
}

fn last_stage(pipeline: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(pipeline.getattr("rank")?.extract::<usize>()? + 1
        == pipeline.getattr("size")?.extract::<usize>()?)
}

fn backend(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("uniserve_worker.model_executor.canvas_runner")
}
