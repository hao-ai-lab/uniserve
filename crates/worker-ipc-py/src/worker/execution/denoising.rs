//! Standalone denoising layout lifetime, preparation and graph dispatch.

use std::collections::HashMap;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::{Execution, GraphBucket, close_all, on_stream};
use crate::worker::cuda_graph::{CUDAGraphError, CUDAGraphRunner};
use crate::worker::host::with_context;

/// Numerical backing for one layout. Graph input views borrow the execution's
/// shared state buffers; constants remain live until queued readers retire.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct DenoisingBuffers {
    #[pyo3(get)]
    backing: Py<PyAny>,
    #[pyo3(get)]
    constants: Py<PyAny>,
    #[pyo3(get)]
    workspace: Py<PyAny>,
    #[pyo3(get)]
    pages: usize,
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
    #[new]
    fn new(
        py: Python<'_>,
        backing: Py<PyAny>,
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

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for value in [
            &self.backing,
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

pub(super) struct Denoising {
    maximum: Py<PyAny>,
    layouts: Py<PyDict>,
    // Immutable pinned sources outlive asynchronous copies. Numerical tensor
    // construction stays with the backend; Rust owns lookup and retention.
    rows: HashMap<(i64, Vec<i64>), Py<PyAny>>,
    slots: HashMap<usize, Py<PyAny>>,
}

impl Denoising {
    pub(super) fn new(py: Python<'_>, maximum: Py<PyAny>) -> Self {
        Self {
            maximum,
            layouts: PyDict::new(py).unbind(),
            rows: HashMap::new(),
            slots: HashMap::new(),
        }
    }

    pub(super) fn contains(&self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.layouts.bind(py).contains(layout)
    }

    pub(super) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.maximum)?;
        visit.call(&self.layouts)?;
        for source in self.rows.values().chain(self.slots.values()) {
            visit.call(source)?;
        }
        Ok(())
    }

    pub(super) fn close(self, py: Python<'_>) -> PyResult<()> {
        close_all(
            py,
            self.layouts.bind(py).values().iter().map(|layout| {
                layout
                    .cast::<DenoisingBuffers>()?
                    .borrow()
                    .backing
                    .call_method0(py, "close")
                    .map(drop)
            }),
        )
    }
}

fn layouts<'py>(execution: &Bound<'py, Execution>) -> PyResult<Bound<'py, PyDict>> {
    execution
        .borrow()
        .denoising
        .as_ref()
        .map(|value| value.layouts.bind(execution.py()).clone())
        .ok_or_else(|| PyRuntimeError::new_err("denoising buffers are not configured"))
}

pub(super) fn layout<'py>(
    execution: &Bound<'py, Execution>,
    key: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, DenoisingBuffers>> {
    layouts(execution)?
        .get_item(key)?
        .ok_or_else(|| PyKeyError::new_err(key.clone().unbind()))?
        .cast_into()
        .map_err(Into::into)
}

pub(super) fn prepare<'py>(
    execution: &Bound<'py, Execution>,
    runner: &Bound<'py, PyAny>,
    key: &Bound<'py, PyAny>,
    pages: usize,
) -> PyResult<Bound<'py, DenoisingBuffers>> {
    let (entries, maximum) = {
        let owner = execution.borrow();
        let buffers = owner
            .denoising
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("denoising buffers are not configured"))?;
        (
            buffers.layouts.bind(execution.py()).clone(),
            buffers.maximum.clone_ref(execution.py()),
        )
    };
    if let Some(value) = entries.get_item(key)? {
        return Ok(value.cast_into()?);
    }

    let first = entries.is_empty();
    if first && !maximum.bind(execution.py()).eq(key)? {
        return Err(PyValueError::new_err(
            "a denoiser runner prepares its maximum first",
        ));
    }

    let allocate = || runner.call_method1("_prepare_layout", (key, pages, first));
    // Captured intermediates use the pool's free blocks. Later layouts must
    // allocate outside that pool so their constants cannot alias replay work.
    let buffers = if execution.borrow().buckets.bind(execution.py()).is_empty() {
        let storage = execution.getattr("storage")?;
        with_context(&storage.call_method1("allocate", (execution,))?, allocate)?
    } else {
        allocate()?
    }
    .cast_into::<DenoisingBuffers>()?;
    entries.set_item(key, &buffers)?;
    Ok(buffers)
}

pub(super) fn retire(execution: &Bound<'_, Execution>, key: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = execution.py();
    let maximum = execution
        .borrow()
        .denoising
        .as_ref()
        .map(|value| value.maximum.clone_ref(py));
    let bucket = execution.borrow().buckets.bind(py).get_item(key)?;
    let maximum = maximum
        .map(|maximum| maximum.bind(py).eq(key))
        .transpose()?
        .unwrap_or(false);
    let captured = bucket
        .map(|bucket| -> PyResult<bool> {
            Ok(bucket
                .cast::<GraphBucket>()?
                .borrow()
                .graphs
                .contains_key(&None))
        })
        .transpose()?
        .unwrap_or(false);
    if maximum || captured {
        return Err(PyValueError::new_err(
            "a captured or maximum layout stays prepared",
        ));
    }

    let entries = layouts(execution)?;
    let Some(value) = entries.get_item(key)? else {
        return Ok(());
    };

    // A retired layout's constants may still be read by queued numerical
    // work. Complete those readers before releasing the backing allocation.
    let stream = execution.borrow().context.bind(py).getattr("stream")?;
    if !stream.is_none() {
        stream.call_method0("synchronize")?;
    }
    entries.del_item(key)?;
    value
        .cast::<DenoisingBuffers>()?
        .borrow()
        .backing
        .call_method0(py, "close")
        .map(drop)
}

pub(super) fn binds(
    execution: &Bound<'_, Execution>,
    runner: &Bound<'_, PyAny>,
    ladder: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    Ok(ladder.getattr("samples")?.is(runner.getattr("samples")?)
        && layouts(execution)?.contains(ladder.getattr("layout")?)?)
}

fn require_bound(
    execution: &Bound<'_, Execution>,
    runner: &Bound<'_, PyAny>,
    ladder: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if !binds(execution, runner, ladder)? {
        return Err(PyValueError::new_err(
            "the ladder was bound by another runner",
        ));
    }
    Ok(())
}

pub(super) fn warm(
    execution: &Bound<'_, Execution>,
    runner: &Bound<'_, PyAny>,
    ladder: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let py = execution.py();
    require_bound(execution, runner, ladder)?;
    let entry = layout(execution, &ladder.getattr("layout")?)?;
    if entry.borrow().warmed {
        return Ok(());
    }
    if !execution.borrow().buckets.bind(py).is_empty() {
        return Err(PyRuntimeError::new_err(
            "warm every denoiser layout before capturing any",
        ));
    }

    let context = execution.borrow().context.bind(py).clone();
    let device = runner.getattr("device")?;
    with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
        on_stream(&context, &device, || {
            with_context(
                &execution
                    .getattr("storage")?
                    .call_method1("allocate", (execution,))?,
                || {
                    runner
                        .call_method1("_warm_step", (&entry, ladder))
                        .map(drop)
                },
            )
        })
    })?;
    if device.getattr("type")?.extract::<String>()? == "cuda" {
        let stream = context.getattr("stream")?;
        let stream = if stream.is_none() {
            py.import("torch.cuda")?
                .call_method1("current_stream", (&device,))?
        } else {
            stream
        };
        stream.call_method0("synchronize")?;
    }
    entry.borrow_mut().warmed = true;
    Ok(())
}

fn bucket<'py>(
    execution: &Bound<'py, Execution>,
    runner: &Bound<'py, PyAny>,
    ladder: &Bound<'py, PyAny>,
    entry: &Bound<'py, DenoisingBuffers>,
) -> PyResult<Bound<'py, GraphBucket>> {
    let py = execution.py();
    let key = ladder.getattr("layout")?;
    let buckets = execution.borrow().buckets.bind(py).clone();
    if let Some(bucket) = buckets.get_item(&key)? {
        check_signature(entry, ladder)?;
        return Ok(bucket.cast_into()?);
    }

    let views = runner.call_method1("_state_views", (ladder,))?;
    let state = views.get_item(0)?.unbind();
    let gathers = views.get_item(1)?.unbind();
    let signature = ladder.getattr("signature")?.unbind();
    {
        let mut buffers = entry.borrow_mut();
        buffers.state = state;
        buffers.gathers = gathers;
        buffers.signature = Some(signature);
    }
    let bucket = Bound::new(py, GraphBucket::new(None))?;
    buckets.set_item(key, &bucket)?;
    Ok(bucket)
}

fn check_signature(entry: &Bound<'_, DenoisingBuffers>, ladder: &Bound<'_, PyAny>) -> PyResult<()> {
    if let Some(signature) = &entry.borrow().signature
        && !signature
            .bind(entry.py())
            .eq(ladder.getattr("signature")?)?
    {
        return Err(PyValueError::new_err(
            "a ladder's structure differs from its layout's captured steps",
        ));
    }
    Ok(())
}

fn copy_indices(
    execution: &Bound<'_, Execution>,
    runner: &Bound<'_, PyAny>,
    entry: &Bound<'_, DenoisingBuffers>,
    ladder: &Bound<'_, PyAny>,
    bank: i64,
    captured: bool,
) -> PyResult<()> {
    let py = execution.py();
    let pages = ladder.getattr("pages")?;
    let key = (bank, pages.extract::<Vec<i64>>()?);
    let cached = execution
        .borrow()
        .denoising
        .as_ref()
        .and_then(|buffers| buffers.rows.get(&key))
        .map(|value| value.clone_ref(py));
    let rows = match cached {
        Some(value) => value,
        None => {
            let value = runner.call_method1("_row_source", (bank, pages))?.unbind();
            execution
                .borrow_mut()
                .denoising
                .as_mut()
                .ok_or_else(|| {
                    PyRuntimeError::new_err(
                        "denoising execution was closed during input preparation",
                    )
                })?
                .rows
                .insert(key, value.clone_ref(py));
            value
        }
    };

    let slot = if captured {
        let slot: usize = ladder.getattr("slot")?.extract()?;
        let cached = execution
            .borrow()
            .denoising
            .as_ref()
            .and_then(|buffers| buffers.slots.get(&slot))
            .map(|value| value.clone_ref(py));
        let source = match cached {
            Some(value) => value,
            None => {
                let value = runner.call_method1("_slot_source", (slot,))?.unbind();
                execution
                    .borrow_mut()
                    .denoising
                    .as_mut()
                    .ok_or_else(|| {
                        PyRuntimeError::new_err(
                            "denoising execution was closed during input preparation",
                        )
                    })?
                    .slots
                    .insert(slot, value.clone_ref(py));
                value
            }
        };
        Some(source)
    } else {
        None
    };

    runner.call_method1("_copy_indices", (entry, rows, slot))?;
    Ok(())
}

pub(super) fn capture(
    execution: &Bound<'_, Execution>,
    runner: &Bound<'_, PyAny>,
    ladder: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let py = execution.py();
    if !runner.getattr("captures")?.is_truthy()? {
        return Err(PyRuntimeError::new_err(
            "denoising graph capture requires a stream",
        ));
    }
    if execution.borrow().sealed {
        return Err(CUDAGraphError::new_err(
            "denoising capture is outside startup preparation",
        ));
    }
    let entry = layout(execution, &ladder.getattr("layout")?)?;
    if !entry.borrow().warmed {
        return Err(PyRuntimeError::new_err(
            "warm a layout before capturing its steps",
        ));
    }
    let bucket = bucket(execution, runner, ladder, &entry)?;
    if bucket.borrow().graphs.contains_key(&None) {
        return Ok(());
    }

    let context = execution.borrow().context.bind(py).clone();
    let device = runner.getattr("device")?;
    let graph = with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
        on_stream(&context, &device, || {
            // Capture uses scratch request pages committed in bank one. Every
            // replay supplies its own committed bank, pages and request slot.
            copy_indices(execution, runner, &entry, ladder, 1, true)?;
            runner.call_method1("_capture_step", (&entry, ladder))
        })
    })?;
    bucket.borrow_mut().graphs.insert(None, graph.unbind());
    Ok(())
}

pub(super) fn step(
    execution: &Bound<'_, Execution>,
    runner: &Bound<'_, PyAny>,
    ladder: &Bound<'_, PyAny>,
    index: usize,
    bank: i64,
) -> PyResult<(Py<PyAny>, &'static str)> {
    let py = execution.py();
    require_bound(execution, runner, ladder)?;
    let key = ladder.getattr("layout")?;
    let entry = layout(execution, &key)?;
    let bucket = execution.borrow().buckets.bind(py).get_item(&key)?;
    let graph = match bucket {
        Some(bucket) => {
            check_signature(&entry, ladder)?;
            bucket.cast::<GraphBucket>()?.borrow().get(py, None)
        }
        None => None,
    };

    let context = execution.borrow().context.bind(py).clone();
    let device = runner.getattr("device")?;
    with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
        on_stream(&context, &device, || {
            copy_indices(execution, runner, &entry, ladder, bank, graph.is_some())?;
            match graph {
                Some(graph) => {
                    let temporal = (
                        ladder.getattr("schedules")?,
                        ladder.getattr("temporal")?.get_item(index)?,
                    );
                    graph
                        .bind(py)
                        .cast::<CUDAGraphRunner>()?
                        .borrow()
                        .replay(py, Some(temporal.into_pyobject(py)?.into_any().unbind()))?;
                    Ok((
                        runner
                            .call_method1("_step_values", (ladder, index))?
                            .unbind(),
                        "graph_replay",
                    ))
                }
                None => Ok((
                    runner
                        .call_method1("_eager_step", (&entry, ladder, index))?
                        .unbind(),
                    "eager",
                )),
            }
        })
    })
}
