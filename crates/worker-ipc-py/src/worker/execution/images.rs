//! Packed vision capture and grouped replay on the execution's graph pool.

use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};
use uniserve_worker_ipc::ForwardStats as NativeStats;

use super::{Execution, GraphBucket};
use crate::stats::ForwardStats;
use crate::worker::cuda_graph::{CUDAGraphError, CUDAGraphRunner};
use crate::worker::host::with_context;
use crate::worker::model_results::ExecutionOutput;

// Bound retained activations independently of the worker's admitted batch size.
// Larger batches replay full groups and then their exact remaining slot count.
const MAX_IMAGES: usize = 16;

pub(super) fn capture(
    execution: &Bound<'_, Execution>,
    runner: &Bound<'_, PyAny>,
    max_images: usize,
    dtype: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let py = execution.py();
    if execution.borrow().image_capacity > 0 {
        return Ok(());
    }
    if execution.borrow().sealed {
        return Err(CUDAGraphError::new_err("packed capture is outside startup"));
    }

    let capacity = max_images.min(MAX_IMAGES);
    let storage = execution.getattr("storage")?;
    let buckets = execution.borrow().buckets.bind(py).clone();
    let buffer = with_context(&storage.call_method1("allocate", (execution,))?, || {
        runner.call_method1("allocate_packed", (capacity, dtype))
    })?;

    // All shapes share one pixel buffer and allocator pool. Largest first lets
    // smaller captures reuse GEMM workspaces without retaining earlier growth.
    for slots in (1..=capacity).rev() {
        let inputs = with_context(&storage.call_method1("allocate", (execution,))?, || {
            runner.call_method1("packed_inputs", (&buffer, slots))
        })?;
        let graph = runner.call_method1("capture_packed", (inputs,))?;
        if let Err(error) = storage.call_method0("check") {
            if let Err(cleanup) = graph.call_method0("close") {
                let _ = error.value(py).call_method1(
                    "add_note",
                    (format!("Graph cleanup also failed: {cleanup}"),),
                );
            }
            return Err(error);
        }
        let bucket = Py::new(py, GraphBucket::new(Some([(None, graph.unbind())].into())))?;
        buckets.set_item(("images", slots), bucket)?;
    }

    execution.borrow_mut().image_capacity = capacity;
    Ok(())
}

pub(super) fn encode(
    execution: &Bound<'_, Execution>,
    runner: &Bound<'_, PyAny>,
    inputs: &Bound<'_, PyAny>,
) -> PyResult<Py<ExecutionOutput>> {
    let py = execution.py();
    let capacity = execution.borrow().image_capacity;
    if capacity == 0 {
        return Err(CUDAGraphError::new_err(
            "packed vision graphs are not resident",
        ));
    }

    let images = inputs.getattr("images")?;
    let shapes = inputs.getattr("grid_shapes")?;
    let count = images.len()?;
    let buckets = execution.borrow().buckets.bind(py).clone();
    let mut values = Vec::with_capacity(count);

    for start in (0..count).step_by(capacity) {
        let stop = (start + capacity).min(count);
        let slice = PySlice::new(py, start as isize, stop as isize, 1);
        let images = images.get_item(&slice)?;
        let shapes = shapes.get_item(&slice)?;
        let bucket = buckets.as_any().get_item(("images", stop - start))?;
        let graph = bucket
            .cast::<GraphBucket>()?
            .borrow()
            .__getitem__(py, None)?;
        runner.call_method1(
            "prepare_packed",
            (
                graph.getattr(py, "inputs")?.getattr(py, "value")?,
                images,
                &shapes,
            ),
        )?;
        let output = graph
            .bind(py)
            .cast::<CUDAGraphRunner>()?
            .borrow()
            .replay(py, None)?;
        let features = runner.call_method1("unpack_packed", (output, shapes))?;
        values.extend(features.cast::<PyTuple>()?.iter().map(Bound::unbind));
    }

    let output = py
        .get_type::<ExecutionOutput>()
        .call1((PyTuple::new(py, values)?,))?;
    let output = output.cast_into::<ExecutionOutput>()?;
    let replays = count.div_ceil(capacity) as u64;
    output.borrow_mut().stats = Some(Py::new(
        py,
        ForwardStats::from(NativeStats {
            cuda_graph_runtime_mode_counts: [("graph_replay".into(), replays)].into(),
            cuda_graph_replays: replays,
            ..NativeStats::default()
        }),
    )?);
    Ok(output.unbind())
}
