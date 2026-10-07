//! Request step inputs and the views a captured denoiser reads.

use std::collections::{BTreeMap, HashMap, HashSet};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::DiffusionRunner;
use crate::worker::cuda_graph::GraphInputs;

/// A request's numerical steps over its sample pages and state slot.
/// Tensor values stay live; static shapes and parameters are fixed at binding.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct DenoisingSequence {
    #[pyo3(get)]
    pub(super) layout: Py<PyAny>,
    #[pyo3(get)]
    pub(super) inputs: Py<PyTuple>,
    #[pyo3(get)]
    pub(super) schedules: Py<PyAny>,
    #[pyo3(get)]
    state: Py<PyAny>,
    pub(super) slot: usize,
    pub(super) pages: Vec<i64>,
    pub(super) signature: Py<PyAny>,
    #[pyo3(get)]
    fields: HashMap<usize, String>,
    // Element offsets within one contiguous request-bank row.
    pub(super) spans: BTreeMap<String, (isize, Vec<usize>)>,
    #[pyo3(get)]
    pub(super) temporal: Py<PyTuple>,
    pub(super) samples: Py<PyAny>,
    pub(super) outputs: Vec<Py<PyDict>>,
}

#[pymethods]
impl DenoisingSequence {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.layout)?;
        visit.call(&self.inputs)?;
        visit.call(&self.schedules)?;
        visit.call(&self.state)?;
        visit.call(&self.signature)?;
        visit.call(&self.temporal)?;
        visit.call(&self.samples)?;
        for output in &self.outputs {
            visit.call(output)?;
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn bind(
    runner: &Bound<'_, DiffusionRunner>,
    layout: Py<PyAny>,
    inputs: &Bound<'_, PyAny>,
    schedules: Py<PyAny>,
    state: Py<PyAny>,
    slot: usize,
    pages: Vec<i64>,
) -> PyResult<DenoisingSequence> {
    let py = runner.py();
    let buffers = DiffusionRunner::layout(runner, layout.bind(py))?;
    let inputs = py
        .get_type::<PyTuple>()
        .call1((inputs,))?
        .cast_into::<PyTuple>()?;
    let (samples, bank, slots, captures, pool_pages) = {
        let owner = runner.borrow();
        let samples = owner
            .samples
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("denoising requires the runner's sample pages"))?;
        (
            samples.clone_ref(py),
            owner.bank.clone_ref(py),
            owner.slots,
            DiffusionRunner::captures(runner.borrow()),
            owner
                .pool
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("denoising requires a sample pool"))?
                .borrow(py)
                .inner
                .num_pages() as i64,
        )
    };
    let page_count = buffers.borrow().pages;
    if slot == 0
        || slot > slots
        || inputs.is_empty()
        || pages.len() < page_count
        || pages
            .iter()
            .take(page_count)
            .any(|&page| page <= 0 || page >= pool_pages)
    {
        return Err(PyValueError::new_err(
            "denoising requires a valid slot, its layout's pages and solver steps",
        ));
    }
    for schedule in schedules.bind(py).call_method0("values")?.try_iter()? {
        if schedule?.getattr("num_steps")?.extract::<usize>()? != inputs.len() {
            return Err(PyValueError::new_err(
                "denoising inputs must cover every solver step",
            ));
        }
    }

    let mut latents = HashSet::new();
    let mut outputs = Vec::with_capacity(inputs.len());
    for input in inputs.iter() {
        let output = PyDict::new(py);
        for item in input
            .getattr("latents")?
            .call_method0("items")?
            .try_iter()?
        {
            let item = item?;
            let values = item
                .get_item(1)?
                .try_iter()?
                .map(|value| {
                    let tensor = value?.getattr("tensor")?;
                    if latents.insert(tensor.as_ptr() as usize) {
                        view_offset(&tensor, samples.bind(py))?;
                    }
                    Ok(tensor)
                })
                .collect::<PyResult<Vec<_>>>()?;
            output.set_item(item.get_item(0)?, PyTuple::new(py, values)?)?;
        }
        outputs.push(output.unbind());
    }

    let mut fields = HashMap::new();
    let mut state_spans = BTreeMap::new();
    if captures {
        for item in state.bind(py).call_method0("items")?.try_iter()? {
            let item = item?;
            let name: String = item.get_item(0)?.extract()?;
            let Some(bank) = bank.bind(py).get_item(&name)? else {
                continue;
            };
            let tensor = item.get_item(1)?;
            let row = bank.get_item(slot - 1)?;
            let start = view_offset(&tensor, &row)?;
            state_spans.insert(name.clone(), (start, tensor.getattr("shape")?.extract()?));
            fields.insert(tensor.as_ptr() as usize, name);
        }
    }

    let values = (&inputs, &schedules).into_pyobject(py)?;
    let shapes = crate::worker::cuda_graph::input_signature(values.as_any())?;
    let signature = (
        shapes,
        PyTuple::new(py, spans(py, &state_spans)?.items().iter())?,
    )
        .into_pyobject(py)?
        .into_any()
        .unbind();
    let mut temporal = Vec::with_capacity(inputs.len());
    for input in inputs.iter() {
        let tensors = GraphInputs::new(py, input.unbind())?.tensors(py)?;
        let sources = tensors.bind(py).iter().filter(|tensor| {
            let pointer = tensor.as_ptr() as usize;
            !fields.contains_key(&pointer) && !latents.contains(&pointer)
        });
        temporal.push(PyTuple::new(py, sources.collect::<Vec<_>>())?.unbind());
    }

    Ok(DenoisingSequence {
        layout,
        inputs: inputs.unbind(),
        schedules,
        state,
        slot,
        pages: pages.into_iter().take(page_count).collect(),
        signature,
        fields,
        spans: state_spans,
        temporal: PyTuple::new(py, temporal)?.unbind(),
        samples,
        outputs,
    })
}

fn spans<'py>(
    py: Python<'py>,
    spans: &BTreeMap<String, (isize, Vec<usize>)>,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    for (name, (offset, shape)) in spans {
        result.set_item(name, (offset, PyTuple::new(py, shape)?))?;
    }
    Ok(result)
}

/// Graph gathers address a contiguous span of its declared backing row.
/// Checking the span once prevents remapping a field into another slot.
fn view_offset(tensor: &Bound<'_, PyAny>, backing: &Bound<'_, PyAny>) -> PyResult<isize> {
    let start = tensor.call_method0("storage_offset")?.extract::<isize>()?
        - backing.call_method0("storage_offset")?.extract::<isize>()?;
    if start < 0
        || !tensor.getattr("dtype")?.eq(backing.getattr("dtype")?)?
        || !tensor.getattr("device")?.eq(backing.getattr("device")?)?
        || !tensor.call_method0("is_contiguous")?.is_truthy()?
        || !tensor
            .call_method0("untyped_storage")?
            .call_method0("data_ptr")?
            .eq(backing
                .call_method0("untyped_storage")?
                .call_method0("data_ptr")?)?
        || start + tensor.call_method0("numel")?.extract::<isize>()?
            > backing.call_method0("numel")?.extract::<isize>()?
    {
        return Err(PyValueError::new_err(
            "denoising input must be a contiguous view within its sample backing or named state slot",
        ));
    }
    Ok(start)
}
