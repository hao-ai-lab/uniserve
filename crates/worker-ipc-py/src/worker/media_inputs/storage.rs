//! Capacity descriptions and aligned sample views for video requests.

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};

use super::MediaBuilder;
use crate::worker::tensor_buffers::buffer_envelope;

// A small page table bounds per-step gather overhead. Modalities start on a
// 256-element boundary; only the last page can contain unused sample storage.
const REQUEST_PAGES: usize = 8;
const SAMPLE_ALIGNMENT: usize = 256;

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct SamplePages {
    #[pyo3(get)]
    pub(in crate::worker) page_units: usize,
    #[pyo3(get)]
    pub(in crate::worker) pages: usize,
    #[pyo3(get)]
    pub(in crate::worker) dtype: Py<PyAny>,
}

#[pymethods]
impl SamplePages {
    #[getter]
    pub(in crate::worker) fn units(&self) -> usize {
        self.page_units * self.pages
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.dtype)
    }
}

impl SamplePages {
    pub(in crate::worker) fn for_slot(&self, slot: usize) -> PyResult<std::ops::Range<usize>> {
        if slot == 0 {
            return Err(PyValueError::new_err(
                "sample pages require a live request slot",
            ));
        }
        Ok(1 + (slot - 1) * self.pages..1 + slot * self.pages)
    }
}

pub(super) fn sample_pages(owner: &MediaBuilder, py: Python<'_>) -> PyResult<SamplePages> {
    let layout = owner.layout(owner.maximum.bind(py))?;
    let state = owner.state(layout.bind(py))?;
    let model = owner.denoiser.bind(py);
    let names: Vec<String> = model.getattr("modalities")?.extract()?;
    let dtype = state.as_any().get_item(&names[0])?.getattr("dtype")?;
    for name in names.iter().skip(1) {
        if !state
            .as_any()
            .get_item(name)?
            .getattr("dtype")?
            .eq(&dtype)?
        {
            return Err(PyValueError::new_err(
                "pooled sample modalities must share one dtype",
            ));
        }
    }
    let mut capacity = 0;
    for layout in owner.largest.bind(py).iter() {
        let mut count = 0;
        for name in &names {
            count += aligned(elements(
                &model.call_method1("latent_shape", (name, &layout))?,
            )?);
        }
        capacity = capacity.max(count);
    }
    let page_units = aligned(capacity.div_ceil(REQUEST_PAGES));
    Ok(SamplePages {
        page_units,
        pages: capacity.div_ceil(page_units),
        dtype: dtype.unbind(),
    })
}

#[allow(clippy::type_complexity)]
pub(super) fn offsets<'py>(
    owner: &MediaBuilder,
    layout: &Bound<'py, PyAny>,
) -> PyResult<(Vec<(String, Vec<usize>, usize)>, usize)> {
    let state = owner.state(layout)?;
    let names: Vec<String> = owner
        .denoiser
        .bind(layout.py())
        .getattr("modalities")?
        .extract()?;
    let mut result = Vec::with_capacity(names.len());
    let mut cursor = 0;
    for name in names {
        let shape: Vec<usize> = state
            .as_any()
            .get_item(&name)?
            .getattr("shape")?
            .extract()?;
        let count = shape.iter().product();
        result.push((name, shape, cursor));
        cursor += aligned(count);
    }
    Ok((result, cursor))
}

pub(super) fn sample_views(
    owner: &MediaBuilder,
    layout: &Bound<'_, PyAny>,
    flat: &Bound<'_, PyAny>,
) -> PyResult<Py<PyDict>> {
    let py = layout.py();
    let flat = flat.call_method1("view", (-1,))?;
    let (offsets, _) = offsets(owner, layout)?;
    let views = PyDict::new(py);
    for (name, shape, start) in offsets {
        let stop = start + shape.iter().product::<usize>();
        let view = flat
            .get_item(PySlice::new(py, start as isize, stop as isize, 1))?
            .call_method1("view", (PyTuple::new(py, shape)?,))?;
        views.set_item(name, view)?;
    }
    Ok(views.unbind())
}

pub(super) fn tables(owner: &MediaBuilder, layout: &Bound<'_, PyAny>) -> PyResult<Py<PyTuple>> {
    let names: Vec<String> = owner
        .denoiser
        .bind(layout.py())
        .getattr("modalities")?
        .extract()?;
    let state = owner.state(layout)?;
    let keys = state
        .keys()
        .iter()
        .map(|key| key.extract::<String>())
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(
        layout.py(),
        keys.into_iter()
            .filter(|key| !names.contains(key))
            .collect::<Vec<_>>(),
    )?
    .unbind())
}

pub(super) fn layout_buffers(
    owner: &MediaBuilder,
    layout: &Bound<'_, PyAny>,
) -> PyResult<Py<PyDict>> {
    let py = layout.py();
    let model = owner.denoiser.bind(py);
    let names: Vec<String> = model.getattr("modalities")?.extract()?;
    let state = owner.state(layout)?;
    let result = PyDict::new(py);
    let replace = py.import("dataclasses")?.getattr("replace")?;
    let config = py.import("uniserve.tensors")?.getattr("BufferConfig")?;
    let torch = py.import("torch")?;
    let host = PyDict::new(py);
    host.set_item("host", true)?;

    // Persistent tables occupy the slot; mutable samples occupy latent pages.
    // Each field has a host source so initialization can use one copy stream.
    for (name, value) in state.iter() {
        let name: String = name.extract()?;
        if !names.contains(&name) {
            result.set_item(&name, &value)?;
        }
        result.set_item(
            format!("{name}_source"),
            replace.call((value,), Some(&host))?,
        )?;
    }
    for name in names {
        result.set_item(
            format!("{name}_noise"),
            config.call(
                (
                    model.call_method1("noise_shape", (&name, layout))?,
                    torch.getattr("float32")?,
                ),
                Some(&host),
            )?,
        )?;
    }
    result.set_item(
        "condition_noise",
        config.call(
            (
                (model.call_method1("condition_noise_capacity", (layout,))?,),
                torch.getattr("float32")?,
            ),
            Some(&host),
        )?,
    )?;
    result.set_item(
        "text_condition",
        config.call1((
            (
                model.call_method1("text_condition_rows", (layout,))?,
                model.getattr("text_condition_width")?,
            ),
            torch.getattr("bfloat16")?,
        ))?,
    )?;
    Ok(result.unbind())
}

pub(super) fn capacity_buffers(owner: &MediaBuilder, py: Python<'_>) -> PyResult<Py<PyDict>> {
    let descriptions = owner
        .largest
        .bind(py)
        .iter()
        .map(|layout| layout_buffers(owner, &layout))
        .collect::<PyResult<Vec<_>>>()?;
    let result = buffer_envelope(py, &PyTuple::new(py, descriptions)?)?;
    let result = result.bind(py);
    let model = owner.denoiser.bind(py);
    let names: Vec<String> = model.getattr("modalities")?.extract()?;
    let replace = py.import("dataclasses")?.getattr("replace")?;
    for name in names {
        let mut capacity = Vec::<usize>::new();
        for layout in owner.largest.bind(py).iter() {
            let shape: Vec<usize> = model
                .call_method1("latent_shape", (&name, layout))?
                .extract()?;
            if capacity.is_empty() {
                capacity = shape;
            } else {
                for (bound, extent) in capacity.iter_mut().zip(shape) {
                    *bound = (*bound).max(extent);
                }
            }
        }
        let key = format!("{name}_source");
        let options = PyDict::new(py);
        options.set_item("capacity_shape", PyTuple::new(py, capacity)?)?;
        result.set_item(
            &key,
            replace.call((result.as_any().get_item(&key)?,), Some(&options))?,
        )?;
    }
    Ok(result.clone().unbind())
}

pub(super) fn elements(shape: &Bound<'_, PyAny>) -> PyResult<usize> {
    Ok(shape.extract::<Vec<usize>>()?.iter().product())
}

fn aligned(elements: usize) -> usize {
    elements.div_ceil(SAMPLE_ALIGNMENT) * SAMPLE_ALIGNMENT
}
