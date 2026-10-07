//! Borrowed image tensors with native sequence and output dimensions.

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{AttentionRow, InputRow};

#[pyclass(extends = AttentionRow, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct DiffusionRow {
    #[pyo3(get)]
    pub(in crate::worker) timestep: Py<PyAny>,
    #[pyo3(get)]
    pub(in crate::worker) latent: Py<PyAny>,
    #[pyo3(get)]
    pub(in crate::worker) image_tokens: usize,
    #[pyo3(get)]
    pub(in crate::worker) image_height: usize,
    #[pyo3(get)]
    pub(in crate::worker) image_width: usize,
}

#[pymethods]
impl DiffusionRow {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (forward_mode, request_pool_idx=0, positions=None, seq_len=0, write_kv=false, causal=Some(true), *, timestep, latent, image_tokens, image_height, image_width))]
    fn new(
        forward_mode: &Bound<'_, PyAny>,
        request_pool_idx: u32,
        positions: Option<Py<PyAny>>,
        seq_len: i64,
        write_kv: bool,
        causal: Option<bool>,
        timestep: Py<PyAny>,
        latent: Py<PyAny>,
        image_tokens: usize,
        image_height: usize,
        image_width: usize,
    ) -> PyResult<PyClassInitializer<Self>> {
        let input = InputRow {
            kind: pythonize::depythonize(forward_mode)?,
            request_pool_idx,
        };
        let attention = AttentionRow {
            positions,
            seq_len,
            write_kv,
            causal,
        };
        Ok(attention.initializer(input).add_subclass(Self {
            timestep,
            latent,
            image_tokens,
            image_height,
            image_width,
        }))
    }

    #[getter]
    fn query_tokens(&self) -> usize {
        self.image_tokens
    }

    #[pyo3(signature = (**fields))]
    fn replace(slf: PyRef<'_, Self>, fields: Option<&Bound<'_, PyDict>>) -> PyResult<Py<Self>> {
        let py = slf.py();
        let mut input = InputRow::clone(slf.as_super().as_super());
        let mut attention = slf.as_super().clone_ref(py);
        let mut row = Self {
            timestep: slf.timestep.clone_ref(py),
            latent: slf.latent.clone_ref(py),
            image_tokens: slf.image_tokens,
            image_height: slf.image_height,
            image_width: slf.image_width,
        };
        if let Some(fields) = fields {
            for (name, value) in fields {
                match name.extract::<&str>()? {
                    "timestep" => row.timestep = value.unbind(),
                    "latent" => row.latent = value.unbind(),
                    "image_tokens" => row.image_tokens = value.extract()?,
                    "image_height" => row.image_height = value.extract()?,
                    "image_width" => row.image_width = value.extract()?,
                    name => attention.field(&mut input, name, &value)?,
                }
            }
        }
        Py::new(py, attention.initializer(input).add_subclass(row))
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.timestep)?;
        visit.call(&self.latent)
    }
}

/// Prepared encoder pixels and optional host-known patch-grid dimensions.
#[pyclass(extends = InputRow, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct VisionRow {
    #[pyo3(get)]
    pub(in crate::worker) encode_pixels: Py<PyAny>,
    #[pyo3(get)]
    pub(in crate::worker) encode_grid: Option<Py<PyAny>>,
    #[pyo3(get)]
    pub(in crate::worker) encode_grid_shape: Option<(usize, usize)>,
}

#[pymethods]
impl VisionRow {
    #[new]
    #[pyo3(signature = (forward_mode, request_pool_idx=0, *, encode_pixels, encode_grid=None, encode_grid_shape=None))]
    fn new(
        forward_mode: &Bound<'_, PyAny>,
        request_pool_idx: u32,
        encode_pixels: Py<PyAny>,
        encode_grid: Option<Py<PyAny>>,
        encode_grid_shape: Option<(usize, usize)>,
    ) -> PyResult<PyClassInitializer<Self>> {
        Ok(PyClassInitializer::from(InputRow {
            kind: pythonize::depythonize(forward_mode)?,
            request_pool_idx,
        })
        .add_subclass(Self {
            encode_pixels,
            encode_grid,
            encode_grid_shape,
        }))
    }

    #[pyo3(signature = (**fields))]
    fn replace(slf: PyRef<'_, Self>, fields: Option<&Bound<'_, PyDict>>) -> PyResult<Py<Self>> {
        let py = slf.py();
        let mut input = InputRow::clone(slf.as_super());
        let mut row = Self {
            encode_pixels: slf.encode_pixels.clone_ref(py),
            encode_grid: slf.encode_grid.as_ref().map(|value| value.clone_ref(py)),
            encode_grid_shape: slf.encode_grid_shape,
        };
        if let Some(fields) = fields {
            for (name, value) in fields {
                match name.extract::<&str>()? {
                    "encode_pixels" => row.encode_pixels = value.unbind(),
                    "encode_grid" => row.encode_grid = value.extract()?,
                    "encode_grid_shape" => row.encode_grid_shape = value.extract()?,
                    name => input.field(name, &value)?,
                }
            }
        }
        Py::new(py, PyClassInitializer::from(input).add_subclass(row))
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.encode_pixels)?;
        visit.call(&self.encode_grid)
    }
}

/// One latent view and the decoder's requested image dimensions.
#[pyclass(extends = InputRow, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct DecodeRow {
    #[pyo3(get)]
    pub(in crate::worker) latent: Py<PyAny>,
    #[pyo3(get)]
    pub(in crate::worker) image_height: usize,
    #[pyo3(get)]
    pub(in crate::worker) image_width: usize,
}

#[pymethods]
impl DecodeRow {
    #[new]
    #[pyo3(signature = (forward_mode, request_pool_idx=0, *, latent, image_height, image_width))]
    fn new(
        forward_mode: &Bound<'_, PyAny>,
        request_pool_idx: u32,
        latent: Py<PyAny>,
        image_height: usize,
        image_width: usize,
    ) -> PyResult<PyClassInitializer<Self>> {
        Ok(PyClassInitializer::from(InputRow {
            kind: pythonize::depythonize(forward_mode)?,
            request_pool_idx,
        })
        .add_subclass(Self {
            latent,
            image_height,
            image_width,
        }))
    }

    #[pyo3(signature = (**fields))]
    fn replace(slf: PyRef<'_, Self>, fields: Option<&Bound<'_, PyDict>>) -> PyResult<Py<Self>> {
        let py = slf.py();
        let mut input = InputRow::clone(slf.as_super());
        let mut row = Self {
            latent: slf.latent.clone_ref(py),
            image_height: slf.image_height,
            image_width: slf.image_width,
        };
        if let Some(fields) = fields {
            for (name, value) in fields {
                match name.extract::<&str>()? {
                    "latent" => row.latent = value.unbind(),
                    "image_height" => row.image_height = value.extract()?,
                    "image_width" => row.image_width = value.extract()?,
                    name => input.field(name, &value)?,
                }
            }
        }
        Py::new(py, PyClassInitializer::from(input).add_subclass(row))
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.latent)
    }
}
