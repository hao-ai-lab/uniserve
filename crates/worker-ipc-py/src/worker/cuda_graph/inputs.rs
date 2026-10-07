//! Captured tensor references and the paths used to update them before replay.

use std::collections::{HashMap, HashSet};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyString, PyTuple};

/// Key tensor structure, layouts and aliases without reading device contents.
#[pyfunction]
pub(in crate::worker) fn input_signature<'py>(
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let py = value.py();
    let options = PyDict::new(py);
    options.set_item(
        "is_leaf",
        py.import("uniserve_worker.model_executor.cuda_graph")?
            .getattr("_register")?,
    )?;
    let flattened =
        py.import("torch.utils._pytree")?
            .call_method("tree_flatten", (value,), Some(&options))?;
    let tensor_type = py.import("torch")?.getattr("Tensor")?;
    let mut aliases = HashMap::new();
    let mut leaves = Vec::new();
    for item in flattened.get_item(0)?.try_iter()? {
        let item = item?;
        let leaf = if item.is_instance(&tensor_type)? {
            let next = aliases.len();
            let ordinal = *aliases.entry(item.as_ptr()).or_insert(next);
            (
                item.getattr("device")?,
                item.getattr("dtype")?,
                PyTuple::new(py, item.getattr("shape")?.extract::<Vec<usize>>()?)?,
                item.call_method0("stride")?,
                ordinal,
            )
                .into_pyobject(py)?
                .into_any()
        } else {
            item
        };
        leaves.push(leaf);
    }
    (flattened.get_item(1)?, PyTuple::new(py, leaves)?).into_pyobject(py)
}

/// A captured numerical PyTree. Tensor aliases are bound once; non-tensor
/// values stay fixed, while copy updates the live tensors on the current stream.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct GraphInputs {
    #[pyo3(get)]
    pub(in crate::worker) value: Py<PyAny>,
    tensors: Vec<InputTensor>,
}

struct InputTensor {
    path: Vec<Accessor>,
    tensor: Py<PyAny>,
    // A broadcast tensor owns only one element along every zero-stride axis.
    slices: Option<Py<PyTuple>>,
}

enum Accessor {
    Attribute(Py<PyString>),
    Item(Py<PyAny>),
    Custom(Py<PyAny>),
}

#[pymethods]
impl GraphInputs {
    #[new]
    pub(in crate::worker) fn new(py: Python<'_>, value: Py<PyAny>) -> PyResult<Self> {
        let pytree = py.import("torch.utils._pytree")?;
        let options = PyDict::new(py);
        options.set_item(
            "is_leaf",
            py.import("uniserve_worker.model_executor.cuda_graph")?
                .getattr("_register")?,
        )?;
        let leaves = pytree
            .call_method("tree_flatten_with_path", (&value,), Some(&options))?
            .get_item(0)?;
        let tensor_type = py.import("torch")?.getattr("Tensor")?;
        let attribute = pytree.getattr("GetAttrKey")?;
        let sequence = pytree.getattr("SequenceKey")?;
        let mapping = pytree.getattr("MappingKey")?;
        let mut seen = HashSet::new();
        let mut tensors = Vec::new();

        for leaf in leaves.try_iter()? {
            let leaf = leaf?;
            let tensor = leaf.get_item(1)?;
            if !tensor.is_instance(&tensor_type)? || !seen.insert(tensor.as_ptr()) {
                continue;
            }
            let path = leaf
                .get_item(0)?
                .try_iter()?
                .map(|key| {
                    let key = key?;
                    Ok(if key.is_instance(&attribute)? {
                        Accessor::Attribute(key.getattr("name")?.extract()?)
                    } else if key.is_instance(&sequence)? {
                        Accessor::Item(key.getattr("idx")?.unbind())
                    } else if key.is_instance(&mapping)? {
                        Accessor::Item(key.getattr("key")?.unbind())
                    } else {
                        // User-registered PyTree nodes may supply their own keys.
                        Accessor::Custom(key.getattr("get")?.unbind())
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            let strides: Vec<i64> = tensor.call_method0("stride")?.extract()?;
            let slices = if strides.contains(&0) {
                Some(
                    PyTuple::new(
                        py,
                        strides.iter().map(|&stride| {
                            PySlice::new(py, 0, if stride == 0 { 1 } else { isize::MAX }, 1)
                        }),
                    )?
                    .unbind(),
                )
            } else {
                None
            };
            tensors.push(InputTensor {
                path,
                tensor: tensor.unbind(),
                slices,
            });
        }

        Ok(Self { value, tensors })
    }

    /// Distinct tensor leaves in traversal order, including borrowed storage.
    #[getter]
    pub(in crate::worker) fn tensors(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.tensors.iter().map(|input| &input.tensor))?.unbind())
    }

    /// Copy live tensors into fixed backing. A mismatched tensor raises before
    /// its copy; preceding copies remain queued. No device values are read.
    pub(in crate::worker) fn copy(&self, live: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = live.py();
        for input in &self.tensors {
            let mut value = live.clone();
            for key in &input.path {
                value = match key {
                    Accessor::Attribute(name) => value.getattr(name.bind(py))?,
                    Accessor::Item(index) => value.get_item(index)?,
                    Accessor::Custom(get) => get.call1(py, (&value,))?.into_bound(py),
                };
            }
            let destination = input.tensor.bind(py);
            if !destination.getattr("shape")?.eq(value.getattr("shape")?)?
                || !destination.getattr("dtype")?.eq(value.getattr("dtype")?)?
                || !destination
                    .getattr("device")?
                    .eq(value.getattr("device")?)?
            {
                return Err(PyValueError::new_err(
                    "graph tensor shape or representation changed",
                ));
            }
            if destination
                .call_method0("data_ptr")?
                .eq(value.call_method0("data_ptr")?)?
            {
                continue;
            }

            if let Some(slices) = &input.slices {
                destination
                    .get_item(slices)?
                    .call_method1("copy_", (value.get_item(slices)?,))?;
            } else {
                destination.call_method1("copy_", (value,))?;
            }
        }
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.value)?;
        for input in &self.tensors {
            visit.call(&input.tensor)?;
            visit.call(&input.slices)?;
            for key in &input.path {
                match key {
                    Accessor::Attribute(name) => visit.call(name)?,
                    Accessor::Item(index) | Accessor::Custom(index) => visit.call(index)?,
                }
            }
        }
        Ok(())
    }
}
