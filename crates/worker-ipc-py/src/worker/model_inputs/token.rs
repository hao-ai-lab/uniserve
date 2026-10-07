//! Token views and resident decode controls.

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker::TokenSelection;

use super::{AttentionRow, InputRow, tensor_len};

/// Indexed decode borrows tokens and positions from resident request state.
/// Ordinary rows retain tensor views; replacing a row never copies their data.
#[pyclass(extends = AttentionRow, module = "uniserve_worker._uniserve_ipc")]
#[derive(Default)]
pub(crate) struct TokenRow {
    #[pyo3(get)]
    pub(in crate::worker) token_ids: Option<Py<PyAny>>,
    #[pyo3(get)]
    pub(in crate::worker) token_embeddings: Option<Py<PyAny>>,
    #[pyo3(get)]
    pub(in crate::worker) token_embedding_mask: Option<Py<PyAny>>,
    pub(in crate::worker) selection: Option<TokenSelection>,
    #[pyo3(get)]
    pub(in crate::worker) decode_predicate: Option<Py<PyAny>>,
    #[pyo3(get)]
    pub(in crate::worker) decode_predicate_tagged: bool,
    #[pyo3(get)]
    pub(in crate::worker) decode_force_finish: bool,
    #[pyo3(get)]
    pub(in crate::worker) request_indexed_decode: bool,
}

#[pymethods]
impl TokenRow {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (forward_mode, request_pool_idx=0, positions=None, seq_len=0, write_kv=false, causal=Some(true), token_ids=None, token_embeddings=None, token_embedding_mask=None, selection=None, decode_predicate=None, decode_predicate_tagged=false, decode_force_finish=false, request_indexed_decode=false))]
    fn new(
        forward_mode: &Bound<'_, PyAny>,
        request_pool_idx: u32,
        positions: Option<Py<PyAny>>,
        seq_len: i64,
        write_kv: bool,
        causal: Option<bool>,
        token_ids: Option<Py<PyAny>>,
        token_embeddings: Option<Py<PyAny>>,
        token_embedding_mask: Option<Py<PyAny>>,
        selection: Option<&Bound<'_, PyAny>>,
        decode_predicate: Option<Py<PyAny>>,
        decode_predicate_tagged: bool,
        decode_force_finish: bool,
        request_indexed_decode: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        let row = InputRow {
            kind: pythonize::depythonize(forward_mode)?,
            request_pool_idx,
        };
        let attention = AttentionRow {
            positions,
            seq_len,
            write_kv,
            causal,
        };
        Ok(attention.initializer(row).add_subclass(Self {
            token_ids,
            token_embeddings,
            token_embedding_mask,
            selection: selection.map(parse_selection).transpose()?,
            decode_predicate,
            decode_predicate_tagged,
            decode_force_finish,
            request_indexed_decode,
        }))
    }

    #[getter]
    fn selection(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.selection
            .map(|selection| selection_to_py(py, selection))
            .transpose()
    }

    #[getter]
    pub(in crate::worker) fn query_tokens(&self, py: Python<'_>) -> PyResult<usize> {
        if self.request_indexed_decode {
            Ok(1)
        } else {
            tensor_len(py, &self.token_ids)
        }
    }

    /// Derive input controls while continuing to borrow the numerical tensors.
    #[pyo3(signature = (**fields))]
    fn replace(slf: PyRef<'_, Self>, fields: Option<&Bound<'_, PyDict>>) -> PyResult<Py<Self>> {
        let py = slf.py();
        let mut input = InputRow::clone(slf.as_super().as_super());
        let mut attention = slf.as_super().clone_ref(py);
        let mut row = slf.clone_ref(py);
        if let Some(fields) = fields {
            for (name, value) in fields {
                match name.extract::<&str>()? {
                    "token_ids" => row.token_ids = value.extract()?,
                    "token_embeddings" => row.token_embeddings = value.extract()?,
                    "token_embedding_mask" => row.token_embedding_mask = value.extract()?,
                    "selection" => {
                        row.selection = if value.is_none() {
                            None
                        } else {
                            Some(parse_selection(&value)?)
                        }
                    }
                    "decode_predicate" => row.decode_predicate = value.extract()?,
                    "decode_predicate_tagged" => row.decode_predicate_tagged = value.extract()?,
                    "decode_force_finish" => row.decode_force_finish = value.extract()?,
                    "request_indexed_decode" => row.request_indexed_decode = value.extract()?,
                    name => attention.field(&mut input, name, &value)?,
                }
            }
        }
        Py::new(py, attention.initializer(input).add_subclass(row))
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.token_ids)?;
        visit.call(&self.token_embeddings)?;
        visit.call(&self.token_embedding_mask)?;
        visit.call(&self.decode_predicate)
    }
}

impl TokenRow {
    /// Materialize ordinary token views; indexed decode needs no per-row
    /// tensors and goes directly to the resident gather during preparation.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::worker) fn with_tokens(
        mut self,
        py: Python<'_>,
        input: InputRow,
        mut attention: AttentionRow,
        tokens: &[u32],
        position: u64,
        current: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<Self>> {
        if !self.request_indexed_decode {
            let options = PyDict::new(py);
            options.set_item("current", current)?;
            let values = py.import("uniserve_worker.execution.token")?.call_method(
                "token_values",
                (PyTuple::new(py, tokens)?, position),
                Some(&options),
            )?;
            self.token_ids = Some(values.get_item(0)?.unbind());
            attention.positions = Some(values.get_item(1)?.unbind());
        }
        Py::new(py, attention.initializer(input).add_subclass(self))
    }

    fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            token_ids: self.token_ids.as_ref().map(|value| value.clone_ref(py)),
            token_embeddings: self
                .token_embeddings
                .as_ref()
                .map(|value| value.clone_ref(py)),
            token_embedding_mask: self
                .token_embedding_mask
                .as_ref()
                .map(|value| value.clone_ref(py)),
            selection: self.selection,
            decode_predicate: self
                .decode_predicate
                .as_ref()
                .map(|value| value.clone_ref(py)),
            decode_predicate_tagged: self.decode_predicate_tagged,
            decode_force_finish: self.decode_force_finish,
            request_indexed_decode: self.request_indexed_decode,
        }
    }
}

pub(in crate::worker) fn parse_selection(value: &Bound<'_, PyAny>) -> PyResult<TokenSelection> {
    match value.extract::<&str>()? {
        "last_logits" => Ok(TokenSelection::LastLogits),
        "all_logits" => Ok(TokenSelection::AllLogits),
        "hidden" => Ok(TokenSelection::Hidden),
        "cache" => Ok(TokenSelection::Cache),
        _ => Err(PyValueError::new_err("unknown text output selection")),
    }
}

pub(in crate::worker) fn parse_selections(
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<TokenSelection>> {
    values
        .try_iter()?
        .map(|value| parse_selection(&value?))
        .collect()
}

pub(in crate::worker) fn selection_to_py(
    py: Python<'_>,
    selection: TokenSelection,
) -> PyResult<Py<PyAny>> {
    Ok(py
        .import("uniserve_worker.sampling.metadata")?
        .getattr("TokenSelection")?
        .call1((selection.as_str(),))?
        .unbind())
}

pub(in crate::worker) fn selections_to_py(
    py: Python<'_>,
    selections: &[TokenSelection],
) -> PyResult<Py<PyTuple>> {
    let kind = py
        .import("uniserve_worker.sampling.metadata")?
        .getattr("TokenSelection")?;
    let values = selections
        .iter()
        .map(|selection| kind.call1((selection.as_str(),)))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, values)?.unbind())
}
