//! Native text capture plans; tensors and graph objects stay with the runner.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use uniserve_worker::{PrefillShape as NativePrefillShape, TextShapes as NativeTextShapes};

#[pyclass(
    frozen,
    eq,
    hash,
    skip_from_py_object,
    module = "uniserve_worker._uniserve_ipc"
)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct PrefillShape {
    pub(super) inner: NativePrefillShape,
}

#[pymethods]
impl PrefillShape {
    #[new]
    #[pyo3(signature = (token_bucket, row_bucket, live_rows, causal=Some(true), embeddings=false, outputs=true))]
    fn new(
        token_bucket: usize,
        row_bucket: usize,
        live_rows: usize,
        causal: Option<bool>,
        embeddings: bool,
        outputs: bool,
    ) -> Self {
        Self {
            inner: NativePrefillShape {
                token_bucket,
                row_bucket,
                live_rows,
                causal,
                embeddings,
                outputs,
            },
        }
    }

    #[getter]
    fn token_bucket(&self) -> usize {
        self.inner.token_bucket
    }

    #[getter]
    fn row_bucket(&self) -> usize {
        self.inner.row_bucket
    }

    #[getter]
    fn live_rows(&self) -> usize {
        self.inner.live_rows
    }

    #[getter]
    fn causal(&self) -> Option<bool> {
        self.inner.causal
    }

    #[getter]
    fn embeddings(&self) -> bool {
        self.inner.embeddings
    }

    #[getter]
    fn outputs(&self) -> bool {
        self.inner.outputs
    }

    // Expert peers exchange these immutable values during capture planning.
    fn __getnewargs__(&self) -> (usize, usize, usize, Option<bool>, bool, bool) {
        let shape = self.inner;
        (
            shape.token_bucket,
            shape.row_bucket,
            shape.live_rows,
            shape.causal,
            shape.embeddings,
            shape.outputs,
        )
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct TextShapes {
    inner: NativeTextShapes,
}

#[pymethods]
impl TextShapes {
    #[new]
    fn new(decode: Vec<usize>, prefill: Vec<PyRef<'_, PrefillShape>>) -> Self {
        Self {
            inner: NativeTextShapes::new(decode, prefill.iter().map(|shape| shape.inner).collect()),
        }
    }

    #[getter]
    fn decode<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.decode().iter().copied())
    }

    #[getter]
    fn prefill<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        shapes_to_py(py, self.inner.prefill().iter().copied())
    }

    #[pyo3(signature = (queries, tokens, *, decode, causal, embeddings, last_logits, cache_only))]
    #[allow(clippy::too_many_arguments)]
    fn select(
        &self,
        py: Python<'_>,
        queries: Vec<usize>,
        tokens: usize,
        decode: bool,
        causal: Option<bool>,
        embeddings: bool,
        last_logits: bool,
        cache_only: bool,
    ) -> PyResult<Option<(usize, usize, bool, bool)>> {
        self.inner
            .select(
                &queries,
                tokens,
                decode,
                causal,
                embeddings,
                last_logits,
                cache_only,
            )
            .map_err(|error| {
                match py.import("uniserve.runtime.cuda_graph").and_then(|module| {
                    module
                        .getattr("CUDAGraphError")?
                        .call1((error.to_string(),))
                }) {
                    Ok(error) => PyErr::from_value(error),
                    Err(error) => error,
                }
            })
    }
}

#[pyfunction]
pub(crate) fn prefill_units(
    pages: Vec<(usize, usize)>,
    rows: usize,
    tokens: usize,
) -> PyResult<usize> {
    validate_pages(&pages)?;
    Ok(uniserve_worker::prefill_units(&pages, rows, tokens))
}

#[pyfunction]
#[pyo3(signature = (token_sizes, row_sizes, *, max_rows, max_tokens, variants, outputs=true, pool=None))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn select_prefill_captures<'py>(
    py: Python<'py>,
    token_sizes: Vec<usize>,
    row_sizes: Vec<usize>,
    max_rows: usize,
    max_tokens: usize,
    variants: Vec<(Option<bool>, bool)>,
    outputs: bool,
    pool: Option<(Vec<(usize, usize)>, usize)>,
) -> PyResult<Bound<'py, PyTuple>> {
    if let Some((pages, _)) = &pool {
        validate_pages(pages)?;
    }
    let shapes = uniserve_worker::prefill_shapes(
        &token_sizes,
        &row_sizes,
        max_rows,
        max_tokens,
        &variants,
        outputs,
        pool.as_ref()
            .map(|(pages, units)| (pages.as_slice(), *units)),
    );
    shapes_to_py(py, shapes)
}

fn validate_pages(pages: &[(usize, usize)]) -> PyResult<()> {
    if pages.iter().any(|&(tokens, _)| tokens == 0) {
        return Err(PyValueError::new_err(
            "KV page token count must be positive",
        ));
    }
    Ok(())
}

fn shapes_to_py<'py>(
    py: Python<'py>,
    shapes: impl IntoIterator<Item = NativePrefillShape>,
) -> PyResult<Bound<'py, PyTuple>> {
    let shapes = shapes
        .into_iter()
        .map(|inner| Py::new(py, PrefillShape { inner }))
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, shapes)
}
