//! Canonical token-sampling parameters shared with the engine and simulator.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyMapping, PyTuple};
use uniserve_core::SamplingParams as NativeParams;

use crate::worker::error::invalid;

/// Immutable FP32 sampling controls. Numerical consumers use the same values
/// that admission, IPC and the simulator use; validation belongs to the core.
#[pyclass(frozen, eq, module = "uniserve_worker._uniserve_ipc")]
#[derive(PartialEq)]
pub(crate) struct SamplingParams {
    pub(crate) inner: NativeParams,
}

#[pymethods]
impl SamplingParams {
    #[new]
    #[pyo3(signature = (**fields))]
    fn new(fields: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut inner = NativeParams::default();
        apply_fields(&mut inner, fields)?;
        inner
            .validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn from_mapping(value: &Bound<'_, PyMapping>) -> PyResult<Self> {
        let fields = PyDict::new(value.py());
        fields.update(value)?;
        Self::new(Some(&fields)).map_err(|error| invalid(value.py(), error.to_string()))
    }

    fn to_mapping<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pythonize::pythonize(py, &self.inner).map_err(Into::into)
    }

    #[pyo3(signature = (**fields))]
    fn replace(&self, fields: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut inner = self.inner.clone();
        apply_fields(&mut inner, fields)?;
        inner
            .validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    fn device_greedy(&self) -> bool {
        self.inner.device_greedy()
    }

    fn uses_penalties(&self) -> bool {
        self.inner.uses_penalties()
    }

    #[getter]
    fn temperature(&self) -> f32 {
        self.inner.temperature
    }

    #[getter]
    fn top_k(&self) -> u32 {
        self.inner.top_k
    }

    #[getter]
    fn top_p(&self) -> f32 {
        self.inner.top_p
    }

    #[getter]
    fn ignore_eos(&self) -> bool {
        self.inner.ignore_eos
    }

    #[getter]
    fn seed(&self) -> Option<u64> {
        self.inner.seed
    }

    #[getter]
    fn min_p(&self) -> f32 {
        self.inner.min_p
    }

    #[getter]
    fn repetition_penalty(&self) -> f32 {
        self.inner.repetition_penalty
    }

    #[getter]
    fn frequency_penalty(&self) -> f32 {
        self.inner.frequency_penalty
    }

    #[getter]
    fn presence_penalty(&self) -> f32 {
        self.inner.presence_penalty
    }

    #[getter]
    fn logit_bias<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.logit_bias)
    }

    #[getter]
    fn min_tokens(&self) -> usize {
        self.inner.min_tokens
    }

    #[getter]
    fn return_logprobs(&self) -> bool {
        self.inner.return_logprobs
    }

    #[getter]
    fn n_logprobs(&self) -> u32 {
        self.inner.n_logprobs
    }

    #[getter]
    fn return_prompt_logprobs(&self) -> bool {
        self.inner.return_prompt_logprobs
    }

    #[getter]
    fn n_prompt_logprobs(&self) -> u32 {
        self.inner.n_prompt_logprobs
    }

    #[getter]
    fn logprob_token_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.logprob_token_ids)
    }

    #[getter]
    fn bad_words_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            self.inner
                .bad_words_ids
                .iter()
                .map(|ids| PyTuple::new(py, ids))
                .collect::<PyResult<Vec<_>>>()?,
        )
    }

    #[getter]
    fn allowed_token_ids<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        self.inner
            .allowed_token_ids
            .as_ref()
            .map(|ids| PyTuple::new(py, ids))
            .transpose()
    }

    #[getter]
    fn typical_p(&self) -> f32 {
        self.inner.typical_p
    }

    #[getter]
    fn forced_token_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.forced_token_ids)
    }

    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyAny>,))> {
        Ok((
            py.get_type::<Self>().getattr("from_mapping")?,
            (self.to_mapping(py)?,),
        ))
    }
}

fn apply_fields(parameters: &mut NativeParams, fields: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
    if let Some(fields) = fields {
        for (name, value) in fields {
            match name.extract::<&str>()? {
                "temperature" => parameters.temperature = crate::convert::mapping_from_py(&value)?,
                "top_k" => parameters.top_k = crate::convert::mapping_from_py(&value)?,
                "top_p" => parameters.top_p = crate::convert::mapping_from_py(&value)?,
                "ignore_eos" => parameters.ignore_eos = crate::convert::mapping_from_py(&value)?,
                "seed" => parameters.seed = crate::convert::mapping_from_py(&value)?,
                "min_p" => parameters.min_p = crate::convert::mapping_from_py(&value)?,
                "repetition_penalty" => {
                    parameters.repetition_penalty = crate::convert::mapping_from_py(&value)?
                }
                "frequency_penalty" => {
                    parameters.frequency_penalty = crate::convert::mapping_from_py(&value)?
                }
                "presence_penalty" => {
                    parameters.presence_penalty = crate::convert::mapping_from_py(&value)?
                }
                "logit_bias" => parameters.logit_bias = crate::convert::mapping_from_py(&value)?,
                "min_tokens" => parameters.min_tokens = crate::convert::mapping_from_py(&value)?,
                "return_logprobs" => {
                    parameters.return_logprobs = crate::convert::mapping_from_py(&value)?
                }
                "n_logprobs" => parameters.n_logprobs = crate::convert::mapping_from_py(&value)?,
                "return_prompt_logprobs" => {
                    parameters.return_prompt_logprobs = crate::convert::mapping_from_py(&value)?
                }
                "n_prompt_logprobs" => {
                    parameters.n_prompt_logprobs = crate::convert::mapping_from_py(&value)?
                }
                "logprob_token_ids" => {
                    parameters.logprob_token_ids = crate::convert::mapping_from_py(&value)?
                }
                "bad_words_ids" => {
                    parameters.bad_words_ids = crate::convert::mapping_from_py(&value)?
                }
                "allowed_token_ids" => {
                    parameters.allowed_token_ids = crate::convert::mapping_from_py(&value)?
                }
                "typical_p" => parameters.typical_p = crate::convert::mapping_from_py(&value)?,
                "forced_token_ids" => {
                    parameters.forced_token_ids = crate::convert::mapping_from_py(&value)?
                }
                name => {
                    return Err(PyTypeError::new_err(format!(
                        "unknown SamplingParams field {name:?}"
                    )));
                }
            }
        }
    }
    Ok(())
}
