//! Immutable worker counters shared by numerical execution and IPC.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyMapping};
use uniserve_worker_ipc::ForwardStats as NativeStats;

use crate::convert::{cached, mapping_from_py};

macro_rules! counters {
    (maps: [$($map:ident),*], scalars: [$($scalar:ident),*]) => {
        /// Immutable counters for completed model calls, including graph use.
        #[pyclass(frozen, eq, module = "uniserve_worker._uniserve_ipc")]
        pub(crate) struct ForwardStats {
            pub(crate) inner: NativeStats,
            $($map: PyOnceLock<Py<PyAny>>,)*
        }

        impl From<NativeStats> for ForwardStats {
            fn from(inner: NativeStats) -> Self {
                Self { inner, $($map: PyOnceLock::new(),)* }
            }
        }

        impl PartialEq for ForwardStats {
            fn eq(&self, other: &Self) -> bool {
                self.inner == other.inner
            }
        }

        #[pymethods]
        impl ForwardStats {
            #[new]
            #[pyo3(signature = (**fields))]
            fn new(fields: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
                let mut inner = NativeStats::default();

                if let Some(fields) = fields {
                    for (key, value) in fields {
                        match key.extract::<&str>()? {
                            $(stringify!($map) => {
                                inner.$map = value.cast::<PyMapping>()?
                                    .items()?
                                    .iter()
                                    .map(|item| item.extract())
                                    .collect::<PyResult<_>>()?;
                            },)*
                            $(stringify!($scalar) => inner.$scalar = value.extract()?,)*
                            key => {
                                return Err(PyTypeError::new_err(format!(
                                    "unknown ForwardStats field {key:?}"
                                )));
                            }
                        }
                    }
                }

                Ok(Self::from(inner))
            }

            #[staticmethod]
            fn from_mapping(value: &Bound<'_, PyAny>) -> PyResult<Self> {
                mapping_from_py::<NativeStats>(value)
                    .map(Self::from)
                    .map_err(|error| crate::worker::error::invalid(value.py(), error.to_string()))
            }

            fn to_mapping<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                pythonize::pythonize(py, &self.inner).map_err(Into::into)
            }

            $(
                #[getter]
                fn $scalar(&self) -> u64 {
                    self.inner.$scalar
                }
            )*

            $(#[getter]
            fn $map<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                cached(py, &self.$map, || {
                    let values = pythonize::pythonize(py, &self.inner.$map)?;
                    py.import("types")?.getattr("MappingProxyType")?.call1((values,))
                })
            })*

            fn __reduce__<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyAny>,))> {
                Ok((py.get_type::<Self>().getattr("from_mapping")?, (self.to_mapping(py)?,)))
            }
        }
    };
}

counters! {
    maps: [
        mode_counts,
        mode_tokens,
        mode_us,
        component_us,
        attention_backend_counts,
        cuda_graph_runtime_mode_counts,
        spec_verify_path_counts
    ],
    scalars: [
        attention_launches,
        attention_us,
        cuda_graph_captures,
        cuda_graph_replays,
        cuda_graph_misses,
        cuda_graph_fallbacks,
        cuda_graph_unpadded_tokens,
        cuda_graph_padded_tokens,
        text_decode_token_relay_hits,
        text_decode_token_relay_misses,
        text_decode_position_relay_hits,
        text_decode_position_relay_misses,
        flashinfer_decode_plan_calls,
        flashinfer_decode_plan_reuses,
        flashinfer_decode_plan_rows,
        flashinfer_decode_plan_indices,
        flashinfer_decode_graph_plan_calls,
        flashinfer_decode_graph_plan_reuses,
        spec_verify_rows,
        spec_verify_draft_tokens,
        spec_verify_accepted_tokens,
        spec_verify_rejected_tokens,
        spec_verify_committed_tokens
    ]
}
