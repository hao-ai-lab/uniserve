//! Bounded diagnostic capture driven by native batch execution.

use std::path::PathBuf;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

struct ProfileConfig {
    output_dir: PathBuf,
    prefix: String,
    activities: Vec<String>,
    start_step: usize,
    num_steps: usize,
    with_stack: bool,
    record_shapes: bool,
    cuda_profiler: bool,
}

/// Capture one execution window. Diagnostic failures never change batch results.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct WorkerProfiler {
    config: Option<ProfileConfig>,
    seen: usize,
    captured: usize,
    start: usize,
    active: bool,
    finished: bool,
    torch: Option<Py<PyAny>>,
}

#[pymethods]
impl WorkerProfiler {
    #[staticmethod]
    #[pyo3(signature = (env=None))]
    pub(super) fn from_env(env: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let value = |name: &str| -> PyResult<Option<String>> {
            match env {
                Some(env) => env.call_method1("get", (name,))?.extract(),
                None => Ok(std::env::var(name).ok()),
            }
        };
        let directory = value("UNISERVE_TORCH_PROFILER_DIR")?.filter(|value| !value.is_empty());
        let config = if let Some(directory) = directory {
            let raw = value("UNISERVE_PROFILE_ACTIVITIES")?.unwrap_or_else(|| "CPU,GPU".to_owned());
            let mut activities = Vec::new();
            for name in raw.replace(',', " ").split_whitespace() {
                let name = name.to_ascii_uppercase();
                if !matches!(name.as_str(), "CPU" | "GPU") {
                    return Err(PyValueError::new_err(format!(
                        "unknown profiler activity {name:?}; expected CPU or GPU"
                    )));
                }
                if !activities.contains(&name) {
                    activities.push(name);
                }
            }
            let steps = |name| -> PyResult<usize> {
                Ok(value(name)?
                    .and_then(|value| value.trim().parse::<i64>().ok())
                    .unwrap_or(1)
                    .max(1) as usize)
            };
            Some(ProfileConfig {
                output_dir: directory.into(),
                prefix: value("UNISERVE_PROFILE_PREFIX")?
                    .filter(|value| !value.is_empty())
                    .unwrap_or_else(|| "uniserve-worker".to_owned()),
                activities,
                start_step: steps("UNISERVE_PROFILE_START_STEP")?,
                num_steps: steps("UNISERVE_PROFILE_STEPS")?,
                with_stack: flag(value("UNISERVE_PROFILE_WITH_STACK")?.as_deref()),
                record_shapes: flag(value("UNISERVE_PROFILE_RECORD_SHAPES")?.as_deref()),
                cuda_profiler: flag(value("UNISERVE_CUDA_PROFILER")?.as_deref()),
            })
        } else {
            None
        };
        Ok(Self {
            config,
            seen: 0,
            captured: 0,
            start: 0,
            active: false,
            finished: false,
            torch: None,
        })
    }

    #[getter]
    fn enabled(&self) -> bool {
        self.config.is_some()
    }

    fn step(slf: Py<Self>, debug_name: String) -> ProfileStep {
        ProfileStep {
            owner: slf,
            name: debug_name,
            range: None,
        }
    }

    /// Export an open window during orderly shutdown, including partial windows.
    pub(super) fn close(&mut self, py: Python<'_>) {
        if self.active {
            self.stop(py);
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.torch)
    }
}

impl WorkerProfiler {
    pub(super) fn with_step<T>(
        owner: &Py<Self>,
        py: Python<'_>,
        name: String,
        execute: impl FnOnce() -> PyResult<T>,
    ) -> PyResult<T> {
        let mut scope = ProfileStep {
            owner: owner.clone_ref(py),
            name,
            range: None,
        };
        scope.__enter__(py);
        let result = execute();
        scope.finish(py);
        result
    }

    fn begin(&mut self, py: Python<'_>) {
        let Some(config) = &self.config else {
            return;
        };
        self.seen += 1;
        if self.active || self.finished || self.seen < config.start_step {
            return;
        }
        self.start = self.seen;
        let started = (|| -> PyResult<()> {
            py.detach(|| std::fs::create_dir_all(&config.output_dir))?;
            let backend = py.import("uniserve_worker.profiling")?;
            let profile = backend.call_method1(
                "_create_profiler",
                (&config.activities, config.with_stack, config.record_shapes),
            )?;
            if !profile.is_none() {
                self.torch = Some(profile.clone().unbind());
                profile.call_method0("start")?;
            }
            if config.cuda_profiler {
                backend.call_method1("_cuda_profiler", (true,))?;
            }
            Ok(())
        })();
        match started {
            Ok(()) => {
                self.active = true;
            }
            Err(error) => {
                // A CUDA-profiler failure after torch started still retires the
                // open torch session; later batches never retry this window.
                if let Some(profile) = self.torch.take() {
                    let _ = profile.call_method0(py, "stop");
                }
                self.finished = true;
                report(
                    py,
                    "failed to start UniServe worker profiler; disabling this capture",
                    error,
                );
            }
        }
    }

    fn end(&mut self, py: Python<'_>) {
        if self.active {
            self.captured += 1;
            if self
                .config
                .as_ref()
                .is_some_and(|config| self.captured >= config.num_steps)
            {
                self.stop(py);
            }
        }
    }

    fn stop(&mut self, py: Python<'_>) {
        self.active = false;
        self.finished = true;
        let profile = self.torch.take();
        let Some(config) = &self.config else {
            return;
        };
        let stopped = if config.cuda_profiler {
            py.import("uniserve_worker.profiling")
                .and_then(|backend| backend.call_method1("_cuda_profiler", (false,)))
                .map(drop)
        } else {
            Ok(())
        };
        if let Err(error) = stopped {
            report(py, "failed to stop CUDA profiler", error);
        }
        let Some(profile) = profile else {
            return;
        };
        let exported = (|| -> PyResult<()> {
            profile.call_method0(py, "stop")?;
            let stamp: String = py
                .import("time")?
                .call_method1("strftime", ("%Y%m%d-%H%M%S",))?
                .extract()?;
            let rank = std::env::var("RANK")
                .ok()
                .filter(|rank| !rank.is_empty())
                .or_else(|| std::env::var("LOCAL_RANK").ok());
            let rank = rank.map(|rank| format!("-rank{rank}")).unwrap_or_default();
            let end = self.start + self.captured.saturating_sub(1);
            let base = format!(
                "{}-pid{}{rank}-steps{}-{end}-{stamp}",
                config.prefix,
                std::process::id(),
                self.start
            );
            let trace = config.output_dir.join(format!("{base}.trace.json.gz"));
            profile.call_method1(
                py,
                "export_chrome_trace",
                (trace.to_string_lossy().as_ref(),),
            )?;
            let options = PyDict::new(py);
            options.set_item(
                "sort_by",
                if config.activities.iter().any(|name| name == "GPU") {
                    "self_device_time_total"
                } else {
                    "self_cpu_time_total"
                },
            )?;
            options.set_item("row_limit", 120)?;
            let summary: String = profile
                .bind(py)
                .call_method0("key_averages")?
                .call_method("table", (), Some(&options))?
                .extract()?;
            py.detach(|| {
                std::fs::write(
                    config.output_dir.join(format!("{base}.summary.txt")),
                    summary,
                )
            })?;
            Ok(())
        })();
        if let Err(error) = exported {
            report(py, "failed to stop/export UniServe worker profiler", error);
        }
    }
}

/// A Python context scope borrows its native owner; execution uses it directly.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
struct ProfileStep {
    owner: Py<WorkerProfiler>,
    name: String,
    range: Option<Py<PyAny>>,
}

#[pymethods]
impl ProfileStep {
    fn __enter__(&mut self, py: Python<'_>) {
        self.owner.borrow_mut(py).begin(py);
        let range = (|| -> PyResult<_> {
            let range = py
                .import("uniserve.profiling")?
                .call_method1("profile_range", (&self.name,))?;
            range.call_method0("__enter__")?;
            Ok(range.unbind())
        })();
        match range {
            Ok(range) => self.range = Some(range),
            Err(error) => report(py, "failed to enter worker profiler range", error),
        }
    }

    fn __exit__(
        &mut self,
        py: Python<'_>,
        _kind: &Bound<'_, PyAny>,
        _error: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) {
        self.finish(py);
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.owner)?;
        visit.call(&self.range)
    }
}

impl ProfileStep {
    fn finish(&mut self, py: Python<'_>) {
        if let Some(range) = self.range.take()
            && let Err(error) =
                range.call_method1(py, "__exit__", (py.None(), py.None(), py.None()))
        {
            report(py, "failed to leave worker profiler range", error);
        }
        self.owner.borrow_mut(py).end(py);
    }
}

#[pyfunction]
pub(super) fn timing_events_enabled() -> bool {
    std::env::var("UNISERVE_TORCH_PROFILER_DIR").is_ok_and(|value| !value.is_empty())
        || ["UNISERVE_NVTX", "UNISERVE_CUDA_PROFILER"]
            .iter()
            .any(|name| flag(std::env::var(name).ok().as_deref()))
}

fn flag(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn report(py: Python<'_>, message: &str, error: PyErr) {
    let _ = py
        .import("logging")
        .and_then(|logging| logging.call_method1("getLogger", ("uniserve_worker.profiling",)))
        .and_then(|logger| logger.call_method1("error", ("%s: %s", message, error.to_string())));
}
