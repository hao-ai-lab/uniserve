//! Select active calls and launch the batch's numerical operations.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use uniserve_worker_ipc::CallStatus;

use super::{BatchState, PythonBackend};
use crate::worker::host::with_context;

impl PythonBackend {
    pub(super) fn run_batch<'py>(
        &self,
        py: Python<'py>,
        batch: &mut BatchState,
    ) -> PyResult<Bound<'py, PyAny>> {
        if batch.plan.calls.is_empty() {
            return Ok(py.None().into_bound(py));
        }

        let numerical = batch.numerical.clone_ref(py);
        let runner = self.runner.bind(py);
        let clock = py.import("time")?.getattr("perf_counter_ns")?;
        let started: u64 = clock.call0()?.extract()?;
        let scope = runner.call_method1("profile_step", (&numerical,))?;
        with_context(&scope, || {
            let mut phase = "batch registration";
            let executed = (|| {
                runner.call_method1("reserve", (&numerical,))?;
                phase = "batch execution";
                self.execute_calls(py, batch)?;
                phase = "batch commit";
                self.commit_batch(py, batch)?;
                Ok(())
            })();

            let Err(error): PyResult<()> = executed else {
                return Ok(py.None().into_bound(py));
            };
            let kwargs = PyDict::new(py);
            kwargs.set_item("phase", phase)?;
            kwargs.set_item("state", &numerical)?;
            kwargs.set_item("committed", batch.committed)?;
            let classified = py.import("uniserve_worker.errors")?.call_method(
                "classify_batch_failure",
                (error.value(py),),
                Some(&kwargs),
            )?;

            // Once stores become visible, rollback could invalidate a reader.
            // Earlier failures retire all reservations on the consuming stream.
            if !batch.committed {
                self.discard_batch(py, batch)?;
            }
            if batch.propagate_errors || classified.getattr("fatal")?.extract::<bool>()? {
                return Err(PyErr::from_value(classified));
            }

            let bound_at = numerical.borrow(py).started_ns;
            let stats = runner.call_method1(
                "execution_stats",
                (
                    &numerical,
                    if bound_at == 0 { started } else { bound_at },
                    py.None(),
                ),
            )?;
            batch.record_execution(py, &stats)?;
            Ok(classified)
        })
    }

    fn execute_calls(&self, py: Python<'_>, batch: &BatchState) -> PyResult<()> {
        let mut active = Vec::new();
        for (index, output) in batch.pending_outputs(py).iter().enumerate() {
            if output.borrow().lock(py)?.output.status != CallStatus::Predicated {
                active.push(index);
            }
        }

        if !active.is_empty() {
            self.runner
                .bind(py)
                .call_method1("execute", (&batch.numerical, active))?;
        }
        Ok(())
    }
}
