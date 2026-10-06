//! Commit complete prompt contexts and retain logits between prompt runs.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use super::token::{next_position, scores_prompt};
use super::{BatchState, ForwardRow, ForwardValue, PythonBackend, SampleCandidate};
use crate::sampling::SamplingParams;
use crate::worker::error::{invalid, unsupported};
use crate::worker::pending::PendingOutput;
use crate::worker::sampling::SamplingMetadata;

impl PythonBackend {
    pub(super) fn score_prompt(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        output: &mut PendingOutput,
        start: u64,
        task: &Bound<'_, PyAny>,
        logits: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let state = self.decode_state.as_ref().ok_or_else(|| {
            unsupported(py, "prompt scoring has no request-indexed runtime state")
        })?;
        let previous = if start == 0 {
            None
        } else {
            if !output.lock(py)?.progress.prompt_logits_ready {
                return Err(invalid(
                    py,
                    "continued prompt scoring has no preceding logits",
                ));
            }
            let retained = output
                .token_update
                .prompt_logits
                .as_ref()
                .map(|value| value.clone_ref(py));
            Some(match retained {
                Some(value) => value,
                None => {
                    let slot = output.request.borrow(py).request.slot();
                    state
                        .bind(py)
                        .getattr("prompt_logits")?
                        .get_item(slot)?
                        .unbind()
                }
            })
        };
        let mut parameters = output
            .request
            .borrow(py)
            .request
            .admission()
            .ar
            .as_ref()
            .ok_or_else(|| {
                invalid(
                    py,
                    "sequence execution requires admitted sampling parameters",
                )
            })?
            .sampling
            .clone();
        parameters.return_logprobs = true;
        parameters.n_logprobs = parameters.n_prompt_logprobs;
        let parameters = Py::new(py, SamplingParams { inner: parameters })?;

        let (retained, details): (Py<PyAny>, Py<PyAny>) = py
            .import("uniserve_worker.execution.token")?
            .call_method1(
                "prompt_logprobs",
                (task.getattr("token_ids")?, logits, previous, parameters),
            )?
            .extract()?;
        output.token_update.prompt_logits = Some(retained);
        output.lock(py)?.progress.prompt_logits_ready = true;

        let buffer = batch.numerical.borrow(py).output_buffer(py)?;
        let spans = buffer.get().capture_logprobs(py, details.bind(py))?;
        // Captured spans retain numerical token order across prompt runs.
        output.lock(py)?.prompt_logprob_ranges.extend(spans);
        Ok(())
    }

    pub(super) fn finish_context(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
        rows: &[(ForwardRow, ForwardValue)],
    ) -> PyResult<Option<SampleCandidate>> {
        let Some(((last, value), preceding)) = rows.split_last() else {
            return Err(PyRuntimeError::new_err("context forward returned no rows"));
        };
        let call = &batch.plan.calls[index];
        let pending = batch.pending(py, index);
        let mut output = pending.borrow_mut(py);
        let task = last.task.bind(py);
        let query = task.getattr("query_tokens")?.extract()?;
        self.record_token_kv(py, &mut output, task, query, true)?;

        let samples = call.token_output.is_some();
        let causal: bool = task.getattr("causal")?.extract()?;
        if scores_prompt(py, &output)? {
            let scored = if samples && causal { preceding } else { rows };
            for (row, value) in scored {
                let task = row.task.bind(py);
                let logits = value.value.bind(py);
                if task.getattr("causal")?.extract::<bool>()? {
                    let start = task.getattr("positions")?.get_item(0)?.extract()?;
                    self.score_prompt(py, batch, &mut output, start, task, logits)?;
                } else {
                    // A vision block predicts the next prompt token, but its
                    // image placeholders are never emitted as prompt scores.
                    let retained = logits.get_item(-1)?.call_method0("detach")?.unbind();
                    output.token_update.prompt_logits = Some(retained);
                    output.lock(py)?.progress.prompt_logits_ready = true;
                }
            }
        }

        let position = next_position(py, task)?;
        if !samples {
            self.visible_length(py, &output)?;
            let fallback = task.getattr("seq_len")?.extract::<u64>()? + query;
            let length = self.token_length(py, &output, fallback)?;
            output
                .lock(py)?
                .advance_tokens(query, Some(length), Some(position), false);
            return Ok(None);
        }
        if causal {
            let start = task.getattr("positions")?.get_item(0)?.extract()?;
            output.lock(py)?.advance_tokens(0, None, Some(start), false);
        }
        let state = self
            .decode_state
            .as_ref()
            .map_or_else(|| py.None().into_bound(py), |state| state.bind(py).clone());
        let metadata = SamplingMetadata::for_call(
            py,
            &batch.call(py, index)?.borrow(),
            &value.value.bind(py).get_item(-1)?,
            &output,
            vec![position],
            value.slot.clone_ref(py),
            &state,
            Vec::new(),
        )?;
        Ok(Some(SampleCandidate {
            index,
            task: last.task.clone_ref(py),
            logits: value.value.clone_ref(py),
            metadata: Some(Py::new(py, metadata)?),
            selection: None,
        }))
    }
}
