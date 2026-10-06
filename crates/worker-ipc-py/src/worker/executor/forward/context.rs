//! Prepare text and visual contexts, then commit their KV and prompt scores.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::token::{next_position, scores_prompt, token_options};
use super::{BatchState, ForwardRow, ForwardValue, PythonBackend, SampleCandidate};
use crate::sampling::SamplingParams;
use crate::worker::error::{invalid, unsupported};
use crate::worker::pending::PendingOutput;
use crate::worker::sampling::SamplingMetadata;

impl PythonBackend {
    pub(super) fn prepare_context_rows(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
    ) -> PyResult<Vec<ForwardRow>> {
        let call = &batch.plan.calls[index];
        let pending = batch.pending(py, index);
        let output = pending.borrow(py);
        let scores = scores_prompt(py, &output)?;
        let mut position = output.lock(py)?.progress.logical_position;
        let mut visible = self.visible_length(py, &output)?;
        let slot = output.request.borrow(py).request.slot();
        let token = py.import("uniserve_worker.execution.token")?;
        let mut rows = Vec::new();
        let mut begin = 0;

        // Text spans and feature blocks share one increasing KV interval.
        // A block can advance M-RoPE by less than its physical token count.
        // The final empty block flushes any trailing text.
        for (block_index, block) in call
            .vision_inputs
            .iter()
            .map(Some)
            .chain(std::iter::once(None))
            .enumerate()
        {
            let end = block.map_or(call.input_token_ids.len(), |block| block.offset as usize);
            if end < begin || end > call.input_token_ids.len() {
                return Err(invalid(
                    py,
                    "context blocks must lie in order within the call's tokens",
                ));
            }

            if end > begin {
                let selection = if scores {
                    "ALL_LOGITS"
                } else if block.is_none() && call.token_output.is_some() {
                    "LAST_LOGITS"
                } else {
                    "CACHE"
                };
                let options = token_options(
                    py,
                    call.code,
                    &call.input_token_ids[begin..end],
                    position,
                    slot,
                    visible,
                    selection,
                )?;
                let row = token.call_method("token_row", (), Some(&options))?;
                position += (end - begin) as u64;
                visible += (end - begin) as u64;
                rows.push(ForwardRow::new(index, row.unbind()));
            }

            if block.is_some() {
                let last = block_index + 1 == call.vision_inputs.len()
                    && end == call.input_token_ids.len();
                let reference = batch
                    .call(py, index)?
                    .getattr("vision_inputs")?
                    .get_item(block_index)?
                    .getattr("feature")?;
                let row = self.feature_row(
                    py,
                    batch,
                    index,
                    reference,
                    position,
                    visible,
                    false,
                    scores || (last && call.token_output.is_some()),
                )?;
                position = next_position(py, row.bind(py))?;
                visible += row.bind(py).getattr("query_tokens")?.extract::<u64>()?;
                rows.push(ForwardRow::new(index, row));
            }

            begin = end;
        }

        let numerical = batch.numerical.borrow(py);
        let descriptors = &numerical.forward_indices[index];
        let inputs = &batch.plan.forward;
        let mut count = 0;
        let mut aligned = descriptors.len() == rows.len();
        for (&descriptor, row) in descriptors.iter().zip(&rows) {
            let query = row
                .task
                .bind(py)
                .getattr("query_tokens")?
                .extract::<u64>()?;
            let prefix = row.task.bind(py).getattr("seq_len")?.extract::<u64>()?;
            count += query;
            aligned &= inputs.request_pool_indices[descriptor] as usize == slot
                && u64::from(inputs.query_lens[descriptor]) == query
                && u64::from(inputs.seq_lens[descriptor]) == prefix + query;
        }
        if !aligned || count > u64::from(call.bounds.max_tokens) {
            return Err(invalid(
                py,
                "context rows disagree with the call's forward rows",
            ));
        }
        Ok(rows)
    }

    pub(super) fn prepare_visual_row(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
    ) -> PyResult<ForwardRow> {
        let plan = &batch.plan.calls[index];
        if plan.vision_inputs.len() + usize::from(plan.latent_feature_input.is_some()) != 1 {
            return Err(invalid(
                py,
                "visual extend requires exactly one feature tensor",
            ));
        }
        let call = batch.call(py, index)?;
        let reference = if plan.latent_feature_input.is_some() {
            call.getattr("latent_feature_input")?
        } else {
            call.getattr("vision_inputs")?
                .get_item(0)?
                .getattr("feature")?
        };
        let pending = batch.pending(py, index);
        let output = pending.borrow(py);
        scores_prompt(py, &output)?;
        let position = output.lock(py)?.progress.logical_position;
        let visible = self.visible_length(py, &output)?;
        let row = self.feature_row(
            py,
            batch,
            index,
            reference,
            position,
            visible,
            true,
            plan.token_output.is_some(),
        )?;
        if row.bind(py).getattr("query_tokens")?.extract::<u64>()?
            > u64::from(plan.bounds.max_tokens)
        {
            return Err(invalid(
                py,
                "image state query span exceeds the call token bound",
            ));
        }
        Ok(ForwardRow::new(index, row))
    }

    /// Acquire one feature read for the call and borrow it until completion.
    /// Numerical builders receive only its tensor, image dimensions and KV coordinates.
    #[allow(clippy::too_many_arguments)]
    fn feature_row(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
        reference: Bound<'_, PyAny>,
        position: u64,
        visible: u64,
        close_image: bool,
        logits: bool,
    ) -> PyResult<Py<PyAny>> {
        let call = batch.call(py, index)?;
        let model = self.model_runner.bind(py);
        let device = model.call_method1("call_devices", (&call,))?.get_item(0)?;
        let read =
            self.tensors
                .get()
                .consume(py, reference, call.getattr("call_id")?, Some(device))?;
        let pending = batch.pending(py, index);
        let output = pending.borrow(py);
        output.feature_reads.bind(py).append(&read)?;
        let read = read.borrow(py);
        let metadata = read
            .metadata
            .as_ref()
            .map(|value| value.bind(py))
            .ok_or_else(|| invalid(py, "visual input requires encoder feature metadata"))?;
        if !metadata.is_instance(
            &py.import("uniserve_worker.storage.tensor_store")?
                .getattr("FeatureMetadata")?,
        )? {
            return Err(invalid(
                py,
                "visual input requires encoder feature metadata",
            ));
        }

        let request = output.request.borrow(py);
        let options = PyDict::new(py);
        options.set_item("slot", request.request.slot())?;
        options.set_item("seq_len", visible)?;
        options.set_item("model_runner", model)?;
        let vision = batch.plan.calls[index].latent_feature_input.is_none();
        if vision {
            let tables = self
                .tables
                .as_ref()
                .ok_or_else(|| unsupported(py, "vision input requires KV storage"))?;
            let capacity = tables
                .borrow(py)
                .tables
                .allocated_length(request.request.slot() as u32);
            if visible > u64::from(capacity) {
                return Err(invalid(
                    py,
                    "vision row prefix lies outside the request's KV extent",
                ));
            }
            let images = request.request.admission().input_images;
            if !close_image && images == 0 {
                return Err(invalid(py, "request admission declares no input images"));
            }
            options.set_item("input_images", (!close_image).then_some(images))?;
            options.set_item("close_image", close_image)?;
            options.set_item("logits", logits)?;
        }

        py.import("uniserve_worker.execution.image")?
            .call_method(
                if vision {
                    "vision_state_row"
                } else {
                    "latent_state_row"
                },
                (
                    &read.tensor,
                    metadata.getattr("height")?,
                    metadata.getattr("width")?,
                    position,
                ),
                Some(&options),
            )
            .map(Bound::unbind)
    }

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
