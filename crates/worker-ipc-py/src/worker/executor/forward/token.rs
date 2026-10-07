//! Prepare token inputs and bind sampling results to pending request coordinates.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyInt, PySlice, PyTuple};
use uniserve_worker::TokenSelection;
use uniserve_worker_ipc::{Call, CallKind, ForwardMode};

use super::{BatchState, ForwardRow, ForwardValue, PythonBackend, SampleCandidate};
use crate::worker::error::{invalid, native_error, unsupported};
use crate::worker::model_inputs::{AttentionRow, InputRow, Row, TokenRow};
use crate::worker::pending::PendingOutput;
use crate::worker::sampling::SamplingMetadata;
use crate::worker::storage::Buffer;

impl PythonBackend {
    pub(super) fn prepare_token_row(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
    ) -> PyResult<ForwardRow> {
        let call = &batch.plan.calls[index];
        let pending = batch.pending(py, index);
        let output = pending.borrow(py);
        let scores = scores_prompt(py, &output)?;
        let visible = self.visible_length(py, &output)?;
        let decode = call.code == CallKind::Forward(ForwardMode::Decode);
        let verify = call.code == CallKind::Forward(ForwardMode::Verify);
        let predicate: Option<(Py<PyAny>, bool)> = output
            .predicate
            .as_ref()
            .map(|predicate| predicate.extract(py))
            .transpose()?;
        let indexed =
            decode && self.indexed_decode && predicate.as_ref().is_some_and(|(_, tagged)| *tagged);
        let relay = call.predicate.is_some()
            && !indexed
            && (decode || verify || call.input_token_ids.is_empty());
        let tokens = if indexed || (decode && relay) {
            &[][..]
        } else if decode {
            call.input_token_ids
                .get(..1)
                .ok_or_else(|| invalid(py, "last-sampled token source has no committed token"))?
        } else {
            &call.input_token_ids
        };
        if verify && !relay && tokens.len() < 2 {
            return Err(invalid(
                py,
                "fixed-parent verification requires current and draft tokens",
            ));
        }
        if tokens.is_empty() && !indexed && !relay {
            return Err(invalid(py, "text extension requires input tokens"));
        }

        let selection = if verify || (!decode && scores) {
            TokenSelection::AllLogits
        } else if decode || call.token_output.is_some() {
            TokenSelection::LastLogits
        } else {
            TokenSelection::Cache
        };
        let position = output.lock(py)?.progress.logical_position;
        let slot = output.request.borrow(py).request.slot();
        let current = if relay {
            if predicate.is_none() {
                return Err(invalid(py, "device token continuation is not registered"));
            }
            let state = self.decode_state.as_ref().ok_or_else(|| {
                unsupported(py, "device continuation has no request runtime state")
            })?;
            Some(
                state
                    .borrow(py)
                    .future_input_tokens(py)?
                    .bind(py)
                    .get_item((slot, PySlice::new(py, 0, 1, 1)))?,
            )
        } else {
            None
        };
        let (decode_predicate, tagged) =
            predicate.map_or((None, false), |(value, tagged)| (Some(value), tagged));
        let row = TokenRow {
            selection: Some(selection),
            decode_predicate,
            decode_predicate_tagged: tagged,
            decode_force_finish: call
                .sampling_state
                .as_ref()
                .is_some_and(|state| state.force_finish),
            request_indexed_decode: indexed,
            ..TokenRow::default()
        }
        .with_tokens(
            py,
            InputRow {
                kind: call.code,
                request_pool_idx: slot as u32,
            },
            AttentionRow {
                positions: None,
                seq_len: visible as i64,
                write_kv: true,
                causal: Some(true),
            },
            tokens,
            position,
            current.as_ref(),
        )?;
        Ok(ForwardRow::new(index, row.into_any()))
    }

    /// Guidance prefixes share this KV extent check but do not update the
    /// request's decode slot. Physical reservation happened before forward.
    pub(super) fn record_token_kv(
        &self,
        py: Python<'_>,
        output: &mut PendingOutput,
        task: &Bound<'_, PyAny>,
        count: u64,
        update_runtime: bool,
    ) -> PyResult<()> {
        let query: u64 = Row::borrow(task)?.query_tokens(py)? as u64;
        if count > query {
            return Err(PyRuntimeError::new_err(
                "KV commit count is outside the task query span",
            ));
        }
        if count == 0 {
            return Ok(());
        }

        let length = task.cast::<AttentionRow>()?.borrow().seq_len as u64 + count;
        let slot: u32 = task.cast::<InputRow>()?.borrow().request_pool_idx;
        let tables = self
            .tables
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("KV commit requires request page tables"))?;
        if length > u64::from(tables.borrow(py).tables.allocated_length(slot)) {
            return Err(PyRuntimeError::new_err(
                "KV task exceeds its scheduler block table",
            ));
        }

        if update_runtime && self.decode_state.is_some() {
            if slot as usize != output.request.borrow(py).request.slot() {
                return Err(PyRuntimeError::new_err(
                    "token KV update crossed request slots",
                ));
            }
            output.token_update.cache_length = Some(length.into_pyobject(py)?.into_any().unbind());
        }
        Ok(())
    }

    pub(super) fn visible_length(&self, py: Python<'_>, output: &PendingOutput) -> PyResult<u64> {
        let tables = self
            .tables
            .as_ref()
            .ok_or_else(|| unsupported(py, "call requires request-to-token storage"))?;
        let slot = output.request.borrow(py).request.slot() as u32;
        let visible = output.lock(py)?.progress.kv_visible_len;
        tables
            .borrow(py)
            .tables
            .coordinates(slot, visible)
            .map(|(_, visible, _)| visible)
            .map_err(|error| native_error(py, error))
    }

    pub(super) fn token_length(
        &self,
        py: Python<'_>,
        output: &PendingOutput,
        fallback: u64,
    ) -> PyResult<u64> {
        let update = &output.token_update;
        match &update.cache_length {
            None => Ok(fallback),
            Some(value) if value.bind(py).is_instance_of::<PyInt>() => value.extract(py),
            // Never extract an integer from a device-selected extent: that
            // would synchronize verification before its result is ready.
            Some(_) => Err(PyRuntimeError::new_err(
                "dynamic KV length requires a speculative selection",
            )),
        }
    }

    fn visual_advance(&self, py: Python<'_>) -> PyResult<u64> {
        let builder = { self.model_runner.borrow(py).image_builder.bind(py).clone() };
        if builder.is_none() {
            Ok(1)
        } else {
            Ok(builder.getattr("rope_advance")?.extract::<i64>()?.max(1) as u64)
        }
    }

    fn finish_visual(&self, py: Python<'_>, output: &PendingOutput, call: &Call) -> PyResult<()> {
        let advance = if call.completion_output.is_some() {
            self.visual_advance(py)?
        } else {
            0
        };
        let length = self.token_length(py, output, self.visible_length(py, output)?)?;
        output
            .lock(py)?
            .advance_tokens(advance, Some(length), None, false);
        Ok(())
    }

    pub(super) fn prepare_sample(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        row: &ForwardRow,
        value: &ForwardValue,
    ) -> PyResult<Option<Py<SamplingMetadata>>> {
        let call = &batch.plan.calls[row.index];
        let pending = batch.pending(py, row.index);
        let mut output = pending.borrow_mut(py);
        let task = row.task.bind(py);
        let start = output.lock(py)?.progress.logical_position;
        let visual = call.writes_visual_state();
        let count = if call.code == CallKind::Forward(ForwardMode::Decode) {
            1
        } else {
            Row::borrow(task)?.query_tokens(py)? as u64
        };
        if visual
            || matches!(
                call.code,
                CallKind::Forward(ForwardMode::Prefill | ForwardMode::Decode)
            )
        {
            self.record_token_kv(
                py,
                &mut output,
                task,
                count,
                call.code != CallKind::Forward(ForwardMode::Decode),
            )?;
            if call.token_output.is_none()
                && (visual || call.code == CallKind::Forward(ForwardMode::Prefill))
            {
                if visual {
                    self.finish_visual(py, &output, call)?;
                } else {
                    let length =
                        self.token_length(py, &output, self.visible_length(py, &output)?)?;
                    output
                        .lock(py)?
                        .advance_tokens(0, Some(length), None, false);
                }
                return Ok(None);
            }
        }

        let (logits, positions, draft) = if visual {
            (
                value.value.bind(py).get_item(-1)?,
                vec![start + self.visual_advance(py)?],
                Vec::new(),
            )
        } else if matches!(
            call.code,
            CallKind::Forward(ForwardMode::Prefill | ForwardMode::Decode)
        ) {
            (
                value.value.bind(py).get_item(-1)?,
                vec![start + count],
                Vec::new(),
            )
        } else {
            let first = usize::from(call.predicate.is_none());
            let draft = call.input_token_ids[first..].to_vec();
            let positions = (start + 1..start + draft.len() as u64 + 2).collect();
            (value.value.bind(py).clone(), positions, draft)
        };
        let state = self.decode_state.as_ref().map_or_else(
            || py.None().into_bound(py),
            |state| state.bind(py).as_any().clone(),
        );
        let metadata = SamplingMetadata::for_call(
            py,
            &batch.call(py, row.index)?.borrow(),
            &logits,
            &output,
            positions,
            value.slot.clone_ref(py),
            &state,
            draft,
        )?;
        Ok(Some(Py::new(py, metadata)?))
    }

    pub(super) fn finish_sample(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        sample: &SampleCandidate,
        selected: Py<PyAny>,
    ) -> PyResult<()> {
        let call = &batch.plan.calls[sample.index];
        let pending = batch.pending(py, sample.index);
        let mut output = pending.borrow_mut(py);
        let task = sample.task.bind(py);
        let progress = output.lock(py)?.progress;
        let start = progress.logical_position;
        let decode = call.code == CallKind::Forward(ForwardMode::Decode);
        let penalty = sample.metadata.as_ref().and_then(|metadata| {
            metadata
                .borrow(py)
                .penalty_base
                .as_ref()
                .map(|value| value.clone_ref(py))
        });

        let (position, sampling_position, increment) = if call.writes_visual_state() {
            output.lock(py)?.advance_tokens(0, None, None, true);
            let advance = if call.completion_output.is_some() {
                self.visual_advance(py)?
            } else {
                1
            };
            self.finish_visual(py, &output, call)?;
            (
                (start + advance).into_pyobject(py)?.into_any().unbind(),
                (progress.rng_counter + 1)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
                false,
            )
        } else if matches!(
            call.code,
            CallKind::Forward(ForwardMode::Prefill | ForwardMode::Decode)
        ) {
            if !decode
                && task
                    .cast::<AttentionRow>()?
                    .borrow()
                    .causal
                    .unwrap_or(false)
                && scores_prompt(py, &output)?
            {
                self.score_prompt(py, batch, &mut output, start, task, sample.logits.bind(py))?;
            }
            let count = if decode {
                1
            } else {
                Row::borrow(task)?.query_tokens(py)? as u64
            };
            let position = if call.writes_context() {
                Some(next_position(py, task)?)
            } else {
                None
            };
            let fallback = if decode {
                task.cast::<AttentionRow>()?.borrow().seq_len as u64 + count
            } else {
                self.visible_length(py, &output)?
            };
            let length = self.token_length(py, &output, fallback)?;
            let mut state = output.lock(py)?;
            state.advance_tokens(count, Some(length), position, true);
            (
                state
                    .progress
                    .logical_position
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
                state
                    .progress
                    .rng_counter
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
                decode,
            )
        } else {
            let metadata = sample
                .metadata
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("verification requires sampling inputs"))?
                .borrow(py);
            let draft = metadata.inner.draft_token_ids.clone();
            let terminal = metadata.inner.terminal_draft_prefix;
            drop(metadata);
            let visible: u64 = task.cast::<AttentionRow>()?.borrow().seq_len as u64;
            let initialized = visible + Row::borrow(task)?.query_tokens(py)? as u64;
            let (length, position, sampling_position): (Py<PyAny>, Py<PyAny>, Py<PyAny>) = py
                .import("uniserve_worker.execution.token")?
                .call_method1(
                    "speculative_positions",
                    (&selected, visible, start, progress.rng_counter),
                )?
                .extract()?;
            output.token_update.cache_length = Some(length);
            output
                .lock(py)?
                .set_speculation(draft, terminal, visible, initialized);
            (position, sampling_position, false)
        };

        if self.decode_state.is_some() {
            let update = &mut output.token_update;
            update.sampled = Some(selected);
            update.logical_position = position;
            update.sampling_position = sampling_position;
            update.penalty_base = penalty;
            update.decode_increment = increment;
        }
        Ok(())
    }

    /// Select reserved device outputs in call order; tensor assembly remains
    /// numerical, and one store operation submits each complete column.
    pub(super) fn write_sample_tensors(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        samples: &[SampleCandidate],
        values: &[Py<PyAny>],
    ) -> PyResult<()> {
        let numerical = py.import("uniserve_worker.sampling.result")?;
        for transitions in [true, false] {
            let mut writes = Vec::new();
            let mut selected = Vec::new();
            for (sample, value) in samples.iter().zip(values) {
                let pending = batch.pending(py, sample.index);
                let output = pending.borrow(py);
                let write = if transitions {
                    &output.transition_write
                } else {
                    &output.token_write
                };
                if let Some(write) = write {
                    writes.push(write.bind(py).cast::<Buffer>()?.clone());
                    selected.push(value.bind(py));
                }
            }
            if writes.is_empty() {
                continue;
            }
            let selected = PyTuple::new(py, selected)?;
            let column = if transitions {
                numerical.call_method1("transition_values", (selected,))?
            } else {
                numerical
                    .call_method1("sample_columns", (selected, ("tagged_tokens",)))?
                    .get_item(0)?
            };
            self.tensors.get().write_scalars(
                py,
                writes,
                column.call_method1("reshape", (-1,))?,
                None,
            )?;
        }
        Ok(())
    }
}

pub(super) fn scores_prompt(py: Python<'_>, output: &PendingOutput) -> PyResult<bool> {
    let request = output.request.borrow(py);
    let params = &request
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
        .sampling;
    Ok(params.return_prompt_logprobs || params.n_prompt_logprobs > 0)
}

pub(super) fn next_position(py: Python<'_>, task: &Bound<'_, PyAny>) -> PyResult<u64> {
    // Context positions are host numerical inputs, including M-RoPE's
    // temporal axis. This does not inspect device-selected decode positions.
    py.import("uniserve_worker.execution.token")?
        .call_method1("next_position", (task,))?
        .extract()
}
