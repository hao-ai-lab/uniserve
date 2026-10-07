//! Execute image trajectories over request-owned guidance prefixes and latent banks.

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};
use uniserve_worker::TokenSelection;
use uniserve_worker_ipc::{CallKind, DrawLayout, ForwardMode, LatentParams};

use super::{BatchState, DiffusionStep, ForwardRow, PythonBackend, Trajectories};
use crate::worker::diffusion_state::DiffusionState;
use crate::worker::error::{invalid, native_error};
use crate::worker::latent::LatentBuffer;
use crate::worker::model_inputs::{AttentionRow, InputRow, Row, TokenRow};
use crate::worker::request::{KVConditioning, Request, resolve_prefix};

impl PythonBackend {
    fn image_builder<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let builder = { self.model_runner.borrow(py).image_builder.bind(py).clone() };
        if builder.is_none() {
            return Err(invalid(
                py,
                "image computation requires its denoiser input builder",
            ));
        }
        Ok(builder)
    }

    /// Refresh numerical schedules only when starting a new image or changing size.
    /// Prefix tokenization belongs to the same image; physical branch coordinates
    /// are refreshed separately for every submitted interval.
    fn image_state(
        &self,
        py: Python<'_>,
        request: &Py<Request>,
        params: &LatentParams,
        fresh: bool,
    ) -> PyResult<Py<DiffusionState>> {
        let size = py
            .import("uniserve.media.image")?
            .getattr("Config")?
            .call1((params.height, params.width))?;
        let previous = request
            .borrow(py)
            .diffusion
            .as_ref()
            .map(|state| state.clone_ref(py));
        if !fresh
            && let Some(previous) = previous
            && previous.borrow(py).size.bind(py).eq(&size)?
        {
            return Ok(previous);
        }

        let image = request.borrow(py).admission.bind(py).getattr("image")?;
        if image.is_none() {
            return Err(invalid(py, "flow call has no admitted image parameters"));
        }
        let state = py
            .import("uniserve_worker.execution.diffusion")?
            .call_method1("image_state", (self.image_builder(py)?, size, image))?
            .extract::<Py<DiffusionState>>()?;
        let mut request = request.borrow_mut(py);
        request.diffusion = Some(state.clone_ref(py));
        request.kv = Some(KVConditioning::default());
        Ok(state)
    }

    fn conditioning(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
    ) -> PyResult<(u32, u64, u32)> {
        let call = &batch.plan.calls[index];
        let buffer = call
            .kv_input
            .ok_or_else(|| invalid(py, "image trajectory requires conditioning KV"))?;
        let output = batch.pending(py, index);
        let output = output.borrow(py);
        let slot = output.request.borrow(py).request.slot() as u32;
        let visible = output.lock(py)?.progress.kv_visible_len;
        let tables = self
            .tables
            .as_ref()
            .ok_or_else(|| invalid(py, "image trajectory requires KV tables"))?;
        let tables = tables.borrow(py);
        let coordinates = tables
            .tables
            .coordinates(slot, visible)
            .map_err(|error| native_error(py, error))?;
        let cache = self
            .cache
            .as_ref()
            .ok_or_else(|| invalid(py, "flow conditioning requires cache export storage"))?;
        cache
            .borrow(py)
            .inner
            .validate_conditioning(call.request_key, buffer, slot, visible, &tables.tables)
            .map_err(|error| native_error(py, error))?;
        Ok(coordinates)
    }

    pub(in super::super) fn prepare_image_latent(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
    ) -> PyResult<()> {
        let call = &batch.plan.calls[index];
        let params = batch.latent_params(py, index)?;
        let cache = self.conditioning(py, batch, index)?;
        let pending = batch.pending(py, index);
        let request = pending.borrow(py).request.clone_ref(py);
        let seed = request
            .borrow(py)
            .request
            .admission()
            .image
            .as_ref()
            .ok_or_else(|| invalid(py, "media preparation has no admitted image parameters"))?
            .seed
            .unwrap_or(0);
        if pending.borrow(py).lock(py)?.progress.flow_step != 0 {
            return Err(invalid(
                py,
                "media preparation repeats an active latent trajectory",
            ));
        }
        let rng = call
            .rng
            .as_ref()
            .ok_or_else(|| invalid(py, "media preparation requires flow-noise RNG coordinates"))?;
        if rng.draw_layout != DrawLayout::FlowNoise
            || rng.seed != seed
            || rng.semantic_index_base < 1
        {
            return Err(invalid(
                py,
                "flow-noise coordinates disagree with the admitted image",
            ));
        }
        if call
            .latent_output
            .as_ref()
            .is_none_or(|output| output.generation < 1)
        {
            return Err(invalid(
                py,
                "media preparation requires a latent output generation",
            ));
        }

        let state = self.image_state(py, &request, params, true)?;
        let buffer = latent_buffer(py, batch, index)?;
        py.import("uniserve_worker.execution.diffusion")?
            .call_method1(
                "initial_latent",
                (
                    self.image_builder(py)?,
                    state.bind(py).getattr("size")?,
                    &buffer.get().value,
                    params.latent_units,
                    seed,
                    rng.semantic_index_base,
                ),
            )?;
        let pool = self
            .latents
            .as_ref()
            .ok_or_else(|| invalid(py, "image trajectory requires a latent pool"))?;
        pool.borrow_mut(py).initialize(
            py,
            i64::from(cache.0),
            &buffer,
            i64::from(params.latent_units),
        )?;
        batch
            .numerical
            .borrow(py)
            .complete_latent(py, call.request_key.request_id.0)?;
        pending.borrow(py).lock(py)?.set_cache_length(cache.1);
        Ok(())
    }

    pub(in super::super) fn initialize_trajectories(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        indices: &[usize],
    ) -> PyResult<Trajectories> {
        let mut trajectories = HashMap::new();
        for &index in indices {
            let call = &batch.plan.calls[index];
            let params = batch.latent_params(py, index)?;
            let cache = self.conditioning(py, batch, index)?;
            let (Some(input), Some(output)) = (&call.latent_input, &call.latent_output) else {
                return Err(invalid(
                    py,
                    "flow call requires latent input and output generations",
                ));
            };
            if call.rng.is_some()
                || input.generation < 1
                || output.generation < 1
                || input == output
            {
                return Err(invalid(
                    py,
                    "flow continuation must inherit its latent and RNG state",
                ));
            }
            let buffer = latent_buffer(py, batch, index)?;
            let pool = self
                .latents
                .as_ref()
                .ok_or_else(|| invalid(py, "image trajectory requires a latent pool"))?;
            pool.borrow_mut(py).gather_current(
                py,
                i64::from(cache.0),
                &buffer,
                i64::from(params.start_step),
                i64::from(input.generation),
                i64::from(params.latent_units),
                i64::from(params.height),
                i64::from(params.width),
            )?;

            let pending = batch.pending(py, index);
            let request = pending.borrow(py).request.clone_ref(py);
            let state = self.image_state(py, &request, params, false)?;
            let mut owner = request.borrow_mut(py);
            let kv = owner.kv.get_or_insert_with(KVConditioning::default);
            kv.cache = cache;
            kv.branches.clear();
            trajectories.insert(index, state);
        }
        Ok(trajectories)
    }

    pub(in super::super) fn finish_diffusion(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        indices: &[usize],
    ) -> PyResult<()> {
        for &index in indices {
            let call = &batch.plan.calls[index];
            let params = batch.latent_params(py, index)?;
            let input = call
                .latent_input
                .as_ref()
                .ok_or_else(|| invalid(py, "flow completion lost its input trajectory"))?;
            let pending = batch.pending(py, index);
            let request = pending.borrow(py).request.clone_ref(py);
            let owner = request.borrow(py);
            let cache = owner
                .kv
                .as_ref()
                .ok_or_else(|| invalid(py, "flow completion lost its conditioning"))?
                .cache;
            let buffer = latent_buffer(py, batch, index)?;
            let pool = self
                .latents
                .as_ref()
                .ok_or_else(|| invalid(py, "image trajectory requires a latent pool"))?;
            pool.borrow_mut(py).write_inactive(
                py,
                i64::from(cache.0),
                &buffer,
                i64::from(params.start_step),
                i64::from(input.generation),
                i64::from(params.latent_units),
                i64::from(params.height),
                i64::from(params.width),
            )?;
            batch
                .numerical
                .borrow(py)
                .complete_latent(py, call.request_key.request_id.0)?;
            pending.borrow(py).lock(py)?.set_cache_length(cache.1);
        }
        Ok(())
    }

    pub(super) fn prepare_diffusion_step(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        steps: &[(usize, u32)],
        trajectories: &Trajectories,
    ) -> PyResult<HashMap<usize, DiffusionStep>> {
        let mut inputs = HashMap::new();
        let mut prefixes = Vec::new();
        for &(index, step) in steps {
            let trajectory = trajectories[&index].borrow(py);
            let schedule = trajectory.schedules.bind(py).get_item("image")?;
            let guidance = trajectory
                .guidance
                .as_ref()
                .ok_or_else(|| invalid(py, "image diffusion requires guidance"))?
                .bind(py);
            let branches = guidance.call_method1("branches", (&schedule, step))?;
            let pending = batch.pending(py, index);
            let slot = pending.borrow(py).request.borrow(py).request.slot() as i64;
            // Image schedules are host tensors. No device scalar is read here.
            let time = schedule
                .getattr("timesteps")?
                .get_item(step)?
                .extract::<f64>()?;
            let pool = self
                .latents
                .as_ref()
                .ok_or_else(|| invalid(py, "image trajectory requires a latent pool"))?;
            let timestep = pool.borrow(py).fill_timestep(py, slot, time)?;

            let (names, rows) = self.guidance_prefixes(py, batch, index, &branches)?;
            prefixes.extend(rows.into_iter().map(|row| ForwardRow::new(index, row)));
            inputs.insert(
                index,
                DiffusionStep {
                    branches: names,
                    timestep,
                    step,
                },
            );
        }

        let values = self.forward_values(py, batch, &prefixes)?;
        for (prefix, result) in prefixes.into_iter().zip(values) {
            if result.is_none() {
                continue;
            }
            let pending = batch.pending(py, prefix.index);
            let task = prefix.task.bind(py);
            let count: u64 = Row::borrow(task)?.query_tokens(py)? as u64;
            self.record_token_kv(py, &mut pending.borrow_mut(py), task, count, false)?;
            let request = pending.borrow(py).request.clone_ref(py);
            let mut request = request.borrow_mut(py);
            let kv = request
                .kv
                .as_mut()
                .ok_or_else(|| invalid(py, "guidance prefix lost its conditioning"))?;
            let slot = task.cast::<InputRow>()?.borrow().request_pool_idx;
            for coordinates in kv.branches.values_mut() {
                if coordinates.0 == slot && coordinates.1 == 0 {
                    coordinates.1 = count;
                }
            }
        }
        Ok(inputs)
    }

    fn guidance_prefixes(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
        branches: &Bound<'_, PyAny>,
    ) -> PyResult<(Vec<String>, Vec<Py<PyAny>>)> {
        let pending = batch.pending(py, index);
        let request = pending.borrow(py).request.clone_ref(py);
        // Tokenization may invoke Python. Keep the host state locally so no
        // mutable request borrow spans a callback, and restore it on failure.
        let mut conditioning = request
            .borrow_mut(py)
            .kv
            .take()
            .ok_or_else(|| invalid(py, "image diffusion lost its conditioning"))?;
        let result = conditioning.prepare_prefixes(py, self, batch, index, branches);
        request.borrow_mut(py).kv = Some(conditioning);
        result
    }

    pub(super) fn prepare_diffusion_rows(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
        input: &DiffusionStep,
        trajectory: &Py<DiffusionState>,
    ) -> PyResult<Vec<ForwardRow>> {
        let pending = batch.pending(py, index);
        let request = pending.borrow(py).request.clone_ref(py);
        let position = pending.borrow(py).lock(py)?.progress.logical_position;
        let coordinates = {
            let request = request.borrow(py);
            let kv = request
                .kv
                .as_ref()
                .ok_or_else(|| invalid(py, "image diffusion lost its conditioning"))?;
            input
                .branches
                .iter()
                .map(|branch| {
                    let &(slot, visible, _) = kv
                        .branches
                        .get(branch)
                        .ok_or_else(|| invalid(py, "guidance branch has no KV prefix"))?;
                    Ok((
                        slot,
                        visible,
                        if branch == "conditioned" {
                            position
                        } else {
                            visible
                        },
                    ))
                })
                .collect::<PyResult<Vec<_>>>()?
        };
        let device = self
            .model_runner
            .borrow(py)
            .call_devices(
                py,
                &*batch
                    .call(py, index)?
                    .extract::<PyRef<crate::calls::Call>>()?,
            )?
            .into_bound(py)
            .get_item(1)?;
        let rows = trajectory.borrow_mut(py).prepare_inputs(
            py,
            &self.image_builder(py)?,
            &self.latent_values(py, batch, index)?,
            input.timestep.bind(py),
            coordinates,
            &device,
        )?;
        rows.bind(py)
            .try_iter()?
            .map(|row| Ok(ForwardRow::new(index, row?.unbind())))
            .collect()
    }

    pub(super) fn integrate_predictions(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        predictions: &IndexMap<usize, Vec<Py<PyAny>>>,
        inputs: &HashMap<usize, DiffusionStep>,
        trajectories: &Trajectories,
    ) -> PyResult<()> {
        for (&index, values) in predictions {
            let step = &inputs[&index];
            let values = PyTuple::new(py, values.iter().map(|value| value.bind(py)))?;
            let latent = self.latent_values(py, batch, index)?;
            let runner = self
                .model_runner
                .borrow(py)
                .diffusion_entry(
                    py,
                    &*batch
                        .call(py, index)?
                        .extract::<PyRef<crate::calls::Call>>()?,
                )?
                .into_bound(py);
            runner.call_method1(
                "integrate",
                (
                    &trajectories[&index],
                    latent,
                    &step.timestep,
                    values,
                    step.step,
                ),
            )?;
        }
        Ok(())
    }

    /// Borrow the sample rows, leaving physical page padding outside model inputs.
    pub(super) fn latent_values<'py>(
        &self,
        py: Python<'py>,
        batch: &BatchState,
        index: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let params = batch.latent_params(py, index)?;
        let buffer = latent_buffer(py, batch, index)?;
        buffer
            .get()
            .value
            .bind(py)
            .get_item(PySlice::new(py, 0, params.latent_units as isize, 1))
    }
}

fn latent_buffer<'py>(
    py: Python<'py>,
    batch: &BatchState,
    index: usize,
) -> PyResult<Bound<'py, LatentBuffer>> {
    let output = batch.pending(py, index);
    let output = output.borrow(py);
    output
        .latent_buffer
        .as_ref()
        .ok_or_else(|| invalid(py, "trajectory call has no bound latent buffer"))?
        .bind(py)
        .clone()
        .cast_into::<LatentBuffer>()
        .map_err(Into::into)
}

impl KVConditioning {
    /// Reuse tokenized source prefixes across intervals, while resolving their
    /// physical slots from the current submission. Prefix forwards precede all
    /// denoising rows and never advance the request's own token coordinates.
    #[allow(clippy::type_complexity)]
    fn prepare_prefixes(
        &mut self,
        py: Python<'_>,
        backend: &PythonBackend,
        batch: &BatchState,
        index: usize,
        branches: &Bound<'_, PyAny>,
    ) -> PyResult<(Vec<String>, Vec<Py<PyAny>>)> {
        let pending = batch.pending(py, index);
        let request = pending.borrow(py).request.clone_ref(py);
        let admitted = std::sync::Arc::clone(&request.borrow(py).request);
        let image = admitted
            .admission()
            .image
            .as_ref()
            .ok_or_else(|| invalid(py, "flow step requires admitted image parameters"))?;
        let negative = admitted
            .admission()
            .ar
            .as_ref()
            .map_or(&[][..], |params| params.negative_token_ids.as_slice());
        let builder = backend.image_builder(py)?;
        let model = backend.model_runner.bind(py);
        let numerical = batch.numerical.borrow(py);
        let descriptors = &numerical.forward_indices[index];
        let inputs = &batch.plan.forward;
        let count = branches.len()?;
        if descriptors.len() < count {
            return Err(invalid(
                py,
                "media denoise has incomplete forward-row metadata",
            ));
        }
        let (prefills, denoising) = descriptors.split_at(descriptors.len() - count);
        let mut names = Vec::with_capacity(count);
        let mut rows = Vec::new();
        let mut initialized_slots = HashSet::new();
        for (branch_index, branch) in branches.try_iter()?.enumerate() {
            let branch = branch?;
            let name: String = branch.getattr("value")?.extract()?;
            names.push(name.clone());

            if self.branches.contains_key(&name) {
                continue;
            }

            let source: String = builder
                .call_method1("branch_source", (&branch,))?
                .extract()?;
            let (tokens, conditioning) = match self.prefixes.entry(source.clone()) {
                std::collections::hash_map::Entry::Occupied(prefix) => prefix.into_mut(),
                std::collections::hash_map::Entry::Vacant(prefix) => prefix.insert(resolve_prefix(
                    py,
                    &{ model.borrow().flow_prompt.bind(py).clone() },
                    &source,
                    image.image_prompts.first().map_or("", String::as_str),
                    &image.negative_prompt,
                    negative,
                    &backend.worker.bind(py).getattr("tokenizer")?,
                )?),
            };
            let cache = self.cache;
            let coordinates = if *conditioning {
                cache
            } else {
                let descriptor = denoising[branch_index];
                let slot = inputs.request_pool_indices[descriptor];
                let has_prefill = prefills.iter().any(|&row| {
                    inputs.request_pool_indices[row] == slot
                        && inputs.seq_lens[row] == inputs.query_lens[row]
                        && inputs.query_lens[row] as usize == tokens.len()
                });
                let visible = if has_prefill {
                    0
                } else {
                    u64::from(inputs.seq_lens[descriptor] - inputs.query_lens[descriptor])
                };
                let tables = backend
                    .tables
                    .as_ref()
                    .ok_or_else(|| invalid(py, "flow prefixes require request page tables"))?;
                tables
                    .borrow(py)
                    .tables
                    .coordinates(slot, visible)
                    .map_err(|error| native_error(py, error))?
            };

            let length = if *conditioning {
                cache.1
            } else {
                tokens.len() as u64
            };
            if length > u64::from(coordinates.2) {
                return Err(invalid(py, "flow prefix exceeds scheduler params"));
            }
            if coordinates.1 != 0 && coordinates.1 != length {
                return Err(invalid(
                    py,
                    "flow branch prefix disagrees with its initialized physical state",
                ));
            }

            self.branches.insert(name, coordinates);

            // Branches that share a physical prefix submit one prefill.
            if coordinates.1 == 0 && !tokens.is_empty() && initialized_slots.insert(coordinates.0) {
                let row = TokenRow {
                    selection: Some(TokenSelection::Hidden),
                    ..TokenRow::default()
                }
                .with_tokens(
                    py,
                    InputRow {
                        kind: CallKind::Forward(ForwardMode::Prefill),
                        request_pool_idx: coordinates.0,
                    },
                    AttentionRow {
                        positions: None,
                        seq_len: coordinates.1 as i64,
                        write_kv: true,
                        causal: Some(true),
                    },
                    tokens,
                    coordinates.1,
                    None,
                )?;
                rows.push(row.into_any());
            }
        }
        Ok((names, rows))
    }
}
