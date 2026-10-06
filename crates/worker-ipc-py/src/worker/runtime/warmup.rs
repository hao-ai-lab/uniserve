//! Bounded startup requests through the native serving executor.

mod scenarios;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use uniserve_core::{ByteAllocator, CallId, RequestId, UnitId};
use uniserve_worker::Request;
use uniserve_worker_ipc::{
    Batch, BatchCommand, BlockTable, BufferAllocation, BufferId, CacheUnitAllocation, Call,
    CallKind, CallStatus, ForwardMode, LatentParams, MediaCall, NewRequest, RequestKey, TensorRef,
    TransferMode, WorkerInfo,
};

use super::{Worker, closed};
use crate::worker::error::{invalid, native_error};
use crate::worker::executor::Executor;
use crate::worker::request::{RequestPool, resolve_prefix};

/// Temporary scheduler allocations for real startup requests. Physical owners
/// remain on Worker; Finish/Free must complete before these addresses are reused.
struct Warmup {
    runner: Py<crate::worker::model_executor::ModelExecutor>,
    tokenizer: Py<PyAny>,
    requests: Py<RequestPool>,
    executor: Py<Executor>,
    info: WorkerInfo,
    batch_id: u64,
    kv_units: HashMap<(RequestKey, usize), Vec<UnitId>>,
    prefix_units: HashMap<(RequestKey, usize), Vec<UnitId>>,
    prefix_slots: HashMap<RequestKey, u32>,
    latent_pages: HashMap<RequestKey, Vec<u32>>,
    buffers: HashMap<BufferId, BufferAllocation>,
    bytes: ByteAllocator,
}

pub(super) fn run(worker: &Bound<'_, Worker>) -> PyResult<()> {
    let py = worker.py();
    let executor = Worker::executor(worker)?;
    let info = executor.borrow(py).info()?.clone();
    let device = py
        .import("torch")?
        .getattr("device")?
        .call1((&info.device,))?;
    if device.getattr("type")?.extract::<String>()? != "cuda" {
        return Ok(());
    }

    let mut warmup = {
        let worker = worker.borrow();
        if !worker
            .requests
            .as_ref()
            .ok_or_else(closed)?
            .borrow(py)
            .pool
            .request_ids()
            .is_empty()
        {
            return Ok(());
        }

        Warmup {
            runner: worker.runner.as_ref().ok_or_else(closed)?.clone_ref(py),
            tokenizer: worker
                .tokenizer
                .as_ref()
                .map_or_else(|| py.None(), |value| value.clone_ref(py)),
            requests: worker.requests.as_ref().ok_or_else(closed)?.clone_ref(py),
            executor,
            bytes: ByteAllocator::new(info.buffer_pool_bytes),
            info,
            batch_id: 0,
            kv_units: HashMap::new(),
            prefix_units: HashMap::new(),
            prefix_slots: HashMap::new(),
            latent_pages: HashMap::new(),
            buffers: HashMap::new(),
        }
    };

    warmup.tokens(py)?;
    warmup.canvas(py)?;
    warmup.flow(py)
}

impl Warmup {
    fn request(&self, py: Python<'_>, key: RequestKey) -> PyResult<Arc<Request>> {
        self.requests
            .borrow(py)
            .pool
            .get(key.request_id.0)
            .map(Arc::clone)
            .map_err(|error| native_error(py, error))
    }

    fn supports(&self, code: CallKind) -> bool {
        self.info.supported_calls.contains(&code)
    }

    fn call(&self, key: RequestKey, row: usize, code: CallKind) -> Call {
        Call::new(key, CallId::new(self.batch_id + 1, row as u32), code)
    }

    fn submit(&mut self, py: Python<'_>, batch: Batch, retain_outputs: bool) -> PyResult<()> {
        let buffers = batch
            .calls
            .iter()
            .flat_map(|call| {
                call.outputs
                    .iter()
                    .chain(call.token_output.iter())
                    .chain(call.completion_output.iter())
                    .chain(call.transition_output.iter())
                    .chain(call.image_output.iter())
                    .map(TensorRef::buffer_id)
            })
            .collect();

        let result = self.executor.borrow_mut(py).run(py, Arc::new(batch))?;
        let failures: Vec<_> = result
            .output
            .completions
            .iter()
            .filter(|output| output.status == CallStatus::Error)
            .map(|output| {
                format!(
                    "request={} call={:?} code={:?}",
                    output.request_key.request_id.0, output.call_id, output.error_code
                )
            })
            .collect();
        if !failures.is_empty() {
            return Err(PyRuntimeError::new_err(format!(
                "startup warmup execution failed: {}",
                failures.join(", ")
            )));
        }

        if !retain_outputs {
            self.free_products(py, buffers)?;
        }
        Ok(())
    }

    fn controls(&mut self, py: Python<'_>, commands: Vec<BatchCommand>) -> PyResult<()> {
        self.batch_id += 1;
        self.submit(
            py,
            Batch::new(self.batch_id, Vec::new(), Vec::new()).with_commands(commands),
            true,
        )
    }

    fn release_buffers(&mut self, buffers: impl IntoIterator<Item = BufferId>) {
        for buffer in buffers {
            if let Some(allocation) = self.buffers.remove(&buffer) {
                self.bytes
                    .free(allocation.offset..allocation.offset + allocation.bytes);
            }
        }
    }

    fn free_products(&mut self, py: Python<'_>, buffers: Vec<BufferId>) -> PyResult<()> {
        if buffers.is_empty() {
            return Ok(());
        }

        self.controls(
            py,
            buffers
                .iter()
                .map(|&buffer| BatchCommand::Free { buffer })
                .collect(),
        )?;
        self.release_buffers(buffers);
        Ok(())
    }

    fn finish(&mut self, py: Python<'_>, key: RequestKey) -> PyResult<()> {
        self.controls(
            py,
            vec![BatchCommand::Finish {
                request_key: key,
                retained_buffers: Vec::new(),
            }],
        )?;

        // Finish drains device and remote readers. Serving keeps the terminal
        // row for its scheduler; startup has no further messages for this row.
        self.requests
            .borrow_mut(py)
            .drop_request(py, key.request_id.0);
        self.kv_units.retain(|(owner, _), _| *owner != key);
        self.prefix_units.retain(|(owner, _), _| *owner != key);
        self.prefix_slots.remove(&key);
        self.latent_pages.remove(&key);

        let buffers: Vec<_> = self
            .buffers
            .keys()
            .copied()
            .filter(|buffer| buffer.owner == key)
            .collect();
        self.release_buffers(buffers);
        Ok(())
    }

    fn allocate_buffer(
        &mut self,
        py: Python<'_>,
        product: &TensorRef,
    ) -> PyResult<BufferAllocation> {
        let buffer = product.buffer_id();
        if let Some(allocation) = self.buffers.get(&buffer) {
            return Ok(*allocation);
        }

        let bytes = product.max_bytes();
        let range = self.bytes.allocate(bytes, 256).ok_or_else(|| {
            invalid(
                py,
                "warmup persistent buffer allocation exceeds resident capacity",
            )
        })?;
        let allocation = BufferAllocation {
            buffer,
            offset: range.start,
            bytes,
        };
        self.buffers.insert(buffer, allocation);
        Ok(allocation)
    }

    fn grow_tables(
        &mut self,
        py: Python<'_>,
        key: RequestKey,
        slot: u32,
        tokens: u32,
        prefix: bool,
        batch: &mut Batch,
    ) -> PyResult<()> {
        let cache = self
            .info
            .kv_cache
            .as_ref()
            .ok_or_else(|| invalid(py, "warmup KV call requires cache storage"))?;
        let occupied: HashSet<_> = self
            .kv_units
            .values()
            .chain(self.prefix_units.values())
            .flatten()
            .copied()
            .collect();
        let mut free = (1..cache.num_units)
            .map(UnitId)
            .filter(|unit| !occupied.contains(unit));
        let leases = if prefix {
            &mut self.prefix_units
        } else {
            &mut self.kv_units
        };

        for (group, shape) in cache.groups.iter().enumerate() {
            let lease = leases.entry((key, group)).or_default();
            let pages = tokens.div_ceil(shape.page_tokens);
            let missing = (pages * shape.units_per_page)
                .checked_sub(lease.len() as u32)
                .ok_or_else(|| invalid(py, "warmup call regresses its KV capacity"))?;
            let allocated: Vec<_> = free.by_ref().take(missing as usize).collect();
            if allocated.len() != missing as usize {
                return Err(invalid(
                    py,
                    "warmup KV allocation exceeds resident capacity",
                ));
            }

            lease.extend(&allocated);
            batch.block_tables.push(BlockTable {
                request_pool_idx: slot,
                group_id: group as u32,
                start_page: 0,
                unit_ids: lease.clone(),
                allocated_tokens: pages * shape.page_tokens,
            });
            if !allocated.is_empty() {
                batch.new_cache_units.push(CacheUnitAllocation {
                    request_pool_idx: slot,
                    group_id: group as u32,
                    unit_ids: allocated,
                });
            }
        }
        Ok(())
    }

    fn plan(
        &mut self,
        py: Python<'_>,
        admissions: Vec<NewRequest>,
        calls: Vec<Call>,
        size: Option<(u32, u32)>,
    ) -> PyResult<Batch> {
        self.batch_id += 1;
        let mut batch = Batch::new(self.batch_id, admissions, calls);
        batch.collective_seq = self.batch_id * 16 + 1;
        let mut allocations = HashMap::new();

        let calls = std::mem::take(&mut batch.calls);
        for (row, call) in calls.iter().enumerate() {
            for product in call.buffer_inputs().chain(call.buffer_outputs()) {
                let allocation = self.allocate_buffer(py, product)?;
                allocations.insert(allocation.buffer, allocation);
            }

            let request = self
                .requests
                .borrow(py)
                .pool
                .peek(call.request_key.request_id.0)
                .cloned();
            let slot = match &request {
                Some(request) => request.slot() as u32,
                None => batch
                    .admissions()
                    .find(|admission| admission.request_key == call.request_key)
                    .map(|admission| admission.request_pool_idx)
                    .ok_or_else(|| invalid(py, "warmup call has no request-pool binding"))?,
            };
            let progress = request
                .as_ref()
                .map(|request| request.progress())
                .transpose()
                .map_err(|error| native_error(py, error))?
                .unwrap_or_default();

            let input_length = match call.code {
                CallKind::Forward(
                    ForwardMode::Prefill | ForwardMode::Decode | ForwardMode::Verify,
                ) => call.bounds.max_tokens,
                _ => 0,
            };
            if matches!(
                call.code,
                CallKind::Forward(_)
                    | CallKind::Transfer(TransferMode::KvExport | TransferMode::KvInstall)
                    | CallKind::Media(MediaCall::LatentPreparation | MediaCall::Denoising)
            ) {
                self.grow_tables(
                    py,
                    call.request_key,
                    slot,
                    progress.kv_visible_len as u32 + input_length,
                    false,
                    &mut batch,
                )?;
            }
            if input_length > 0 {
                add_row(
                    &mut batch,
                    row,
                    slot,
                    progress.kv_visible_len as u32,
                    input_length,
                    true,
                );
            } else if call.code == CallKind::Forward(ForwardMode::TokenDenoising) {
                add_row(
                    &mut batch,
                    row,
                    slot,
                    progress.kv_visible_len as u32,
                    call.bounds.max_tokens,
                    false,
                );
            }

            if matches!(
                call.code,
                CallKind::Media(MediaCall::LatentPreparation | MediaCall::Denoising)
            ) || call.latent_input.is_some()
            {
                let (height, width) = size.map_or_else(|| self.image_size(py), Ok)?;
                let shape = self.latent_shape(py, height, width)?;
                let units = shape[0];
                let pages = units.div_ceil(self.info.latent_page_units);
                let occupied: HashSet<_> = self.latent_pages.values().flatten().copied().collect();
                let table = self.latent_pages.entry(call.request_key).or_default();
                let missing = pages.checked_sub(table.len() as u32).ok_or_else(|| {
                    invalid(py, "warmup latent allocation regresses its physical extent")
                })?;
                let allocated: Vec<_> = (1..self.info.latent_pages)
                    .filter(|page| !occupied.contains(page))
                    .take(missing as usize)
                    .collect();
                if allocated.len() != missing as usize {
                    return Err(invalid(
                        py,
                        "warmup latent allocation exceeds resident capacity",
                    ));
                }

                table.extend(allocated);
                batch.latent_params.push(LatentParams {
                    request_key: call.request_key,
                    call_id: call.call_id,
                    page_table: table.clone(),
                    latent_units: units,
                    height,
                    width,
                    start_step: progress.flow_step as u32,
                    step_count: if call.code == CallKind::Media(MediaCall::Denoising) {
                        call.bounds.max_tokens
                    } else {
                        0
                    },
                });
                if call.code == CallKind::Media(MediaCall::Denoising) {
                    self.flow_rows(py, call, row, slot, &mut batch)?;
                }
            }
        }

        batch.calls = calls;
        batch.buffer_allocations = allocations.into_values().collect();
        Ok(batch)
    }

    fn flow_rows(
        &mut self,
        py: Python<'_>,
        call: &Call,
        row: usize,
        main_slot: u32,
        batch: &mut Batch,
    ) -> PyResult<()> {
        let request = self.request(py, call.request_key)?;
        let image = request
            .admission()
            .image
            .as_ref()
            .ok_or_else(|| invalid(py, "generation warmup has no admitted image runtime"))?;
        let progress = request
            .progress()
            .map_err(|error| native_error(py, error))?;
        let view = self
            .requests
            .borrow(py)
            .get(py, call.request_key.request_id.0)?;
        let builder = self.runner.bind(py).getattr("image_builder")?;

        // Latent preparation already constructed the numerical trajectory.
        // Read its active guidance branches without rebuilding its schedules.
        let trajectory = view
            .borrow(py)
            .diffusion
            .as_ref()
            .ok_or_else(|| invalid(py, "warmup image has no prepared trajectory"))?
            .clone_ref(py);
        let schedule = trajectory
            .bind(py)
            .getattr("schedules")?
            .get_item("image")?;
        let branches = trajectory
            .bind(py)
            .getattr("guidance")?
            .call_method1("branches", (schedule, progress.flow_step))?;
        let query: u32 = builder
            .call_method1("sequence_length", (trajectory.bind(py).getattr("size")?,))?
            .extract()?;
        let prompt = self.runner.bind(py).getattr("flow_prompt")?;
        let mut prefixes = Vec::new();
        let mut alternative: Option<Vec<u32>> = None;
        for branch in branches.try_iter()? {
            let source: String = builder
                .call_method1("branch_source", (branch?,))?
                .extract()?;
            let (tokens, conditioned) = resolve_prefix(
                py,
                &prompt,
                &source,
                image.image_prompts.first().map_or("", String::as_str),
                &image.negative_prompt,
                &[],
                self.tokenizer.bind(py),
            )?;
            if !conditioned {
                if alternative.as_ref().is_some_and(|prefix| *prefix != tokens) {
                    return Err(invalid(
                        py,
                        "warmup flow has multiple distinct alternative prefixes",
                    ));
                }
                alternative = Some(tokens.clone());
            }
            prefixes.push((tokens, conditioned));
        }

        let mut alternative_slot = main_slot;
        if let Some(tokens) = alternative.filter(|tokens| !tokens.is_empty()) {
            let next = self.info.request_slots - self.prefix_slots.len() as u32;
            alternative_slot = *self.prefix_slots.entry(call.request_key).or_insert(next);
            self.grow_tables(
                py,
                call.request_key,
                alternative_slot,
                tokens.len() as u32,
                true,
                batch,
            )?;
            add_row(batch, row, alternative_slot, 0, tokens.len() as u32, true);
        }
        for (tokens, conditioned) in prefixes {
            add_row(
                batch,
                row,
                if conditioned {
                    main_slot
                } else {
                    alternative_slot
                },
                if conditioned {
                    progress.kv_visible_len as u32
                } else {
                    tokens.len() as u32
                },
                query,
                false,
            );
        }
        Ok(())
    }
}

fn add_row(batch: &mut Batch, call: usize, slot: u32, prefix: u32, query: u32, write: bool) {
    batch
        .forward
        .push(call as u32, slot, prefix + query, query, write);
}

fn image_size(py: Python<'_>, height: u32, width: u32) -> PyResult<Bound<'_, PyAny>> {
    py.import("uniserve.media.image")?
        .getattr("Config")?
        .call1((height, width))
}

fn key(request: u64) -> RequestKey {
    RequestKey::new(0, RequestId(request), 1)
}
