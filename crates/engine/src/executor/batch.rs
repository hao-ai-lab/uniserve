//! Logical submissions and their worker-local protocol projection.
//!
//! The scheduler builds an [`ExecutionBatch`] of calls, each paired with the
//! [`RequestPlacement`] it chose, plus lifecycle commands, and hands it to an
//! `Executor`. `WorkerExecutor` splits it into one `ExecutionBatch` per target
//! worker and lowers each with `ExecutionBatch::into_protocol`, which gathers
//! the per-call placement fields into the flat, call-indexed arrays of the
//! wire `Batch`.

use super::WorkerId;
use std::collections::BTreeMap;
use uniserve_worker_ipc::{
    Batch, BatchCommand, BlockTable, BufferAllocation, CacheUnitAllocation, Call, DecodeRange,
    ForwardBatch, LatentParams, NewRequest, RequestKey, TensorPublication,
};

/// Physical placement selected by the scheduler for a computation.
///
/// The computation itself is the shared IPC `Call`. These fields describe
/// its worker and allocations, which are gathered into physical batch arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestPlacement {
    /// Worker routing stays outside the computation sent across IPC.
    pub worker: WorkerId,
    /// Worker-local request row when a replicated route owns its own address
    /// space. Unset placements retain the admission's canonical row;
    /// `WorkerExecutor` applies the override to the `Start` it sends the worker.
    pub request_pool_idx: Option<u32>,
    /// KV tables and newly acquired units used by this computation.
    pub block_tables: Vec<BlockTable>,
    pub new_cache_units: Vec<CacheUnitAllocation>,
    /// Rows use a local call index until gathered into the physical batch:
    /// `ExecutionBatch::validate` requires every row's index to be zero, and
    /// `into_protocol` offsets it by the call's position.
    pub forward: ForwardBatch,
    pub latent: Option<LatentParams>,
    pub decode: Option<DecodeRange>,
    /// Worker-local persistent spans for the call's buffer inputs and outputs.
    /// Request retirement retains every physical reader and writer allocation.
    pub buffers: Vec<BufferAllocation>,
    /// For a video request's call, the worker routed to each component that
    /// reads its products, keyed by component name, as the request's graph
    /// connects its calls; empty for a call whose products leave the graph.
    /// Admission fixes a video request's route, so a product is published
    /// only toward the replicas that read it. `None` for calls outside a
    /// video graph, whose readers follow from their call kind
    /// (`generation_consuming_calls`) or their transfer edges.
    pub readers: Option<BTreeMap<String, WorkerId>>,
}

/// One logical executor submission. Its rank projections are derived only inside an executor.
///
/// `WorkerExecutor` also uses this type for the part of a logical batch routed
/// to one worker; that part keeps the logical `id`, so completions from every
/// worker correlate to the same batch.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionBatch {
    /// Logical batch identity used to correlate partial completions.
    pub id: u64,
    /// Shared call kinds paired with physical placement in scheduler order.
    pub requests: Vec<(Call, RequestPlacement)>,
    /// Ordered lifecycle and resource commands.
    pub commands: Vec<BatchCommand>,
    /// Published transfer descriptors supplied by an external storage owner.
    pub input_transfers: Vec<TensorPublication>,
    /// External cache publications consumed by explicit KV installation.
    ///
    /// `ExecutionBatch::new` leaves this empty; `WorkerExecutor` fills it on
    /// the worker-local batches it builds.
    pub kv_inputs: Vec<uniserve_worker_ipc::KvTransfer>,
}

impl ExecutionBatch {
    /// Constructs a logical executor submission.
    pub fn new(
        id: u64,
        requests: Vec<(Call, RequestPlacement)>,
        commands: Vec<BatchCommand>,
        input_transfers: Vec<TensorPublication>,
    ) -> Self {
        Self {
            id,
            requests,
            commands,
            input_transfers,
            kv_inputs: Vec::new(),
        }
    }

    /// Iterates over request admissions carried by batch commands.
    pub fn admissions(&self) -> impl Iterator<Item = &NewRequest> {
        self.commands.iter().filter_map(|command| match command {
            BatchCommand::Start { request } => Some(request.as_ref()),
            _ => None,
        })
    }

    /// Removes unstarted work for terminated request epochs while preserving
    /// independent calls.
    ///
    /// Drops the calls and `Start` commands of every request key in
    /// `requests`, then the input transfers and KV inputs no remaining call
    /// reads. The requests' `Finish` and `Free` commands stay, since they still
    /// retire physical state and reader obligations on the workers. Returns the
    /// removed calls with the placements describing their resources. The batch
    /// may be left with neither calls nor commands; callers decide whether to
    /// discard it.
    pub(crate) fn retire_requests(
        &mut self,
        requests: &std::collections::HashSet<RequestKey>,
    ) -> Vec<(Call, RequestPlacement)> {
        let (retired, active): (Vec<_>, Vec<_>) = std::mem::take(&mut self.requests)
            .into_iter()
            .partition(|(call, _)| requests.contains(&call.request_key));
        self.requests = active;
        self.commands.retain_mut(|command| {
            if !requests.contains(&command.request_key()) {
                return true;
            }
            !matches!(command, BatchCommand::Start { .. })
        });
        let inputs = self
            .requests
            .iter()
            .flat_map(|(call, _)| call.tensor_inputs().chain(call.predicate.iter()))
            .collect::<std::collections::HashSet<_>>();
        self.input_transfers
            .retain(|payload| inputs.contains(&payload.product));
        self.kv_inputs.retain(|publication| {
            self.requests
                .iter()
                .any(|(call, _)| call.kv_input == Some(publication.source))
        });
        retired
    }

    /// Validates call identities, execution ownership, and command payloads.
    ///
    /// Requires at least one call or command; checks each call and its
    /// placement, that call identities are unique and belong to this batch,
    /// and that every `Start` is unique and names a request with a call in
    /// this batch; and validates every command, input transfer, and KV input.
    /// Wire-level constraints between calls (one kind and component per batch,
    /// allocation tables matching their calls, non-overlapping spans) are left
    /// to `Batch::validate`, which `into_protocol` runs.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.requests.is_empty() || !self.commands.is_empty(),
            "logical batch must carry at least one call or command"
        );

        let mut requests = std::collections::HashSet::with_capacity(self.requests.len());
        let mut identities = std::collections::HashSet::with_capacity(self.requests.len());
        for (call, placement) in &self.requests {
            call.validate()?;
            anyhow::ensure!(
                call.call_id.batch_id == self.id,
                "computation identity belongs to another logical batch"
            );
            requests.insert(call.request_key);
            // `WorkerId`'s public field admits any string; only its
            // constructor checks the identifier syntax.
            WorkerId::new(placement.worker.0.clone())?;
            anyhow::ensure!(!call.component.is_empty(), "call requires a component");
            anyhow::ensure!(
                identities.insert(call.call_id),
                "logical batch repeats a call identity"
            );

            // A placement describes one call, so its forward rows may only
            // name local call index zero.
            placement.forward.validate(1)?;
            for table in &placement.block_tables {
                table.validate()?;
            }
            for units in &placement.new_cache_units {
                units.validate()?;
            }
            if let Some(latent) = &placement.latent {
                latent.validate()?;
                anyhow::ensure!(
                    (latent.request_key, latent.call_id) == (call.request_key, call.call_id),
                    "logical call carries another call's latent execution"
                );
            }
            if let Some(decode) = &placement.decode {
                decode.validate()?;
                anyhow::ensure!(
                    (decode.request_key, decode.call_id) == (call.request_key, call.call_id),
                    "logical call carries another call's decode execution"
                );
            }
            for buffer in &placement.buffers {
                buffer.validate()?;
                anyhow::ensure!(
                    call.buffer_inputs()
                        .chain(call.buffer_outputs())
                        .any(|tensor| tensor.buffer_id() == buffer.buffer),
                    "logical call carries a buffer execution for an unrelated tensor"
                );
            }
        }

        let mut admitted = std::collections::HashSet::new();
        for admission in self.admissions() {
            admission.validate()?;
            anyhow::ensure!(
                admitted.insert(admission.request_key),
                "logical batch repeats a request start"
            );
            anyhow::ensure!(
                requests.contains(&admission.request_key),
                "logical batch starts a request without a call"
            );
        }

        for command in &self.commands {
            command.validate()?;
        }

        for transfer in &self.input_transfers {
            transfer.validate()?;
        }
        for publication in &self.kv_inputs {
            publication.validate()?;
        }
        Ok(())
    }
}

impl ExecutionBatch {
    /// Lowers a logical batch into a validated wire batch.
    ///
    /// Concatenates the placements' tables, pages, latent parameters, decode
    /// ranges, and buffer allocations in call order, and rebases each
    /// placement's forward rows onto the call's position in `calls`.
    /// `collective_seq` is the worker's next collective sequence, which
    /// `WorkerExecutor` commits only after the worker accepts the batch. Fails
    /// when adding the call's position to a forward row's call index overflows
    /// `u32` or `Batch::validate` rejects the result.
    pub(crate) fn into_protocol(self, collective_seq: u64) -> anyhow::Result<Batch> {
        let Self {
            id: batch_id,
            requests,
            commands,
            input_transfers: input_products,
            kv_inputs,
        } = self;
        let mut block_tables = Vec::new();
        let mut new_cache_units = Vec::new();
        let mut forward = ForwardBatch::default();
        let mut latent_params = Vec::new();
        let mut decode_ranges = Vec::new();
        let mut buffer_allocations = Vec::new();
        let mut calls = Vec::with_capacity(requests.len());

        for (call_index, (call, placement)) in requests.into_iter().enumerate() {
            block_tables.extend(placement.block_tables);
            new_cache_units.extend(placement.new_cache_units);
            forward.append(placement.forward, call_index as u32)?;
            latent_params.extend(placement.latent);
            decode_ranges.extend(placement.decode);
            buffer_allocations.extend(placement.buffers);
            calls.push(call);
        }

        let batch = Batch {
            batch_id,
            collective_seq,
            calls,
            block_tables,
            new_cache_units,
            forward,
            latent_params,
            decode_ranges,
            buffer_allocations,
            commands,
            input_products,
            kv_inputs,
        };
        batch.validate()?;
        Ok(batch)
    }
}
