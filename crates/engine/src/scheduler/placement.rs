//! Request residency and capability-based worker placement.
//!
//! The scheduler asks `Placement` which loaded worker replica executes a call.
//! Candidates are the workers whose reported `supported_calls` and component
//! bindings cover the call; once a request is placed on a replica for a
//! component, its later calls for that component stay there.

use super::*;

#[derive(Default)]
pub(super) struct Placement {
    /// Replica that holds each request's state for one bound component name.
    ///
    /// Admission records the prefill worker of a token request and every route
    /// of a media request; `select_worker` records an entry each time it places
    /// a call, keyed by the name the worker binds (which may be
    /// `DEFAULT_COMPONENT` rather than the requested name).
    /// `Scheduler::refill_executor` prunes entries of requests that are no
    /// longer running or retiring.
    pub(super) affinity: HashMap<(RequestKey, String), crate::WorkerId>,
}

impl Placement {
    /// Enumerates loaded components that can execute the requested call.
    ///
    /// Which component serves a media call is the model's state, not the
    /// engine's: a worker resolves it from the capabilities its components
    /// implement and reports it as `media_components`. That report is the
    /// only authority here. Calls without an explicit media route use the
    /// protocol's default component for an undivided model.
    pub(super) fn worker_candidates<'a>(
        &'a self,
        executor: &'a dyn Executor,
        info: &'a WorkerInfo,
        kind: CallKind,
    ) -> impl Iterator<Item = (&'a crate::WorkerId, &'a str, &'a WorkerInfo)> {
        let component = match kind {
            CallKind::Media(media_call) => info
                .media_components
                .get(&media_call)
                .map_or(DEFAULT_COMPONENT, String::as_str),
            _ => DEFAULT_COMPONENT,
        };
        self.component_candidates(executor, kind, component)
    }

    /// Enumerates workers that support `kind` and can serve `component`, with
    /// the component name each one binds.
    ///
    /// A worker binding `component` by name serves it under that name. A
    /// worker with no component bindings, or with a `DEFAULT_COMPONENT`
    /// binding, serves any requested component under `DEFAULT_COMPONENT`.
    /// Other workers are excluded.
    pub(super) fn component_candidates<'a>(
        &'a self,
        executor: &'a dyn Executor,
        kind: CallKind,
        component: &'a str,
    ) -> impl Iterator<Item = (&'a crate::WorkerId, &'a str, &'a WorkerInfo)> {
        executor
            .info()
            .workers
            .iter()
            .filter_map(move |(id, info)| {
                if !info.supported_calls.contains(&kind) {
                    return None;
                }
                let bound_component = if info
                    .components
                    .iter()
                    .any(|binding| binding.name == component)
                {
                    component
                } else if info.components.is_empty()
                    || info
                        .components
                        .iter()
                        .any(|binding| binding.name == DEFAULT_COMPONENT)
                {
                    DEFAULT_COMPONENT
                } else {
                    return None;
                };
                Some((id, bound_component, info))
            })
    }

    /// Selects one replica while preserving request residency.
    ///
    /// Returns `None` when no worker can serve the component, when the replica
    /// that already holds this request's component state is no longer a
    /// candidate or has no submission capacity, when a replica already holding
    /// another of the request's components has no capacity, or when no other
    /// candidate has capacity. Records nothing; `select_worker` records the
    /// choice.
    pub(super) fn component_target<'a>(
        &'a self,
        executor: &'a dyn Executor,
        request: RequestKey,
        kind: CallKind,
        component: &'a str,
    ) -> Option<(&'a crate::WorkerId, &'a str)> {
        let candidates = self
            .component_candidates(executor, kind, component)
            .collect::<Vec<_>>();
        let bound = candidates.first()?.1;

        // Once a request has used this component, its mutable request state and
        // graph slot live on that exact replica. Queue pressure may delay the
        // next call but must never migrate it.
        if let Some(owner) = self.affinity.get(&(request, bound.to_owned())) {
            let (id, component, _) = candidates.into_iter().find(|(id, _, _)| *id == owner)?;
            return executor.has_capacity(id).then_some((id, component));
        }

        // Prefer a worker already selected for another component of this
        // request. This keeps a replicated denoiser and its media decoders in
        // one process and avoids publishing their large latent products.
        let resident = self
            .affinity
            .iter()
            .filter_map(|((key, _), worker)| (*key == request).then_some(worker))
            .collect::<HashSet<_>>();
        if let Some((id, component, _)) = candidates
            .iter()
            .copied()
            .find(|(id, _, _)| resident.contains(id))
        {
            return executor.has_capacity(id).then_some((id, component));
        }

        // Among replicas with queue capacity, a new request goes to the one
        // holding the fewest distinct resident requests, so a burst fans out
        // instead of filling the first configured worker. Queue capacity is a
        // hard condition; ties go to the earliest worker in
        // `ExecutorInfo::workers`.
        candidates
            .into_iter()
            .filter(|(id, _, _)| executor.has_capacity(id))
            .min_by_key(|(id, _, _)| {
                self.affinity
                    .iter()
                    .filter_map(|((key, _), worker)| (worker == *id).then_some(*key))
                    .collect::<HashSet<_>>()
                    .len()
            })
            .map(|(id, component, _)| (id, component))
    }

    /// Selects a replica for `kind` using the aggregate media routing in
    /// `info`, as `worker_candidates` does, while preserving residency.
    pub(super) fn worker_target<'a>(
        &'a self,
        executor: &'a dyn Executor,
        info: &'a WorkerInfo,
        request: RequestKey,
        kind: CallKind,
    ) -> Option<(&'a crate::WorkerId, &'a str)> {
        let component = match kind {
            CallKind::Media(media_call) => info
                .media_components
                .get(&media_call)
                .map_or("model", String::as_str),
            _ => "model",
        };
        self.component_target(executor, request, kind, component)
    }

    /// Binds planned work to its configured component and records request residency.
    ///
    /// Returns `None`, recording nothing, when `component_target` finds no
    /// eligible replica with submission capacity.
    pub(super) fn select_worker(
        &mut self,
        executor: &dyn Executor,
        info: &WorkerInfo,
        call: &Call,
    ) -> Option<(crate::WorkerId, String)> {
        let (id, bound_component) = if call.component == DEFAULT_COMPONENT {
            self.worker_target(executor, info, call.request_key, call.code)
        } else {
            self.component_target(executor, call.request_key, call.code, &call.component)
        }?;
        let target = (id.clone(), bound_component.to_owned());
        self.affinity
            .insert((call.request_key, target.1.clone()), target.0.clone());
        Some(target)
    }

    /// Includes lifecycle commands in destination ordering, even when a batch has no compute.
    ///
    /// Returns each call's placed worker plus every worker holding residency
    /// for a command's request. `Scheduler::dispatch_submissions` uses this
    /// set to hold back later batches that share a blocked destination.
    pub(super) fn batch_workers(&self, batch: &ExecutionBatch) -> HashSet<crate::WorkerId> {
        let mut targets = batch
            .requests
            .iter()
            .map(|(_, placement)| placement.worker.clone())
            .collect::<HashSet<_>>();
        for command in &batch.commands {
            targets.extend(self.affinity.iter().filter_map(|((request, _), worker)| {
                (*request == command.request_key()).then_some(worker.clone())
            }));
        }
        targets
    }
}
