//! Request residency and capability-based worker placement.

use super::*;

#[derive(Default)]
pub(super) struct Placement {
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
        // request. This keeps a replicated H3 denoiser and its media decoders
        // in one process and avoids publishing their large latent products.
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

        // A new request goes to the least-resident ready replica. Queue
        // capacity remains a hard admission condition; residency breaks ties
        // so a burst fans out instead of filling the first configured worker.
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
    pub(super) fn select_worker(
        &mut self,
        executor: &dyn Executor,
        info: &WorkerInfo,
        call: &Call,
    ) -> (crate::WorkerId, String) {
        let (id, bound_component) = if call.component == DEFAULT_COMPONENT {
            self.worker_target(executor, info, call.request_key, call.code)
        } else {
            self.component_target(executor, call.request_key, call.code, &call.component)
        }
        .expect("planned call retains an executable component");
        let target = (id.clone(), bound_component.to_owned());
        self.affinity
            .insert((call.request_key, target.1.clone()), target.0.clone());
        target
    }

    /// Includes lifecycle commands in destination ordering, even when a batch has no compute.
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
