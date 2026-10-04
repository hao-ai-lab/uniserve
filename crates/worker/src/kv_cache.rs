//! Physical KV intervals retained by model execution and asynchronous transfer.

use std::collections::{HashMap, HashSet};
use std::ops::{Deref, Range};

use uniserve_worker_ipc::{BufferId, RequestKey};

use crate::{Completion, Error, Result};

type Ranges = HashMap<u32, Range<u64>>;

struct Execution<C> {
    completion: C,
    requests: HashSet<RequestKey>,
    ranges: Ranges,
}

struct Export<C> {
    ranges: Ranges,
    retirements: Vec<C>,
    released: bool,
}

struct Import<C> {
    ranges: Ranges,
    retirement: C,
}

/// Track physical use of scheduler-assigned KV units.
///
/// A model stream orders its own accesses. Independent imports and immutable
/// exports prevent overlapping writes; recycling a unit also waits for model
/// accesses. Completion handles retain backend resources without inspecting
/// them. Only successful physical completion removes an access.
/// Handles must keep their completion at a stable address, as `Arc` does.
pub struct KVCacheManager<C> {
    executions: HashMap<usize, Execution<C>>,
    units: HashMap<u32, HashSet<usize>>,
    exports: HashMap<BufferId, Export<C>>,
    imports: HashMap<BufferId, Import<C>>,
}

impl<C> Default for KVCacheManager<C> {
    fn default() -> Self {
        Self {
            executions: HashMap::new(),
            units: HashMap::new(),
            exports: HashMap::new(),
            imports: HashMap::new(),
        }
    }
}

impl<E, N, C: Deref<Target = Completion<E, N>>> KVCacheManager<C> {
    pub fn has_pending_accesses(&mut self) -> bool {
        self.reap();
        !self.executions.is_empty() || self.has_transfers()
    }

    pub fn has_transfers(&self) -> bool {
        !self.exports.is_empty() || !self.imports.is_empty()
    }

    /// Coalesce the ranges of calls sharing one batch completion. The caller
    /// reports an already finished completion's error before retaining it.
    pub fn retain_execution(
        &mut self,
        request: RequestKey,
        spans: &[(u32, u32, u32)],
        completion: C,
    ) {
        if completion.succeeded() || spans.is_empty() {
            return;
        }

        let key = std::ptr::from_ref(&*completion) as usize;
        let execution = self.executions.entry(key).or_insert_with(|| Execution {
            completion,
            requests: HashSet::new(),
            ranges: HashMap::new(),
        });
        execution.requests.insert(request);
        for &(unit, offset, count) in spans {
            let start = u64::from(offset);
            let end = start + u64::from(count);
            execution
                .ranges
                .entry(unit)
                .and_modify(|range| {
                    range.start = range.start.min(start);
                    range.end = range.end.max(end);
                })
                .or_insert(start..end);
            self.units.entry(unit).or_default().insert(key);
        }
    }

    pub fn reserve_export(&mut self, buffer: BufferId, spans: &[(u32, u32, u32)]) -> Result<()> {
        self.reap();
        if spans.is_empty() || self.exports.contains_key(&buffer) {
            return Err(Error::Invalid(
                "KV export has an empty or already registered interval".into(),
            ));
        }

        self.exports.insert(
            buffer,
            Export {
                ranges: ranges(spans),
                retirements: Vec::new(),
                released: false,
            },
        );
        Ok(())
    }

    pub fn retain_export(&mut self, buffer: BufferId, retirement: C) -> Result<()> {
        let export = self
            .exports
            .get_mut(&buffer)
            .filter(|export| !export.released)
            .ok_or_else(|| Error::Invalid("KV export reservation is no longer active".into()))?;
        export.retirements.push(retirement);
        Ok(())
    }

    pub fn release_exports(&mut self, buffers: &[BufferId]) {
        for buffer in buffers {
            if let Some(export) = self.exports.get_mut(buffer) {
                export.released = true;
            }
        }
        self.reap();
    }

    pub fn exported_buffers(&self) -> Vec<BufferId> {
        self.exports.keys().copied().collect()
    }

    /// The import owner completes retirement after adoption or abandonment,
    /// once the copy stream and every transport reader have finished.
    pub fn reserve_import(
        &mut self,
        buffer: BufferId,
        spans: &[(u32, u32, u32)],
        retirement: C,
    ) -> Result<()> {
        self.require_reusable(spans)?;
        if self.imports.contains_key(&buffer) {
            return Err(Error::Invalid(
                "KV import destination is already reserved".into(),
            ));
        }
        self.imports.insert(
            buffer,
            Import {
                ranges: ranges(spans),
                retirement,
            },
        );
        Ok(())
    }

    /// Roll back a reservation whose copy task was never submitted.
    pub fn discard_import(&mut self, buffer: BufferId) {
        self.imports.remove(&buffer);
    }

    pub fn write_dependencies(&mut self, spans: &[(u32, u32, u32)]) -> Vec<&C> {
        self.reap();
        let mut dependencies: Vec<_> = self
            .execution_keys(spans)
            .into_iter()
            .map(|key| &self.executions[&key].completion)
            .collect();
        dependencies.extend(
            self.imports
                .values()
                .filter(|import| overlaps(&import.ranges, spans))
                .map(|import| &import.retirement),
        );
        dependencies.extend(
            self.exports
                .values()
                .filter(|export| overlaps(&export.ranges, spans))
                .flat_map(|export| &export.retirements),
        );
        dependencies
    }

    pub fn require_writable(&mut self, spans: &[(u32, u32, u32)]) -> Result<()> {
        self.reap();
        if self
            .exports
            .values()
            .any(|export| overlaps(&export.ranges, spans))
        {
            return Err(Error::Resource("KV interval still has a published version"));
        }
        if self
            .imports
            .values()
            .any(|import| overlaps(&import.ranges, spans))
        {
            return Err(Error::Resource(
                "KV interval still has an import destination",
            ));
        }
        Ok(())
    }

    pub fn require_reusable(&mut self, spans: &[(u32, u32, u32)]) -> Result<()> {
        self.require_writable(spans)?;
        if !self.execution_keys(spans).is_empty() {
            return Err(Error::Resource(
                "KV interval still has an executing producer or consumer",
            ));
        }
        Ok(())
    }

    /// Borrow the selected owners' signals for backend error reporting.
    /// Releasing a buffer does not retire other model accesses by its request.
    pub fn retirement_completions(
        &self,
        buffers: &HashSet<BufferId>,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
    ) -> Vec<&C> {
        let selected = |buffer: &BufferId| {
            buffers.contains(buffer)
                || (requests.contains(&buffer.owner) && !retained.contains(buffer))
        };
        self.exports
            .iter()
            .filter(|(buffer, _)| selected(buffer))
            .flat_map(|(_, export)| &export.retirements)
            .chain(
                self.imports
                    .iter()
                    .filter(|(buffer, _)| selected(buffer))
                    .map(|(_, import)| &import.retirement),
            )
            .chain(
                self.executions
                    .values()
                    .filter(|execution| !execution.requests.is_disjoint(requests))
                    .map(|execution| &execution.completion),
            )
            .collect()
    }

    pub fn retirement_ready(
        &mut self,
        buffers: &HashSet<BufferId>,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
    ) -> bool {
        self.reap();
        let selected = |buffer: &BufferId| {
            buffers.contains(buffer)
                || (requests.contains(&buffer.owner) && !retained.contains(buffer))
        };
        !self.exports.keys().chain(self.imports.keys()).any(selected)
            && !self
                .executions
                .values()
                .any(|execution| !execution.requests.is_disjoint(requests))
    }

    pub fn require_retired(&mut self) -> Result<()> {
        self.reap();
        if !self.executions.is_empty() {
            return Err(Error::Resource(
                "KV cache still has executing producers or consumers",
            ));
        }
        if !self.exports.is_empty() {
            return Err(Error::Resource(
                "KV cache still has unretired physical publications",
            ));
        }
        if !self.imports.is_empty() {
            return Err(Error::Resource("KV imports still own physical storage"));
        }
        Ok(())
    }

    /// Backend handles retained by the manager, for language-runtime tracing.
    pub fn completions(&self) -> impl Iterator<Item = &C> {
        self.executions
            .values()
            .map(|execution| &execution.completion)
            .chain(self.exports.values().flat_map(|export| &export.retirements))
            .chain(self.imports.values().map(|import| &import.retirement))
    }

    fn execution_keys(&self, spans: &[(u32, u32, u32)]) -> HashSet<usize> {
        spans
            .iter()
            .flat_map(|&(unit, offset, count)| {
                self.units
                    .get(&unit)
                    .into_iter()
                    .flatten()
                    .filter_map(move |&key| {
                        overlaps(&self.executions[&key].ranges, &[(unit, offset, count)])
                            .then_some(key)
                    })
            })
            .collect()
    }

    fn reap(&mut self) {
        self.executions.retain(|key, execution| {
            if !execution.completion.succeeded() {
                return true;
            }
            for unit in execution.ranges.keys() {
                if let Some(accesses) = self.units.get_mut(unit) {
                    accesses.remove(key);
                    if accesses.is_empty() {
                        self.units.remove(unit);
                    }
                }
            }
            false
        });
        self.exports.retain(|_, export| {
            !export.released
                || export
                    .retirements
                    .iter()
                    .any(|completion| !completion.succeeded())
        });
        self.imports
            .retain(|_, import| !import.retirement.succeeded());
    }
}

fn ranges(spans: &[(u32, u32, u32)]) -> Ranges {
    spans
        .iter()
        .map(|&(unit, offset, count)| {
            (
                unit,
                u64::from(offset)..u64::from(offset) + u64::from(count),
            )
        })
        .collect()
}

fn overlaps(ranges: &Ranges, spans: &[(u32, u32, u32)]) -> bool {
    spans.iter().any(|&(unit, offset, count)| {
        ranges.get(&unit).is_some_and(|range| {
            range.start.max(u64::from(offset)) < range.end.min(u64::from(offset) + u64::from(count))
        })
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use uniserve_core::{CallId, RequestId};

    use super::*;

    type Fence = Arc<Completion<String, ()>>;

    fn buffer(request: u64) -> BufferId {
        BufferId {
            owner: RequestKey::new(1, RequestId(request), 1),
            producer_call_id: CallId::new(1, 0),
            output_index: 0,
            generation: 1,
        }
    }

    #[test]
    fn model_accesses_share_a_batch_fence_without_blocking_independent_intervals() -> Result<()> {
        let mut cache = KVCacheManager::default();
        let fence = Fence::default();
        let first = buffer(1);
        let second = buffer(2);
        cache.retain_execution(first.owner, &[(1, 0, 4)], Arc::clone(&fence));
        cache.retain_execution(second.owner, &[(1, 4, 2), (2, 0, 8)], Arc::clone(&fence));

        cache.require_writable(&[(1, 0, 6)])?;
        assert!(cache.require_reusable(&[(1, 5, 1)]).is_err());
        cache.require_reusable(&[(1, 6, 2), (3, 0, 8)])?;
        assert!(cache.retirement_ready(&HashSet::from([first]), &HashSet::new(), &HashSet::new()));
        assert!(!cache.retirement_ready(
            &HashSet::new(),
            &HashSet::from([second.owner]),
            &HashSet::new()
        ));

        fence.complete(Ok(()))?;
        assert!(!cache.has_pending_accesses());
        cache.require_reusable(&[(1, 0, 8), (2, 0, 8)])?;
        cache.require_retired()
    }

    #[test]
    fn exported_prefixes_wait_for_all_readers_and_preserve_owner_scoped_failures() -> Result<()> {
        let mut cache = KVCacheManager::default();
        let first = buffer(1);
        let left = Fence::default();
        let right = Fence::default();
        cache.reserve_export(first, &[(1, 0, 3)])?;
        cache.retain_export(first, Arc::clone(&left))?;
        cache.retain_export(first, Arc::clone(&right))?;
        cache.require_writable(&[(1, 3, 5)])?;
        cache.require_writable(&[(1, 2, 0)])?;
        assert!(cache.require_writable(&[(1, 2, 1)]).is_err());

        cache.release_exports(&[first]);
        left.complete(Ok(()))?;
        assert!(cache.require_reusable(&[(1, 0, 8)]).is_err());
        right.complete(Err("reader completion unknown".into()))?;
        assert!(cache.require_reusable(&[(1, 0, 8)]).is_err());
        assert!(cache.retirement_ready(
            &HashSet::new(),
            &HashSet::from([buffer(2).owner]),
            &HashSet::new()
        ));
        let failures =
            cache.retirement_completions(&HashSet::from([first]), &HashSet::new(), &HashSet::new());
        assert!(failures.iter().any(|completion| matches!(completion.outcome(), Some(crate::Outcome::Failed(error)) if *error == "reader completion unknown")));
        assert!(cache.retirement_ready(
            &HashSet::new(),
            &HashSet::from([first.owner]),
            &HashSet::from([first])
        ));
        assert!(cache.require_retired().is_err());
        Ok(())
    }

    #[test]
    fn import_reservations_survive_rejection_and_end_only_at_physical_retirement() -> Result<()> {
        let mut cache = KVCacheManager::default();
        let first = buffer(1);
        let fence = Fence::default();
        cache.reserve_import(first, &[(2, 1, 4)], Arc::clone(&fence))?;
        assert!(
            cache
                .reserve_import(first, &[(3, 0, 8)], Fence::default())
                .is_err()
        );
        assert!(
            cache
                .reserve_import(buffer(2), &[(2, 4, 2)], Fence::default())
                .is_err()
        );
        cache.require_writable(&[(2, 0, 1), (2, 5, 3)])?;
        assert!(cache.require_writable(&[(2, 1, 4)]).is_err());

        cache.release_exports(&[first]);
        assert!(!cache.retirement_ready(&HashSet::from([first]), &HashSet::new(), &HashSet::new()));
        fence.complete(Ok(()))?;
        cache.require_reusable(&[(2, 0, 8)])?;
        cache.require_retired()
    }
}
