//! Resident KV transfers, incremental transfer bases, and physical accesses.

use std::collections::{HashMap, HashSet};
use std::ops::{Deref, Range};
use std::sync::Arc;

use uniserve_core::CallId;
use uniserve_worker_ipc::{BufferId, KvTransfer, RequestKey};

use crate::{Completion, Error, Result};

type Ranges = HashMap<u32, Range<u64>>;
type Destinations = HashMap<(RequestKey, String), (BufferId, u32)>;

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

/// Own resident KV transfers and physical use of scheduler-assigned units.
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
    resident: HashMap<BufferId, Arc<KvTransfer>>,
    destination_bases: Destinations,
    installed_bases: Destinations,
}

impl<C> Default for KVCacheManager<C> {
    fn default() -> Self {
        Self {
            executions: HashMap::new(),
            units: HashMap::new(),
            exports: HashMap::new(),
            imports: HashMap::new(),
            resident: HashMap::new(),
            destination_bases: HashMap::new(),
            installed_bases: HashMap::new(),
        }
    }
}

impl<C> KVCacheManager<C> {
    pub fn resident(&self, buffer: BufferId) -> Option<&Arc<KvTransfer>> {
        self.resident.get(&buffer)
    }

    /// The latest extent sent to this destination, retained even after its
    /// descriptor is released. The receiver already owns that prefix.
    pub fn destination_base(
        &self,
        request: RequestKey,
        destination: &str,
    ) -> Option<(BufferId, u32)> {
        self.destination_bases
            .get(&(request, destination.to_owned()))
            .copied()
    }

    /// Check the destination's accepted prefix before copying an incremental
    /// transfer into its assigned physical pages.
    pub fn validate_install(&self, transfer: &KvTransfer) -> Result<()> {
        let installed = self
            .installed_bases
            .get(&(transfer.source.owner, transfer.destination.clone()))
            .copied();
        validate_base(transfer, installed)
    }

    /// Check all touched versions before any resource owner commits. Updates
    /// to one destination must form a chain in batch order; rejection leaves
    /// both resident descriptors and accepted bases unchanged.
    pub fn validate_exports(
        &self,
        exports: &[(BufferId, KvTransfer)],
        installations: &[(BufferId, BufferId, KvTransfer)],
    ) -> Result<()> {
        let mut resident = HashMap::new();
        let mut destination_bases = Destinations::new();
        let mut installed_bases = Destinations::new();

        for (buffer, transfer) in exports {
            if *buffer != transfer.source {
                return Err(Error::Invalid(
                    "KV export buffer differs from its source".into(),
                ));
            }
            let existing = resident
                .get(buffer)
                .copied()
                .or_else(|| self.resident(*buffer).map(AsRef::as_ref));
            if existing.is_some_and(|existing| existing != transfer) {
                return Err(Error::Invalid(
                    "KV export conflicts with its resident buffer".into(),
                ));
            }

            stage_base(transfer, &self.destination_bases, &mut destination_bases)?;
            resident.insert(*buffer, transfer);
        }

        for (source, installed, transfer) in installations {
            if *source != transfer.source || installed.owner != source.owner {
                return Err(Error::Invalid(
                    "installed KV buffer does not belong to its source".into(),
                ));
            }

            stage_base(transfer, &self.installed_bases, &mut installed_bases)?;
        }
        Ok(())
    }

    /// Commit an already checked batch. The executor must leave this
    /// directory unchanged between validation and the shared resource commit.
    pub fn apply_exports(
        &mut self,
        exports: Vec<(BufferId, KvTransfer)>,
        installations: Vec<(BufferId, BufferId, KvTransfer)>,
    ) {
        for (buffer, transfer) in exports {
            self.destination_bases.insert(
                (buffer.owner, transfer.destination.clone()),
                (transfer.source, transfer.exported_extent),
            );
            self.resident.insert(buffer, Arc::new(transfer));
        }

        for (source, installed, transfer) in installations {
            self.installed_bases.insert(
                (installed.owner, transfer.destination.clone()),
                (transfer.source, transfer.exported_extent),
            );
            let transfer = Arc::new(transfer);
            self.resident.insert(source, Arc::clone(&transfer));
            self.resident.insert(installed, transfer);
        }
    }

    /// Remove consumed descriptors and return their buffers for revocation.
    /// Physical exports and transfer bases have independent lifetimes.
    pub fn release_calls(&mut self, calls: &[(RequestKey, CallId)]) -> Vec<BufferId> {
        let calls: HashSet<_> = calls.iter().copied().collect();
        let mut released = Vec::new();
        self.resident.retain(|buffer, _| {
            if calls.contains(&(buffer.owner, buffer.producer_call_id)) {
                released.push(*buffer);
                false
            } else {
                true
            }
        });
        released
    }

    /// Forget a retired request after its physical accesses have drained.
    pub fn drop_request(&mut self, request_id: u64) {
        self.resident
            .retain(|buffer, _| buffer.owner.request_id.0 != request_id);
        self.destination_bases
            .retain(|(request, _), _| request.request_id.0 != request_id);
        self.installed_bases
            .retain(|(request, _), _| request.request_id.0 != request_id);
    }

    /// Forget transfer descriptions and bases when the cache owner closes.
    /// Physical access retirement is independent and must already be drained.
    pub fn clear_resident(&mut self) {
        self.resident.clear();
        self.destination_bases.clear();
        self.installed_bases.clear();
    }
}

fn validate_base(transfer: &KvTransfer, current: Option<(BufferId, u32)>) -> Result<()> {
    let current = current.map_or((None, 0), |(buffer, extent)| (Some(buffer), extent));
    if current != (transfer.base, transfer.base_extent) {
        return Err(Error::Invalid(
            "KV transfer base does not match destination".into(),
        ));
    }
    Ok(())
}

fn stage_base(
    transfer: &KvTransfer,
    resident: &Destinations,
    staged: &mut Destinations,
) -> Result<()> {
    let key = (transfer.source.owner, transfer.destination.clone());
    validate_base(
        transfer,
        staged.get(&key).or_else(|| resident.get(&key)).copied(),
    )?;
    staged.insert(key, (transfer.source, transfer.exported_extent));
    Ok(())
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
                "KV cache still has unretired physical exports",
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

    fn empty_transfer(source: BufferId, base: Option<BufferId>) -> KvTransfer {
        KvTransfer {
            groups: Vec::new(),
            source,
            destination: "decoder".into(),
            base,
            base_extent: 0,
            exported_extent: 0,
            compute_dtype: "bfloat16".into(),
        }
    }

    #[test]
    fn rejected_transfer_chains_leave_the_accepted_base_and_resident_values_unchanged() -> Result<()>
    {
        let mut cache = KVCacheManager::<()>::default();
        let first = empty_transfer(buffer(1), None);
        let second = empty_transfer(
            BufferId {
                producer_call_id: CallId::new(2, 0),
                ..first.source
            },
            Some(first.source),
        );
        let mut third = empty_transfer(
            BufferId {
                producer_call_id: CallId::new(3, 0),
                ..first.source
            },
            Some(first.source),
        );
        cache.validate_exports(&[(first.source, first.clone())], &[])?;
        cache.apply_exports(vec![(first.source, first.clone())], vec![]);

        // Even an empty suffix advances the accepted transfer base. The
        // third update cannot skip the second update in the same batch.
        assert!(
            cache
                .validate_exports(
                    &[
                        (second.source, second.clone()),
                        (third.source, third.clone())
                    ],
                    &[],
                )
                .is_err()
        );
        assert_eq!(
            cache.destination_base(first.source.owner, "decoder"),
            Some((first.source, 0))
        );
        assert_eq!(
            cache.resident(first.source).map(AsRef::as_ref),
            Some(&first)
        );
        assert_eq!(cache.resident(second.source), None);

        third.base = Some(second.source);
        let batch = vec![
            (second.source, second.clone()),
            (third.source, third.clone()),
        ];
        cache.validate_exports(&batch, &[])?;
        cache.apply_exports(batch, vec![]);
        cache.release_calls(&[
            (first.source.owner, first.source.producer_call_id),
            (second.source.owner, second.source.producer_call_id),
        ]);
        assert_eq!(cache.resident(first.source), None);
        assert_eq!(cache.resident(second.source), None);
        assert_eq!(
            cache.resident(third.source).map(AsRef::as_ref),
            Some(&third)
        );
        assert_eq!(
            cache.destination_base(first.source.owner, "decoder"),
            Some((third.source, 0))
        );
        Ok(())
    }

    #[test]
    fn installed_aliases_and_transfer_bases_retire_with_their_request() -> Result<()> {
        let mut cache = KVCacheManager::<()>::default();
        let first = empty_transfer(buffer(1), None);
        let local = BufferId {
            producer_call_id: CallId::new(5, 0),
            ..first.source
        };
        let other = empty_transfer(buffer(2), None);
        let installs = vec![
            (first.source, local, first.clone()),
            (other.source, other.source, other.clone()),
        ];
        cache.validate_install(&first)?;
        cache.validate_exports(&[], &installs)?;
        cache.apply_exports(vec![], installs);
        cache.release_calls(&[(first.source.owner, first.source.producer_call_id)]);
        assert_eq!(cache.resident(first.source), None);
        assert_eq!(cache.resident(local).map(AsRef::as_ref), Some(&first));

        let next = empty_transfer(
            BufferId {
                producer_call_id: CallId::new(2, 0),
                ..first.source
            },
            Some(first.source),
        );
        cache.validate_install(&next)?;
        assert!(cache.validate_install(&first).is_err());
        cache.drop_request(first.source.owner.request_id.0);
        cache.validate_install(&first)?;
        assert!(cache.validate_install(&next).is_err());
        assert_eq!(cache.resident(local), None);
        assert_eq!(
            cache.resident(other.source).map(AsRef::as_ref),
            Some(&other)
        );
        assert!(cache.validate_install(&other).is_err());
        Ok(())
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
