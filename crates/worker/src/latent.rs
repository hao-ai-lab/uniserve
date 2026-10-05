//! Paged diffusion trajectories and the physical lifetime of their two banks.

use std::collections::{HashMap, HashSet};
use std::ops::Deref;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use uniserve_worker_ipc::{BufferId, Call, CallKind, LatentParams, MediaCall, RequestKey};

use crate::{Completion, Error, Result};

#[derive(Default)]
struct LatentSlot {
    pages: Vec<usize>,
    bank: u8,
    step: i64,
    generation: i64,
    units: usize,
    height: i64,
    width: i64,
}

/// A batch's prepared trajectory commit or release. The executor validates
/// all updates before making any result visible, then applies the same values.
#[derive(Clone)]
pub struct LatentUpdate {
    pub request_pool_idx: i64,
    pub params: LatentParams,
    pub expected_generation: i64,
    pub expected_step: i64,
    pub generation: i64,
    pub step: i64,
    pub release: bool,
}

impl LatentUpdate {
    /// Derive a completed numerical call's trajectory change from its plan.
    /// Standalone media keeps one slot-local trajectory, versioned by step;
    /// transferred image trajectories use the call's explicit generations.
    pub fn for_call(slot: usize, call: &Call, params: &LatentParams) -> Result<Self> {
        let start = i64::from(params.start_step);
        let end = start + i64::from(params.step_count);
        let (expected_generation, expected_step, generation, step, release) = match call.code {
            CallKind::Media(MediaCall::LatentPreparation) => (
                0,
                0,
                call.latent_output
                    .as_ref()
                    .map_or(1, |output| i64::from(output.generation)),
                0,
                false,
            ),
            CallKind::Media(MediaCall::Denoising) => (
                call.latent_input
                    .as_ref()
                    .map_or(start + 1, |input| i64::from(input.generation)),
                start,
                call.latent_output
                    .as_ref()
                    .map_or(end + 1, |output| i64::from(output.generation)),
                end,
                false,
            ),
            CallKind::Media(MediaCall::ImageDecoding) => {
                let input = call.latent_input.as_ref().ok_or_else(|| {
                    Error::Invalid("image decoding has no latent trajectory".into())
                })?;
                (0, 0, i64::from(input.generation), start, true)
            }
            _ => {
                return Err(Error::Invalid(
                    "call does not update a latent trajectory".into(),
                ));
            }
        };

        Ok(Self {
            request_pool_idx: slot as i64,
            params: params.clone(),
            expected_generation,
            expected_step,
            generation,
            step,
            release,
        })
    }
}

struct ImportState<T> {
    transfers: Vec<Arc<T>>,
    adopted: bool,
    released: bool,
}

/// Destination pages retained until adoption or abandonment and physical
/// completion. Imported values occupy bank zero. T retains a transfer and
/// dereferences its native physical completion, independently of its result.
pub struct LatentImport<T> {
    pub buffer: BufferId,
    pub request_pool_idx: usize,
    pub pages: Vec<usize>,
    pub units: usize,
    state: Mutex<ImportState<T>>,
}

impl<T> LatentImport<T> {
    pub fn adopted(&self) -> bool {
        self.state().adopted
    }

    pub fn released(&self) -> bool {
        self.state().released
    }

    /// Retain handles outside the import lock before invoking transport work.
    pub fn transfers(&self) -> Vec<Arc<T>> {
        self.state().transfers.clone()
    }

    fn state(&self) -> MutexGuard<'_, ImportState<T>> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

struct ExportState<C> {
    retirements: Vec<Arc<C>>,
    released: bool,
}

/// An immutable page-bank version retained by transport readers, including
/// versions exported before the first trajectory commit.
pub struct LatentExport<C> {
    pub buffer: BufferId,
    pub request_pool_idx: usize,
    pub bank: u8,
    pub pages: Vec<usize>,
    state: Mutex<ExportState<C>>,
}

impl<C> LatentExport<C> {
    pub fn retirements(&self) -> Vec<Arc<C>> {
        self.state().retirements.clone()
    }

    fn state(&self) -> MutexGuard<'_, ExportState<C>> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Own scheduler-assigned pages, committed trajectories and transfer access.
/// I and S retain import/export objects at stable addresses, as Arc does. The
/// numerical backend owns tensor allocation, views, copies and stream waits.
pub struct LatentPool<I, S> {
    request_pool_size: usize,
    num_pages: usize,
    page_units: usize,
    slots: Vec<LatentSlot>,
    owners: Vec<usize>,
    imports: HashMap<usize, I>,
    exports: HashMap<BufferId, S>,
    retiring: HashSet<usize>,
}

impl<I, S> LatentPool<I, S> {
    pub fn new(request_pool_size: usize, num_pages: usize, page_units: usize) -> Result<Self> {
        if request_pool_size == 0 || num_pages < 2 || page_units == 0 {
            return Err(Error::Invalid(
                "latent-pool shape must contain slots, pages, and elements".into(),
            ));
        }

        Ok(Self {
            request_pool_size,
            num_pages,
            page_units,
            slots: (0..=request_pool_size)
                .map(|_| LatentSlot::default())
                .collect(),
            owners: vec![0; num_pages],
            imports: HashMap::new(),
            exports: HashMap::new(),
            retiring: HashSet::new(),
        })
    }

    pub fn slot(&self, slot: i64) -> Result<usize> {
        if slot < 1 || slot as usize > self.request_pool_size {
            return Err(Error::Invalid(
                "latent request slot is outside physical capacity".into(),
            ));
        }
        Ok(slot as usize)
    }

    pub fn request_pool_size(&self) -> usize {
        self.request_pool_size
    }

    pub fn num_pages(&self) -> usize {
        self.num_pages
    }

    pub fn page_units(&self) -> usize {
        self.page_units
    }

    pub fn capacity_units(&self) -> usize {
        (self.num_pages - 1) * self.page_units
    }

    pub fn validate_pages(&self, pages: &[i64], units: i64) -> Result<Vec<usize>> {
        if units < 1
            || pages.len() != (units as usize).div_ceil(self.page_units)
            || pages
                .iter()
                .any(|&page| page < 1 || page as usize >= self.num_pages)
            || pages.iter().collect::<HashSet<_>>().len() != pages.len()
        {
            return Err(Error::Invalid(
                "latent page table is outside physical pool bounds".into(),
            ));
        }
        Ok(pages.iter().map(|&page| page as usize).collect())
    }

    pub fn idle(&self) -> bool {
        self.slots.iter().all(|slot| slot.pages.is_empty())
            && self.imports.is_empty()
            && self.exports.is_empty()
    }

    /// Return page tables and their contiguous scratch offset. Occupied views
    /// supply their logical pages and offset in the shared page-index buffer;
    /// callers retain them through numerical completion.
    pub fn bind(
        &self,
        page_tables: &[Vec<i64>],
        latent_units: &[i64],
        occupied: &[(Vec<usize>, usize)],
    ) -> Result<(Vec<Vec<usize>>, usize)> {
        if page_tables.is_empty() || page_tables.len() != latent_units.len() {
            return Err(Error::Invalid(
                "latent buffer columns are not aligned".into(),
            ));
        }
        let tables = page_tables
            .iter()
            .zip(latent_units)
            .map(|(pages, &units)| self.validate_pages(pages, units))
            .collect::<Result<Vec<_>>>()?;
        let total: usize = tables.iter().map(Vec::len).sum();
        if total >= self.num_pages {
            return Err(Error::Invalid(
                "latent buffer exceeds the fixed step buffer".into(),
            ));
        }

        let mut pages = HashSet::new();
        for &page in tables.iter().flatten() {
            if !pages.insert(page) {
                return Err(Error::Invalid("latent buffer page tables overlap".into()));
            }
        }
        let mut ranges = Vec::with_capacity(occupied.len());
        for (held, start) in occupied {
            if held.iter().any(|page| pages.contains(page)) {
                return Err(Error::Invalid(
                    "latent buffer page tables overlap live calls".into(),
                ));
            }
            ranges.push((*start, held.len()));
        }

        ranges.sort_unstable();
        let mut offset = 0;
        for (start, count) in ranges {
            if offset + total <= start {
                break;
            }
            offset = offset.max(start + count);
        }
        if offset + total >= self.num_pages {
            return Err(Error::Resource(
                "live latent buffer exceeds the fixed step buffer",
            ));
        }
        Ok((tables, offset))
    }

    pub fn imports(&self) -> impl Iterator<Item = &I> {
        self.imports.values()
    }

    pub fn exports(&self) -> impl Iterator<Item = &S> {
        self.exports.values()
    }

    pub fn require_retired(&self) -> Result<()> {
        if !self.imports.is_empty() || !self.exports.is_empty() {
            return Err(Error::Resource(
                "latent physical reads must retire before pool shutdown",
            ));
        }
        Ok(())
    }
}

impl<T, C, I, S> LatentPool<I, S>
where
    I: Deref<Target = LatentImport<T>>,
    S: Deref<Target = LatentExport<C>>,
{
    pub fn require_initial(&self, slot: usize, pages: &[usize]) -> Result<()> {
        self.require_empty(slot)?;
        self.require_owners(pages, 0, None)?;
        self.require_writable(1, pages)
    }

    /// Check the trajectory addressed by a call and return its committed bank.
    #[allow(clippy::too_many_arguments)]
    pub fn current_bank(
        &self,
        slot: usize,
        step: i64,
        generation: i64,
        units: i64,
        height: i64,
        width: i64,
        pages: &[usize],
    ) -> Result<u8> {
        let current = &self.slots[slot];
        if current.step != step
            || current.generation != generation
            || current.units as i64 != units
            || current.height != height
            || current.width != width
        {
            return Err(Error::Invalid(
                "latent allocation does not name the committed trajectory".into(),
            ));
        }
        if current.pages != pages {
            return Err(Error::Invalid(
                "latent page table does not match its committed trajectory".into(),
            ));
        }

        self.require_owners(pages, slot, None)?;
        Ok(current.bank)
    }

    pub fn next_bank(&self, slot: usize) -> u8 {
        1 - self.slots[slot].bank
    }

    pub fn require_writable(&self, bank: u8, pages: &[usize]) -> Result<()> {
        if self.exports.values().any(|source| {
            source.bank == bank && source.pages.iter().any(|page| pages.contains(page))
        }) {
            return Err(Error::Invalid(
                "latent bank is retained by a published version".into(),
            ));
        }
        Ok(())
    }

    /// Prepare an export before the backend constructs its numerical views.
    /// Register it before allowing another owner operation on this pool.
    pub fn prepare_export(
        &self,
        buffer: BufferId,
        slot: usize,
        bank: u8,
        pages: Vec<usize>,
    ) -> Result<LatentExport<C>> {
        if self.exports.contains_key(&buffer) {
            return Err(Error::Invalid(
                "latent export generation is already registered".into(),
            ));
        }

        Ok(LatentExport {
            buffer,
            request_pool_idx: slot,
            bank,
            pages,
            state: Mutex::new(ExportState {
                retirements: Vec::new(),
                released: false,
            }),
        })
    }

    /// Install the prepared object after numerical view creation succeeds.
    pub fn register_export(&mut self, source: S) {
        self.exports.insert(source.buffer, source);
    }

    pub fn retain_export(&self, source: &LatentExport<C>, retirement: C) -> Result<()> {
        let mut state = source.state();
        if state.released
            || self
                .exports
                .get(&source.buffer)
                .is_none_or(|current| !std::ptr::eq(&**current, source))
        {
            return Err(Error::Invalid(
                "latent export reservation is no longer active".into(),
            ));
        }
        state.retirements.push(Arc::new(retirement));
        Ok(())
    }

    pub fn release_exports(&self, buffers: &[BufferId]) {
        for buffer in buffers {
            if let Some(source) = self.exports.get(buffer) {
                source.state().released = true;
            }
        }
    }

    pub fn write_dependencies(&self, slot: usize, pages: &[usize]) -> Vec<Arc<C>> {
        let bank = self.next_bank(slot);
        self.exports
            .values()
            .filter(|source| {
                source.bank == bank && source.pages.iter().any(|page| pages.contains(page))
            })
            .flat_map(|source| source.retirements())
            .collect()
    }

    /// Prepare bank-zero ownership before any numerical write. The owner may
    /// build views and initialize padding, then installs the returned object
    /// without changing pool assignments between these operations.
    pub fn prepare_import(
        &self,
        buffer: BufferId,
        slot: usize,
        pages: Vec<usize>,
        units: usize,
    ) -> Result<LatentImport<T>> {
        self.require_empty(slot)?;
        self.require_owners(&pages, 0, None)?;

        Ok(LatentImport {
            buffer,
            request_pool_idx: slot,
            pages,
            units,
            state: Mutex::new(ImportState {
                transfers: Vec::new(),
                adopted: false,
                released: false,
            }),
        })
    }

    /// Install the prepared import after destination views are constructed.
    pub fn register_import(&mut self, write: I) {
        let slot = write.request_pool_idx;
        for &page in &write.pages {
            self.owners[page] = slot;
        }
        self.slots[slot].pages.clone_from(&write.pages);
        self.imports.insert(slot, write);
    }

    pub fn retain_transfer(&self, write: &LatentImport<T>, transfer: T) -> Result<()> {
        self.require_import(write)?;
        let mut state = write.state();
        if state.adopted {
            return Err(Error::Invalid(
                "resident latent import cannot accept another read".into(),
            ));
        }
        state.transfers.push(Arc::new(transfer));
        Ok(())
    }

    pub fn validate_adoption(
        &self,
        write: &LatentImport<T>,
        generation: i64,
        step: i64,
        height: i64,
        width: i64,
    ) -> Result<()> {
        self.require_import(write)?;
        let state = write.state();
        if state.adopted || state.transfers.is_empty() {
            return Err(Error::Invalid("latent import cannot be adopted".into()));
        }
        self.validate_metadata(generation, step, write.units as i64, height, width)?;
        if generation != i64::from(write.buffer.generation) {
            return Err(Error::Invalid(
                "latent import generation disagrees with its product".into(),
            ));
        }
        Ok(())
    }

    /// Apply a validated adoption after every transfer result has ordered its
    /// producer fence on the consuming stream. Readiness alone is insufficient.
    pub fn adopt_import(
        &mut self,
        write: &LatentImport<T>,
        generation: i64,
        step: i64,
        height: i64,
        width: i64,
    ) {
        let current = &mut self.slots[write.request_pool_idx];
        current.bank = 0;
        current.step = step;
        current.generation = generation;
        current.units = write.units;
        current.height = height;
        current.width = width;
        write.state().adopted = true;
    }

    /// Revoke the import and return transfers to cancel outside native locks.
    pub fn abandon_import(&self, write: &LatentImport<T>) -> Result<Vec<Arc<T>>> {
        if write.released() {
            return Ok(Vec::new());
        }
        self.require_import(write)?;
        let mut state = write.state();
        if state.adopted {
            return Err(Error::Invalid(
                "resident latent import cannot be abandoned".into(),
            ));
        }

        state.released = true;
        Ok(state.transfers.clone())
    }

    pub fn cancel_imports(&self, requests: &HashSet<RequestKey>) -> Vec<Arc<T>> {
        let mut transfers = Vec::new();
        for write in self.imports.values() {
            if !requests.contains(&write.buffer.owner) {
                continue;
            }
            let mut state = write.state();
            if !state.adopted && !state.released {
                state.released = true;
                transfers.extend(state.transfers.iter().cloned());
            }
        }
        transfers
    }

    /// Validate all changes without partially committing a rejected batch.
    pub fn validate_updates(&self, updates: &[LatentUpdate]) -> Result<()> {
        let mut slots = HashSet::new();
        let mut claimed = HashSet::new();
        for update in updates {
            let params = &update.params;
            let slot = self.slot(update.request_pool_idx)?;
            if !slots.insert(slot) {
                return Err(Error::Invalid(
                    "latent commit repeats a request slot".into(),
                ));
            }
            let units = i64::from(params.latent_units);
            let height = i64::from(params.height);
            let width = i64::from(params.width);
            let pages = self.validate_pages(
                &params
                    .page_table
                    .iter()
                    .map(|&page| i64::from(page))
                    .collect::<Vec<_>>(),
                units,
            )?;

            if update.release {
                self.current_bank(
                    slot,
                    update.step,
                    update.generation,
                    units,
                    height,
                    width,
                    &pages,
                )?;
                continue;
            }

            self.validate_metadata(update.generation, update.step, units, height, width)?;
            if update.expected_generation == 0 {
                if update.expected_step != 0 || update.step != 0 {
                    return Err(Error::Invalid(
                        "latent initialization must publish step zero".into(),
                    ));
                }
                self.require_empty(slot)?;
                // A provisional export from this slot retains the bank, but
                // does not conflict with its own initial trajectory commit.
                self.require_owners(&pages, 0, Some(slot))?;
            } else {
                self.current_bank(
                    slot,
                    update.expected_step,
                    update.expected_generation,
                    units,
                    height,
                    width,
                    &pages,
                )?;
                if update.step <= update.expected_step {
                    return Err(Error::Invalid(
                        "latent successor does not advance its step".into(),
                    ));
                }
            }

            if update.generation <= update.expected_generation {
                return Err(Error::Invalid(
                    "latent commit does not advance its generation".into(),
                ));
            }
            for page in pages {
                if !claimed.insert(page) {
                    return Err(Error::Invalid(
                        "latent commits overlap physical pages".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Apply the previously validated updates. The executor must not mutate
    /// assignments or updates between validation and application. Returned
    /// transfers belong to released slots and must be cancelled by their owner.
    pub fn apply_updates(&mut self, updates: &[LatentUpdate]) -> Vec<Arc<T>> {
        let mut transfers = Vec::new();
        for update in updates {
            let params = &update.params;
            let slot = update.request_pool_idx as usize;
            if update.release {
                transfers.extend(self.clear_slot(slot));
                continue;
            }

            let current = &mut self.slots[slot];
            if update.expected_generation == 0 {
                current.pages = params
                    .page_table
                    .iter()
                    .map(|&page| page as usize)
                    .collect();
                for &page in &current.pages {
                    self.owners[page] = slot;
                }
            }
            current.bank = 1 - current.bank;
            current.generation = update.generation;
            current.step = update.step;
            current.units = params.latent_units as usize;
            current.height = i64::from(params.height);
            current.width = i64::from(params.width);
        }
        transfers
    }

    pub fn release_slots(&mut self, slots: &[i64]) -> Result<Vec<Arc<T>>> {
        let slots = slots
            .iter()
            .map(|&slot| self.slot(slot))
            .collect::<Result<Vec<_>>>()?;
        if slots.iter().copied().collect::<HashSet<_>>().len() != slots.len() {
            return Err(Error::Invalid(
                "latent release repeats a request slot".into(),
            ));
        }
        Ok(slots
            .into_iter()
            .flat_map(|slot| self.clear_slot(slot))
            .collect())
    }

    pub fn retirement_ready(&self, requests: &HashSet<RequestKey>) -> bool {
        !self
            .imports
            .values()
            .any(|write| requests.contains(&write.buffer.owner))
            && !self
                .exports
                .values()
                .any(|source| requests.contains(&source.buffer.owner))
    }

    fn require_empty(&self, slot: usize) -> Result<()> {
        let current = &self.slots[slot];
        if self.imports.contains_key(&slot)
            || current.generation != 0
            || current.units != 0
            || self.retiring.contains(&slot)
        {
            return Err(Error::Invalid(
                "request slot already owns a committed trajectory".into(),
            ));
        }
        Ok(())
    }

    fn require_owners(
        &self,
        pages: &[usize],
        owner: usize,
        export_slot: Option<usize>,
    ) -> Result<()> {
        if owner == 0
            && self.exports.values().any(|source| {
                Some(source.request_pool_idx) != export_slot
                    && source.pages.iter().any(|page| pages.contains(page))
            })
        {
            return Err(Error::Invalid(
                "latent pages are owned by a published version".into(),
            ));
        }
        if pages.iter().any(|&page| self.owners[page] != owner) {
            return Err(Error::Invalid(
                "latent page table is not owned by its request slot".into(),
            ));
        }
        Ok(())
    }

    fn validate_metadata(
        &self,
        generation: i64,
        step: i64,
        units: i64,
        height: i64,
        width: i64,
    ) -> Result<()> {
        if generation < 1 || step < 0 || units < 1 || height < 1 || width < 1 {
            return Err(Error::Invalid(
                "latent trajectory metadata is invalid".into(),
            ));
        }
        if units as usize > self.capacity_units() {
            return Err(Error::Invalid(
                "latent trajectory exceeds physical capacity".into(),
            ));
        }
        Ok(())
    }

    fn require_import(&self, write: &LatentImport<T>) -> Result<()> {
        if write.released()
            || self
                .imports
                .get(&write.request_pool_idx)
                .is_none_or(|current| !std::ptr::eq(&**current, write))
        {
            return Err(Error::Invalid(
                "latent import reservation is no longer writable".into(),
            ));
        }
        Ok(())
    }

    fn clear_slot(&mut self, slot: usize) -> Vec<Arc<T>> {
        self.retiring.insert(slot);
        if let Some(write) = self.imports.get(&slot) {
            let mut state = write.state();
            state.released = true;
            if !state.adopted {
                return state.transfers.clone();
            }
        }
        Vec::new()
    }
}

impl<E, N, F, M, T, C, I, S> LatentPool<I, S>
where
    T: Deref<Target = Completion<E, N>>,
    C: Deref<Target = Completion<F, M>>,
    I: Deref<Target = LatentImport<T>>,
    S: Deref<Target = LatentExport<C>>,
{
    /// Failed or cancelled physical completions retain their pages. Logical
    /// cancellation revokes visibility; it never establishes physical reuse.
    pub fn reap(&mut self) {
        self.exports.retain(|_, source| {
            let state = source.state();
            !state.released
                || state
                    .retirements
                    .iter()
                    .any(|completion| !completion.succeeded())
        });
        self.imports.retain(|&slot, write| {
            let state = write.state();
            if (!state.adopted && !state.released)
                || state.transfers.iter().any(|transfer| !transfer.succeeded())
            {
                return true;
            }
            if state.released {
                self.retiring.insert(slot);
            }
            false
        });

        self.retiring.retain(|&slot| {
            if self.imports.contains_key(&slot)
                || self
                    .exports
                    .values()
                    .any(|source| source.request_pool_idx == slot)
            {
                return true;
            }

            let current = &mut self.slots[slot];
            for &page in &current.pages {
                self.owners[page] = 0;
            }
            *current = LatentSlot::default();
            false
        });
    }
}
