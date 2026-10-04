//! Double-buffered latent pages and committed diffusion trajectories.
//!
//! The executor owns metadata changes. Tensor operations remain in the numerical
//! backend; import tickets and export completions keep pages out of reuse.

use std::collections::{HashMap, HashSet};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PySlice, PyTuple};
use uniserve_worker_ipc::{BufferId, RequestKey};

use super::completion::Completion;
use super::error::{invalid, resource};
use super::protocol::{buffer_id, request_key};
use super::transfer::TransferTicket;

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

/// A prepared trajectory commit or release, applied with its batch's outputs.
#[pyclass(get_all, set_all, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct LatentUpdate {
    request_pool_idx: i64,
    params: Option<Py<PyAny>>,
    expected_generation: i64,
    expected_step: i64,
    generation: i64,
    step: i64,
    release: bool,
}

#[pymethods]
impl LatentUpdate {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (request_pool_idx, params=None, expected_generation=0, expected_step=0, generation=0, step=0, release=false))]
    fn new(
        request_pool_idx: i64,
        params: Option<Py<PyAny>>,
        expected_generation: i64,
        expected_step: i64,
        generation: i64,
        step: i64,
        release: bool,
    ) -> Self {
        Self {
            request_pool_idx,
            params,
            expected_generation,
            expected_step,
            generation,
            step,
            release,
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.params)
    }
}

/// Destination pages retained until a transfer is adopted or abandoned and
/// every physical read has ended. Imported values occupy bank zero.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct LatentImport {
    id: BufferId,
    #[pyo3(get)]
    product: Py<PyAny>,
    #[pyo3(get)]
    request_pool_idx: usize,
    pages: Vec<usize>,
    units: usize,
    #[pyo3(get)]
    spans: Py<PyTuple>,
    transfers: Vec<Py<TransferTicket>>,
    #[pyo3(get)]
    adopted: bool,
    #[pyo3(get)]
    released: bool,
}

#[pymethods]
impl LatentImport {
    #[getter]
    fn page_table<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.pages)
    }

    #[getter]
    fn transfers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.transfers)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.product)?;
        visit.call(&self.spans)?;
        for ticket in &self.transfers {
            visit.call(ticket)?;
        }
        Ok(())
    }
}

/// An immutable page-bank version held by its transport readers.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct LatentExport {
    id: BufferId,
    #[pyo3(get)]
    buffer: Py<PyAny>,
    #[pyo3(get)]
    request_pool_idx: usize,
    #[pyo3(get)]
    bank: u8,
    pages: Vec<usize>,
    #[pyo3(get)]
    spans: Py<PyTuple>,
    retirements: Vec<Py<Completion>>,
    released: bool,
}

#[pymethods]
impl LatentExport {
    #[getter]
    fn page_table<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.pages)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.buffer)?;
        visit.call(&self.spans)?;
        for future in &self.retirements {
            visit.call(future)?;
        }
        Ok(())
    }
}

/// Own fixed latent backing, page assignments, and request-slot visibility.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct LatentPool {
    #[pyo3(get)]
    request_pool_size: usize,
    #[pyo3(get)]
    num_pages: usize,
    #[pyo3(get)]
    page_units: usize,
    #[pyo3(get)]
    latent_width: usize,
    #[pyo3(get)]
    capacity_units: usize,
    #[pyo3(get)]
    dtype: Py<PyAny>,
    #[pyo3(get)]
    device: Py<PyAny>,
    #[pyo3(get)]
    storage: Py<PyAny>,
    #[pyo3(get)]
    step_buffer: Py<PyAny>,
    #[pyo3(get)]
    page_table_buffer: Py<PyAny>,
    #[pyo3(get)]
    timesteps: Py<PyAny>,
    page_staging: Py<PyAny>,
    slots: Vec<LatentSlot>,
    owners: Vec<usize>,
    imports: HashMap<usize, Py<LatentImport>>,
    sources: HashMap<BufferId, Py<LatentExport>>,
    retiring: HashSet<usize>,
    #[pyo3(get)]
    exports: Py<PyDict>,
}

#[pymethods]
impl LatentPool {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (*, request_pool_size, num_pages, page_units, latent_width, dtype, device, staging=true))]
    fn new(
        py: Python<'_>,
        request_pool_size: i64,
        num_pages: i64,
        page_units: i64,
        latent_width: i64,
        dtype: Py<PyAny>,
        device: &Bound<'_, PyAny>,
        staging: bool,
    ) -> PyResult<Self> {
        if request_pool_size < 1 || num_pages < 2 || page_units < 1 || latent_width < 1 {
            return Err(PyValueError::new_err(
                "latent-pool shape must contain slots, pages, and elements",
            ));
        }
        if !dtype.bind(py).getattr("is_floating_point")?.is_truthy()? {
            return Err(PyValueError::new_err(
                "latent-pool storage must use a floating dtype",
            ));
        }
        let device = py.import("torch")?.getattr("device")?.call1((device,))?;
        let allocated = backend(py)?.getattr("_allocate")?.call1((
            request_pool_size,
            num_pages,
            page_units,
            latent_width,
            &dtype,
            &device,
            staging,
        ))?;

        Ok(Self {
            request_pool_size: request_pool_size as usize,
            num_pages: num_pages as usize,
            page_units: page_units as usize,
            latent_width: latent_width as usize,
            capacity_units: (num_pages as usize - 1) * page_units as usize,
            dtype,
            device: device.unbind(),
            storage: allocated.get_item(0)?.unbind(),
            step_buffer: allocated.get_item(1)?.unbind(),
            page_table_buffer: allocated.get_item(2)?.unbind(),
            timesteps: allocated.get_item(3)?.unbind(),
            page_staging: allocated.get_item(4)?.unbind(),
            slots: (0..=request_pool_size)
                .map(|_| LatentSlot::default())
                .collect(),
            owners: vec![0; num_pages as usize],
            imports: HashMap::new(),
            sources: HashMap::new(),
            retiring: HashSet::new(),
            exports: PyDict::new(py).unbind(),
        })
    }

    #[getter]
    fn persistent_bytes(&self, py: Python<'_>) -> PyResult<usize> {
        let mut bytes = 0;
        for value in [
            &self.storage,
            &self.step_buffer,
            &self.page_table_buffer,
            &self.timesteps,
        ] {
            let value = value.bind(py);
            bytes += value.call_method0("numel")?.extract::<usize>()?
                * value.call_method0("element_size")?.extract::<usize>()?;
        }
        Ok(bytes)
    }

    fn startup_values(slf: Bound<'_, Self>, rows: usize, units: i64) -> PyResult<Py<PyAny>> {
        Ok(backend(slf.py())?
            .getattr("_startup_values")?
            .call1((slf, rows, units))?
            .unbind())
    }

    fn _startup_staging(&self, py: Python<'_>, rows: usize, units: i64) -> PyResult<Py<PyTuple>> {
        if self.slots.iter().any(|slot| !slot.pages.is_empty())
            || !self.imports.is_empty()
            || !self.sources.is_empty()
        {
            return Err(PyRuntimeError::new_err(
                "startup scratch requires an idle latent pool",
            ));
        }
        if units < 1 {
            return Err(invalid(py, "latent startup requires positive units"));
        }
        let count = (units as usize).div_ceil(self.page_units);
        let tables = (0..rows)
            .map(|row| {
                (1 + row * count..1 + (row + 1) * count)
                    .map(|page| page as i64)
                    .collect()
            })
            .collect();
        let staged = self.stage(py, tables, vec![units; rows], Vec::new())?;
        let views = staged
            .bind(py)
            .iter()
            .map(|item| {
                item.getattr("value")?
                    .get_item(PySlice::new(py, 0, units as isize, 1))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, views)?.unbind())
    }

    /// Borrow contiguous staging alongside every still-running call in
    /// `occupied`. The caller retains those views through numerical completion;
    /// omitting a live view permits its scratch range to be overwritten.
    #[pyo3(signature = (page_tables, latent_units, *, occupied=Vec::new()))]
    fn stage(
        &self,
        py: Python<'_>,
        page_tables: Vec<Vec<i64>>,
        latent_units: Vec<i64>,
        occupied: Vec<Py<PyAny>>,
    ) -> PyResult<Py<PyTuple>> {
        if page_tables.is_empty() || page_tables.len() != latent_units.len() {
            return Err(invalid(py, "latent staging columns are not aligned"));
        }

        let tables = page_tables
            .iter()
            .zip(latent_units)
            .map(|(pages, units)| self.validate_pages(py, pages, units))
            .collect::<PyResult<Vec<_>>>()?;
        let total: usize = tables.iter().map(Vec::len).sum();
        if total >= self.num_pages {
            return Err(invalid(py, "latent staging exceeds the fixed step buffer"));
        }

        let mut pages = HashSet::new();
        for &page in tables.iter().flatten() {
            if !pages.insert(page) {
                return Err(invalid(py, "latent staging page tables overlap"));
            }
        }

        let mut ranges = Vec::new();
        for item in &occupied {
            let item = item.bind(py);
            let held: Vec<usize> = item.getattr("page_table")?.extract()?;
            if held.iter().any(|page| pages.contains(page)) {
                return Err(invalid(py, "latent staging page tables overlap live calls"));
            }
            let start = item
                .getattr("pages")?
                .call_method0("storage_offset")?
                .extract::<usize>()?;
            ranges.push((start, held.len()));
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
            return Err(resource(
                py,
                "live latent staging exceeds the fixed step buffer",
            ));
        }
        Ok(backend(py)?
            .getattr("_stage")?
            .call1((
                tables,
                offset,
                self.page_units,
                &self.page_table_buffer,
                &self.step_buffer,
                &self.page_staging,
            ))?
            .cast_into::<PyTuple>()?
            .unbind())
    }

    /// Scatter the initial values into bank one. The prepared values become
    /// visible when the batch applies its trajectory update.
    #[pyo3(signature = (request_pool_idx, staging, *, latent_units))]
    fn initialize(
        &mut self,
        py: Python<'_>,
        request_pool_idx: i64,
        staging: &Bound<'_, PyAny>,
        latent_units: i64,
    ) -> PyResult<()> {
        self.reap(py)?;
        let slot = self.slot(py, request_pool_idx)?;
        let pages = self.staging_pages(py, staging, latent_units)?;
        self.require_initial(py, slot, &pages)?;

        backend(py)?
            .getattr("_scatter")?
            .call1((&self.storage, 1, staging))?;
        Ok(())
    }

    /// Flattened page rows borrowed by numerical runners, with both banks.
    #[getter]
    fn page_rows(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self
            .storage
            .bind(py)
            .call_method1("view", (2 * self.num_pages, -1))?
            .unbind())
    }

    /// Borrow a contiguous bank range without a gather. The caller orders
    /// accesses and uses initial_bank or step_banks before writing.
    fn bank_view(&self, py: Python<'_>, bank: i64, page_table: Vec<i64>) -> PyResult<Py<PyAny>> {
        if !(0..=1).contains(&bank)
            || page_table.is_empty()
            || page_table[0] < 1
            || page_table[0] as usize + page_table.len() > self.num_pages
            || page_table
                .iter()
                .enumerate()
                .any(|(index, &page)| page != page_table[0] + index as i64)
        {
            return Err(invalid(
                py,
                "latent bank view requires consecutive pool pages",
            ));
        }
        let start = page_table[0] as isize;
        Ok(self
            .storage
            .bind(py)
            .get_item((
                bank,
                PySlice::new(py, start, start + page_table.len() as isize, 1),
            ))?
            .call_method1("view", (-1, self.latent_width))?
            .unbind())
    }

    #[pyo3(signature = (request_pool_idx, page_table, *, latent_units))]
    fn initial_bank(
        &mut self,
        py: Python<'_>,
        request_pool_idx: i64,
        page_table: Vec<i64>,
        latent_units: i64,
    ) -> PyResult<u8> {
        self.reap(py)?;
        let slot = self.slot(py, request_pool_idx)?;
        let pages = self.validate_pages(py, &page_table, latent_units)?;
        self.require_initial(py, slot, &pages)?;
        Ok(1)
    }

    /// Select the committed input bank and a writable successor bank.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (request_pool_idx, page_table, *, step, generation, latent_units, height, width))]
    fn step_banks(
        &mut self,
        py: Python<'_>,
        request_pool_idx: i64,
        page_table: Vec<i64>,
        step: i64,
        generation: i64,
        latent_units: i64,
        height: i64,
        width: i64,
    ) -> PyResult<(u8, u8)> {
        self.reap(py)?;
        let slot = self.slot(py, request_pool_idx)?;
        let pages = self.validate_pages(py, &page_table, latent_units)?;
        self.require_current(
            py,
            slot,
            step,
            generation,
            latent_units,
            height,
            width,
            &pages,
        )?;
        let bank = self.slots[slot].bank;
        self.require_writable(py, 1 - bank, &pages)?;
        Ok((bank, 1 - bank))
    }

    /// Gather the requested committed trajectory, preserving page-table order.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (request_pool_idx, staging, *, step, generation, latent_units, height, width))]
    fn gather_current(
        &mut self,
        py: Python<'_>,
        request_pool_idx: i64,
        staging: &Bound<'_, PyAny>,
        step: i64,
        generation: i64,
        latent_units: i64,
        height: i64,
        width: i64,
    ) -> PyResult<Py<PyAny>> {
        self.reap(py)?;
        let slot = self.slot(py, request_pool_idx)?;
        let pages = self.staging_pages(py, staging, latent_units)?;
        self.require_current(
            py,
            slot,
            step,
            generation,
            latent_units,
            height,
            width,
            &pages,
        )?;
        Ok(backend(py)?
            .getattr("_gather")?
            .call1((&self.storage, self.slots[slot].bank, staging, latent_units))?
            .unbind())
    }

    /// Write the hidden successor without changing the committed trajectory.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (request_pool_idx, staging, *, expected_step, expected_generation, latent_units, height, width))]
    fn write_inactive(
        &mut self,
        py: Python<'_>,
        request_pool_idx: i64,
        staging: &Bound<'_, PyAny>,
        expected_step: i64,
        expected_generation: i64,
        latent_units: i64,
        height: i64,
        width: i64,
    ) -> PyResult<()> {
        self.reap(py)?;
        let slot = self.slot(py, request_pool_idx)?;
        let pages = self.staging_pages(py, staging, latent_units)?;
        self.require_current(
            py,
            slot,
            expected_step,
            expected_generation,
            latent_units,
            height,
            width,
            &pages,
        )?;
        let bank = 1 - self.slots[slot].bank;
        self.require_writable(py, bank, &pages)?;
        backend(py)?
            .getattr("_scatter")?
            .call1((&self.storage, bank, staging))?;
        Ok(())
    }

    /// Retain the prepared successor before its batch commits. Attach every
    /// transport retirement with retain_publication, and release on abandonment.
    #[pyo3(signature = (product, *, request_pool_idx, page_table, latent_units))]
    fn reserve_publication(
        &mut self,
        py: Python<'_>,
        product: &Bound<'_, PyAny>,
        request_pool_idx: i64,
        page_table: Vec<i64>,
        latent_units: i64,
    ) -> PyResult<Py<LatentExport>> {
        self.reap(py)?;
        let slot = self.slot(py, request_pool_idx)?;
        let pages = self.validate_pages(py, &page_table, latent_units)?;
        let bank = 1 - self.slots[slot].bank;
        self.require_writable(py, bank, &pages)?;
        self.reserve_source(py, product, slot, bank, pages, latent_units)
    }

    /// Export the committed bank without advancing the trajectory. Multiple
    /// outputs may share that bank; every output must retire before reuse.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (product, *, request_pool_idx, page_table, generation, step, latent_units, height, width))]
    fn reserve_current_publication(
        &mut self,
        py: Python<'_>,
        product: &Bound<'_, PyAny>,
        request_pool_idx: i64,
        page_table: Vec<i64>,
        generation: i64,
        step: i64,
        latent_units: i64,
        height: i64,
        width: i64,
    ) -> PyResult<Py<LatentExport>> {
        self.reap(py)?;
        let slot = self.slot(py, request_pool_idx)?;
        let pages = self.validate_pages(py, &page_table, latent_units)?;
        self.require_current(
            py,
            slot,
            step,
            generation,
            latent_units,
            height,
            width,
            &pages,
        )?;
        self.reserve_source(
            py,
            product,
            slot,
            self.slots[slot].bank,
            pages,
            latent_units,
        )
    }

    fn retain_publication(
        &self,
        py: Python<'_>,
        source: &Bound<'_, LatentExport>,
        retirement: Py<Completion>,
    ) -> PyResult<()> {
        let mut value = source.borrow_mut();
        if value.released
            || self
                .sources
                .get(&value.id)
                .is_none_or(|registered| !registered.bind(py).is(source))
        {
            return Err(invalid(
                py,
                "latent publication reservation is no longer active",
            ));
        }
        value.retirements.push(retirement);
        Ok(())
    }

    /// Revoke exports while retaining pages through physical reader completion.
    fn release_buffers(slf: Bound<'_, Self>, buffers: Vec<Py<PyAny>>) -> PyResult<()> {
        let py = slf.py();
        let exports = {
            let owner = slf.borrow();
            for buffer in &buffers {
                if let Some(source) = owner.sources.get(&buffer_id(buffer.bind(py))?) {
                    source.borrow_mut(py).released = true;
                }
            }
            owner.exports.clone_ref(py)
        };

        // Transport retirement can notify observers synchronously. The pool
        // must be available to them while each backend revokes its reads.
        py.import("uniserve_worker.transport.exports")?
            .getattr("release_exports")?
            .call1((exports, buffers))?;
        slf.borrow_mut().reap(py)
    }

    /// Reader completions that must precede writing the next bank.
    fn write_dependencies(
        &mut self,
        py: Python<'_>,
        request_pool_idx: i64,
        page_table: Vec<usize>,
    ) -> PyResult<Py<PyTuple>> {
        self.reap(py)?;
        let slot = self.slot(py, request_pool_idx)?;
        let bank = 1 - self.slots[slot].bank;
        let pages: HashSet<_> = page_table.into_iter().collect();
        let mut futures = Vec::new();
        for source in self.sources.values() {
            let source = source.borrow(py);
            if source.bank == bank && source.pages.iter().any(|page| pages.contains(page)) {
                futures.extend(source.retirements.iter().map(|future| future.clone_ref(py)));
            }
        }
        Ok(PyTuple::new(py, futures)?.unbind())
    }

    fn stage_timestep(
        &self,
        py: Python<'_>,
        request_pool_idx: i64,
        value: f64,
    ) -> PyResult<Py<PyAny>> {
        let slot = self.slot(py, request_pool_idx)?;
        let row = self.timesteps.bind(py).get_item(slot)?;
        row.call_method1("fill_", (value,))?;
        Ok(row.unbind())
    }

    /// Check all slot changes before publishing any of the batch's outputs.
    fn validate_updates(&mut self, py: Python<'_>, updates: Vec<Py<LatentUpdate>>) -> PyResult<()> {
        self.reap(py)?;
        let mut slots = HashSet::new();
        let mut claimed = HashSet::new();
        for update in &updates {
            let update = update.borrow(py);
            let Some(params) = &update.params else {
                continue;
            };
            let params = params.bind(py);
            let slot = self.slot(py, update.request_pool_idx)?;
            if !slots.insert(slot) {
                return Err(invalid(py, "latent commit repeats a request slot"));
            }

            let units: i64 = params.getattr("latent_units")?.extract()?;
            let height = params.getattr("height")?.extract()?;
            let width = params.getattr("width")?.extract()?;
            let pages = self.validate_pages(
                py,
                &params.getattr("page_table")?.extract::<Vec<i64>>()?,
                units,
            )?;

            if update.release {
                self.require_current(
                    py,
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

            self.validate_metadata(py, update.generation, update.step, units, height, width)?;
            if update.expected_generation == 0 {
                if update.expected_step != 0 || update.step != 0 {
                    return Err(invalid(py, "latent initialization must publish step zero"));
                }
                self.require_empty(py, slot)?;

                // The successor may already have been exported before this
                // commit. Its own provisional bank does not conflict here.
                self.require_owners(py, &pages, 0, Some(slot))?;
            } else {
                self.require_current(
                    py,
                    slot,
                    update.expected_step,
                    update.expected_generation,
                    units,
                    height,
                    width,
                    &pages,
                )?;
                if update.step <= update.expected_step {
                    return Err(invalid(py, "latent successor does not advance its step"));
                }
            }

            if update.generation <= update.expected_generation {
                return Err(invalid(
                    py,
                    "latent publication does not advance its generation",
                ));
            }
            for page in pages {
                if !claimed.insert(page) {
                    return Err(invalid(
                        py,
                        "latent commit publications overlap physical pages",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Apply an already validated batch without rechecking its values. The
    /// executor must not change the pool or updates between these two calls.
    fn apply_updates(&mut self, py: Python<'_>, updates: Vec<Py<LatentUpdate>>) -> PyResult<()> {
        for update in &updates {
            let update = update.borrow(py);
            let Some(params) = &update.params else {
                continue;
            };
            let slot = update.request_pool_idx as usize;
            if update.release {
                self.clear_slot(py, slot)?;
                continue;
            }
            let params = params.bind(py);
            let current = &mut self.slots[slot];
            if update.expected_generation == 0 {
                current.pages = params.getattr("page_table")?.extract()?;
                for &page in &current.pages {
                    self.owners[page] = slot;
                }
            }
            current.bank = 1 - current.bank;
            current.generation = update.generation;
            current.step = update.step;
            current.units = params.getattr("latent_units")?.extract()?;
            current.height = params.getattr("height")?.extract()?;
            current.width = params.getattr("width")?.extract()?;
        }
        self.reap(py)
    }

    /// Assign destination pages in bank zero before any read starts. Padding is
    /// initialized here; transfers write only the logical spans.
    #[pyo3(signature = (product, *, request_pool_idx, page_table, latent_units))]
    fn reserve_import(
        &mut self,
        py: Python<'_>,
        product: Py<PyAny>,
        request_pool_idx: i64,
        page_table: Vec<i64>,
        latent_units: i64,
    ) -> PyResult<Py<LatentImport>> {
        self.reap(py)?;
        let slot = self.slot(py, request_pool_idx)?;
        let pages = self.validate_pages(py, &page_table, latent_units)?;
        self.require_empty(py, slot)?;
        self.require_owners(py, &pages, 0, None)?;
        let spans = self.spans(py, 0, &pages, latent_units)?;
        let id = buffer_id(&product.bind(py).getattr("buffer_id")?)?;

        // Only the final page's padding is initialized here. It is outside
        // every span granted to the independent transfer stream.
        let last = pages[pages.len() - 1];
        let used = latent_units as usize - (pages.len() - 1) * self.page_units;
        self.storage
            .bind(py)
            .get_item((
                0,
                last,
                PySlice::new(py, used as isize, self.page_units as isize, 1),
            ))?
            .call_method0("zero_")?;

        let write = Py::new(
            py,
            LatentImport {
                id,
                product,
                request_pool_idx: slot,
                pages: pages.clone(),
                units: latent_units as usize,
                spans,
                transfers: Vec::new(),
                adopted: false,
                released: false,
            },
        )?;
        for &page in &pages {
            self.owners[page] = slot;
        }
        self.slots[slot].pages = pages;
        self.imports.insert(slot, write.clone_ref(py));
        Ok(write)
    }

    fn retain_transfer(
        &self,
        py: Python<'_>,
        write: &Bound<'_, LatentImport>,
        ticket: Py<TransferTicket>,
    ) -> PyResult<()> {
        self.require_import(py, write)?;
        let mut write = write.borrow_mut();
        if write.adopted {
            return Err(invalid(
                py,
                "resident latent import cannot accept another read",
            ));
        }
        write.transfers.push(ticket);
        Ok(())
    }

    /// Expose imported pages after ordering their producer fences on the
    /// consuming stream. A pending or failed transfer cannot be adopted.
    #[pyo3(signature = (write, *, generation, step, height, width))]
    fn adopt_import(
        &mut self,
        py: Python<'_>,
        write: &Bound<'_, LatentImport>,
        generation: i64,
        step: i64,
        height: i64,
        width: i64,
    ) -> PyResult<()> {
        self.require_import(py, write)?;
        let mut write = write.borrow_mut();
        if write.adopted || write.transfers.is_empty() {
            return Err(invalid(py, "latent import cannot be adopted"));
        }
        self.validate_metadata(py, generation, step, write.units as i64, height, width)?;
        if generation != i64::from(write.id.generation) {
            return Err(invalid(
                py,
                "latent import generation disagrees with its product",
            ));
        }

        // Results insert producer fences on the consuming stream. Readiness
        // alone is insufficient to expose imported device values.
        for ticket in &write.transfers {
            ticket.get().result(py, None)?;
        }

        let current = &mut self.slots[write.request_pool_idx];
        current.bank = 0;
        current.step = step;
        current.generation = generation;
        current.units = write.units;
        current.height = height;
        current.width = width;
        write.adopted = true;
        drop(write);
        self.reap(py)
    }

    fn abandon_import(&mut self, py: Python<'_>, write: &Bound<'_, LatentImport>) -> PyResult<()> {
        if write.borrow().released {
            return Ok(());
        }
        self.require_import(py, write)?;
        let mut write = write.borrow_mut();
        if write.adopted {
            return Err(invalid(py, "resident latent import cannot be abandoned"));
        }
        write.released = true;
        for ticket in &write.transfers {
            ticket.get().cancel(py)?;
        }
        drop(write);
        self.reap(py)
    }

    /// Report physical completion failures only to the affected requests.
    /// Independent requests can continue using their own pages.
    fn retirement_ready(&mut self, py: Python<'_>, requests: Vec<Py<PyAny>>) -> PyResult<bool> {
        let requests: HashSet<RequestKey> = requests
            .iter()
            .map(|value| request_key(value.bind(py)))
            .collect::<PyResult<_>>()?;
        for write in self.imports.values() {
            let write = write.borrow(py);
            if requests.contains(&write.id.owner) {
                for ticket in &write.transfers {
                    ticket.get().retirement_ready(py)?;
                }
            }
        }
        for source in self.sources.values() {
            let source = source.borrow(py);
            if requests.contains(&source.id.owner) {
                for future in &source.retirements {
                    let completion = future.borrow(py);
                    if completion.done() {
                        completion.result(py, None)?;
                    }
                }
            }
        }
        self.reap(py)?;
        Ok(!self
            .imports
            .values()
            .any(|write| requests.contains(&write.borrow(py).id.owner))
            && !self
                .sources
                .values()
                .any(|source| requests.contains(&source.borrow(py).id.owner)))
    }

    fn cancel_imports(&mut self, py: Python<'_>, requests: Vec<Py<PyAny>>) -> PyResult<()> {
        let requests: HashSet<RequestKey> = requests
            .iter()
            .map(|value| request_key(value.bind(py)))
            .collect::<PyResult<_>>()?;
        for write in self.imports.values() {
            let mut write = write.borrow_mut(py);
            if requests.contains(&write.id.owner) && !write.adopted && !write.released {
                write.released = true;
                for ticket in &write.transfers {
                    ticket.get().cancel(py)?;
                }
            }
        }
        self.reap(py)
    }

    /// Release slot ownership after all registered imports and exports retire.
    fn release_slots(&mut self, py: Python<'_>, request_pool_indices: Vec<i64>) -> PyResult<()> {
        let slots = request_pool_indices
            .iter()
            .map(|&slot| self.slot(py, slot))
            .collect::<PyResult<Vec<_>>>()?;
        if slots.iter().copied().collect::<HashSet<_>>().len() != slots.len() {
            return Err(invalid(py, "latent release repeats a request slot"));
        }
        for slot in slots {
            self.clear_slot(py, slot)?;
        }
        self.reap(py)
    }

    /// Drop backing only after reads drain. Pending or failed physical access
    /// raises a resource error and keeps the storage retained.
    fn close(slf: Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let buffers = {
            let owner = slf.borrow();
            owner.page_staging.bind(py).call_method0("close")?;
            owner
                .sources
                .values()
                .map(|source| source.borrow(py).buffer.clone_ref(py))
                .collect()
        };
        Self::release_buffers(slf.clone(), buffers)?;
        let mut owner = slf.borrow_mut();
        let slots: HashSet<usize> = owner
            .imports
            .keys()
            .copied()
            .chain(
                owner
                    .sources
                    .values()
                    .map(|source| source.borrow(py).request_pool_idx),
            )
            .collect();
        for slot in slots {
            owner.clear_slot(py, slot)?;
        }
        owner.reap(py)?;
        if !owner.imports.is_empty() || !owner.sources.is_empty() {
            return Err(resource(
                py,
                "latent physical reads must retire before pool shutdown",
            ));
        }
        owner.exports.bind(py).clear();
        let empty = backend(py)?
            .getattr("_empty_storage")?
            .call1((&owner.dtype, &owner.device))?;
        owner.storage = empty.get_item(0)?.unbind();
        owner.step_buffer = empty.get_item(1)?.unbind();
        owner.page_table_buffer = empty.get_item(2)?.unbind();
        owner.timesteps = empty.get_item(3)?.unbind();
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for value in [
            &self.dtype,
            &self.device,
            &self.storage,
            &self.step_buffer,
            &self.page_table_buffer,
            &self.timesteps,
            &self.page_staging,
        ] {
            visit.call(value)?;
        }
        visit.call(&self.exports)?;
        for write in self.imports.values() {
            visit.call(write)?;
        }
        for source in self.sources.values() {
            visit.call(source)?;
        }
        Ok(())
    }
}

impl LatentPool {
    fn slot(&self, py: Python<'_>, slot: i64) -> PyResult<usize> {
        if slot < 1 || slot as usize > self.request_pool_size {
            return Err(invalid(
                py,
                "latent request slot is outside physical capacity",
            ));
        }
        Ok(slot as usize)
    }

    fn validate_pages(&self, py: Python<'_>, pages: &[i64], units: i64) -> PyResult<Vec<usize>> {
        if units < 1
            || pages.len() != (units as usize).div_ceil(self.page_units)
            || pages
                .iter()
                .any(|&page| page < 1 || page as usize >= self.num_pages)
            || pages.iter().collect::<HashSet<_>>().len() != pages.len()
        {
            return Err(invalid(
                py,
                "latent page table is outside physical pool bounds",
            ));
        }
        Ok(pages.iter().map(|&page| page as usize).collect())
    }

    fn staging_pages(
        &self,
        py: Python<'_>,
        staging: &Bound<'_, PyAny>,
        units: i64,
    ) -> PyResult<Vec<usize>> {
        let pages = staging.getattr("page_table")?.extract::<Vec<i64>>()?;
        let pages = self.validate_pages(py, &pages, units)?;
        backend(py)?
            .getattr("_check_staging")?
            .call1((staging, units, &self.storage))?;
        Ok(pages)
    }

    fn require_empty(&self, py: Python<'_>, slot: usize) -> PyResult<()> {
        let current = &self.slots[slot];
        if self.imports.contains_key(&slot)
            || current.generation != 0
            || current.units != 0
            || self.retiring.contains(&slot)
        {
            return Err(invalid(
                py,
                "request slot already owns a committed trajectory",
            ));
        }
        Ok(())
    }

    fn require_initial(&self, py: Python<'_>, slot: usize, pages: &[usize]) -> PyResult<()> {
        self.require_empty(py, slot)?;
        self.require_owners(py, pages, 0, None)?;
        self.require_writable(py, 1, pages)
    }

    fn require_owners(
        &self,
        py: Python<'_>,
        pages: &[usize],
        owner: usize,
        publication_slot: Option<usize>,
    ) -> PyResult<()> {
        // A producer can export its first bank before committing the trajectory.
        // Such pages are retained even though no slot owns them yet.
        if owner == 0
            && self.sources.values().any(|source| {
                let source = source.borrow(py);
                Some(source.request_pool_idx) != publication_slot
                    && source.pages.iter().any(|page| pages.contains(page))
            })
        {
            return Err(invalid(py, "latent pages are owned by a published version"));
        }
        if pages.iter().any(|&page| self.owners[page] != owner) {
            return Err(invalid(
                py,
                "latent page table is not owned by its request slot",
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn require_current(
        &self,
        py: Python<'_>,
        slot: usize,
        step: i64,
        generation: i64,
        units: i64,
        height: i64,
        width: i64,
        pages: &[usize],
    ) -> PyResult<()> {
        let current = &self.slots[slot];
        if current.step != step
            || current.generation != generation
            || current.units as i64 != units
            || current.height != height
            || current.width != width
        {
            return Err(invalid(
                py,
                "latent allocation does not name the committed trajectory",
            ));
        }
        if current.pages != pages {
            return Err(invalid(
                py,
                "latent page table does not match its committed trajectory",
            ));
        }
        self.require_owners(py, pages, slot, None)
    }

    fn require_writable(&self, py: Python<'_>, bank: u8, pages: &[usize]) -> PyResult<()> {
        if self.sources.values().any(|source| {
            let source = source.borrow(py);
            source.bank == bank && source.pages.iter().any(|page| pages.contains(page))
        }) {
            return Err(invalid(
                py,
                "latent bank is retained by a published version",
            ));
        }
        Ok(())
    }

    fn validate_metadata(
        &self,
        py: Python<'_>,
        generation: i64,
        step: i64,
        units: i64,
        height: i64,
        width: i64,
    ) -> PyResult<()> {
        if generation < 1 || step < 0 || units < 1 || height < 1 || width < 1 {
            return Err(invalid(py, "latent publication metadata is invalid"));
        }
        if units as usize > self.capacity_units {
            return Err(invalid(py, "latent publication exceeds physical capacity"));
        }
        Ok(())
    }

    fn spans(
        &self,
        py: Python<'_>,
        bank: u8,
        pages: &[usize],
        units: i64,
    ) -> PyResult<Py<PyTuple>> {
        Ok(backend(py)?
            .getattr("_spans")?
            .call1((&self.storage, bank, pages, units))?
            .cast_into::<PyTuple>()?
            .unbind())
    }

    fn reserve_source(
        &mut self,
        py: Python<'_>,
        product: &Bound<'_, PyAny>,
        slot: usize,
        bank: u8,
        pages: Vec<usize>,
        units: i64,
    ) -> PyResult<Py<LatentExport>> {
        let buffer = product.getattr("buffer_id")?;
        let id = buffer_id(&buffer)?;
        if self.sources.contains_key(&id) {
            return Err(invalid(
                py,
                "latent publication generation is already registered",
            ));
        }

        let spans = self.spans(py, bank, &pages, units)?;
        let source = Py::new(
            py,
            LatentExport {
                id,
                buffer: buffer.unbind(),
                request_pool_idx: slot,
                bank,
                pages,
                spans,
                retirements: Vec::new(),
                released: false,
            },
        )?;
        self.sources.insert(id, source.clone_ref(py));
        Ok(source)
    }

    fn require_import(&self, py: Python<'_>, write: &Bound<'_, LatentImport>) -> PyResult<()> {
        let value = write.borrow();
        if value.released
            || self
                .imports
                .get(&value.request_pool_idx)
                .is_none_or(|registered| !registered.bind(py).is(write))
        {
            return Err(invalid(
                py,
                "latent import reservation is no longer writable",
            ));
        }
        Ok(())
    }

    fn clear_slot(&mut self, py: Python<'_>, slot: usize) -> PyResult<()> {
        self.retiring.insert(slot);
        if let Some(write) = self.imports.get(&slot) {
            let mut write = write.borrow_mut(py);
            write.released = true;
            if !write.adopted {
                for ticket in &write.transfers {
                    ticket.get().cancel(py)?;
                }
            }
        }
        Ok(())
    }

    fn reap(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut exports = Vec::new();
        for (id, source) in &self.sources {
            let source = source.borrow(py);
            if !source.released {
                continue;
            }
            let mut retired = true;
            for future in &source.retirements {
                retired &= future.borrow(py).succeeded();
            }
            if retired {
                exports.push(*id);
            }
        }
        for id in exports {
            self.sources.remove(&id);
        }

        let mut imports = Vec::new();
        for (&slot, write) in &self.imports {
            let write = write.borrow(py);
            if !write.adopted && !write.released {
                continue;
            }
            let mut retired = true;
            for ticket in &write.transfers {
                retired &= ticket.get().retired(py)?;
            }
            if retired {
                imports.push(slot);
                if write.released {
                    self.retiring.insert(slot);
                }
            }
        }
        for slot in imports {
            self.imports.remove(&slot);
        }

        // Cancellation removes visibility immediately; reuse waits for all
        // copies and readers, including readers of an earlier bank version.
        self.retiring.retain(|&slot| {
            if self.imports.contains_key(&slot)
                || self
                    .sources
                    .values()
                    .any(|source| source.borrow(py).request_pool_idx == slot)
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
        Ok(())
    }
}

fn backend(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("uniserve_worker.storage.latent_pool")
}
