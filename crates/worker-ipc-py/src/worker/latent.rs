//! PyTorch views and transfer operations for native latent trajectories.

use std::collections::HashSet;
use std::ops::Deref;
use std::sync::Arc;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PySlice, PyTuple};
use uniserve_worker::{
    LatentExport as NativeLatentExport, LatentImport as NativeLatentImport,
    LatentPool as NativeLatentPool, LatentUpdate as NativeLatentUpdate,
};
use uniserve_worker_ipc::{LatentParams, RequestKey};

use super::completion::{Completion, CompletionRef};
use super::error::{invalid, native_error};
use super::protocol::{buffer_id, call_id, request_key};
use super::transfer::{TransferRef, TransferTicket};

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
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct LatentImport {
    inner: Arc<NativeLatentImport<TransferRef>>,
    #[pyo3(get)]
    product: Py<PyAny>,
    #[pyo3(get)]
    spans: Py<PyTuple>,
}

#[pymethods]
impl LatentImport {
    #[getter]
    fn request_pool_idx(&self) -> usize {
        self.inner.request_pool_idx
    }

    #[getter]
    fn adopted(&self) -> bool {
        self.inner.adopted()
    }

    #[getter]
    fn released(&self) -> bool {
        self.inner.released()
    }

    #[getter]
    fn page_table<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.pages)
    }

    #[getter]
    fn transfers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            self.inner
                .transfers()
                .iter()
                .map(|ticket| ticket.owner.clone_ref(py)),
        )
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.product)?;
        visit.call(&self.spans)?;
        for ticket in self.inner.transfers() {
            visit.call(&ticket.owner)?;
        }
        Ok(())
    }
}

/// An immutable page-bank version held by its transport readers.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct LatentExport {
    inner: Arc<NativeLatentExport<CompletionRef>>,
    #[pyo3(get)]
    buffer: Py<PyAny>,
    #[pyo3(get)]
    spans: Py<PyTuple>,
}

#[pymethods]
impl LatentExport {
    #[getter]
    fn request_pool_idx(&self) -> usize {
        self.inner.request_pool_idx
    }

    #[getter]
    fn bank(&self) -> u8 {
        self.inner.bank
    }

    #[getter]
    fn page_table<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.pages)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.buffer)?;
        visit.call(&self.spans)?;
        for completion in self.inner.retirements() {
            visit.call(&completion.owner)?;
        }
        Ok(())
    }
}

// Keep the public wrappers alive for Python GC while the core uses their
// stable native objects. There is one owner of each import/export lifecycle.
struct ImportRef {
    owner: Py<LatentImport>,
    inner: Arc<NativeLatentImport<TransferRef>>,
}

impl Deref for ImportRef {
    type Target = NativeLatentImport<TransferRef>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

struct ExportRef {
    owner: Py<LatentExport>,
    inner: Arc<NativeLatentExport<CompletionRef>>,
}

impl Deref for ExportRef {
    type Target = NativeLatentExport<CompletionRef>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Own fixed latent backing, page assignments, and request-slot visibility.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct LatentPool {
    #[pyo3(get)]
    latent_width: usize,
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
    inner: NativeLatentPool<ImportRef, ExportRef>,
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
        request_pool_size: usize,
        num_pages: usize,
        page_units: usize,
        latent_width: usize,
        dtype: Py<PyAny>,
        device: &Bound<'_, PyAny>,
        staging: bool,
    ) -> PyResult<Self> {
        if latent_width < 1 {
            return Err(PyValueError::new_err(
                "latent-pool shape must contain slots, pages, and elements",
            ));
        }
        if !dtype.bind(py).getattr("is_floating_point")?.is_truthy()? {
            return Err(PyValueError::new_err(
                "latent-pool storage must use a floating dtype",
            ));
        }
        let inner = NativeLatentPool::new(request_pool_size, num_pages, page_units)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
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
            latent_width,
            dtype,
            device: device.unbind(),
            storage: allocated.get_item(0)?.unbind(),
            step_buffer: allocated.get_item(1)?.unbind(),
            page_table_buffer: allocated.get_item(2)?.unbind(),
            timesteps: allocated.get_item(3)?.unbind(),
            page_staging: allocated.get_item(4)?.unbind(),
            inner,
            exports: PyDict::new(py).unbind(),
        })
    }

    #[getter]
    fn request_pool_size(&self) -> usize {
        self.inner.request_pool_size()
    }

    #[getter]
    fn num_pages(&self) -> usize {
        self.inner.num_pages()
    }

    #[getter]
    fn page_units(&self) -> usize {
        self.inner.page_units()
    }

    #[getter]
    fn capacity_units(&self) -> usize {
        self.inner.capacity_units()
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
        if !self.inner.idle() {
            return Err(PyRuntimeError::new_err(
                "startup scratch requires an idle latent pool",
            ));
        }
        if units < 1 {
            return Err(invalid(py, "latent startup requires positive units"));
        }
        let count = (units as usize).div_ceil(self.inner.page_units());
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
        let occupied = occupied
            .iter()
            .map(|item| {
                let item = item.bind(py);
                Ok::<_, PyErr>((
                    item.getattr("page_table")?.extract::<Vec<usize>>()?,
                    item.getattr("pages")?
                        .call_method0("storage_offset")?
                        .extract::<usize>()?,
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let (tables, offset) = self
            .inner
            .stage(&page_tables, &latent_units, &occupied)
            .map_err(|error| native_error(py, error))?;

        Ok(backend(py)?
            .getattr("_stage")?
            .call1((
                tables,
                offset,
                self.inner.page_units(),
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
        self.inner.reap();
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
            .call_method1("view", (2 * self.inner.num_pages(), -1))?
            .unbind())
    }

    /// Borrow a contiguous bank range without a gather. The caller orders
    /// accesses and uses initial_bank or step_banks before writing.
    fn bank_view(&self, py: Python<'_>, bank: i64, page_table: Vec<i64>) -> PyResult<Py<PyAny>> {
        if !(0..=1).contains(&bank)
            || page_table.is_empty()
            || page_table[0] < 1
            || page_table[0] as usize + page_table.len() > self.inner.num_pages()
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
        self.inner.reap();
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
        self.inner.reap();
        let slot = self.slot(py, request_pool_idx)?;
        let pages = self.validate_pages(py, &page_table, latent_units)?;
        let bank = self.current_bank(
            py,
            slot,
            step,
            generation,
            latent_units,
            height,
            width,
            &pages,
        )?;
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
        self.inner.reap();
        let slot = self.slot(py, request_pool_idx)?;
        let pages = self.staging_pages(py, staging, latent_units)?;
        let bank = self.current_bank(
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
            .call1((&self.storage, bank, staging, latent_units))?
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
        self.inner.reap();
        let slot = self.slot(py, request_pool_idx)?;
        let pages = self.staging_pages(py, staging, latent_units)?;
        let bank = 1 - self.current_bank(
            py,
            slot,
            expected_step,
            expected_generation,
            latent_units,
            height,
            width,
            &pages,
        )?;
        self.require_writable(py, bank, &pages)?;
        backend(py)?
            .getattr("_scatter")?
            .call1((&self.storage, bank, staging))?;
        Ok(())
    }

    /// Retain the prepared successor before its batch commits. Attach every
    /// transport retirement with retain_export, and release on abandonment.
    #[pyo3(signature = (product, *, request_pool_idx, page_table, latent_units))]
    fn reserve_export(
        &mut self,
        py: Python<'_>,
        product: &Bound<'_, PyAny>,
        request_pool_idx: i64,
        page_table: Vec<i64>,
        latent_units: i64,
    ) -> PyResult<Py<LatentExport>> {
        self.inner.reap();
        let slot = self.slot(py, request_pool_idx)?;
        let pages = self.validate_pages(py, &page_table, latent_units)?;
        let bank = self.inner.next_bank(slot);
        self.require_writable(py, bank, &pages)?;
        self.reserve_source(py, product, slot, bank, pages, latent_units)
    }

    /// Export the committed bank without advancing the trajectory. Multiple
    /// outputs may share that bank; every output must retire before reuse.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (product, *, request_pool_idx, page_table, generation, step, latent_units, height, width))]
    fn reserve_current_export(
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
        self.inner.reap();
        let slot = self.slot(py, request_pool_idx)?;
        let pages = self.validate_pages(py, &page_table, latent_units)?;
        let bank = self.current_bank(
            py,
            slot,
            step,
            generation,
            latent_units,
            height,
            width,
            &pages,
        )?;
        self.reserve_source(py, product, slot, bank, pages, latent_units)
    }

    fn retain_export(
        &self,
        py: Python<'_>,
        source: &Bound<'_, LatentExport>,
        retirement: Py<Completion>,
    ) -> PyResult<()> {
        self.inner
            .retain_export(&source.get().inner, CompletionRef::new(py, retirement))
            .map_err(|error| native_error(py, error))
    }

    /// Revoke exports while retaining pages through physical reader completion.
    fn release_buffers(slf: Bound<'_, Self>, buffers: Vec<Py<PyAny>>) -> PyResult<()> {
        let py = slf.py();
        let ids = buffers
            .iter()
            .map(|buffer| buffer_id(buffer.bind(py)))
            .collect::<PyResult<Vec<_>>>()?;
        let exports = {
            let owner = slf.borrow();
            owner.inner.release_exports(&ids);
            owner.exports.clone_ref(py)
        };

        // Backend revocation may notify observers synchronously, so the pool
        // must be available while those callbacks run.
        py.import("uniserve_worker.transport.exports")?
            .getattr("release_exports")?
            .call1((exports, buffers))?;
        slf.borrow_mut().inner.reap();
        Ok(())
    }

    /// Reader completions that must precede writing the next bank.
    fn write_dependencies(
        &mut self,
        py: Python<'_>,
        request_pool_idx: i64,
        page_table: Vec<usize>,
    ) -> PyResult<Py<PyTuple>> {
        self.inner.reap();
        let slot = self.slot(py, request_pool_idx)?;
        let dependencies = self.inner.write_dependencies(slot, &page_table);
        Ok(PyTuple::new(
            py,
            dependencies
                .iter()
                .map(|completion| completion.owner.clone_ref(py)),
        )?
        .unbind())
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

    /// Check every trajectory update before committing any batch output.
    fn validate_updates(&mut self, py: Python<'_>, updates: Vec<Py<LatentUpdate>>) -> PyResult<()> {
        self.inner.reap();
        self.inner
            .validate_updates(&lower_updates(py, &updates)?)
            .map_err(|error| native_error(py, error))
    }

    /// The executor applies the same updates without intervening pool changes.
    fn apply_updates(&mut self, py: Python<'_>, updates: Vec<Py<LatentUpdate>>) -> PyResult<()> {
        cancel_transfers(py, self.inner.apply_updates(&lower_updates(py, &updates)?))?;
        self.inner.reap();
        Ok(())
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
        self.inner.reap();
        let slot = self.slot(py, request_pool_idx)?;
        let pages = self.validate_pages(py, &page_table, latent_units)?;
        let id = buffer_id(&product.bind(py).getattr("buffer_id")?)?;
        let inner = Arc::new(
            self.inner
                .prepare_import(id, slot, pages, latent_units as usize)
                .map_err(|error| native_error(py, error))?,
        );
        let spans = self.spans(py, 0, &inner.pages, latent_units)?;

        // Padding lies outside every span granted to the transfer stream.
        // The owning call prevents pool changes until registration below.
        let last = inner.pages[inner.pages.len() - 1];
        let used = inner.units - (inner.pages.len() - 1) * self.inner.page_units();
        self.storage
            .bind(py)
            .get_item((
                0,
                last,
                PySlice::new(py, used as isize, self.inner.page_units() as isize, 1),
            ))?
            .call_method0("zero_")?;

        let write = Py::new(
            py,
            LatentImport {
                inner: Arc::clone(&inner),
                product,
                spans,
            },
        )?;
        self.inner.register_import(ImportRef {
            owner: write.clone_ref(py),
            inner,
        });
        Ok(write)
    }

    fn retain_transfer(
        &self,
        py: Python<'_>,
        write: &Bound<'_, LatentImport>,
        ticket: Py<TransferTicket>,
    ) -> PyResult<()> {
        self.inner
            .retain_transfer(&write.get().inner, TransferRef::new(py, ticket)?)
            .map_err(|error| native_error(py, error))
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
        let write = &write.get().inner;
        self.inner
            .validate_adoption(write, generation, step, height, width)
            .map_err(|error| native_error(py, error))?;

        // Results order producer fences on the consuming stream before the
        // native trajectory can expose the imported bank.
        for ticket in write.transfers() {
            ticket.owner.get().result(py, None)?;
        }
        self.inner
            .adopt_import(write, generation, step, height, width);
        self.inner.reap();
        Ok(())
    }

    fn abandon_import(&mut self, py: Python<'_>, write: &Bound<'_, LatentImport>) -> PyResult<()> {
        let transfers = self
            .inner
            .abandon_import(&write.get().inner)
            .map_err(|error| native_error(py, error))?;
        cancel_transfers(py, transfers)?;
        self.inner.reap();
        Ok(())
    }

    /// Report physical completion failures only to the affected requests.
    /// Independent requests can continue using their own pages.
    fn retirement_ready(&mut self, py: Python<'_>, requests: Vec<Py<PyAny>>) -> PyResult<bool> {
        let requests = request_set(py, &requests)?;
        for write in self.inner.imports() {
            if requests.contains(&write.buffer.owner) {
                for ticket in write.transfers() {
                    ticket.owner.get().retirement_ready(py)?;
                }
            }
        }
        for source in self.inner.exports() {
            if requests.contains(&source.buffer.owner) {
                for completion in source.retirements() {
                    if completion.done() {
                        completion.owner.borrow(py).result(py, None)?;
                    }
                }
            }
        }

        self.inner.reap();
        Ok(self.inner.retirement_ready(&requests))
    }

    fn cancel_imports(&mut self, py: Python<'_>, requests: Vec<Py<PyAny>>) -> PyResult<()> {
        cancel_transfers(py, self.inner.cancel_imports(&request_set(py, &requests)?))?;
        self.inner.reap();
        Ok(())
    }

    /// Request cancellation retains page ownership through physical retirement.
    fn release_slots(&mut self, py: Python<'_>, request_pool_indices: Vec<i64>) -> PyResult<()> {
        let transfers = self
            .inner
            .release_slots(&request_pool_indices)
            .map_err(|error| native_error(py, error))?;
        cancel_transfers(py, transfers)?;
        self.inner.reap();
        Ok(())
    }

    /// Drop backing only after reads drain. Pending or failed physical access
    /// raises a resource error and keeps the storage retained.
    fn close(slf: Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let buffers = {
            let owner = slf.borrow();
            owner.page_staging.bind(py).call_method0("close")?;
            owner
                .inner
                .exports()
                .map(|source| source.owner.get().buffer.clone_ref(py))
                .collect()
        };
        Self::release_buffers(slf.clone(), buffers)?;
        let mut owner = slf.borrow_mut();
        let slots = owner
            .inner
            .imports()
            .map(|write| write.request_pool_idx as i64)
            .chain(
                owner
                    .inner
                    .exports()
                    .map(|source| source.request_pool_idx as i64),
            )
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        owner.release_slots(py, slots)?;
        owner
            .inner
            .require_retired()
            .map_err(|error| native_error(py, error))?;

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
        for write in self.inner.imports() {
            visit.call(&write.owner)?;
        }
        for source in self.inner.exports() {
            visit.call(&source.owner)?;
        }
        Ok(())
    }
}

impl LatentPool {
    fn slot(&self, py: Python<'_>, slot: i64) -> PyResult<usize> {
        self.inner
            .slot(slot)
            .map_err(|error| native_error(py, error))
    }

    fn validate_pages(&self, py: Python<'_>, pages: &[i64], units: i64) -> PyResult<Vec<usize>> {
        self.inner
            .validate_pages(pages, units)
            .map_err(|error| native_error(py, error))
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

    fn require_initial(&self, py: Python<'_>, slot: usize, pages: &[usize]) -> PyResult<()> {
        self.inner
            .require_initial(slot, pages)
            .map_err(|error| native_error(py, error))
    }

    #[allow(clippy::too_many_arguments)]
    fn current_bank(
        &self,
        py: Python<'_>,
        slot: usize,
        step: i64,
        generation: i64,
        units: i64,
        height: i64,
        width: i64,
        pages: &[usize],
    ) -> PyResult<u8> {
        self.inner
            .current_bank(slot, step, generation, units, height, width, pages)
            .map_err(|error| native_error(py, error))
    }

    fn require_writable(&self, py: Python<'_>, bank: u8, pages: &[usize]) -> PyResult<()> {
        self.inner
            .require_writable(bank, pages)
            .map_err(|error| native_error(py, error))
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
        let inner = Arc::new(
            self.inner
                .prepare_export(buffer_id(&buffer)?, slot, bank, pages)
                .map_err(|error| native_error(py, error))?,
        );
        let spans = self.spans(py, bank, &inner.pages, units)?;
        let source = Py::new(
            py,
            LatentExport {
                inner: Arc::clone(&inner),
                buffer: buffer.unbind(),
                spans,
            },
        )?;
        self.inner.register_export(ExportRef {
            owner: source.clone_ref(py),
            inner,
        });
        Ok(source)
    }
}

fn cancel_transfers(py: Python<'_>, transfers: Vec<Arc<TransferRef>>) -> PyResult<()> {
    for ticket in transfers {
        ticket.owner.get().cancel(py)?;
    }
    Ok(())
}

fn request_set(py: Python<'_>, requests: &[Py<PyAny>]) -> PyResult<HashSet<RequestKey>> {
    requests
        .iter()
        .map(|value| request_key(value.bind(py)))
        .collect()
}

fn lower_updates(
    py: Python<'_>,
    updates: &[Py<LatentUpdate>],
) -> PyResult<Vec<NativeLatentUpdate>> {
    updates
        .iter()
        .map(|update| {
            let update = update.borrow(py);
            let params = update
                .params
                .as_ref()
                .map(|params| {
                    let params = params.bind(py);
                    Ok::<_, PyErr>(LatentParams {
                        request_key: request_key(&params.getattr("request_key")?)?,
                        call_id: call_id(&params.getattr("call_id")?)?,
                        page_table: params.getattr("page_table")?.extract()?,
                        latent_units: params.getattr("latent_units")?.extract()?,
                        height: params.getattr("height")?.extract()?,
                        width: params.getattr("width")?.extract()?,
                        start_step: params.getattr("start_step")?.extract()?,
                        step_count: params.getattr("step_count")?.extract()?,
                    })
                })
                .transpose()?;
            Ok(NativeLatentUpdate {
                request_pool_idx: update.request_pool_idx,
                params,
                expected_generation: update.expected_generation,
                expected_step: update.expected_step,
                generation: update.generation,
                step: update.step,
                release: update.release,
            })
        })
        .collect()
}

fn backend(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("uniserve_worker.storage.latent_pool")
}
