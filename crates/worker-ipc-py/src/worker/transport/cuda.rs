//! CUDA export allocation, producer readiness and failed-copy retirement.

use std::collections::HashMap;
use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyBaseException, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyBytes, PyDict};
use uniserve_worker::cuda::{DeviceGuard, Stream};

use super::{ExportView, Transport, numerical, transfer_types};
use crate::worker::descriptor_grants::DescriptorGrants;
use crate::worker::events::{CUDAEvent, current_stream};
use crate::worker::host::with_context;
use crate::worker::registry::{BufferRegistry, TransportBuffer};
use crate::worker::vmm_pool::{PoolChunk, VmmPool};

pub(super) struct CudaExports {
    pub(super) buffers: Py<BufferRegistry>,
    pub(super) slot: usize,
    cross_host: bool,
    pools: Mutex<HashMap<String, Py<VmmPool>>>,
    grants: Mutex<Option<Py<DescriptorGrants>>>,
    // A failed drain leaves device access unresolved. Retain every view and
    // descriptor, keep capacity occupied, and reject subsequent exports.
    failed: Mutex<Option<FailedExport>>,
}

struct FailedExport {
    error: Py<PyBaseException>,
    source: Py<PyAny>,
    input: Py<PyAny>,
    event: Option<Py<CUDAEvent>>,
    allocation: Option<(Py<VmmPool>, Py<PoolChunk>)>,
    grants: Option<Py<DescriptorGrants>>,
    _descriptor: Option<OwnedFd>,
}

impl CudaExports {
    pub(super) fn new(buffers: Py<BufferRegistry>, slot: usize, cross_host: bool) -> Self {
        Self {
            buffers,
            slot,
            cross_host,
            pools: Mutex::new(HashMap::new()),
            grants: Mutex::new(None),
            failed: Mutex::new(None),
        }
    }

    pub(super) fn check(&self, py: Python<'_>) -> PyResult<()> {
        match lock(py, &self.failed).as_ref() {
            Some(failed) => Err(PyErr::from_value(failed.error.bind(py).clone().into_any())),
            None => Ok(()),
        }
    }

    pub(super) fn validate(&self, view: &ExportView<'_>) -> PyResult<()> {
        numerical(view.tensor.py(), "cuda_vmm")?.call_method1("_export_spans", (&view.tensor,))?;
        Ok(())
    }

    pub(super) fn export<'py>(
        &self,
        owner: &Transport,
        view: &ExportView<'py>,
        consumers: Vec<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = view.tensor.py();
        let device = view.first.getattr("device")?;
        let mut source = view.tensor.clone();
        let mut event: Option<Py<CUDAEvent>> = None;
        let mut copied = None;
        let mut allocation: Option<(Py<VmmPool>, Py<PoolChunk>)> = None;
        let mut descriptor = None;
        let mut grants: Option<Py<DescriptorGrants>> = None;
        let mut backing: Option<Py<TransportBuffer>> = None;
        let mut submitted = false;
        let export_id = uuid::Uuid::new_v4().simple().to_string();

        let result = (|| {
            let exported: Option<(Vec<u8>, u64, u64)> = py
                .import("uniserve_kernels.peer_storage")?
                .call_method1("export_handle", (&view.first,))?
                .extract()?;
            let (handle, size, offset) = if let Some((handle, size, offset)) = exported {
                // A direct POSIX export owns this descriptor. A pool's handle
                // stays with its allocation and is never closed by an export.
                if let Ok(bytes) = <[u8; size_of::<i32>()]>::try_from(handle.as_slice()) {
                    descriptor = Some(unsafe { OwnedFd::from_raw_fd(i32::from_ne_bytes(bytes)) });
                }
                (handle, size, offset)
            } else {
                let pool = self.pool(owner, &device)?;
                let chunk = Py::new(py, pool.get().reserve(py, view.nbytes as usize)?)?;
                allocation = Some((pool.clone_ref(py), chunk.clone_ref(py)));
                let shared = chunk
                    .get()
                    .storage
                    .bind(py)
                    .call_method1("view", (view.first.getattr("dtype")?,))?
                    .call_method1("view", (&view.shape,))?;
                submitted = true;
                numerical(py, "cuda_vmm")?.call_method1("_copy_export", (&source, &shared))?;
                copied = Some(source.clone().unbind());
                source = shared;
                (
                    pool.get().handle(py)?.as_bytes().to_vec(),
                    pool.get().capacity(py)? as u64,
                    chunk.get().payload_offset() as u64,
                )
            };

            // CUDA IPC events are host-local. Cross-host exports establish
            // readiness before sending their allocation handle to readers.
            if self.cross_host {
                let profile = py
                    .import("uniserve.profiling")?
                    .call_method1("profile_range", ("vmm_export_synchronize",))?;
                with_context(&profile, || wait_stream(&device))?;
            }
            let ready = owner.record_event(&device, !self.cross_host)?;
            event = Some(ready.clone_ref(py));
            if let Ok(bytes) = <[u8; size_of::<i32>()]>::try_from(handle.as_slice()) {
                let grant = self.grants(py)?;
                let raw = i32::from_ne_bytes(bytes);
                grant.get().register(py, &export_id, raw)?;
                grants = Some(grant);
            }
            let buffer = Py::new(
                py,
                TransportBuffer::cuda(
                    source.clone().unbind(),
                    ready.clone_ref(py),
                    view.nbytes,
                    owner.capacity.clone_ref(py),
                    descriptor.take(),
                    copied.as_ref().map(|source| source.clone_ref(py)),
                    allocation
                        .as_ref()
                        .map(|(pool, chunk)| (pool.clone_ref(py), chunk.clone_ref(py))),
                    if allocation.is_some() {
                        consumers
                    } else {
                        Vec::new()
                    },
                    grants.as_ref().map(|grants| grants.clone_ref(py)),
                    &export_id,
                ),
            )?;
            backing = Some(buffer.clone_ref(py));

            // Shape and stride are numerical view metadata. Rust supplies the
            // allocation coordinates, readiness and acknowledgment lifetime.
            let args = numerical(py, "cuda_vmm")?
                .call_method1("_export_layout", (&source, offset))?
                .cast_into::<PyDict>()?;
            args.set_item("endpoint", owner.endpoint())?;
            args.set_item("export_id", &export_id)?;
            args.set_item("storage_size_bytes", size)?;
            args.set_item("allocation_handle", PyBytes::new(py, &handle))?;
            let ready_handle = if self.cross_host {
                Vec::new()
            } else {
                ready
                    .borrow(py)
                    .inner
                    .ipc_handle()
                    .map_err(PyRuntimeError::new_err)?
                    .to_vec()
            };
            args.set_item("ready_event_handle", PyBytes::new(py, &ready_handle))?;
            args.set_item(
                "acknowledgment_offset",
                allocation
                    .as_ref()
                    .map_or(-1, |(_, chunk)| chunk.get().offset() as i64),
            )?;
            let handle = transfer_types(py)?
                .getattr("CudaVmmTransfer")?
                .call((), Some(&args))?;
            let locator = view.locator(owner, handle)?;
            self.buffers
                .get()
                .register(py, locator.clone().unbind(), buffer)?;
            Ok(locator)
        })();

        if let Err(error) = result {
            if let Some(backing) = backing {
                TransportBuffer::reclaim(py, backing, &owner.events.borrow(py), None)?;
                return Err(error);
            }

            // An early failure may precede a usable event. Drain the producer
            // stream before returning its chunk, grant or byte reservation.
            let cleanup = (|| {
                if submitted || event.is_some() {
                    wait_stream(&device)?;
                }
                if let Some(event) = &event {
                    owner.events.borrow(py).defer_events(
                        vec![Arc::clone(&event.borrow(py).inner)],
                        source.clone().unbind(),
                        None,
                    )?;
                }
                Ok::<_, PyErr>(())
            })();
            if let Err(cleanup) = cleanup {
                *lock(py, &self.failed) = Some(FailedExport {
                    error: cleanup.value(py).clone().unbind(),
                    source: source.unbind(),
                    input: view.tensor.clone().unbind(),
                    event,
                    allocation,
                    grants,
                    _descriptor: descriptor,
                });
                return Err(cleanup);
            }
            if let Some(grants) = grants {
                grants.get().release(py, &export_id);
            }
            if let Some((pool, chunk)) = allocation {
                pool.get().release(py, chunk.get())?;
            }
            owner.return_bytes(py, view.nbytes)?;
            return Err(error);
        }
        result
    }

    fn pool(&self, owner: &Transport, device: &Bound<'_, PyAny>) -> PyResult<Py<VmmPool>> {
        let py = device.py();
        let key = device.str()?.to_str()?.to_owned();
        if let Some(pool) = lock(py, &self.pools).get(&key) {
            return Ok(pool.clone_ref(py));
        }
        let pool = Py::new(
            py,
            VmmPool::new(py, device, owner.capacity.get().inner.capacity() as usize)?,
        )?;
        let mut pools = lock(py, &self.pools);
        Ok(pools.entry(key).or_insert(pool).clone_ref(py))
    }

    fn grants(&self, py: Python<'_>) -> PyResult<Py<DescriptorGrants>> {
        if let Some(grants) = lock(py, &self.grants).as_ref() {
            return Ok(grants.clone_ref(py));
        }
        let grants = Py::new(py, DescriptorGrants::new(py, self.buffers.get().name())?)?;
        let mut installed = lock(py, &self.grants);
        Ok(installed.get_or_insert(grants).clone_ref(py))
    }

    fn pools(&self, py: Python<'_>) -> Vec<Py<VmmPool>> {
        lock(py, &self.pools)
            .values()
            .map(|pool| pool.clone_ref(py))
            .collect()
    }

    pub(super) fn reap(&self, py: Python<'_>) -> PyResult<()> {
        for pool in self.pools(py) {
            pool.get().reap(py)?;
        }
        Ok(())
    }

    pub(super) fn awaiting_acknowledgment(&self, py: Python<'_>) -> PyResult<bool> {
        for pool in self.pools(py) {
            if pool.get().awaiting_acknowledgment(py)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(super) fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.check(py)?;
        let mut result = Ok(());
        for pool in self.pools(py) {
            result = result.and(pool.get().close(py));
        }
        if result.is_ok() {
            let pools = std::mem::take(&mut *lock(py, &self.pools));
            drop(pools);
        }
        let grants = lock(py, &self.grants).take();
        if let Some(grants) = grants {
            result = result.and(grants.get().close(py));
        }
        result
    }

    pub(super) fn visit(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Ok(pools) = self.pools.try_lock() {
            for pool in pools.values() {
                visit.call(pool)?;
            }
        }
        if let Ok(grants) = self.grants.try_lock() {
            visit.call(&*grants)?;
        }
        if let Ok(failed) = self.failed.try_lock()
            && let Some(failed) = &*failed
        {
            visit.call(&failed.error)?;
            visit.call(&failed.source)?;
            visit.call(&failed.input)?;
            visit.call(&failed.event)?;
            visit.call(&failed.grants)?;
            if let Some((pool, chunk)) = &failed.allocation {
                visit.call(pool)?;
                visit.call(chunk)?;
            }
        }
        Ok(())
    }
}

impl Drop for CudaExports {
    #[allow(
        clippy::mem_forget,
        reason = "unresolved device access must retain its backing until process teardown"
    )]
    fn drop(&mut self) {
        if let Some(failed) = self
            .failed
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            std::mem::forget(failed);
        }
    }
}

fn lock<'a, T>(py: Python<'_>, mutex: &'a Mutex<T>) -> MutexGuard<'a, T> {
    mutex
        .lock_py_attached(py)
        .unwrap_or_else(PoisonError::into_inner)
}

fn wait_stream(device: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = device.py();
    let index = device.getattr("index")?.extract()?;
    let stream = current_stream(py, device)?;
    py.detach(|| {
        let _device = DeviceGuard::new(index)?;
        Stream::borrowed(stream).wait()
    })
    .map_err(PyRuntimeError::new_err)
}
