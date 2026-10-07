//! Stream-ordered NCCL operations and registered allocation lifetime.

mod ffi;

use std::sync::Arc;

use indexmap::IndexMap;

use crate::CUDAStream;
use crate::cuda::{DeviceGuard, Event};
use ffi::{Handle, api, check};

/// A communicator bound to one computation stream. Its transfer stream shares
/// the computation's SM partition; consumers join asynchronous work by event.
/// Registered backing lives through communicator retirement, independently of
/// the execution contexts and graphs borrowing it.
pub struct NcclCommunicator<T> {
    device: i32,
    resources: Option<Resources<T>>,
}

struct Resources<T> {
    handle: usize,
    computation: usize,
    transfer: CUDAStream,
    pending: Option<Arc<Event>>,
    windows: IndexMap<(usize, usize), (usize, T)>,
}

impl<T> NcclCommunicator<T> {
    pub fn unique_id() -> Result<[u8; 128], String> {
        let mut id = ffi::UniqueId { bytes: [0; 128] };
        unsafe {
            check((api()?.unique_id)(&mut id), "ncclGetUniqueId")?;
        }
        Ok(id.bytes)
    }

    /// Every group member calls this once, with the same distributed unique ID.
    pub fn new(owner: &CUDAStream, size: i32, rank: i32, id: [u8; 128]) -> Result<Self, String> {
        let device = owner.device();
        let _device = DeviceGuard::new(device)?;
        let mut handle = std::ptr::null_mut();
        let mut config = ffi::Config::new(owner.full_device());
        let created = unsafe {
            check(
                (api()?.init)(
                    &mut handle,
                    size,
                    ffi::UniqueId { bytes: id },
                    rank,
                    &mut config,
                ),
                "ncclCommInitRankConfig",
            )
        };

        if let Err(error) = created {
            if !handle.is_null() {
                let _ = unsafe { (api()?.abort)(handle) };
            }
            return Err(error);
        }

        let transfer = match owner.fork() {
            Ok(stream) => stream,
            Err(error) => {
                let _ = unsafe { (api()?.abort)(handle) };
                return Err(error);
            }
        };

        Ok(Self {
            device,
            resources: Some(Resources {
                handle: handle as usize,
                computation: owner.handle()?,
                transfer,
                pending: None,
                windows: IndexMap::new(),
            }),
        })
    }

    pub fn transfer_stream(&self) -> Result<usize, String> {
        self.resources()?.transfer.handle()
    }

    fn resources(&self) -> Result<&Resources<T>, String> {
        self.resources
            .as_ref()
            .ok_or_else(|| "computation collective is closed".into())
    }

    fn resources_mut(&mut self) -> Result<&mut Resources<T>, String> {
        self.resources
            .as_mut()
            .ok_or_else(|| "computation collective is closed".into())
    }

    fn arguments(&mut self, asynchronous: bool) -> Result<(Handle, Handle), String> {
        let resources = self.resources_mut()?;
        let stream = if asynchronous {
            resources.transfer.handle()?
        } else {
            // A projection can invoke another collective before consuming its
            // gathered tail. Keep both streams on the same communicator order.
            if let Some(pending) = resources.pending.take() {
                pending.wait_on(resources.computation)?;
            }
            resources.computation
        };
        Ok((resources.handle as Handle, stream as Handle))
    }

    fn start(
        &mut self,
        call: impl FnOnce(&mut Self) -> Result<(), String>,
    ) -> Result<Arc<Event>, String> {
        let resources = self.resources()?;
        resources.transfer.wait(resources.computation)?;
        let completed = Arc::new(Event::new(self.device, false, false));
        let result = call(self);
        let resources = self.resources_mut()?;
        completed.record(resources.transfer.handle()?)?;
        if let Err(error) = result {
            // Join partial submissions before the caller can retire inputs.
            completed.wait_on(resources.computation)?;
            return Err(error);
        }
        resources.pending = Some(Arc::clone(&completed));
        Ok(completed)
    }

    pub fn join(&mut self, event: &Arc<Event>, consumer: usize) -> Result<(), String> {
        event.wait_on(consumer)?;
        let resources = self.resources_mut()?;
        if consumer == resources.computation
            && resources
                .pending
                .as_ref()
                .is_some_and(|pending| Arc::ptr_eq(pending, event))
        {
            // A completed eager join must not leak an event dependency into
            // a subsequent, independent graph capture.
            resources.pending = None;
        }
        Ok(())
    }

    /// # Safety
    /// Input and output spans must hold the declared elements on this device
    /// and remain alive through the submitted operation and graph replays.
    pub unsafe fn all_gather(
        &mut self,
        output: usize,
        input: usize,
        count: usize,
        dtype: i32,
        asynchronous: bool,
    ) -> Result<Option<Arc<Event>>, String> {
        let submit = |owner: &mut Self| {
            let (comm, stream) = owner.arguments(asynchronous)?;
            unsafe {
                check(
                    (api()?.all_gather)(
                        input as Handle,
                        output as Handle,
                        count,
                        dtype,
                        comm,
                        stream,
                    ),
                    "ncclAllGather",
                )
            }
        };

        if asynchronous {
            self.start(submit).map(Some)
        } else {
            submit(self).map(|()| None)
        }
    }

    /// # Safety
    /// Each span holds `count` elements per rank and outlives all device uses.
    pub unsafe fn all_to_all(
        &mut self,
        output: usize,
        input: usize,
        count: usize,
        dtype: i32,
        asynchronous: bool,
    ) -> Result<Option<Arc<Event>>, String> {
        let submit = |owner: &mut Self| {
            let (comm, stream) = owner.arguments(asynchronous)?;
            unsafe {
                check(
                    (api()?.all_to_all)(
                        input as Handle,
                        output as Handle,
                        count,
                        dtype,
                        comm,
                        stream,
                    ),
                    "ncclAlltoAll",
                )
            }
        };

        if asynchronous {
            self.start(submit).map(Some)
        } else {
            submit(self).map(|()| None)
        }
    }

    /// # Safety
    /// The in-place tensor holds `count` elements and outlives device uses.
    pub unsafe fn all_reduce(
        &mut self,
        data: usize,
        count: usize,
        dtype: i32,
        op: i32,
    ) -> Result<(), String> {
        let (comm, stream) = self.arguments(false)?;
        unsafe {
            check(
                (api()?.all_reduce)(
                    data as Handle,
                    data as Handle,
                    count,
                    dtype,
                    op,
                    comm,
                    stream,
                ),
                "ncclAllReduce",
            )
        }
    }

    /// # Safety
    /// The in-place tensor holds `count` elements and outlives device uses.
    pub unsafe fn broadcast(
        &mut self,
        data: usize,
        count: usize,
        dtype: i32,
        root: i32,
    ) -> Result<(), String> {
        let (comm, stream) = self.arguments(false)?;
        unsafe {
            check(
                (api()?.broadcast)(
                    data as Handle,
                    data as Handle,
                    count,
                    dtype,
                    root,
                    comm,
                    stream,
                ),
                "ncclBroadcast",
            )
        }
    }

    /// # Safety
    /// Input holds `count` elements per rank and output holds `count` elements.
    /// Both remain allocated through device reads and writes.
    pub unsafe fn reduce_scatter(
        &mut self,
        output: usize,
        input: usize,
        count: usize,
        dtype: i32,
    ) -> Result<(), String> {
        let (comm, stream) = self.arguments(false)?;
        unsafe {
            check(
                (api()?.reduce_scatter)(
                    input as Handle,
                    output as Handle,
                    count,
                    dtype,
                    0,
                    comm,
                    stream,
                ),
                "ncclReduceScatter",
            )
        }
    }

    /// # Safety
    /// The span holds `count` elements until its transfer and graph uses finish.
    pub unsafe fn send(
        &mut self,
        data: usize,
        count: usize,
        dtype: i32,
        peer: i32,
    ) -> Result<(), String> {
        let (comm, stream) = self.arguments(false)?;
        unsafe {
            check(
                (api()?.send)(data as Handle, count, dtype, peer, comm, stream),
                "ncclSend",
            )
        }
    }

    /// # Safety
    /// The destination holds `count` elements until its transfer and graph uses finish.
    pub unsafe fn recv(
        &mut self,
        data: usize,
        count: usize,
        dtype: i32,
        peer: i32,
    ) -> Result<(), String> {
        let (comm, stream) = self.arguments(false)?;
        unsafe {
            check(
                (api()?.recv)(data as Handle, count, dtype, peer, comm, stream),
                "ncclRecv",
            )
        }
    }

    /// Group matching point-to-point operations into one NCCL submission.
    pub fn grouped<R>(
        &mut self,
        call: impl FnOnce(&mut Self) -> Result<R, String>,
    ) -> Result<R, String> {
        ffi::grouped(|| call(self))
    }

    /// Retain one symmetric allocation through all contexts sharing this stream.
    /// # Safety
    /// `backing` keeps the VMM allocation `[data, data + bytes)` live; every rank
    /// registers matching windows in the same order, outside graph capture.
    pub unsafe fn register(&mut self, data: usize, bytes: usize, backing: T) -> Result<(), String> {
        let resources = self.resources_mut()?;
        if resources.windows.contains_key(&(data, bytes)) {
            return Ok(());
        }
        let mut window = std::ptr::null_mut();
        unsafe {
            check(
                (api()?.register)(
                    resources.handle as Handle,
                    data as Handle,
                    bytes,
                    &mut window,
                    0x01,
                ),
                "ncclCommWindowRegister",
            )?;
        }
        resources
            .windows
            .insert((data, bytes), (window as usize, backing));
        Ok(())
    }

    /// Close collectively after all borrowed graph and tensor accesses retire.
    /// Aborted execution retains handles, mappings and streams until process exit.
    #[allow(clippy::mem_forget)]
    pub fn close(&mut self, aborted: bool) -> Result<(), String> {
        let Some(mut resources) = self.resources.take() else {
            return Ok(());
        };
        if aborted {
            std::mem::forget(resources);
            return Ok(());
        }

        let result = (|| {
            resources.transfer.synchronize()?;
            let _device = DeviceGuard::new(self.device)?;
            for (window, _) in resources.windows.values() {
                unsafe {
                    check(
                        (api()?.deregister)(resources.handle as Handle, *window as Handle),
                        "ncclCommWindowDeregister",
                    )?;
                }
            }
            unsafe {
                check(
                    (api()?.destroy)(resources.handle as Handle),
                    "ncclCommDestroy",
                )?;
            }
            resources.transfer.close(false)
        })();

        if result.is_err() {
            std::mem::forget(resources);
        }
        result
    }

    /// Visit backing references held by a foreign-language resource owner.
    pub fn backings(&self) -> impl Iterator<Item = &T> {
        self.resources
            .iter()
            .flat_map(|resources| resources.windows.values().map(|(_, owner)| owner))
    }
}

impl<T> Drop for NcclCommunicator<T> {
    fn drop(&mut self) {
        // Implicit destruction cannot enter a collective that peers may never
        // reach. Normal callers explicitly close after retiring graph readers.
        let _ = self.close(true);
    }
}
