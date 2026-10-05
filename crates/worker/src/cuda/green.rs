//! SM partition allocation through the CUDA 13 Green Context API.

use std::sync::{Arc, OnceLock};

use super::{DeviceGuard, Handle, Status, Stream, check, driver};

// CUdevResource_v1: a tag, 92 opaque bytes, and a 48-byte resource union.
// Only smCount is consumed here; the driver fills and interprets the rest.
#[repr(C)]
struct Resource {
    kind: i32,
    internal: [u8; 92],
    sm_count: u32,
    reserved: [u8; 44],
}

impl Resource {
    fn empty() -> Self {
        Self {
            kind: 0,
            internal: [0; 92],
            sm_count: 0,
            reserved: [0; 44],
        }
    }
}

struct Api {
    device_resource: unsafe extern "C" fn(i32, *mut Resource, i32) -> Status,
    green_resource: unsafe extern "C" fn(Handle, *mut Resource, i32) -> Status,
    context_resource: unsafe extern "C" fn(Handle, *mut Resource, i32) -> Status,
    split: unsafe extern "C" fn(
        *mut Resource,
        *mut u32,
        *const Resource,
        *mut Resource,
        u32,
        u32,
    ) -> Status,
    descriptor: unsafe extern "C" fn(*mut Handle, *mut Resource, u32) -> Status,
    create: unsafe extern "C" fn(*mut Handle, Handle, i32, u32) -> Status,
    destroy: unsafe extern "C" fn(Handle) -> Status,
}

fn api() -> Result<&'static Api, String> {
    static API: OnceLock<Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let library = &driver()?._library;
        unsafe {
            Ok(Api {
                device_resource: *library
                    .get(b"cuDeviceGetDevResource\0")
                    .map_err(|e| e.to_string())?,
                green_resource: *library
                    .get(b"cuGreenCtxGetDevResource\0")
                    .map_err(|e| e.to_string())?,
                context_resource: *library
                    .get(b"cuCtxGetDevResource\0")
                    .map_err(|e| e.to_string())?,
                split: *library
                    .get(b"cuDevSmResourceSplitByCount\0")
                    .map_err(|e| e.to_string())?,
                descriptor: *library
                    .get(b"cuDevResourceGenerateDesc\0")
                    .map_err(|e| e.to_string())?,
                create: *library
                    .get(b"cuGreenCtxCreate\0")
                    .map_err(|e| e.to_string())?,
                destroy: *library
                    .get(b"cuGreenCtxDestroy\0")
                    .map_err(|e| e.to_string())?,
            })
        }
    })
    .as_ref()
    .map_err(Clone::clone)
}

pub(super) fn stream_sms(stream: Handle) -> Result<u32, String> {
    let mut context = std::ptr::null_mut();
    let mut resource = Resource::empty();
    unsafe {
        check(
            (driver()?.stream_context)(stream, &mut context),
            "cuStreamGetCtx",
        )?;
        check(
            (api()?.context_resource)(context, &mut resource, 1),
            "cuCtxGetDevResource",
        )?;
    }
    Ok(resource.sm_count)
}

pub(super) struct GreenContext {
    handle: Handle,
    pub(super) sm_count: u32,
}

// SAFETY: this immutable owner never makes the green context current. Callers
// serialize context use across its streams; references retain it through the
// last stream and event, including when they move between native threads.
unsafe impl Send for GreenContext {}
unsafe impl Sync for GreenContext {}

impl GreenContext {
    fn new(device: i32, mut resource: Resource) -> Result<Self, String> {
        let api = api()?;
        let mut descriptor = std::ptr::null_mut();
        let mut handle = std::ptr::null_mut();
        unsafe {
            check(
                (api.descriptor)(&mut descriptor, &mut resource, 1),
                "cuDevResourceGenerateDesc",
            )?;
            check(
                (api.create)(&mut handle, descriptor, device, 1),
                "cuGreenCtxCreate",
            )?;
        }
        Ok(Self {
            handle,
            sm_count: resource.sm_count,
        })
    }

    fn resource(&self) -> Result<Resource, String> {
        let mut resource = Resource::empty();
        unsafe {
            check(
                (api()?.green_resource)(self.handle, &mut resource, 1),
                "cuGreenCtxGetDevResource",
            )?;
        }
        Ok(resource)
    }
}

impl Drop for GreenContext {
    fn drop(&mut self) {
        if let Ok(api) = api() {
            unsafe {
                (api.destroy)(self.handle);
            }
        }
    }
}

fn split(resource: &Resource, count: u32) -> Result<(Resource, Resource), String> {
    let mut group = Resource::empty();
    let mut remainder = Resource::empty();
    let mut groups = 1;
    unsafe {
        check(
            (api()?.split)(&mut group, &mut groups, resource, &mut remainder, 0, count),
            "cuDevSmResourceSplitByCount",
        )?;
    }
    // CUDA may round minCount up. Lane capacities require the requested
    // allocation, so an unrepresentable budget is reported before execution.
    if groups != 1 || group.sm_count != count {
        return Err(format!(
            "CUDA cannot allocate exactly {count} SMs (received {})",
            group.sm_count
        ));
    }
    Ok((group, remainder))
}

pub(super) fn partition(device: i32, counts: &[u32]) -> Result<Vec<Stream>, String> {
    if counts.is_empty() {
        return Ok(Vec::new());
    }
    if counts.contains(&0) {
        return Err("stream SM counts must be positive".into());
    }

    let requested = counts
        .iter()
        .try_fold(0u32, |total, count| total.checked_add(*count))
        .ok_or("stream SM budget exceeds the device limit")?;

    let _device = DeviceGuard::new(device)?;
    let mut resource = Resource::empty();
    unsafe {
        check(
            (api()?.device_resource)(device, &mut resource, 1),
            "cuDeviceGetDevResource",
        )?;
    }

    if requested > resource.sm_count {
        return Err(format!(
            "stream SM budgets require {requested} SMs but the device exposes {}",
            resource.sm_count
        ));
    }

    let (aggregate, _) = split(&resource, requested)?;
    let mut parents = vec![GreenContext::new(device, aggregate)?];
    let mut streams = Vec::with_capacity(counts.len());

    for (index, count) in counts.iter().enumerate() {
        let resource = parents[index].resource()?;
        let (lane, remainder) = if index + 1 == counts.len() {
            (resource, None)
        } else {
            let (lane, remainder) = split(&resource, *count)?;
            (lane, Some(remainder))
        };
        let green = Arc::new(GreenContext::new(device, lane)?);
        let mut handle = std::ptr::null_mut();
        unsafe {
            check(
                (driver()?.create_green_stream)(&mut handle, green.handle, 1, 0),
                "cuGreenCtxStreamCreate",
            )?;
        }
        streams.push(Stream {
            handle,
            owned: true,
            green: Some(green),
        });

        if let Some(remainder) = remainder {
            // A split result is used for another split only after CUDA has
            // materialized it as a context resource. All lanes share one tree.
            parents.push(GreenContext::new(device, remainder)?);
        }
    }
    Ok(streams)
}
