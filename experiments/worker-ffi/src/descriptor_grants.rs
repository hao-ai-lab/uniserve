//! Native descriptor ownership and local grants through TVM-FFI.

use std::os::fd::IntoRawFd;
use std::sync::Arc;

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::{Function, Object, ObjectArc, Result, String};

use crate::execution::failure;
use crate::{method, object};

#[repr(C)]
#[derive(Object)]
#[type_key = "uniserve.ffi.DescriptorGrants"]
pub struct DescriptorGrantsObj {
    object: Object,
    grants: Arc<uniserve_worker::DescriptorGrants>,
}

#[derive(Clone, ObjectRef)]
pub struct DescriptorGrants {
    data: ObjectArc<DescriptorGrantsObj>,
}

impl DescriptorGrants {
    pub(crate) fn native(&self) -> Arc<uniserve_worker::DescriptorGrants> {
        Arc::clone(&self.data.grants)
    }
}

pub fn fetch_descriptor(endpoint: String, export: String) -> Result<i32> {
    uniserve_worker::fetch_descriptor(endpoint.as_str(), export.as_str())
        .map(IntoRawFd::into_raw_fd)
        .map_err(|error| failure(error.to_string()))
}

pub fn register() -> Result<()> {
    object::<DescriptorGrantsObj>();
    method::<DescriptorGrantsObj>(
        "__ffi_init__",
        Function::from_typed(|endpoint: String| -> Result<DescriptorGrants> {
            let grants = uniserve_worker::DescriptorGrants::new(endpoint.as_str())
                .map_err(|error| failure(error.to_string()))?;
            Ok(DescriptorGrants {
                data: ObjectArc::new(DescriptorGrantsObj {
                    object: Object::new(),
                    grants: Arc::new(grants),
                }),
            })
        }),
        "Serve local allocation descriptor grants on a native thread.",
    )?;
    method::<DescriptorGrantsObj>(
        "register",
        Function::from_typed(|owner: DescriptorGrants, export: String, fd: i32| {
            owner
                .data
                .grants
                .register(export.as_str(), fd)
                .map_err(|error| failure(error.to_string()))
        }),
        "Retain an allocation descriptor until revocation.",
    )?;
    method::<DescriptorGrantsObj>(
        "release",
        Function::from_typed(|owner: DescriptorGrants, export: String| -> Result<()> {
            owner.data.grants.release(export.as_str());
            Ok(())
        }),
        "Refuse later requests while received descriptors remain usable.",
    )?;
    method::<DescriptorGrantsObj>(
        "close",
        Function::from_typed(|owner: DescriptorGrants| {
            owner
                .data
                .grants
                .close()
                .map_err(|error| failure(error.to_string()))
        }),
        "Revoke descriptors and join the native service thread.",
    )?;
    Ok(())
}
