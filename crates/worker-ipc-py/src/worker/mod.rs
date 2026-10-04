//! Python interfaces to rank-local execution and storage.

use pyo3::prelude::*;

mod batch;
mod block_tables;
mod buffer;
mod completion;
mod error;
mod events;
mod executor;
mod exports;
mod host;
mod inputs;
mod kv_cache;
mod kv_import;
mod latent;
mod output;
mod pending;
mod protocol;
mod registry;
mod request;
mod storage;
mod transfer;

/// Register the worker objects in the common native extension.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(exports::release_exports, module)?)?;
    module.add_class::<batch::BatchState>()?;
    module.add_class::<block_tables::BlockTables>()?;
    module.add_class::<block_tables::GroupShape>()?;
    module.add_class::<block_tables::GroupTable>()?;
    module.add_class::<completion::Completion>()?;
    module.add_class::<events::CUDAEvent>()?;
    module.add_class::<events::EventPool>()?;
    module.add(
        "EventPoolError",
        module.py().get_type::<events::EventPoolError>(),
    )?;
    module.add_class::<host::HostLane>()?;
    module.add_class::<host::HostTask>()?;
    module.add_class::<inputs::BatchInputs>()?;
    module.add_class::<kv_cache::KVCacheManager>()?;
    module.add_class::<kv_import::KVImport>()?;
    module.add_class::<kv_import::KVImporter>()?;
    module.add_class::<executor::Executor>()?;
    module.add_class::<executor::Submission>()?;
    module.add_class::<buffer::BufferBinding>()?;
    module.add_class::<buffer::BufferPool>()?;
    module.add_class::<latent::LatentPool>()?;
    module.add_class::<latent::LatentImport>()?;
    module.add_class::<latent::LatentExport>()?;
    module.add_class::<latent::LatentUpdate>()?;
    module.add_class::<output::OutputBuffer>()?;
    module.add_class::<output::OutputPool>()?;
    module.add_class::<pending::PendingOutput>()?;
    module.add_class::<registry::BufferRegistry>()?;
    module.add_class::<request::Request>()?;
    module.add_class::<request::RequestPool>()?;
    module.add_class::<storage::Buffer>()?;
    module.add_class::<storage::TensorRead>()?;
    module.add_class::<storage::TensorImport>()?;
    module.add_class::<storage::TensorStore>()?;
    module.add_class::<transfer::TransferCapacity>()?;
    module.add_class::<transfer::ReadReservation>()?;
    module.add_class::<transfer::TransferTicket>()?;
    module.add_class::<transfer::TransferPool>()?;
    Ok(())
}
