//! Python interfaces to rank-local execution and storage.

use pyo3::prelude::*;

mod block_tables;
mod buffer;
mod completion;
mod error;
mod executor;
mod kv_cache;
mod latent;
mod protocol;
mod registry;
mod request;
mod storage;
mod transfer;

/// Register the worker objects in the common native extension.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<block_tables::BlockTables>()?;
    module.add_class::<block_tables::GroupShape>()?;
    module.add_class::<block_tables::GroupTable>()?;
    module.add_class::<completion::Completion>()?;
    module.add_class::<kv_cache::KVCacheManager>()?;
    module.add_class::<executor::Executor>()?;
    module.add_class::<executor::Submission>()?;
    module.add_class::<buffer::BufferBinding>()?;
    module.add_class::<buffer::BufferPool>()?;
    module.add_class::<latent::LatentPool>()?;
    module.add_class::<latent::LatentImport>()?;
    module.add_class::<latent::LatentExport>()?;
    module.add_class::<latent::LatentUpdate>()?;
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
