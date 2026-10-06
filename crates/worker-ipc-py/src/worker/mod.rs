//! Python interfaces to rank-local execution and storage.

use pyo3::prelude::*;

mod batch;
mod block_tables;
mod buffer;
mod completion;
mod descriptor_grants;
pub(crate) mod error;
mod events;
mod execution;
mod executor;
mod expert_exchange;
mod exports;
mod fetch;
mod graph_shapes;
mod graph_storage;
mod host;
mod host_buffers;
mod inputs;
mod kv_cache;
mod kv_import;
mod latent;
pub(crate) mod locator;
mod microbatches;
mod model_results;
mod model_runners;
mod output;
mod pending;
pub(crate) mod protocol;
mod registry;
mod request;
mod sampling;
mod shared_buffer;
mod storage;
mod stream;
mod transfer;
mod transport;
mod vmm_pool;
mod weight_prefetch;

/// Register the worker objects in the common native extension.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<sampling::SamplingMetadata>()?;
    module.add_function(wrap_pyfunction!(sampling::sample, module)?)?;
    module.add_function(wrap_pyfunction!(sampling::sample_graph, module)?)?;
    module.add(
        "TOKEN_CONTINUATION_BIT",
        uniserve_worker::TOKEN_CONTINUATION_BIT,
    )?;
    module.add("TOKEN_VALUE_MASK", uniserve_worker::TOKEN_VALUE_MASK)?;
    module.add_class::<locator::Locator>()?;
    module.add_function(wrap_pyfunction!(pending::store_media_bytes, module)?)?;
    module.add_class::<descriptor_grants::DescriptorGrants>()?;
    module.add_function(wrap_pyfunction!(
        descriptor_grants::fetch_descriptor,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(fetch::fetch_tensor, module)?)?;
    module.add_function(wrap_pyfunction!(exports::export_tensor, module)?)?;
    module.add_function(wrap_pyfunction!(exports::release_exports, module)?)?;
    module.add_function(wrap_pyfunction!(
        graph_storage::graph_storage_budget_bytes,
        module
    )?)?;
    module.add_class::<graph_storage::GraphStorage>()?;
    module.add_class::<execution::Execution>()?;
    module.add_class::<execution::GraphBucket>()?;
    module.add_class::<graph_shapes::PrefillShape>()?;
    module.add_class::<graph_shapes::TextShapes>()?;
    module.add_function(wrap_pyfunction!(graph_shapes::prefill_units, module)?)?;
    module.add_function(wrap_pyfunction!(
        graph_shapes::select_prefill_captures,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(microbatches::yield_microbatch, module)?)?;
    module.add_class::<microbatches::Microbatches>()?;
    module.add_class::<model_runners::ModelRunners>()?;
    module.add_class::<crate::stats::ForwardStats>()?;
    module.add_class::<model_results::ExecutionOutput>()?;
    module.add_class::<weight_prefetch::WeightPrefetch>()?;
    module.add_class::<expert_exchange::ExpertExchange>()?;
    module.add_class::<batch::BatchState>()?;
    module.add_class::<block_tables::BlockTables>()?;
    module.add_class::<block_tables::GroupShape>()?;
    module.add_class::<block_tables::GroupTable>()?;
    module.add_class::<block_tables::TablePages>()?;
    module.add_function(wrap_pyfunction!(block_tables::table_pages, module)?)?;
    module.add_class::<completion::Completion>()?;
    module.add_class::<events::CUDAEvent>()?;
    module.add_class::<stream::CUDAStream>()?;
    module.add_class::<events::EventPool>()?;
    module.add(
        "EventPoolError",
        module.py().get_type::<events::EventPoolError>(),
    )?;
    module.add_class::<host::HostLane>()?;
    module.add_class::<host::HostTask>()?;
    module.add_class::<host_buffers::HostBuffers>()?;
    module.add_class::<inputs::BatchInputs>()?;
    module.add_class::<kv_cache::KVCacheManager>()?;
    module.add_class::<kv_import::KVImport>()?;
    module.add_class::<kv_import::KVImporter>()?;
    module.add_class::<executor::Executor>()?;
    module.add_class::<executor::Submission>()?;
    module.add_class::<buffer::BufferBinding>()?;
    module.add_class::<buffer::BufferPool>()?;
    module.add_class::<latent::LatentPool>()?;
    module.add_class::<latent::LatentBuffer>()?;
    module.add_class::<latent::LatentImport>()?;
    module.add_class::<latent::LatentExport>()?;
    module.add_class::<latent::LatentUpdate>()?;
    module.add_class::<output::OutputBuffer>()?;
    module.add_class::<output::OutputPool>()?;
    module.add_class::<pending::PendingOutput>()?;
    module.add_class::<pending::TokenUpdate>()?;
    module.add_class::<shared_buffer::SharedBuffer>()?;
    module.add_class::<vmm_pool::VmmPool>()?;
    module.add_class::<vmm_pool::PoolChunk>()?;
    module.add(
        "PoolExhaustedError",
        module.py().get_type::<vmm_pool::PoolExhaustedError>(),
    )?;
    module.add_class::<shared_buffer::SharedRead>()?;
    module.add("SHM_HEADER_BYTES", uniserve_worker::SHM_HEADER_BYTES)?;
    module.add_function(wrap_pyfunction!(shared_buffer::open_shared_memory, module)?)?;
    module.add_class::<request::Request>()?;
    module.add_class::<request::RequestProgress>()?;
    module.add_class::<request::RequestPool>()?;
    module.add_class::<storage::Buffer>()?;
    module.add_class::<storage::TensorRead>()?;
    module.add_class::<storage::TensorImport>()?;
    module.add_class::<storage::TensorStore>()?;
    module.add_class::<transfer::TransferCapacity>()?;
    module.add_class::<transfer::ReadReservation>()?;
    module.add_class::<transfer::TransferTicket>()?;
    module.add_class::<transport::Transport>()?;
    Ok(())
}
