//! Batch input admission, transport reads and preparation wakeups.

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use uniserve_worker::Submission as NativeSubmission;
use uniserve_worker_ipc::{
    CallKind, MediaCall, TensorRef, TensorTransfer, TransferHandle, TransferTransport,
};

use super::super::error::{invalid, native_error, resource, unsupported};
use super::super::fetch::{plan_reads, submit_reads};
use super::super::inputs::{BatchInputs, Input};
use super::super::kv_import::KVImporter;
use super::super::transfer::TransferCapacity;
use super::{BatchState, PythonBackend, Submission};

impl PythonBackend {
    /// Resume at the first refused import. Accepted imports retain their
    /// destinations until consumption or batch cleanup, even while waiting
    /// for another import's read tickets.
    fn prepare_products(&self, py: Python<'_>, batch: &BatchState) -> PyResult<()> {
        let plan = &batch.plan;
        let start = batch.inputs.borrow(py).started();
        if start == plan.input_products.len() {
            return Ok(());
        }

        let transports = self.worker.bind(py).getattr("transports")?;
        let transports = transports.cast::<PyDict>()?;
        if transports.is_empty() {
            return Err(unsupported(
                py,
                "cross-call input requires a configured transport",
            ));
        }

        let numerical = batch.numerical.borrow(py).batch.clone_ref(py);
        let numerical = numerical.bind(py);
        let products = numerical.getattr("input_products")?;
        let calls = numerical.getattr("calls")?;
        let allocations = PyDict::new(py);
        for allocation in numerical.getattr("buffer_allocations")?.try_iter()? {
            let allocation = allocation?;
            allocations.set_item(allocation.getattr("buffer")?, allocation)?;
        }

        // A product may feed several calls, but it has one destination device.
        // Resolve placement once per consumer, including predicate-only uses.
        let mut consumers: HashMap<&TensorRef, Vec<usize>> = HashMap::new();
        for (index, call) in plan.calls.iter().enumerate() {
            for product in call.tensor_inputs().chain(call.predicate.as_ref()) {
                let indices = consumers.entry(product).or_default();
                if indices.last() != Some(&index) {
                    indices.push(index);
                }
            }
        }
        let mut devices: HashMap<usize, Bound<'_, PyAny>> = HashMap::new();
        let has_shm = transports.contains("shm")?;

        for (index, export) in plan.input_products.iter().enumerate().skip(start) {
            batch.inputs.borrow_mut(py).set_started(index);
            let product = &export.product;
            let tensor = &export.value.tensors()[0];
            let consumers = &consumers[product];
            let borrowed = consumers.iter().any(|&index| {
                let call = &plan.calls[index];
                call.inputs.contains(product)
                    && (call.code == CallKind::Media(MediaCall::Muxing)
                        || (call.code == CallKind::Media(MediaCall::VideoEncoding)
                            && has_shm
                            && !matches!(export.value, TransferHandle::Latent { .. })
                            && tensor.locations.iter().any(|location| {
                                matches!(location.transport, TransferTransport::PosixShm { .. })
                                    && location.source.node == self.info.endpoint.node
                            })))
            });
            if borrowed {
                // Media codecs read local segments and encoded row prefixes
                // in place. Importing their capacity would read unused padding.
                batch
                    .inputs
                    .borrow_mut(py)
                    .insert(product.buffer_id(), Input::Borrowed);
                continue;
            }

            for &index in consumers {
                if let std::collections::hash_map::Entry::Vacant(value) = devices.entry(index) {
                    value.insert(
                        self.model_runner
                            .bind(py)
                            .call_method1("call_devices", (calls.get_item(index)?,))?
                            .get_item(0)?,
                    );
                }
            }
            // IPC admission requires a consumer for every supplied input.
            let device = devices[&consumers[0]].clone();
            for index in &consumers[1..] {
                if !device.eq(&devices[index])? {
                    return Err(invalid(
                        py,
                        "transferred product requires one consumer device per batch",
                    ));
                }
            }

            let view = products.get_item(index)?;
            let reference = view.getattr("product")?;
            let tensor_view = view.getattr("value")?.getattr("tensor")?;
            let bindings = self.input_transports(py, tensor, &tensor_view, transports)?;
            let slot = self
                .requests
                .borrow(py)
                .pool
                .peek(product.request_key.request_id.0)
                .filter(|request| request.key() == product.request_key)
                .map(|request| request.slot());

            if let TransferHandle::Latent {
                height,
                width,
                latent_units,
                step,
                ..
            } = &export.value
            {
                if consumers.len() != 1 {
                    return Err(invalid(py, "latent transfer must have one consumer"));
                }
                let consumer = &plan.calls[consumers[0]];
                let params = plan
                    .latent_params
                    .iter()
                    .find(|params| {
                        params.request_key == consumer.request_key
                            && params.call_id == consumer.call_id
                    })
                    .ok_or_else(|| invalid(py, "latent transfer has no scheduler parameters"))?;
                if (*latent_units, *height, *width, *step)
                    != (
                        params.latent_units,
                        params.height,
                        params.width,
                        params.start_step,
                    )
                {
                    return Err(invalid(
                        py,
                        "latent transfer disagrees with its scheduler parameters",
                    ));
                }
                let pool = self.latents.as_ref().ok_or_else(|| {
                    invalid(py, "latent transfer requires physical latent storage")
                })?;
                pool.borrow(py)
                    .validate_transfer(py, tensor, *latent_units)?;
                let slot =
                    slot.ok_or_else(|| invalid(py, "latent transfer has no request slot"))?;
                let write = pool.borrow_mut(py).reserve_import(
                    py,
                    reference.unbind(),
                    slot as i64,
                    params
                        .page_table
                        .iter()
                        .map(|&page| i64::from(page))
                        .collect(),
                    i64::from(*latent_units),
                )?;
                batch
                    .inputs
                    .borrow_mut(py)
                    .insert(product.buffer_id(), Input::Latent(write.clone_ref(py)));

                // Register the destination before submission: any partial
                // transport failure leaves physical reads with their owner.
                let reads = plan_reads(
                    py,
                    &tensor_view,
                    write.get().spans.bind(py).as_any(),
                    bindings.as_any(),
                    &tensor.shape.iter().map(|&size| 0..size).collect::<Vec<_>>(),
                )?;
                if let Err(error) = submit_reads(py, &reads, |ticket| {
                    pool.borrow(py)
                        .retain_transfer(py, write.bind(py), ticket.clone_ref(py))
                }) {
                    if error.is_instance(py, self.read_backpressure.bind(py)) {
                        // Capacity refusal starts no reads. Release these
                        // pages so the same input can be admitted on resumption.
                        batch.inputs.borrow_mut(py).remove(product.buffer_id());
                        pool.borrow_mut(py).abandon_import(py, write.bind(py))?;
                    }
                    return Err(error);
                }
            } else {
                let slots = PyDict::new(py);
                if let Some(slot) = slot {
                    slots.set_item(reference.getattr("request_key")?, slot)?;
                }
                let metadata = py.import("uniserve_worker.storage.tensor_store")?;
                let metadata = match &export.value {
                    TransferHandle::Encoder { height, width, .. } => Some(
                        metadata
                            .getattr("FeatureMetadata")?
                            .call1((*height, *width))?,
                    ),
                    TransferHandle::DeviceProduct {
                        height,
                        width,
                        value_range,
                        ..
                    } if *height != 0 => {
                        let range = match value_range.as_str() {
                            "signed_unit" => Some((-1.0, 1.0)),
                            "unit" => Some((0.0, 1.0)),
                            _ => None,
                        };
                        Some(
                            metadata
                                .getattr("ImageMetadata")?
                                .call1((*height, *width, range))?,
                        )
                    }
                    _ => None,
                };
                let read = self.tensors.get().import_tensor(
                    py,
                    reference,
                    tensor_view,
                    device,
                    bindings.into_any(),
                    slots.into_any(),
                    allocations.clone().into_any(),
                    metadata,
                )?;
                batch
                    .inputs
                    .borrow_mut(py)
                    .insert(product.buffer_id(), Input::Tensor(read));
            }
        }
        batch
            .inputs
            .borrow_mut(py)
            .set_started(plan.input_products.len());
        Ok(())
    }

    /// Bind available transports; a POSIX segment is addressable only on its
    /// producer's node. Remote inputs use their channel or device locations.
    fn input_transports<'py>(
        &self,
        py: Python<'py>,
        tensor: &TensorTransfer,
        view: &Bound<'py, PyAny>,
        transports: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let bindings = PyDict::new(py);
        let locations = view.getattr("locations")?;
        for (index, location) in tensor.locations.iter().enumerate() {
            if matches!(location.transport, TransferTransport::PosixShm { .. })
                && location.source.node != self.info.endpoint.node
            {
                continue;
            }
            let view = locations.get_item(index)?;
            let backend = view.getattr("backend")?;
            if let Some(transport) = transports.get_item(&backend)? {
                bindings.set_item((view.getattr("source")?, backend), transport)?;
            }
        }
        Ok(bindings)
    }

    pub(super) fn advance_inputs(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        submission: &Arc<NativeSubmission>,
    ) -> PyResult<()> {
        let inputs = &batch.inputs;
        if inputs.borrow(py).closed() {
            return Ok(());
        }

        if !inputs.borrow(py).submitted() {
            // Reusing an import destination must wait for its previous
            // reader. Batches without imports can prepare numerical inputs
            // while those same dependencies still gate model execution.
            if batch.imports {
                if !inputs.borrow(py).storage_ready()? {
                    return Ok(());
                }
                inputs.borrow(py).require_storage(py)?;
            }

            inputs.borrow_mut(py).set_awaiting_reads(false);

            // Only product admission can resume after read capacity returns.
            // Later phases may already have submitted non-repeatable work.
            let mut resumable = true;
            let prepared: PyResult<()> = (|| {
                self.prepare_products(py, batch)?;
                resumable = false;
                self.prepare_kv(py, batch)?;
                self.prepare_predicates(py, batch)?;
                Ok(())
            })();
            if let Err(error) = prepared {
                let backpressure = error.is_instance(py, self.read_backpressure.bind(py));
                if !resumable || !backpressure {
                    let error = if backpressure {
                        resource(py, error.to_string())
                    } else {
                        error
                    };
                    // Both numerical preparation and native imports use this
                    // cleanup path. Abandon every accepted destination before
                    // propagating the error; physical accesses retire later.
                    if let Err(cleanup) = BatchInputs::close(
                        inputs.bind(py),
                        self.tensors.get(),
                        self.latents.as_ref().map(|pool| pool.bind(py)),
                        self.cache_imports.as_ref().map(|imports| imports.bind(py)),
                    ) {
                        let _ = error.value(py).call_method1(
                            "add_note",
                            (format!("batch input cleanup failed: {cleanup}"),),
                        );
                    }
                    return Err(error);
                }

                let error = error.value(py);
                let capacity: Py<TransferCapacity> = error.getattr("capacity")?.extract()?;
                let returns = error.getattr("returns")?.extract()?;
                inputs.borrow_mut(py).set_awaiting_reads(true);

                // The return counter closes the gap between refusal and
                // subscription, including a return on another host thread.
                capacity.get().notify_reads_returned(
                    py,
                    Self::input_wake(py, submission)?,
                    returns,
                )?;
                return Ok(());
            }
            inputs.borrow_mut(py).set_submitted(true);
        }

        // Numerical callbacks may acquire or close inputs. Never retain an
        // input-set borrow across a callback or notification registration.
        self.capture_predicates(py, batch)?;
        Ok(())
    }

    pub(super) fn input_wake(
        py: Python<'_>,
        submission: &Arc<NativeSubmission>,
    ) -> PyResult<Py<PyAny>> {
        Py::new(
            py,
            Submission {
                submission: Arc::clone(submission),
            },
        )?
        .bind(py)
        .getattr("notify_ready")
        .map(Bound::unbind)
    }

    /// Start KV copies after storage dependencies and tensor input admission.
    /// Record each accepted import immediately so failure of a later import
    /// releases earlier destinations through the same batch input owner.
    fn prepare_kv(&self, py: Python<'_>, batch: &BatchState) -> PyResult<()> {
        let plan = &batch.plan;
        if plan.kv_inputs.is_empty() && batch.resident_kv.is_empty() {
            return Ok(());
        }

        let transports = self.worker.bind(py).getattr("transports")?;
        if !transports.is_truthy()? {
            return Err(unsupported(
                py,
                "cross-call input requires a configured transport",
            ));
        }
        let (imports, tables) = self
            .cache_imports
            .as_ref()
            .zip(self.tables.as_ref())
            .ok_or_else(|| invalid(py, "KV input requires physical cache storage"))?;

        // IPC admission checks supplied descriptors. Resident exports are
        // resolved afterward and must also have one installation consumer.
        if !batch.resident_kv.is_empty() {
            let mut consumers = HashMap::new();
            for call in &plan.calls {
                if let Some(source) = call.kv_input {
                    *consumers.entry(source).or_insert(0) += 1;
                }
            }
            if batch
                .resident_kv
                .iter()
                .any(|transfer| consumers.get(&transfer.source) != Some(&1))
            {
                return Err(invalid(py, "KV input requires one installation consumer"));
            }
        }

        let transfers = plan
            .kv_inputs
            .iter()
            .cloned()
            .map(Arc::new)
            .chain(batch.resident_kv.iter().cloned());
        for transfer in transfers {
            let source = transfer.source;
            // Starts were installed before dependency resolution. The native
            // request pool is the sole source of a batch consumer's slot.
            let slot = self
                .requests
                .borrow(py)
                .pool
                .get(source.owner.request_id.0)
                .map_err(|error| native_error(py, error))?
                .slot() as u32;
            let tables = tables
                .borrow(py)
                .tables
                .for_batch(slot, &plan.block_tables)
                .map_err(|error| native_error(py, error))?;
            let initialized = plan
                .new_cache_units
                .iter()
                .filter(|allocation| allocation.request_pool_idx == slot)
                .flat_map(|allocation| allocation.unit_ids.iter().map(|unit| unit.0))
                .collect();
            let write = KVImporter::reserve(
                imports.bind(py),
                transfer,
                slot,
                tables,
                initialized,
                transports.clone().unbind(),
            )?;
            batch
                .inputs
                .borrow_mut(py)
                .insert(source, Input::Cache(write));
        }
        Ok(())
    }
}
