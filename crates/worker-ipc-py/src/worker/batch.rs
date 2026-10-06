//! Batch-owned inputs, output rows and borrowed numerical views.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker_ipc::{Batch, CallKind, CallStatus, DType, MediaCall, TransferHandle};

use super::block_tables::BlockTables;
use super::completion::CompletionRef;
use super::error::{invalid, native_error};
use super::inputs::{BatchInputs, Input};
use super::kv_cache::KVCacheManager;
use super::kv_import::KVImporter;
use super::latent::LatentPool;
use super::output::OutputBuffer;
use super::pending::PendingOutput;
use super::request::RequestPool;
use super::storage::{TensorRead, TensorStore};

/// Resources retained until the executor delivers or abandons one submission.
/// Numerical callbacks borrow this object; they do not advance the executor.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct BatchState {
    plan: Arc<Batch>,
    request_indexes: HashMap<u64, usize>,
    pub(super) outputs: Vec<Py<PendingOutput>>,
    pub(super) buffer: Option<Py<OutputBuffer>>,

    #[pyo3(get)]
    pub(super) batch: Py<PyAny>,
    #[pyo3(get)]
    pub(super) inputs: Py<BatchInputs>,

    // Rows name native call indices and their readback spans. A missing span
    // waits for an imported source; all rows are filled before sealing.
    pub(super) predicate_captures: Vec<(usize, Option<(usize, usize)>)>,

    #[pyo3(get, set)]
    pub(super) stream: Option<Py<PyAny>>,
    // Python perf_counter_ns at reservation, shared with numerical timers.
    #[pyo3(get)]
    pub(super) started_ns: u64,

    pub(super) forward_stats: uniserve_worker_ipc::ForwardStats,
    #[pyo3(get)]
    pub(super) component_us: Py<PyDict>,
    pub(super) forward_indices: Vec<Vec<usize>>,
}

impl BatchState {
    pub(super) fn new(py: Python<'_>, batch: Py<PyAny>, plan: Arc<Batch>) -> PyResult<Self> {
        let request_indexes = plan
            .calls
            .iter()
            .enumerate()
            .map(|(index, call)| (call.request_key.request_id.0, index))
            .collect();

        let mut forward_indices = vec![Vec::new(); plan.calls.len()];
        for (row, &call) in plan.forward.call_indices.iter().enumerate() {
            forward_indices[call as usize].push(row);
        }

        Ok(Self {
            plan,
            request_indexes,
            outputs: Vec::new(),
            buffer: None,
            batch,
            inputs: Py::new(py, BatchInputs::new())?,
            predicate_captures: Vec::new(),
            stream: None,
            started_ns: 0,
            forward_stats: uniserve_worker_ipc::ForwardStats::default(),
            component_us: PyDict::new(py).unbind(),
            forward_indices,
        })
    }

    pub(super) fn close(
        slf: &Bound<'_, Self>,
        tensors: &TensorStore,
        latents: Option<&Bound<'_, LatentPool>>,
        imports: Option<&Bound<'_, KVImporter>>,
    ) -> PyResult<()> {
        let py = slf.py();
        let (inputs, outputs) = {
            let this = slf.borrow();
            let outputs = this
                .outputs
                .iter()
                .map(|output| output.clone_ref(py))
                .collect::<Vec<_>>();
            (this.inputs.clone_ref(py), outputs)
        };

        // Completion observers can reenter numerical consumers. Release the
        // batch borrow before closing, and attempt every owner after failures.
        let mut failure = BatchInputs::close(inputs.bind(py), tensors, latents, imports).err();
        for output in outputs {
            if let Err(error) = PendingOutput::abandon(output.bind(py)) {
                if let Some(first) = &failure {
                    let _ = first.value(py).call_method1(
                        "add_note",
                        (format!("batch output cleanup failed: {error}"),),
                    );
                } else {
                    failure = Some(error);
                }
            }
        }
        failure.map_or(Ok(()), Err)
    }
}

impl BatchState {
    /// Complete ready inputs for active numerical consumers. The executor
    /// establishes readiness; owners order device fences and propagate failures.
    pub(super) fn complete_inputs(
        slf: &Bound<'_, Self>,
        tensors: &TensorStore,
        latents: Option<&Bound<'_, LatentPool>>,
        cache: Option<&Bound<'_, KVCacheManager>>,
    ) -> PyResult<()> {
        let py = slf.py();
        let (plan, inputs) = {
            let this = slf.borrow();
            (Arc::clone(&this.plan), this.inputs.clone_ref(py))
        };

        // KV descriptors can have become resident since read admission. Check
        // those mutable owners before exposing any imported tensor or latent.
        let cache_inputs: HashSet<_> = plan.calls.iter().filter_map(|call| call.kv_input).collect();
        for write in inputs.borrow(py).cache_imports(py) {
            let write = write.get();
            if !cache_inputs.contains(&write.export.source) {
                continue;
            }
            let cache =
                cache.ok_or_else(|| invalid(py, "KV input requires cache export storage"))?;
            if let Some(existing) = cache.borrow().inner.resident(write.export.source)
                && existing.as_ref() != write.export.as_ref()
            {
                return Err(invalid(
                    py,
                    "prepared KV import conflicts with its resident export",
                ));
            }
        }

        if plan.input_products.is_empty() {
            return Ok(());
        }

        // Admission already resolved input roles, placement and immutable
        // transfer dimensions. Only calls whose predicates passed consume.
        let outputs = slf
            .borrow()
            .outputs
            .iter()
            .map(|output| output.clone_ref(py))
            .collect::<Vec<_>>();
        let mut consumers = HashMap::new();
        for (index, call) in plan.calls.iter().enumerate() {
            if outputs[index].borrow(py).lock(py)?.output.status == CallStatus::Predicated {
                continue;
            }
            for product in call.tensor_inputs().chain(call.predicate.as_ref()) {
                consumers.entry(product).or_insert(index);
            }
        }

        for export in &plan.input_products {
            let Some(&index) = consumers.get(&export.product) else {
                continue;
            };
            let input = inputs
                .borrow(py)
                .get(py, export.product.buffer_id())
                .ok_or_else(|| {
                    PyRuntimeError::new_err("transferred input lost its prepared destination")
                })?;

            // No batch/input borrow spans completion callbacks. The cloned
            // input retains its views while its owner makes them visible.
            match (input, &export.value) {
                (Input::Borrowed, _) => {}
                (Input::Tensor(read), _) => tensors.complete_import(py, read.bind(py))?,
                (
                    Input::Latent(write),
                    TransferHandle::Latent {
                        height,
                        width,
                        step,
                        ..
                    },
                ) => {
                    let output = &outputs[index];
                    let (slot, current_step) = {
                        let output = output.borrow(py);
                        let slot = output.request.borrow(py).request.slot();
                        let step = output.lock(py)?.progress.flow_step;
                        (slot, step)
                    };
                    if current_step != 0 {
                        return Err(invalid(
                            py,
                            "latent transfer destination already owns a trajectory",
                        ));
                    }
                    if write.get().inner.request_pool_idx != slot {
                        return Err(invalid(
                            py,
                            "latent import has a different destination slot",
                        ));
                    }

                    let pool = latents
                        .ok_or_else(|| invalid(py, "latent import requires physical storage"))?;
                    pool.borrow_mut().adopt_import(
                        py,
                        write.bind(py),
                        i64::from(export.product.generation),
                        i64::from(*step),
                        i64::from(*height),
                        i64::from(*width),
                    )?;
                    output.borrow(py).lock(py)?.progress.flow_step = u64::from(*step);
                }
                _ => {
                    return Err(PyRuntimeError::new_err(
                        "tensor product has no prepared tensor or latent import",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Bind active trajectory intervals and their numerical latent views.
    /// Model callbacks describe shapes; the batch owns progress and selection.
    pub(super) fn bind_latents(
        slf: &Bound<'_, Self>,
        pool: Option<&Bound<'_, LatentPool>>,
        image_builder: Option<&Bound<'_, PyAny>>,
        media_builder: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let py = slf.py();
        let (plan, batch, selected, outputs) = {
            let this = slf.borrow();
            if this.plan.latent_params.is_empty() {
                return Ok(());
            }

            let mut selected = Vec::new();
            for (parameter, params) in this.plan.latent_params.iter().enumerate() {
                let index = this.request_indexes[&params.request_key.request_id.0];
                let output = &this.outputs[index];
                if output.borrow(py).lock(py)?.output.status != CallStatus::Predicated {
                    selected.push((parameter, index, output.clone_ref(py)));
                }
            }

            (
                Arc::clone(&this.plan),
                this.batch.clone_ref(py),
                selected,
                this.outputs
                    .iter()
                    .map(|output| output.clone_ref(py))
                    .collect::<Vec<_>>(),
            )
        };
        if selected.is_empty() {
            return Ok(());
        }

        let pool = pool.ok_or_else(|| invalid(py, "latent inputs require a resident pool"))?;

        let image_shape = image_builder
            .map(|builder| {
                Ok::<_, PyErr>((
                    builder.getattr("denoiser")?.getattr("latent_shape")?,
                    py.import("uniserve.media.image")?.getattr("Config")?,
                ))
            })
            .transpose()?;
        let sample_pages = if image_shape.is_none() {
            let builder =
                media_builder.ok_or_else(|| invalid(py, "latent inputs require a denoiser"))?;
            Some((
                builder.getattr("slot_pages")?,
                builder
                    .getattr("sample_pages")?
                    .getattr("units")?
                    .extract::<u64>()?,
            ))
        } else {
            None
        };
        let imported_steps: HashMap<_, _> = plan
            .input_products
            .iter()
            .filter_map(|export| match &export.value {
                TransferHandle::Latent { step, .. } => Some((&export.product, u64::from(*step))),
                _ => None,
            })
            .collect();

        // Validate every interval before any writable image view is assigned.
        for &(parameter, index, ref output) in &selected {
            let params = &plan.latent_params[parameter];
            let call = &plan.calls[index];
            let (request, mut committed_step) = {
                let output = output.borrow(py);
                let request = Arc::clone(&output.request.borrow(py).request);
                let step = output.lock(py)?.progress.flow_step;
                (request, step)
            };

            if let Some((pages, units)) = &sample_pages {
                let pages = pages.call1((request.slot(),))?.extract::<Vec<u32>>()?;
                if params.page_table != pages || u64::from(params.latent_units) != *units {
                    return Err(invalid(
                        py,
                        "latent inputs do not name the request slot's pages",
                    ));
                }
            }

            let total_steps = if let Some((shape, size)) = &image_shape {
                let image = request.admission().image.as_ref().ok_or_else(|| {
                    invalid(py, "latent inputs have no admitted image dimensions")
                })?;
                let size = size.call1((params.height, params.width))?;
                let units = shape
                    .call1(("image", size))?
                    .get_item(0)?
                    .extract::<u64>()?;
                if params.height != image.height
                    || params.width != image.width
                    || u64::from(params.latent_units) != units
                {
                    return Err(invalid(
                        py,
                        "latent inputs disagree with admitted model dimensions",
                    ));
                }
                if let Some(step) = call
                    .latent_input
                    .as_ref()
                    .and_then(|input| imported_steps.get(input))
                {
                    committed_step = *step;
                }

                Some(u64::from(image.steps))
            } else {
                None
            };

            let start = u64::from(params.start_step);
            let count = u64::from(params.step_count);
            let valid = match call.code {
                CallKind::Media(MediaCall::LatentPreparation) => start == 0 && count == 0,
                CallKind::Media(MediaCall::Denoising) => {
                    start == committed_step
                        && match total_steps {
                            Some(total) => {
                                count > 0
                                    && start + count <= total
                                    && (call.bounds.max_tokens == 0
                                        || count <= u64::from(call.bounds.max_tokens))
                            }
                            None => count == 1,
                        }
                }
                _ => start == committed_step && count == 0,
            };
            if !valid {
                return Err(invalid(
                    py,
                    "latent interval disagrees with request progress or admitted schedule",
                ));
            }
        }

        // Image rows borrow one shared scratch allocation. Standalone media
        // runners bind their own numerical views over their fixed slot pages.
        let views = if image_shape.is_some() {
            let mut occupied = Vec::new();
            for output in outputs {
                if let Some(view) = &output.borrow(py).latent_buffer {
                    occupied.push(view.clone_ref(py));
                }
            }

            let tables = selected
                .iter()
                .map(|&(parameter, _, _)| {
                    plan.latent_params[parameter]
                        .page_table
                        .iter()
                        .map(|&page| i64::from(page))
                        .collect()
                })
                .collect();
            let units = selected
                .iter()
                .map(|&(parameter, _, _)| i64::from(plan.latent_params[parameter].latent_units))
                .collect();
            Some(pool.borrow().bind(py, tables, units, occupied)?)
        } else {
            None
        };

        let parameters = batch.bind(py).getattr("latent_params")?;
        for (row, (parameter, _, output)) in selected.iter().enumerate() {
            let mut output = output.borrow_mut(py);
            output.latent_params = Some(parameters.get_item(*parameter)?.unbind());
            if let Some(views) = &views {
                output.latent_buffer = Some(views[row].clone_ref(py));
            }
        }

        Ok(())
    }

    /// Install active calls' KV assignments and retain their physical access.
    /// Callbacks only copy numerical tables and reset recycled cache units.
    pub(super) fn bind_cache(
        &self,
        py: Python<'_>,
        cache: &Bound<'_, KVCacheManager>,
        tables: &Bound<'_, BlockTables>,
        copy: &Bound<'_, PyAny>,
        recycle: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let started = Instant::now();
        let forward = &self.plan.forward;
        let mut active = Vec::new();
        let mut slots = HashSet::new();
        for (index, output) in self.outputs.iter().enumerate() {
            let output = output.borrow(py);
            let pending = output.lock(py)?;
            if pending.output.status == CallStatus::Predicated {
                continue;
            }

            let slot = output.request.borrow(py).request.slot() as u32;
            active.push((index, slot, pending.progress.kv_visible_len));
            slots.insert(slot);
            slots.extend(
                self.forward_indices[index]
                    .iter()
                    .map(|&row| forward.request_pool_indices[row]),
            );
        }

        // Alternative-prefix rows need their own assignments even though they
        // share a call's progress. Numerical copying precedes host-table commit.
        let assignments = self
            .plan
            .block_tables
            .iter()
            .filter(|table| slots.contains(&table.request_pool_idx))
            .cloned()
            .collect::<Vec<_>>();
        tables.borrow_mut().install(py, &assignments, copy)?;

        // Import copies reset their own destinations. All other new units are
        // recycled together before this batch can submit model writes.
        let mut initialized = HashSet::new();
        for write in self.inputs.borrow(py).cache_imports(py) {
            let write = write.get();
            initialized.extend(
                write
                    .initialized_units
                    .iter()
                    .map(|&unit| (write.request_pool_idx() as u32, unit)),
            );
        }

        let mut recycled = Vec::new();
        let mut seen = HashSet::new();
        for allocation in &self.plan.new_cache_units {
            if !slots.contains(&allocation.request_pool_idx) {
                continue;
            }
            for unit in &allocation.unit_ids {
                if initialized.contains(&(allocation.request_pool_idx, unit.0)) {
                    continue;
                }
                if !seen.insert(unit.0) {
                    return Err(invalid(py, "KV allocation repeats a physical unit"));
                }
                recycled.push(unit.0);
            }
        }

        if !recycled.is_empty() {
            let unit_tokens = tables
                .borrow()
                .tables
                .groups()
                .iter()
                .map(|group| group.page_tokens)
                .max()
                .unwrap_or(0);
            let spans = recycled
                .iter()
                .map(|&unit| (unit, 0, unit_tokens))
                .collect::<Vec<_>>();
            cache
                .borrow_mut()
                .inner
                .require_reusable(&spans)
                .map_err(|error| native_error(py, error))?;
            recycle.call1((PyTuple::new(py, recycled)?,))?;
        }

        let completion = self.output_buffer(py)?.borrow(py).completion(py)?;
        let mut tables = tables.borrow_mut();
        let tables = &mut tables.tables;
        let mut cache = cache.borrow_mut();
        for (index, main_slot, visible) in active {
            let call = &self.plan.calls[index];
            let rows = &self.forward_indices[index];
            if let Some(&row) = rows
                .iter()
                .find(|&&row| forward.request_pool_indices[row] == main_slot)
            {
                let declared = u64::from(forward.seq_lens[row] - forward.query_lens[row]);
                let relayed = call
                    .predicate
                    .as_ref()
                    .is_some_and(|value| value.dtype == DType::I64);
                // A queued token relay carries a capacity bound. Its device
                // predicate determines the actual visible length at execution.
                if declared < visible || (!relayed && declared != visible) {
                    return Err(invalid(
                        py,
                        "forward row sequence length disagrees with execution state",
                    ));
                }
            }

            let mut spans = Vec::new();
            for &row in rows {
                let slot = forward.request_pool_indices[row];
                if slot != main_slot {
                    if forward.seq_lens[row] - forward.query_lens[row]
                        > tables.allocated_length(slot)
                    {
                        return Err(invalid(
                            py,
                            "forward row exceeds alternative-prefix capacity",
                        ));
                    }
                    tables.retain_prefix(call.request_key, slot);
                }

                // Read-only rows retain the prefix; writers also retain query
                // tokens. A windowed group starts at its first resident page.
                let length = forward.seq_lens[row]
                    - if forward.write_kv[row] {
                        0
                    } else {
                        forward.query_lens[row]
                    };
                for group in 0..tables.groups().len() as u32 {
                    let table = tables
                        .table(slot, group)
                        .map_err(|error| native_error(py, error))?;
                    let start = u64::from(table.start_page) * u64::from(table.shape.page_tokens);
                    if u64::from(length) > start {
                        spans.extend(
                            table
                                .spans(start, u64::from(length) - start)
                                .map_err(|error| native_error(py, error))?,
                        );
                    }
                }
            }
            cache.inner.retain_execution(
                call.request_key,
                &spans,
                CompletionRef::new(py, completion.clone_ref(py)),
            );
        }

        let components = self.component_us.bind(py);
        let previous = components
            .get_item("bc_tables")?
            .map_or(Ok(0), |value| value.extract::<u64>())?;
        components.set_item("bc_tables", previous + started.elapsed().as_micros() as u64)?;
        Ok(())
    }

    /// Bind all output rows before exposing any to numerical execution.
    pub(super) fn bind_outputs(
        slf: &Bound<'_, Self>,
        requests: &Bound<'_, RequestPool>,
        buffer: Py<OutputBuffer>,
        started_ns: u64,
        predicated: HashSet<u64>,
    ) -> PyResult<()> {
        let py = slf.py();
        let (plan, batch) = {
            let this = slf.borrow();
            (Arc::clone(&this.plan), this.batch.clone_ref(py))
        };
        let bindings = {
            let requests = requests.borrow();
            let rows = plan
                .calls
                .iter()
                .map(|call| {
                    let request = requests
                        .pool
                        .get(call.request_key.request_id.0)
                        .map_err(|error| native_error(py, error))?;
                    Ok((call.request_key, call.call_id, request.slot()))
                })
                .collect::<PyResult<Vec<_>>>()?;
            requests
                .pool
                .bind_calls(&rows)
                .map_err(|error| native_error(py, error))?
                .iter()
                .map(|request| requests.get(py, request.key().request_id.0))
                .collect::<PyResult<Vec<_>>>()?
        };

        let calls = batch.bind(py).getattr("calls")?;
        let mut outputs = Vec::with_capacity(plan.calls.len());
        for (index, (call, request)) in plan.calls.iter().zip(bindings).enumerate() {
            let output = PendingOutput::for_call(
                py,
                calls.get_item(index)?.unbind(),
                call,
                request,
                buffer.clone_ref(py),
                index,
            )?;
            if predicated.contains(&call.request_key.request_id.0) {
                output.lock(py)?.output.status = CallStatus::Predicated;
            }
            outputs.push(Py::new(py, output)?);
        }

        let mut this = slf.borrow_mut();
        this.outputs = outputs;
        this.buffer = Some(buffer);
        this.started_ns = started_ns;
        Ok(())
    }

    /// Read completed U8 predicates once. Device-gated predicates remain tensors.
    pub(super) fn resolve_predicates(slf: &Bound<'_, Self>) -> PyResult<HashSet<u64>> {
        let py = slf.py();
        let (inputs, plan, captures) = {
            let this = slf.borrow();
            (
                this.inputs.clone_ref(py),
                Arc::clone(&this.plan),
                this.predicate_captures.clone(),
            )
        };
        let Some(buffer) = inputs.borrow(py).predicate(py) else {
            return Ok(HashSet::new());
        };
        if !buffer.get().ready(py)? {
            return Err(PyRuntimeError::new_err(
                "prepared predicates were observed before readiness",
            ));
        }

        let mut predicated = HashSet::new();
        let read = (|| -> PyResult<()> {
            for (row, (index, span)) in captures.into_iter().enumerate() {
                let (offset, count) = span.ok_or_else(|| {
                    PyRuntimeError::new_err("sealed predicate buffer has an uncaptured row")
                })?;
                let value = buffer.get().with_readback(py, |buffer| {
                    match buffer.read_tokens(offset, count)? {
                        [0] => Ok(false),
                        [1] => Ok(true),
                        _ => Err(uniserve_worker::Error::Invalid(
                            "call predicate is not a canonical boolean".into(),
                        )),
                    }
                })?;
                if !value {
                    predicated.insert(plan.calls[index].request_key.request_id.0);
                }
                buffer.get().observe(py, row)?;
            }
            Ok(())
        })();
        if let Err(error) = read {
            OutputBuffer::abandon(buffer.bind(py))?;
            return Err(error);
        }

        inputs.borrow_mut(py).set_predicate(None);
        Ok(predicated)
    }
}

#[pymethods]
impl BatchState {
    /// Borrow a call's single resident tensor and keep its lease through
    /// commit or discard. Numerical consumers and transfers use this owner.
    pub(super) fn consume_tensor(
        &self,
        py: Python<'_>,
        request_id: u64,
        tensor_store: &TensorStore,
        device: Bound<'_, PyAny>,
    ) -> PyResult<Py<TensorRead>> {
        let &index = self
            .request_indexes
            .get(&request_id)
            .ok_or_else(|| invalid(py, "request has no output in this batch"))?;
        let call = &self.plan.calls[index];
        let output = self.outputs[index].borrow(py);
        let numerical = output.call.bind(py);
        let feature_count =
            call.vision_inputs.len() + usize::from(call.latent_feature_input.is_some());
        let reference = if feature_count != 0 {
            if feature_count != 1 {
                return Err(invalid(py, "feature transfer requires one source"));
            }
            if call.vision_inputs.is_empty() {
                numerical.getattr("latent_feature_input")?
            } else {
                numerical
                    .getattr("vision_inputs")?
                    .get_item(0)?
                    .getattr("feature")?
            }
        } else {
            if call.inputs.len()
                + usize::from(call.token_input.is_some())
                + usize::from(call.image_input.is_some())
                != 1
            {
                return Err(invalid(py, "tensor transfer requires one resident source"));
            }
            if !call.inputs.is_empty() {
                numerical.getattr("inputs")?.get_item(0)?
            } else if call.token_input.is_some() {
                numerical.getattr("token_input")?
            } else {
                numerical.getattr("image_input")?
            }
        };
        let read =
            tensor_store.consume(py, reference, numerical.getattr("call_id")?, Some(device))?;
        let reads = if feature_count != 0 {
            &output.feature_reads
        } else {
            &output.device_reads
        };
        reads.bind(py).append(read.bind(py))?;
        if feature_count != 0 {
            let source = read.borrow(py);
            let feature = match &source.metadata {
                Some(metadata) => metadata.bind(py).is_instance(
                    &py.import("uniserve_worker.storage.tensor_store")?
                        .getattr("FeatureMetadata")?,
                )?,
                None => false,
            };
            if !feature {
                return Err(invalid(py, "feature transfer requires spatial metadata"));
            }
        }
        Ok(read)
    }

    /// Export the numerical values owned by this rank. Deferred host writes
    /// remain with their pending output and become visible together here.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (request_id, values, tensor_store, transports, *, host=false, regions=None))]
    pub(super) fn export_tensors(
        &self,
        py: Python<'_>,
        request_id: u64,
        values: Vec<Bound<'_, PyAny>>,
        tensor_store: &TensorStore,
        transports: &Bound<'_, PyAny>,
        host: bool,
        regions: Option<Vec<Vec<Bound<'_, PyTuple>>>>,
    ) -> PyResult<()> {
        let &index = self
            .request_indexes
            .get(&request_id)
            .ok_or_else(|| invalid(py, "request has no output in this batch"))?;
        let call = &self.plan.calls[index];
        PendingOutput::export_tensors(
            self.outputs[index].bind(py),
            call,
            values,
            tensor_store,
            transports,
            host,
            regions,
        )
    }

    /// Return the decoder-unit interval assigned to this request's call.
    fn decode_range<'py>(&self, py: Python<'py>, request_id: u64) -> PyResult<Bound<'py, PyAny>> {
        let &index = self
            .request_indexes
            .get(&request_id)
            .ok_or_else(|| invalid(py, "request has no output in this batch"))?;
        let call = &self.plan.calls[index];
        let params = self
            .plan
            .decode_ranges
            .iter()
            .find(|params| params.request_key == call.request_key && params.call_id == call.call_id)
            .ok_or_else(|| invalid(py, "video decode call has no decode params"))?;
        crate::convert::decode_range_to_py(
            py,
            params,
            &mut crate::convert::RequestConversion::new(py)?,
        )
    }

    /// Record completion only after the numerical consumer has written or read
    /// its latent bank. Ranks without that consumer leave their update empty.
    pub(super) fn complete_latent(&self, py: Python<'_>, request_id: u64) -> PyResult<()> {
        let index = self
            .request_indexes
            .get(&request_id)
            .ok_or_else(|| invalid(py, "request has no output in this batch"))?;
        let call = &self.plan.calls[*index];
        let params = self
            .plan
            .latent_params
            .iter()
            .find(|params| params.request_key == call.request_key && params.call_id == call.call_id)
            .ok_or_else(|| invalid(py, "call has no bound latent parameters"))?;
        let output = self.outputs[*index].borrow(py);
        let slot = output.request.borrow(py).request.slot();
        let update = uniserve_worker::LatentUpdate::for_call(slot, call, params)
            .map_err(|error| native_error(py, error))?;
        output.lock(py)?.latent_update = Some(update);
        Ok(())
    }

    /// Add one completed numerical invocation without retaining Python results.
    pub(super) fn record_forward(&mut self, stats: PyRef<'_, crate::stats::ForwardStats>) {
        self.forward_stats.merge(&stats.inner);
    }

    pub(super) fn execution_stats(&self, py: Python<'_>) -> PyResult<crate::stats::ForwardStats> {
        if self.started_ns == 0 {
            return Ok(crate::stats::ForwardStats::from(
                uniserve_worker_ipc::ForwardStats::default(),
            ));
        }

        let mut stats = self.forward_stats.clone();
        for (name, elapsed) in self.component_us.bind(py) {
            let elapsed = elapsed.extract::<i64>()?.max(0) as u64;
            let count = stats.component_us.entry(name.extract()?).or_default();
            *count = count.saturating_add(elapsed);
        }

        Ok(crate::stats::ForwardStats::from(stats))
    }

    #[getter]
    fn batch_id(&self) -> u64 {
        self.plan.batch_id
    }

    /// Return numerical forward rows in their scheduler-supplied order.
    fn forward_rows<'py>(&self, py: Python<'py>, request_id: u64) -> PyResult<Bound<'py, PyTuple>> {
        let rows = self
            .request_indexes
            .get(&request_id)
            .map_or(&[][..], |&index| self.forward_indices[index].as_slice());
        PyTuple::new(py, rows)
    }

    #[getter]
    pub(super) fn route(&self) -> Option<&'static str> {
        self.plan.calls.first().map(|call| call.code.as_str())
    }

    /// Enter the numerical stream, or retain the caller's current stream.
    pub(super) fn scope(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let stream = slf
            .borrow()
            .stream
            .as_ref()
            .map(|stream| stream.clone_ref(py));
        match stream {
            Some(stream) => py.import("torch.cuda")?.call_method1("stream", (stream,)),
            None => py.import("contextlib")?.call_method0("nullcontext"),
        }
        .map(Bound::unbind)
    }

    #[getter]
    pub(super) fn output_buffer(&self, py: Python<'_>) -> PyResult<Py<OutputBuffer>> {
        self.buffer
            .as_ref()
            .map(|buffer| buffer.clone_ref(py))
            .ok_or_else(|| PyRuntimeError::new_err("batch has no reserved output buffer"))
    }

    /// Borrow the complete set of output rows in scheduler call order.
    fn pending_outputs(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        if self.outputs.len() != self.plan.calls.len() {
            return Err(PyRuntimeError::new_err(
                "batch has no reserved pending outputs",
            ));
        }
        PyTuple::new(py, &self.outputs).map(Bound::unbind)
    }

    fn pending_output(&self, py: Python<'_>, request_id: u64) -> PyResult<Py<PendingOutput>> {
        let index = self
            .request_indexes
            .get(&request_id)
            .ok_or_else(|| invalid(py, format!("batch has no request {request_id}")))?;
        self.outputs
            .get(*index)
            .map(|output| output.clone_ref(py))
            .ok_or_else(|| PyRuntimeError::new_err("request has no reserved pending output"))
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.batch)?;
        visit.call(&self.inputs)?;
        visit.call(&self.buffer)?;
        for output in &self.outputs {
            visit.call(output)?;
        }
        visit.call(&self.stream)?;
        visit.call(&self.component_us)
    }

    fn __clear__(&mut self, py: Python<'_>) {
        self.batch = py.None();
        self.outputs.clear();
        self.buffer = None;
        self.stream = None;
        self.predicate_captures.clear();
        self.forward_stats = uniserve_worker_ipc::ForwardStats::default();
        self.component_us = PyDict::new(py).unbind();
    }
}
