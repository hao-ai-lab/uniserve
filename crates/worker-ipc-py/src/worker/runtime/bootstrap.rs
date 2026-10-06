//! Load a rank and bind its numerical resources before execution starts.

use std::collections::HashSet;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyType};

use super::{Worker, abort, closed};
use crate::worker::buffer::BufferPool;
use crate::worker::capacity::{self, inputs as capacity_inputs, pools, report};
use crate::worker::error::unsupported;
use crate::worker::events::EventPool;
use crate::worker::execution::close_all;
use crate::worker::host::HostLane;
use crate::worker::latent::LatentPool;
use crate::worker::model_runners::ModelRunners;
use crate::worker::output::OutputPool;
use crate::worker::request::RequestPool;
use crate::worker::storage::TensorStore;

pub(super) fn from_config<'py>(
    cls: &Bound<'py, PyType>,
    config: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = cls.py();
    // Interpret checkpoint declarations before process groups exist so an
    // invalid component placement cannot leave peers waiting for startup.
    let loader = py.import("uniserve_worker.bootstrap.model_loader")?;
    let prepared = loader.call_method1("prepare_worker_model", (config,))?;
    let source = prepared.get_item(0)?;
    let description = prepared.get_item(1)?;
    let declarations = prepared.get_item(2)?;
    let execution = config.getattr("execution")?;
    let execution_native = crate::worker::config::native(&execution)?;
    let experts = config.getattr("expert_parallel")?;
    let options = PyDict::new(py);
    options.set_item("rank", execution_native.rank)?;
    options.set_item("world_size", execution_native.world_size)?;
    options.set_item("device", &execution_native.device)?;
    options.set_item("local_rank", config.getattr("local_rank")?)?;
    options.set_item("backend", config.getattr("distributed_backend")?)?;
    options.set_item("rendezvous", config.getattr("rendezvous")?)?;
    if !experts.is_none() {
        options.set_item(
            "experts",
            (
                experts.getattr("rank")?,
                experts.getattr("size")?,
                experts.getattr("rendezvous")?,
            ),
        )?;
    }
    let groups = py
        .import("uniserve.runtime.process_groups")?
        .call_method("initialize_process_groups", (), Some(&options))
        .map_err(|error| setup_error(py, error))?;

    let result = (|| {
        let components = config.getattr("components")?;
        let bindings = if execution_native.role == "experts" {
            PyDict::new(py).into_any()
        } else {
            let options = PyDict::new(py);
            options.set_item("declarations", &declarations)?;
            let components = py.get_type::<PyDict>().call1((&components,))?;
            py.import("uniserve_worker.bootstrap.distributed")?
                .call_method(
                    "initialize_components",
                    (&groups, components),
                    Some(&options),
                )
                .map_err(|error| setup_error(py, error))?
        };
        let options = PyDict::new(py);
        options.set_item("source", source)?;
        options.set_item("description", description)?;
        options.set_item("declarations", &declarations)?;
        options.set_item("experts", groups.getattr("experts")?)?;
        let loaded =
            loader.call_method("load_worker_model", (config, &bindings), Some(&options))?;

        // Sampling uses the TP group of the text component independently of
        // the parallel layout of encoders, decoders and denoisers.
        let causal_lm = py.import("uniserve.model")?.getattr("CausalLM")?;
        let mut model_mesh = py.None().into_bound(py);
        'components: for item in declarations.call_method0("items")?.try_iter()? {
            let item = item?;
            let name = item.get_item(0)?;
            if !bindings.contains(&name)? {
                continue;
            }
            for call in item.get_item(1)?.try_iter()? {
                if call?.getattr("module")?.is_instance(&causal_lm)? {
                    model_mesh = bindings.get_item(&name)?.getattr("mesh")?;
                    break 'components;
                }
            }
        }
        // A rank outside the text component has no sampling mesh.
        let sampling_group = if model_mesh.is_none() {
            model_mesh
        } else {
            model_mesh.call_method1("get_group", ("tp",))?
        };
        let options = PyDict::new(py);
        options.set_item("bindings", bindings)?;
        options.set_item("sampling_group", sampling_group)?;
        for name in [
            "entry_points",
            "tokenizer",
            "image_processor",
            "flow_prompt",
            "model_name",
        ] {
            options.set_item(name, loaded.getattr(name)?)?;
        }
        options.set_item("worker_config", loaded.getattr("config")?)?;
        options.set_item("allowed_calls", config.getattr("supported_calls")?)?;
        options.set_item("worker_id", config.getattr("worker_id")?)?;
        let plane = config.getattr("data_plane")?;
        options.set_item("transfer_backends", plane.getattr("backends")?)?;
        options.set_item("export_backends", plane.getattr("export_backends")?)?;
        if !experts.is_none() {
            options.set_item("attention_ranks", experts.getattr("attention_ranks")?)?;
        }
        let ipc = config.getattr("ipc")?;
        for name in [
            "queue_depth",
            "acknowledgment_slot",
            "host_slots",
            "products_cross_hosts",
        ] {
            options.set_item(name, ipc.getattr(name)?)?;
        }
        options.set_item(
            "completion_payload_bytes",
            ipc.getattr("max_payload_bytes")?,
        )?;
        options.set_item("components", components)?;
        options.set_item("process_groups", &groups)?;
        cls.call((loaded.getattr("model")?,), Some(&options))
    })();
    match result {
        Ok(worker) => Ok(worker),
        Err(error) => close_all(py, [Err(error), abort(py, &groups)]).map(|()| unreachable!()),
    }
}

fn setup_error(py: Python<'_>, error: PyErr) -> PyErr {
    if error.is_instance_of::<PyValueError>(py) {
        let mapped = unsupported(py, error.value(py).to_string());
        mapped.set_cause(py, Some(error));
        mapped
    } else {
        error
    }
}

impl Worker {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn allocate(
        &mut self,
        py: Python<'_>,
        model: &Bound<'_, PyAny>,
        allowed_calls: Option<Py<PyAny>>,
        queue_depth: usize,
        completion_payload_bytes: usize,
        acknowledgment_slot: usize,
        host_slots: Vec<usize>,
        products_cross_hosts: bool,
        attention: Option<Py<PyAny>>,
        transfer_backends: Vec<String>,
        export_backends: Vec<String>,
        worker_id: &str,
        image_processor: Option<Py<PyAny>>,
        flow_prompt: Option<Py<PyAny>>,
        components: Option<Py<PyAny>>,
        bindings: Option<Py<PyAny>>,
        model_name: Option<Py<PyAny>>,
        attention_ranks: usize,
        entry_points: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let mut config = self.worker_config.clone_ref(py).into_bound(py);
        let config_native = crate::worker::config::native(&config)?;
        let torch = py.import("torch")?;
        if !model.is_instance(&py.import("torch.nn")?.getattr("Module")?)? {
            return Err(unsupported(
                py,
                "worker model has no supported execution surface",
            ));
        }
        if queue_depth == 0 {
            return Err(unsupported(py, "worker pipeline depth must be positive"));
        }
        if export_backends.is_empty()
            || export_backends.iter().collect::<HashSet<_>>().len() != export_backends.len()
            || export_backends
                .iter()
                .any(|name| !transfer_backends.contains(name))
        {
            return Err(unsupported(
                py,
                "export backends must be unique bound transports",
            ));
        }
        if transfer_backends.iter().any(|name| name == "cuda_vmm")
            && torch
                .getattr("device")?
                .call1((&config_native.device,))?
                .getattr("type")?
                .extract::<String>()?
                != "cuda"
        {
            return Err(unsupported(py, "CUDA VMM requires a CUDA worker device"));
        }

        // Ownership starts after argument validation and before numerical
        // initialization can enqueue device accesses.
        self.model = Some(model.clone().unbind());
        let group = match &self.process_groups {
            Some(groups) => groups.bind(py).getattr("process_group")?,
            None => py.None().into_bound(py),
        };
        let options = PyDict::new(py);
        options.set_item("bindings", &bindings)?;
        options.set_item("attention", attention)?;
        options.set_item("entry_points", entry_points)?;
        options.set_item("image_processor", &image_processor)?;
        options.set_item("flow_prompt", flow_prompt)?;
        options.set_item("max_inflight", queue_depth)?;
        if let Some(groups) = &self.process_groups {
            options.set_item("expert_group", groups.bind(py).getattr("experts")?)?;
        }
        options.set_item("attention_ranks", attention_ranks)?;
        let runner = py
            .import("uniserve_worker.execution.model_executor")?
            .getattr("ModelExecutor")?
            .call((&model, &config), Some(&options))?;
        self.runner = Some(runner.clone().unbind());
        self.attention = Some(runner.getattr("attention")?.unbind());

        // Capacity is measured after model inputs and workspaces are bound.
        // The page size is resolved first so every later pool uses that size.
        let capacity_group = (!group.is_none()).then_some(&group);
        config = capacity_inputs::resolve_page_size(
            model,
            &config,
            self.attention.as_ref().ok_or_else(closed)?.bind(py),
            capacity_group,
        )?;
        runner.setattr("worker_config", &config)?;
        config = pools::resolve_request_capacity(
            model,
            &config,
            queue_depth,
            capacity_group,
            bindings.as_ref().map(|value| value.bind(py)),
            Some(&runner.getattr("state_buffers")?),
        )?;
        let config_native = crate::worker::config::native(&config)?;
        self.worker_config = config.clone().unbind();
        runner.setattr("worker_config", &config)?;

        let endpoint = py
            .import("uniserve_worker.protocol.transfer")?
            .getattr("WorkerEndpoint")?
            .call_method1("local", (worker_id, config_native.rank))?;
        let components = components
            .map(|value| value.into_bound(py))
            .unwrap_or_else(|| PyTuple::empty(py).into_any());
        let layout = report::build_worker_layout(
            model,
            &config,
            image_processor.as_ref().map(|value| value.bind(py)),
            model_name
                .as_ref()
                .map(|value| value.extract(py))
                .transpose()?,
            queue_depth,
            Some(&endpoint),
            capacity_group.or_else(|| self.sampling_group.as_ref().map(|value| value.bind(py))),
            allowed_calls.as_ref().map(|value| value.bind(py)),
            transfer_backends.clone(),
            Some(&components),
            &runner
                .getattr("attention")?
                .getattr("name")?
                .extract::<String>()?,
            bindings.as_ref().map(|value| value.bind(py)),
            Some(&runner.getattr("state_buffers")?),
        )?;
        self.check_auxiliary_storage(py, &layout)?;
        let info = &layout.info;
        self.info = Some(info.clone());
        let arena = &layout.arena;
        self.device_product_bytes = arena.device_product_bytes;
        let inputs = py.import("uniserve_worker.bootstrap.inputs")?;
        let text = inputs.call_method1(
            "capability",
            (&model, py.import("uniserve.model")?.getattr("CausalLM")?),
        )?;
        if !text.is_none() {
            self.allocate_cache(py, &text, info, queue_depth)?;
        }

        let options = PyDict::new(py);
        let state_buffers = runner.getattr("state_buffers")?;
        if state_buffers.is_truthy()? {
            options.set_item("state_buffers", state_buffers)?;
        }
        options.set_item("device", &config_native.device)?;
        let requests = py
            .get_type::<RequestPool>()
            .call((info.request_slots,), Some(&options))?
            .cast_into::<RequestPool>()?;
        self.requests = Some(requests.clone().unbind());
        let dtype_name = config_native.model_dtype.trim_start_matches("torch.");
        let dtype = torch
            .getattr(dtype_name)
            .map_err(|_| unsupported(py, format!("unsupported model dtype {dtype_name:?}")))?;
        if !dtype.is_instance(&torch.getattr("dtype")?)? {
            return Err(unsupported(
                py,
                format!("unsupported model dtype {dtype_name:?}"),
            ));
        }
        if let Some(tables) = &self.block_tables {
            let options = PyDict::new(py);
            options.set_item("request_pool_size", info.request_slots)?;
            options.set_item(
                "vocab_size",
                text.getattr("backbone")?.getattr("vocab_size")?,
            )?;
            options.set_item("continuation_width", 1)?;
            options.set_item("device", &config_native.device)?;
            options.set_item("logits_dtype", dtype)?;
            options.set_item(
                "valid_cache_lengths",
                tables.bind(py).getattr("verified_lengths")?,
            )?;
            self.decode_state = Some(
                py.import("uniserve_worker.storage.decode_state")?
                    .getattr("DecodeState")?
                    .call((), Some(&options))?
                    .unbind(),
            );
        }

        let denoises = runner.getattr("denoises")?.is_truthy()?;
        let generation_device = config_native
            .generation_device
            .as_deref()
            .unwrap_or(&config_native.device);
        if let Some(plan) = &layout.latent_plan
            && (denoises || !inputs.call_method1("image_builder", (&model,))?.is_none())
        {
            let options = PyDict::new(py);
            options.set_item("request_pool_size", info.request_slots)?;
            options.set_item("num_pages", info.latent_pages)?;
            options.set_item("page_units", info.latent_page_units)?;
            options.set_item("latent_width", plan.latent_width)?;
            options.set_item("dtype", &plan.dtype)?;
            options.set_item("with_workspace", plan.with_workspace)?;
            options.set_item("device", generation_device)?;
            self.latent_pool = Some(
                py.get_type::<LatentPool>()
                    .call((), Some(&options))?
                    .extract()?,
            );
        }
        if denoises {
            runner.call_method1(
                "bind_diffusion_storage",
                (
                    requests.getattr("storage")?.getattr("bank")?,
                    &self.latent_pool,
                ),
            )?;
        }

        // All asynchronous producers use bounded rank-local owners. Graphs and
        // numerical runners borrow their storage without owning its retirement.
        self.device_events = Some(Py::new(py, EventPool::new())?);
        let options = PyDict::new(py);
        let max_calls = info.max_batch_calls as usize;
        options.set_item("capacity", queue_depth * max_calls)?;
        // Four completion fields plus payload words, matching native leases.
        options.set_item(
            "max_words",
            max_calls * (4 + completion_payload_bytes.div_ceil(4)),
        )?;
        options.set_item("event_pool", &self.device_events)?;
        self.output_pool = Some(
            py.get_type::<OutputPool>()
                .call((), Some(&options))?
                .extract()?,
        );
        let devices = self.devices(py)?;
        let options = PyDict::new(py);
        let physical = layout.physical_buffer_pool_bytes;
        options.set_item("byte_capacity", physical)?;
        options.set_item("devices", &devices)?;
        options.set_item("compact", physical < info.buffer_pool_bytes)?;
        self.buffer_pool = Some(
            py.get_type::<BufferPool>()
                .call((), Some(&options))?
                .extract()?,
        );
        let options = PyDict::new(py);
        options.set_item("capacity", arena.tensor_store)?;
        options.set_item("byte_capacity", arena.device_product_bytes)?;
        options.set_item("max_feature_bytes", info.encoder_entry_bytes.max(1))?;
        options.set_item("devices", &devices)?;
        options.set_item("request_capacity", info.request_slots)?;
        options.set_item("relay_depth", info.max_unresolved_calls as usize + 1)?;
        options.set_item("buffer_pool", &self.buffer_pool)?;
        options.set_item("event_pool", &self.device_events)?;
        self.tensor_store = Some(
            py.get_type::<TensorStore>()
                .call((), Some(&options))?
                .extract()?,
        );

        let component_names = components
            .try_iter()?
            .map(|item| item?.get_item(0))
            .collect::<PyResult<Vec<_>>>()?;
        self.codec_slot = py
            .import("uniserve_worker.bootstrap.components")?
            .call_method1("holds_host_components", (component_names,))?
            .extract()?;
        let host_capacity = if self.codec_slot {
            1
        } else {
            arena.host_lane_inflight
        };
        self.host_tasks = Some(Py::new(
            py,
            HostLane::new(py, host_capacity, host_capacity.min(4), "worker-host-lane")?,
        )?);

        let mut transfer_bytes = arena.transfer_bytes as usize;
        if runner.getattr("state_buffers")?.is_truthy()?
            || !inputs
                .call_method1(
                    "capability",
                    (
                        &model,
                        py.import("uniserve.model")?.getattr("VideoPostprocessor")?,
                    ),
                )?
                .is_none()
        {
            // One export credit per representation plus one read credit per
            // remote rank keeps request tensor lifetimes within the pool.
            transfer_bytes *= export_backends.len() + config_native.world_size.saturating_sub(1);
        }
        let options = PyDict::new(py);
        options.set_item("source", endpoint)?;
        options.set_item("byte_capacity", transfer_bytes)?;
        options.set_item("ticket_capacity", arena.transfer_tickets)?;
        options.set_item("event_pool", &self.device_events)?;
        options.set_item("acknowledgment_slot", acknowledgment_slot)?;
        options.set_item("host_slots", host_slots)?;
        options.set_item("cross_host_consumers", products_cross_hosts)?;
        let transports = py
            .import("uniserve_worker.transport")?
            .call_method("make_transports", (transfer_backends,), Some(&options))?
            .cast_into::<PyDict>()?;
        self.transports = Some(transports.clone().unbind());
        let exports = PyDict::new(py);
        for name in export_backends {
            exports.set_item(&name, transports.as_any().get_item(&name)?)?;
        }
        self.export_transports = Some(exports.unbind());

        if let Some(decode_state) = &self.decode_state {
            let cache = self.kv_cache.as_ref().ok_or_else(closed)?.bind(py);
            let widths = capacity_inputs::graph_table_widths(model, &config, cache)?;
            let runners = runner
                .getattr("batch_runners")?
                .cast_into::<ModelRunners>()?;
            ModelRunners::configure_inputs(
                &runners,
                &runner,
                layout.input_config.as_ref().ok_or_else(closed)?.bind(py),
                cache,
                &self
                    .latent_pool
                    .as_ref()
                    .map(|pool| pool.bind(py).as_any().clone())
                    .unwrap_or_else(|| py.None().into_bound(py)),
                &decode_state.bind(py).getattr("predicates")?,
                max_calls,
                info.request_slots as usize,
                (info.latent_pages.saturating_sub(1) * info.latent_page_units) as usize,
                widths.extract()?,
                queue_depth,
            )?;

            let canvas = runner.getattr("canvas_runner")?;
            let sampling = config.getattr("canvas_sampling")?;
            if !canvas.is_none() && !sampling.is_none() {
                let slots = py.import("uniserve_worker.storage.canvas_slots")?;
                let denoiser =
                    slots.call_method1("generating_denoiser", (canvas.getattr("model")?,))?;
                if !denoiser.is_none() {
                    let options = PyDict::new(py);
                    options.set_item("request_pool_size", info.request_slots)?;
                    options.set_item("sampling", sampling)?;
                    options.set_item("device", canvas.getattr("device")?)?;
                    self.canvas_slots = Some(
                        slots
                            .getattr("CanvasSlots")?
                            .call_method("for_denoiser", (denoiser,), Some(&options))?
                            .unbind(),
                    );
                    runner.call_method1("bind_canvas_slots", (&self.canvas_slots,))?;
                }
            }
        } else if config_native.role == "experts" {
            ModelRunners::configure_experts(
                &runner
                    .getattr("batch_runners")?
                    .cast_into::<ModelRunners>()?,
                &runner,
            )?;
        }
        Ok(())
    }

    fn allocate_cache(
        &mut self,
        py: Python<'_>,
        text: &Bound<'_, PyAny>,
        info: &uniserve_worker_ipc::WorkerInfo,
        queue_depth: usize,
    ) -> PyResult<()> {
        let config = self.worker_config.bind(py);
        let config_native = crate::worker::config::native(config)?;
        let cache = text.getattr("cache_config")?;
        if cache.is_none() {
            return Ok(());
        }
        let cache_info = info
            .kv_cache
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("causal language model has no KV capacity"))?;
        let kv_info = py
            .import("uniserve_worker.protocol.worker_info")?
            .getattr("KVCacheInfo")?
            .call_method1(
                "from_mapping",
                (pythonize::pythonize(py, cache_info)?, "worker KV cache"),
            )?;
        let planner = py.import("uniserve_worker.bootstrap.cache")?;
        let planes = planner.call_method1("plan_cache", (text, config))?;
        let storage = planner.call_method1("storage", (config,))?;
        let options = PyDict::new(py);
        options.set_item("num_units", cache_info.num_units)?;
        options.set_item("block_size", config_native.block_size)?;
        options.set_item("device", &config_native.device)?;
        options.set_item("dtype", storage.get_item(0)?)?;
        if storage.get_item(1)?.is_truthy()? {
            let quantization = PyDict::new(py);
            let quantizer = py.import("uniserve.quantization")?.getattr("Quantizer")?;
            let quantizer_options = PyDict::new(py);
            quantizer_options.set_item("axis", 0)?;
            for name in cache.getattr("layers")?.try_iter()? {
                quantization
                    .set_item(name?, quantizer.call(("fp8",), Some(&quantizer_options))?)?;
            }
            options.set_item("quantization", quantization)?;
        }
        let prefix_cache = py
            .import("uniserve.runtime")?
            .getattr("PrefixCache")?
            .call((&cache,), Some(&options))?;
        let options = PyDict::new(py);
        options.set_item("info", kv_info)?;
        options.set_item(
            "group_layers",
            planner.call_method1("group_layers", (text, &planes))?,
        )?;
        options.set_item("import_capacity", info.max_unresolved_calls)?;
        options.set_item("request_pool_size", info.request_slots)?;
        options.set_item(
            "table_width",
            capacity_inputs::resident_width(&planes, config_native.max_sequence_tokens as u64)?,
        )?;
        options.set_item("host_buffer_depth", queue_depth)?;
        let cache = py
            .import("uniserve_worker.storage.kv_cache")?
            .getattr("KVCacheManager")?
            .call((prefix_cache,), Some(&options))?;
        self.block_tables = Some(cache.getattr("block_tables")?.unbind());
        self.kv_cache = Some(cache.unbind());
        Ok(())
    }

    fn devices<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let config = self.worker_config.bind(py);
        let native = crate::worker::config::native(config)?;
        let devices = native
            .devices()
            .map(|device| Ok(device.into_pyobject(config.py())?.into_any()))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(devices)
    }

    fn check_auxiliary_storage(
        &self,
        py: Python<'_>,
        layout: &report::WorkerLayout,
    ) -> PyResult<()> {
        let config = self.worker_config.bind(py);
        let config_native = crate::worker::config::native(config)?;
        let device = py.import("uniserve.runtime.device")?;
        // KV sizing already charges the primary device's fixed storage. Other
        // CUDA devices need their own grant for fixed backing plus graphs.
        for (name, bytes) in &layout.fixed_device_bytes {
            let target = device.call_method1("canonical_device", (name,))?;
            if *name == config_native.device
                || target.getattr("type")?.extract::<String>()? != "cuda"
            {
                continue;
            }
            let available = device
                .call_method1(
                    "device_storage_budget",
                    (&name, config_native.kv_storage_fraction),
                )?
                .get_item(0)?
                .extract::<u64>()?;
            let reserved = bytes
                + uniserve_worker::graph_storage_budget_bytes(
                    capacity::device_total_bytes(&target)? as i64,
                );
            if reserved > available {
                return Err(unsupported(
                    py,
                    format!(
                        "runtime storage on {name} requires {reserved} bytes, but its static storage grant is {available} bytes"
                    ),
                ));
            }
        }
        Ok(())
    }

    pub(super) fn bind_graph_budgets(&self, py: Python<'_>) -> PyResult<()> {
        let config = crate::worker::config::native(self.worker_config.bind(py))?;
        let devices = self.devices(py)?;
        let storage = self
            .runner
            .as_ref()
            .ok_or_else(closed)?
            .bind(py)
            .getattr("graph_storage")?;
        let resident = storage.call_method0("resident_bytes")?;
        let products = self.device_product_bytes / devices.len() as u64;
        let tensors = self.tensor_store.as_ref().ok_or_else(closed)?.bind(py);
        let device = py.import("uniserve.runtime.device")?;
        for name in devices {
            let target = device.call_method1("canonical_device", (name,))?;
            if target.getattr("type")?.extract::<String>()? != "cuda" {
                continue;
            }
            let available = device
                .call_method1(
                    "device_storage_budget",
                    (&target, config.kv_storage_fraction),
                )?
                .get_item(0)?
                .extract::<u64>()?;
            let pending = products.saturating_sub(
                tensors
                    .call_method1("resident_bytes", (&target,))?
                    .extract::<u64>()?,
            );
            let budget = resident
                .call_method1("get", (&target, 0))?
                .extract::<u64>()?
                + available.saturating_sub(pending);
            storage.call_method1("set_budget", (target, budget))?;
        }
        Ok(())
    }
}
