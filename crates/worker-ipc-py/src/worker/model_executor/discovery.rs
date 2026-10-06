//! Bind loaded numerical modules once, before requests begin.

use std::collections::HashMap;
use std::sync::Arc;

use indexmap::IndexMap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::{HostLane, ModelExecutor, NativeModelRunners, WorkerConfig, resources, streams};
use crate::worker::graph_storage::GraphStorage;

#[allow(clippy::too_many_arguments)]
pub(super) fn create(
    py: Python<'_>,
    model: Py<PyAny>,
    worker_config: Py<WorkerConfig>,
    bindings: Option<&Bound<'_, PyAny>>,
    entry_points: Option<&Bound<'_, PyAny>>,
    attention: Option<&str>,
    image_processor: Option<Py<PyAny>>,
    flow_prompt: Option<Py<PyAny>>,
    max_inflight: usize,
    expert_group: Option<Py<PyAny>>,
    attention_ranks: usize,
) -> PyResult<Py<ModelExecutor>> {
    let config = Arc::clone(&worker_config.borrow(py).inner);
    let device = py
        .import("uniserve.runtime.device")?
        .call_method1("canonical_device", (&config.device,))?;
    let options = PyDict::new(py);
    options.set_item("device", &device)?;
    options.set_item(
        "flashinfer",
        py.import("uniserve.runtime.backends.attention.flashinfer")?
            .call_method1("Backend", (worker_config.bind(py).getattr("flashinfer")?,))?,
    )?;
    let attention = py
        .import("uniserve.runtime.backends.attention")?
        .call_method(
            "resolve",
            (attention
                .or(config.attention_backend.as_deref())
                .unwrap_or("auto"),),
            Some(&options),
        )?;

    let inputs = py.import("uniserve_worker.bootstrap.inputs")?;
    let types = py.import("uniserve.model")?;
    let capability = |name: &str| inputs.call_method1("capability", (&model, types.getattr(name)?));
    let text = capability("CausalLM")?;
    let video_decoder = capability("VideoDecoder")?;
    let audio_decoder = capability("AudioDecoder")?;
    let video_postprocessor = capability("VideoPostprocessor")?;
    let image_builder = inputs.call_method1("image_builder", (&model,))?;
    let media_builder = inputs.call_method1("media_builder", (&model, &worker_config))?;
    let outputs = py
        .import("uniserve_worker.bootstrap.outputs")?
        .call_method1("resolve_outputs", (&model, &worker_config))?;
    let components = py.import("uniserve_worker.bootstrap.components")?;
    let options = PyDict::new(py);
    options.set_item("entries", entry_points)?;
    let declarations = components
        .call_method("describe_components", (&model,), Some(&options))?
        .cast_into::<PyDict>()?;
    let bindings = match bindings {
        Some(bindings) => py
            .get_type::<PyDict>()
            .call1((bindings,))?
            .cast_into::<PyDict>()?,
        None => local_bindings(py, &declarations, &device, config.rank, config.world_size)?,
    };
    let options = PyDict::new(py);
    options.set_item("declarations", &declarations)?;
    components.call_method("bind_components", (&model, &bindings), Some(&options))?;
    let mut numerical = config.role == "experts";
    for (_, binding) in bindings.iter() {
        numerical |= binding.getattr("calls")?.is_truthy()?;
    }
    let state_buffers = py
        .import("uniserve_worker.model_executor.resources")?
        .call_method1("media_state_buffers", (&bindings, &media_builder))?;
    let graph_storage = py.get_type::<GraphStorage>().call0()?.extract()?;
    let kernels = py
        .import("uniserve_worker.execution.kernel_table")?
        .call_method0("KernelRecords")?;
    let owner = Py::new(
        py,
        ModelExecutor {
            config,
            worker_config,
            model,
            expert_group: expert_group.unwrap_or_else(|| py.None()),
            attention_ranks,
            attention: attention.unbind(),
            processor: image_processor.unwrap_or_else(|| py.None()),
            flow_prompt: flow_prompt.unwrap_or_else(|| py.None()),
            image_builder: image_builder.unbind(),
            media_builder: media_builder.unbind(),
            text: text.unbind(),
            video_decoder: video_decoder.unbind(),
            audio_decoder: audio_decoder.unbind(),
            video_postprocessor: video_postprocessor.unbind(),
            outputs: outputs.clone().unbind(),
            declarations: declarations.unbind(),
            bindings: py
                .import("types")?
                .call_method1("MappingProxyType", (&bindings,))?
                .unbind(),
            numerical,
            state_buffers: state_buffers.unbind(),
            graph_storage: Some(graph_storage),
            flow_captures: PyTuple::empty(py).unbind(),
            flow_cfg_branches: Vec::new(),
            table_widths: Vec::new(),
            decode_predicates: py.None(),
            kv_cache: py.None(),
            expert_weights: py.None(),
            diffusion_bank: PyDict::new(py).unbind(),
            latent_pool: py.None(),
            canvas_slots: py.None(),
            noise_draws: None,
            kernels: kernels.unbind(),
            kernel_choices: 0,
            inner: NativeModelRunners::default(),
            modules: IndexMap::new(),
            output_names: HashMap::new(),
            streams: Arc::new(streams::Streams::new(max_inflight + 1)),
            closed: false,
            sealed: false,
            diffusion: None,
            diffusion_layouts: None,
            exchanges: Vec::new(),
            expert_executions: Vec::new(),
            expert_runners: Vec::new(),
        },
    )?;

    let initialized: PyResult<()> = (|| {
        for (name, binding) in bindings.iter() {
            for call in binding.getattr("calls")?.try_iter()? {
                let call = call?;
                let kinds = components.call_method1("call_kinds", ((&call,),))?;
                if !kinds.is_truthy()? {
                    continue;
                }
                let encoder = if call
                    .getattr("entry_point")?
                    .getattr("method")?
                    .extract::<String>()?
                    == "encode"
                {
                    Some(encoder_kind(&call.getattr("module")?)?.to_owned())
                } else {
                    None
                };
                let declared = outputs.call_method1("get", (&name, PyTuple::empty(py)))?;
                owner.borrow_mut(py).register(
                    py,
                    name.extract()?,
                    binding.clone().unbind(),
                    call.unbind(),
                    &kinds,
                    declared.cast()?,
                    encoder,
                )?;
            }
        }
        if owner.borrow(py).denoises(py) {
            let draws = HostLane::new(
                py,
                owner.borrow(py).config.max_request_pool_size,
                1,
                "worker-noise",
            )?;
            owner.borrow_mut(py).noise_draws = Some(Py::new(py, draws)?);
        }
        Ok(())
    })();
    if let Err(error) = initialized {
        if let Err(cleanup) = resources::close(owner.bind(py), true) {
            let _ = error.value(py).call_method1(
                "add_note",
                (format!("execution resource cleanup failed: {cleanup}"),),
            );
        }
        return Err(error);
    }
    Ok(owner)
}

fn local_bindings<'py>(
    py: Python<'py>,
    declarations: &Bound<'py, PyDict>,
    device: &Bound<'py, PyAny>,
    rank: usize,
    world_size: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let distributed = py.import("uniserve.distributed")?;
    let options = PyDict::new(py);
    options.set_item("device", device)?;
    let group = distributed
        .getattr("Communicator")?
        .call((PyTuple::new(py, 0..world_size)?, rank), Some(&options))?;
    let decoder = py.import("uniserve.model")?.getattr("VideoDecoder")?;
    let bindings = PyDict::new(py);
    for (name, calls) in declarations.iter() {
        let mut temporal = false;
        for call in calls.try_iter()? {
            temporal |= call?.getattr("module")?.is_instance(&decoder)?;
        }
        let options = PyDict::new(py);
        options.set_item("distribution", temporal.then_some("temporal_units"))?;
        let config = py
            .import("uniserve_worker.config.deployment")?
            .getattr("ComponentConfig")?
            .call(((rank,),), Some(&options))?;
        let dimensions: Vec<(String, usize)> = config
            .getattr("parallel_config")?
            .getattr("dimensions")?
            .extract()?;
        let options = PyDict::new(py);
        options.set_item("ranks", (rank,))?;
        options.set_item("rank", rank)?;
        options.set_item(
            "shape",
            PyTuple::new(py, dimensions.iter().map(|(_, size)| *size))?,
        )?;
        options.set_item(
            "axes",
            PyTuple::new(py, dimensions.iter().map(|(axis, _)| axis))?,
        )?;
        let mesh = distributed
            .getattr("DeviceMesh")?
            .call((), Some(&options))?;
        let binding = py
            .import("uniserve_worker.model_executor.component_binding")?
            .call_method1("ComponentBinding", (&name, config, &group, mesh, device))?;
        bindings.set_item(name, binding)?;
    }
    Ok(bindings)
}

fn encoder_kind(module: &Bound<'_, PyAny>) -> PyResult<&'static str> {
    let py = module.py();
    let types = py.import("uniserve.model")?;
    for (class, name) in [
        ("TextEncoder", "text"),
        ("PatchEncoder", "vision"),
        ("VideoEncoder", "video_condition"),
        ("AudioEncoder", "audio_condition"),
    ] {
        if module.is_instance(&types.getattr(class)?)? {
            return Ok(name);
        }
    }
    if module.is_instance(&py.import("uniserve.nn.vae")?.getattr("PatchAutoencoder")?)? {
        return Ok("latent");
    }
    Ok("conditioning")
}
