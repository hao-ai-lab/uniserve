//! Numerical runner bindings and batched dispatch.

mod binding;
mod diffusion;
mod discovery;
mod dispatch;
mod encoders;
mod execute;
mod experts;
mod forward;
mod graphs;
mod inputs;
mod modules;
mod outputs;
mod resources;
mod startup;
mod streams;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyTuple};
use uniserve_worker::ModelRunners as NativeModelRunners;
use uniserve_worker_ipc::{CallKind, MediaCall};

use crate::calls::Call;

use super::config::WorkerConfig;
use super::execution::Execution;
use super::graph_storage::GraphStorage;
use super::host::{HostLane, with_context};
use super::model_results::ExecutionOutput;
use modules::{Module, ensure_open, input_error};

/// One rank's model bindings, numerical executions and their resource lifetime.
/// Tensor construction and model mathematics remain in the numerical backend.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct ModelExecutor {
    pub(super) config: Arc<uniserve_worker::WorkerConfig>,
    #[pyo3(get)]
    pub(super) worker_config: Py<WorkerConfig>,
    #[pyo3(get)]
    pub(super) attention_ranks: usize,
    #[pyo3(get)]
    pub(super) numerical: bool,
    // Borrowed numerical modules and their resolved component placement.
    #[pyo3(get)]
    pub(super) model: Py<PyAny>,
    #[pyo3(get)]
    pub(super) expert_group: Py<PyAny>,
    #[pyo3(get)]
    pub(super) attention: Py<PyAny>,
    #[pyo3(get)]
    pub(super) processor: Py<PyAny>,
    #[pyo3(get)]
    pub(super) flow_prompt: Py<PyAny>,
    #[pyo3(get)]
    pub(super) image_builder: Py<PyAny>,
    #[pyo3(get)]
    pub(super) media_builder: Py<PyAny>,
    #[pyo3(get)]
    text: Py<PyAny>,
    #[pyo3(get)]
    pub(super) video_decoder: Py<PyAny>,
    #[pyo3(get)]
    pub(super) audio_decoder: Py<PyAny>,
    #[pyo3(get)]
    pub(super) video_postprocessor: Py<PyAny>,
    #[pyo3(get)]
    pub(super) bindings: Py<PyAny>,

    // Request storage is lent by Worker; executions retain its numerical views.
    #[pyo3(get)]
    pub(super) state_buffers: Py<PyAny>,
    #[pyo3(get)]
    pub(super) decode_predicates: Py<PyAny>,
    #[pyo3(get)]
    pub(super) kv_cache: Py<PyAny>,
    #[pyo3(get)]
    pub(super) expert_weights: Py<PyAny>,
    #[pyo3(get)]
    pub(super) latent_pool: Py<PyAny>,
    #[pyo3(get)]
    pub(super) canvas_slots: Py<PyAny>,
    #[pyo3(get)]
    pub(super) noise_draws: Option<Py<HostLane>>,
    #[pyo3(get)]
    pub(super) outputs: Py<PyAny>,
    declarations: Py<PyDict>,
    #[pyo3(get)]
    pub(super) diffusion_bank: Py<PyDict>,

    graph_storage: Option<Py<GraphStorage>>,
    #[pyo3(get)]
    pub(super) flow_captures: Py<PyTuple>,
    flow_cfg_branches: Vec<usize>,
    table_widths: Vec<usize>,
    kernels: Py<PyAny>,
    kernel_choices: usize,

    inner: NativeModelRunners<Py<PyAny>>,
    modules: IndexMap<(String, String, String), Arc<modules::Module>>,
    output_names: HashMap<String, Vec<String>>,
    // Creation order matters when closing communicators shared by ranks.
    streams: Arc<streams::Streams>,
    closed: bool,
    #[pyo3(get)]
    sealed: bool,
    diffusion: Option<Py<PyAny>>,
    diffusion_layouts: Option<Py<PyDict>>,
    exchanges: Vec<Py<PyAny>>,
    expert_executions: Vec<Py<Execution>>,
    // Buffered numerical calls in collective order, each with its microbatch
    // executions. Idle participation needs no Python runner lookup.
    expert_runners: Vec<Vec<Py<Execution>>>,
}

#[pymethods]
impl ModelExecutor {
    /// Standalone media denoising is distinct from KV-conditioned image batches.
    #[getter]
    pub(super) fn denoises(&self, py: Python<'_>) -> bool {
        !self.media_builder.is_none(py) && self.modules.values().any(|module| module.denoiser)
    }

    /// Borrow request storage before constructing the standalone denoiser.
    pub(super) fn bind_diffusion_storage(
        slf: &Bound<'_, Self>,
        bank: &Bound<'_, PyAny>,
        pool: Py<PyAny>,
    ) -> PyResult<()> {
        if slf.borrow().diffusion.is_some() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "bind request storage before preparing diffusion runners",
            ));
        }
        let bank = slf.py().get_type::<PyDict>().call1((bank,))?.extract()?;
        let mut owner = slf.borrow_mut();
        owner.diffusion_bank = bank;
        owner.latent_pool = pool;
        Ok(())
    }

    #[getter]
    pub(super) fn canvas_runner(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.inner
            .first(CallKind::Forward(
                uniserve_worker_ipc::ForwardMode::TokenDenoising,
            ))
            .map(|runner| runner.clone_ref(py))
    }

    pub(super) fn bind_canvas_slots(slf: &Bound<'_, Self>, slots: Py<PyAny>) -> PyResult<()> {
        let py = slf.py();
        let runner = slf.borrow().canvas_runner(py).ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("this rank runs no token denoising")
        })?;
        for peer in runner.bind(py).getattr("peers")?.try_iter()? {
            peer?.call_method1("bind_canvas_slots", (&slots,))?;
        }
        slf.borrow_mut().canvas_slots = slots;
        Ok(())
    }

    pub(super) fn diffusion_entry(&self, py: Python<'_>, call: &Call) -> PyResult<Py<PyAny>> {
        let runner = self
            .inner
            .get(&call.inner.component, CallKind::Media(MediaCall::Denoising));
        let class = py
            .import("uniserve_worker.model_executor.diffusion_runner")?
            .getattr("DiffusionRunner")?;
        if let Some(runner) = runner
            && runner.bind(py).is_instance(&class)?
        {
            return Ok(runner.clone_ref(py));
        }
        Err(crate::worker::error::invalid(
            py,
            "denoising call has no batched diffusion runner",
        ))
    }

    pub(super) fn call_devices(&self, py: Python<'_>, call: &Call) -> PyResult<Py<PyTuple>> {
        let device = py.import("uniserve.runtime.device")?;
        let binding = self
            .bindings
            .bind(py)
            .call_method1("get", (&call.inner.component,))?;
        let generation = !self.image_builder.is_none(py)
            && matches!(
                call.inner.code,
                CallKind::Media(
                    MediaCall::LatentPreparation | MediaCall::Denoising | MediaCall::ImageDecoding
                )
            );
        let source = if generation {
            device.call_method1(
                "canonical_device",
                (self
                    .config
                    .generation_device
                    .as_deref()
                    .unwrap_or(&self.config.device),),
            )?
        } else if binding.is_none() {
            device.call_method1("canonical_device", (&self.config.device,))?
        } else {
            binding.getattr("device")?
        };
        let input = match self.inner.get(&call.inner.component, call.inner.code) {
            Some(runner) => runner.bind(py).getattr("device")?,
            None => source.clone(),
        };
        Ok(PyTuple::new(py, [&source, &input, &source])?.unbind())
    }

    pub(super) fn image_processor(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if !self.processor.bind(py).is_instance(
            &py.import("uniserve.processing")?
                .getattr("ImageProcessor")?,
        )? {
            return Err(crate::worker::error::invalid(
                py,
                "call requires model image processing",
            ));
        }
        Ok(self.processor.clone_ref(py))
    }

    #[pyo3(signature = (kind, *values, **options))]
    fn run_encoder(
        slf: &Bound<'_, Self>,
        kind: &str,
        values: &Bound<'_, PyTuple>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<ExecutionOutput>> {
        encoders::run(slf, kind, values, options)
    }

    fn encode_vision(
        slf: &Bound<'_, Self>,
        inputs: &Bound<'_, PyAny>,
    ) -> PyResult<Py<ExecutionOutput>> {
        let module = slf.borrow().encoder_module(slf.py(), "vision")?;
        let args = PyTuple::new(slf.py(), [inputs])?;
        Self::run_module(
            slf,
            &module.name,
            &args,
            None,
            Some("encode"),
            Some(&module.path),
            None,
        )
    }

    #[pyo3(signature = (token_ids, *, visual=None, image_grids=None, video_grids=None))]
    fn encode_text(
        slf: &Bound<'_, Self>,
        token_ids: Vec<i64>,
        visual: Option<&Bound<'_, PyAny>>,
        image_grids: Option<&Bound<'_, PyTuple>>,
        video_grids: Option<&Bound<'_, PyTuple>>,
    ) -> PyResult<Py<ExecutionOutput>> {
        encoders::text(slf, token_ids, visual, image_grids, video_grids)
    }

    fn encode_conditioning(
        slf: &Bound<'_, Self>,
        features: &Bound<'_, PyAny>,
    ) -> PyResult<Py<ExecutionOutput>> {
        encoders::conditioning(slf, features)
    }

    fn prepare_text_tokens(slf: &Bound<'_, Self>, tokens: Vec<i64>) -> PyResult<Py<PyAny>> {
        encoders::tokens(slf, &tokens)
    }

    fn prepare_text_capacities(slf: &Bound<'_, Self>) -> PyResult<()> {
        encoders::prepare(slf)
    }

    #[pyo3(signature = (entry, output_index, media, decode, num_prompt_tokens, conditions=None))]
    pub(super) fn output_layout(
        slf: &Bound<'_, Self>,
        entry: &str,
        output_index: usize,
        media: &Bound<'_, PyAny>,
        decode: &Bound<'_, PyAny>,
        num_prompt_tokens: usize,
        conditions: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        outputs::layout(
            slf,
            entry,
            output_index,
            media,
            decode,
            num_prompt_tokens,
            conditions,
        )
    }

    pub(super) fn warmup(slf: &Bound<'_, Self>, storage: &Bound<'_, PyAny>) -> PyResult<()> {
        if !slf.borrow().media_builder.is_none(slf.py()) {
            let media = slf.py().import("uniserve_worker.execution.media")?;
            media.call_method1("prepare_denoising", (slf, storage))?;
            encoders::prepare(slf)?;
            media.call_method1("warmup_decoders", (slf,))?;
            media.call_method1("warmup_postprocess", (slf, storage))?;
        }
        resources::synchronize(slf)
    }

    pub(super) fn complete_startup(slf: &Bound<'_, Self>) -> PyResult<()> {
        startup::complete(slf)
    }

    /// Numerical declarations in binding order, for startup input construction.
    fn calls<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            self.modules
                .values()
                .map(|module| (&module.name, module.binding.bind(py), module.call.bind(py))),
        )
    }

    #[pyo3(signature = (kind, *, capability_type=None))]
    fn component(
        &self,
        py: Python<'_>,
        kind: &Bound<'_, PyAny>,
        capability_type: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let kind: CallKind = pythonize::depythonize(kind)?;
        let mut found = Vec::new();
        for module in self
            .modules
            .values()
            .filter(|module| module.kinds.contains(&kind))
        {
            let value = module.call.bind(py).getattr("module")?;
            if capability_type
                .map(|kind| value.is_instance(kind))
                .transpose()?
                .unwrap_or(true)
            {
                found.push(value);
            }
        }
        if found.len() != 1 {
            return Err(input_error(
                py,
                format!(
                    "computation {} requires one local capability",
                    kind.as_str()
                ),
            ));
        }
        Ok(found.remove(0).unbind())
    }

    #[getter]
    fn encoder_kinds<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        PyFrozenSet::new(
            py,
            self.modules
                .values()
                .filter_map(|module| module.encoder.as_deref()),
        )
    }

    /// Select the module's execution stream from the native call and outputs.
    /// Buffered calls use their batch runner's stream instead.
    pub(super) fn call_stream(slf: &Bound<'_, Self>, call: &Call) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let call = &call.inner;
        let (mut modules, condition) = {
            let runners = slf.borrow();
            if runners.inner.get(&call.component, call.code).is_some()
                || (call.code == CallKind::Media(MediaCall::LatentPreparation)
                    && runners
                        .inner
                        .get(&call.component, CallKind::Media(MediaCall::Denoising))
                        .is_some())
            {
                return Ok(py.None());
            }
            let modules: Vec<_> = runners
                .modules
                .values()
                .filter(|module| module.name == call.component && module.kinds.contains(&call.code))
                .cloned()
                .collect();
            let names = runners.output_names.get(&call.component);
            let has = |name: &str| {
                call.outputs.iter().any(|output| {
                    names
                        .and_then(|names| names.get(output.output_index as usize))
                        .is_some_and(|value| value == name)
                })
            };
            let condition = if call.code != CallKind::Media(MediaCall::LatentEncoding) {
                None
            } else if has("condition_video_latents") {
                Some("video_condition")
            } else if has("condition_audio_latents") {
                Some("audio_condition")
            } else {
                None
            };
            (modules, condition)
        };
        if modules.is_empty() {
            return Ok(py.None());
        }
        match call.code {
            // Initialization and conditioning share a call, but the denoiser
            // owns its stream. Encoders keep their ordinary tensor dependencies.
            CallKind::Media(MediaCall::LatentPreparation)
                if modules.iter().any(|module| module.denoiser) =>
            {
                modules.retain(|module| module.denoiser);
            }
            CallKind::Media(MediaCall::VideoDecoding) => {
                modules.retain(|module| module.video_decoder)
            }
            CallKind::Media(MediaCall::LatentEncoding) if condition.is_some() => {
                modules.retain(|module| module.encoder.as_deref() == condition);
            }
            _ => {}
        }
        if modules.len() != 1 {
            return Err(input_error(
                py,
                "call requires one bound numerical capability",
            ));
        }
        ensure_open(slf)?;
        let stream = modules[0].stream(slf)?;
        Ok(if stream.is_none() {
            stream
        } else {
            stream.getattr("stream")?
        }
        .unbind())
    }

    /// Prepare a numerical signature before its first invocation. Startup sizes
    /// share one graph pool: prepare all contexts before capturing any size.
    #[pyo3(signature = (name, size, *, method=None, path=None))]
    fn prepare_module(
        slf: &Bound<'_, Self>,
        name: &str,
        size: &Bound<'_, PyAny>,
        method: Option<&str>,
        path: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        ensure_open(slf)?;
        let module = slf.borrow().module(slf.py(), name, method, path)?;
        let key = slf
            .py()
            .import("uniserve_worker.model_executor.cuda_graph")?
            .call_method1("input_signature", (size,))?;
        module.prepare(slf, size, &key).map(Bound::unbind)
    }

    #[pyo3(signature = (name, *, method=None, path=None))]
    fn module_stream(
        slf: &Bound<'_, Self>,
        name: &str,
        method: Option<&str>,
        path: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        ensure_open(slf)?;
        let module = slf.borrow().module(slf.py(), name, method, path)?;
        module.stream(slf).map(Bound::unbind)
    }

    /// Prepare and invoke one numerical method, retiring a failed graph's
    /// context before propagating its error. Results own their tensor storage.
    #[pyo3(signature = (name, *args, size=None, method=None, path=None, **kwargs))]
    #[allow(clippy::too_many_arguments)]
    fn run_module(
        slf: &Bound<'_, Self>,
        name: &str,
        args: &Bound<'_, PyTuple>,
        size: Option<&Bound<'_, PyAny>>,
        method: Option<&str>,
        path: Option<&str>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<ExecutionOutput>> {
        let py = slf.py();
        let empty = PyDict::new(py);
        let kwargs = kwargs.unwrap_or(&empty);
        ensure_open(slf)?;
        let module = slf.borrow().module(py, name, method, path)?;
        let none = py.None().into_bound(py);
        let size = size.unwrap_or(&none);
        let key = py
            .import("uniserve_worker.model_executor.cuda_graph")?
            .call_method1("input_signature", (size,))?;
        with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
            let runner = module.prepare(slf, size, &key)?;
            let rank: usize = Arc::clone(&slf.borrow().config).rank;
            let result = with_context(
                &dispatch::profile(
                    py,
                    &format!("uniserve.model.module rank={rank} work={name}"),
                )?,
                || graphs::run_module(&runner, args, kwargs),
            );
            if let Err(error) = &result {
                let graph_error = py
                    .import("uniserve.runtime.cuda_graph")?
                    .getattr("CUDAGraphError")?;
                if error.value(py).is_instance(&graph_error)? {
                    module.remove(&key)?;
                }
            }
            let result = result?;
            ModelExecutor::report_new_kernels(slf)?;
            Ok(result)
        })
    }

    /// Bind numerical backing and graph buckets once for every local lane.
    #[pyo3(signature = (*, input_config, kv_cache, latent_pool, decode_predicates, max_calls, request_slots, latent_capacity_units, table_widths, max_inflight))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn configure_inputs(
        slf: &Bound<'_, Self>,
        input_config: &Bound<'_, PyAny>,
        kv_cache: &Bound<'_, PyAny>,
        latent_pool: &Bound<'_, PyAny>,
        decode_predicates: &Bound<'_, PyAny>,
        max_calls: usize,
        request_slots: usize,
        latent_capacity_units: usize,
        table_widths: Vec<usize>,
        max_inflight: usize,
    ) -> PyResult<()> {
        binding::configure(
            slf,
            input_config,
            kv_cache,
            latent_pool,
            decode_predicates,
            max_calls,
            request_slots,
            latent_capacity_units,
            table_widths,
            max_inflight,
        )
    }

    #[getter]
    pub(super) fn experts(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.exchanges
            .first()
            .map(|exchange| exchange.clone_ref(py))
    }

    pub(super) fn configure_experts(slf: &Bound<'_, Self>) -> PyResult<()> {
        experts::configure_worker(slf)
    }

    #[pyo3(signature = (*, leaving=false))]
    pub(super) fn join_expert_step(slf: &Bound<'_, Self>, leaving: bool) -> PyResult<bool> {
        experts::idle(slf, leaving)
    }

    #[pyo3(signature = (*, tokenizer, latents))]
    pub(super) fn capture(
        slf: &Bound<'_, Self>,
        tokenizer: &Bound<'_, PyAny>,
        latents: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        startup::prepare(slf, tokenizer, latents)
    }

    #[getter]
    pub(super) fn diffusion<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        diffusion::runner(slf)
    }

    fn prepare_layouts<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        diffusion::prepare_layouts(slf)
    }

    pub(super) fn diffusion_layout(
        slf: &Bound<'_, Self>,
        layout: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        diffusion::layout(slf, layout)
    }

    pub(super) fn run_denoising(
        slf: &Bound<'_, Self>,
        ladder: &Bound<'_, PyAny>,
        index: usize,
        bank: i64,
    ) -> PyResult<Py<ExecutionOutput>> {
        diffusion::run(slf, ladder, index, bank)
    }

    fn prepare_denoising(slf: &Bound<'_, Self>, storage: &Bound<'_, PyAny>) -> PyResult<()> {
        diffusion::prepare(slf, storage)
    }

    fn preparing_inputs(
        slf: &Bound<'_, Self>,
        transfers: Py<PyTuple>,
    ) -> PyResult<inputs::InputCopies> {
        let py = slf.py();
        ensure_open(slf)?;
        let device = py.import("uniserve.runtime.device")?.call_method1(
            "canonical_device",
            (&Arc::clone(&slf.borrow().config).device,),
        )?;
        if streams::cuda_index(&device)?.is_none() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "input preparation requires a live CUDA model executor",
            ));
        }
        let streams = Arc::clone(&slf.borrow().streams);
        Ok(inputs::InputCopies {
            owner: slf.clone().unbind(),
            stream: streams.preparation(&device)?.unbind(),
            transfers,
        })
    }

    pub(super) fn synchronize(slf: &Bound<'_, Self>) -> PyResult<()> {
        resources::synchronize(slf)
    }

    /// Destroy captured graphs after their borrowed output readers drain.
    /// Contexts and pools remain alive until the enclosing worker closes them.
    pub(super) fn close_graphs(slf: &Bound<'_, Self>) -> PyResult<()> {
        resources::close_graphs(slf)
    }

    /// Normal shutdown drains execution before releasing resources. An aborted
    /// worker retains backing until process exit without waiting on failed peers.
    #[pyo3(signature = (*, aborted=false))]
    pub(super) fn close(slf: &Bound<'_, Self>, aborted: bool) -> PyResult<()> {
        resources::close(slf, aborted)
    }

    /// Bind one loaded model to this rank and own its numerical executions.
    #[new]
    #[pyo3(signature = (model, worker_config, *, bindings=None, entry_points=None, attention=None, image_processor=None, flow_prompt=None, max_inflight=1, expert_group=None, attention_ranks=0))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
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
    ) -> PyResult<Py<Self>> {
        discovery::create(
            py,
            model,
            worker_config,
            bindings,
            entry_points,
            attention,
            image_processor,
            flow_prompt,
            max_inflight,
            expert_group,
            attention_ranks,
        )
    }

    #[getter]
    pub(super) fn graph_storage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, GraphStorage>> {
        self.graph_storage
            .as_ref()
            .map(|storage| storage.bind(py).clone())
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("model executor is closed"))
    }

    #[getter]
    fn table_widths<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.table_widths)
    }

    #[getter]
    fn flow_cfg_branches<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.flow_cfg_branches)
    }

    /// Evaluate a prepared numerical batch in its context, including experts.
    /// The caller orders input and output accesses with the runner's stream.
    fn run_eager(
        &self,
        py: Python<'_>,
        runner: &Bound<'_, PyAny>,
        batch: &Bound<'_, PyAny>,
        forward: &Bound<'_, PyAny>,
    ) -> PyResult<Py<ExecutionOutput>> {
        super::host::with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
            super::host::with_context(&context(runner)?.call_method0("activate")?, || {
                graphs::run_eager(runner, batch, forward)
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (tasks, *, cache, tables, states))]
    pub(super) fn forward(
        slf: &Bound<'_, Self>,
        tasks: Py<PyTuple>,
        cache: Py<PyAny>,
        tables: Py<PyAny>,
        states: Py<PyAny>,
    ) -> PyResult<forward::ModelBatches> {
        forward::prepare(
            slf.py(),
            &slf.borrow().inner,
            slf.clone().unbind(),
            tasks,
            cache,
            tables,
            states,
        )
    }

    /// Execute prepared numerical rows with native stream and result ordering.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (runner, rows, *, calls, cache, tables, states))]
    fn run_batch<'py>(
        slf: &Bound<'py, Self>,
        runner: &Bound<'py, PyAny>,
        rows: &Bound<'py, PyTuple>,
        calls: &Bound<'py, PyTuple>,
        cache: &Bound<'py, PyAny>,
        tables: &Bound<'py, PyAny>,
        states: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, ExecutionOutput>> {
        execute::run_batch(
            slf.py(),
            slf,
            runner,
            rows,
            calls,
            cache,
            tables,
            states,
            false,
        )
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.model)?;
        visit.call(&self.expert_group)?;
        visit.call(&self.attention)?;
        visit.call(&self.processor)?;
        visit.call(&self.flow_prompt)?;
        visit.call(&self.image_builder)?;
        visit.call(&self.media_builder)?;
        visit.call(&self.text)?;
        visit.call(&self.video_decoder)?;
        visit.call(&self.audio_decoder)?;
        visit.call(&self.video_postprocessor)?;
        visit.call(&self.bindings)?;
        visit.call(&self.state_buffers)?;
        visit.call(&self.decode_predicates)?;
        visit.call(&self.kv_cache)?;
        visit.call(&self.expert_weights)?;
        visit.call(&self.latent_pool)?;
        visit.call(&self.canvas_slots)?;
        visit.call(&self.noise_draws)?;
        visit.call(&self.worker_config)?;
        visit.call(&self.outputs)?;
        visit.call(&self.declarations)?;
        visit.call(&self.diffusion_bank)?;
        visit.call(&self.graph_storage)?;
        visit.call(&self.flow_captures)?;
        visit.call(&self.kernels)?;
        for runner in self.inner.iter() {
            visit.call(runner)?;
        }
        for module in self.modules.values() {
            // An active numerical call holds its own native reference. Its
            // Python resources remain roots until that call returns.
            if Arc::strong_count(module) == 1 {
                module.traverse(&visit)?;
            }
        }
        visit.call(&self.diffusion)?;
        visit.call(&self.diffusion_layouts)?;
        for exchange in &self.exchanges {
            visit.call(exchange)?;
        }
        for execution in &self.expert_executions {
            visit.call(execution)?;
        }
        for runner in self.expert_runners.iter().flatten() {
            visit.call(runner)?;
        }
        self.streams.traverse(&visit)?;
        Ok(())
    }

    fn __clear__(&mut self, py: Python<'_>) {
        self.clear();
        self.model = py.None();
        self.expert_group = py.None();
        self.attention = py.None();
        self.processor = py.None();
        self.flow_prompt = py.None();
        self.image_builder = py.None();
        self.media_builder = py.None();
        self.text = py.None();
        self.video_decoder = py.None();
        self.audio_decoder = py.None();
        self.video_postprocessor = py.None();
        self.bindings = py.None();
        self.state_buffers = py.None();
        self.decode_predicates = py.None();
        self.kv_cache = py.None();
        self.expert_weights = py.None();
        self.latent_pool = py.None();
        self.canvas_slots = py.None();
        self.noise_draws = None;
        self.kernels = py.None();
        self.outputs = py.None();
        self.declarations = PyDict::new(py).unbind();
        self.diffusion_bank = PyDict::new(py).unbind();
        self.flow_captures = PyTuple::empty(py).unbind();
        self.graph_storage = None;
    }

    fn get(
        &self,
        py: Python<'_>,
        component: &str,
        kind: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<PyAny>>> {
        Ok(self
            .inner
            .get(component, pythonize::depythonize(kind)?)
            .map(|runner| runner.clone_ref(py)))
    }
}

pub(super) fn execution<'py>(runner: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Execution>> {
    Ok(runner.getattr("execution")?.cast_into()?)
}

pub(super) fn context<'py>(runner: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    Ok(execution(runner)?
        .borrow()
        .context
        .bind(runner.py())
        .clone())
}

impl ModelExecutor {
    /// Bind whole-row numerical peers to their shared native execution group.
    fn bind_microbatches(&self, py: Python<'_>, peers: &Bound<'_, PyTuple>) -> PyResult<()> {
        let executions = peers
            .iter()
            .map(|peer| execution(&peer).map(Bound::unbind))
            .collect::<PyResult<Vec<_>>>()?;
        Execution::bind_microbatches(py, executions)?;
        for peer in peers {
            peer.setattr("peers", peers)?;
        }
        Ok(())
    }

    /// The caller drains streams and output readers before releasing contexts.
    /// Every close is attempted; shared scratch outlives the contexts using it.
    pub(super) fn close_modules(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let modules: Vec<_> = slf.borrow().modules.values().cloned().collect();
        let mut actions = Vec::new();
        for module in &modules {
            for cache in [&module.resident, &module.serving] {
                let cache = cache.bind(py);
                for (_, runner) in cache.iter() {
                    actions.push(runner.getattr("close")?);
                }
                cache.clear();
            }
        }
        for module in &modules {
            let scratch = module
                .scratch
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .as_ref()
                .map(|scratch| scratch.clone_ref(py));
            if let Some(scratch) = scratch {
                actions.push(scratch.bind(py).getattr("close")?);
            }
        }
        py.import("uniserve.runtime.resources")?
            .getattr("close_resources")?
            .call1(PyTuple::new(py, actions)?)?;
        Ok(())
    }
}

impl ModelExecutor {
    #[allow(clippy::too_many_arguments)]
    fn register(
        &mut self,
        py: Python<'_>,
        name: String,
        binding: Py<PyAny>,
        call: Py<PyAny>,
        kinds: &Bound<'_, PyAny>,
        outputs: &Bound<'_, PyTuple>,
        encoder: Option<String>,
    ) -> PyResult<()> {
        let path: String = call.bind(py).getattr("path")?.extract()?;
        let method: String = call
            .bind(py)
            .getattr("entry_point")?
            .getattr("method")?
            .extract()?;
        let runner_type = py
            .import("uniserve_worker.model_executor.model_runner")?
            .call_method1("runner_type", (call.bind(py).getattr("module")?,))?
            .unbind();
        let kinds = kinds
            .try_iter()?
            .map(|kind| Ok(pythonize::depythonize(&kind?)?))
            .collect::<PyResult<_>>()?;
        let module = call.bind(py).getattr("module")?;
        let types = py.import("uniserve.model")?;
        let denoiser = module.is_instance(&types.getattr("Denoiser")?)?;
        let video_decoder = module.is_instance(&types.getattr("VideoDecoder")?)?;
        if !self.output_names.contains_key(&name) {
            let names = outputs
                .iter()
                .map(|output| output.getattr("name")?.extract())
                .collect::<PyResult<_>>()?;
            self.output_names.insert(name.clone(), names);
        }
        self.modules.insert(
            (name.clone(), path.clone(), method.clone()),
            Arc::new(Module {
                name,
                path,
                method,
                kinds,
                encoder,
                denoiser,
                video_decoder,
                binding,
                call,
                runner_type,
                resident: PyDict::new(py).unbind(),
                serving: PyDict::new(py).unbind(),
                stream: Mutex::new(None),
                scratch: Mutex::new(None),
                streams: Arc::clone(&self.streams),
            }),
        );
        Ok(())
    }

    fn prepared<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let values = self.modules.values().flat_map(|module| {
            module
                .resident
                .bind(py)
                .iter()
                .chain(module.serving.bind(py).iter())
                .map(|(_, runner)| runner)
        });
        PyTuple::new(py, values.collect::<Vec<_>>())
    }

    fn all<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let mut runners: Vec<_> = slf
            .borrow()
            .inner
            .iter()
            .map(|runner| runner.bind(py).clone())
            .collect();
        runners.extend(slf.borrow().prepared(py)?.iter());
        if let Some(diffusion) = &slf.borrow().diffusion {
            runners.push(diffusion.bind(py).clone());
        }
        PyTuple::new(py, runners)
    }

    fn clear(&mut self) {
        self.inner.clear();
        self.modules.clear();
        self.output_names.clear();
        self.diffusion = None;
        self.diffusion_layouts = None;
        self.exchanges.clear();
        self.expert_executions.clear();
        self.expert_runners.clear();
        self.streams.clear();
    }
}

impl ModelExecutor {
    /// Capacity fitting replaces the immutable configuration before binding inputs.
    pub(super) fn set_config(&mut self, py: Python<'_>, config: Py<WorkerConfig>) {
        self.config = Arc::clone(&config.borrow(py).inner);
        self.worker_config = config;
    }
}
