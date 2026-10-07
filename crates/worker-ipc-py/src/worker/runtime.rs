//! Resource ownership and lifecycle of one model worker rank.

mod bootstrap;
pub(super) mod launch;
mod warmup;

use super::model_executor::ModelExecutor;
use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyType};

use super::buffer::BufferPool;
use super::canvas_slots::CanvasSlots;
use super::decode_state::DecodeState;
use super::events::EventPool;
use super::execution::close_all;
use super::executor::{Executor, Submission};
use super::host::HostLane;
use super::latent::LatentPool;
use super::output::OutputPool;
use super::request::RequestPool;
use super::storage::TensorStore;
use crate::PyServer;

/// Own a rank's model, storage, execution, and transports.
///
/// Construction allocates resources without graph capture or IPC I/O. Warmup
/// prepares numerical execution before admission. Normal close drains readers
/// before freeing backing; aborted close retains device resources until exit.
#[pyclass(module = "uniserve_worker._uniserve_ipc", subclass, weakref)]
pub(crate) struct Worker {
    #[pyo3(get)]
    worker_config: Py<PyAny>,
    pub(super) info: Option<uniserve_worker_ipc::WorkerInfo>,
    info_view: Option<Py<PyAny>>,
    device_product_bytes: u64,

    // Resources are filled during construction and released on normal close.
    #[pyo3(get)]
    model: Option<Py<PyAny>>,
    #[pyo3(get)]
    runner: Option<Py<ModelExecutor>>,
    #[pyo3(get)]
    tokenizer: Option<Py<PyAny>>,
    #[pyo3(get)]
    attention: Option<Py<PyAny>>,
    #[pyo3(get)]
    sampling_group: Option<Py<PyAny>>,
    #[pyo3(get)]
    process_groups: Option<Py<super::process_groups::ProcessGroups>>,

    #[pyo3(get)]
    requests: Option<Py<RequestPool>>,
    #[pyo3(get)]
    decode_state: Option<Py<DecodeState>>,
    #[pyo3(get)]
    block_tables: Option<Py<PyAny>>,
    #[pyo3(get)]
    kv_cache: Option<Py<PyAny>>,
    #[pyo3(get)]
    latent_pool: Option<Py<LatentPool>>,
    #[pyo3(get)]
    canvas_slots: Option<Py<CanvasSlots>>,

    #[pyo3(get)]
    output_pool: Option<Py<OutputPool>>,
    #[pyo3(get)]
    tensor_store: Option<Py<TensorStore>>,
    #[pyo3(get)]
    buffer_pool: Option<Py<BufferPool>>,
    #[pyo3(get)]
    device_events: Option<Py<EventPool>>,
    #[pyo3(get)]
    host_tasks: Option<Py<HostLane>>,

    #[pyo3(get)]
    transports: Option<Py<PyDict>>,
    #[pyo3(get)]
    export_transports: Option<Py<PyDict>>,

    #[pyo3(get)]
    ipc_endpoint: Option<Py<PyServer>>,
    #[pyo3(get)]
    pub(super) profiler: Option<Py<super::profiling::WorkerProfiler>>,
    #[pyo3(get)]
    codec_slot: bool,

    executor: Option<Py<Executor>>,
    closed: bool,
    warmed_up: bool,
    run_started: bool,
}

#[pymethods]
impl Worker {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (model, *, worker_config, sampling_group, tokenizer,
        allowed_calls, queue_depth, completion_payload_bytes,
        acknowledgment_slot=0, host_slots=Vec::new(), products_cross_hosts=false,
        attention=None, transfer_backends=vec!["local".to_owned()],
        export_backends=vec!["local".to_owned()], worker_id="worker",
        image_processor=None, flow_prompt=None, components=None,
        process_groups=None, bindings=None, model_name=None, attention_ranks=0,
        entry_points=None))]
    fn new(
        py: Python<'_>,
        model: Py<PyAny>,
        worker_config: Py<PyAny>,
        sampling_group: Option<Py<PyAny>>,
        tokenizer: Option<Py<PyAny>>,
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
        process_groups: Option<Py<super::process_groups::ProcessGroups>>,
        bindings: Option<Py<PyAny>>,
        model_name: Option<Py<PyAny>>,
        attention_ranks: usize,
        entry_points: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let mut worker = Self {
            worker_config,
            info: None,
            info_view: None,
            device_product_bytes: 0,
            model: None,
            sampling_group,
            tokenizer,
            process_groups,
            runner: None,
            attention: None,
            requests: None,
            decode_state: None,
            block_tables: None,
            kv_cache: None,
            latent_pool: None,
            canvas_slots: None,
            output_pool: None,
            tensor_store: None,
            buffer_pool: None,
            device_events: None,
            host_tasks: None,
            transports: None,
            export_transports: None,
            ipc_endpoint: None,
            profiler: None,
            codec_slot: false,
            executor: None,
            closed: false,
            warmed_up: false,
            run_started: false,
        };
        let result = worker.allocate(
            py,
            model.bind(py),
            allowed_calls,
            queue_depth,
            completion_payload_bytes,
            acknowledgment_slot,
            host_slots,
            products_cross_hosts,
            attention,
            transfer_backends,
            export_backends,
            worker_id,
            image_processor,
            flow_prompt,
            components,
            bindings,
            model_name,
            attention_ranks,
            entry_points,
        );
        if let Err(error) = result {
            if worker.model.is_none() {
                return Err(error);
            }
            // Keep partial allocations alive while GPU accesses may still be
            // in flight. Aborting the host lane and runner does not wait.
            worker.closed = true;
            let worker = Py::new(py, worker)?;
            let cleanup = (|| {
                py.import("uniserve.runtime.resources")?
                    .call_method1("retain_until_exit", (&worker,))?;
                let worker = worker.borrow(py);
                let mut results = Vec::new();
                if let Some(host) = &worker.host_tasks {
                    host.borrow(py).abort();
                }
                if let Some(runner) = &worker.runner {
                    results.push(ModelExecutor::close(runner.bind(py), true));
                }
                close_all(py, results)
            })();
            return close_all(py, [Err(error), cleanup]).map(|()| unreachable!());
        }
        Ok(worker)
    }

    /// Load the model and build its rank-local resources. Process groups are
    /// retained after a construction failure; the caller must exit the process.
    #[classmethod]
    fn from_config<'py>(
        cls: &Bound<'py, PyType>,
        config: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        bootstrap::from_config(cls, config)
    }

    /// Materialize the Python report only for a direct library consumer.
    #[getter]
    fn info(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if let Some(view) = &slf.borrow().info_view {
            return Ok(view.clone_ref(py));
        }
        let view = super::capacity::report::info_to_py(
            py,
            slf.borrow().info.as_ref().ok_or_else(closed)?,
        )?
        .unbind();
        slf.borrow_mut().info_view = Some(view.clone_ref(py));
        Ok(view)
    }

    /// The shared executor is initialized on first execution, after all resource
    /// owners exist. Both direct calls and the IPC service use this instance.
    #[getter]
    fn executor(slf: &Bound<'_, Self>) -> PyResult<Py<Executor>> {
        let py = slf.py();
        if let Some(executor) = &slf.borrow().executor {
            return Ok(executor.clone_ref(py));
        }
        slf.borrow().require_open()?;
        let executor = Py::new(py, Executor::new(py, slf.as_any())?)?;
        slf.borrow_mut().executor = Some(executor.clone_ref(py));
        Ok(executor)
    }

    fn __enter__(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
        slf.borrow().require_open()?;
        Ok(slf)
    }

    fn __exit__(
        slf: &Bound<'_, Self>,
        _exc_type: &Bound<'_, PyAny>,
        exc_value: &Bound<'_, PyAny>,
        traceback: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let py = slf.py();
        let aborted = !exc_value.is_none();
        if aborted {
            // Record the initiating failure before teardown, which itself may
            // encounter a failed CUDA context or an unavailable peer.
            let options = PyDict::new(py);
            options.set_item("exc_info", (exc_value.get_type(), exc_value, traceback))?;
            py.import("logging")?
                .call_method1("getLogger", ("uniserve_worker.worker",))?
                .call_method(
                    "error",
                    ("worker execution failed before resource cleanup",),
                    Some(&options),
                )?;
        }
        match Self::close(slf, aborted) {
            Err(error) if aborted => {
                exc_value.call_method1(
                    "add_note",
                    (format!("Resource cleanup also failed: {error}"),),
                )?;
                Ok(())
            }
            result => result,
        }
    }

    /// Prepare and capture numerical execution once, before the first submission.
    fn warmup(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let runner = {
            let worker = slf.borrow();
            worker.require_open()?;
            if worker.warmed_up {
                return Ok(());
            }
            worker.runner.as_ref().ok_or_else(closed)?.clone_ref(py)
        };
        let executor = Self::executor(slf)?;
        if executor.borrow(py).started() {
            return Err(PyRuntimeError::new_err(
                "warmup must precede the first serving submission",
            ));
        }
        if runner.borrow(py).numerical {
            slf.borrow().bind_graph_budgets(py)?;
            let (slots, tokenizer, latents) = {
                let worker = slf.borrow();
                (
                    PyTuple::new(
                        py,
                        &worker
                            .requests
                            .as_ref()
                            .ok_or_else(closed)?
                            .borrow(py)
                            .storage
                            .borrow(py)
                            .tensor_slots,
                    )?
                    .into_any(),
                    worker.tokenizer.as_ref().map(|value| value.clone_ref(py)),
                    worker.latent_pool.as_ref().map(|value| value.clone_ref(py)),
                )
            };
            ModelExecutor::warmup(runner.bind(py), &slots)?;
            ModelExecutor::capture(
                runner.bind(py),
                &tokenizer
                    .map(|value| value.into_bound(py))
                    .unwrap_or_else(|| py.None().into_bound(py)),
                &latents
                    .map(|value| value.into_bound(py).into_any())
                    .unwrap_or_else(|| py.None().into_bound(py)),
            )?;
            warmup::run(slf)?;
        }
        if slf.borrow().codec_slot {
            let media = py.import("uniserve_worker.media.container")?;
            media.call_method1(
                "require_media_codecs",
                (media.getattr("VIDEO_CODEC")?, media.getattr("AUDIO_CODEC")?),
            )?;
        }
        ModelExecutor::complete_startup(runner.bind(py))?;
        if slf
            .borrow()
            .requests
            .as_ref()
            .ok_or_else(closed)?
            .bind(py)
            .call_method0("request_ids")?
            .is_truthy()?
        {
            return Err(PyRuntimeError::new_err(
                "startup completed with resident requests",
            ));
        }
        // Synthetic startup batches consume IDs and collective sequence numbers;
        // reset only once their requests have all retired.
        executor.borrow_mut(py).reset(py)?;
        {
            let worker = slf.borrow();
            super::capacity::check_startup_storage(
                worker.worker_config.bind(py),
                worker.device_product_bytes,
                worker
                    .tensor_store
                    .as_ref()
                    .ok_or_else(closed)?
                    .bind(py)
                    .as_any(),
            )?;
        }
        slf.borrow_mut().warmed_up = true;
        Ok(())
    }

    /// Drain readers and release backing in dependency order. An aborted rank
    /// retains asynchronous resources until exit without waiting on peers or GPU.
    /// The borrowed IPC endpoint remains open in either case.
    #[pyo3(signature = (*, aborted=false))]
    fn close(slf: &Bound<'_, Self>, aborted: bool) -> PyResult<()> {
        let py = slf.py();
        {
            let mut worker = slf.borrow_mut();
            if worker.closed {
                return Ok(());
            }
            worker.closed = true;
        }
        let result = if aborted {
            py.import("uniserve.runtime.resources")?
                .call_method1("retain_until_exit", (slf,))?;
            slf.borrow().abort(py)
        } else {
            slf.borrow().drain(py)
        };
        let mut worker = slf.borrow_mut();
        worker.ipc_endpoint = None;
        if !aborted {
            worker.release();
        }
        result
    }

    /// Wake result polling when a host task, transfer or CUDA producer completes.
    fn set_completion_wake(
        &self,
        py: Python<'_>,
        wake: Option<Py<PyAny>>,
        wake_on_stream: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        if let Some(events) = &self.device_events {
            events.borrow(py).set_completion_wake(wake_on_stream);
        }
        if let Some(host) = &self.host_tasks {
            host.borrow(py)
                .set_completion_wake(wake.as_ref().map(|value| value.clone_ref(py)));
        }
        if let Some(transports) = &self.transports {
            for transport in transports.bind(py).values() {
                transport.call_method1("set_completion_wake", (&wake,))?;
            }
        }
        if let Some(cache) = &self.kv_cache {
            cache
                .bind(py)
                .getattr("imports")?
                .call_method1("set_completion_wake", (wake,))?;
        }
        Ok(())
    }

    /// Borrow an open endpoint for one synchronous service run.
    fn bind<'py>(
        slf: Bound<'py, Self>,
        endpoint: Option<Bound<'py, PyServer>>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        {
            let worker = slf.borrow();
            worker.require_open()?;
            if worker.ipc_endpoint.is_some() {
                return Err(PyRuntimeError::new_err(
                    "worker already has a bound IPC endpoint",
                ));
            }
        }
        let endpoint = endpoint
            .ok_or_else(|| PyValueError::new_err("worker binding requires an open IPC endpoint"))?;
        if endpoint.borrow().closed()? {
            return Err(PyValueError::new_err(
                "worker binding requires an open IPC endpoint",
            ));
        }
        let profiler = Py::new(py, super::profiling::WorkerProfiler::from_env(None)?)?;
        let wake = endpoint.getattr("wake")?.unbind();
        let wake_on_stream = endpoint.getattr("wake_on_stream")?.unbind();
        {
            let mut worker = slf.borrow_mut();
            worker.profiler = Some(profiler);
            worker.ipc_endpoint = Some(endpoint.unbind());
        }
        slf.borrow()
            .set_completion_wake(py, Some(wake), Some(wake_on_stream))?;
        Ok(slf)
    }

    /// Warm up and run the bound native service once, on the calling thread.
    fn run(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let endpoint = {
            let mut worker = slf.borrow_mut();
            worker.require_open()?;
            if worker.run_started {
                return Err(PyRuntimeError::new_err("worker can only run once"));
            }
            let endpoint = worker
                .ipc_endpoint
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("worker has no bound IPC endpoint"))?
                .clone_ref(py);
            worker.run_started = true;
            endpoint
        };
        Self::warmup(slf)?;
        Self::executor(slf)?
            .borrow_mut(py)
            .serve(py, &endpoint.borrow(py))
    }

    /// Admit a batch. Queue saturation leaves its ID available for resubmission.
    #[pyo3(signature = (batch, *, propagate_errors=false))]
    fn submit(
        slf: &Bound<'_, Self>,
        batch: &Bound<'_, crate::batches::Batch>,
        propagate_errors: bool,
    ) -> PyResult<Submission> {
        slf.borrow().require_open()?;
        Self::executor(slf)?
            .borrow_mut(slf.py())
            .submit(slf.py(), batch, propagate_errors)
    }

    /// Progress admitted work without consuming results.
    fn advance(slf: &Bound<'_, Self>) -> PyResult<()> {
        slf.borrow().require_open()?;
        Self::executor(slf)?
            .borrow_mut(slf.py())
            .advance(slf.py())
            .map(drop)
    }

    /// Consume a submission's result, or return None while it is pending.
    fn poll(slf: &Bound<'_, Self>, submission: &Submission) -> PyResult<Option<Py<PyAny>>> {
        slf.borrow().require_open()?;
        Self::executor(slf)?
            .borrow_mut(slf.py())
            .poll(slf.py(), submission)
    }

    /// Release a drained request and its admission.
    fn drop_request(slf: &Bound<'_, Self>, request_id: u64) -> PyResult<()> {
        slf.borrow().require_open()?;
        Self::executor(slf)?
            .borrow_mut(slf.py())
            .drop_request(slf.py(), request_id)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.worker_config)?;
        visit.call(&self.info_view)?;
        visit.call(&self.executor)?;
        visit.call(&self.model)?;
        visit.call(&self.runner)?;
        visit.call(&self.tokenizer)?;
        visit.call(&self.attention)?;
        visit.call(&self.sampling_group)?;
        visit.call(&self.process_groups)?;
        visit.call(&self.requests)?;
        visit.call(&self.decode_state)?;
        visit.call(&self.block_tables)?;
        visit.call(&self.kv_cache)?;
        visit.call(&self.latent_pool)?;
        visit.call(&self.canvas_slots)?;
        visit.call(&self.output_pool)?;
        visit.call(&self.tensor_store)?;
        visit.call(&self.buffer_pool)?;
        visit.call(&self.device_events)?;
        visit.call(&self.host_tasks)?;
        visit.call(&self.transports)?;
        visit.call(&self.export_transports)?;
        visit.call(&self.ipc_endpoint)?;
        visit.call(&self.profiler)?;
        Ok(())
    }

    fn __clear__(&mut self) {
        self.closed = true;
        self.release();
    }
}

impl Worker {
    fn require_open(&self) -> PyResult<()> {
        if self.closed {
            return Err(closed());
        }
        Ok(())
    }

    fn release(&mut self) {
        self.executor = None;
        self.model = None;
        self.runner = None;
        self.tokenizer = None;
        self.attention = None;
        self.sampling_group = None;
        self.process_groups = None;
        self.requests = None;
        self.decode_state = None;
        self.block_tables = None;
        self.kv_cache = None;
        self.latent_pool = None;
        self.canvas_slots = None;
        self.output_pool = None;
        self.tensor_store = None;
        self.buffer_pool = None;
        self.device_events = None;
        self.host_tasks = None;
        self.transports = None;
        self.export_transports = None;
        self.ipc_endpoint = None;
        self.profiler = None;
    }

    fn abort(&self, py: Python<'_>) -> PyResult<()> {
        let mut results = vec![self.set_completion_wake(py, None, None)];
        if let Some(host) = &self.host_tasks {
            host.borrow(py).abort();
        }
        if let Some(runner) = &self.runner {
            results.push(ModelExecutor::close(runner.bind(py), true));
        }
        if let Some(groups) = &self.process_groups {
            results.push(super::process_groups::ProcessGroups::close(
                groups.bind(py),
                true,
            ));
        }
        close_all(py, results)
    }

    fn drain(&self, py: Python<'_>) -> PyResult<()> {
        let runner = self.runner.as_ref().map(|value| value.bind(py));
        let mut results = Vec::new();
        if let Some(runner) = runner {
            results.push(ModelExecutor::synchronize(runner));
        }
        if let Some(profiler) = &self.profiler {
            profiler.borrow_mut(py).close(py);
        }
        if let Some(executor) = &self.executor {
            results.push(executor.borrow_mut(py).close(py));
        }
        if let Some(host) = &self.host_tasks {
            results.push(host.borrow(py).close(py));
        }

        // Stop remote and host readers before destroying graph executables or
        // the KV, canvas and latent storage they borrow.
        if let Some(cache) = &self.kv_cache {
            results.push(
                cache
                    .bind(py)
                    .getattr("imports")
                    .and_then(|imports| imports.call_method0("stop"))
                    .map(drop),
            );
        }
        if let Some(transports) = &self.transports {
            results.extend(
                transports
                    .bind(py)
                    .values()
                    .iter()
                    .map(|transport| transport.call_method0("close").map(drop)),
            );
        }
        if let Some(output) = &self.output_pool {
            results.push(output.borrow(py).close(py));
        }
        if let Some(runner) = runner {
            results.push(ModelExecutor::close_graphs(runner));
        }
        if let Some(cache) = &self.kv_cache {
            results.push(cache.call_method0(py, "close").map(drop));
        }
        if let Some(canvas) = &self.canvas_slots {
            canvas.borrow(py).close(py);
        }
        if let Some(latents) = &self.latent_pool {
            results.push(LatentPool::close(latents.bind(py).clone()));
        }
        if let Some(tensors) = &self.tensor_store {
            results.push(tensors.bind(py).call_method0("close").map(drop));
        }
        if let Some(buffers) = &self.buffer_pool {
            results.push(buffers.borrow(py).close(py));
        }
        if let Some(events) = &self.device_events {
            results.push(events.borrow(py).close(py));
        }
        if let Some(runner) = runner {
            results.push(ModelExecutor::close(runner, false));
        }
        if let Some(requests) = &self.requests {
            results.push(requests.borrow_mut(py).close(py));
        }
        if let Some(groups) = &self.process_groups {
            results.push(super::process_groups::ProcessGroups::close(
                groups.bind(py),
                false,
            ));
        }
        results.push(self.set_completion_wake(py, None, None));
        close_all(py, results)
    }
}

fn closed() -> PyErr {
    PyRuntimeError::new_err("worker is closed")
}
