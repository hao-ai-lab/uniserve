//! A numerical binding's execution resources and CUDA graph dispatch.

mod canvas;
mod diffusion;
mod encoder;
mod text;
pub(super) use canvas::{CanvasRunner, SLOT_BUCKETS};
pub(super) use diffusion::{DenoisingBuffers, DenoisingSequence, DiffusionRunner};
pub(super) use encoder::EncoderRunner;
pub(super) use text::TextRunner;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};
use uniserve_worker_ipc::CallKind;

use super::cuda_graph::{CUDAGraphError, CUDAGraphRunner, batch as graph_batch};
use super::execution::{Execution, close_all};
use super::execution_context::ExecutionContext;
use super::graph_storage::GraphStorage;
use super::input_buffers::InputBuffers;
use super::model_inputs::InputBatch;
use super::model_results::ExecutionOutput;

/// One bound model call on a borrowed stream. Numerical subclasses implement
/// tensor preparation and forward; this owner retains inputs and graph pools.
#[pyclass(module = "uniserve_worker._uniserve_ipc", subclass)]
pub(crate) struct ModelRunner {
    #[pyo3(get)]
    pub(super) execution: Py<Execution>,
    #[pyo3(get)]
    pub(super) name: String,
    #[pyo3(get)]
    call: Py<PyAny>,
    #[pyo3(get)]
    pub(super) device: Py<PyAny>,
    #[pyo3(get)]
    model: Py<PyAny>,
    kinds: Vec<CallKind>,
    #[pyo3(get)]
    cuda_stream: Py<PyAny>,
    #[pyo3(get)]
    pub(super) input_buffers: Option<Py<InputBuffers>>,
    #[pyo3(get)]
    exact_graphs: bool,
    #[pyo3(get)]
    cache: Py<PyAny>,
    #[pyo3(get)]
    decode_predicates: Py<PyAny>,
    #[pyo3(get)]
    rank: usize,
    #[pyo3(get, set)]
    table_widths: Vec<usize>,
    // None denotes the single-runner case without a reference to self.
    peers: Option<Py<PyTuple>>,
}

#[pymethods]
impl ModelRunner {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (name, call, device, kinds, stream, context, inputs=None, *, storage, devices, exact_graphs=false, cache=None, predicates=None, rank=0, share=None))]
    fn new(
        py: Python<'_>,
        name: String,
        call: Py<PyAny>,
        device: Py<PyAny>,
        kinds: &Bound<'_, PyAny>,
        stream: Py<PyAny>,
        context: Py<ExecutionContext>,
        inputs: Option<Py<InputBuffers>>,
        storage: Py<GraphStorage>,
        devices: &Bound<'_, PyAny>,
        exact_graphs: bool,
        cache: Option<Py<PyAny>>,
        predicates: Option<Py<PyAny>>,
        rank: usize,
        share: Option<PyRef<'_, Self>>,
    ) -> PyResult<Self> {
        let method: String = call
            .bind(py)
            .getattr("entry_point")?
            .getattr("method")?
            .extract()?;
        let model = call.bind(py).getattr("module")?.unbind();
        let kinds = kinds
            .try_iter()?
            .map(|kind| Ok(pythonize::depythonize(&kind?)?))
            .collect::<PyResult<_>>()?;
        let execution = Execution::new(
            py,
            format!("{name}.{method}"),
            context.into_any(),
            devices,
            Some(storage),
            share.as_ref().map(|runner| runner.execution.bind(py)),
        )?;

        Ok(Self {
            execution,
            name,
            call,
            device,
            model,
            kinds,
            cuda_stream: stream,
            input_buffers: inputs,
            exact_graphs,
            cache: cache.unwrap_or_else(|| py.None()),
            decode_predicates: predicates.unwrap_or_else(|| py.None()),
            rank,
            table_widths: Vec::new(),
            peers: None,
        })
    }

    #[getter]
    fn peers(slf: &Bound<'_, Self>) -> PyResult<Py<PyTuple>> {
        match &slf.borrow().peers {
            Some(peers) => Ok(peers.clone_ref(slf.py())),
            None => Ok(PyTuple::new(slf.py(), [slf])?.unbind()),
        }
    }

    #[setter]
    fn set_peers(&mut self, peers: Py<PyTuple>) {
        self.peers = Some(peers);
    }

    #[getter]
    fn call_kinds(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let convert = crate::convert::RequestConversion::new(py)?;
        Ok(PyTuple::new(py, self.kinds.iter().map(|&kind| convert.kind(kind)))?.unbind())
    }

    /// Prepare the call in this lane's resident input backing.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (rows, *, forward_mode, attention=None, cache=None, tables=None, states=None))]
    fn prepare_inputs(
        &self,
        py: Python<'_>,
        rows: &Bound<'_, PyTuple>,
        forward_mode: &Bound<'_, PyAny>,
        attention: Option<&Bound<'_, PyAny>>,
        cache: Option<&Bound<'_, PyAny>>,
        tables: Option<&Bound<'_, PyAny>>,
        states: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let inputs = self
            .input_buffers
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("capability has no batched input buffers"))?;
        InputBuffers::prepare(
            inputs.bind(py),
            rows,
            pythonize::depythonize(forward_mode)?,
            attention,
            cache,
            tables,
            states,
        )
    }

    #[pyo3(signature = (_batch, *, padded=false))]
    fn batch_forward(&self, _batch: &Bound<'_, InputBatch>, padded: bool) -> PyResult<()> {
        let _ = padded;
        Err(PyNotImplementedError::new_err(
            "runner has no batched numerical forward",
        ))
    }

    fn graph_tokens(&self, _key: &Bound<'_, PyAny>) -> PyResult<usize> {
        Err(PyNotImplementedError::new_err(
            "runner has no bucketed graph",
        ))
    }

    fn expert_tokens(&self, batch: PyRef<'_, InputBatch>) -> PyResult<usize> {
        batch
            .query_tokens
            .ok_or_else(|| PyValueError::new_err("expert execution requires host query lengths"))
    }

    fn capture_plan(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let mut kinds = self.kinds.clone();
        kinds.sort_by_key(|kind| kind.as_str());
        let convert = crate::convert::RequestConversion::new(py)?;
        Ok(PyTuple::new(py, kinds.into_iter().map(|kind| convert.kind(kind)))?.unbind())
    }

    /// Exact numerical variants are captured at startup; a serving miss is eager.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (batch, *, eligible))]
    fn select_graph_shape(
        &self,
        py: Python<'_>,
        batch: Py<InputBatch>,
        eligible: bool,
    ) -> PyResult<Option<(Py<PyAny>, Py<InputBatch>, bool)>> {
        if !eligible || !self.exact_graphs || self.execution.borrow(py).pools.bind(py).is_empty() {
            return Ok(None);
        }
        let inputs = self
            .input_buffers
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("graph execution has no input buffers"))?;
        let graph = py.import("uniserve_worker.model_executor.graph_inputs")?;
        let widened = graph.call_method1(
            "widen_prefix",
            (batch, inputs.bind(py).getattr("table_widths")?),
        )?;
        let key = super::cuda_graph::attention::exact_key(widened.cast::<InputBatch>()?)?;
        Ok(Some((key.unbind(), widened.extract()?, false)))
    }

    fn resources<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        PyDict::new(py)
    }

    fn kernels(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.execution
            .borrow(py)
            .context
            .call_method0(py, "kernels")
    }

    fn close_graphs(&self, py: Python<'_>) -> PyResult<()> {
        Execution::close_graphs(self.execution.bind(py))
    }

    /// Graphs retire before their context and input storage. Attempt every
    /// release even when a numerical backend reports a cleanup failure.
    fn close(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let (execution, inputs) = {
            let owner = slf.borrow();
            (
                owner.execution.clone_ref(py),
                owner
                    .input_buffers
                    .as_ref()
                    .map(|inputs| inputs.clone_ref(py)),
            )
        };
        let graphs = slf.call_method0("close_graphs").map(drop);
        let context = Execution::close(execution.bind(py));
        let inputs = inputs.map_or(Ok(()), |inputs| inputs.borrow_mut(py).close(py));
        slf.borrow_mut().peers = Some(PyTuple::empty(py).unbind());
        close_all(py, [graphs, context, inputs])
    }

    fn batch_graph(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let execution = self.execution.borrow(py);
        let capacity = if execution.expert_order.is_some() {
            execution
                .context
                .bind(py)
                .getattr("experts")?
                .getattr("capacity")?
        } else {
            py.None().into_bound(py)
        };
        Ok(execution
            .buckets
            .bind(py)
            .get_item(key)?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?
            .get_item(capacity)?
            .unbind())
    }

    fn capture_graph(
        slf: &Bound<'_, Self>,
        _key: &Bound<'_, PyAny>,
        execution: Py<InputBatch>,
        forward: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let (owner, cache, predicates) = {
            let runner = slf.borrow();
            (
                runner.execution.clone_ref(py),
                runner.cache.clone_ref(py),
                runner.decode_predicates.clone_ref(py),
            )
        };
        let (context, pools) = {
            let owner = owner.borrow(py);
            (
                owner.context.extract::<Py<ExecutionContext>>(py)?,
                owner.pools.clone_ref(py),
            )
        };
        let joined = joining_experts(py, forward, context.clone_ref(py))?;
        let graph = graph_batch::capture_batch(
            py,
            context,
            execution,
            joined,
            Some(pools.bind(py).as_any()),
            Some(cache),
            Some(predicates),
            Some(owner.bind(py).getattr("warm_experts")?.unbind()),
        )?;
        Ok(Py::new(py, graph)?.into_any())
    }

    #[pyo3(signature = (key, execution, batch, *, borrow))]
    fn replay_graph(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        execution: Py<InputBatch>,
        batch: PyRef<'_, InputBatch>,
        borrow: bool,
    ) -> PyResult<Py<ExecutionOutput>> {
        let graph: Py<CUDAGraphRunner> = self.batch_graph(py, key)?.extract(py)?;
        let output = graph_batch::replay_batch(
            py,
            graph.borrow(py),
            execution,
            Some(batch.row_count),
            borrow,
        )?;
        if batch.decode_force_finish.is_none() {
            output.borrow_mut(py).greedy = None;
        }
        Ok(output)
    }

    #[staticmethod]
    fn result(py: Python<'_>, result: &Bound<'_, PyAny>) -> PyResult<Py<ExecutionOutput>> {
        Ok(py
            .import("uniserve_worker.model_executor.model_runner")?
            .call_method1("tensor_result", (result,))?
            .extract()?)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.execution)?;
        visit.call(&self.call)?;
        visit.call(&self.device)?;
        visit.call(&self.model)?;
        visit.call(&self.cuda_stream)?;
        visit.call(&self.input_buffers)?;
        visit.call(&self.cache)?;
        visit.call(&self.decode_predicates)?;
        visit.call(&self.peers)
    }

    fn __clear__(&mut self) {
        self.peers = None;
    }
}

/// Capture each forward with the unused expert layers of its collective step.
#[pyfunction]
pub(super) fn joining_experts(
    py: Python<'_>,
    forward: Py<PyAny>,
    context: Py<ExecutionContext>,
) -> PyResult<Py<PyAny>> {
    Ok(py
        .import("functools")?
        .call_method1(
            "partial",
            (
                wrap_pyfunction!(forward_with_experts, py)?,
                forward,
                context,
            ),
        )?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (forward, context, *args))]
fn forward_with_experts(
    py: Python<'_>,
    forward: Py<PyAny>,
    context: Py<ExecutionContext>,
    args: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    if let Some(experts) = context
        .borrow(py)
        .experts(py)
        .filter(|value| !value.is_none(py))
    {
        let experts = experts.bind(py);
        if experts.getattr("capacity")?.extract::<usize>()? > 0 {
            experts.call_method0("reset_layers")?;
        }
    }
    let result = forward.call1(py, args)?;
    ExecutionContext::join_expert_layers(context.bind(py))?;
    Ok(result)
}

/// Copy a smaller captured output into the largest bucket's retained backing.
/// Serial graph variants share addresses, including independent EP capacities.
fn copy_graph_output(value: &Bound<'_, PyAny>, backing: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let py = value.py();
    let shape: Vec<usize> = value.getattr("shape")?.extract()?;
    let capacity: Vec<usize> = backing.getattr("shape")?.extract()?;
    let Some((&rows, dimensions)) = shape.split_first() else {
        return Err(CUDAGraphError::new_err(
            "graph output requires a token axis",
        ));
    };
    let Some((&maximum, target_dimensions)) = capacity.split_first() else {
        return Err(CUDAGraphError::new_err(
            "graph output backing requires a token axis",
        ));
    };
    if rows > maximum
        || dimensions != target_dimensions
        || !value.getattr("dtype")?.eq(backing.getattr("dtype")?)?
        || !value.getattr("device")?.eq(backing.getattr("device")?)?
    {
        return Err(CUDAGraphError::new_err(
            "graphs capture largest first; output exceeds shared backing",
        ));
    }
    let output = backing.get_item(PySlice::new(py, 0, rows as isize, 1))?;
    output.call_method1("copy_", (value,))?;
    Ok(output.unbind())
}
