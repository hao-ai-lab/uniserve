//! Prepared numerical resources and graph retirement.

mod denoising;
mod images;
mod joins;
pub(super) use denoising::DenoisingBuffers;
pub(super) use joins::JoinGraphs;

use std::collections::{BTreeMap, BTreeSet};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet};

use super::expert_exchange::ExpertExchange;
use super::graph_storage::GraphStorage;
use super::host::with_context;
use super::microbatches::Microbatches;

/// Graph variants of one numerical shape, indexed by expert transfer capacity.
/// Independent numerical calls use None. Every variant retains its backing.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct GraphBucket {
    graphs: BTreeMap<Option<usize>, Py<PyAny>>,
    pub(super) layers: BTreeSet<usize>,
}

#[pymethods]
impl GraphBucket {
    #[new]
    #[pyo3(signature = (graphs=None))]
    fn new(graphs: Option<BTreeMap<Option<usize>, Py<PyAny>>>) -> Self {
        Self {
            graphs: graphs.unwrap_or_default(),
            layers: BTreeSet::new(),
        }
    }

    fn __getitem__(&self, py: Python<'_>, capacity: Option<usize>) -> PyResult<Py<PyAny>> {
        self.get(py, capacity)
            .ok_or_else(|| PyKeyError::new_err(capacity))
    }

    fn __setitem__(&mut self, capacity: Option<usize>, graph: Py<PyAny>) {
        self.graphs.insert(capacity, graph);
    }

    fn __contains__(&self, capacity: Option<usize>) -> bool {
        self.graphs.contains_key(&capacity)
    }

    #[pyo3(signature = (capacity=None))]
    fn get(&self, py: Python<'_>, capacity: Option<usize>) -> Option<Py<PyAny>> {
        self.graphs.get(&capacity).map(|graph| graph.clone_ref(py))
    }

    #[getter]
    fn expert_layers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        PyFrozenSet::new(py, &self.layers)
    }

    #[setter]
    fn set_expert_layers(&mut self, layers: BTreeSet<usize>) {
        self.layers = layers;
    }

    fn close(slf: &Bound<'_, Self>) -> PyResult<()> {
        let graphs = std::mem::take(&mut slf.borrow_mut().graphs);
        close_all(
            slf.py(),
            graphs
                .into_values()
                .map(|graph| graph.call_method0(slf.py(), "close").map(drop)),
        )
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for graph in self.graphs.values() {
            visit.call(graph)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.graphs.clear();
    }
}

/// Own graph variants, allocator pools and one prepared numerical context.
/// External readers must drain before close; graphs retire before their context
/// and allocation pools. Microbatch peers share one native host rotation.
#[pyclass(module = "uniserve_worker._uniserve_ipc", weakref)]
pub(crate) struct Execution {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    pub(super) context: Py<PyAny>,
    #[pyo3(get)]
    pub(super) buckets: Py<PyDict>,
    #[pyo3(get)]
    pub(super) pools: Py<PyDict>,
    #[pyo3(get)]
    storage: Option<Py<GraphStorage>>,
    #[pyo3(get)]
    pub(super) microbatches: Option<Py<Microbatches>>,
    peer: usize,
    #[pyo3(get)]
    pub(super) sealed: bool,
    #[pyo3(get)]
    pub(super) expert_order: Option<i64>,
    pub(super) joins: Option<Py<JoinGraphs>>,
    pub(super) microbatch_joins: Option<Py<JoinGraphs>>,
    denoising: Option<denoising::Denoising>,
    pub(super) image_capacity: usize,
    closed: bool,
}

#[pymethods]
impl Execution {
    #[new]
    #[pyo3(signature = (name, context, *, devices, storage=None, share=None))]
    fn new(
        py: Python<'_>,
        name: String,
        context: Py<PyAny>,
        devices: &Bound<'_, PyAny>,
        storage: Option<Py<GraphStorage>>,
        share: Option<&Bound<'_, Self>>,
    ) -> PyResult<Py<Self>> {
        let storage = match storage {
            Some(storage) => storage,
            None => py.get_type::<GraphStorage>().call0()?.extract()?,
        };
        let owner = Py::new(
            py,
            Self {
                name,
                context,
                buckets: PyDict::new(py).unbind(),
                pools: PyDict::new(py).unbind(),
                storage: Some(storage.clone_ref(py)),
                microbatches: None,
                peer: 0,
                sealed: false,
                expert_order: None,
                joins: None,
                microbatch_joins: None,
                denoising: None,
                image_capacity: 0,
                closed: false,
            },
        )?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("share", share)?;
        let pools = storage
            .bind(py)
            .call_method("reserve", (&owner, devices), Some(&kwargs))?;
        owner.borrow_mut(py).pools = pools.extract()?;
        Ok(owner)
    }

    /// Encode image groups while retaining results across graph buffer reuse.
    pub(super) fn encode_images(
        slf: &Bound<'_, Self>,
        runner: &Bound<'_, PyAny>,
        inputs: &Bound<'_, PyAny>,
    ) -> PyResult<Py<super::model_results::ExecutionOutput>> {
        images::encode(slf, runner, inputs)
    }

    fn configure_denoising(&mut self, py: Python<'_>, maximum: Py<PyAny>) {
        self.denoising = Some(denoising::Denoising::new(py, maximum));
    }

    pub(super) fn prepare_denoising<'py>(
        slf: &Bound<'py, Self>,
        runner: &Bound<'py, PyAny>,
        layout: &Bound<'py, PyAny>,
        pages: usize,
    ) -> PyResult<Bound<'py, DenoisingBuffers>> {
        denoising::prepare(slf, runner, layout, pages)
    }

    pub(super) fn denoising_layout<'py>(
        slf: &Bound<'py, Self>,
        layout: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, DenoisingBuffers>> {
        denoising::layout(slf, layout)
    }

    pub(super) fn retire_denoising(
        slf: &Bound<'_, Self>,
        layout: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        denoising::retire(slf, layout)
    }

    pub(super) fn binds_denoising(
        slf: &Bound<'_, Self>,
        runner: &Bound<'_, PyAny>,
        ladder: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        denoising::binds(slf, runner, ladder)
    }

    pub(super) fn warm_denoising(
        slf: &Bound<'_, Self>,
        runner: &Bound<'_, PyAny>,
        ladder: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        denoising::warm(slf, runner, ladder)
    }

    pub(super) fn capture_denoising(
        slf: &Bound<'_, Self>,
        runner: &Bound<'_, PyAny>,
        ladder: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        denoising::capture(slf, runner, ladder)
    }

    pub(super) fn step_denoising(
        slf: &Bound<'_, Self>,
        runner: &Bound<'_, PyAny>,
        ladder: &Bound<'_, PyAny>,
        index: usize,
        bank: i64,
    ) -> PyResult<(Py<PyAny>, &'static str)> {
        denoising::step(slf, runner, ladder, index, bank)
    }

    fn seal(&mut self) {
        self.sealed = true;
    }

    /// Bind independently prepared contexts to one ordered host rotation.
    #[staticmethod]
    pub(super) fn bind_microbatches(py: Python<'_>, peers: Vec<Py<Self>>) -> PyResult<()> {
        let contexts = peers
            .iter()
            .map(|peer| peer.borrow(py).context.clone_ref(py))
            .collect();
        let rotation = Py::new(py, Microbatches::new(py, contexts)?)?;
        for (index, peer) in peers.into_iter().enumerate() {
            let mut peer = peer.borrow_mut(py);
            peer.peer = index;
            peer.microbatches = Some(rotation.clone_ref(py));
        }
        Ok(())
    }

    pub(super) fn begin_expert_step(&self, py: Python<'_>, capacity: usize) -> PyResult<()> {
        for context in self.contexts(py) {
            control(&context)?.borrow().begin(capacity)?;
        }
        Ok(())
    }

    pub(super) fn end_expert_step(&self, py: Python<'_>) -> PyResult<()> {
        for context in self.contexts(py) {
            control(&context)?.borrow().end();
        }
        Ok(())
    }

    /// Complete every peer's expert layers without touching model inputs.
    pub(super) fn join_expert_step(&self, py: Python<'_>, capacity: usize) -> PyResult<()> {
        self.begin_expert_step(py, capacity)?;
        let joined: PyResult<()> = (|| {
            let calls = self
                .contexts(py)
                .into_iter()
                .map(|context| context.getattr("join_expert_layers").map(Bound::unbind))
                .collect::<PyResult<Vec<_>>>()?;
            if let Some(rotation) = &self.microbatches {
                rotation.borrow(py).__call__(py, calls)?;
            } else {
                with_context(&self.context.bind(py).call_method0("activate")?, || {
                    calls[0].call0(py).map(drop)
                })?;
            }
            Ok(())
        })();
        let ended = self.end_expert_step(py);
        joined?;
        ended
    }

    /// Warm this peer while other contexts join with empty inputs. Only the
    /// selected peer writes the scratch request's KV pages.
    pub(super) fn warm_experts(
        &self,
        py: Python<'_>,
        call: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let Some(rotation) = &self.microbatches else {
            return Ok(call.call1((value,))?.unbind());
        };
        let mut calls = Vec::new();
        for (index, context) in self.contexts(py).into_iter().enumerate() {
            control(&context)?.borrow().reset_layers();
            let call = if index == self.peer {
                py.import("functools")?
                    .getattr("partial")?
                    .call1((call, value))?
            } else {
                context.getattr("join_expert_layers")?
            };
            calls.push(call.unbind());
        }
        let results = rotation.borrow(py).__call__(py, calls)?;
        Ok(results[self.peer].clone_ref(py))
    }

    /// Retire all captured variants after their external readers have drained.
    pub(super) fn close_graphs(slf: &Bound<'_, Self>) -> PyResult<()> {
        // Detach before running destructors, so cleanup can inspect the owner.
        let buckets = {
            let owner = slf.borrow();
            let buckets = owner.buckets.bind(slf.py());
            let values = buckets.values();
            buckets.clear();
            values
        };
        let (joins, microbatch_joins) = {
            let mut owner = slf.borrow_mut();
            owner.image_capacity = 0;
            (owner.joins.take(), owner.microbatch_joins.take())
        };
        close_all(
            slf.py(),
            buckets
                .iter()
                .map(|bucket| bucket.call_method0("close").map(drop))
                .chain(
                    joins
                        .into_iter()
                        .chain(microbatch_joins)
                        .map(|joins| joins.borrow(slf.py()).close(slf.py())),
                ),
        )
    }

    fn close(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let (context, storage, rotation, first) = {
            let mut owner = slf.borrow_mut();
            if owner.closed {
                return Ok(());
            }
            owner.closed = true;
            (
                owner.context.clone_ref(py),
                owner.storage.as_ref().map(|value| value.clone_ref(py)),
                owner.microbatches.take(),
                owner.peer == 0,
            )
        };
        let graphs = Self::close_graphs(slf);
        let rotated = match rotation.filter(|_| first) {
            Some(rotation) => rotation.borrow(py).close(py),
            None => Ok(()),
        };
        let context = context.call_method0(py, "close").map(drop);
        let denoising = slf.borrow_mut().denoising.take();
        let buffers = denoising.map_or(Ok(()), |value| value.close(py));
        let storage = storage.map_or(Ok(()), |storage| {
            storage.bind(py).call_method1("release", (slf,)).map(drop)
        });
        slf.borrow().pools.bind(py).clear();
        close_all(py, [graphs, rotated, context, buffers, storage])
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.context)?;
        visit.call(&self.buckets)?;
        visit.call(&self.pools)?;
        visit.call(&self.storage)?;
        visit.call(&self.microbatches)?;
        visit.call(&self.joins)?;
        visit.call(&self.microbatch_joins)?;
        if let Some(denoising) = &self.denoising {
            denoising.traverse(&visit)?;
        }
        Ok(())
    }

    fn __clear__(&mut self, py: Python<'_>) {
        self.buckets.bind(py).clear();
        self.pools.bind(py).clear();
        self.context = py.None();
        self.storage = None;
        self.microbatches = None;
        self.joins = None;
        self.microbatch_joins = None;
        self.denoising = None;
        self.image_capacity = 0;
    }
}

impl Execution {
    /// Capture every packed image count through the bounded vision capacity.
    pub(super) fn capture_images(
        slf: &Bound<'_, Self>,
        runner: &Bound<'_, PyAny>,
        max_images: usize,
        dtype: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        images::capture(slf, runner, max_images, dtype)
    }

    pub(super) fn has_denoising_layout(
        &self,
        py: Python<'_>,
        layout: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        self.denoising
            .as_ref()
            .map_or(Ok(false), |value| value.contains(py, layout))
    }

    fn contexts<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyAny>> {
        match &self.microbatches {
            Some(rotation) => rotation.borrow(py).contexts.bind(py).iter().collect(),
            None => vec![self.context.bind(py).clone()],
        }
    }
}

fn control<'py>(context: &Bound<'py, PyAny>) -> PyResult<Bound<'py, ExpertExchange>> {
    Ok(context
        .getattr("experts")?
        .getattr("_control")?
        .cast_into()?)
}

/// Attempt every release and retain the first error, adding later failures.
pub(super) fn close_all(
    py: Python<'_>,
    results: impl IntoIterator<Item = PyResult<()>>,
) -> PyResult<()> {
    let mut failure: Option<PyErr> = None;
    for result in results {
        if let Err(error) = result {
            if let Some(first) = &failure {
                let _ = first.value(py).call_method1(
                    "add_note",
                    (format!("Resource cleanup also failed: {error}"),),
                );
            } else {
                failure = Some(error);
            }
        }
    }
    failure.map_or(Ok(()), Err)
}

/// Order a numerical context between the caller's stream accesses, including
/// exceptional exits. Activation restores PyTorch's current stream.
pub(super) fn on_stream<T>(
    context: &Bound<'_, PyAny>,
    device: &Bound<'_, PyAny>,
    call: impl FnOnce() -> PyResult<T>,
) -> PyResult<T> {
    let py = context.py();
    let stream = context.getattr("stream")?;
    let cuda = py.import("torch.cuda")?;
    let current = || cuda.call_method1("current_stream", (device,));
    if !stream.is_none() {
        stream.call_method1("wait", (current()?,))?;
    }
    let result = with_context(&context.call_method0("activate")?, call);
    let joined = if stream.is_none() {
        Ok(())
    } else {
        current()?
            .call_method1("wait_stream", (stream.getattr("stream")?,))
            .map(drop)
    };
    let result = result?;
    joined?;
    Ok(result)
}
