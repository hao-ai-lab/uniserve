//! Text graph buckets and output storage shared across their captured shapes.

mod outputs;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};
use uniserve_worker::TokenSelection;

use super::{ModelRunner, joining_experts};
use crate::worker::cuda_graph::{CUDAGraphError, CUDAGraphRunner, batch as graph_batch};
use crate::worker::execution::Execution;
use crate::worker::execution_context::ExecutionContext;
use crate::worker::graph_shapes::TextShapes;
use crate::worker::graph_storage::GraphStorage;
use crate::worker::host::with_context;
use crate::worker::input_buffers::InputBuffers;
use crate::worker::model_inputs::{InputBatch, parse_selections, selection_to_py};
use crate::worker::model_results::ExecutionOutput;

/// Prefill buckets share hidden output; decode buckets share vocabulary output.
/// The numerical subclass computes the backbone, projections and broadcasts.
#[pyclass(extends = ModelRunner, subclass, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct TextRunner {
    #[pyo3(get, set)]
    shapes: Py<TextShapes>,
    #[pyo3(get)]
    pipeline: Py<PyAny>,
    #[pyo3(get)]
    vocab: Py<PyAny>,
    prefill_output: Option<Py<PyAny>>,
    decode_output: Option<Py<PyAny>>,
}

#[pymethods]
impl TextRunner {
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
        share: Option<PyRef<'_, ModelRunner>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let (pipeline, vocab) = backend(py)?
            .call_method1("bind_vocabulary", (call.bind(py).getattr("module")?,))?
            .extract()?;
        let base = ModelRunner::new(
            py,
            name,
            call,
            device,
            kinds,
            stream,
            context,
            inputs,
            storage,
            devices,
            exact_graphs,
            cache,
            predicates,
            rank,
            share,
        )?;
        Ok(PyClassInitializer::from(base).add_subclass(Self {
            shapes: Py::new(
                py,
                TextShapes {
                    inner: uniserve_worker::TextShapes::new(Vec::new(), Vec::new()),
                },
            )?,
            pipeline,
            vocab,
            prefill_output: None,
            decode_output: None,
        }))
    }

    /// Execute a packed backbone and select independent per-sequence outputs.
    fn __call__(
        slf: &Bound<'_, Self>,
        inputs: &Bound<'_, PyAny>,
        selections: &Bound<'_, PyAny>,
    ) -> PyResult<Py<ExecutionOutput>> {
        outputs::forward(slf, inputs, &parse_selections(selections)?)
    }

    /// Gather vocabulary and hidden outputs using host-known query lengths.
    /// copy_hidden preserves results when subsequent forwards reuse the backing.
    #[pyo3(signature = (hidden, inputs, selections, *, copy_hidden=false))]
    fn select_outputs(
        slf: &Bound<'_, Self>,
        hidden: &Bound<'_, PyAny>,
        inputs: &Bound<'_, PyAny>,
        selections: &Bound<'_, PyAny>,
        copy_hidden: bool,
    ) -> PyResult<Py<ExecutionOutput>> {
        outputs::select(
            slf,
            hidden,
            inputs,
            &parse_selections(selections)?,
            copy_hidden,
        )
    }

    #[pyo3(signature = (batch, *, padded=false))]
    fn batch_forward(
        slf: &Bound<'_, Self>,
        batch: PyRef<'_, InputBatch>,
        padded: bool,
    ) -> PyResult<Py<ExecutionOutput>> {
        if padded && batch.token_selections.first() == Some(&TokenSelection::LastLogits) {
            return Ok(slf
                .call_method1("last_logits", (&batch.inputs,))?
                .extract()?);
        }
        outputs::forward(slf, batch.inputs.bind(slf.py()), &batch.token_selections)
    }

    fn graph_tokens(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        key.get_item(1)?.get_item(1)?.extract()
    }

    fn capture_plan(slf: PyRef<'_, Self>) -> PyResult<Py<PyTuple>> {
        let py = slf.py();
        Ok((
            slf.as_super().capture_plan(py)?,
            slf.shapes.bind(py).getattr("decode")?,
            slf.shapes.bind(py).getattr("prefill")?,
        )
            .into_pyobject(py)?
            .unbind())
    }

    /// Select configured buckets using host lengths, then bind padded views.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (batch, *, eligible))]
    fn select_graph_shape(
        slf: PyRef<'_, Self>,
        mut batch: Py<InputBatch>,
        eligible: bool,
    ) -> PyResult<Option<(Py<PyTuple>, Py<InputBatch>, bool)>> {
        let py = slf.py();
        let base = slf.as_super();
        if !eligible || base.execution.borrow(py).pools.bind(py).is_empty() {
            return Ok(None);
        }
        let graph = py.import("uniserve_worker.model_executor.graph_inputs")?;
        let selected =
            slf.shapes
                .borrow(py)
                .select_batch(py, &batch.borrow(py), &base.table_widths)?;
        let Some((shape, outputs)) = selected else {
            return Ok(None);
        };
        let shape = shape.into_bound(py);
        let decode: bool = shape.get_item(3)?.extract()?;
        let buffers = base
            .input_buffers
            .as_ref()
            .ok_or_else(|| CUDAGraphError::new_err("text graphs require input buffers"))?;

        // Direct token rows supply an inert finish column to the same decode
        // graph that serves resident continuations. Only the latter consumes it.
        if decode && batch.borrow(py).decode_force_finish.is_none() {
            let count = batch.borrow(py).row_count;
            let finish = buffers
                .bind(py)
                .getattr("decode_force_finish")?
                .get_item(PySlice::new(py, 0, count as isize, 1))?;
            finish.call_method0("zero_")?;
            let mut extended = batch.borrow(py).clone_ref(py);
            extended.decode_force_finish = Some(finish.unbind());
            batch = Py::new(py, extended)?;
        }
        let options = PyDict::new(py);
        options.set_item("buffers", buffers)?;
        let args = PyTuple::new(
            py,
            std::iter::once(batch.bind(py).as_any().clone())
                .chain(shape.iter())
                .collect::<Vec<_>>(),
        )?;
        let padded: Py<InputBatch> = graph
            .call_method("pad_text", args, Some(&options))?
            .extract()?;
        let value = padded.borrow(py);
        let inputs = value.inputs.bind(py);
        let attention = inputs
            .getattr("attention")?
            .getattr("entries")?
            .call_method0("values")?
            .try_iter()?
            .next()
            .ok_or_else(|| CUDAGraphError::new_err("text graph has no attention table"))??;
        let causal = if !decode && !attention.getattr("causal_values")?.is_none() {
            py.None().into_bound(py)
        } else {
            attention.getattr("causal")?.get_item(0)?
        };
        let selection = if decode {
            selection_to_py(py, value.token_selections[0])?.into_bound(py)
        } else {
            outputs.into_pyobject(py)?.to_owned().into_any()
        };
        let key = (
            if decode { "text" } else { "prefill" },
            shape,
            inputs.getattr("input_ids")?.getattr("dtype")?,
            inputs.getattr("positions")?.getattr("ndim")?,
            !inputs.getattr("embeddings")?.is_none(),
            causal,
            selection,
        )
            .into_pyobject(py)?
            .unbind();
        drop(value);
        Ok(Some((key, padded, true)))
    }

    fn capture_graph(
        slf: &Bound<'_, Self>,
        key: &Bound<'_, PyAny>,
        execution: Py<InputBatch>,
        forward: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let base = slf.as_any().cast::<ModelRunner>()?;
        if key.get_item(0)?.extract::<&str>()? != "prefill" {
            return ModelRunner::capture_graph(base, key, execution, forward);
        }
        let (owner, cache) = {
            let base = base.borrow();
            (base.execution.clone_ref(py), base.cache.clone_ref(py))
        };
        let (context, pools) = {
            let owner = owner.borrow(py);
            (
                owner.context.extract::<Py<ExecutionContext>>(py)?,
                owner.pools.clone_ref(py),
            )
        };
        let partial = py.import("functools")?.getattr("partial")?;
        let call = partial.call1((
            wrap_pyfunction!(prefill_forward, py)?,
            slf,
            key.get_item(-1)?,
        ))?;
        let call = joining_experts(py, call.unbind(), context.clone_ref(py))?;
        let warmup = partial
            .call1((owner.bind(py).getattr("warm_experts")?, &call))?
            .unbind();
        let graph = graph_batch::capture_hidden(
            py,
            context,
            execution,
            call,
            Some(pools.bind(py).as_any()),
            Some(cache),
            Some(warmup),
        )?;
        Ok(Py::new(py, graph)?.into_any())
    }

    #[pyo3(signature = (key, execution, batch, *, borrow))]
    fn replay_graph(
        slf: &Bound<'_, Self>,
        key: &Bound<'_, PyAny>,
        execution: Py<InputBatch>,
        batch: Py<InputBatch>,
        borrow: bool,
    ) -> PyResult<Py<ExecutionOutput>> {
        let py = slf.py();
        let base = slf.as_any().cast::<ModelRunner>()?;
        if key.get_item(0)?.extract::<&str>()? != "prefill" {
            return base
                .borrow()
                .replay_graph(py, key, execution, batch.borrow(py), borrow);
        }
        let graph: Py<CUDAGraphRunner> = base.borrow().batch_graph(py, key)?.extract(py)?;
        let hidden = graph_batch::replay_hidden(graph.borrow(py), execution.bind(py))?;
        let batch = batch.borrow(py);
        if !key.get_item(-1)?.extract::<bool>()? {
            return outputs::cache_rows(slf, batch.row_count);
        }
        let context: Py<ExecutionContext> =
            base.borrow().execution.borrow(py).context.extract(py)?;
        ExecutionContext::with_active(context.bind(py), || {
            outputs::select(
                slf,
                hidden.bind(py),
                batch.inputs.bind(py),
                &batch.token_selections,
                true,
            )
        })
    }

    fn close_graphs(slf: &Bound<'_, Self>) -> PyResult<()> {
        let execution = slf.borrow().as_super().execution.clone_ref(slf.py());
        let closed = Execution::close_graphs(execution.bind(slf.py()));
        let mut owner = slf.borrow_mut();
        owner.prefill_output = None;
        owner.decode_output = None;
        closed
    }

    /// Evaluate hidden states into backing shared by the prefill buckets.
    fn hidden_states(slf: &Bound<'_, Self>, inputs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let model = slf.borrow().as_super().model.clone_ref(slf.py());
        let hidden = model.call1(slf.py(), (inputs,))?;
        retain_output(slf, hidden, true)
    }

    /// Reuse the largest decode graph's vocabulary output across EP variants.
    fn decode_output(slf: &Bound<'_, Self>, values: Py<PyAny>) -> PyResult<Py<PyAny>> {
        retain_output(slf, values, false)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.shapes)?;
        visit.call(&self.pipeline)?;
        visit.call(&self.vocab)?;
        visit.call(&self.prefill_output)?;
        visit.call(&self.decode_output)
    }
}

fn retain_output(
    slf: &Bound<'_, TextRunner>,
    value: Py<PyAny>,
    prefill: bool,
) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    let backing = {
        let owner = slf.borrow();
        if prefill {
            &owner.prefill_output
        } else {
            &owner.decode_output
        }
        .as_ref()
        .map(|value| value.clone_ref(py))
    };
    let backing = match backing {
        Some(backing) => backing,
        None => {
            let capturing: bool = py
                .import("torch.cuda")?
                .call_method0("is_current_stream_capturing")?
                .extract()?;
            let backing = if prefill {
                if capturing {
                    return Err(CUDAGraphError::new_err(
                        "prefill output backing must exist before capture",
                    ));
                }
                let execution = slf.borrow().as_super().execution.clone_ref(py);
                let scope = execution
                    .bind(py)
                    .getattr("storage")?
                    .call_method1("allocate", (&execution,))?;
                with_context(&scope, || {
                    Ok(py
                        .import("torch")?
                        .call_method1("empty_like", (&value,))?
                        .unbind())
                })?
            } else {
                if capturing {
                    slf.borrow_mut().decode_output = Some(value.clone_ref(py));
                }
                return Ok(value);
            };
            slf.borrow_mut().prefill_output = Some(backing.clone_ref(py));
            backing
        }
    };

    super::copy_graph_output(value.bind(py), backing.bind(py))
}

#[pyfunction]
fn prefill_forward(
    slf: &Bound<'_, TextRunner>,
    outputs: bool,
    batch: PyRef<'_, InputBatch>,
) -> PyResult<Py<PyAny>> {
    if outputs {
        TextRunner::hidden_states(slf, batch.inputs.bind(slf.py()))
    } else {
        let model = slf.borrow().as_super().model.clone_ref(slf.py());
        model.call_method1(slf.py(), "fill_cache", (&batch.inputs,))
    }
}

fn backend(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("uniserve_worker.model_executor.text_runner")
}
