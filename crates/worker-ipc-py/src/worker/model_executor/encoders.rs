//! Encoder capacity selection and dispatch over borrowed numerical tensors.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};

use super::modules::{Module, input_error};
use super::{ExecutionOutput, ModelExecutor, with_context};

impl ModelExecutor {
    pub(super) fn encoder_module(&self, py: Python<'_>, kind: &str) -> PyResult<Arc<Module>> {
        let mut modules = self
            .modules
            .values()
            .filter(|module| module.encoder.as_deref() == Some(kind));
        match (modules.next(), modules.next()) {
            (Some(module), None) => Ok(Arc::clone(module)),
            _ => Err(input_error(
                py,
                format!("rank does not participate in {kind} encoding"),
            )),
        }
    }

    fn text_capacity(&self, py: Python<'_>, tokens: usize) -> PyResult<usize> {
        if self.media_builder.is_none(py) {
            return Ok(tokens);
        }
        self.media_builder
            .bind(py)
            .getattr("text_capacities")?
            .extract::<Vec<usize>>()?
            .into_iter()
            .filter(|&capacity| capacity >= tokens)
            .min()
            .ok_or_else(|| PyValueError::new_err("prompt exceeds the configured text capacities"))
    }
}

pub(super) fn run(
    owner: &Bound<'_, ModelExecutor>,
    kind: &str,
    values: &Bound<'_, PyTuple>,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<ExecutionOutput>> {
    let py = owner.py();
    let module = owner.borrow().encoder_module(py, kind)?;
    let (inputs, size): (Bound<'_, PyTuple>, Bound<'_, PyAny>) = backend(py)?
        .call_method1("encoder_inputs", (kind, values))?
        .extract()?;
    ModelExecutor::run_module(
        owner,
        &module.name,
        &PyTuple::new(py, [inputs])?,
        Some(&size),
        Some("encode"),
        Some(&module.path),
        options,
    )
}

pub(super) fn tokens(owner: &Bound<'_, ModelExecutor>, tokens: &[i64]) -> PyResult<Py<PyAny>> {
    let py = owner.py();
    let module = owner.borrow().encoder_module(py, "text")?;
    let size = py
        .import("uniserve.model")?
        .call_method1("TextSize", (tokens.len(), 1))?;
    let runner = ModelExecutor::prepare_module(
        owner,
        &module.name,
        &size,
        Some("encode"),
        Some(&module.path),
    )?;
    let capacity = {
        let owner = owner.borrow();
        owner.text_capacity(py, owner.config.max_sequence_tokens)?
    };
    let options = PyDict::new(py);
    options.set_item("capacity", capacity)?;
    Ok(runner
        .bind(py)
        .call_method(
            "prepare_tokens",
            (PyTuple::new(py, tokens)?,),
            Some(&options),
        )?
        .unbind())
}

pub(super) fn text(
    owner: &Bound<'_, ModelExecutor>,
    mut token_ids: Vec<i64>,
    visual: Option<&Bound<'_, PyAny>>,
    image_grids: Option<&Bound<'_, PyTuple>>,
    video_grids: Option<&Bound<'_, PyTuple>>,
) -> PyResult<Py<ExecutionOutput>> {
    let py = owner.py();
    let count = token_ids.len();
    let capacity = owner.borrow().text_capacity(py, count)?;
    // Causal attention keeps trailing padding out of the prompt's features.
    token_ids.resize(capacity, 0);
    let inputs = tokens(owner, &token_ids)?;
    let options = PyDict::new(py);
    if let Some(visual) = visual {
        let module = owner.borrow().encoder_module(py, "text")?;
        let empty = PyTuple::empty(py);
        let positions = backend(py)?.call_method1(
            "text_positions",
            (
                module.call.bind(py).getattr("module")?,
                PyTuple::new(py, token_ids)?,
                inputs.bind(py).getattr("device")?,
                image_grids.unwrap_or(&empty),
                video_grids.unwrap_or(&empty),
            ),
        )?;
        options.set_item("positions", (positions,))?;
        options.set_item("visual", (visual,))?;
    }
    trim(
        py,
        run(owner, "text", &PyTuple::new(py, [inputs])?, Some(&options))?,
        count,
    )
}

pub(super) fn conditioning(
    owner: &Bound<'_, ModelExecutor>,
    features: &Bound<'_, PyAny>,
) -> PyResult<Py<ExecutionOutput>> {
    let py = owner.py();
    let module = owner.borrow().encoder_module(py, "conditioning")?;
    if !module
        .call
        .bind(py)
        .getattr("module")?
        .is_instance(&py.import("uniserve.model")?.getattr("TextConditioner")?)?
    {
        return run(owner, "conditioning", &PyTuple::new(py, [features])?, None);
    }
    let count = features.getattr("shape")?.get_item(0)?.extract()?;
    let capacity = owner.borrow().text_capacity(py, count)?;
    let (padded, lengths): (Bound<'_, PyAny>, Bound<'_, PyAny>) = backend(py)?
        .call_method1("conditioning_inputs", (features, capacity))?
        .extract()?;
    let options = PyDict::new(py);
    options.set_item("lengths", lengths)?;
    trim(
        py,
        run(
            owner,
            "conditioning",
            &PyTuple::new(py, [padded])?,
            Some(&options),
        )?,
        count,
    )
}

fn trim(
    py: Python<'_>,
    result: Py<ExecutionOutput>,
    count: usize,
) -> PyResult<Py<ExecutionOutput>> {
    let values = result.borrow(py).values.clone_ref(py);
    let rows = PySlice::new(py, 0, count as isize, 1);
    let values = values
        .bind(py)
        .iter()
        .map(|value| value.get_item(&rows))
        .collect::<PyResult<Vec<_>>>()?;
    let mut output = result.borrow(py).clone_ref(py);
    output.values = PyTuple::new(py, values)?.unbind();
    Py::new(py, output)
}

pub(super) fn prepare(owner: &Bound<'_, ModelExecutor>) -> PyResult<()> {
    let py = owner.py();
    let builder = owner.borrow().media_builder.bind(py).clone();
    if builder.is_none() {
        return Ok(());
    }
    let (text_module, conditioner) = {
        let owner = owner.borrow();
        let find = |kind| {
            owner
                .modules
                .values()
                .find(|module| module.encoder.as_deref() == Some(kind))
                .cloned()
        };
        (find("text"), find("conditioning"))
    };
    if text_module.is_none() && conditioner.is_none() {
        return Ok(());
    }
    let model = owner.borrow().model.bind(py).clone();
    let text_encoder = py
        .import("uniserve_worker.bootstrap.inputs")?
        .call_method1(
            "capability",
            (model, py.import("uniserve.model")?.getattr("TextEncoder")?),
        )?;
    let capacities: Vec<usize> = builder.getattr("text_capacities")?.extract()?;
    let dtype = py
        .import("torch")?
        .getattr(owner.borrow().config.model_dtype.as_str())?;
    let features = |capacity, module: &Module| {
        backend(py)?.call_method1(
            "text_features",
            (
                &text_encoder,
                capacity,
                &dtype,
                module.binding.bind(py).getattr("device")?,
            ),
        )
    };
    with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
        // All persistent contexts precede the first capture into a shared pool.
        // Largest-first warmup prevents later calls from growing its scratch.
        for &capacity in capacities.iter().rev() {
            if let Some(module) = &text_module {
                let size = py
                    .import("uniserve.model")?
                    .call_method1("TextSize", (capacity, 1))?;
                ModelExecutor::prepare_module(
                    owner,
                    &module.name,
                    &size,
                    Some("encode"),
                    Some(&module.path),
                )?;
            }
            if let Some(module) = &conditioner
                && !text_encoder.is_none()
            {
                let shape = text_encoder
                    .call_method1("output_layout", (capacity, &dtype))?
                    .get_item("conditioning")?
                    .getattr("shape")?;
                ModelExecutor::prepare_module(
                    owner,
                    &module.name,
                    PyTuple::new(py, [shape])?.as_any(),
                    Some("encode"),
                    Some(&module.path),
                )?;
            }
        }
        for &capacity in capacities.iter().rev() {
            if text_module.is_some() {
                text(owner, vec![0; capacity], None, None, None)?;
            }
            if let Some(module) = &conditioner
                && !text_encoder.is_none()
            {
                conditioning(owner, &features(capacity, module)?)?;
            }
        }
        Ok(())
    })
}

fn backend(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("uniserve_worker.model_executor.encoder_runner")
}
