//! Per-row output selection over one packed language-model invocation.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};
use uniserve_worker::TokenSelection;

use super::TextRunner;
use crate::worker::model_results::ExecutionOutput;

pub(super) fn forward(
    runner: &Bound<'_, TextRunner>,
    inputs: &Bound<'_, PyAny>,
    selections: &[TokenSelection],
) -> PyResult<Py<ExecutionOutput>> {
    let py = runner.py();
    let model = runner.borrow().as_super().model.clone_ref(py);
    if selections
        .iter()
        .all(|selection| *selection == TokenSelection::Cache)
    {
        model.call_method1(py, "fill_cache", (inputs,))?;
        return cache_rows(runner, selections.len());
    }
    let hidden = model.call1(py, (inputs,))?;
    select(runner, hidden.bind(py), inputs, selections, false)
}

pub(super) fn cache_rows(
    runner: &Bound<'_, TextRunner>,
    rows: usize,
) -> PyResult<Py<ExecutionOutput>> {
    let py = runner.py();
    let (model, device) = {
        let owner = runner.borrow();
        (
            owner.as_super().model.clone_ref(py),
            owner.as_super().device.clone_ref(py),
        )
    };
    let options = pyo3::types::PyDict::new(py);
    options.set_item("device", device)?;
    let empty = py.import("torch")?.call_method(
        "empty",
        ((
            0,
            model.bind(py).getattr("backbone")?.getattr("hidden_size")?,
        ),),
        Some(&options),
    )?;
    output(
        py,
        vec![empty; rows],
        (0..rows).map(|_| py.None()).collect(),
    )
}

pub(super) fn select(
    runner: &Bound<'_, TextRunner>,
    hidden: &Bound<'_, PyAny>,
    inputs: &Bound<'_, PyAny>,
    selections: &[TokenSelection],
    copy_hidden: bool,
) -> PyResult<Py<ExecutionOutput>> {
    let py = runner.py();
    let queries = inputs.getattr("attention")?.getattr("queries")?;
    let lengths: Vec<usize> = if queries.is_none() {
        let rows: usize = inputs.getattr("batch_size")?.extract()?;
        let tokens: usize = inputs
            .getattr("input_ids")?
            .call_method0("numel")?
            .extract()?;
        vec![tokens / rows; rows]
    } else {
        let host = queries.getattr("host")?;
        if host.is_none() {
            return Err(PyValueError::new_err(
                "text output selection requires host query lengths",
            ));
        }
        host.extract()?
    };
    if lengths.len() != selections.len() {
        return Err(PyValueError::new_err(
            "text output selections must align with the input sequences",
        ));
    }

    // A single vocabulary projection gathers all requested token positions.
    let logit_counts: Vec<_> = lengths
        .iter()
        .zip(selections)
        .filter_map(|(&tokens, &selection)| {
            selection.projects().then_some(selection.logit_rows(tokens))
        })
        .collect();
    let logits_count: usize = logit_counts.iter().sum();
    let vocab = runner.borrow().vocab.clone_ref(py);
    let logits = if logits_count == 0 {
        let local = vocab.bind(py).getattr("local_slice")?;
        let width = local.getattr("stop")?.extract::<usize>()?
            - local.getattr("start")?.extract::<usize>()?;
        hidden.call_method1("new_empty", ((0, width),))?
    } else {
        let all_last = lengths
            .iter()
            .zip(selections)
            .all(|(&tokens, &selection)| tokens > 0 && selection == TokenSelection::LastLogits);
        let indices = if !queries.is_none() && all_last {
            queries
                .getattr("offsets")?
                .get_item(PySlice::new(py, 1, (lengths.len() + 1) as isize, 1))?
                .call_method1("__sub__", (1,))?
        } else {
            let mut positions = Vec::with_capacity(logits_count);
            let mut start = 0;
            for (&length, &selection) in lengths.iter().zip(selections) {
                let stop = start + length;
                positions.extend(stop - selection.logit_rows(length)..stop);
                start = stop;
            }
            indices(runner, hidden, positions)?
        };
        runner.call_method1("_project_logits", (hidden, indices, logits_count))?
    };

    let hidden_lengths: Vec<usize> = lengths
        .iter()
        .zip(selections)
        .filter_map(|(&length, &selection)| (selection == TokenSelection::Hidden).then_some(length))
        .collect();
    let hidden_count: usize = hidden_lengths.iter().sum();
    let hidden_rows = if !hidden_lengths.is_empty() {
        let indices = if selections
            .iter()
            .all(|selection| *selection == TokenSelection::Hidden)
        {
            py.None().into_bound(py)
        } else {
            let mut positions = Vec::with_capacity(hidden_count);
            let mut start = 0;
            for (&length, &selection) in lengths.iter().zip(selections) {
                if selection == TokenSelection::Hidden {
                    positions.extend(start..start + length);
                }
                start += length;
            }
            indices(runner, hidden, positions)?
        };
        runner
            .call_method1(
                "_select_hidden",
                (hidden, indices, hidden_count, copy_hidden),
            )?
            .call_method1("split", (PyTuple::new(py, hidden_lengths)?,))?
            .cast_into::<PyTuple>()?
    } else {
        PyTuple::empty(py)
    };

    let width = hidden.getattr("shape")?.get_item(1)?;
    let empty_hidden = hidden.call_method1("new_empty", ((0, width),))?;
    let logit_rows = logits
        .call_method1("split", (PyTuple::new(py, logit_counts)?,))?
        .cast_into::<PyTuple>()?;
    let mut values = Vec::with_capacity(selections.len());
    let mut vocabularies = Vec::with_capacity(selections.len());
    let mut logits_index = 0;
    let mut hidden_index = 0;
    for &selection in selections {
        if selection.projects() {
            values.push(logit_rows.get_item(logits_index)?);
            vocabularies.push(vocab.clone_ref(py));
            logits_index += 1;
        } else if selection == TokenSelection::Hidden {
            values.push(hidden_rows.get_item(hidden_index)?);
            vocabularies.push(py.None());
            hidden_index += 1;
        } else {
            values.push(empty_hidden.clone());
            vocabularies.push(py.None());
        }
    }
    output(py, values, vocabularies)
}

fn indices<'py>(
    runner: &Bound<'py, TextRunner>,
    hidden: &Bound<'py, PyAny>,
    values: Vec<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    super::backend(runner.py())?.call_method1("_indices", (values, hidden.getattr("device")?))
}

fn output(
    py: Python<'_>,
    values: Vec<Bound<'_, PyAny>>,
    vocabularies: Vec<Py<PyAny>>,
) -> PyResult<Py<ExecutionOutput>> {
    Py::new(
        py,
        ExecutionOutput::new(
            py,
            PyTuple::new(py, values)?.unbind(),
            Some(PyTuple::new(py, vocabularies)?.unbind()),
            None,
            None,
            None,
            None,
            None,
        )?,
    )
}
