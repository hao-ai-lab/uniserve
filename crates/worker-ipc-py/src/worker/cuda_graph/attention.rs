//! Attention host metadata for exact capture keys and replay preparation.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::worker::model_inputs::InputBatch;

/// Keep captured tensor addresses while binding the live sequence coordinates.
pub(in crate::worker) fn bind<'py>(
    fixed: &Bound<'py, PyAny>,
    live: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = fixed.py();
    let queries = replace(
        &fixed.getattr("queries")?,
        &[("host", live.getattr("queries")?.getattr("host")?)],
    )?;
    let entries = PyDict::new(py);
    let source = fixed.getattr("entries")?;
    let current = live.getattr("entries")?;
    let paged = py.import("uniserve.nn.attention")?.getattr("PagedInput")?;

    for table in source.try_iter()? {
        let table = table?;
        let entry = source.get_item(&table)?;
        let live = current.get_item(&table)?;
        let blocks = entry.getattr("block_table")?;
        let blocks = if blocks.getattr("start_page")?.is_none() {
            blocks
        } else {
            replace(
                &blocks,
                &[(
                    "start_page_host",
                    live.getattr("block_table")?.getattr("start_page_host")?,
                )],
            )?
        };
        let prefixes = replace(
            &entry.getattr("prefixes")?,
            &[("host", live.getattr("prefixes")?.getattr("host")?)],
        )?;
        let mut fields = vec![
            ("queries", queries.clone()),
            ("prefixes", prefixes),
            ("block_table", blocks),
        ];
        if entry.is_instance(&paged)? {
            fields.push(("causal", live.getattr("causal")?));
        }
        entries.set_item(table, replace(&entry, &fields)?)?;
    }
    attention(py, &entries, &queries)
}

/// Prefix lengths and start pages are rebound before replay, so their values
/// do not select a distinct capture. Their presence and row counts still do.
pub(in crate::worker) fn exact_key<'py>(
    batch: &Bound<'py, InputBatch>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = batch.py();
    let inputs = batch.borrow().inputs.bind(py).clone();
    let mut value = batch.clone();
    if let Some(bound) = inputs
        .getattr_opt("attention")?
        .filter(|value| !value.is_none())
    {
        let types = py.import("uniserve.nn.attention")?;
        let paged = types.getattr("PagedInput")?;
        let segmented = types.getattr("SegmentedInput")?;
        let source = bound.getattr("entries")?;
        let supported = source.call_method0("values")?.try_iter()?.try_fold(
            true,
            |supported, entry| -> PyResult<bool> {
                let entry = entry?;
                Ok(supported && (entry.is_instance(&paged)? || entry.is_instance(&segmented)?))
            },
        )?;
        if supported {
            let entries = PyDict::new(py);
            for table in source.try_iter()? {
                let table = table?;
                let entry = source.get_item(&table)?;
                let blocks = entry.getattr("block_table")?;
                let host = blocks.getattr("start_page_host")?;
                let blocks = if host.is_none() {
                    blocks
                } else {
                    replace(&blocks, &[("start_page_host", mask_lengths(&host)?)])?
                };
                let prefixes = entry.getattr("prefixes")?;
                let prefixes = replace(
                    &prefixes,
                    &[("host", mask_lengths(&prefixes.getattr("host")?)?)],
                )?;
                entries.set_item(
                    table,
                    replace(&entry, &[("prefixes", prefixes), ("block_table", blocks)])?,
                )?;
            }
            let attention = attention(py, &entries, &bound.getattr("queries")?)?;
            let mut masked = batch.borrow().clone_ref(py);
            masked.inputs = replace(&inputs, &[("attention", attention)])?.unbind();
            value = Bound::new(py, masked)?;
        }
    }
    let signature = super::inputs::input_signature(value.as_any())?;
    Ok(("exact", signature).into_pyobject(py)?.into_any())
}

fn attention<'py>(
    py: Python<'py>,
    entries: &Bound<'py, PyDict>,
    queries: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("uniserve.nn.attention")?
        .getattr("AttentionBatch")?
        .call1((entries, queries))
}

fn mask_lengths<'py>(host: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if host.is_none() {
        Ok(host.clone())
    } else {
        Ok(PyTuple::new(host.py(), (0..host.len()?).map(|_| 0))?.into_any())
    }
}

fn replace<'py>(
    value: &Bound<'py, PyAny>,
    fields: &[(&str, Bound<'py, PyAny>)],
) -> PyResult<Bound<'py, PyAny>> {
    let options = PyDict::new(value.py());
    for (name, field) in fields {
        options.set_item(name, field)?;
    }
    value
        .py()
        .import("dataclasses")?
        .call_method("replace", (value,), Some(&options))
}
