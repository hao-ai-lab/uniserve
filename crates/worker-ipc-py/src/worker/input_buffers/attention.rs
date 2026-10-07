//! Native row columns feed the existing device page-table gather.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::{InputBuffers, ROW_SECTIONS, copy, numerical, prefix, with_host};
use crate::worker::block_tables::pages_to_py;
use crate::worker::error::invalid;
use crate::worker::kv_cache::KVCacheManager;

impl InputBuffers {
    pub(super) fn pages<'py>(
        rows: &Bound<'py, PyTuple>,
        cache: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Vec<uniserve_worker::TablePages>> {
        let py = rows.py();
        let cache =
            cache.ok_or_else(|| invalid(py, "paged attention requires resident KV storage"))?;
        let rows = rows
            .iter()
            .map(|row| {
                Ok((
                    row.getattr("request_pool_idx")?.extract()?,
                    row.getattr("seq_len")?.extract()?,
                    row.getattr("query_tokens")?.extract()?,
                    row.getattr("write_kv")?.extract()?,
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;
        cache
            .getattr("_manager")?
            .cast::<KVCacheManager>()?
            .borrow_mut()
            .prepare_attention(py, rows)
    }

    pub(super) fn attention(
        slf: &Bound<'_, Self>,
        rows: &Bound<'_, PyTuple>,
        attention: Option<&Bound<'_, PyAny>>,
        cache: Option<&Bound<'_, PyAny>>,
        tables: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if let Some(attention) = attention {
            return Ok(numerical(py)?
                .call_method1("copy_attention", (slf, attention))?
                .unbind());
        }
        let tables =
            tables.ok_or_else(|| invalid(py, "paged attention requires request tables"))?;
        let pages = Self::pages(rows, cache)?;
        let mut queries = Vec::with_capacity(rows.len());
        let mut prefixes = Vec::with_capacity(rows.len());
        let mut writes = Vec::with_capacity(rows.len());
        let mut causal = Vec::with_capacity(rows.len());
        for row in rows {
            queries.push(row.getattr("query_tokens")?.extract::<usize>()?);
            prefixes.push(row.getattr("seq_len")?.extract::<i64>()?);
            writes.push(row.getattr("write_kv")?.is_truthy()?);
            causal.push(row.getattr("causal")?.is_truthy()?);
        }
        if !writes.iter().any(|value| *value) && causal.iter().any(|value| *value) {
            return Err(PyValueError::new_err(
                "read-only prefix/current calls require noncausal current sequences",
            ));
        }
        let total = queries.iter().sum::<usize>();
        let (ring, destination, stride, widths) = {
            let owner = slf.borrow();
            (
                owner
                    .rows
                    .as_ref()
                    .ok_or_else(|| PyValueError::new_err("attention requires row storage"))?
                    .clone_ref(py),
                owner.column(py, "row_columns")?,
                owner.max_rows + 1,
                owner.table_widths.clone(),
            )
        };
        if pages.len() != widths.len() {
            return Err(PyValueError::new_err(
                "attention tables exceed input-buffer capacity",
            ));
        }

        // The gather reads six scalar/offset sections, page counts for each
        // table, and one row index per token. Offset sections include row N.
        let section = (ROW_SECTIONS + pages.len()) * stride;
        let count = section + total;
        let (slot, host) = ring.borrow(py).acquire(py)?;
        with_host::<i64>(host.bind(py), count, |columns| {
            let mut query_offset = 0;
            let mut prefix_offset = 0;
            columns[4 * stride].set(0);
            columns[5 * stride].set(0);
            for (index, row) in rows.iter().enumerate() {
                columns[index].set(row.getattr("request_pool_idx")?.extract()?);
                columns[stride + index].set(prefixes[index]);
                columns[2 * stride + index].set(queries[index] as i64);
                columns[3 * stride + index].set(i64::from(writes[index]));
                for token in
                    &columns[section + query_offset..section + query_offset + queries[index]]
                {
                    token.set(index as i64);
                }
                query_offset += queries[index];
                prefix_offset += prefixes[index];
                columns[4 * stride + index + 1].set(query_offset as i64);
                columns[5 * stride + index + 1].set(prefix_offset);
            }
            for (number, table) in pages.iter().enumerate() {
                if table.width() > widths[number] {
                    return Err(PyValueError::new_err(
                        "block tables exceed input-buffer capacity",
                    ));
                }
                let start = (ROW_SECTIONS + number) * stride;
                for (cell, length) in columns[start..start + rows.len()]
                    .iter()
                    .zip(&table.lengths)
                {
                    cell.set(*length as i64);
                }
            }
            Ok(())
        })?;
        copy(
            &prefix(&destination, count)?,
            &prefix(host.bind(py), count)?,
        )?;
        ring.borrow(py).record_copy(py, slot)?;

        Ok(numerical(py)?
            .call_method1(
                "_gather_attention",
                (
                    slf,
                    pages_to_py(py, pages)?,
                    PyTuple::new(py, queries)?,
                    PyTuple::new(py, prefixes)?,
                    PyTuple::new(py, writes)?,
                    PyTuple::new(py, causal)?,
                    tables,
                ),
            )?
            .unbind())
    }
}
