//! Execute direct and converted KV imports using native transfer descriptions.

use std::ops::Range;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::{Copy, TransferRef, TransferTicket, Workspace, drain, native_error};
use crate::worker::fetch;
use crate::worker::locator::Locator;

impl Copy {
    pub(super) fn copy_groups(&self, py: Python<'_>, workspace: &Workspace) -> PyResult<()> {
        let owner = self.owner.borrow(py);
        let pool = owner.pool.bind(py);
        let info = pool.getattr("info")?;
        let dtype: String = info.getattr("dtype")?.extract()?;
        let write = self.write.get();

        for (index, group) in write.export.groups.iter().enumerate() {
            if group.tensors.is_empty() {
                continue;
            }

            let advertised = info.getattr("groups")?.get_item(index)?;
            let layer: u64 = pool
                .getattr("axes")?
                .get_item(index)?
                .getattr("offset")?
                .extract()?;
            let layers = advertised.getattr("layer_ids")?.len()? as u64;
            let head: u64 = advertised.getattr("kv_head_offset")?.extract()?;
            let heads: u64 = advertised.getattr("num_kv_heads")?.extract()?;
            let width: u64 = advertised.getattr("head_dim")?.extract()?;
            let axes = [layer..layer + layers, head..head + heads, 0..width];
            let source_dtype = &group.tensors[0].locations[0].dtype;
            let mut direct = source_dtype == &dtype;
            if direct && dtype == "float8_e4m3fn" {
                let page_tokens = write.tables[index].shape.page_tokens;
                let scale_heads = group.tensors[0].shape[2] / group.tensors[2].shape[3];
                let compute_dtype = pool.getattr("compute_dtypes")?.get_item(index)?.str()?;
                // Partial pages retain their destination scale; an intact
                // source scale can only serve one destination head group.
                direct = group.page_tokens == page_tokens
                    && group.start.is_multiple_of(page_tokens)
                    && write.export.compute_dtype
                        == compute_dtype.to_str()?.trim_start_matches("torch.")
                    && head / scale_heads == (head + heads - 1) / scale_heads;
            }

            let mut locators: Vec<_> = group
                .tensors
                .iter()
                .map(|tensor| vec![None; tensor.locations.len()])
                .collect();
            if direct {
                self.copy_direct(py, workspace, index, axes, &mut locators)?;
            } else {
                self.copy_converted(py, workspace, index, axes, &mut locators)?;
            }
        }
        Ok(())
    }

    fn copy_direct<'py>(
        &self,
        py: Python<'py>,
        workspace: &Workspace,
        index: usize,
        axes: [Range<u64>; 3],
        locators: &mut [Vec<Option<Bound<'py, Locator>>>],
    ) -> PyResult<()> {
        let owner = self.owner.borrow(py);
        let pool = owner.pool.bind(py);
        let write = self.write.get();
        let group = &write.export.groups[index];
        let table = &write.tables[index];
        let carried = u64::from(write.export.exported_extent - group.start);
        let spans = table
            .spans(u64::from(group.start), carried)
            .map_err(|error| native_error(py, error))?;
        let columns: usize = pool
            .getattr("cache")?
            .getattr("planes")?
            .getattr("columns")?
            .extract()?;
        let views = py
            .import("uniserve_worker.storage.cache_imports")?
            .getattr("_direct_views")?;
        let rows = table.shape.units_per_page as usize;

        // Bound physical fan-out by one layer, regardless of model depth.
        for (column, layer) in axes[0].clone().enumerate() {
            let selected = PyTuple::new(
                py,
                spans.iter().skip(column / columns).step_by(rows).copied(),
            )?;
            let fields: Vec<(Bound<'_, PyAny>, Option<Bound<'_, PyAny>>)> =
                views.call1((pool, index, column, selected))?.extract()?;
            let mut tickets = Vec::new();
            for (field, (values, scales)) in fields.into_iter().enumerate() {
                let region = [
                    0..carried,
                    layer..layer + 1,
                    axes[1].clone(),
                    axes[2].clone(),
                ];
                tickets.extend(self.fetch(
                    py,
                    index,
                    field,
                    &values,
                    &region,
                    &mut locators[field],
                )?);
                if let Some(scales) = scales {
                    let scale_heads = group.tensors[0].shape[2] / group.tensors[2].shape[3];
                    let head = axes[1].start / scale_heads;
                    let pages = (spans.len() / rows) as u64;
                    let field = field as u64;
                    let region = [0..pages, field..field + 1, layer..layer + 1, head..head + 1];
                    tickets.extend(self.fetch(py, index, 2, &scales, &region, &mut locators[2])?);
                }
            }
            self.consume(py, workspace, &tickets)?;
            Self::close_reads(py, tickets)?;
        }

        owner
            .inner
            .require_active(&write.inner)
            .map_err(|error| native_error(py, error))?;
        pool.getattr("cache")?.call_method1(
            "mark_initialized",
            (PyTuple::new(py, spans.iter().map(|span| span.0))?,),
        )?;
        Ok(())
    }

    fn copy_converted<'py>(
        &self,
        py: Python<'py>,
        workspace: &Workspace,
        index: usize,
        axes: [Range<u64>; 3],
        locators: &mut [Vec<Option<Bound<'py, Locator>>>],
    ) -> PyResult<()> {
        let owner = self.owner.borrow(py);
        let pool = owner.pool.bind(py);
        let write = self.write.get();
        let group = &write.export.groups[index];
        let table = &write.tables[index];
        let numerical = py.import("uniserve_worker.storage.cache_imports")?;
        let dtype = &group.tensors[0].locations[0].dtype;
        let quantized = dtype == "float8_e4m3fn";
        let scale_heads = if quantized {
            group.tensors[0].shape[2] / group.tensors[2].shape[3]
        } else {
            0
        };
        let compute_dtype = py
            .import("torch")?
            .getattr(write.export.compute_dtype.as_str())?;
        let page_tokens = u64::from(table.shape.page_tokens);
        let start = u64::from(group.start);
        let end = u64::from(write.export.exported_extent);
        let mut position = start;

        while position < end {
            let page = position / page_tokens;
            let offset = position % page_tokens;
            let count = (end - position).min(page_tokens - offset);
            let shape = [
                count,
                axes[0].end - axes[0].start,
                axes[1].end - axes[1].start,
                axes[2].end - axes[2].start,
            ];
            let source_offset = position % u64::from(group.page_tokens);
            let scale_region = if quantized {
                let first =
                    position / u64::from(group.page_tokens) - start / u64::from(group.page_tokens);
                let pages = (source_offset + count).div_ceil(u64::from(group.page_tokens));
                Some([
                    first..first + pages,
                    0..2,
                    axes[0].clone(),
                    axes[1].start / scale_heads..axes[1].end.div_ceil(scale_heads),
                ])
            } else {
                None
            };
            let scale_shape: Vec<_> = scale_region
                .iter()
                .flatten()
                .map(|axis| axis.end - axis.start)
                .collect();
            let (raw, scales): (Bound<'_, PyTuple>, Option<Bound<'_, PyAny>>) = numerical
                .call_method1(
                    "_conversion_views",
                    (
                        workspace.values.bind(py),
                        dtype,
                        PyTuple::new(py, shape)?,
                        PyTuple::new(py, scale_shape)?,
                    ),
                )?
                .extract()?;
            let region = [
                position - start..position - start + count,
                axes[0].clone(),
                axes[1].clone(),
                axes[2].clone(),
            ];
            let mut tickets = Vec::new();
            for (field, target) in raw.iter().enumerate() {
                tickets.extend(self.fetch(
                    py,
                    index,
                    field,
                    &target,
                    &region,
                    &mut locators[field],
                )?);
            }
            if let (Some(scales), Some(region)) = (&scales, scale_region) {
                tickets.extend(self.fetch(py, index, 2, scales, &region, &mut locators[2])?);
            }
            self.consume(py, workspace, &tickets)?;

            let rows = table.shape.units_per_page as usize;
            let first = (page - u64::from(table.start_page)) as usize * rows;
            let units = PyTuple::new(
                py,
                table.units[first..first + rows].iter().map(|unit| unit.0),
            )?;
            numerical.call_method1(
                "_copy_converted",
                (
                    pool,
                    index,
                    workspace.values.bind(py),
                    raw,
                    scales,
                    units,
                    offset,
                    source_offset,
                    group.page_tokens,
                    scale_heads,
                    &compute_dtype,
                ),
            )?;

            // The next span reuses this conversion page. Retire numerical
            // consumption before allowing another transport to overwrite it.
            drain(py, workspace)?;
            Self::close_reads(py, tickets)?;
            position += count;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn fetch<'py>(
        &self,
        py: Python<'py>,
        group: usize,
        field: usize,
        destination: &Bound<'py, PyAny>,
        region: &[Range<u64>],
        locators: &mut [Option<Bound<'py, Locator>>],
    ) -> PyResult<Vec<Py<TransferTicket>>> {
        let owner = self.owner.borrow(py);
        let write = self.write.get();
        owner
            .inner
            .require_active(&write.inner)
            .map_err(|error| native_error(py, error))?;
        let reads = fetch::plan_native_reads(
            py,
            &write.export.groups[group].tensors[field],
            destination,
            self.transports.bind(py),
            region,
            locators,
        )?;
        fetch::submit_reads(py, &reads, |ticket| {
            let read = Arc::new(TransferRef::new(py, ticket.clone_ref(py))?);
            let cancelled = write.inner.retain(read);
            ticket
                .get()
                .add_retirement_callback(py, self.owner.bind(py).getattr("_reap")?.unbind())?;
            if cancelled {
                ticket.get().cancel(py)?;
            }
            Ok(())
        })
    }

    fn consume(
        &self,
        py: Python<'_>,
        workspace: &Workspace,
        tickets: &[Py<TransferTicket>],
    ) -> PyResult<()> {
        let stream = workspace.torch_stream.bind(py);
        for ticket in tickets {
            ticket.get().wait_ready(py)?;
            ticket
                .get()
                .result(py, (!stream.is_none()).then(|| stream.clone()))?;
        }
        Ok(())
    }

    fn close_reads(py: Python<'_>, tickets: Vec<Py<TransferTicket>>) -> PyResult<()> {
        for ticket in tickets {
            TransferTicket::close(ticket.into_bound(py))?;
        }
        Ok(())
    }
}
