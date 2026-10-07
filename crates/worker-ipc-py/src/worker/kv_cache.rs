//! Numerical views of native KV transfer state and storage lifetimes.

use std::collections::HashSet;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use uniserve_worker::KVCacheManager as NativeKVCacheManager;
use uniserve_worker_ipc::{BufferId, KvGroupTransfer, KvTransfer, RequestKey};

use super::block_tables::{BlockTables, GroupTable, pages_to_py};
use super::completion::{Completion, CompletionRef};
use super::error::{invalid, native_error};
use super::protocol::{buffer_id, call_id, request_key};
use crate::convert;

#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct KVCacheManager {
    pub(super) inner: NativeKVCacheManager<CompletionRef>,
    tables: Py<BlockTables>,
    compute_dtype: String,
    export_group: Py<PyAny>,
}

impl KVCacheManager {
    /// Select visible pages once for native execution and Python consumers.
    pub(super) fn prepare_attention(
        &mut self,
        py: Python<'_>,
        rows: Vec<(u32, i64, i64, bool)>,
    ) -> PyResult<Vec<uniserve_worker::TablePages>> {
        let rows = rows
            .into_iter()
            .map(|(slot, prefix, query, write)| {
                let length = |value| {
                    u32::try_from(value)
                        .map_err(|_| invalid(py, "forward attention lengths are invalid"))
                };
                Ok(uniserve_worker::AttentionRow {
                    slot,
                    prefix: length(prefix)?,
                    query: length(query)?,
                    write,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mut spans = Vec::new();
        let transfers = self.inner.has_transfers();
        let pages = self
            .tables
            .borrow(py)
            .tables
            .prepare_attention(&rows, |table, prefix, query| {
                if transfers {
                    spans.extend(table.spans(u64::from(prefix), u64::from(query))?);
                }
                Ok(())
            })
            .map_err(|error| native_error(py, error))?;

        if !spans.is_empty() {
            self.inner
                .require_writable(&spans)
                .map_err(|error| native_error(py, error))?;
        }
        Ok(pages)
    }

    /// Reserve the visible suffix before exposing numerical page views.
    /// Completed descriptions stay native until a direct Python caller asks
    /// for one; serving retains them through batch commit and IPC delivery.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn export(
        slf: &Bound<'_, Self>,
        slot: u32,
        visible: u64,
        destination: &str,
        buffer: BufferId,
        transports: &Bound<'_, PyAny>,
        consumers: &[u32],
    ) -> PyResult<(KvTransfer, Py<PyList>)> {
        let py = slf.py();
        let (mut transfer, tables, export_group) = {
            let mut owner = slf.borrow_mut();
            let (base, base_extent) = owner
                .inner
                .destination_base(buffer.owner, destination)
                .map_or((None, 0), |(buffer, extent)| (Some(buffer), extent));
            let (tables, groups, spans) = {
                let tables = owner.tables.borrow(py);
                let tables = &tables.tables;
                if visible > u64::from(tables.allocated_length(slot)) {
                    return Err(invalid(py, "KV export exceeds its scheduler block table"));
                }
                if visible < u64::from(base_extent) {
                    return Err(invalid(py, "KV export destination is ahead of its source"));
                }

                let mut selected = Vec::new();
                let mut groups = Vec::new();
                let mut spans = Vec::new();
                for group in 0..tables.groups().len() {
                    let table = tables
                        .table(slot, group as u32)
                        .map_err(|error| native_error(py, error))?;
                    let start = table.shape.window.map_or(base_extent, |window| {
                        base_extent.max((visible as u32).saturating_sub(window))
                    });
                    spans.extend(
                        table
                            .spans(u64::from(start), visible - u64::from(start))
                            .map_err(|error| native_error(py, error))?,
                    );
                    if visible > u64::from(base_extent) {
                        groups.push(KvGroupTransfer {
                            start,
                            page_tokens: table.shape.page_tokens,
                            tensors: Vec::new(),
                        });
                    }
                    selected.push(table);
                }
                (selected, groups, spans)
            };
            if visible > u64::from(base_extent) {
                owner
                    .inner
                    .reserve_export(buffer, &spans)
                    .map_err(|error| native_error(py, error))?;
            }
            (
                KvTransfer {
                    source: buffer,
                    destination: destination.to_owned(),
                    base,
                    base_extent,
                    exported_extent: visible as u32,
                    compute_dtype: owner.compute_dtype.clone(),
                    groups,
                },
                tables,
                owner.export_group.clone_ref(py),
            )
        };

        // A transport can expose a locator before a later group fails. Keep
        // every locator in one list so failure revokes all accepted exports.
        let locators = PyList::empty(py);
        let result = (|| -> PyResult<()> {
            let kwargs = PyDict::new(py);
            kwargs.set_item("source", convert::buffer_id_to_py(py, &buffer)?)?;
            kwargs.set_item("transports", transports)?;
            kwargs.set_item("consumers", consumers)?;
            kwargs.set_item("locators", &locators)?;
            for (index, (group, table)) in transfer.groups.iter_mut().zip(tables).enumerate() {
                let table = Py::new(py, GroupTable { table })?;
                let tensors = export_group
                    .bind(py)
                    .call((index, table, group.start, visible), Some(&kwargs))?;
                group.tensors = tensors
                    .try_iter()?
                    .map(|tensor| {
                        convert::tensor_transfer_from_py(&tensor?.call_method0("to_mapping")?)
                            .ok_or_else(|| invalid(py, "invalid exported KV tensor"))
                    })
                    .collect::<PyResult<_>>()?;
            }
            Ok(())
        })();
        if let Err(error) = result {
            for locator in &locators {
                let released = (|| -> PyResult<()> {
                    transports
                        .get_item(locator.getattr("backend")?)?
                        .call_method1("release", (locator,))?;
                    Ok(())
                })();
                if let Err(cleanup) = released {
                    let _ = error
                        .value(py)
                        .call_method1("add_note", (cleanup.to_string(),));
                }
            }
            slf.borrow_mut().inner.release_exports(&[buffer]);
            return Err(error);
        }
        Ok((transfer, locators.unbind()))
    }
}

#[pymethods]
impl KVCacheManager {
    #[new]
    fn new(tables: Py<BlockTables>, compute_dtype: String, export_group: Py<PyAny>) -> Self {
        Self {
            inner: NativeKVCacheManager::default(),
            tables,
            compute_dtype,
            export_group,
        }
    }

    #[pyo3(name = "export", signature = (*, request_pool_idx, visible_length, destination, buffer, transports, consumers=Vec::new()))]
    #[allow(clippy::too_many_arguments)]
    fn export_py(
        slf: &Bound<'_, Self>,
        request_pool_idx: u32,
        visible_length: u64,
        destination: &str,
        buffer: &Bound<'_, PyAny>,
        transports: &Bound<'_, PyAny>,
        consumers: Vec<u32>,
    ) -> PyResult<Py<PyAny>> {
        let (transfer, _) = Self::export(
            slf,
            request_pool_idx,
            visible_length,
            destination,
            buffer_id(buffer)?,
            transports,
            &consumers,
        )?;
        convert::kv_transfer_to_py(slf.py(), &transfer).map(Bound::unbind)
    }

    fn resident(&self, py: Python<'_>, buffer: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .resident(buffer_id(buffer)?)
            .map(|transfer| convert::kv_transfer_to_py(py, transfer).map(Bound::unbind))
            .transpose()
    }

    #[pyo3(signature = (request, buffer, *, request_pool_idx, visible_length))]
    fn validate_conditioning(
        &self,
        py: Python<'_>,
        request: Bound<'_, PyAny>,
        buffer: Bound<'_, PyAny>,
        request_pool_idx: u32,
        visible_length: u64,
    ) -> PyResult<Py<PyAny>> {
        let tables = self.tables.borrow(py);
        let export = self
            .inner
            .validate_conditioning(
                request_key(&request)?,
                buffer_id(&buffer)?,
                request_pool_idx,
                visible_length,
                &tables.tables,
            )
            .map_err(|error| native_error(py, error))?;
        convert::kv_transfer_to_py(py, export).map(Bound::unbind)
    }

    fn destination_base(
        &self,
        py: Python<'_>,
        request: &Bound<'_, PyAny>,
        destination: &str,
    ) -> PyResult<Option<(Py<PyAny>, u32)>> {
        self.inner
            .destination_base(request_key(request)?, destination)
            .map(|(buffer, extent)| Ok((convert::buffer_id_to_py(py, &buffer)?.unbind(), extent)))
            .transpose()
    }

    fn validate_exports(
        &self,
        py: Python<'_>,
        exports: &Bound<'_, PyAny>,
        installations: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.inner
            .validate_exports(
                &exports_from_py(exports)?,
                &installations_from_py(installations)?,
            )
            .map_err(|error| native_error(py, error))
    }

    fn apply_exports(
        &mut self,
        exports: &Bound<'_, PyAny>,
        installations: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.inner.apply_exports(
            exports_from_py(exports)?,
            installations_from_py(installations)?,
        );
        Ok(())
    }

    fn release_calls(&mut self, py: Python<'_>, calls: &Bound<'_, PyAny>) -> PyResult<Py<PyTuple>> {
        let calls = calls
            .try_iter()?
            .map(|call| {
                let (request, call): (Bound<'_, PyAny>, Bound<'_, PyAny>) = call?.extract()?;
                Ok((request_key(&request)?, call_id(&call)?))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let buffers = self
            .inner
            .release_calls(&calls)
            .iter()
            .map(|buffer| convert::buffer_id_to_py(py, buffer))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, buffers)?.unbind())
    }

    fn drop_request(&mut self, request_id: u64) {
        self.inner.drop_request(request_id);
    }

    fn clear_resident(&mut self) {
        self.inner.clear_resident();
    }

    #[getter]
    fn has_pending_accesses(&mut self) -> bool {
        self.inner.has_pending_accesses()
    }

    fn retain_execution(
        &mut self,
        py: Python<'_>,
        request: &Bound<'_, PyAny>,
        spans: Vec<(u32, u32, u32)>,
        completion: Py<Completion>,
    ) -> PyResult<()> {
        if completion.borrow(py).done() {
            completion.borrow(py).result(py, None)?;
            return Ok(());
        }
        self.inner.retain_execution(
            request_key(request)?,
            &spans,
            CompletionRef::new(py, completion),
        );
        Ok(())
    }

    fn reserve_export(
        &mut self,
        py: Python<'_>,
        buffer: &Bound<'_, PyAny>,
        spans: Vec<(u32, u32, u32)>,
    ) -> PyResult<()> {
        self.inner
            .reserve_export(buffer_id(buffer)?, &spans)
            .map_err(|error| native_error(py, error))
    }

    fn retain_export(
        &mut self,
        py: Python<'_>,
        buffer: &Bound<'_, PyAny>,
        retirement: Py<Completion>,
    ) -> PyResult<()> {
        self.inner
            .retain_export(buffer_id(buffer)?, CompletionRef::new(py, retirement))
            .map_err(|error| native_error(py, error))
    }

    fn release_exports(&mut self, buffers: &Bound<'_, PyAny>) -> PyResult<()> {
        let buffers = buffers
            .try_iter()?
            .map(|buffer| buffer_id(&buffer?))
            .collect::<PyResult<Vec<_>>>()?;
        self.inner.release_exports(&buffers);
        Ok(())
    }

    fn exported_buffers(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let buffers = self
            .inner
            .exported_buffers()
            .into_iter()
            .map(|buffer| crate::convert::buffer_id_to_py(py, &buffer))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, buffers)?.unbind())
    }

    fn reserve_import(
        &mut self,
        py: Python<'_>,
        buffer: &Bound<'_, PyAny>,
        spans: Vec<(u32, u32, u32)>,
        retirement: Py<Completion>,
    ) -> PyResult<()> {
        self.inner
            .reserve_import(
                buffer_id(buffer)?,
                &spans,
                CompletionRef::new(py, retirement),
            )
            .map_err(|error| native_error(py, error))
    }

    fn discard_import(&mut self, buffer: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.discard_import(buffer_id(buffer)?);
        Ok(())
    }

    fn write_dependencies(
        &mut self,
        py: Python<'_>,
        spans: Vec<(u32, u32, u32)>,
    ) -> PyResult<Py<PyTuple>> {
        let completions = self
            .inner
            .write_dependencies(&spans)
            .into_iter()
            .map(|completion| completion.owner.clone_ref(py));
        Ok(PyTuple::new(py, completions)?.unbind())
    }

    /// Borrow visible KV pages after capacity and transfer checks.
    #[pyo3(name = "prepare_attention")]
    fn prepare_attention_py<'py>(
        &mut self,
        py: Python<'py>,
        rows: Vec<(u32, i64, i64, bool)>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        pages_to_py(py, self.prepare_attention(py, rows)?)
    }

    fn require_reusable(&mut self, py: Python<'_>, spans: Vec<(u32, u32, u32)>) -> PyResult<()> {
        self.inner
            .require_reusable(&spans)
            .map_err(|error| native_error(py, error))
    }

    fn retirement_ready(
        &mut self,
        py: Python<'_>,
        buffers: &Bound<'_, PyAny>,
        requests: &Bound<'_, PyAny>,
        retained: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let buffers = buffer_set(buffers)?;
        let requests: HashSet<RequestKey> = requests
            .try_iter()?
            .map(|request| request_key(&request?))
            .collect::<PyResult<_>>()?;
        let retained = buffer_set(retained)?;

        self.retirement_ready_for(py, &buffers, &requests, &retained)
    }

    fn require_retired(&mut self, py: Python<'_>) -> PyResult<()> {
        self.inner
            .require_retired()
            .map_err(|error| native_error(py, error))
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.tables)?;
        visit.call(&self.export_group)?;
        for completion in self.inner.completions() {
            visit.call(&completion.owner)?;
        }
        Ok(())
    }

    fn __clear__(&mut self, py: Python<'_>) {
        self.inner = NativeKVCacheManager::default();
        self.export_group = py.None();
    }
}

pub(super) fn transfer_from_py(value: &Bound<'_, PyAny>) -> PyResult<KvTransfer> {
    convert::kv_transfer_from_py(&value.call_method0("to_mapping")?)
        .ok_or_else(|| invalid(value.py(), "invalid KV transfer"))
}

pub(super) fn exports_from_py(values: &Bound<'_, PyAny>) -> PyResult<Vec<(BufferId, KvTransfer)>> {
    values
        .try_iter()?
        .map(|value| {
            let (buffer, transfer): (Bound<'_, PyAny>, Bound<'_, PyAny>) = value?.extract()?;
            Ok((buffer_id(&buffer)?, transfer_from_py(&transfer)?))
        })
        .collect()
}

pub(super) fn installations_from_py(
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<(BufferId, BufferId, KvTransfer)>> {
    values
        .try_iter()?
        .map(|value| {
            let (source, installed, transfer): (
                Bound<'_, PyAny>,
                Bound<'_, PyAny>,
                Bound<'_, PyAny>,
            ) = value?.extract()?;
            Ok((
                buffer_id(&source)?,
                buffer_id(&installed)?,
                transfer_from_py(&transfer)?,
            ))
        })
        .collect()
}

fn buffer_set(buffers: &Bound<'_, PyAny>) -> PyResult<HashSet<BufferId>> {
    buffers
        .try_iter()?
        .map(|buffer| buffer_id(&buffer?))
        .collect()
}

impl KVCacheManager {
    pub(super) fn retirement_ready_for(
        &mut self,
        py: Python<'_>,
        buffers: &HashSet<BufferId>,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
    ) -> PyResult<bool> {
        for completion in self
            .inner
            .retirement_completions(buffers, requests, retained)
        {
            if completion.done() {
                completion.owner.borrow(py).result(py, None)?;
            }
        }
        Ok(self.inner.retirement_ready(buffers, requests, retained))
    }
}
