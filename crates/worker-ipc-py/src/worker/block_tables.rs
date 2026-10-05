//! Native KV page tables with borrowed Python tensor-copy callbacks.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use uniserve_core::UnitId;
use uniserve_worker::{
    BlockTables as NativeBlockTables, GroupShape as NativeGroupShape,
    GroupTable as NativeGroupTable,
};
use uniserve_worker_ipc::{BlockTable, RequestKey};

use super::error::{invalid, native_error};
use super::protocol::request_key;

#[pyclass(
    frozen,
    eq,
    hash,
    skip_from_py_object,
    module = "uniserve_worker._uniserve_ipc"
)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct GroupShape {
    shape: NativeGroupShape,
}

#[pymethods]
impl GroupShape {
    #[new]
    #[pyo3(signature = (page_tokens, units_per_page, window=None))]
    fn new(
        py: Python<'_>,
        page_tokens: u32,
        units_per_page: u32,
        window: Option<u32>,
    ) -> PyResult<Self> {
        Ok(Self {
            shape: NativeGroupShape::new(page_tokens, units_per_page, window)
                .map_err(|error| native_error(py, error))?,
        })
    }

    #[getter]
    fn page_tokens(&self) -> u32 {
        self.shape.page_tokens
    }

    #[getter]
    fn units_per_page(&self) -> u32 {
        self.shape.units_per_page
    }

    #[getter]
    fn window(&self) -> Option<u32> {
        self.shape.window
    }
}

#[pyclass(
    frozen,
    eq,
    hash,
    skip_from_py_object,
    module = "uniserve_worker._uniserve_ipc"
)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct GroupTable {
    pub(super) table: Arc<NativeGroupTable>,
}

#[pymethods]
impl GroupTable {
    #[new]
    fn new(shape: &GroupShape, start_page: u32, units: Vec<u32>, allocated_tokens: u32) -> Self {
        Self {
            table: Arc::new(NativeGroupTable {
                shape: shape.shape,
                start_page,
                units: units.into_iter().map(UnitId).collect(),
                allocated_tokens,
            }),
        }
    }

    #[getter]
    fn shape(&self) -> GroupShape {
        GroupShape {
            shape: self.table.shape,
        }
    }

    #[getter]
    fn start_page(&self) -> u32 {
        self.table.start_page
    }

    #[getter]
    fn end_page(&self) -> u64 {
        self.table.end_page()
    }

    #[getter]
    fn allocated_tokens(&self) -> u32 {
        self.table.allocated_tokens
    }

    #[getter]
    fn units<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.table.units.iter().map(|unit| unit.0))
    }

    fn row<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.table.row(index))
    }

    fn spans<'py>(
        &self,
        py: Python<'py>,
        start: i64,
        length: i64,
    ) -> PyResult<Bound<'py, PyTuple>> {
        if start < 0 || length < 0 {
            return Err(invalid(py, "KV token interval exceeds its unit table"));
        }

        let spans = self
            .table
            .spans(start as u64, length as u64)
            .map_err(|error| native_error(py, error))?;
        PyTuple::new(py, spans)
    }
}

#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct BlockTables {
    pub(super) tables: NativeBlockTables,
}

#[pymethods]
impl BlockTables {
    #[new]
    fn new(
        py: Python<'_>,
        groups: Vec<PyRef<'_, GroupShape>>,
        request_pool_size: i64,
        width: i64,
        num_units: i64,
    ) -> PyResult<Self> {
        let invalid_dimensions = |_| invalid(py, "request-to-token pool dimensions are invalid");
        let request_pool_size = u32::try_from(request_pool_size).map_err(invalid_dimensions)?;
        let width = usize::try_from(width).map_err(invalid_dimensions)?;
        let num_units = u32::try_from(num_units).map_err(invalid_dimensions)?;
        let groups = groups.iter().map(|group| group.shape).collect();
        Ok(Self {
            tables: NativeBlockTables::new(groups, request_pool_size, width, num_units)
                .map_err(|error| native_error(py, error))?,
        })
    }

    #[getter]
    fn first_table<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.tables.first_table().iter().copied())
    }

    /// The copy callback borrows one update; it is never retained by Rust.
    #[pyo3(name = "install")]
    fn install_py(
        &mut self,
        py: Python<'_>,
        tables: &Bound<'_, PyAny>,
        copy: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut values = Vec::new();
        for table in tables.try_iter()? {
            let (slot, group, start, units, allocated): (i64, i64, i64, Vec<i64>, i64) =
                table?.extract()?;
            let unsigned = |value| {
                u32::try_from(value).map_err(|_| invalid(py, "scheduler block table is invalid"))
            };
            values.push(BlockTable {
                request_pool_idx: unsigned(slot)?,
                group_id: unsigned(group)?,
                start_page: unsigned(start)?,
                unit_ids: units
                    .into_iter()
                    .map(|unit| unsigned(unit).map(UnitId))
                    .collect::<PyResult<_>>()?,
                allocated_tokens: unsigned(allocated)?,
            });
        }

        self.install(py, &values, copy)
    }

    fn table(&self, py: Python<'_>, request_pool_idx: i64, group_id: i64) -> PyResult<GroupTable> {
        let missing = |_| invalid(py, "request slot has no installed block table");
        let slot = u32::try_from(request_pool_idx).map_err(missing)?;
        let group = u32::try_from(group_id).map_err(missing)?;
        Ok(GroupTable {
            table: self
                .tables
                .table(slot, group)
                .map_err(|error| native_error(py, error))?,
        })
    }

    fn allocated_length(&self, request_pool_idx: i64) -> u32 {
        u32::try_from(request_pool_idx).map_or(0, |slot| self.tables.allocated_length(slot))
    }

    fn retain_prefix(&mut self, request: &Bound<'_, PyAny>, slot: u32) -> PyResult<()> {
        self.tables.retain_prefix(request_key(request)?, slot);
        Ok(())
    }

    #[pyo3(signature = (request, copy, slots=None))]
    fn release_prefixes(
        &mut self,
        py: Python<'_>,
        request: &Bound<'_, PyAny>,
        copy: &Bound<'_, PyAny>,
        slots: Option<Vec<i64>>,
    ) -> PyResult<()> {
        let key = request_key(request)?;
        let slots = match slots {
            Some(slots) => self.slots(py, slots)?,
            None => self
                .tables
                .release_slots(&self.tables.prefix_slots(key))
                .map_err(|error| native_error(py, error))?,
        };
        self.clear_prefixes(key, &slots, copy)
    }

    fn release(
        &mut self,
        py: Python<'_>,
        slots: Vec<i64>,
        copy: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let slots = self.slots(py, slots)?;
        if !slots.is_empty() {
            copy.call1((&slots,))?;
        }
        self.tables.release(&slots);
        Ok(())
    }

    fn clear(&mut self) {
        self.tables.clear();
    }
}

impl BlockTables {
    /// Clear bound prefix slots after their final numerical use. The caller
    /// supplies validated slots; device clears precede host-table removal.
    pub(super) fn clear_prefixes(
        &mut self,
        key: RequestKey,
        slots: &[u32],
        copy: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if !slots.is_empty() {
            copy.call1((slots,))?;
        }
        self.tables.release_prefixes(key, slots);
        Ok(())
    }

    pub(super) fn install(
        &mut self,
        py: Python<'_>,
        values: &[BlockTable],
        copy: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let update = self
            .tables
            .prepare(values)
            .map_err(|error| native_error(py, error))?;
        if !update.rows.is_empty()
            || !update.start_values.is_empty()
            || !update.length_slots.is_empty()
        {
            copy.call1((
                &update.rows,
                &update.row_tables,
                &update.row_slots,
                &update.start_groups,
                &update.start_slots,
                &update.start_values,
                &update.length_slots,
                &update.length_values,
            ))?;
        }
        self.tables.commit(update);
        Ok(())
    }

    fn slots(&self, py: Python<'_>, slots: Vec<i64>) -> PyResult<Vec<u32>> {
        let slots = slots
            .into_iter()
            .map(|slot| {
                u32::try_from(slot)
                    .map_err(|_| invalid(py, "released request slot is outside capacity"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        self.tables
            .release_slots(&slots)
            .map_err(|error| native_error(py, error))
    }
}
