//! Visible KV pages for numerical attention inputs.

use std::ops::Range;
use std::sync::Arc;

use super::{BlockTables, GroupShape, GroupTable};
use crate::{Error, Result};

/// Host lengths and placement of one numerical attention row.
pub struct AttentionRow {
    pub slot: u32,
    pub prefix: u32,
    pub query: u32,
    pub write: bool,
}

/// Borrowed pages for one numerical block table, in batch row order.
/// Device gathers need only the lengths and first pages. Physical units are
/// expanded only for callers that build an attention tensor on the host.
pub struct TablePages {
    pub shape: GroupShape,
    pub starts: Vec<u32>,
    pub lengths: Vec<usize>,
    position: usize,
    tables: Vec<Arc<GroupTable>>,
}

impl TablePages {
    pub fn width(&self) -> usize {
        self.lengths.iter().copied().max().unwrap_or(0).max(1)
    }

    pub fn rows(&self) -> impl Iterator<Item = impl Iterator<Item = u32> + '_> + '_ {
        self.tables.iter().enumerate().map(|(row, table)| {
            let first = (self.starts[row] - table.start_page) as usize;
            let step = self.shape.units_per_page as usize;
            table
                .units
                .iter()
                .skip(first * step + self.position)
                .step_by(step)
                .take(self.lengths[row])
                .map(|unit| unit.0)
        })
    }
}

impl GroupTable {
    /// The first query at `prefix` reaches furthest back into the window.
    fn first_page(&self, prefix: u32) -> u32 {
        self.shape.window.map_or(0, |window| {
            prefix.saturating_sub(window) / self.shape.page_tokens
        })
    }

    fn visible_pages(&self, prefix: u32, query: u32) -> Result<Range<u64>> {
        let first = u64::from(self.first_page(prefix));
        if first < u64::from(self.start_page) {
            return Err(Error::Invalid(
                "attention row reads retired window pages".into(),
            ));
        }

        let end = if self.shape.window.is_some() {
            self.end_page().min(
                (u64::from(prefix) + u64::from(query)).div_ceil(u64::from(self.shape.page_tokens)),
            )
        } else {
            self.end_page()
        };
        Ok(first..end.max(first))
    }
}

/// Select the same page ranges for resident execution and startup scratch KV.
pub fn table_pages(
    tables: &[Vec<Arc<GroupTable>>],
    prefixes: &[u32],
    queries: &[u32],
) -> Result<Vec<TablePages>> {
    let Some(first) = tables.first() else {
        return Err(Error::Invalid(
            "attention tables require at least one row".into(),
        ));
    };
    if tables.len() != prefixes.len()
        || tables.len() != queries.len()
        || tables.iter().any(|row| row.len() != first.len())
    {
        return Err(Error::Invalid(
            "attention tables and lengths must align by row".into(),
        ));
    }

    let mut selected = Vec::new();
    for (group, table) in first.iter().enumerate() {
        let mut starts = Vec::with_capacity(tables.len());
        let mut lengths = Vec::with_capacity(tables.len());
        let mut rows = Vec::with_capacity(tables.len());
        for ((row, &prefix), &query) in tables.iter().zip(prefixes).zip(queries) {
            let row_table = &row[group];
            if row_table.shape != table.shape {
                return Err(Error::Invalid(
                    "attention rows must share cache group dimensions".into(),
                ));
            }
            let pages = row_table.visible_pages(prefix, query)?;
            starts.push(pages.start as u32);
            lengths.push((pages.end - pages.start) as usize);
            rows.push(Arc::clone(row_table));
        }

        for position in 0..table.shape.units_per_page as usize {
            selected.push(TablePages {
                shape: table.shape,
                starts: starts.clone(),
                lengths: lengths.clone(),
                position,
                tables: rows.clone(),
            });
        }
    }
    Ok(selected)
}

impl BlockTables {
    /// Resolve an attention batch while checking scheduler-assigned capacity.
    /// The caller supplies the cache's write check, so physical intervals never
    /// need to pass through the numerical backend.
    pub fn prepare_attention(
        &self,
        rows: &[AttentionRow],
        mut require_writable: impl FnMut(&GroupTable, u32, u32) -> Result<()>,
    ) -> Result<Vec<TablePages>> {
        if rows.is_empty() {
            return Err(Error::Invalid(
                "attention metadata requires forward rows".into(),
            ));
        }

        let mut tables = Vec::with_capacity(rows.len());
        let mut prefixes = Vec::with_capacity(rows.len());
        let mut queries = Vec::with_capacity(rows.len());
        for row in rows {
            if row.query == 0 {
                return Err(Error::Invalid(
                    "forward attention lengths are invalid".into(),
                ));
            }
            let resulting =
                u64::from(row.prefix) + u64::from(if row.write { row.query } else { 0 });
            if resulting > u64::from(self.allocated_length(row.slot)) {
                return Err(Error::Invalid(
                    "forward row exceeds its scheduler block table".into(),
                ));
            }

            let mut groups = Vec::with_capacity(self.groups.len());
            for group in 0..self.groups.len() {
                let table = self.table(row.slot, group as u32)?;
                if row.write {
                    require_writable(&table, row.prefix, row.query)?;
                }
                groups.push(table);
            }
            tables.push(groups);
            prefixes.push(row.prefix);
            queries.push(row.query);
        }
        table_pages(&tables, &prefixes, &queries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_core::UnitId;

    #[test]
    fn window_pages_split_layer_units_and_keep_full_history() -> Result<()> {
        let window = Arc::new(GroupTable {
            shape: GroupShape::new(4, 2, Some(4))?,
            start_page: 1,
            units: (10..18).map(UnitId).collect(),
            allocated_tokens: 20,
        });
        let full = Arc::new(GroupTable {
            shape: GroupShape::new(8, 1, None)?,
            start_page: 0,
            units: vec![UnitId(21), UnitId(22), UnitId(23)],
            allocated_tokens: 20,
        });
        let tables = vec![
            vec![Arc::clone(&window), Arc::clone(&full)],
            vec![Arc::clone(&window), Arc::clone(&full)],
        ];

        // Token 8 reaches page 1; token 12 reaches page 2. The latter row
        // extends beyond the stored prefix, as a read-only canvas may do.
        let pages = table_pages(&tables, &[8, 12], &[1, 16])?;
        let values: Vec<Vec<Vec<u32>>> = pages
            .iter()
            .map(|table| table.rows().map(Iterator::collect).collect())
            .collect();
        assert_eq!(
            values,
            vec![
                vec![vec![10, 12], vec![12, 14, 16]],
                vec![vec![11, 13], vec![13, 15, 17]],
                vec![vec![21, 22, 23], vec![21, 22, 23]],
            ]
        );
        assert_eq!(pages[0].starts, [1, 2]);
        assert_eq!(pages[0].lengths, [2, 3]);
        assert_eq!(pages[0].width(), 3);
        assert_eq!(pages[2].starts, [0, 0]);

        // A retired history page cannot be reconstructed from later units.
        assert!(table_pages(&tables, &[7, 12], &[1, 1]).is_err());
        Ok(())
    }
}
