//! Host decoding for the sampler's packed score columns.

use std::collections::{HashMap, HashSet};

use uniserve_core::TokenLogprob;

use crate::{Error, Result};

/// Row selection and padded widths of one packed logprob column.
/// Scores carry float32 bits in the low half of each int64 word.
pub struct LogprobLayout {
    pub rows: Vec<usize>,
    pub counts: Vec<usize>,
    pub requested_ids: Vec<Vec<u32>>,
    pub max_count: usize,
    pub max_requested: usize,
}

impl LogprobLayout {
    pub(super) fn validate(&self, words: usize) -> Result<()> {
        let width = self.max_count.checked_mul(3).and_then(|top| {
            self.max_requested
                .checked_mul(2)
                .and_then(|requested| top.checked_add(requested)?.checked_add(3))
        });
        if width.and_then(|width| self.rows.len().checked_mul(width)) != Some(words)
            || self.counts.len() != self.rows.len()
            || self.requested_ids.len() != self.rows.len()
            || self.counts.iter().any(|&count| count > self.max_count)
            || self
                .requested_ids
                .iter()
                .any(|ids| ids.len() > self.max_requested)
        {
            return Err(Error::Invariant(
                "logprob completion does not match its row layout".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn entries(&self, row: usize) -> Result<usize> {
        let local = self
            .rows
            .iter()
            .position(|&index| index == row)
            .ok_or(Error::State("logprob capture has no requested row"))?;
        Ok(1 + self.counts[local] + self.requested_ids[local].len())
    }

    pub(super) fn decode(&self, words: &[i64]) -> Result<HashMap<usize, Vec<TokenLogprob>>> {
        let rows = self.rows.len();

        // Each field contains all rows before the next field begins. Top and
        // requested-token fields are padded; only their row counts are visible.
        let top_ids = rows * 3;
        let top_values = top_ids + rows * self.max_count;
        let top_ranks = top_values + rows * self.max_count;
        let requested_values = top_ranks + rows * self.max_count;
        let requested_ranks = requested_values + rows * self.max_requested;
        let mut result = HashMap::with_capacity(rows);

        for (local, &row) in self.rows.iter().enumerate() {
            let mut entries =
                Vec::with_capacity(1 + self.counts[local] + self.requested_ids[local].len());
            let mut seen = HashSet::with_capacity(entries.capacity());
            let mut append = |token, score, rank| -> Result<()> {
                let token_id = u32::try_from(token)
                    .map_err(|_| Error::State("logprob token is outside the token id range"))?;
                if seen.insert(token_id) {
                    entries.push(TokenLogprob {
                        token_id,
                        logprob: f32::from_bits(score as u32),
                        rank: u32::try_from(rank)
                            .map_err(|_| Error::State("logprob rank is outside the rank range"))?,
                    });
                }
                Ok(())
            };

            append(words[local], words[rows + local], words[rows * 2 + local])?;
            for index in 0..self.counts[local] {
                let column = local * self.max_count + index;
                append(
                    words[top_ids + column],
                    words[top_values + column],
                    words[top_ranks + column],
                )?;
            }
            for (index, &token) in self.requested_ids[local].iter().enumerate() {
                let column = local * self.max_requested + index;
                append(
                    i64::from(token),
                    words[requested_values + column],
                    words[requested_ranks + column],
                )?;
            }
            result.insert(row, entries);
        }
        Ok(result)
    }
}
