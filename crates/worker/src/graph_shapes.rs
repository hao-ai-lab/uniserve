//! Capture sizes and serving selection for homogeneous text graphs.

use std::collections::BTreeSet;

use crate::{Error, Result};

/// One prefill capture, including its extra padding row and attention mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PrefillShape {
    /// Flat tokens and rows held by the captured numerical input.
    pub token_bucket: usize,
    pub row_bucket: usize,
    /// Least rows served by this bucket: one, or the preceding row bucket.
    /// Startup uses this count to fit scratch KV within the physical pool.
    pub live_rows: usize,
    /// None selects per-row device causality flags.
    pub causal: Option<bool>,
    pub embeddings: bool,
    /// False captures KV writes alone, without computing hidden outputs.
    pub outputs: bool,
}

/// Fewest units held by independent rows over all cache groups.
/// Each group supplies `(tokens per page, pool units per page)` in `pages`.
pub fn prefill_units(pages: &[(usize, usize)], rows: usize, tokens: usize) -> usize {
    pages
        .iter()
        .map(|&(page, units)| units * rows.max(tokens.div_ceil(page)))
        .sum()
}

/// Build configured prefill captures that can serve an admitted batch.
/// A row bucket needs an extra sequence for padding. Its first token bucket
/// serves one token per live row; later buckets begin above the previous size.
#[allow(clippy::too_many_arguments)]
pub fn prefill_shapes(
    token_sizes: &[usize],
    row_sizes: &[usize],
    max_rows: usize,
    max_tokens: usize,
    variants: &[(Option<bool>, bool)],
    outputs: bool,
    pool: Option<(&[(usize, usize)], usize)>,
) -> Vec<PrefillShape> {
    let mut tokens: BTreeSet<_> = token_sizes
        .iter()
        .copied()
        .filter(|&size| size <= max_tokens)
        .collect();
    tokens.insert(max_tokens);
    let rows: BTreeSet<_> = row_sizes.iter().copied().filter(|&size| size > 1).collect();
    let mut shapes = Vec::new();
    let mut live_rows = 1;

    for row_bucket in rows {
        if live_rows > max_rows {
            break;
        }
        let minimum = if live_rows == 1 { 1 } else { live_rows + 1 };
        let mut least = live_rows;
        for &token_bucket in tokens.range(minimum..) {
            if pool.is_some_and(|(pages, units)| prefill_units(pages, live_rows, least) > units) {
                break;
            }
            shapes.extend(variants.iter().map(|&(causal, embeddings)| PrefillShape {
                token_bucket,
                row_bucket,
                live_rows,
                causal,
                embeddings,
                outputs,
            }));
            least = token_bucket + 1;
        }
        live_rows = row_bucket;
    }
    shapes
}

/// Immutable capture sizes of one text runner. The numerical backend owns
/// graph objects and fixed buffers; this owner chooses their serving shape.
pub struct TextShapes {
    decode: Vec<usize>,
    prefill: Vec<PrefillShape>,
    cache_only: bool,
    outputs: bool,
}

impl TextShapes {
    pub fn new(decode: Vec<usize>, prefill: Vec<PrefillShape>) -> Self {
        Self {
            cache_only: prefill.iter().any(|shape| !shape.outputs),
            outputs: prefill.iter().any(|shape| shape.outputs),
            decode,
            prefill,
        }
    }

    pub fn decode(&self) -> &[usize] {
        &self.decode
    }

    pub fn prefill(&self) -> &[PrefillShape] {
        &self.prefill
    }

    /// Return `(rows, tokens, decode, outputs)`, or None for eager prefill.
    /// Decode never borrows a prefill bucket. A cache-only prefill uses a
    /// hidden-state graph only when no cache-only graph family was captured.
    #[allow(clippy::too_many_arguments)]
    pub fn select(
        &self,
        queries: &[usize],
        tokens: usize,
        decode: bool,
        causal: Option<bool>,
        embeddings: bool,
        last_logits: bool,
        cache_only: bool,
    ) -> Result<Option<(usize, usize, bool, bool)>> {
        let rows = queries.len();
        if rows == 0 {
            return Err(Error::Invalid(
                "text graphs require a nonempty query batch".into(),
            ));
        }
        if decode && queries.iter().all(|&length| length == 1) {
            if causal == Some(true)
                && last_logits
                && let Some(&size) = self.decode.iter().find(|&&size| size >= rows)
            {
                return Ok(Some((size, size, true, true)));
            }
            return Err(Error::Invalid(format!(
                "no decode graph for {rows} rows; configured maximum is {} rows",
                self.decode.iter().copied().max().unwrap_or(0)
            )));
        }

        let outputs = !(cache_only && self.cache_only);
        if outputs && !self.outputs {
            return Ok(None);
        }
        if !queries.contains(&0)
            && let Some(shape) = self
                .prefill
                .iter()
                .filter(|shape| {
                    shape.outputs == outputs
                        && shape.causal == causal
                        && shape.embeddings == embeddings
                        && shape.row_bucket > rows
                        && shape.token_bucket >= tokens
                })
                .min_by_key(|shape| (shape.row_bucket, shape.token_bucket))
        {
            return Ok(Some((shape.row_bucket, shape.token_bucket, false, outputs)));
        }
        Err(Error::Invalid(format!(
            "no prefill graph for {rows} rows and {tokens} tokens with causal={causal:?}, embeddings={embeddings} and outputs={outputs}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefill(rows: usize, tokens: usize, causal: Option<bool>, outputs: bool) -> PrefillShape {
        PrefillShape {
            row_bucket: rows,
            token_bucket: tokens,
            live_rows: 1,
            causal,
            embeddings: false,
            outputs,
        }
    }

    #[test]
    fn decode_keeps_configured_order_and_never_uses_prefill_capacity() -> Result<()> {
        let shapes = TextShapes::new(vec![8, 4, 16], vec![prefill(32, 64, Some(true), true)]);
        assert_eq!(
            shapes.select(&[1; 3], 3, true, Some(true), false, true, false)?,
            Some((8, 8, true, true))
        );
        assert!(
            shapes
                .select(&[1; 17], 17, true, Some(true), false, true, false)
                .is_err()
        );
        assert!(
            shapes
                .select(&[1], 1, true, Some(false), false, true, false)
                .is_err()
        );
        assert!(
            shapes
                .select(&[1], 1, true, Some(true), false, false, false)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn prefill_selects_padding_capacity_and_matching_numerical_semantics() -> Result<()> {
        let shapes = TextShapes::new(
            vec![],
            vec![
                prefill(8, 16, Some(true), true),
                prefill(4, 64, Some(true), true),
                prefill(4, 32, None, true),
                prefill(4, 16, Some(true), false),
            ],
        );
        assert_eq!(
            shapes.select(&[2, 3], 5, false, Some(true), false, false, false)?,
            Some((4, 64, false, true))
        );
        assert_eq!(
            shapes.select(&[2, 3], 5, false, Some(true), false, false, true)?,
            Some((4, 16, false, false))
        );
        assert_eq!(
            shapes.select(&[2, 3], 5, false, None, false, false, false)?,
            Some((4, 32, false, true))
        );
        assert_eq!(
            shapes.select(&[1; 4], 4, false, Some(true), false, true, false)?,
            Some((8, 16, false, true))
        );
        assert!(
            shapes
                .select(&[2], 2, false, Some(true), true, false, false)
                .is_err()
        );
        assert!(
            shapes
                .select(&[0], 0, false, Some(true), false, false, false)
                .is_err()
        );

        let output_only = TextShapes::new(vec![], vec![prefill(2, 16, Some(true), true)]);
        assert_eq!(
            output_only.select(&[2], 2, false, Some(true), false, false, true)?,
            Some((2, 16, false, true))
        );
        let cache_only = TextShapes::new(vec![], vec![prefill(2, 16, Some(true), false)]);
        assert_eq!(
            cache_only.select(&[2], 2, false, Some(true), false, true, false)?,
            None
        );
        Ok(())
    }

    #[test]
    fn captures_exclude_buckets_no_admitted_batch_can_reach() {
        let shapes = prefill_shapes(
            &[16, 32],
            &[1, 2, 4, 8],
            3,
            64,
            &[(Some(true), false)],
            true,
            Some((&[(4, 1)], 4)),
        );
        assert_eq!(
            shapes
                .iter()
                .map(|shape| (shape.row_bucket, shape.token_bucket, shape.live_rows))
                .collect::<Vec<_>>(),
            vec![(2, 16, 1), (4, 16, 2)]
        );
    }
}
