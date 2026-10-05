//! Step selection and layer participation for distributed routed experts.

use std::collections::HashSet;

use crate::{Error, Result};

/// Coordinate a serialized sequence of expert calls over one rank group.
///
/// Sources executing one forward together must all have the same call kind
/// ready. Other sources and expert-only ranks participate without local rows.
/// The transport supplies host records; selection performs no communication.
pub struct ExpertExchange {
    rank: usize,
    ranks: usize,
    sources: Vec<Vec<usize>>,
    max_tokens: usize,
    capacities: Vec<usize>,
    capacity: usize,
    kind: i64,
    active: bool,
    released: bool,
    ready: bool,
    invoked: HashSet<usize>,
}

impl ExpertExchange {
    /// Memberships use global ranks. `rank` is the local index in `ranks`.
    /// Fused transports retain maximum-sized symmetric storage; other
    /// transports use power-of-two capacities plus the configured maximum.
    pub fn new(
        ranks: &[usize],
        rank: usize,
        memberships: &[Vec<usize>],
        attention_ranks: usize,
        max_tokens: usize,
        fused: bool,
        needs_warmup: bool,
    ) -> Result<Self> {
        if ranks.len() < 2 || rank >= ranks.len() || max_tokens == 0 {
            return Err(Error::Invalid(
                "an expert exchange spans two or more ranks and one token".into(),
            ));
        }

        let source_count = if attention_ranks == 0 {
            ranks.len()
        } else {
            attention_ranks
        };
        if source_count > ranks.len() || memberships.len() != ranks.len() {
            return Err(Error::Invalid(
                "expert source membership is incomplete".into(),
            ));
        }

        let mut sources = Vec::new();
        for (index, members) in memberships.iter().enumerate() {
            let peers = members
                .iter()
                .map(|member| ranks.iter().position(|rank| rank == member))
                .collect::<Option<Vec<_>>>()
                .filter(|peers| {
                    if peers.is_empty() {
                        return index >= source_count;
                    }

                    peers.contains(&index)
                        && peers.iter().copied().collect::<HashSet<_>>().len() == peers.len()
                        && peers
                            .iter()
                            .all(|&peer| peer < source_count && memberships[peer] == *members)
                })
                .ok_or_else(|| {
                    Error::Invalid(
                        "expert sources must declare disjoint, consistent groups of ranks that execute each forward together".into(),
                    )
                })?;

            if peers.first() == Some(&index) {
                sources.push(peers);
            }
        }

        let capacities = if fused {
            vec![max_tokens]
        } else {
            (0..usize::BITS - (max_tokens - 1).leading_zeros())
                .map(|power| 1 << power)
                .chain([max_tokens])
                .collect()
        };

        Ok(Self {
            rank,
            ranks: ranks.len(),
            sources,
            max_tokens,
            capacities,
            capacity: 0,
            kind: -1,
            active: false,
            released: false,
            ready: !needs_warmup,
            invoked: HashSet::new(),
        })
    }

    /// Select from completed host records `(tokens, leaving, kind)`.
    ///
    /// The accessor borrows the transport's fixed receive buffer without
    /// copying it into a second record array. It must return stable values
    /// throughout this call. Ready kinds rotate in increasing cyclic order.
    pub fn select(&mut self, record: impl Fn(usize) -> (usize, bool, i64)) -> Result<usize> {
        self.active = false;
        self.released = (0..self.ranks).all(|rank| {
            let (tokens, leaving, _) = record(rank);
            tokens == 0 && leaving
        });

        let mut selected: Option<(i64, usize, bool)> = None;
        for members in &self.sources {
            let (_, _, kind) = record(members[0]);
            let mut most = 0;
            if !members.iter().all(|&rank| {
                let (tokens, _, tag) = record(rank);
                most = most.max(tokens);
                tokens != 0 && tag == kind
            }) {
                continue;
            }

            let active = members.contains(&self.rank);
            match &mut selected {
                Some((tag, tokens, local)) if *tag == kind => {
                    *tokens = (*tokens).max(most);
                    *local |= active;
                }
                Some((tag, _, _)) if (kind <= self.kind, kind) >= (*tag <= self.kind, *tag) => {}
                _ => selected = Some((kind, most, active)),
            }
        }

        let Some((kind, most, active)) = selected else {
            return Ok(0);
        };
        self.kind = kind;
        self.active = active;

        self.capacities
            .iter()
            .copied()
            .find(|&capacity| capacity >= most)
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "expert step of {most} tokens exceeds the configured {} tokens per rank",
                    self.max_tokens
                ))
            })
    }

    pub fn capacities(&self) -> &[usize] {
        &self.capacities
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn kind(&self) -> i64 {
        self.kind
    }

    pub fn active(&self) -> bool {
        self.active
    }

    pub fn released(&self) -> bool {
        self.released
    }

    pub fn begin(&mut self, capacity: usize) -> Result<()> {
        if capacity == 0 || capacity > self.max_tokens {
            return Err(Error::Invalid(format!(
                "step capacity {capacity} is outside the exchange's {} tokens",
                self.max_tokens
            )));
        }

        self.capacity = capacity;
        self.reset_layers();
        Ok(())
    }

    pub fn end(&mut self) {
        self.capacity = 0;
        self.reset_layers();
    }

    /// The first communication waits for every peer's numerical preparation.
    /// A transport marks that readiness only after its host barrier succeeds.
    pub fn needs_warmup(&self) -> Result<bool> {
        if self.capacity == 0 {
            return Err(Error::State("an expert exchange runs inside an open step"));
        }
        Ok(!self.ready)
    }

    pub fn enter(&mut self, module: usize) -> Result<()> {
        self.needs_warmup()?;
        self.ready = true;
        self.invoked.insert(module);
        Ok(())
    }

    /// Return the unvisited tail in collective order, before any join launches.
    pub fn pending_layers(&self, modules: impl IntoIterator<Item = usize>) -> Result<Vec<usize>> {
        let mut pending = Vec::new();
        for module in modules {
            if !self.invoked.contains(&module) {
                pending.push(module);
            } else if !pending.is_empty() {
                return Err(Error::State(
                    "a forward skipped an expert layer before one it reached",
                ));
            }
        }

        Ok(pending)
    }

    pub fn invoked(&self) -> impl Iterator<Item = usize> + '_ {
        self.invoked.iter().copied()
    }

    pub fn reset_layers(&mut self) {
        self.invoked.clear();
    }

    /// Graph replay runs no host layer hooks; the captured call names them.
    pub fn record_layers(&mut self, modules: impl IntoIterator<Item = usize>) {
        self.invoked.extend(modules);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn exchange(rank: usize, fused: bool) -> ExpertExchange {
        ExpertExchange::new(
            &[8, 12, 20, 40, 41],
            rank,
            &[vec![8, 20], vec![12], vec![8, 20], vec![], vec![]],
            3,
            192,
            fused,
            true,
        )
        .unwrap()
    }

    #[test]
    fn source_groups_wait_together_and_ready_kinds_take_turns() {
        for rank in 0..5 {
            let mut exchange = exchange(rank, false);
            let mut records = [
                (129, false, 2),
                (7, false, 0),
                (0, false, 2),
                (0, false, 0),
                (0, false, 0),
            ];
            assert_eq!(exchange.select(|rank| records[rank]).unwrap(), 8);
            assert_eq!(exchange.kind(), 0);
            assert_eq!(exchange.active(), rank == 1);

            records[2].0 = 64;
            assert_eq!(exchange.select(|rank| records[rank]).unwrap(), 192);
            assert_eq!(exchange.kind(), 2);
            assert_eq!(exchange.active(), rank == 0 || rank == 2);
            assert_eq!(exchange.select(|rank| records[rank]).unwrap(), 8);

            records[2].2 = 1;
            assert_eq!(exchange.select(|rank| records[rank]).unwrap(), 8);
            records[2].2 = 2;
            records[0].0 = 193;
            assert!(exchange.select(|rank| records[rank]).is_err());
        }
    }

    #[test]
    fn release_waits_for_every_rank_and_no_remaining_tokens() {
        let mut exchange = exchange(1, true);
        let mut records = [(0, true, 0); 5];
        records[4].1 = false;
        assert_eq!(exchange.select(|rank| records[rank]).unwrap(), 0);
        assert!(!exchange.released());

        records[4].1 = true;
        records[1].0 = 1;
        assert_eq!(exchange.select(|rank| records[rank]).unwrap(), 192);
        assert!(!exchange.released());

        records[1].0 = 0;
        assert_eq!(exchange.select(|rank| records[rank]).unwrap(), 0);
        assert!(exchange.released());
    }

    #[test]
    fn eager_and_captured_layers_share_one_step() {
        let mut exchange = exchange(0, false);
        assert!(exchange.enter(10).is_err());
        exchange.begin(192).unwrap();
        assert!(exchange.needs_warmup().unwrap());
        exchange.enter(10).unwrap();
        exchange.record_layers([20, 30]);
        assert_eq!(exchange.pending_layers([10, 20, 30, 40]).unwrap(), [40]);
        assert!(exchange.pending_layers([10, 15, 20]).is_err());
        assert!(!exchange.needs_warmup().unwrap());

        exchange.reset_layers();
        assert_eq!(exchange.capacity(), 192);
        assert_eq!(exchange.pending_layers([10, 20]).unwrap(), [10, 20]);
        exchange.enter(10).unwrap();
        exchange.end();
        assert_eq!(exchange.capacity(), 0);
        assert!(exchange.enter(20).is_err());
    }

    #[test]
    fn overlapping_source_groups_are_rejected() {
        assert!(
            ExpertExchange::new(
                &[0, 1, 2],
                0,
                &[vec![0, 2], vec![1, 2], vec![0, 2]],
                0,
                32,
                false,
                false,
            )
            .is_err()
        );
    }
}
