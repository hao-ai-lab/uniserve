//! Rank membership and logical model-parallel degrees of worker components.
//!
//! A worker group names each numerical component it runs and the ordered
//! ranks that run it (`WorkerConfig::components` in the engine). A
//! [`ComponentConfig`] either forms one model-parallel group over its ranks
//! or, with a [`ComponentDistribution`], runs an independent local instance on
//! every rank and deals independent units among them.
//!
//! [`ComponentConfig::validate`] checks membership and parallel degrees for
//! launch, worker registration and Python callers. Process-world bounds belong
//! to the owner of that world. The same configuration crosses worker IPC as
//! the flatbuffer `ParallelConfig` and `ComponentInfo` tables.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

/// Invalid component membership or parallel degrees.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ParallelConfigError(pub &'static str);

/// Serde default for an omitted sequence-parallel degree and for
/// `units_per_rank`.
const fn one() -> usize {
    1
}

/// One mutually exclusive sequence attention strategy.
///
/// Serialized with a `kind` tag (`local`, `ulysses`, `allgather`, `hybrid`);
/// an omitted degree defaults to one. The flatbuffer schema carries the same
/// choice as the `SequenceParallel` union.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SequenceParallel {
    /// No sequence parallelism.
    #[default]
    Local,
    /// Ulysses sequence parallelism over `ulysses_degree` ranks.
    Ulysses {
        #[serde(default = "one")]
        ulysses_degree: usize,
    },
    /// Context parallelism that gathers every context partition's keys and
    /// values onto each of `allgather_degree` ranks.
    Allgather {
        #[serde(default = "one")]
        allgather_degree: usize,
    },
    /// Ulysses and all-gather context parallelism composed; the sequence
    /// degree is the product of both.
    Hybrid {
        #[serde(default = "one")]
        ulysses_degree: usize,
        #[serde(default = "one")]
        allgather_degree: usize,
    },
}

impl SequenceParallel {
    /// Context ranks precede the Ulysses ranks, which vary fastest.
    pub fn dimensions(&self) -> [(&'static str, usize); 2] {
        let (context, ulysses) = match *self {
            Self::Local => (1, 1),
            Self::Ulysses { ulysses_degree } => (1, ulysses_degree),
            Self::Allgather { allgather_degree } => (allgather_degree, 1),
            Self::Hybrid {
                ulysses_degree,
                allgather_degree,
            } => (allgather_degree, ulysses_degree),
        };
        [("cp", context), ("ulysses", ulysses)]
    }

    /// Derives sequence degree from the selected strategy's independent dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`ParallelConfigError`] when a degree is zero or the product
    /// overflows.
    pub fn size(&self) -> Result<usize, ParallelConfigError> {
        let degrees = match *self {
            Self::Local => vec![1],
            Self::Ulysses { ulysses_degree } => vec![ulysses_degree],
            Self::Allgather { allgather_degree } => vec![allgather_degree],
            Self::Hybrid {
                ulysses_degree,
                allgather_degree,
            } => vec![ulysses_degree, allgather_degree],
        };
        checked_product(&degrees)
    }
}

/// Multiplies degrees, rejecting a zero degree and overflow.
fn checked_product(degrees: &[usize]) -> Result<usize, ParallelConfigError> {
    degrees.iter().try_fold(1usize, |size, &degree| {
        if degree == 0 {
            return Err(ParallelConfigError("parallel degrees must be positive"));
        }
        size.checked_mul(degree)
            .ok_or(ParallelConfigError("parallel degree product overflow"))
    })
}

/// Logical model-parallel degrees for one independently placed component.
///
/// Omitted fields take their defaults: degree one and [`SequenceParallel::Local`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ParallelConfig {
    pub tensor_parallel_size: usize,
    pub pipeline_parallel_size: usize,
    pub sequence_parallel: SequenceParallel,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            sequence_parallel: SequenceParallel::Local,
        }
    }
}

impl ParallelConfig {
    /// Row-major rank order shared by placement, numerical meshes and routing.
    pub fn dimensions(&self) -> [(&'static str, usize); 4] {
        let [context, ulysses] = self.sequence_parallel.dimensions();
        [
            ("pp", self.pipeline_parallel_size),
            context,
            ("tp", self.tensor_parallel_size),
            ulysses,
        ]
    }

    /// Returns the rank count one model-parallel group needs: the product of
    /// the tensor, pipeline, and sequence degrees.
    ///
    /// # Errors
    ///
    /// Returns [`ParallelConfigError`] when a degree is zero or the product
    /// overflows.
    pub fn world_size(&self) -> Result<usize, ParallelConfigError> {
        checked_product(&[
            self.tensor_parallel_size,
            self.pipeline_parallel_size,
            self.sequence_parallel.size()?,
        ])
    }
}

/// Components may distribute independent temporal decode units over local decoders.
///
/// The flatbuffer `ComponentDistribution` enum encodes an absent distribution
/// (`None` in [`ComponentConfig::distribution`]) as `Local`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentDistribution {
    /// Every member rank runs its own local instance, and independent media
    /// units are dealt to the ranks in `ranks` order, `units_per_rank`
    /// consecutive units each.
    TemporalUnits,
}

/// Rank membership and parallel layout of one named component in a worker
/// group.
///
/// Without a `distribution`, `ranks` form one model-parallel group and must
/// number exactly `parallel_config.world_size()`. With a distribution, the
/// parallel world size must be one. Call [`Self::validate`] after parsing and
/// before using the rank-selection methods.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentConfig {
    /// Indices into the worker group's ordered rank list; their order assigns
    /// each rank its position in the parallel degrees or unit schedule.
    pub ranks: Vec<usize>,
    #[serde(default)]
    pub parallel_config: ParallelConfig,
    /// How independent units are spread over `ranks`; `None` runs the
    /// component as one model-parallel group.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distribution: Option<ComponentDistribution>,
    /// Consecutive units each rank takes under `distribution`; must be
    /// positive.
    #[serde(default = "one")]
    pub units_per_rank: usize,
}

impl ComponentConfig {
    /// Check a component's membership and its independent parallel degrees.
    /// Process-world bounds are checked by the caller that owns that world.
    pub fn validate(&self) -> Result<(), ParallelConfigError> {
        if self.ranks.is_empty()
            || self.ranks.iter().collect::<HashSet<_>>().len() != self.ranks.len()
        {
            return Err(ParallelConfigError(
                "component ranks must be unique and non-empty",
            ));
        }
        if self.units_per_rank == 0 {
            return Err(ParallelConfigError("units_per_rank must be positive"));
        }

        let degree = self.parallel_config.world_size()?;
        if self.distribution.is_some() {
            if degree != 1 {
                return Err(ParallelConfigError(
                    "temporal unit distribution requires local decoder parallelism",
                ));
            }
        } else if degree != self.ranks.len() {
            return Err(ParallelConfigError(
                "component membership must equal TP × sequence × pipeline degrees",
            ));
        }
        Ok(())
    }

    /// First pipeline-stage input ranks, in component order.
    pub fn input_ranks(&self) -> &[usize] {
        if self.distribution.is_some() {
            &self.ranks
        } else {
            &self.ranks[..self.ranks.len() / self.parallel_config.pipeline_parallel_size]
        }
    }

    /// Final-stage output ranks with tensor replicas counted once.
    pub fn output_ranks(&self) -> Vec<usize> {
        if self.distribution.is_some() {
            return self.ranks.clone();
        }
        let stage = self.ranks.len() / self.parallel_config.pipeline_parallel_size;
        let ulysses = self.parallel_config.sequence_parallel.dimensions()[1].1;
        self.ranks
            .iter()
            .enumerate()
            .filter_map(|(index, &rank)| {
                (index >= self.ranks.len() - stage
                    && (index / ulysses).is_multiple_of(self.parallel_config.tensor_parallel_size))
                .then_some(rank)
            })
            .collect()
    }

    /// Builds an undistributed component that runs `parallel_config` over
    /// `ranks`.
    pub fn parallel(ranks: Vec<usize>, parallel_config: ParallelConfig) -> Self {
        Self {
            ranks,
            parallel_config,
            distribution: None,
            units_per_rank: 1,
        }
    }

    /// Reports whether every member rank publishes its own copy of the
    /// component's products.
    ///
    /// A distributed component's ranks each publish the units dealt to them,
    /// and sequence-parallel ranks each publish their own shard. A tensor
    /// replica publishes once, from tensor-parallel coordinate zero, and only
    /// the final pipeline stage publishes, so a component with either degree
    /// above one leaves some member ranks without a copy. The worker's
    /// `ComponentBinding.output_ranks` selects the publishing ranks by the
    /// same rule.
    pub fn publishes_on_every_rank(&self) -> bool {
        self.distribution.is_some()
            || (self.parallel_config.tensor_parallel_size == 1
                && self.parallel_config.pipeline_parallel_size == 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_replicated_or_pipelined_components_leave_ranks_without_a_copy() {
        let component = |tensor_parallel_size, pipeline_parallel_size, sequence_parallel| {
            ComponentConfig::parallel(
                vec![0, 1, 2, 3],
                ParallelConfig {
                    tensor_parallel_size,
                    pipeline_parallel_size,
                    sequence_parallel,
                },
            )
        };
        let ulysses = |ulysses_degree| SequenceParallel::Ulysses { ulysses_degree };

        assert!(component(1, 1, ulysses(4)).publishes_on_every_rank());
        assert!(!component(4, 1, SequenceParallel::Local).publishes_on_every_rank());
        assert!(!component(2, 1, ulysses(2)).publishes_on_every_rank());
        assert!(!component(1, 4, SequenceParallel::Local).publishes_on_every_rank());

        let mut distributed = ComponentConfig::parallel(vec![0, 1, 2, 3], Default::default());
        distributed.distribution = Some(ComponentDistribution::TemporalUnits);
        assert!(distributed.publishes_on_every_rank());
    }
}
