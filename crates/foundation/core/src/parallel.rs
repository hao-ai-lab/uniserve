//! Rank membership and logical model-parallel degrees of worker components.
//!
//! A worker group names each numerical component it runs and the ordered
//! ranks that run it (`WorkerConfig::components` in the engine). A
//! [`ComponentConfig`] either forms one model-parallel group over its ranks
//! or, with a [`ComponentDistribution`], runs an independent local instance on
//! every rank and deals independent units among them.
//!
//! These types describe the layout only. [`SequenceParallel::size`] and
//! [`ParallelConfig::world_size`] reject zero degrees and overflow, but the
//! agreement between a component's ranks and its degrees is checked by
//! `WorkerConfig::validate_members` in the engine, by `WorkerInfo::validate`
//! in the worker IPC crate, and by the Python mirror's `ComponentConfig` in
//! `uniserve_worker.config.deployment`. The same shape crosses the worker
//! boundary as the flatbuffer `ParallelConfig` and `ComponentInfo` tables.

use serde::{Deserialize, Serialize};

/// Invalid parallel degrees: a zero degree or a degree product that overflows
/// `usize`.
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
/// parallel world size must be one. `WorkerConfig::validate_members` in the
/// engine and `WorkerInfo::validate` in the worker IPC crate enforce both
/// rules in Rust; this type does not.
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
}
