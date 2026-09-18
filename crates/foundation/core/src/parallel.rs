//! Shared logical parallelism and ordered component params declarations.

use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ParallelConfigError(pub &'static str);

const fn one() -> usize {
    1
}

/// One mutually exclusive sequence attention strategy.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SequenceParallel {
    #[default]
    Local,
    Ulysses {
        #[serde(default = "one")]
        ulysses_degree: usize,
    },
    Ring {
        #[serde(default = "one")]
        ring_degree: usize,
    },
    Hybrid {
        #[serde(default = "one")]
        ulysses_degree: usize,
        #[serde(default = "one")]
        ring_degree: usize,
    },
    Allgather {
        #[serde(default = "one")]
        allgather_degree: usize,
    },
    Attention2d {
        #[serde(default = "one")]
        attn2d_row_size: usize,
        #[serde(default = "one")]
        attn2d_col_size: usize,
        #[serde(default = "one")]
        ulysses_degree: usize,
    },
}

impl SequenceParallel {
    /// Derives sequence degree from the selected strategy's independent dimensions.
    pub fn size(&self) -> Result<usize, ParallelConfigError> {
        let degrees = match *self {
            Self::Local => vec![1],
            Self::Ulysses { ulysses_degree } => vec![ulysses_degree],
            Self::Ring { ring_degree } => vec![ring_degree],
            Self::Hybrid {
                ulysses_degree,
                ring_degree,
            } => vec![ulysses_degree, ring_degree],
            Self::Allgather { allgather_degree } => vec![allgather_degree],
            Self::Attention2d {
                attn2d_row_size,
                attn2d_col_size,
                ulysses_degree,
            } => {
                vec![attn2d_row_size, attn2d_col_size, ulysses_degree]
            }
        };
        checked_product(&degrees)
    }
}

fn checked_product(degrees: &[usize]) -> Result<usize, ParallelConfigError> {
    degrees.iter().try_fold(1usize, |size, &degree| {
        if degree == 0 {
            return Err(ParallelConfigError("parallel degrees must be positive"));
        }
        size.checked_mul(degree)
            .ok_or_else(|| ParallelConfigError("parallel degree product overflow"))
    })
}

/// Logical model-parallel degrees for one independently placed component.
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
    pub fn world_size(&self) -> Result<usize, ParallelConfigError> {
        checked_product(&[
            self.tensor_parallel_size,
            self.pipeline_parallel_size,
            self.sequence_parallel.size()?,
        ])
    }
}

/// Components may distribute independent temporal decode units over local decoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentDistribution {
    TemporalUnits,
}

/// CallKind entry members index the ordered ranks of a Worker instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentConfig {
    pub ranks: Vec<usize>,
    #[serde(default)]
    pub parallel_config: ParallelConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distribution: Option<ComponentDistribution>,
    #[serde(default = "one")]
    pub units_per_rank: usize,
}

impl ComponentConfig {
    pub fn parallel(ranks: Vec<usize>, parallel_config: ParallelConfig) -> Self {
        Self {
            ranks,
            parallel_config,
            distribution: None,
            units_per_rank: 1,
        }
    }
}
