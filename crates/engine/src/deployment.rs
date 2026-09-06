//! Component membership and model-parallel degrees resolved before worker launch.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub use uniserve_core::{
    ComponentDeployConfig, ComponentDistribution, ParallelConfig, SequenceParallel,
};

/// A serving instance's physical placement and independently configured components.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageDeployConfig {
    pub devices: Vec<usize>,
    #[serde(flatten)]
    pub components: BTreeMap<String, ComponentDeployConfig>,
}

impl StageDeployConfig {
    /// Validates expanded membership without claiming model/backend capability support.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.devices.is_empty(),
            "deployment devices must not be empty"
        );
        anyhow::ensure!(
            self.devices.iter().collect::<BTreeSet<_>>().len() == self.devices.len(),
            "deployment devices must be unique"
        );
        anyhow::ensure!(
            !self.components.is_empty(),
            "deployment must assign components"
        );
        for (name, component) in &self.components {
            anyhow::ensure!(!name.is_empty(), "component name must not be empty");
            anyhow::ensure!(
                !component.ranks.is_empty(),
                "component {name} requires members"
            );
            anyhow::ensure!(
                component
                    .ranks
                    .iter()
                    .all(|&rank| rank < self.devices.len()),
                "component {name} contains a rank outside deployment devices"
            );
            anyhow::ensure!(
                component.ranks.iter().collect::<BTreeSet<_>>().len() == component.ranks.len(),
                "component {name} repeats ranks"
            );
            let degree = component.parallel_config.world_size()?;
            if component.distribution.is_some() {
                anyhow::ensure!(
                    degree == 1,
                    "temporal component {name} requires local parallel_config"
                );
            } else {
                anyhow::ensure!(
                    degree == component.ranks.len(),
                    "component {name} has {} members but TP × sequence × pipeline requires {degree}",
                    component.ranks.len()
                );
            }
            anyhow::ensure!(
                component.units_per_rank > 0,
                "component {name} units_per_rank must be positive"
            );
        }
        Ok(())
    }

    /// Resolves the explicit device budget into the established H3 Ulysses placement.
    pub fn h3(devices: Vec<usize>) -> Self {
        let ranks: Vec<_> = (0..devices.len()).collect();
        let denoiser = ParallelConfig {
            sequence_parallel: SequenceParallel::Ulysses {
                ulysses_degree: ranks.len(),
            },
            ..ParallelConfig::default()
        };
        let encoder = ParallelConfig {
            tensor_parallel_size: ranks.len(),
            ..ParallelConfig::default()
        };
        Self {
            devices,
            components: BTreeMap::from([
                (
                    "denoiser".into(),
                    ComponentDeployConfig::parallel(ranks.clone(), denoiser),
                ),
                (
                    "text_encoder".into(),
                    ComponentDeployConfig::parallel(ranks.clone(), encoder),
                ),
                (
                    "video_decoder".into(),
                    ComponentDeployConfig {
                        ranks,
                        parallel_config: ParallelConfig::default(),
                        distribution: Some(ComponentDistribution::TemporalUnits),
                        units_per_rank: 1,
                    },
                ),
                (
                    "audio_decoder".into(),
                    ComponentDeployConfig::parallel(vec![0], ParallelConfig::default()),
                ),
                (
                    "output".into(),
                    ComponentDeployConfig::parallel(vec![0], ParallelConfig::default()),
                ),
            ]),
        }
    }
}

impl std::str::FromStr for StageDeployConfig {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let deployment: Self = serde_json::from_str(value)?;
        deployment.validate()?;
        Ok(deployment)
    }
}
