//! Worker physical-resource classes and pressure reports.

use serde::{Deserialize, Serialize};

/// The kinds of physical resource exposed by worker store telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceClass {
    KvBlock,
    EncoderOutput,
    ImageLatent,
}

impl ResourceClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            ResourceClass::KvBlock => "kv_block",
            ResourceClass::EncoderOutput => "encoder_output",
            ResourceClass::ImageLatent => "image_latent",
        }
    }

    pub fn unit(&self) -> &'static str {
        match self {
            ResourceClass::KvBlock => "pages",
            ResourceClass::EncoderOutput => "handles",
            ResourceClass::ImageLatent => "bytes",
        }
    }
}

/// Worker-reported pressure for one physical resource class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourcePressure {
    pub class: ResourceClass,
    pub total: u64,
    pub used: u64,
    pub evictable: u64,
    pub free: u64,
}
