//! Load-bound model control tokens carried by the startup handshake.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationControlTokens {
    pub bos: u32,
    pub eos: Vec<u32>,
    pub end_of_image: u32,
}
