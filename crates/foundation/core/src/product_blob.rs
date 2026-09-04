//! Binary encodings for bounded values carried by worker result products.
//!
//! Payloads use deterministic little-endian, length-prefixed layouts so worker
//! producers and host consumers agree on the exact representation.

/// A rejected [`LogprobBlob`] byte payload.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProductBlobError {
    /// The payload ends before a declared value is complete.
    #[error("logprob blob is truncated")]
    Truncated,
    /// The sampled-logprob presence flag is outside its binary domain.
    #[error("logprob blob sampled flag {0} is invalid")]
    InvalidSampledFlag(u8),
    /// Bytes remain after the declared payload ends.
    #[error("logprob blob has trailing bytes")]
    TrailingBytes,
}

/// One ranked vocabulary candidate at a generated or prompt token position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RankedToken {
    /// Vocabulary token identity.
    pub token_id: u32,
    /// Natural-log probability assigned to the token.
    pub logprob: f32,
    /// Zero-based probability rank at the position.
    pub rank: u32,
}

/// The logprob values a token operation produced: the sampled logprob and its
/// ranked candidates for the committed token, plus per-prompt-position ranked
/// candidates for a prefill that scored its prompt.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LogprobBlob {
    /// Log probability of the sampled token, when requested.
    pub sampled_logprob: Option<f32>,
    /// Ranked candidates for the generated position.
    pub top_logprobs: Vec<RankedToken>,
    /// Ranked candidates for each scored prompt position.
    pub prompt_logprobs: Vec<Vec<RankedToken>>,
}

impl LogprobBlob {
    /// Returns whether the blob contains no sampled or positional scores.
    pub fn is_empty(&self) -> bool {
        self.sampled_logprob.is_none()
            && self.top_logprobs.is_empty()
            && self.prompt_logprobs.is_empty()
    }

    /// Encodes the blob into its deterministic little-endian representation.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();

        // Prefix the sampled score with a one-byte presence discriminant.
        match self.sampled_logprob {
            Some(value) => {
                out.push(1);
                out.extend_from_slice(&value.to_le_bytes());
            }
            None => out.push(0),
        }

        // Each candidate collection carries its own element count so readers
        // can advance without schema-dependent offsets.
        encode_tokens(&mut out, &self.top_logprobs);
        out.extend_from_slice(&(self.prompt_logprobs.len() as u32).to_le_bytes());
        for position in &self.prompt_logprobs {
            encode_tokens(&mut out, position);
        }
        out
    }

    /// Decodes and validates one complete blob representation.
    pub fn decode(bytes: &[u8]) -> Result<Self, ProductBlobError> {
        let mut reader = Reader { bytes, offset: 0 };

        // Decode the discriminant before consuming the optional score payload.
        let sampled_logprob = match reader.u8()? {
            0 => None,
            1 => Some(reader.f32()?),
            other => return Err(ProductBlobError::InvalidSampledFlag(other)),
        };

        // Candidate groups use the same length-prefixed record encoding for
        // generated and prompt positions.
        let top_logprobs = reader.tokens()?;
        let position_count = reader.u32()? as usize;
        let mut prompt_logprobs = Vec::with_capacity(position_count);
        for _ in 0..position_count {
            prompt_logprobs.push(reader.tokens()?);
        }

        // A blob is one complete value; unconsumed bytes indicate a producer
        // and consumer schema mismatch.
        if reader.offset != reader.bytes.len() {
            return Err(ProductBlobError::TrailingBytes);
        }

        Ok(Self {
            sampled_logprob,
            top_logprobs,
            prompt_logprobs,
        })
    }
}

/// Appends one length-prefixed sequence of fixed-width token records.
fn encode_tokens(out: &mut Vec<u8>, tokens: &[RankedToken]) {
    out.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    for token in tokens {
        out.extend_from_slice(&token.token_id.to_le_bytes());
        out.extend_from_slice(&token.logprob.to_le_bytes());
        out.extend_from_slice(&token.rank.to_le_bytes());
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl Reader<'_> {
    /// Advances over exactly `len` bytes or rejects a truncated payload.
    fn take(&mut self, len: usize) -> Result<&[u8], ProductBlobError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(ProductBlobError::Truncated)?;
        if end > self.bytes.len() {
            return Err(ProductBlobError::Truncated);
        }

        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    /// Reads one byte at the current cursor.
    fn u8(&mut self) -> Result<u8, ProductBlobError> {
        Ok(self.take(1)?[0])
    }

    /// Reads one little-endian 32-bit integer at the current cursor.
    fn u32(&mut self) -> Result<u32, ProductBlobError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    /// Reads one little-endian IEEE 754 value at the current cursor.
    fn f32(&mut self) -> Result<f32, ProductBlobError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    /// Reads one length-prefixed sequence of fixed-width token records.
    fn tokens(&mut self) -> Result<Vec<RankedToken>, ProductBlobError> {
        let count = self.u32()? as usize;
        let mut tokens = Vec::with_capacity(count);
        for _ in 0..count {
            let token_id = self.u32()?;
            let logprob = self.f32()?;
            let rank = self.u32()?;
            tokens.push(RankedToken {
                token_id,
                logprob,
                rank,
            });
        }
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logprob_blob_round_trips() {
        let blob = LogprobBlob {
            sampled_logprob: Some(-0.5),
            top_logprobs: vec![
                RankedToken {
                    token_id: 7,
                    logprob: -0.5,
                    rank: 0,
                },
                RankedToken {
                    token_id: 9,
                    logprob: -1.25,
                    rank: 1,
                },
            ],
            prompt_logprobs: vec![
                vec![RankedToken {
                    token_id: 3,
                    logprob: -2.0,
                    rank: 0,
                }],
                Vec::new(),
            ],
        };
        let decoded = LogprobBlob::decode(&blob.encode()).expect("decode");
        assert_eq!(decoded, blob);
    }

    #[test]
    fn empty_blob_round_trips_and_reports_empty() {
        let blob = LogprobBlob::default();
        assert!(blob.is_empty());
        assert_eq!(LogprobBlob::decode(&blob.encode()).expect("decode"), blob);
    }

    #[test]
    fn truncated_blob_is_rejected() {
        let bytes = LogprobBlob {
            sampled_logprob: Some(1.0),
            ..LogprobBlob::default()
        }
        .encode();
        assert!(LogprobBlob::decode(&bytes[..bytes.len() - 1]).is_err());
    }
}
