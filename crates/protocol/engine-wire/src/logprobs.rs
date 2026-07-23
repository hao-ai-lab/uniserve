mod array;
#[cfg(test)]
mod tests;
mod wire;

use std::ops::{Deref, DerefMut};

use enum_as_inner::EnumAsInner;
use serde::{Deserialize, Deserializer, Serialize};

use self::wire::*;
use super::{EngineCoreOutput, EngineCoreOutputs, decode_msgpack};
use crate::error::{Error, Result, bail_ext_value_decode, ext_value_decode};
use crate::tensor::{WireArrayData, WireNdArray};

/// One token candidate and its logprob metadata for a single sequence position.
///
/// The first entry in a [`PositionLogprobs`] is always the sampled/selected
/// token for that position. Any remaining entries follow the engine's returned
/// top-k candidate order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenLogprob {
    pub token_id: u32,
    pub logprob: f32,
    /// The sampled/selected token uses its actual vocab rank. Remaining entries
    /// use 1-based top-k ranks matching the engine's returned candidate
    /// order.
    pub rank: u32,
}

/// Logprob payload for one sequence position.
///
/// This is the semantic Rust representation used by the public client API after
/// the lower-level ndarray/tensor wire payload has been decoded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PositionLogprobs {
    pub entries: Vec<TokenLogprob>,
}

impl PositionLogprobs {
    /// Convert one decoded logprobs row into this per-position form by grouping
    /// each token/logprob pair together with the sampled/selected token's
    /// actual vocab rank.
    ///
    /// `token_ids` is still in raw signed form: the engine right-pads ragged
    /// top-k rows with a `-1` sentinel (see [`array::TOKEN_ID_PADDING_SENTINEL`])
    /// in the trailing columns. Those padding slots carry no real token and are
    /// skipped here rather than decoded into bogus [`TokenLogprob`] entries. The
    /// sampled/selected token in column 0 is never padded.
    fn from_decoded_row(token_ids: &[i64], logprobs: &[f32], sampled_rank: u32) -> Result<Self> {
        if token_ids.len() != logprobs.len() {
            bail_ext_value_decode!(
                "logprobs row length mismatch: token_ids={}, logprobs={}",
                token_ids.len(),
                logprobs.len()
            );
        }
        if sampled_rank == 0 {
            bail_ext_value_decode!("token_ranks must be >= 1 for decoded engine logprobs");
        }

        let mut entries = Vec::with_capacity(token_ids.len());
        for (index, (&token_id, &logprob)) in token_ids.iter().zip(logprobs.iter()).enumerate() {
            let Some(token_id) = array::token_id_from_i64(token_id, "logprobs row")? else {
                // Padding sentinel: trailing ragged-row slot with no real token.
                continue;
            };
            let rank = if index == 0 {
                sampled_rank
            } else {
                index as u32
            };
            entries.push(TokenLogprob {
                token_id,
                logprob,
                rank,
            });
        }
        Ok(Self { entries })
    }
}

/// Decoded per-request logprobs payload for one engine output.
///
/// Unlike the Python wire payload, this public Rust type is already fully
/// semantic: one [`PositionLogprobs`] per scored position, each containing the
/// sampled/selected token plus any returned top-k alternatives for that same
/// position.
///
/// The Python engine still sends logprobs as ndarray/tensor-shaped wire tuples.
/// Rust resolves that lower-level representation during decode and exposes only
/// this per-position form to callers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Logprobs {
    /// One decoded logprobs record per scored position in this engine
    /// output.
    pub positions: Vec<PositionLogprobs>,
}

impl Logprobs {
    /// Returns the number of scored positions in this payload.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Returns whether the payload contains no scored positions.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

/// Output field wrapper that is initially deserialized from the Python wire
/// shape, then resolved into [`Logprobs`] before the decoded message is
/// returned to callers.
#[derive(Clone, PartialEq, Debug, EnumAsInner)]
pub enum MaybeWireLogprobs {
    /// The logprobs are still in the wire format and need to be resolved by
    /// looking up aux frames and decoding raw views. Should only be used
    /// internally during deserialization.
    Wire(Box<WireLogprobs>),
    /// The actual decoded logprobs value,
    Direct(Logprobs),
}

impl Deref for MaybeWireLogprobs {
    type Target = Logprobs;

    /// Dereferences to the decoded [`Logprobs`].
    ///
    /// # Panics
    ///
    /// Panics if the value is still in [`MaybeWireLogprobs::Wire`] form (i.e.
    /// [`resolve`](MaybeWireLogprobs::resolve) has not yet run). The `Deref`
    /// trait cannot signal this fallibly, so callers that may hold an
    /// unresolved value must use the [`as_direct`](Self::as_direct) /
    /// [`into_direct`](Self::into_direct) accessors instead.
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Wire(_) => panic!(
                "MaybeWireLogprobs dereferenced while still in wire format; \
                 resolve() must run first, or use as_direct()/into_direct()"
            ),
            Self::Direct(value) => value,
        }
    }
}

impl DerefMut for MaybeWireLogprobs {
    /// Mutably dereferences to the decoded [`Logprobs`].
    ///
    /// # Panics
    ///
    /// Panics if the value is still in [`MaybeWireLogprobs::Wire`] form (i.e.
    /// [`resolve`](MaybeWireLogprobs::resolve) has not yet run). The `DerefMut`
    /// trait cannot signal this fallibly, so callers that may hold an
    /// unresolved value must use the [`as_direct_mut`](Self::as_direct_mut)
    /// accessor instead.
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Wire(_) => panic!(
                "MaybeWireLogprobs mutably dereferenced while still in wire format; \
                 resolve() must run first, or use as_direct_mut()"
            ),
            Self::Direct(value) => value,
        }
    }
}

impl<'de> Deserialize<'de> for MaybeWireLogprobs {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // When deserializing, it's always in the wire form.
        WireLogprobs::deserialize(deserializer).map(|v| Self::Wire(Box::new(v)))
    }
}

impl Serialize for MaybeWireLogprobs {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // For testing purposes only. We don't actually serialize it into aux frames.
        match self {
            Self::Wire(value) => value.serialize(serializer),
            Self::Direct(value) => WireLogprobs::from_direct(value)
                .map_err(serde::ser::Error::custom)?
                .serialize(serializer),
        }
    }
}

impl MaybeWireLogprobs {
    /// Resolve the wire representation into decoded logprobs by looking up aux
    /// frames and decoding raw views as needed.
    fn resolve<Frame>(self, frames: &[Frame], field_prefix: &str) -> Result<Self>
    where
        Frame: AsRef<[u8]>,
    {
        match self {
            Self::Direct(value) => Ok(Self::Direct(value)),
            Self::Wire(value) => value.resolve(frames, field_prefix).map(Self::Direct),
        }
    }
}

impl EngineCoreOutputs {
    /// Resolve all wire-format fields in-place by looking up aux frames and
    /// decoding raw-view payloads as needed.
    fn resolve_in_place<Frame>(&mut self, frames: &[Frame]) -> Result<()>
    where
        Frame: AsRef<[u8]>,
    {
        for output in &mut self.outputs {
            output.resolve_in_place(frames)?;
        }
        Ok(())
    }
}

impl EngineCoreOutput {
    /// Resolve all wire-format fields in-place by looking up aux frames and
    /// decoding raw-view payloads as needed.
    fn resolve_in_place<Frame>(&mut self, frames: &[Frame]) -> Result<()>
    where
        Frame: AsRef<[u8]>,
    {
        self.new_logprobs = (self.new_logprobs.take())
            .map(|value| value.resolve(frames, "new_logprobs"))
            .transpose()?;
        self.new_prompt_logprobs_tensors = (self.new_prompt_logprobs_tensors.take())
            .map(|value| value.resolve(frames, "new_prompt_logprobs_tensors"))
            .transpose()?;
        Ok(())
    }
}

impl WireLogprobs {
    /// Convert semantic per-position logprobs into the Python wire tuple shape.
    ///
    /// This exists mainly so Rust-side tests can inject semantic logprobs into
    /// mocked engine outputs without manually building ndarray
    /// raw-view tuples.
    fn from_direct(value: &Logprobs) -> std::result::Result<Self, String> {
        let rows = value.positions.len();
        let cols = value
            .positions
            .first()
            .map(|position| position.entries.len())
            .unwrap_or(0);

        let mut token_ids = Vec::with_capacity(rows.saturating_mul(cols).saturating_mul(8));
        let mut logprobs = Vec::with_capacity(rows.saturating_mul(cols).saturating_mul(4));
        let mut token_ranks = Vec::with_capacity(rows.saturating_mul(8));

        for (row_index, position) in value.positions.iter().enumerate() {
            if position.entries.len() != cols {
                return Err(format!(
                    "logprobs row {row_index} length mismatch: expected {cols}, got {}",
                    position.entries.len()
                ));
            }
            let Some((sampled, _)) = position.entries.split_first() else {
                return Err(format!("logprobs row {row_index} is empty"));
            };

            token_ranks.extend_from_slice(&(sampled.rank as i64).to_le_bytes());
            for entry in &position.entries {
                token_ids.extend_from_slice(&(entry.token_id as i64).to_le_bytes());
                logprobs.extend_from_slice(&entry.logprob.to_le_bytes());
            }
        }

        Ok(Self {
            logprob_token_ids: WireNdArray {
                dtype: "<i8".to_string(),
                shape: vec![rows, cols],
                data: WireArrayData::RawView(token_ids),
            },
            logprobs: WireNdArray {
                dtype: "<f4".to_string(),
                shape: vec![rows, cols],
                data: WireArrayData::RawView(logprobs),
            },
            token_ranks: WireNdArray {
                dtype: "<i8".to_string(),
                shape: vec![rows],
                data: WireArrayData::RawView(token_ranks),
            },
        })
    }

    /// Resolve the wire-format logprobs into semantic [`Logprobs`] records by
    /// looking up aux frames, decoding raw views, and grouping each row
    /// into one [`PositionLogprobs`].
    fn resolve<Frame>(self, frames: &[Frame], field_prefix: &str) -> Result<Logprobs>
    where
        Frame: AsRef<[u8]>,
    {
        let token_ids = array::decode_array2_token_ids(
            self.logprob_token_ids,
            &format!("{field_prefix}.logprob_token_ids"),
            frames,
        )?;
        let logprobs =
            array::decode_array2_f32(self.logprobs, &format!("{field_prefix}.logprobs"), frames)?;
        let token_ranks = array::decode_array1_u32(
            self.token_ranks,
            &format!("{field_prefix}.token_ranks"),
            frames,
        )?;

        if token_ids.rows != logprobs.rows || token_ids.cols != logprobs.cols {
            bail_ext_value_decode!(
                "{field_prefix}: row shape mismatch between token ids ({}, {}) and logprobs ({}, {})",
                token_ids.rows,
                token_ids.cols,
                logprobs.rows,
                logprobs.cols
            );
        }
        if token_ids.rows != token_ranks.len() {
            bail_ext_value_decode!(
                "{field_prefix}: token_ranks length {} does not match row count {}",
                token_ranks.len(),
                token_ids.rows
            );
        }

        let mut positions = Vec::with_capacity(token_ids.rows);
        for ((token_ids_row, logprobs_row), sampled_rank) in token_ids
            .data
            .chunks(token_ids.cols)
            .zip(logprobs.data.chunks(logprobs.cols))
            .zip(token_ranks)
        {
            positions.push(PositionLogprobs::from_decoded_row(
                token_ids_row,
                logprobs_row,
                sampled_rank,
            )?);
        }

        Ok(Logprobs { positions })
    }
}

/// Decode one ordinary or multipart engine output message into the strong
/// typed public protocol shape.
pub fn decode_engine_outputs<Frame>(frames: &[Frame]) -> Result<EngineCoreOutputs>
where
    Frame: AsRef<[u8]>,
{
    let first_frame = frames
        .first()
        .ok_or_else(|| ext_value_decode!("missing output frame"))?;

    let mut outputs: EngineCoreOutputs = decode_msgpack(first_frame.as_ref())?;
    outputs.resolve_in_place(frames)?;
    Ok(outputs)
}
