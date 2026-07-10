use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value;

use crate::tokenizer::Result;

pub(crate) const SPECIAL_TOKEN_MARKER: u8 = 0xff;

#[derive(Debug, Deserialize)]
struct AddedToken {
    id: usize,
    content: String,
    #[serde(default)]
    special: bool,
}

#[derive(Debug, Clone)]
enum DecoderKind {
    Raw,
    ByteLevel,
    ByteFallback { space: char },
    WordPiece { prefix: String },
    Metaspace { replacement: char },
}

pub(crate) fn huggingface_token_bytes(path: &Path) -> Result<Arc<[Vec<u8>]>> {
    let raw = std::fs::read_to_string(path).map_err(|error| {
        tokenizer_error!(
            "failed to read tokenizer vocabulary {}: {error}",
            path.display()
        )
    })?;
    let value: Value = serde_json::from_str(&raw).map_err(|error| {
        tokenizer_error!(
            "failed to parse tokenizer vocabulary {}: {error}",
            path.display()
        )
    })?;
    let kind = decoder_kind(value.get("decoder"));
    let added: Vec<AddedToken> = serde_json::from_value(
        value
            .get("added_tokens")
            .cloned()
            .unwrap_or_else(|| Value::Array(Vec::new())),
    )
    .map_err(|error| tokenizer_error!("invalid added-token vocabulary: {error}"))?;
    let vocab = model_vocab_entries(
        value
            .pointer("/model/vocab")
            .ok_or_else(|| tokenizer_error!("tokenizer vocabulary has no model.vocab value"))?,
    )?;

    let max_id = added
        .iter()
        .map(|token| token.id)
        .chain(vocab.iter().map(|(_, id)| *id))
        .max()
        .unwrap_or(0);
    let mut bytes = vec![Vec::new(); max_id.saturating_add(1)];
    for token in added {
        let mut value = token.content.into_bytes();
        if token.special {
            value.insert(0, SPECIAL_TOKEN_MARKER);
        }
        bytes[token.id] = value;
    }
    let byte_level_map = matches!(&kind, DecoderKind::ByteLevel).then(build_byte_level_map);
    for (token, id) in vocab {
        if !bytes[id].is_empty() {
            continue;
        }
        bytes[id] = decode_vocab_token(&token, &kind, byte_level_map.as_ref())?;
    }
    Ok(Arc::from(bytes))
}

fn model_vocab_entries(vocab: &Value) -> Result<Vec<(String, usize)>> {
    let entries = match vocab {
        Value::Object(tokens) => tokens
            .iter()
            .map(|(token, id)| {
                let id = id.as_u64().ok_or_else(|| {
                    tokenizer_error!("model vocabulary ID for {token:?} is not an unsigned integer")
                })?;
                let id = u32::try_from(id).map_err(|_| {
                    tokenizer_error!("model vocabulary ID for {token:?} exceeds u32")
                })?;
                Ok((token.clone(), id as usize))
            })
            .collect::<Result<Vec<_>>>()?,
        Value::Array(tokens) => tokens
            .iter()
            .enumerate()
            .map(|(id, entry)| {
                let pair = entry.as_array().ok_or_else(|| {
                    tokenizer_error!("Unigram vocabulary entry {id} is not a [token, score] pair")
                })?;
                if pair.len() != 2 || !pair[1].is_number() {
                    return Err(tokenizer_error!(
                        "Unigram vocabulary entry {id} is not a [token, score] pair"
                    ));
                }
                let token = pair[0].as_str().ok_or_else(|| {
                    tokenizer_error!("Unigram vocabulary token {id} is not a string")
                })?;
                let id = u32::try_from(id)
                    .map_err(|_| tokenizer_error!("Unigram vocabulary exceeds u32 token IDs"))?;
                Ok((token.to_string(), id as usize))
            })
            .collect::<Result<Vec<_>>>()?,
        _ => {
            return Err(tokenizer_error!(
                "model vocabulary must be a token-to-ID object or Unigram array"
            ));
        }
    };
    let mut ids = HashSet::with_capacity(entries.len());
    if let Some((token, id)) = entries.iter().find(|(_, id)| !ids.insert(*id)) {
        return Err(tokenizer_error!(
            "model vocabulary assigns duplicate token ID {id} at token {token:?}"
        ));
    }
    Ok(entries)
}

fn decoder_kind(decoder: Option<&Value>) -> DecoderKind {
    let Some(decoder) = decoder else {
        return DecoderKind::Raw;
    };
    match decoder.get("type").and_then(Value::as_str) {
        Some("ByteLevel") => DecoderKind::ByteLevel,
        Some("ByteFallback") => DecoderKind::ByteFallback { space: ' ' },
        Some("WordPiece") => DecoderKind::WordPiece {
            prefix: decoder
                .get("prefix")
                .and_then(Value::as_str)
                .unwrap_or("##")
                .to_string(),
        },
        Some("Metaspace") => DecoderKind::Metaspace {
            replacement: decoder
                .get("replacement")
                .and_then(Value::as_str)
                .and_then(|value| value.chars().next())
                .unwrap_or('\u{2581}'),
        },
        Some("Sequence") => sequence_decoder_kind(decoder),
        _ => DecoderKind::Raw,
    }
}

fn sequence_decoder_kind(decoder: &Value) -> DecoderKind {
    let decoders = decoder
        .get("decoders")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    if decoders
        .iter()
        .any(|decoder| decoder.get("type").and_then(Value::as_str) == Some("ByteLevel"))
    {
        return DecoderKind::ByteLevel;
    }
    if decoders
        .iter()
        .any(|decoder| decoder.get("type").and_then(Value::as_str) == Some("ByteFallback"))
    {
        let space = decoders
            .iter()
            .find(|decoder| {
                decoder.get("type").and_then(Value::as_str) == Some("Replace")
                    && decoder.get("content").and_then(Value::as_str) == Some(" ")
            })
            .and_then(|decoder| decoder.pointer("/pattern/String"))
            .and_then(Value::as_str)
            .and_then(|value| value.chars().next())
            .unwrap_or('\u{2581}');
        return DecoderKind::ByteFallback { space };
    }
    decoders
        .iter()
        .map(|decoder| decoder_kind(Some(decoder)))
        .find(|kind| !matches!(kind, DecoderKind::Raw))
        .unwrap_or(DecoderKind::Raw)
}

fn decode_vocab_token(
    token: &str,
    kind: &DecoderKind,
    byte_level_map: Option<&HashMap<char, u8>>,
) -> Result<Vec<u8>> {
    match kind {
        DecoderKind::Raw => Ok(token.as_bytes().to_vec()),
        DecoderKind::ByteLevel => token
            .chars()
            .map(|character| {
                byte_level_map
                    .and_then(|mapping| mapping.get(&character).copied())
                    .ok_or_else(|| {
                        tokenizer_error!(
                            "byte-level vocabulary contains unmapped character {character:?}"
                        )
                    })
            })
            .collect(),
        DecoderKind::ByteFallback { space } => {
            if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
                return u8::from_str_radix(&token[3..5], 16)
                    .map(|value| vec![value])
                    .map_err(|error| tokenizer_error!("invalid byte-fallback token: {error}"));
            }
            Ok(token.replace(*space, " ").into_bytes())
        }
        DecoderKind::WordPiece { prefix } => Ok(token
            .strip_prefix(prefix)
            .unwrap_or(token)
            .as_bytes()
            .to_vec()),
        DecoderKind::Metaspace { replacement } => Ok(token.replace(*replacement, " ").into_bytes()),
    }
}

fn build_byte_level_map() -> HashMap<char, u8> {
    fn self_mapped(character: char) -> bool {
        matches!(character, '!'..='~' | '\u{00a1}'..='\u{00ac}' | '\u{00ae}'..='\u{00ff}')
    }

    let mut mapping = HashMap::new();
    let mut next = 0x100u32;
    for byte in 0..=u8::MAX {
        let direct = byte as char;
        let character = if self_mapped(direct) {
            direct
        } else {
            let mapped = char::from_u32(next).unwrap_or('\u{fffd}');
            next += 1;
            mapped
        };
        mapping.insert(character, byte);
    }
    mapping
}
