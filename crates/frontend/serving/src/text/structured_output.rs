use serde::Deserialize;
use uniserve_core::GrammarSpec;
use uniserve_model_profile::tokenizer::Tokenizer;
use xgrammar::{Grammar, GrammarCompiler, StructuralTagItem, TokenizerInfo};

use crate::StructuredOutputIntent;
use crate::text::error::{Error, Result};

#[derive(Debug, Deserialize)]
struct StructuralTagCollection {
    structures: Vec<StructuralTagCollectionItem>,
    triggers: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct StructuralTagCollectionItem {
    begin: String,
    schema: serde_json::Value,
    end: String,
}

pub(crate) fn compile_structured_output(
    intent: Option<&StructuredOutputIntent>,
    tokenizer: &dyn Tokenizer,
    stop_token_ids: &[u32],
) -> Result<Option<GrammarSpec>> {
    let Some(intent) = intent else {
        return Ok(None);
    };
    if let StructuredOutputIntent::Choice(choices) = intent {
        if choices.is_empty() {
            return Err(Error::StructuredOutput(
                "choice requires at least one alternative".to_string(),
            ));
        }
        let token_sequences = choices
            .iter()
            .map(|choice| tokenizer.encode(choice, false).map_err(Error::from))
            .collect::<Result<Vec<_>>>()?;
        if token_sequences.iter().any(Vec::is_empty) {
            return Err(Error::StructuredOutput(
                "choice alternatives must tokenize to at least one token".to_string(),
            ));
        }
        return Ok(Some(GrammarSpec::Choice { token_sequences }));
    }
    let whitespace_pattern = match intent {
        StructuredOutputIntent::JsonObject {
            whitespace_pattern, ..
        }
        | StructuredOutputIntent::JsonSchema {
            whitespace_pattern, ..
        } => whitespace_pattern.as_ref(),
        _ => None,
    };
    if whitespace_pattern.is_some() {
        return Err(Error::StructuredOutput(
            "custom JSON whitespace patterns are not supported by the configured grammar compiler"
                .to_string(),
        ));
    }

    let token_bytes = tokenizer.grammar_token_bytes()?;
    let stop_ids = stop_token_ids
        .iter()
        .copied()
        .map(|id| {
            i32::try_from(id).map_err(|_| {
                Error::StructuredOutput(format!("stop token id {id} exceeds the grammar id space"))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let metadata = serde_json::json!({
        "vocab_type": 0,
        "vocab_size": token_bytes.len(),
        "add_prefix_space": false,
        "stop_token_ids": stop_ids,
    })
    .to_string();
    let tokenizer_info =
        TokenizerInfo::from_vocab_and_metadata_bytes(token_bytes.iter(), &metadata);
    let mut compiler = GrammarCompiler::new(&tokenizer_info, 1, true, 64 * 1024 * 1024)
        .map_err(Error::StructuredOutput)?;
    let compiled = if let StructuredOutputIntent::JsonSchema {
        schema,
        disable_any_whitespace,
        disable_additional_properties,
        ..
    } = intent
    {
        let schema = match schema {
            serde_json::Value::String(schema) => schema.clone(),
            value => value.to_string(),
        };
        compiler
            .compile_json_schema(
                &schema,
                !disable_any_whitespace,
                None,
                None::<(&str, &str)>,
                *disable_additional_properties,
                None,
            )
            .map_err(Error::StructuredOutput)?
    } else if let StructuredOutputIntent::JsonObject {
        disable_any_whitespace,
        ..
    } = intent
    {
        compiler
            .compile_json_schema(
                r#"{"type":"object"}"#,
                !disable_any_whitespace,
                None,
                None::<(&str, &str)>,
                false,
                None,
            )
            .map_err(Error::StructuredOutput)?
    } else if let StructuredOutputIntent::Regex(regex) = intent {
        compiler
            .compile_regex(regex)
            .map_err(Error::StructuredOutput)?
    } else if let StructuredOutputIntent::Grammar(grammar) = intent {
        let root = grammar
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty() && !line.starts_with('#'))
            .and_then(|line| line.split_once("::="))
            .map(|(name, _)| name.trim())
            .filter(|name| !name.is_empty())
            .ok_or_else(|| {
                Error::StructuredOutput(
                    "grammar must be EBNF with a named `::=` root rule".to_string(),
                )
            })?;
        compiler
            .compile_grammar_from_ebnf(grammar, root)
            .map_err(Error::StructuredOutput)?
    } else if let StructuredOutputIntent::StructuralTag(structural_tag) = intent {
        let value: serde_json::Value = serde_json::from_str(structural_tag).map_err(|error| {
            Error::StructuredOutput(format!("structural tag is not valid JSON: {error}"))
        })?;
        if value.get("format").is_some() {
            let grammar =
                Grammar::from_structural_tag(structural_tag).map_err(Error::StructuredOutput)?;
            compiler
                .compile_grammar(&grammar)
                .map_err(Error::StructuredOutput)?
        } else {
            let collection: StructuralTagCollection =
                serde_json::from_value(value).map_err(|error| {
                    Error::StructuredOutput(format!(
                        "structural tag must contain `structures` and `triggers`: {error}"
                    ))
                })?;
            if collection.structures.is_empty() || collection.triggers.is_empty() {
                return Err(Error::StructuredOutput(
                    "structural tag requires non-empty structures and triggers".to_string(),
                ));
            }
            let tags = collection
                .structures
                .iter()
                .map(|item| StructuralTagItem::new(&item.begin, item.schema.to_string(), &item.end))
                .collect::<Vec<_>>();
            compiler
                .compile_structural_tag(&tags, &collection.triggers)
                .map_err(Error::StructuredOutput)?
        }
    } else {
        unreachable!("choice constraints return before grammar compilation")
    };

    Ok(Some(GrammarSpec::Compiled {
        token_bytes: token_bytes.iter().cloned().collect(),
        compiled_grammar_json: compiled.serialize_json(),
        stop_token_ids: stop_token_ids.to_vec(),
    }))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use uniserve_model_profile::tokenizer::{Result as TokenizerResult, Tokenizer};

    struct AsciiTokenizer;

    impl Tokenizer for AsciiTokenizer {
        fn encode(&self, text: &str, _add_special_tokens: bool) -> TokenizerResult<Vec<u32>> {
            Ok(text.bytes().map(u32::from).collect())
        }

        fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> TokenizerResult<String> {
            Ok(token_ids
                .iter()
                .filter_map(|id| u8::try_from(*id).ok())
                .map(char::from)
                .collect())
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            (token.len() == 1).then(|| u32::from(token.as_bytes()[0]))
        }

        fn grammar_token_bytes(&self) -> TokenizerResult<Arc<[Vec<u8>]>> {
            let mut vocabulary = (0_u8..=126).map(|byte| vec![byte]).collect::<Vec<_>>();
            vocabulary.push(vec![0xff]);
            Ok(vocabulary.into())
        }
    }

    #[test]
    fn compiles_choice_and_regex_constraints() {
        let choice = StructuredOutputIntent::Choice(vec!["yes".to_string(), "no".to_string()]);
        assert!(matches!(
            compile_structured_output(Some(&choice), &AsciiTokenizer, &[127]).unwrap(),
            Some(GrammarSpec::Choice { token_sequences }) if token_sequences == vec![vec![121, 101, 115], vec![110, 111]]
        ));

        let regex = StructuredOutputIntent::Regex("a(b|c)".to_string());
        assert!(matches!(
            compile_structured_output(Some(&regex), &AsciiTokenizer, &[127]).unwrap(),
            Some(GrammarSpec::Compiled { compiled_grammar_json, .. }) if !compiled_grammar_json.is_empty()
        ));

        let structural = StructuredOutputIntent::StructuralTag(
            serde_json::json!({
                "type": "structural_tag",
                "format": {"type": "const_string", "value": "ok"}
            })
            .to_string(),
        );
        assert!(matches!(
            compile_structured_output(Some(&structural), &AsciiTokenizer, &[127]).unwrap(),
            Some(GrammarSpec::Compiled { compiled_grammar_json, .. }) if !compiled_grammar_json.is_empty()
        ));
    }

    #[test]
    fn rejects_unsupported_custom_json_whitespace() {
        let whitespace = StructuredOutputIntent::JsonObject {
            disable_any_whitespace: false,
            whitespace_pattern: Some("[ ]?".to_string()),
        };
        assert!(compile_structured_output(Some(&whitespace), &AsciiTokenizer, &[127]).is_err());
    }
}
