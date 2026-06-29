pub mod reasoning;
pub mod tool;

use std::collections::HashMap;
use std::convert::Infallible;
use std::fmt;
use std::str::FromStr;

use serde_with::{DeserializeFromStr, SerializeDisplay};

/// Specify which reasoning or tool-call parser implementation to use.
#[derive(Debug, Clone, PartialEq, Eq, Default, DeserializeFromStr, SerializeDisplay)]
pub enum ParserSelection {
 /// Use model-based auto-detection.
    #[default]
    Auto,
 /// Disable the parser entirely.
    None,
 /// Force one specific parser implementation by name.
    Explicit(String),
}

impl ParserSelection {
    pub const AUTO_LITERAL: &str = "auto";
    pub const NONE_LITERAL: &str = "none";
}

impl FromStr for ParserSelection {
    type Err = Infallible;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Ok(if value.eq_ignore_ascii_case(Self::AUTO_LITERAL) {
            Self::Auto
        } else if value.eq_ignore_ascii_case(Self::NONE_LITERAL) {
            Self::None
        } else {
            Self::Explicit(value.to_owned())
        })
    }
}

impl fmt::Display for ParserSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str(Self::AUTO_LITERAL),
            Self::None => f.write_str(Self::NONE_LITERAL),
            Self::Explicit(name) => f.write_str(name),
        }
    }
}

/// Registry and model matcher for reasoning and tool parsers.
#[derive(Clone)]
pub struct ParserFactory<C> {
    creators: HashMap<String, C>,
    patterns: Vec<(String, String)>,
}

impl<C> Default for ParserFactory<C> {
    fn default() -> Self {
        Self {
            creators: HashMap::new(),
            patterns: Vec::new(),
        }
    }
}

impl<C> ParserFactory<C> {
 /// Register a creator for a parser by an exact name.
    pub fn register_creator(&mut self, name: &str, creator: C) -> &mut Self {
        self.creators.insert(name.to_string(), creator);
        self
    }

 /// Add a case-insensitive substring match from model ID to parser name.

 /// Matching is **first-match-wins over registration order**: when several
 /// patterns are substrings of the same model ID (e.g. both `qwen` and
    /// `qwen3-coder` match `qwen3-coder-480b`), [`resolve_name_for_model`] returns
 /// the parser of the pattern that was registered *first*. Callers must
 /// therefore register more-specific patterns before broader ones (register
 /// `qwen3-coder` before `qwen`, `deepseek-v3.2`/`deepseek-v3.1` before
 /// `deepseek-v3`, `glm-4.7` before `glm-4.6`/`glm-4.5`, etc.). This ordering
 /// is load-bearing; reordering the registration sites can silently change
 /// which parser a model resolves to.

    /// [`resolve_name_for_model`]: Self::resolve_name_for_model
    pub fn register_pattern(&mut self, pattern: &str, parser_name: &str) -> &mut Self {
        self.patterns
            .push((pattern.to_lowercase(), parser_name.to_string()));
        self
    }

 /// Return the parser name of the first registered pattern that is a
 /// case-insensitive substring of `model_id`, or `None` if none match.

 /// See [`register_pattern`](Self::register_pattern) for the order-sensitive,
 /// first-match-wins contract that governs overlapping patterns.
    pub fn resolve_name_for_model(&self, model_id: &str) -> Option<&str> {
        let model_lower = model_id.to_lowercase();
        self.patterns
            .iter()
            .find(|(pattern, _)| model_lower.contains(pattern))
            .map(|(_, parser_name)| parser_name.as_str())
    }

 /// Return true if the exact parser name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.creators.contains_key(name)
    }

 /// Return all registered parser names sorted for stable display.
    pub fn list(&self) -> Vec<String> {
        let mut names: Vec<_> = self.creators.keys().cloned().collect();
        names.sort_unstable();
        names
    }

 /// Get the constructor for a parser by its exact registered name, or return
 /// None if not found.
    pub fn creator(&self, name: &str) -> Option<&C> {
        self.creators.get(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

 /// Pins the documented first-match-wins, order-sensitive, case-insensitive
    /// substring contract of `register_pattern` / `resolve_name_for_model` so a
 /// future reordering of the registration sites cannot silently change which
 /// parser a model resolves to.
    #[test]
    fn resolve_name_is_first_match_wins_and_order_sensitive() {
 // Specific-before-broad: the first registered overlapping pattern wins.
        let mut specific_first = ParserFactory::<()>::default();
        specific_first
            .register_pattern("qwen3-coder", "qwen3_coder")
            .register_pattern("qwen", "qwen3_xml");
        assert_eq!(
            specific_first.resolve_name_for_model("Qwen/Qwen3-Coder-480B"),
            Some("qwen3_coder"),
        );

 // Reversing the order flips the resolved parser for the same model ID,
 // demonstrating the ordering is load-bearing.
        let mut broad_first = ParserFactory::<()>::default();
        broad_first
            .register_pattern("qwen", "qwen3_xml")
            .register_pattern("qwen3-coder", "qwen3_coder");
        assert_eq!(
            broad_first.resolve_name_for_model("Qwen/Qwen3-Coder-480B"),
            Some("qwen3_xml"),
        );
    }

    #[test]
    fn resolve_name_returns_none_when_no_pattern_matches() {
        let mut factory = ParserFactory::<()>::default();
        factory.register_pattern("deepseek-v3", "deepseek_v3");
        assert_eq!(factory.resolve_name_for_model("gpt-4o"), None);
    }
}
