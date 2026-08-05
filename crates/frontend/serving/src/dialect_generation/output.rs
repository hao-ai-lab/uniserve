use crate::text::tokenizer::DynTokenizer;
use uniserve_model_profile::dialect::{DelimitedTextPolicy, OutputFilterPolicy};
use uniserve_model_profile::reasoning::DelimitedReasoningParser;

pub(crate) struct DialectOutputProcessor {
    reasoning: Option<DelimitedReasoningParser>,
    visible_wrappers: DialectVisibleWrapperFilter,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct DialectTextDelta {
    pub(crate) visible: String,
    pub(crate) reasoning: String,
}

impl DialectOutputProcessor {
    pub(crate) fn new(
        mut spec: OutputFilterPolicy,
        tokenizer: DynTokenizer,
        prompt_token_ids: &[u32],
        profile_reasoning: bool,
    ) -> uniserve_model_profile::reasoning::Result<Self> {
        if !profile_reasoning {
            spec.reasoning = None;
        }
        for wrapper in &spec.visible_wrappers {
            if wrapper.start.is_empty() {
                return Err(
                    uniserve_model_profile::reasoning::ReasoningError::EmptyDelimiter {
                        field: "visible wrapper start",
                    },
                );
            }
            if wrapper.end.is_empty() {
                return Err(
                    uniserve_model_profile::reasoning::ReasoningError::EmptyDelimiter {
                        field: "visible wrapper end",
                    },
                );
            }
        }
        let reasoning = if let Some(reasoning) = spec.reasoning.clone() {
            let mut parser =
                DelimitedReasoningParser::new(tokenizer, reasoning.start, reasoning.end, false)?;
            parser.initialize(prompt_token_ids);
            Some(parser)
        } else {
            None
        };
        Ok(Self {
            reasoning,
            visible_wrappers: DialectVisibleWrapperFilter::new(spec.visible_wrappers),
        })
    }

    pub(crate) fn push(&mut self, text: &str) -> DialectTextDelta {
        let (content, reasoning) = if let Some(parser) = self.reasoning.as_mut() {
            let delta = parser.push(text);
            (
                delta.content.unwrap_or_default(),
                delta.reasoning.unwrap_or_default(),
            )
        } else {
            (text.to_string(), String::new())
        };
        DialectTextDelta {
            visible: self.visible_wrappers.push(&content),
            reasoning,
        }
    }
}

struct DialectVisibleWrapperFilter {
    wrappers: Vec<DelimitedTextPolicy>,
    pending: String,
}

impl DialectVisibleWrapperFilter {
    fn new(wrappers: Vec<DelimitedTextPolicy>) -> Self {
        Self {
            wrappers,
            pending: String::new(),
        }
    }

    fn push(&mut self, text: &str) -> String {
        if self.wrappers.is_empty() {
            return text.to_string();
        }
        self.pending.push_str(text);
        let mut visible = String::new();

        loop {
            if let Some((start, marker_len)) = self.first_marker() {
                visible.push_str(&self.pending[..start]);
                self.pending.drain(..start + marker_len);
                continue;
            }

            let keep = trailing_marker_prefix_len_any(&self.pending, &self.markers());
            let emit_to = self.pending.len().saturating_sub(keep);
            visible.push_str(&self.pending[..emit_to]);
            self.pending.drain(..emit_to);
            break;
        }

        visible
    }

    fn markers(&self) -> Vec<&str> {
        let mut markers = Vec::new();
        for wrapper in &self.wrappers {
            markers.push(wrapper.start.as_str());
            markers.push(wrapper.end.as_str());
        }
        markers
    }

    fn first_marker(&self) -> Option<(usize, usize)> {
        let mut matches = Vec::new();
        for wrapper in &self.wrappers {
            matches.push(wrapper.start.as_str());
            matches.push(wrapper.end.as_str());
        }
        matches
            .into_iter()
            .filter_map(|marker| self.pending.find(marker).map(|idx| (idx, marker.len())))
            .min_by_key(|(idx, _)| *idx)
    }
}

fn trailing_marker_prefix_len_any(text: &str, markers: &[&str]) -> usize {
    markers
        .iter()
        .map(|marker| trailing_marker_prefix_len(text, marker))
        .max()
        .unwrap_or(0)
}

fn trailing_marker_prefix_len(text: &str, marker: &str) -> usize {
    let max = text.len().min(marker.len().saturating_sub(1));
    for len in (1..=max).rev() {
        if text.is_char_boundary(text.len() - len)
            && marker.is_char_boundary(len)
            && text[text.len() - len..].eq(&marker[..len])
        {
            return len;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::text::tokenizer::{DynTokenizer, Tokenizer};
    use uniserve_model_profile::dialect::{DelimitedTextPolicy, OutputFilterPolicy};

    use super::DialectOutputProcessor;

    #[derive(Debug)]
    struct FilterTokenizer;

    impl Tokenizer for FilterTokenizer {
        fn encode(
            &self,
            text: &str,
            _add_special_tokens: bool,
        ) -> crate::text::tokenizer::Result<Vec<u32>> {
            Ok(text.bytes().map(u32::from).collect())
        }

        fn decode(
            &self,
            token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> crate::text::tokenizer::Result<String> {
            Ok(
                String::from_utf8_lossy(&token_ids.iter().map(|id| *id as u8).collect::<Vec<_>>())
                    .into_owned(),
            )
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            match token {
                "<think>" => Some(1),
                "</think>" => Some(2),
                _ => None,
            }
        }
    }

    fn tokenizer() -> DynTokenizer {
        Arc::new(FilterTokenizer)
    }

    fn filter() -> DialectOutputProcessor {
        DialectOutputProcessor::new(
            OutputFilterPolicy {
                reasoning: Some(DelimitedTextPolicy {
                    start: "<think>".into(),
                    end: "</think>".into(),
                }),
                visible_wrappers: vec![DelimitedTextPolicy {
                    start: "<answer>".into(),
                    end: "</answer>".into(),
                }],
            },
            tokenizer(),
            &[],
            true,
        )
        .unwrap()
    }

    #[test]
    fn reasoning_tag_filter_strips_split_think_delimiters() {
        let mut filter = filter();
        let mut out = String::new();

        for chunk in ["Here ", "<thi", "nk>secret", "</thi", "nk> guide"] {
            out.push_str(&filter.push(chunk).visible);
        }

        assert_eq!(out, "Here  guide");
    }

    #[test]
    fn reasoning_tag_filter_preserves_text_around_multiple_blocks() {
        let mut filter = filter();
        let chunks = [
            "Sonoma <think>plan</think>",
            " Sequoia",
            " <think>more</think>Tahoe",
            " and Golden Gate.",
        ];
        let out = chunks
            .into_iter()
            .map(|chunk| filter.push(chunk).visible)
            .collect::<String>();

        assert_eq!(out, "Sonoma  Sequoia Tahoe and Golden Gate.");
    }

    #[test]
    fn reasoning_tag_filter_hides_unclosed_reasoning_until_close() {
        let mut filter = filter();

        assert_eq!(filter.push("<think>draft plan").visible, "");
        assert_eq!(filter.push(" and more</thi").visible, "");
        assert_eq!(filter.push("nk>Final answer.").visible, "Final answer.");
    }

    #[test]
    fn reasoning_tag_filter_strips_answer_wrappers() {
        let mut filter = filter();
        let chunks = ["<ans", "wer>Sonoma guide", "</ans", "wer>"];
        let out = chunks
            .into_iter()
            .map(|chunk| filter.push(chunk).visible)
            .collect::<String>();

        assert_eq!(out, "Sonoma guide");
    }

    #[test]
    fn output_filter_rejects_empty_visible_wrapper_delimiter() {
        let result = DialectOutputProcessor::new(
            OutputFilterPolicy {
                reasoning: None,
                visible_wrappers: vec![DelimitedTextPolicy {
                    start: String::new(),
                    end: "</answer>".into(),
                }],
            },
            tokenizer(),
            &[],
            true,
        );

        assert!(matches!(
            result,
            Err(
                uniserve_model_profile::reasoning::ReasoningError::EmptyDelimiter {
                    field: "visible wrapper start"
                }
            )
        ));
    }
}
