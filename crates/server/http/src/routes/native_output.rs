use uniserve_native_api::{NativeDelimitedText, NativeOutputFilter};
use uniserve_reasoning_parser::DelimitedReasoningParser;
use uniserve_text::tokenizer::DynTokenizer;

pub(crate) struct NativeTextOutputFilter {
    reasoning: Option<DelimitedReasoningParser>,
    visible_wrappers: NativeVisibleWrapperFilter,
}

impl NativeTextOutputFilter {
    pub(crate) fn new(
        spec: NativeOutputFilter,
        tokenizer: DynTokenizer,
        prompt_token_ids: &[u32],
    ) -> uniserve_reasoning_parser::Result<Self> {
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
            visible_wrappers: NativeVisibleWrapperFilter::new(spec.visible_wrappers),
        })
    }

    pub(crate) fn push(&mut self, text: &str) -> String {
        let content = if let Some(reasoning) = self.reasoning.as_mut() {
            reasoning.push(text).content.unwrap_or_default()
        } else {
            text.to_string()
        };
        self.visible_wrappers.push(&content)
    }
}

struct NativeVisibleWrapperFilter {
    wrappers: Vec<NativeDelimitedText>,
    pending: String,
}

impl NativeVisibleWrapperFilter {
    fn new(wrappers: Vec<NativeDelimitedText>) -> Self {
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

    use uniserve_native_api::{NativeDelimitedText, NativeOutputFilter};
    use uniserve_text::tokenizer::{DynTokenizer, Tokenizer};

    use super::NativeTextOutputFilter;

    #[derive(Debug)]
    struct FilterTokenizer;

    impl Tokenizer for FilterTokenizer {
        fn encode(
            &self,
            text: &str,
            _add_special_tokens: bool,
        ) -> uniserve_text::tokenizer::Result<Vec<u32>> {
            Ok(text.bytes().map(u32::from).collect())
        }

        fn decode(
            &self,
            token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> uniserve_text::tokenizer::Result<String> {
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

    fn filter() -> NativeTextOutputFilter {
        NativeTextOutputFilter::new(
            NativeOutputFilter {
                reasoning: Some(NativeDelimitedText {
                    start: "<think>".into(),
                    end: "</think>".into(),
                }),
                visible_wrappers: vec![NativeDelimitedText {
                    start: "<answer>".into(),
                    end: "</answer>".into(),
                }],
            },
            tokenizer(),
            &[],
        )
        .unwrap()
    }

    #[test]
    fn native_reasoning_tag_filter_strips_split_think_delimiters() {
        let mut filter = filter();
        let mut out = String::new();

        for chunk in ["Here ", "<thi", "nk>secret", "</thi", "nk> guide"] {
            out.push_str(&filter.push(chunk));
        }

        assert_eq!(out, "Here  guide");
    }

    #[test]
    fn native_reasoning_tag_filter_preserves_text_around_multiple_blocks() {
        let mut filter = filter();
        let chunks = [
            "Sonoma <think>plan</think>",
            " Sequoia",
            " <think>more</think>Tahoe",
            " and Golden Gate.",
        ];
        let out = chunks
            .into_iter()
            .map(|chunk| filter.push(chunk))
            .collect::<String>();

        assert_eq!(out, "Sonoma  Sequoia Tahoe and Golden Gate.");
    }

    #[test]
    fn native_reasoning_tag_filter_hides_unclosed_reasoning_until_close() {
        let mut filter = filter();

        assert_eq!(filter.push("<think>draft plan"), "");
        assert_eq!(filter.push(" and more</thi"), "");
        assert_eq!(filter.push("nk>Final answer."), "Final answer.");
    }

    #[test]
    fn native_reasoning_tag_filter_strips_answer_wrappers() {
        let mut filter = filter();
        let chunks = ["<ans", "wer>Sonoma guide", "</ans", "wer>"];
        let out = chunks
            .into_iter()
            .map(|chunk| filter.push(chunk))
            .collect::<String>();

        assert_eq!(out, "Sonoma guide");
    }
}
