use crate::profile::omni::{DelimitedTextPolicy, OutputFilterPolicy};
use crate::profile::reasoning::DelimitedReasoningParser;
use crate::serving::text::tokenizer::DynTokenizer;

pub(crate) struct SenseNovaOutputProcessor {
    reasoning: Option<DelimitedReasoningParser>,
    visible_wrappers: VisibleWrapperFilter,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct SenseNovaTextDelta {
    pub(crate) visible: String,
    pub(crate) reasoning: String,
}

impl SenseNovaOutputProcessor {
    pub(crate) fn new(
        spec: OutputFilterPolicy,
        tokenizer: DynTokenizer,
        prompt_token_ids: &[u32],
    ) -> crate::profile::reasoning::Result<Self> {
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
            visible_wrappers: VisibleWrapperFilter::new(spec.visible_wrappers),
        })
    }

    pub(crate) fn push(&mut self, text: &str) -> SenseNovaTextDelta {
        let (content, reasoning) = if let Some(parser) = self.reasoning.as_mut() {
            let delta = parser.push(text);
            (
                delta.content.unwrap_or_default(),
                delta.reasoning.unwrap_or_default(),
            )
        } else {
            (text.to_string(), String::new())
        };
        SenseNovaTextDelta {
            visible: self.visible_wrappers.push(&content),
            reasoning,
        }
    }
}

struct VisibleWrapperFilter {
    wrappers: Vec<DelimitedTextPolicy>,
    pending: String,
}

impl VisibleWrapperFilter {
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
