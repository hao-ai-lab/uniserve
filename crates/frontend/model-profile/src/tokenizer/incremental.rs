use std::mem::take;

use crate::tokenizer::{Result, Tokenizer};

/// Stateful incremental decoder that emits text chunks one token at a time.
pub trait IncrementalDecoder: Send {
    /// Push one generated token and return how many new string bytes were
    /// added.
    fn push_token(&mut self, token_id: u32) -> Result<usize>;

    /// Consume any text which is currently ready.
    fn next_chunk(&mut self) -> Option<String>;

    /// Flush any remaining buffered text that has not yet been emitted.
    ///
    /// Called after the final generated token to force out buffered/incomplete
    /// fragments.
    fn flush(&mut self, truncate_output_to: Option<usize>) -> Result<(Option<String>, String)>;

    /// Return cumulative decoded text so far.
    fn output(&self) -> &str;
}
/// [`IncrementalDecoder`] built on [`Tokenizer::decode`] with prefix-diffing.
///
/// This is the same sliding-window algorithm used by `tokenizers::DecodeStream`
pub(crate) struct DecodeStream<'a, T: Tokenizer + ?Sized> {
    tokenizer: &'a T,
    skip_special_tokens: bool,
    min_bytes_to_buffer: usize,
    // mutated state
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
    cumulative_output: String,
    output_index: usize,
    /// Whether the one-shot prompt seed has run. Tracked explicitly (rather than
    /// inferring it from `prefix.is_empty`) so a prompt whose decode ends in
    /// U+FFFD — leaving `prefix` empty — does not re-trigger seeding on every
    /// push and re-emit the prompt.
    prompt_seeded: bool,
    /// Consecutive `push_token` calls that produced no emitted bytes (decode
    /// shrank or still ends in U+FFFD). Bounds the buffered window: see
    /// [`MAX_PENDING_TOKENS`].
    pending_since_emit: usize,
}

impl<'a, T: Tokenizer + ?Sized> DecodeStream<'a, T> {
    pub(crate) fn new(
        tokenizer: &'a T,
        prompt_token_ids: &[u32],
        skip_special_tokens: bool,
        min_bytes_to_buffer: usize,
    ) -> Self {
        Self {
            tokenizer,
            skip_special_tokens,
            min_bytes_to_buffer,
            ids: prompt_token_ids.to_vec(),
            prefix: String::new(),
            prefix_index: 0,
            cumulative_output: String::new(),
            output_index: 0,
            prompt_seeded: false,
            pending_since_emit: 0,
        }
    }
}

/// Try a short tail suffix first (covers a CJK glyph straddling 1-2 token
/// boundaries); beyond 6 tokens the fallback full-prompt decode is no worse
/// than baseline so widening the sweep just adds overhead.
const SAFE_SUFFIX_MIN: usize = 4;
const SAFE_SUFFIX_MAX: usize = 6;

/// Maximum number of consecutive non-emitting pushes (decode keeps ending in
/// U+FFFD or shrinking) before the buffered bytes are force-emitted. A genuine
/// incomplete UTF-8 scalar resolves within a handful of tokens, so this bound
/// never trips on well-formed streams; it exists only to keep the decode window
/// bounded on a pathological stream that never resolves its trailing U+FFFD
/// and to surface a legitimately-emitted U+FFFD incrementally instead of
/// withholding it until flush.
const MAX_PENDING_TOKENS: usize = 32;

impl<T: Tokenizer + ?Sized> DecodeStream<'_, T> {
    /// Seed `self.prefix` from the shortest trailing suffix whose decoded text
    /// has no U+FFFD — a clean decode means the suffix starts and ends at
    /// valid UTF-8/token boundaries, so priming from it is equivalent to
    /// priming from the full prompt.
    fn seed_prefix(&mut self) -> Result<()> {
        let prompt_len = self.ids.len();
        if prompt_len > SAFE_SUFFIX_MIN {
            let max_try = SAFE_SUFFIX_MAX.min(prompt_len - 1);
            for suffix_len in SAFE_SUFFIX_MIN..=max_try {
                let start = prompt_len - suffix_len;
                let decoded = self
                    .tokenizer
                    .decode(&self.ids[start..], self.skip_special_tokens)?;
                if !decoded.contains('\u{FFFD}') {
                    self.prefix = decoded;
                    self.ids.drain(..start);
                    self.prefix_index = self.ids.len();
                    return Ok(());
                }
            }
        }
        let decoded = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
        if decoded.ends_with('\u{FFFD}') {
            // The prompt tail is an incomplete UTF-8 scalar. Strip the trailing
            // replacement char(s) so `prefix.len` does not include them — a
            // same-byte-length completing scalar would otherwise be masked by the
            // `string.len <= prefix_len` guard in `push_token` and never emitted
            //. Keep every prompt id in the window (`prefix_index = 0`) so
            // the incomplete tail can still combine with the generated tokens.
            self.prefix = decoded.trim_end_matches('\u{FFFD}').to_string();
            self.prefix_index = 0;
        } else {
            self.prefix = decoded;
            self.prefix_index = self.ids.len();
        }
        Ok(())
    }
}

impl<T: Tokenizer + ?Sized> IncrementalDecoder for DecodeStream<'_, T> {
    fn push_token(&mut self, token_id: u32) -> Result<usize> {
        if !self.prompt_seeded {
            self.prompt_seeded = true;
            if !self.ids.is_empty() {
                self.seed_prefix()?;
            }
        }

        self.ids.push(token_id);
        let string = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
        let prefix_len = self.prefix.len();
        // Normally hold back a decode that shrank or still ends in an incomplete
        // U+FFFD. But once `MAX_PENDING_TOKENS` non-emitting pushes have piled up,
        // force the emit so the window stays bounded and a genuine trailing
        // U+FFFD surfaces incrementally rather than only at flush.
        let force = self.pending_since_emit >= MAX_PENDING_TOKENS;
        if !force && (string.len() <= prefix_len || string.ends_with('\u{FFFD}')) {
            self.pending_since_emit += 1;
            return Ok(0);
        }
        // Ensure we split at a utf-8 char boundary (clamps to string.len when
        // prefix_len exceeds it, i.e. a forced emit on a shrunk decode).
        let new_chunk = &string[string.floor_char_boundary(prefix_len)..];
        self.cumulative_output.push_str(new_chunk);
        self.ids.drain(..self.prefix_index);
        self.prefix = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
        self.prefix_index = self.ids.len();
        self.pending_since_emit = 0;
        Ok(new_chunk.len())
    }

    fn next_chunk(&mut self) -> Option<String> {
        let cutoff = self
            .cumulative_output
            .len()
            .saturating_sub(self.min_bytes_to_buffer);
        // Ensure we split at a utf-8 char boundary.
        let cutoff = self.cumulative_output.floor_char_boundary(cutoff);
        (cutoff > self.output_index).then(|| {
            let chunk = self.cumulative_output[self.output_index..cutoff].to_string();
            self.output_index = cutoff;
            chunk
        })
    }

    fn flush(&mut self, truncate_output_to: Option<usize>) -> Result<(Option<String>, String)> {
        if !self.ids.is_empty() {
            let string = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
            let prefix_len = self.prefix.len();
            self.ids.clear();
            self.prefix.clear();
            self.prefix_index = 0;
            self.pending_since_emit = 0;
            // Ensure we split at a utf-8 char boundary.
            self.cumulative_output
                .push_str(&string[string.floor_char_boundary(prefix_len)..]);
        }
        if let Some(truncate_output_to) = truncate_output_to {
            self.cumulative_output.truncate(truncate_output_to);
        }
        let last_chunk = (self.output_index < self.cumulative_output.len())
            .then(|| self.cumulative_output[self.output_index..].to_string());
        self.output_index = 0;
        Ok((last_chunk, take(&mut self.cumulative_output)))
    }

    fn output(&self) -> &str {
        &self.cumulative_output
    }
}
