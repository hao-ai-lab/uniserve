//! Stateful incremental decoding with stable-prefix emission.
//!
//! Each generated token is decoded together with a window of preceding
//! tokens, and only the text past the window's previous decode is emitted.
//! Decoding a window instead of single tokens lets a UTF-8 character split
//! across tokens appear once its last byte arrives; until then the decode ends
//! in U+FFFD and `push_token` holds it back. Emitted text accumulates in
//! `output`, where the serving output stages match stop strings, and
//! `next_chunk` streams it while withholding a trailing `min_bytes_to_buffer`
//! bytes so a matched stop string can still be truncated by `flush`.

use std::mem::take;

use crate::profile::tokenizer::{HuggingFaceTokenizer, Result};

/// Stateful incremental decoder built on the configured Hugging Face tokenizer with prefix-diffing.
///
/// One decoder serves one generation stream: `push_token` for every generated
/// token, `next_chunk` for streamed text, and `flush` once at the end of the
/// stream or at a stop-string match.
pub struct IncrementalDecoder<'a> {
    tokenizer: &'a HuggingFaceTokenizer,
    skip_special_tokens: bool,
    min_bytes_to_buffer: usize,
    /// Token window decoded on every push. It starts as the prompt, which
    /// seeding may trim to a trailing suffix; each emit drops the tokens
    /// before `prefix_index`, keeping the tokens that produced the emitted
    /// text as left context for the next decode.
    ids: Vec<u32>,
    /// Decode of the window at the last emit or seed (minus trailing U+FFFD
    /// when the prompt ends in an incomplete character). Text past
    /// `prefix.len()` in a later window decode is new.
    prefix: String,
    /// Number of leading window tokens the next emit drops.
    prefix_index: usize,
    /// All text emitted by `push_token`, returned by `output`.
    cumulative_output: String,
    /// Byte offset in `cumulative_output` up to which `next_chunk` has
    /// returned text.
    output_index: usize,
    /// Whether the one-shot prompt seed has run. Tracked explicitly (rather than
    /// inferring it from `prefix.is_empty`) because a seeded `prefix` can be
    /// empty, for example when the seeded tokens are all skipped special
    /// tokens or the prompt decodes only to incomplete UTF-8 bytes; seeding
    /// again on a later push would treat generated tokens already in the
    /// window as prompt. Seeding runs on the first push, so this is also
    /// whether any generated token has entered the window, which `flush`
    /// needs to avoid emitting the unseeded prompt as output.
    prompt_seeded: bool,
    /// Consecutive `push_token` calls that held back their decode (no longer
    /// than `prefix`, or ending in U+FFFD). Bounds the buffered window: see
    /// [`MAX_PENDING_TOKENS`].
    pending_since_emit: usize,
}

impl<'a> IncrementalDecoder<'a> {
    /// Creates an incremental decoder for one generation stream.
    ///
    /// Construction does not decode. The prompt is seeded as decode context on
    /// the first `push_token`, which reports any decode error from seeding.
    pub(crate) fn new(
        tokenizer: &'a HuggingFaceTokenizer,
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

/// Shortest trailing prompt suffix, in tokens, that `seed_prefix` tries as
/// decode context before it falls back to decoding the full prompt.
const SAFE_SUFFIX_MIN: usize = 4;
/// Longest trailing prompt suffix, in tokens, that `seed_prefix` tries.
const SAFE_SUFFIX_MAX: usize = 6;

/// Maximum number of consecutive non-emitting pushes before `push_token`
/// forces an emit.
///
/// A push emits nothing while the window decode is no longer than `prefix` or
/// ends in U+FFFD. An incomplete UTF-8 character in well-formed output
/// resolves within a few tokens, well before this bound. The bound keeps the
/// decode window from growing without limit when a trailing U+FFFD never
/// resolves, and limits how long a U+FFFD that is part of the generated text
/// is withheld while it ends the output. Pushes that add no visible text, such
/// as special tokens removed by `skip_special_tokens`, also count toward the
/// bound; when the forcing push adds no text either, the forced emit appends
/// nothing and only slides the window.
const MAX_PENDING_TOKENS: usize = 32;

impl IncrementalDecoder<'_> {
    /// Seeds `self.prefix` from the shortest trailing suffix whose decoded text
    /// has no U+FFFD.
    ///
    /// A clean decode means the suffix neither starts nor ends inside a
    /// multi-byte UTF-8 character, so it can stand in for the full prompt as
    /// left context. Only prompts longer than `SAFE_SUFFIX_MIN` tokens try
    /// suffixes, and a suffix never spans the whole prompt. Without a clean
    /// suffix the full prompt is decoded and stays in the window as left
    /// context.
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
            // replacement char(s) so `prefix.len` does not include them; a
            // completing scalar of the same byte length would otherwise be
            // masked by the `string.len <= prefix_len` guard in `push_token`.
            // Keep every prompt id in the window (`prefix_index = 0`) so the
            // incomplete tail can still combine with the generated tokens.
            self.prefix = decoded.trim_end_matches('\u{FFFD}').to_string();
            self.prefix_index = 0;
        } else {
            self.prefix = decoded;
            self.prefix_index = self.ids.len();
        }
        Ok(())
    }
}

impl IncrementalDecoder<'_> {
    /// Pushes one generated token and returns how many bytes it appended to
    /// `output`.
    ///
    /// Returns 0 when nothing is appended, as while new text is held back.
    /// The first call also seeds the prompt context. Errors are tokenizer
    /// decode failures, such as an unknown token ID on the byte-level decode
    /// path.
    pub fn push_token(&mut self, token_id: u32) -> Result<usize> {
        if !self.prompt_seeded {
            self.prompt_seeded = true;
            if !self.ids.is_empty() {
                self.seed_prefix()?;
            }
        }

        self.ids.push(token_id);
        let string = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
        let prefix_len = self.prefix.len();
        // Normally hold back a decode that did not grow or still ends in an
        // incomplete U+FFFD. But once `MAX_PENDING_TOKENS` non-emitting pushes have piled up,
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

        // Slide the window: the tokens behind this emit become the context
        // whose decode is the next `prefix`.
        self.ids.drain(..self.prefix_index);
        self.prefix = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
        self.prefix_index = self.ids.len();
        self.pending_since_emit = 0;
        Ok(new_chunk.len())
    }

    /// Returns emitted text that no earlier call has returned, excluding the
    /// trailing `min_bytes_to_buffer` bytes of `output`.
    ///
    /// The cutoff is rounded down to a character boundary, so at least
    /// `min_bytes_to_buffer` bytes stay withheld (all of `output` when it is
    /// shorter). Returns `None` when no text
    /// before the cutoff is pending. This does not decode.
    pub fn next_chunk(&mut self) -> Option<String> {
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

    /// Flushes buffered tokens and optionally truncates final output bytes.
    ///
    /// Decodes the generated tokens the window still holds (including a
    /// trailing U+FFFD) into the output, then truncates the output to
    /// `truncate_output_to` bytes when given. The offset must lie on a
    /// character boundary, as a stop-string match offset does, or
    /// `String::truncate` panics. Before the first `push_token` the window
    /// holds only the unseeded prompt, which is decode context rather than
    /// output, so a stream that generated no token flushes no text.
    ///
    /// Returns the text not yet returned by `next_chunk` (`None` if there is
    /// none) and the complete output. The window and output are emptied, so
    /// callers flush once per stream.
    pub fn flush(&mut self, truncate_output_to: Option<usize>) -> Result<(Option<String>, String)> {
        if !self.prompt_seeded {
            self.ids.clear();
        } else if !self.ids.is_empty() {
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

    /// Returns all text emitted by `push_token` so far, including the bytes
    /// that `next_chunk` still withholds.
    pub fn output(&self) -> &str {
        &self.cumulative_output
    }
}

#[cfg(test)]
mod tests {
    /// A stream that ends before its first generated token, such as an
    /// image-only generation or a request whose first sampled token is EOS,
    /// produced no text: the prompt is decode context, never output. Prompts
    /// shorter and longer than the seed's trailing-suffix window are both
    /// covered because prompt seeding treats them differently.
    #[test]
    fn flush_without_generated_tokens_emits_nothing() {
        let tokenizer = crate::serving::test_support::configured_tokenizer();
        for prompt in ["p", "a longer prompt"] {
            let prompt_token_ids: Vec<u32> = prompt.bytes().map(u32::from).collect();
            let mut decoder = tokenizer.create_decode_stream(&prompt_token_ids, true, 0);

            let (last_chunk, output) = decoder.flush(None).unwrap();

            assert_eq!(last_chunk, None, "prompt {prompt:?}");
            assert_eq!(output, "", "prompt {prompt:?}");
        }
    }
}
