//! Streaming lexical helpers shared by tool parsers.
//!
//! The parsing helpers are `winnow` parsers over `Partial` input: they return
//! `Incomplete` when the buffered text cannot be classified yet, and
//! `parse_buffered_event` turns that into "wait for the next chunk".

use winnow::error::{ContextError, ErrMode, ModalResult, Needed, StrContext, StrContextValue};
use winnow::stream::{Offset, Partial, Stream};

use super::Result;

/// Upper bound on a tool parser's per-stream pending buffer, in bytes.
///
/// A streaming parser retains the input that has not yet formed a complete
/// event, such as an unterminated tool call. Exceeding the cap fails
/// `ToolParser::parse_into` with a parse error instead of growing memory
/// without bound.
pub(super) const MAX_BUFFER_BYTES: usize = 1 << 20;

/// Returns the byte length of the longest proper prefix of `token` that is also
/// a suffix of `buffer`.
///
/// Streaming parsers use this to keep only the trailing fragment that might
/// still grow into a full marker after the next decoded chunk arrives.
///
/// The returned length is always a valid UTF-8 boundary in `token`, so callers
/// can safely slice `&token[..len]` for ASCII and non-ASCII markers.
pub(super) fn partial_prefix_len(buffer: &str, token: &str) -> usize {
    let Some(first_byte) = token.as_bytes().first().copied() else {
        return 0;
    };

    let max_len = buffer.len().min(token.len().saturating_sub(1));
    let tail_start = buffer.len() - max_len;
    let buffer_bytes = buffer.as_bytes();
    let token_bytes = token.as_bytes();

    // Scan from the longest possible suffix to preserve overlapping prefixes.
    for index in tail_start..buffer.len() {
        if buffer_bytes[index] != first_byte {
            continue;
        }

        let len = buffer.len() - index;
        if buffer.is_char_boundary(index)
            && token.is_char_boundary(len)
            && token_bytes[..len] == buffer_bytes[index..]
        {
            return len;
        }
    }

    0
}

/// Parses a safe text run before the next marker.
///
/// Safe text cannot be part of `marker`: everything before its first full
/// occurrence or, without one, everything except a trailing partial-marker
/// prefix. Returns the text length in bytes, and advances the input. Returns
/// `Incomplete` when nothing can be emitted yet (empty input, or input that is
/// entirely a possible marker prefix), and `Ok(0)` without consuming when the
/// input starts with `marker`.
pub(super) fn safe_text_len(input: &mut Partial<&str>, marker: &str) -> ModalResult<usize> {
    let text = **input;
    if text.is_empty() {
        return incomplete();
    }

    if let Some(start_idx) = text.find(marker) {
        input.next_slice(start_idx);
        return Ok(start_idx);
    }

    let keep_len = partial_prefix_len(text, marker);
    let emit_len = text.len().saturating_sub(keep_len);
    if emit_len == 0 {
        return incomplete();
    }

    input.next_slice(emit_len);
    Ok(emit_len)
}

/// Streaming lexical state for a top-level JSON object.
///
/// Carries the scan position across chunks so `take_json_object` resumes where
/// the previous chunk ended. Object and array depths are counted
/// independently rather than as a nesting stack, so the scan checks only
/// balance, not correct interleaving.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct JsonObjectScanState {
    object_depth: usize,
    array_depth: usize,
    /// Whether the scan is inside a string literal, where braces and brackets
    /// are not structural.
    in_string: bool,
    /// Whether the previous string byte was an unescaped backslash.
    escape: bool,
    phase: JsonObjectScanPhase,
}

/// Progress of one `take_json_object` scan.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum JsonObjectScanPhase {
    /// No byte consumed; the next byte must be `{`.
    #[default]
    Initial,
    Scanning,
    Complete,
}

impl JsonObjectScanState {
    /// Returns whether the top-level JSON object has closed.
    pub(super) const fn complete(&self) -> bool {
        matches!(self.phase, JsonObjectScanPhase::Complete)
    }
}

/// Parses a raw top-level JSON object argument prefix.
///
/// The returned length is safe to emit as raw argument text. This scans only
/// lexical boundaries from `{` through the matching `}`, preserving
/// malformed-but-balanced JSON without deserializing or normalizing it.
///
/// On success a call consumes either every buffered byte or the bytes through
/// the closing `}`, which marks `state` complete. Returns `Incomplete` on empty
/// input and a `Cut` error when `state` is already complete, when the first
/// byte is not `{`, or when braces and brackets do not balance (an unmatched
/// `]`, or the top-level object closing while an array is still open).
pub(super) fn take_json_object(
    input: &mut Partial<&str>,
    state: &mut JsonObjectScanState,
) -> ModalResult<usize> {
    let text = **input;
    if text.is_empty() {
        return incomplete();
    }
    if state.complete() {
        return Err(json_scan_error(
            "JSON object argument",
            StrContextValue::Description("active JSON object scan"),
        ));
    }

    let bytes = text.as_bytes();
    let just_started = matches!(state.phase, JsonObjectScanPhase::Initial);
    if just_started {
        if bytes[0] != b'{' {
            return Err(json_scan_error(
                "JSON object argument",
                StrContextValue::CharLiteral('{'),
            ));
        }
        state.phase = JsonObjectScanPhase::Scanning;
        state.object_depth = 1;
    }

    // Scanning bytes is UTF-8 safe: every structural byte is ASCII, which never
    // occurs inside a multi-byte sequence, so each returned length ends on a
    // character boundary.
    let mut index = usize::from(just_started);

    while index < bytes.len() {
        let byte = bytes[index];
        index += 1;

        if state.in_string {
            if state.escape {
                state.escape = false;
            } else if byte == b'\\' {
                state.escape = true;
            } else if byte == b'"' {
                state.in_string = false;
            }
            continue;
        }

        match byte {
            b'"' => state.in_string = true,
            b'{' => state.object_depth += 1,
            b'}' => {
                state.object_depth = state.object_depth.checked_sub(1).ok_or_else(|| {
                    json_scan_error(
                        "JSON object argument",
                        StrContextValue::Description("balanced object braces"),
                    )
                })?;
                if state.object_depth == 0 && state.array_depth == 0 {
                    state.phase = JsonObjectScanPhase::Complete;
                    input.next_slice(index);
                    return Ok(index);
                }
                if state.object_depth == 0 {
                    return Err(json_scan_error(
                        "JSON object argument",
                        StrContextValue::Description(
                            "nested arrays to close before the top-level object",
                        ),
                    ));
                }
            }
            b'[' => state.array_depth += 1,
            b']' => {
                state.array_depth = state.array_depth.checked_sub(1).ok_or_else(|| {
                    json_scan_error(
                        "JSON object argument",
                        StrContextValue::Description("balanced array brackets"),
                    )
                })?;
            }
            _ => {}
        }
    }

    input.next_slice(text.len());
    Ok(text.len())
}

/// Parses a JSON string literal and returns its unescaped value.
///
/// Returns `Incomplete` until the closing quote is buffered, and a `Cut` error
/// when the input does not start with `"` or `serde_json` rejects the literal.
pub(super) fn json_str(input: &mut Partial<&str>) -> ModalResult<String> {
    let text = **input;
    if text.is_empty() {
        return incomplete();
    }

    let bytes = text.as_bytes();
    if bytes[0] != b'"' {
        return Err(json_scan_error(
            "JSON string",
            StrContextValue::CharLiteral('"'),
        ));
    }

    let mut escape = false;
    let mut index = 1;
    while index < bytes.len() {
        let byte = bytes[index];
        index += 1;

        if escape {
            escape = false;
            continue;
        }

        match byte {
            b'\\' => escape = true,
            b'"' => {
                let raw = &text[..index];
                let value = serde_json::from_str::<String>(raw).map_err(|_| {
                    json_scan_error(
                        "JSON string",
                        StrContextValue::Description("valid JSON string"),
                    )
                })?;
                input.next_slice(index);
                return Ok(value);
            }
            _ => {}
        }
    }

    incomplete()
}

/// Builds a non-backtracking (`Cut`) parser error labelled `label` that
/// reports `expected` as the expected input.
fn json_scan_error(label: &'static str, expected: StrContextValue) -> ErrMode<ContextError> {
    let mut error = ContextError::new();
    error.push(StrContext::Label(label));
    error.push(StrContext::Expected(expected));
    ErrMode::Cut(error)
}

/// Parses one event from a buffered streaming input.
///
/// Returns:
/// - `Ok(Some((event, consumed_len)))` if an event was successfully parsed, along with the number
///   of bytes consumed from the buffer.
/// - `Ok(None)` if the buffer does not contain a full event yet, and more data is needed.
/// - `Err` if a parsing error occurred.
///
/// A parse that succeeds without consuming input also yields `Ok(None)`, so a
/// caller looping until `Ok(None)` cannot spin on an empty event.
pub(super) fn parse_buffered_event<E>(
    buffer: &str,
    parse: impl FnOnce(&mut Partial<&str>) -> ModalResult<E>,
) -> Result<Option<(E, usize)>> {
    let mut input = Partial::new(buffer);
    let checkpoint = input.checkpoint();
    let event = match parse(&mut input) {
        Ok(event) => event,
        Err(ErrMode::Incomplete(_)) => return Ok(None),
        Err(ErrMode::Backtrack(e) | ErrMode::Cut(e)) => {
            // The message is the rendered winnow context chain that the
            // grammar functions attach, such as a label and expected input.
            return Err(parsing_failed!("{}", e));
        }
    };
    let consumed_len = input.offset_from(&checkpoint);
    if consumed_len == 0 {
        return Ok(None);
    }

    Ok(Some((event, consumed_len)))
}

/// Returns the `Incomplete` error that asks for more input.
pub(super) fn incomplete<T>() -> ModalResult<T> {
    Err(ErrMode::Incomplete(Needed::Unknown))
}
