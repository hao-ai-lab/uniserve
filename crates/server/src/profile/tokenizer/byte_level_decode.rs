//! GPT-2 byte-level detokenization into a contiguous byte buffer.
//!
//! Token pieces are unescaped directly into one allocation before UTF-8
//! validation. `HuggingFaceTokenizer::decode` uses this path when the
//! tokenizer's decoder is exactly one byte-level step.

/// Reverse GPT-2 byte-to-unicode mapping: codepoint → original byte. The GPT-2
/// table only emits codepoints in U+0000..U+0143, so a flat array suffices.
const CHAR_TO_BYTE: [u8; 324] = build_char_to_byte();

/// Returns whether GPT-2's byte-to-unicode table maps byte `b` to the
/// codepoint of the same value: printable ASCII other than space, and
/// 0xA1..=0xFF except 0xAD.
const fn is_nice(b: u8) -> bool {
    (b >= b'!' && b <= b'~') || (b >= 0xA1 && b <= 0xAC) || b >= 0xAE
}

/// Builds the inverse of GPT-2's byte-to-unicode table.
///
/// The table is indexed by codepoint. A self-mapped byte sits at its own
/// value; every other byte sits at `256 + k`, where `k` is its rank among the
/// remapped bytes in ascending order.
const fn build_char_to_byte() -> [u8; 324] {
    let mut table = [0u8; 324];
    let mut b: u16 = 0;
    while b < 256 {
        let cp = if is_nice(b as u8) {
            b as u32
        } else {
            256 + nice_offset(b as u8)
        };
        table[cp as usize] = b as u8;
        b += 1;
    }
    table
}

/// Returns how many bytes below `b` are not self-mapped, which is the rank
/// of a remapped byte `b` in GPT-2's table.
const fn nice_offset(b: u8) -> u32 {
    let mut i: u16 = 0;
    let mut n: u32 = 0;
    while i < b as u16 {
        if !is_nice(i as u8) {
            n += 1;
        }
        i += 1;
    }
    n
}

/// Decodes byte-level encoded token strings into a single UTF-8 string,
/// matching `fastokens::decoders::ByteLevelDecoder`.
///
/// Byte sequences that are not valid UTF-8, such as a multi-byte character
/// cut off at either end of the token list, become U+FFFD replacement
/// characters.
pub(crate) fn decode_byte_level<'a, I: IntoIterator<Item = &'a str>>(tokens: I) -> String {
    let iter = tokens.into_iter();
    let (lower, _) = iter.size_hint();
    // Capacity is a heuristic from the token count; the buffer grows as needed.
    let mut bytes: Vec<u8> = Vec::with_capacity(lower.saturating_mul(4));
    for token in iter {
        for c in token.chars() {
            let cp = c as usize;
            if cp < CHAR_TO_BYTE.len() {
                bytes.push(CHAR_TO_BYTE[cp]);
            } else {
                // Codepoints beyond the table pass through as their UTF-8
                // bytes.
                let mut buf = [0u8; 4];
                bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    // Valid UTF-8 reuses the buffer; only invalid input pays for a lossy copy.
    String::from_utf8(bytes).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
}
