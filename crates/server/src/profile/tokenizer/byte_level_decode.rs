//! GPT-2 byte-level detokenization into a contiguous byte buffer.
//!
//! Token pieces are unescaped directly into one allocation before UTF-8
//! validation.

/// Reverse GPT-2 byte-to-unicode mapping: codepoint → original byte. The GPT-2
/// table only emits codepoints in U+0000..U+0143, so a flat array suffices.
const CHAR_TO_BYTE: [u8; 324] = build_char_to_byte();

/// Returns whether a byte offset is safe for direct decoding.
const fn is_nice(b: u8) -> bool {
    (b >= b'!' && b <= b'~') || (b >= 0xA1 && b <= 0xAC) || b >= 0xAE
}
/// Builds the char to byte.
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

/// Returns the nearest safe byte offset at or before the requested position.
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
pub(crate) fn decode_byte_level<'a, I: IntoIterator<Item = &'a str>>(tokens: I) -> String {
    let iter = tokens.into_iter();
    let (lower, _) = iter.size_hint();
    let mut bytes: Vec<u8> = Vec::with_capacity(lower.saturating_mul(4));
    for token in iter {
        for c in token.chars() {
            let cp = c as usize;
            if cp < CHAR_TO_BYTE.len() {
                bytes.push(CHAR_TO_BYTE[cp]);
            } else {
                // Non-GPT-2 codepoints pass through unchanged.
                let mut buf = [0u8; 4];
                bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    String::from_utf8(bytes).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
}
