//! Python text forms of decoded JSON values.
//!
//! The readout prompt writes structured values the way the reference encoder
//! does, with Python's `json.dumps(value, ensure_ascii=False)`, and the
//! validation errors name an invalid question type the way pydantic does, with
//! Python's `str()` of the decoded value. Both forms are reproduced here for
//! values parsed by `serde_json` (with key order preserved).
//!
//! `serde_json` decodes some numbers differently from Python's `json` module,
//! so their text can differ: integers outside the 64-bit range become floats,
//! and `-0` becomes the float `-0.0` (Python keeps the integer `0`).

use serde_json::{Number, Value};

/// Returns `value` as Python's `json.dumps(value, ensure_ascii=False)` writes it.
///
/// Items are separated by `", "` and keys from values by `": "`, object keys
/// keep their order, non-ASCII characters are written as themselves, and
/// floats use Python's shortest round-trip `repr`.
pub(super) fn json_dumps(value: &Value) -> String {
    let mut out = String::new();
    write_json(&mut out, value);
    out
}

fn write_json(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => out.push_str(&number_repr(number)),
        Value::String(text) => write_json_string(out, text),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_json(out, item);
            }
            out.push(']');
        }
        Value::Object(fields) => {
            out.push('{');
            for (index, (key, item)) in fields.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_json_string(out, key);
                out.push_str(": ");
                write_json(out, item);
            }
            out.push('}');
        }
    }
}

/// Writes a JSON string literal with `ensure_ascii=False` escaping: only the
/// quote, the backslash, and control characters below U+0020 are escaped.
fn write_json_string(out: &mut String, text: &str) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            control if u32::from(control) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", u32::from(control)));
            }
            other => out.push(other),
        }
    }
    out.push('"');
}

/// Returns a decoded JSON number as Python writes the corresponding `int` or
/// `float`.
fn number_repr(number: &Number) -> String {
    if let Some(integer) = number.as_i64() {
        integer.to_string()
    } else if let Some(integer) = number.as_u64() {
        integer.to_string()
    } else {
        // serde_json holds every other number as a finite f64.
        float_repr(number.as_f64().unwrap_or(f64::NAN))
    }
}

/// Returns Python's `repr` of a float.
///
/// Python writes the shortest digit string that round-trips, in positional
/// notation with at least one fractional digit when the decimal exponent lies
/// in `[-4, 16)`, and otherwise as `d.ddde±XX` with at least two exponent
/// digits. Rust's `{:e}` formatting yields the same shortest digits.
pub(super) fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_owned();
    }
    if value.is_infinite() {
        return if value > 0.0 { "Infinity" } else { "-Infinity" }.to_owned();
    }
    let sign = if value.is_sign_negative() { "-" } else { "" };
    if value == 0.0 {
        return format!("{sign}0.0");
    }

    let scientific = format!("{:e}", value.abs());
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let exponent: i32 = exponent.parse().unwrap_or(0);
    // Position of the decimal point relative to the start of `digits`.
    let point = exponent + 1;

    if !(-4 < point && point <= 16) {
        let (first, rest) = digits.split_at(1);
        let fraction = if rest.is_empty() {
            String::new()
        } else {
            format!(".{rest}")
        };
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        format!(
            "{sign}{first}{fraction}e{exponent_sign}{:02}",
            exponent.abs()
        )
    } else if point <= 0 {
        format!(
            "{sign}0.{}{digits}",
            "0".repeat(point.unsigned_abs() as usize)
        )
    } else {
        let point = point as usize;
        if point >= digits.len() {
            format!("{sign}{digits}{}.0", "0".repeat(point - digits.len()))
        } else {
            format!("{sign}{}.{}", &digits[..point], &digits[point..])
        }
    }
}

/// Returns Python's `str()` of a decoded JSON value: a string as itself and
/// any other value as its `repr` (`None`, `True`, `['a', 1]`, `{'k': 1.5}`).
pub(super) fn python_str(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => python_repr(other),
    }
}

fn python_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Number(number) => number_repr(number),
        Value::String(text) => string_repr(text),
        Value::Array(items) => {
            let items: Vec<String> = items.iter().map(python_repr).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Object(fields) => {
            let fields: Vec<String> = fields
                .iter()
                .map(|(key, item)| format!("{}: {}", string_repr(key), python_repr(item)))
                .collect();
            format!("{{{}}}", fields.join(", "))
        }
    }
}

/// Returns Python's `repr` of a string.
///
/// The quote is `'` unless the text contains `'` and no `"`. Backslashes, the
/// quote, and the named whitespace escapes are escaped, and so are the C0 and
/// C1 control characters and the separator and format characters Python does
/// not print; other characters are written as themselves.
fn string_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::from(quote);
    for character in text.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if !python_printable(c) => {
                let code = u32::from(c);
                if code <= 0xff {
                    out.push_str(&format!("\\x{code:02x}"));
                } else if code <= 0xffff {
                    out.push_str(&format!("\\u{code:04x}"));
                } else {
                    out.push_str(&format!("\\U{code:08x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Approximates Python's `str.isprintable` for one character: control,
/// separator (other than the space), and common format characters are not
/// printable. Unassigned code points, which Python also escapes, are written
/// as themselves.
fn python_printable(character: char) -> bool {
    !matches!(
        character,
        '\u{00}'..='\u{1f}'
            | '\u{7f}'..='\u{a0}'
            | '\u{ad}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200f}'
            | '\u{2028}'..='\u{202f}'
            | '\u{205f}'..='\u{2064}'
            | '\u{3000}'
            | '\u{feff}'
    )
}

/// Returns whether Python's `str.isspace` holds for a character.
///
/// Python's whitespace is Unicode `White_Space` plus the information
/// separators U+001C..U+001F.
pub(super) fn python_space(character: char) -> bool {
    character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{float_repr, json_dumps, python_str};

    /// Each expected text is what CPython 3.12 prints for the same decoded JSON.
    #[test]
    fn structured_values_print_as_python_json_dumps() {
        let value: serde_json::Value = serde_json::from_str(
            r#"{"subject": "Doppelte Abbuchung — ñ 你好", "amount": 12.5, "count": 3,
                "big": 1e16, "small": 0.00001, "flags": [true, false, null],
                "nested": {"quote": "a \"b\" \\ c\nd\te\u0001\u007f", "empty": {}, "list": []}}"#,
        )
        .unwrap();

        assert_eq!(
            json_dumps(&value),
            r#"{"subject": "Doppelte Abbuchung — ñ 你好", "amount": 12.5, "count": 3, "big": 1e+16, "small": 1e-05, "flags": [true, false, null], "nested": {"quote": "a \"b\" \\ c\nd\te\u0001"#
                .to_owned()
                + "\u{7f}"
                + r#"", "empty": {}, "list": []}}"#
        );
        assert_eq!(
            json_dumps(&json!(["a", -7, 18446744073709551615u64])),
            r#"["a", -7, 18446744073709551615]"#
        );
    }

    #[test]
    fn floats_print_as_python_repr() {
        for (value, expected) in [
            (1.0, "1.0"),
            (100.0, "100.0"),
            (0.1, "0.1"),
            (-2.5, "-2.5"),
            (0.0001, "0.0001"),
            (0.00012, "0.00012"),
            (0.00001, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.2345e17, "1.2345e+17"),
            (1.5e300, "1.5e+300"),
            (123456.789, "123456.789"),
            (-0.0, "-0.0"),
            (5e-324, "5e-324"),
        ] {
            assert_eq!(float_repr(value), expected, "{value:e}");
        }
    }

    #[test]
    fn non_string_values_print_as_python_str() {
        assert_eq!(python_str(&json!("boolean")), "boolean");
        assert_eq!(python_str(&json!(null)), "None");
        assert_eq!(python_str(&json!(true)), "True");
        assert_eq!(python_str(&json!(1.5)), "1.5");
        assert_eq!(
            python_str(&json!(["noul", 1.5, null, true, {"k": "it's"}])),
            r#"['noul', 1.5, None, True, {'k': "it's"}]"#
        );
        assert_eq!(python_str(&json!(["a'b\"c\n"])), r#"['a\'b"c\n']"#);
    }
}
