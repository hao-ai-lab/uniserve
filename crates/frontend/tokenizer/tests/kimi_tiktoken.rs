//! Kimi-K2.5 tiktoken loading behavior for the tokenizer crate.
//!
//! These integration tests drive only the public [`TiktokenTokenizer`] /
//! [`Tokenizer`] API against a small committed offline fixture under
//! `tests/fixtures/kimi_mini` rather than downloading a real Kimi model. The
//! fixture is a hand-built `*.tiktoken` BPE file (single-byte base vocab plus
//! two Han merges) paired with a `config.json` declaring `model_type =
//! "kimi_k25"` and a `tokenizer_config.json` declaring Kimi's reasoning and
//! tool-call markers as non-special added tokens. Together they exercise the
//! observable behaviors called out for the `tokenizer-kimi` cluster:
//!
//! 1. The Kimi vocab resolves `<think>` / `</think>` /
//!    `<|tool_calls_section_begin|>` to their declared ids (and back).
//! 2. The Kimi `model_type` selects Kimi's BPE pre-tokenization pattern, not
//!    the `cl100k_base` default — observed through a token-count probe on a
//!    space+Han string that the two patterns split differently.
//! 3. Decode round-trips Kimi-flavored text (reasoning + tool-call markers and
//!    CJK) back to the original string.
//! 4. `token_to_id` returns `None` for special-looking text that is not
//!    registered in the vocabulary, without panicking.
//!
//! Both supported backends (`riptoken` and `tiktoken-rs`) are covered, since
//! the loader can use either and they must agree on these behaviors.
//!
//! The production-exact ids of the real `moonshotai/Kimi-K2.5` checkpoint are
//! deferred to an `#[ignore]`d online test below, since asserting them would
//! require downloading the real (uncommitted) model assets.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs;
use std::path::{Path, PathBuf};

use uniserve_tokenizer::{TiktokenTokenizer, Tokenizer};

/// Directory holding the committed mini Kimi fixture (`kimi_mini.tiktoken`,
/// `config.json`, `tokenizer_config.json`).
fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kimi_mini")
}

/// Path to the committed mini Kimi `*.tiktoken` BPE file. Loading it picks up
/// the sibling `config.json` (Kimi `model_type`, `vocab_size`) and
/// `tokenizer_config.json` (special-token declarations) automatically.
fn fixture_bpe_path() -> PathBuf {
    fixture_dir().join("kimi_mini.tiktoken")
}

/// Both supported tiktoken backends loaded from the committed fixture, so each
/// behavior is asserted identically for `riptoken` and `tiktoken-rs`.
fn kimi_backends() -> Vec<(&'static str, TiktokenTokenizer)> {
    let path = fixture_bpe_path();
    vec![
        (
            "riptoken",
            TiktokenTokenizer::new_riptoken(&path).expect("load riptoken backend from fixture"),
        ),
        (
            "tiktoken_rs",
            TiktokenTokenizer::new_tiktoken_rs(&path)
                .expect("load tiktoken-rs backend from fixture"),
        ),
    ]
}

/// Load a backend over the *committed* fixture's BPE bytes but with a chosen
/// `model_type` written into a temp-dir `config.json`. This keeps the
/// load-bearing BPE merges (which decide whether a leading space attaches to a
/// Han run) committed, while varying only the pattern-selecting `model_type`.
/// Used by the pattern probe to compare Kimi vs `cl100k_base` splitting.
fn riptoken_with_model_type(dir: &Path, model_type: Option<&str>) -> TiktokenTokenizer {
    let bpe_bytes = fs::read(fixture_bpe_path()).expect("read committed fixture bpe");
    let bpe_path = dir.join("vocab.tiktoken");
    fs::write(&bpe_path, &bpe_bytes).expect("write bpe into temp dir");
    let config = match model_type {
        Some(mt) => format!(r#"{{"model_type":"{mt}","vocab_size":384}}"#),
        None => r#"{"vocab_size":384}"#.to_string(),
    };
    fs::write(dir.join("config.json"), config).expect("write config.json into temp dir");
    TiktokenTokenizer::new_riptoken(&bpe_path).expect("load riptoken backend with model_type")
}

// ---------------------------------------------------------------------------
// 1. Special-token id resolution (both directions).
// ---------------------------------------------------------------------------

/// The Kimi reasoning and tool-call markers declared in the fixture's
/// `tokenizer_config.json` resolve to their declared ids via `token_to_id`,
/// and the same ids map back to those strings via `id_to_token`.
#[test]
fn kimi_special_markers_resolve_to_declared_ids() {
    for (name, backend) in kimi_backends() {
        assert_eq!(
            backend.token_to_id("<think>"),
            Some(258),
            "{name}: <think> id"
        );
        assert_eq!(
            backend.token_to_id("</think>"),
            Some(259),
            "{name}: </think> id"
        );
        assert_eq!(
            backend.token_to_id("<|tool_calls_section_begin|>"),
            Some(260),
            "{name}: tool-calls-section-begin id"
        );

        assert_eq!(
            backend.id_to_token(258).as_deref(),
            Some("<think>"),
            "{name}: id 258 -> <think>"
        );
        assert_eq!(
            backend.id_to_token(259).as_deref(),
            Some("</think>"),
            "{name}: id 259 -> </think>"
        );
        assert_eq!(
            backend.id_to_token(260).as_deref(),
            Some("<|tool_calls_section_begin|>"),
            "{name}: id 260 -> tool-calls-section-begin"
        );
    }
}

/// Kimi declares its reasoning and tool-call markers as `"special": false`, so
/// `is_special_id` reports them as non-special (they survive
/// `skip_special_tokens = true`), while a marker declared `"special": true`
/// (the fixture's `[BOS]`) is reported as special.
#[test]
fn kimi_non_special_markers_are_not_flagged_special() {
    for (name, backend) in kimi_backends() {
        assert!(
            !backend.is_special_id(258),
            "{name}: <think> should not be special"
        );
        assert!(
            !backend.is_special_id(260),
            "{name}: tool-calls-section-begin should not be special"
        );
        // 262 is declared "[BOS]" with "special": true in the fixture.
        assert!(
            backend.is_special_id(262),
            "{name}: [BOS] should be special"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Kimi BPE pattern selection, observed through a token-count probe.
// ---------------------------------------------------------------------------

/// The Kimi `model_type` selects Kimi's pre-tokenization pattern rather than
/// the `cl100k_base` default. The two patterns split a space immediately
/// followed by a Han character differently: `cl100k_base`'s
/// `[^\r\n\p{L}\p{N}]?\p{L}+` branch lets the leading space attach to the Han
/// "letter" run (so the committed `" \u{4F60}"` merge fires → 1 token), while
/// Kimi's leading `[\p{Han}]+` branch has no leading-space allowance (so the
/// space stays its own pre-token and cannot merge → strictly more tokens).
///
/// The assertion is on the *direction* of the count difference (Kimi splits
/// more), which follows purely from the documented pattern difference and is
/// decoupled from the exact BPE merge mechanics.
#[test]
fn kimi_pattern_splits_space_before_han_differently_than_cl100k() {
    let kimi_dir = tempfile::tempdir().expect("kimi temp dir");
    let cl100k_dir = tempfile::tempdir().expect("cl100k temp dir");
    let kimi = riptoken_with_model_type(kimi_dir.path(), Some("kimi_k25"));
    let cl100k = riptoken_with_model_type(cl100k_dir.path(), Some("gpt2"));

    // " 你": space directly before a Han char — the discriminating case.
    let space_han = " \u{4F60}";
    let kimi_ids = kimi.encode(space_han, false).expect("kimi encode");
    let cl100k_ids = cl100k.encode(space_han, false).expect("cl100k encode");

    assert!(
        kimi_ids.len() > cl100k_ids.len(),
        "Kimi pattern must keep the leading space separate from the Han run \
         (more tokens) while cl100k attaches it: kimi={kimi_ids:?} cl100k={cl100k_ids:?}"
    );

    // Both still decode back to the same input — the split differs, the
    // covered bytes do not.
    assert_eq!(kimi.decode(&kimi_ids, false).unwrap(), space_han);
    assert_eq!(cl100k.decode(&cl100k_ids, false).unwrap(), space_han);
}

/// Control for the probe above: a bare Han run with no leading space is not a
/// discriminating case, so the Kimi and `cl100k_base` patterns produce the
/// same tokenization. This shows the difference in the sibling test is
/// specifically the Kimi pattern's leading-space handling for Han, not a vocab
/// artifact that would differ for any input.
#[test]
fn kimi_and_cl100k_agree_on_bare_han_run() {
    let kimi_dir = tempfile::tempdir().expect("kimi temp dir");
    let cl100k_dir = tempfile::tempdir().expect("cl100k temp dir");
    let kimi = riptoken_with_model_type(kimi_dir.path(), Some("kimi_k25"));
    let cl100k = riptoken_with_model_type(cl100k_dir.path(), Some("gpt2"));

    let bare_han = "\u{4F60}\u{597D}"; // 你好, no leading space
    let kimi_ids = kimi.encode(bare_han, false).expect("kimi encode");
    let cl100k_ids = cl100k.encode(bare_han, false).expect("cl100k encode");

    assert_eq!(
        kimi_ids, cl100k_ids,
        "bare Han run is not a discriminating case: kimi={kimi_ids:?} cl100k={cl100k_ids:?}"
    );
}

/// Omitting `model_type` from `config.json` (no Kimi signal) falls back to the
/// `cl100k_base` pattern: the same space+Han probe attaches the leading space,
/// matching the explicit non-Kimi `model_type` case rather than the Kimi one.
#[test]
fn missing_model_type_uses_cl100k_pattern_for_space_before_han() {
    let none_dir = tempfile::tempdir().expect("none temp dir");
    let cl100k_dir = tempfile::tempdir().expect("cl100k temp dir");
    let no_model_type = riptoken_with_model_type(none_dir.path(), None);
    let cl100k = riptoken_with_model_type(cl100k_dir.path(), Some("gpt2"));

    let space_han = " \u{4F60}";
    assert_eq!(
        no_model_type.encode(space_han, false).unwrap(),
        cl100k.encode(space_han, false).unwrap(),
        "absent model_type should tokenize like cl100k, not Kimi"
    );
}

// ---------------------------------------------------------------------------
// 3. Decode round-trips Kimi-flavored text.
// ---------------------------------------------------------------------------

/// A string mixing the reasoning markers, the tool-call section marker, and CJK
/// text round-trips through encode -> decode without `skip_special_tokens`.
/// Because Kimi declares the markers as non-special, they also survive a
/// `skip_special_tokens = true` decode, so both decodes reproduce the input.
#[test]
fn kimi_flavored_text_round_trips_through_encode_decode() {
    let text = "<think>\u{4F60}\u{597D}</think><|tool_calls_section_begin|>";
    for (name, backend) in kimi_backends() {
        let ids = backend
            .encode(text, false)
            .expect("encode kimi-flavored text");

        let decoded_keep = backend.decode(&ids, false).expect("decode keep-special");
        assert_eq!(decoded_keep, text, "{name}: keep-special round-trip");

        // Markers are declared non-special, so skipping specials does not drop
        // them — the text is identical.
        let decoded_skip = backend.decode(&ids, true).expect("decode skip-special");
        assert_eq!(decoded_skip, text, "{name}: skip-special round-trip");
    }
}

/// The special markers encode to exactly their single declared ids when they
/// appear as standalone literal text, confirming the loader registers them
/// with the inner BPE encoder (not just in the reverse `token_to_id` map).
#[test]
fn kimi_markers_encode_to_single_special_ids() {
    for (name, backend) in kimi_backends() {
        assert_eq!(
            backend.encode("<think>", false).unwrap(),
            vec![258],
            "{name}: <think> encodes to its id"
        );
        assert_eq!(
            backend
                .encode("<|tool_calls_section_begin|>", false)
                .unwrap(),
            vec![260],
            "{name}: tool-calls-section-begin encodes to its id"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Unregistered special-like text resolves to None (no panic).
// ---------------------------------------------------------------------------

/// `token_to_id` returns `None` for special-looking strings that are not
/// registered in the Kimi vocabulary, on both backends, without panicking.
/// (`tiktoken-rs` historically could panic on unregistered special-looking
/// input; the loader's reverse-map + ordinary-encode fallback prevents that.)
#[test]
fn unregistered_special_like_text_resolves_to_none() {
    for (name, backend) in kimi_backends() {
        assert_eq!(
            backend.token_to_id("<|tool_calls_section_never_registered|>"),
            None,
            "{name}: unregistered tool-call-like marker"
        );
        assert_eq!(
            backend.token_to_id("<|definitely_not_a_real_kimi_token|>"),
            None,
            "{name}: unregistered reserved-looking marker"
        );
        assert_eq!(
            backend.token_to_id("</not_think>"),
            None,
            "{name}: unregistered angle-bracket marker"
        );
    }
}

// ---------------------------------------------------------------------------
// Gated: production-exact ids against the real Kimi-K2.5 checkpoint. This
// needs the real model's `tiktoken.model` + config, which are not committed
// offline, so it is ignored by default and documents the intended check.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires downloading the real moonshotai/Kimi-K2.5 tiktoken assets (not committed offline)"]
fn online_kimi_k25_production_exact_special_ids() {
    // Intentionally not implemented: enabling this requires fetching the real
    // `tiktoken.model`, `config.json`, and `tokenizer_config.json` from the
    // `moonshotai/Kimi-K2.5` repo (see benches/tiktoken.rs for the hf-hub
    // fetch). It would then load the tokenizer and assert the *production*
    // ids of `<think>`, `</think>`, and `<|tool_calls_section_begin|>` match
    // the checkpoint's `tokenizer_config.json`. The committed-fixture tests
    // above cover the loading/resolution behavior with offline-stable ids.
    unreachable!("ignored: enable with the real Kimi-K2.5 model assets");
}
