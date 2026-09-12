//! Cross-language fixtures for the sampling RNG and inverse-CDF boundary.
//!
//! Each fixture records a Philox key, output words, uniform draw, and selected
//! token for a semantic sampling coordinate. The Python worker recomputes the
//! same values to verify byte-identical protocol behavior.

use std::path::PathBuf;

use uniserve_core::philox::{philox4x32_10, sampling_key, sampling_uniform};

/// A fixed dyadic distribution (each probability is a multiple of 2^-4, so the
/// cumulative sums are exact in binary and the draw comparison is reproducible
/// across languages). Ascending token order.
const PROBS: [f32; 6] = [
    2.0 / 16.0,
    3.0 / 16.0,
    1.0 / 16.0,
    4.0 / 16.0,
    5.0 / 16.0,
    1.0 / 16.0,
];

fn select_token(draw: f32) -> u32 {
    let mut cumulative = 0.0f32;
    for (index, &probability) in PROBS.iter().enumerate() {
        cumulative += probability;
        if draw <= cumulative {
            return index as u32;
        }
    }
    (PROBS.len() - 1) as u32
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("tests")
        .join("python")
        .join("generated")
        .join("sampling_rng_parity.json")
}

#[test]
fn emit_sampling_rng_parity_fixture() {
    let session_seeds = [0u64, 1, 0x0123_4567_89ab_cdef, u64::MAX];
    let identities = [(0u64, 0u64, 0u64), (7, 11, 3), (4_000_000_000, 5, 900)];
    let draw_layouts = [0u64, 1, 2];
    let coordinates = [
        (0u64, 0u64, 0u64),
        (1, 0, 0),
        (40, 0, 0),
        (5_000_000_000, 2, 7),
    ];

    // Enumerate the protocol coordinate space and record both the raw bijection
    // output and its derived sampling decisions.
    let mut cases = Vec::new();
    for &session_seed in &session_seeds {
        for &(engine_id, request_id, request_epoch) in &identities {
            for &draw_layout in &draw_layouts {
                let key = sampling_key(
                    session_seed,
                    engine_id,
                    request_id,
                    request_epoch,
                    draw_layout,
                );
                for &(semantic_token_index, processor_stage, draw_index) in &coordinates {
                    let counter = [
                        semantic_token_index as u32,
                        (semantic_token_index >> 32) as u32,
                        processor_stage as u32,
                        draw_index as u32,
                    ];
                    let words = philox4x32_10(counter, [key as u32, (key >> 32) as u32]);
                    let uniform =
                        sampling_uniform(key, semantic_token_index, processor_stage, draw_index);
                    cases.push(serde_json::json!({
                        "session_seed": session_seed,
                        "engine_id": engine_id,
                        "request_id": request_id,
                        "request_epoch": request_epoch,
                        "draw_layout": draw_layout,
                        "semantic_token_index": semantic_token_index,
                        "processor_stage": processor_stage,
                        "draw_index": draw_index,
                        "key": key,
                        "words": words,
                        "uniform_bits": uniform.to_bits(),
                        "selected_token": select_token(uniform),
                    }));
                }
            }
        }
    }

    let fixture = serde_json::json!({
        "probs": PROBS,
        "cases": cases,
    });

    // Keep the cross-language fixture under the shared Python test-data tree.
    let path = fixture_path();
    std::fs::create_dir_all(path.parent().unwrap()).expect("create fixture directory");
    std::fs::write(&path, serde_json::to_vec_pretty(&fixture).unwrap()).expect("write fixture");
}
