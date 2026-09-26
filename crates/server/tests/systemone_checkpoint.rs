#![allow(clippy::expect_used, clippy::unwrap_used)]
//! System One readout plans for the published DiffusionGemma checkpoints.
//!
//! Every plan must equal, token for token, the golden plan of
//! `tests/python/fixtures/diffusion_gemma_readout_prompts.json`: the reference
//! DJev encoder's prompts and canvases for requests it accepts, and the prompt
//! format 1 rules rendered by the Hugging Face chat template for the rest.
//! Both checkpoints produce the same golden plans.
//!
//! The tests read the checkpoint directories named by
//! `UNISERVE_DIFFUSION_GEMMA_MODEL` (BF16) and
//! `UNISERVE_DIFFUSION_GEMMA_NVFP4_MODEL` (NVFP4) and run only on request:
//! `cargo test -p uniserve-server --test systemone_checkpoint -- --include-ignored`.

use std::collections::BTreeSet;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value;
use uniserve_server::profile::assets::ResolvedModelFiles;
use uniserve_server::profile::tokenizer::HuggingFaceTokenizer;
use uniserve_server::profile::{ModelConfig, ModelDescription, ModelParameters};
use uniserve_server::serving::systemone::{
    CanvasMode, ImagePlacement, ImageSize, ReadoutEncoder, ReadoutLayout, ReadoutOptions,
    ReadoutPlan, SystemOneRequest,
};

#[derive(Deserialize)]
struct Fixture {
    canvas_length: usize,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    layout: String,
    canvas: String,
    request: Value,
    #[serde(default)]
    images: Vec<Size>,
    prompts: Vec<Prompt>,
    shared_prefix_tokens: u32,
    input_tokens: u32,
}

#[derive(Deserialize)]
struct Size {
    width: u32,
    height: u32,
}

#[derive(Deserialize)]
struct Prompt {
    token_ids: Vec<u32>,
    images: Vec<Placement>,
    rows: Vec<Row>,
}

#[derive(Deserialize)]
struct Placement {
    source: usize,
    offset: u32,
    soft_tokens: u32,
}

/// A canvas row: explicit tokens up to `<turn|>` padded to `length`, or a
/// copy of an earlier row with one position set.
#[derive(Deserialize)]
struct Row {
    token_ids: Option<Vec<u32>>,
    length: Option<usize>,
    like_row: Option<usize>,
    set: Option<SetToken>,
    slots: Vec<Slot>,
}

#[derive(Deserialize)]
struct SetToken {
    position: usize,
    token: u32,
}

#[derive(Deserialize)]
struct Slot {
    position: u32,
    candidates: Vec<u32>,
}

#[tokio::test]
#[ignore = "requires the BF16 DiffusionGemma checkpoint named by UNISERVE_DIFFUSION_GEMMA_MODEL"]
async fn bf16_readout_plans_match_the_golden_plans() {
    check_checkpoint("UNISERVE_DIFFUSION_GEMMA_MODEL", false).await;
}

#[tokio::test]
#[ignore = "requires the NVFP4 DiffusionGemma checkpoint named by UNISERVE_DIFFUSION_GEMMA_NVFP4_MODEL"]
async fn nvfp4_readout_plans_match_the_golden_plans() {
    check_checkpoint("UNISERVE_DIFFUSION_GEMMA_NVFP4_MODEL", true).await;
}

async fn check_checkpoint(variable: &str, quantized: bool) {
    let directory = std::env::var(variable)
        .unwrap_or_else(|_| panic!("{variable} must name the checkpoint directory"));
    let files = ResolvedModelFiles::new(&directory).await.unwrap();
    let tokenizer = Arc::new(HuggingFaceTokenizer::new(&files.tokenizer_path).unwrap());
    let config = ModelConfig::from_files("diffusion-gemma", &directory, &files, None, &tokenizer)
        .await
        .unwrap();

    // Checkpoint facts the server serves from.
    assert_eq!(config.description(), ModelDescription::DiffusionGemma);
    assert_eq!(config.max_model_tokens, Some(262_144));
    assert_eq!(config.eos_token_ids, BTreeSet::from([1, 50, 106]));
    let ModelParameters::DiffusionGemma(profile) = &config.parameters else {
        unreachable!("the description is DiffusionGemma");
    };
    assert_eq!(profile.canvas_length, 256);
    let tokens = profile.tokens;
    assert_eq!(
        (tokens.mask, tokens.pad, tokens.turn_end),
        (4, 0, 106),
        "canvas control tokens"
    );
    assert_eq!(
        (tokens.image, tokens.image_start, tokens.image_end),
        (258_880, 255_999, 258_882),
        "image tokens"
    );
    let images = profile.images;
    assert_eq!(
        (
            images.patch_size,
            images.pooling_kernel_size,
            images.max_soft_tokens
        ),
        (16, 3, 280)
    );
    let denoising = profile.denoising;
    assert_eq!(denoising.max_denoising_steps, 48);
    assert_eq!(denoising.entropy_bound, 0.1);
    assert_eq!((denoising.t_min, denoising.t_max), (0.4, 0.8));
    assert_eq!(denoising.confidence_threshold, 0.005);
    assert_eq!(denoising.stability_threshold, 1);
    assert_eq!(profile.quantized, quantized);

    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../tests/python/fixtures/diffusion_gemma_readout_prompts.json"
    ))
    .unwrap();
    assert_eq!(fixture.canvas_length, profile.canvas_length as usize);
    for case in fixture.cases {
        let options = ReadoutOptions {
            layout: case.layout.parse::<ReadoutLayout>().unwrap(),
            canvas: case.canvas.parse::<CanvasMode>().unwrap(),
        };
        let encoder =
            ReadoutEncoder::load(&files, Arc::clone(&tokenizer), profile, options, 262_144)
                .unwrap();
        let request = SystemOneRequest::from_value(&case.request).unwrap();
        let sizes: Vec<ImageSize> = case
            .images
            .iter()
            .map(|size| ImageSize {
                width: size.width,
                height: size.height,
            })
            .collect();

        let plan = encoder.plan(&request, &sizes).unwrap();

        assert_plan(&case, &plan, tokens.pad);
    }
}

fn assert_plan(case: &Case, plan: &ReadoutPlan, pad: u32) {
    let name = case.name.as_str();
    assert_eq!(
        plan.prompts.len(),
        case.prompts.len(),
        "{name}: prompt count"
    );
    for (index, (actual, expected)) in plan.prompts.iter().zip(&case.prompts).enumerate() {
        assert_eq!(
            actual.token_ids, expected.token_ids,
            "{name}: prompt {index} tokens"
        );
        let placements: Vec<ImagePlacement> = expected
            .images
            .iter()
            .map(|image| ImagePlacement {
                source: image.source,
                offset: image.offset,
                soft_tokens: image.soft_tokens,
            })
            .collect();
        assert_eq!(actual.images, placements, "{name}: prompt {index} images");

        assert_eq!(
            actual.rows.len(),
            expected.rows.len(),
            "{name}: prompt {index} rows"
        );
        let mut expected_tokens: Vec<Vec<u32>> = Vec::new();
        for (row_index, (row, golden)) in actual.rows.iter().zip(&expected.rows).enumerate() {
            let tokens = match (&golden.token_ids, golden.like_row, &golden.set) {
                (Some(tokens), _, _) => {
                    let mut tokens = tokens.clone();
                    tokens.resize(golden.length.unwrap(), pad);
                    tokens
                }
                (None, Some(base), Some(set)) => {
                    let mut tokens = expected_tokens[base].clone();
                    tokens[set.position] = set.token;
                    tokens
                }
                _ => unreachable!("{name}: malformed golden row"),
            };
            assert_eq!(
                row.token_ids, tokens,
                "{name}: prompt {index} row {row_index} tokens"
            );
            let slots: Vec<(u32, &[u32])> = golden
                .slots
                .iter()
                .map(|slot| (slot.position, slot.candidates.as_slice()))
                .collect();
            let actual_slots: Vec<(u32, &[u32])> = row
                .slots
                .iter()
                .map(|slot| (slot.position, slot.candidates.as_slice()))
                .collect();
            assert_eq!(
                actual_slots, slots,
                "{name}: prompt {index} row {row_index} slots"
            );
            expected_tokens.push(tokens);
        }
    }
    assert_eq!(
        plan.shared_prefix_tokens, case.shared_prefix_tokens,
        "{name}: shared prefix"
    );
    assert_eq!(plan.input_tokens, case.input_tokens, "{name}: input tokens");
}
