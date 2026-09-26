//! Readout planning and answer assembly over a synthetic checkpoint.
//!
//! The synthetic tokenizer maps each ASCII character to one token and merges
//! a space with a following letter (`" A"`), and `" yes"` and `" no"` into
//! single tokens. Its scaffold costs therefore equal the DiffusionGemma
//! tokenizer's (`"1:"` is two tokens, `"10:"` three), so canvas capacities
//! behave as in the served checkpoint. Token-exact prompts for the real
//! checkpoints are covered by the `systemone_checkpoint` integration test.

use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

use serde_json::{Value, json};
use tempfile::TempDir;
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};

use super::{
    CanvasMode, ImageSize, ReadoutEncoder, ReadoutLayout, ReadoutOptions, ReadoutPlan,
    StartupError, SystemOneError, SystemOneRequest,
};
use crate::profile::assets::ResolvedModelFiles;
use crate::profile::diffusion_gemma::{
    ControlTokens, DenoisingDefaults, DiffusionGemmaProfile, PatchBudget,
};
use crate::profile::tokenizer::{DynTokenizer, HuggingFaceTokenizer};

const SPECIAL_TOKENS: [&str; 11] = [
    "<pad>",
    "<eos>",
    "<bos>",
    "<mask>",
    "<|turn>",
    "<turn|>",
    "<|channel>",
    "<channel|>",
    "<|image>",
    "<|image|>",
    "<image|>",
];

/// A Gemma-4-shaped template: one user turn, images before text, and an open
/// model turn.
const TEMPLATE: &str = r"{{ bos_token }}{% for message in messages %}{{ '<|turn>user\n' }}{% if message['content'] is string %}{{ message['content'] | trim }}{% else %}{% for item in message['content'] %}{% if item['type'] == 'image' %}{{ '<|image|>' }}{% else %}{{ item['text'] | trim }}{% endif %}{% endfor %}{% endif %}{{ '<turn|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|turn>model\n' }}{% endif %}";

struct Checkpoint {
    _directory: TempDir,
    files: ResolvedModelFiles,
    tokenizer: DynTokenizer,
    profile: DiffusionGemmaProfile,
}

impl Checkpoint {
    /// Writes the synthetic checkpoint; `extra_merges` adds BPE merges after
    /// the answer-token merges.
    fn new(extra_merges: &[(&str, &str)]) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let mut vocab = Vocab::from_iter([("<unk>".to_owned(), 0_u32)]);
        for code in 1_u32..=127 {
            vocab.insert(char::from_u32(code).unwrap().to_string(), code);
        }
        let mut merges = Vec::new();
        let mut merge = |vocab: &mut Vocab, left: &str, right: &str| {
            let next = vocab.len() as u32;
            vocab.entry(format!("{left}{right}")).or_insert(next);
            merges.push((left.to_owned(), right.to_owned()));
        };
        for letter in ('A'..='Z').chain('a'..='z') {
            merge(&mut vocab, " ", &letter.to_string());
        }
        merge(&mut vocab, " y", "e");
        merge(&mut vocab, " ye", "s");
        merge(&mut vocab, " n", "o");
        for (left, right) in extra_merges {
            merge(&mut vocab, left, right);
        }
        let model = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .unk_token("<unk>".to_owned())
            .build()
            .unwrap();
        let mut builder = TokenizerBuilder::new(model);
        builder.add_special_tokens(
            &SPECIAL_TOKENS
                .iter()
                .map(|token| AddedToken::from(*token, true))
                .collect::<Vec<_>>(),
        );
        let tokenizer_path = directory.path().join("tokenizer.json");
        builder.save(&tokenizer_path, false).unwrap();
        let tokenizer_config_path = directory.path().join("tokenizer_config.json");
        fs::write(
            &tokenizer_config_path,
            json!({ "bos_token": "<bos>", "eos_token": "<eos>", "chat_template": TEMPLATE })
                .to_string(),
        )
        .unwrap();

        let tokenizer = Arc::new(HuggingFaceTokenizer::new(&tokenizer_path).unwrap());
        let id = |token: &str| tokenizer.token_to_id(token).unwrap();
        let profile = DiffusionGemmaProfile {
            canvas_length: 256,
            tokens: ControlTokens {
                mask: id("<mask>"),
                pad: id("<pad>"),
                turn_end: id("<turn|>"),
                image: id("<|image|>"),
                image_start: id("<|image>"),
                image_end: id("<image|>"),
            },
            images: PatchBudget {
                patch_size: 16,
                pooling_kernel_size: 3,
                max_soft_tokens: 280,
            },
            denoising: DenoisingDefaults {
                max_denoising_steps: 48,
                entropy_bound: 0.1,
                t_min: 0.4,
                t_max: 0.8,
                confidence_threshold: 0.005,
                stability_threshold: 1,
            },
            quantized: false,
        };
        let files = ResolvedModelFiles {
            tokenizer_path,
            tokenizer_config_path: Some(tokenizer_config_path),
            generation_config_path: None,
            preprocessor_config_path: None,
            chat_template_path: None,
            config_path: None,
        };
        Self {
            _directory: directory,
            files,
            tokenizer,
            profile,
        }
    }

    fn encoder(&self, options: ReadoutOptions, max_model_len: u32) -> ReadoutEncoder {
        ReadoutEncoder::load(
            &self.files,
            Arc::clone(&self.tokenizer),
            &self.profile,
            options,
            max_model_len,
        )
        .unwrap()
    }

    fn token(&self, text: &str) -> u32 {
        match self.tokenizer.encode(text, false).unwrap().as_slice() {
            [token] => *token,
            tokens => panic!("{text:?} is {tokens:?}"),
        }
    }
}

fn request(questions: Value) -> SystemOneRequest {
    request_with(json!({ "model": "m", "state": "A support ticket.", "questions": questions }))
}

fn request_with(body: Value) -> SystemOneRequest {
    SystemOneRequest::from_json(&serde_json::to_vec(&body).unwrap()).unwrap()
}

fn noul(instructions: &str) -> Value {
    json!({ "type": "noul", "instructions": instructions })
}

fn choice(options: usize) -> Value {
    let criteria: serde_json::Map<String, Value> = (0..options)
        .map(|option| (format!("o{option}"), json!(format!("Option {option}"))))
        .collect();
    json!({ "type": "choice", "instructions": "Pick one.", "criteria": criteria })
}

/// Log-probabilities for every planned candidate, from a function of the
/// prompt index, the row index, the slot's canvas position, and the candidate token.
fn logprobs(plan: &ReadoutPlan, probability: impl Fn(usize, usize, u32, u32) -> f64) -> Vec<f32> {
    let mut out = Vec::with_capacity(plan.candidate_count());
    for (prompt_index, prompt) in plan.prompts.iter().enumerate() {
        for (row_index, row) in prompt.rows.iter().enumerate() {
            for slot in &row.slots {
                for &candidate in &slot.candidates {
                    out.push(
                        probability(prompt_index, row_index, slot.position, candidate).ln() as f32,
                    );
                }
            }
        }
    }
    out
}

fn assert_close(actual: &Value, expected: f64) {
    let actual = actual.as_f64().unwrap();
    assert!(
        (actual - expected).abs() < 1e-6,
        "{actual} differs from {expected}"
    );
}

#[test]
fn answers_follow_the_candidate_probability_math() {
    let checkpoint = Checkpoint::new(&[]);
    let encoder = checkpoint.encoder(ReadoutOptions::default(), 4096);
    let plan = encoder
        .plan(
            &request(json!({
                "spam": noul("Is this spam?"),
                "team": {"type": "choice", "instructions": "Which team?",
                         "criteria": {"billing": "Payments", "technical": null, "sales": ["pricing"]}},
                "urgency": {"type": "score", "criteria": ["Can wait", {"level": "soon"}, "Now"]},
            })),
            &[],
        )
        .unwrap();

    // Spellings: " no"/" yes" are the only single-token yes/no forms here,
    // and each label sums " X" and "X".
    let table: HashMap<u32, f64> = [
        (" no", 0.2),
        (" yes", 0.6),
        (" A", 0.3),
        ("A", 0.1),
        (" B", 0.2),
        ("B", 0.0),
        (" C", 0.1),
        ("C", 0.1),
    ]
    .into_iter()
    .map(|(text, probability)| (checkpoint.token(text), probability))
    .collect();
    let slots = &plan.prompts[0].rows[0].slots;
    let values = logprobs(&plan, |_, _, position, token| {
        // The score slot is the third; its labels have their own probabilities.
        if position == slots[2].position {
            let score: HashMap<u32, f64> = [
                (" A", 0.1),
                ("A", 0.1),
                (" B", 0.2),
                ("B", 0.0),
                (" C", 0.5),
                ("C", 0.1),
            ]
            .into_iter()
            .map(|(text, probability)| (checkpoint.token(text), probability))
            .collect();
            score[&token]
        } else {
            table[&token]
        }
    });
    let response = serde_json::to_value(plan.assemble("m", &values).unwrap()).unwrap();

    assert_eq!(response["model"], "m");
    assert_eq!(
        response["usage"]["input_tokens"],
        plan.prompts[0].token_ids.len()
    );
    assert_eq!(response["usage"]["output_tokens"], 0);
    let answers = response["answers"].as_object().unwrap();
    assert_eq!(
        answers.keys().collect::<Vec<_>>(),
        ["spam", "team", "urgency"]
    );

    // noul = P(yes) / (P(yes) + P(no)).
    assert_eq!(answers["spam"]["type"], "noul");
    assert_close(&answers["spam"]["noul"], 0.75);
    assert_close(&answers["spam"]["x_candidate_mass"], 0.8);
    assert_eq!(answers["spam"].as_object().unwrap().len(), 3);

    // Per-option mass 0.4, 0.2, 0.2 normalizes to 0.5, 0.25, 0.25.
    let team = &answers["team"];
    assert_eq!(team["type"], "choice");
    assert_eq!(team["choice"], "billing");
    assert_eq!(
        team["probabilities"]
            .as_object()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        ["billing", "technical", "sales"]
    );
    assert_close(&team["probabilities"]["billing"], 0.5);
    assert_close(&team["probabilities"]["technical"], 0.25);
    assert_close(&team["probabilities"]["sales"], 0.25);
    assert_close(&team["confidence"], (3.0 * 0.5 - 1.0) / 2.0);
    assert_close(&team["x_candidate_mass"], 0.8);

    // Per-level mass 0.2, 0.2, 0.6: score = 0.2 + 2 * 0.6.
    let urgency = &answers["urgency"];
    assert_eq!(urgency["type"], "score");
    assert_close(&urgency["score"], 1.4);
    assert_eq!(
        urgency["legend"],
        json!({"0": "Can wait", "1": {"level": "soon"}, "2": "Now"})
    );
    assert_close(&urgency["probabilities"]["2"], 0.6);
    assert_close(&urgency["confidence"], (3.0 * 0.6 - 1.0) / 2.0);
    assert_close(&urgency["x_candidate_mass"], 1.0);
}

#[test]
fn a_single_option_or_level_is_fully_confident() {
    let checkpoint = Checkpoint::new(&[]);
    let encoder = checkpoint.encoder(ReadoutOptions::default(), 4096);
    let plan = encoder
        .plan(
            &request(json!({
                "one": {"type": "choice", "criteria": {"only": null}},
                "flat": {"type": "score", "criteria": ["flat"]},
            })),
            &[],
        )
        .unwrap();
    let values = logprobs(&plan, |_, _, _, _| 0.01);
    let response = serde_json::to_value(plan.assemble("m", &values).unwrap()).unwrap();

    assert_eq!(response["answers"]["one"]["choice"], "only");
    assert_close(&response["answers"]["one"]["probabilities"]["only"], 1.0);
    assert_close(&response["answers"]["one"]["confidence"], 1.0);
    assert_close(&response["answers"]["one"]["x_candidate_mass"], 0.02);
    assert_close(&response["answers"]["flat"]["score"], 0.0);
    assert_close(&response["answers"]["flat"]["confidence"], 1.0);
}

#[test]
fn choices_beyond_52_options_read_two_letters_left_to_right() {
    let checkpoint = Checkpoint::new(&[]);
    let encoder = checkpoint.encoder(ReadoutOptions::default(), 8192);
    let plan = encoder
        .plan(
            &request(json!({"code": choice(53), "spam": noul("Spam?")})),
            &[],
        )
        .unwrap();

    // The first row reads the first letters A-C of the 53 labels (A A..A Z,
    // B A..B Z, C A) and the noul slot; one row per first letter follows.
    let letter = |text: &str| checkpoint.token(text);
    let rows = &plan.prompts[0].rows;
    assert_eq!(rows.len(), 4);
    let first_slot = rows[0].slots[0].position;
    assert_eq!(
        rows[0].slots[0].candidates,
        [letter(" A"), letter(" B"), letter(" C")]
    );
    assert_eq!(
        rows[0].token_ids[first_slot as usize],
        checkpoint.profile.tokens.mask
    );
    assert_eq!(
        rows[0].token_ids[first_slot as usize + 1],
        checkpoint.profile.tokens.mask
    );
    let second_letters: Vec<u32> = ('A'..='Z').map(|y| letter(&format!(" {y}"))).collect();
    for (row, (first, count)) in rows[1..].iter().zip([(" A", 26), (" B", 26), (" C", 1)]) {
        let mut expected = rows[0].token_ids.clone();
        expected[first_slot as usize] = letter(first);
        assert_eq!(row.token_ids, expected, "row conditioned on {first}");
        assert_eq!(row.slots.len(), 1);
        assert_eq!(row.slots[0].position, first_slot + 1);
        assert_eq!(row.slots[0].candidates, second_letters[..count]);
    }

    // P(A)=0.5, P(B)=0.3, P(C)=0.1; given A, " B" has 0.5 and every other
    // second letter 1/64; given B every second letter has 1/64; given C, 0.5.
    let values = logprobs(&plan, |_, row, position, token| match row {
        0 if position == first_slot => {
            if token == letter(" A") {
                0.5
            } else if token == letter(" B") {
                0.3
            } else {
                0.1
            }
        }
        0 => 0.5,
        1 if token == letter(" B") => 0.5,
        1 | 2 => 1.0 / 64.0,
        _ => 0.5,
    });
    let response = serde_json::to_value(plan.assemble("m", &values).unwrap()).unwrap();
    let code = &response["answers"]["code"];

    let mass = 0.5 * 0.5 + 25.0 * 0.5 / 64.0 + 26.0 * 0.3 / 64.0 + 0.1 * 0.5;
    assert_eq!(code["choice"], "o1");
    assert_close(&code["x_candidate_mass"], mass);
    assert_close(&code["probabilities"]["o1"], 0.25 / mass);
    assert_close(&code["probabilities"]["o0"], 0.5 / 64.0 / mass);
    assert_close(&code["probabilities"]["o30"], 0.3 / 64.0 / mass);
    assert_close(&code["probabilities"]["o52"], 0.05 / mass);
    assert_close(&code["confidence"], (53.0 * 0.25 / mass - 1.0) / 52.0);
    assert_eq!(code["probabilities"].as_object().unwrap().len(), 53);
    assert_close(&response["answers"]["spam"]["noul"], 0.5);
}

#[test]
fn joint_canvases_split_at_their_token_capacity() {
    let checkpoint = Checkpoint::new(&[]);
    let encoder = checkpoint.encoder(ReadoutOptions::default(), 65_536);
    let turn_end = checkpoint.profile.tokens.turn_end;
    let questions = |count: usize, two_letter: usize| {
        let entries: serde_json::Map<String, Value> = (0..count)
            .map(|index| {
                let question = if index < two_letter {
                    choice(53)
                } else {
                    noul(&format!("Item {index}?"))
                };
                (format!("q{index}"), question)
            })
            .collect();
        request(Value::Object(entries))
    };
    // Canvas questions per prompt, and the canvas length its scaffold uses.
    let layout = |plan: &ReadoutPlan| -> Vec<(usize, usize)> {
        plan.prompts
            .iter()
            .map(|prompt| {
                let row = &prompt.rows[0];
                let used = row.token_ids.iter().position(|t| *t == turn_end).unwrap() + 1;
                (row.slots.len(), used)
            })
            .collect()
    };

    // 52 one-slot questions take 252 tokens with <turn|>; the 53rd would need 257.
    assert_eq!(
        layout(&encoder.plan(&questions(52, 0), &[]).unwrap()),
        [(52, 252)]
    );
    assert_eq!(
        layout(&encoder.plan(&questions(53, 0), &[]).unwrap()),
        [(52, 252), (1, 5)]
    );
    // Each two-letter question takes one more token: with four, 52 questions
    // fill all 256 tokens; with five, the first 51 take 252 and the 52nd
    // would need 257.
    assert_eq!(
        layout(&encoder.plan(&questions(52, 4), &[]).unwrap()),
        [(52, 256)]
    );
    assert_eq!(
        layout(&encoder.plan(&questions(52, 5), &[]).unwrap()),
        [(51, 252), (1, 5)]
    );

    // Answers stay keyed by question id across canvases, and the prompts'
    // shared prefix is counted once.
    let plan = encoder.plan(&questions(53, 0), &[]).unwrap();
    let yes = checkpoint.token(" yes");
    let values = logprobs(&plan, |prompt, _, position, token| {
        let question = if prompt == 0 {
            plan.prompts[0].rows[0]
                .slots
                .iter()
                .position(|slot| slot.position == position)
                .unwrap()
        } else {
            52
        };
        let yes_share = (question + 1) as f64 / 100.0;
        if token == yes {
            yes_share
        } else {
            1.0 - yes_share
        }
    });
    let response = serde_json::to_value(plan.assemble("m", &values).unwrap()).unwrap();
    for question in [0, 17, 51, 52] {
        assert_close(
            &response["answers"][format!("q{question}")]["noul"],
            (question + 1) as f64 / 100.0,
        );
    }
    let [first, second] = [&plan.prompts[0].token_ids, &plan.prompts[1].token_ids];
    let shared = plan.shared_prefix_tokens as usize;
    assert_eq!(first[..shared], second[..shared]);
    assert_ne!(first[shared], second[shared]);
    assert_eq!(
        plan.input_tokens as usize,
        first.len() + second.len() - shared
    );
    assert_eq!(response["usage"]["input_tokens"], plan.input_tokens);
}

#[test]
fn independent_layout_gives_every_question_its_own_prompt() {
    let checkpoint = Checkpoint::new(&[]);
    let options = ReadoutOptions {
        layout: ReadoutLayout::Independent,
        canvas: CanvasMode::Full,
    };
    let encoder = checkpoint.encoder(options, 4096);
    let plan = encoder
        .plan(
            &request(json!({"a": noul("A?"), "b": choice(2), "c": noul("C?")})),
            &[],
        )
        .unwrap();
    let joint = checkpoint
        .encoder(ReadoutOptions::default(), 4096)
        .plan(&request(json!({"b": choice(2)})), &[])
        .unwrap();

    assert_eq!(plan.prompts.len(), 3);
    // A question's prompt and canvas do not depend on the other questions.
    assert_eq!(plan.prompts[1], joint.prompts[0]);
    for prompt in &plan.prompts {
        assert_eq!(prompt.rows[0].slots.len(), 1);
        assert_eq!(prompt.rows[0].token_ids.len(), 256);
    }
}

#[test]
fn compact_canvases_end_at_the_next_multiple_of_16() {
    let checkpoint = Checkpoint::new(&[]);
    let options = ReadoutOptions {
        layout: ReadoutLayout::Joint,
        canvas: CanvasMode::Compact,
    };
    let encoder = checkpoint.encoder(options, 4096);
    let tokens = checkpoint.profile.tokens;

    // Three questions use 13 scaffold tokens with <turn|>, four use 17.
    for (count, length) in [(3, 16), (4, 32)] {
        let questions: serde_json::Map<String, Value> =
            (0..count).map(|i| (format!("q{i}"), noul("?"))).collect();
        let plan = encoder
            .plan(&request(Value::Object(questions)), &[])
            .unwrap();
        let canvas = &plan.prompts[0].rows[0].token_ids;
        assert_eq!(canvas.len(), length);
        assert_eq!(canvas[4 * count], tokens.turn_end);
        assert!(
            canvas[4 * count + 1..]
                .iter()
                .all(|token| *token == tokens.pad)
        );
    }
}

#[test]
fn images_are_placed_before_the_text_with_their_soft_tokens() {
    let checkpoint = Checkpoint::new(&[]);
    let encoder = checkpoint.encoder(ReadoutOptions::default(), 8192);
    let tokens = checkpoint.profile.tokens;
    let request = request_with(json!({
        "model": "m",
        "state": "Compare the images.",
        "questions": {"same": noul("Same scene?")},
        "x_images": ["https://example.com/1.png", "data:image/png;base64,AAAA"],
    }));
    let plan = encoder
        .plan(
            &request,
            &[
                ImageSize {
                    width: 640,
                    height: 480,
                },
                ImageSize {
                    width: 300,
                    height: 1200,
                },
            ],
        )
        .unwrap();

    let prompt = &plan.prompts[0];
    assert_eq!(prompt.images.len(), 2);
    for (index, (placement, soft_tokens)) in prompt.images.iter().zip([266, 264]).enumerate() {
        let offset = placement.offset as usize;
        assert_eq!(placement.source, index);
        assert_eq!(placement.soft_tokens, soft_tokens);
        assert_eq!(prompt.token_ids[offset - 1], tokens.image_start);
        assert!(
            prompt.token_ids[offset..offset + soft_tokens as usize]
                .iter()
                .all(|token| *token == tokens.image)
        );
        assert_eq!(
            prompt.token_ids[offset + soft_tokens as usize],
            tokens.image_end
        );
    }
    // Both images directly follow the opening of the user turn.
    assert_eq!(prompt.images[1].offset, prompt.images[0].offset + 266 + 2);
    let image_tokens = prompt
        .token_ids
        .iter()
        .filter(|t| **t == tokens.image)
        .count();
    assert_eq!(image_tokens, 266 + 264);
    assert_eq!(plan.input_tokens as usize, prompt.token_ids.len());
}

#[test]
fn request_text_cannot_smuggle_image_placeholders() {
    let checkpoint = Checkpoint::new(&[]);
    let encoder = checkpoint.encoder(ReadoutOptions::default(), 4096);
    let request = request_with(json!({
        "model": "m",
        "state": "Look at <|image|> here.",
        "questions": {"a": noul("?")},
    }));

    let error = encoder.plan(&request, &[]).unwrap_err();

    assert_eq!(error.status_code().as_u16(), 422);
    assert_eq!(error.body()["detail"][0]["loc"], json!(["body"]));
}

#[test]
fn encoded_requests_beyond_their_limits_are_rejected() {
    let checkpoint = Checkpoint::new(&[]);
    // Request context min(64k, 700) = 700; state plus longest question at most 350.
    let encoder = checkpoint.encoder(ReadoutOptions::default(), 700);
    let plan = |state: &str, questions: Value| {
        encoder.plan(
            &request_with(json!({"model": "m", "state": state, "questions": questions})),
            &[],
        )
    };
    let detail = |error: SystemOneError| {
        assert_eq!(error.status_code().as_u16(), 422);
        error.body()["detail"][0].clone()
    };

    // One character is one token: a 330-token state leaves too little for a
    // question of more than 20 tokens.
    let issue = detail(
        plan(
            &"s".repeat(330),
            json!({"short": noul("?"), "long": noul(&"q".repeat(40))}),
        )
        .unwrap_err(),
    );
    assert_eq!(issue["loc"], json!(["body", "questions", "long"]));
    assert_eq!(issue["type"], "value_error");

    // A small state with many questions exceeds the 700-token request context.
    let many: serde_json::Map<String, Value> =
        (0..40).map(|i| (format!("q{i}"), noul("Is it?"))).collect();
    let issue = detail(plan("s", Value::Object(many)).unwrap_err());
    assert_eq!(issue["loc"], json!(["body"]));

    // A prompt within the request context still needs room for its 256-token canvas.
    let issue = detail(plan(&"s".repeat(300), json!({"a": noul("?")})).unwrap_err());
    assert_eq!(issue["loc"], json!(["body"]));
    assert!(plan(&"s".repeat(100), json!({"a": noul("?")})).is_ok());
}

#[test]
fn a_question_that_cannot_fit_an_empty_canvas_is_rejected() {
    let mut checkpoint = Checkpoint::new(&[]);
    // "1:", one slot, "\n", and <turn|> fill a 5-token canvas exactly; a
    // two-letter question needs one token more.
    checkpoint.profile.canvas_length = 5;
    let encoder = checkpoint.encoder(ReadoutOptions::default(), 4096);

    assert!(encoder.plan(&request(json!({"a": noul("?")})), &[]).is_ok());
    let error = encoder
        .plan(&request(json!({"a": noul("?"), "wide": choice(53)})), &[])
        .unwrap_err();

    assert_eq!(error.status_code().as_u16(), 422);
    assert_eq!(
        error.body()["detail"][0]["loc"],
        json!(["body", "questions", "wide"])
    );
}

#[test]
fn an_image_without_area_is_rejected() {
    let checkpoint = Checkpoint::new(&[]);
    let encoder = checkpoint.encoder(ReadoutOptions::default(), 4096);
    let request = request_with(json!({
        "model": "m", "state": "s", "questions": {"a": noul("?")},
        "x_images": ["https://example.com/1.png"],
    }));

    let error = encoder
        .plan(
            &request,
            &[ImageSize {
                width: 0,
                height: 10,
            }],
        )
        .unwrap_err();

    assert_eq!(
        error.body()["detail"][0]["loc"],
        json!(["body", "x_images", 0])
    );
}

#[test]
fn assembly_refuses_malformed_readouts() {
    let checkpoint = Checkpoint::new(&[]);
    let encoder = checkpoint.encoder(ReadoutOptions::default(), 4096);
    let plan = encoder
        .plan(&request(json!({"a": noul("?")})), &[])
        .unwrap();

    let short = plan.assemble("m", &[]).unwrap_err();
    let nan = plan
        .assemble("m", &vec![f32::NAN; plan.candidate_count()])
        .unwrap_err();

    assert_eq!(short.status_code().as_u16(), 500);
    assert_eq!(nan.status_code().as_u16(), 500);
}

#[test]
fn a_tokenizer_that_merges_two_letter_labels_is_refused_at_startup() {
    let checkpoint = Checkpoint::new(&[(" C", " D")]);

    let error = ReadoutEncoder::load(
        &checkpoint.files,
        Arc::clone(&checkpoint.tokenizer),
        &checkpoint.profile,
        ReadoutOptions::default(),
        4096,
    )
    .err()
    .unwrap();

    assert!(
        matches!(&error, StartupError::Contract(message) if message.contains("\" C D\"")),
        "{error}"
    );
}
