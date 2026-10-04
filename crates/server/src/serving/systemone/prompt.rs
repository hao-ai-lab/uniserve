//! Readout prompt text, prompt format 1.
//!
//! The text is the reference encoder's (DJev `Encoder.encode`) character for
//! character for every request it accepts:
//!
//! ```text
//! <PREAMBLE>
//! [N image(s) are attached above, in order: Image 1, ..., Image N.]
//!
//! State:
//! <state>
//!
//! Question 1 (yes/no): <instructions>
//! Answer "yes" if: <true criterion>
//! Answer "no" if: <false criterion>
//!
//! Question 2 (choice): <instructions>
//! Options:
//! A) <name>: <description>
//!
//! Question 3 (score): <instructions>
//! Levels, ordered from lowest to highest:
//! A) <level>
//!
//! Answer format (one line per question):
//! 1: <yes or no>
//! 2: <one option letter, A-B>
//! 3: <one level letter, A-C>
//! ```
//!
//! The official question types map onto the reference blocks: `noul` is the
//! yes/no block, `choice` and `score` keep their names. Non-string values are
//! written as Python `json.dumps(value, ensure_ascii=False)`. The official
//! schema also admits inputs the reference refuses, written as follows:
//!
//! - absent or `null` instructions drop the `: <instructions>` suffix of the
//!   question line (`Question 1 (yes/no)`);
//! - a `null` noul criterion drops its `Answer ... if:` line;
//! - a `null` option description writes the option line as `A) <name>`;
//! - a single option or level is hinted as `<one option letter, A>`;
//! - a choice with more than 52 options uses two-letter labels, `A A` to
//!   `J Z` (first letter A-J, second A-Z, separated by a space), written as
//!   `A A) <name>: <description>` and hinted as
//!   `<one option label, A A to C A>` with the question's last label.

use serde_json::Value;

use crate::serving::systemone::python::json_dumps;
use crate::serving::systemone::request::{Criteria, Question};

/// Prompt format version: the text, template, and canvas rules of this
/// module and of the canvas scaffold. Any change to them is a new version.
pub const PROMPT_FORMAT: u32 = 1;

/// The fixed instruction that opens every readout prompt.
pub(super) const PREAMBLE: &str = "You are a decision model. Read the state, then answer every \
     question below. Judge each question independently, using only the state and that \
     question's own instructions. Reply with exactly one line per question in the answer format \
     given at the end, and nothing else.";

/// Option and level labels, in order; each `" X"` is one vocabulary token.
pub(super) const LABELS: [char; 52] = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
];

/// Letters per position of a two-letter label: the second letter runs A-Z.
pub(super) const SECOND_LETTERS: usize = 26;

/// Returns whether a choice with `options` options uses two-letter labels.
pub(super) fn uses_letter_pairs(options: usize) -> bool {
    options > LABELS.len()
}

/// Returns `value` as prompt text: a string as itself, any other JSON value
/// as Python's `json.dumps(value, ensure_ascii=False)`.
pub(super) fn value_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => json_dumps(other),
    }
}

/// Returns the label of option `index` among `options` options.
fn label(options: usize, index: usize) -> String {
    if uses_letter_pairs(options) {
        format!(
            "{} {}",
            LABELS[index / SECOND_LETTERS],
            LABELS[index % SECOND_LETTERS]
        )
    } else {
        LABELS[index].to_string()
    }
}

/// Returns the answer-format hint for a question with `count` options or levels.
fn label_hint(noun: &str, count: usize) -> String {
    if uses_letter_pairs(count) {
        format!("<one option label, A A to {}>", label(count, count - 1))
    } else if count == 1 {
        format!("<one {noun} letter, A>")
    } else {
        format!("<one {noun} letter, A-{}>", LABELS[count - 1])
    }
}

/// The text of one question within a prompt.
pub(super) struct QuestionText {
    /// The question's block lines, without the blank line that follows it.
    pub(super) block: Vec<String>,
    /// The question's answer-format line.
    pub(super) hint: String,
}

impl QuestionText {
    /// Renders `question` as question `number` (1-based) of its prompt.
    pub(super) fn new(number: usize, question: &Question) -> Self {
        let kind = match question.criteria {
            Criteria::Noul { .. } => "yes/no",
            Criteria::Choice(_) => "choice",
            Criteria::Score(_) => "score",
        };
        let mut block = vec![match &question.instructions {
            Some(instructions) => {
                format!("Question {number} ({kind}): {}", value_text(instructions))
            }
            None => format!("Question {number} ({kind})"),
        }];

        let hint = match &question.criteria {
            Criteria::Noul { yes, no } => {
                if let Some(yes) = yes {
                    block.push(format!("Answer \"yes\" if: {}", value_text(yes)));
                }
                if let Some(no) = no {
                    block.push(format!("Answer \"no\" if: {}", value_text(no)));
                }
                "<yes or no>".to_owned()
            }
            Criteria::Choice(options) => {
                block.push("Options:".to_owned());
                for (index, option) in options.iter().enumerate() {
                    let label = label(options.len(), index);
                    block.push(match &option.description {
                        Some(description) => {
                            format!("{label}) {}: {}", option.name, value_text(description))
                        }
                        None => format!("{label}) {}", option.name),
                    });
                }
                label_hint("option", options.len())
            }
            Criteria::Score(levels) => {
                block.push("Levels, ordered from lowest to highest:".to_owned());
                for (index, level) in levels.iter().enumerate() {
                    block.push(format!("{}) {}", LABELS[index], value_text(level)));
                }
                label_hint("level", levels.len())
            }
        };

        Self {
            block,
            hint: format!("{number}: {hint}"),
        }
    }
}

/// Returns the prompt lines every question of a request shares: the
/// preamble, the image line, and the state.
pub(super) fn shared_lines(state: &str, image_count: usize) -> Vec<String> {
    let mut lines = vec![PREAMBLE.to_owned()];
    if image_count > 0 {
        let images: Vec<String> = (1..=image_count).map(|i| format!("Image {i}")).collect();
        lines.push(format!(
            "{image_count} image(s) are attached above, in order: {}.",
            images.join(", ")
        ));
    }
    lines.extend([
        String::new(),
        "State:".to_owned(),
        state.to_owned(),
        String::new(),
    ]);
    lines
}

/// Joins the shared lines and one prompt's questions into the user message.
pub(super) fn prompt_text(shared: &[String], questions: &[QuestionText]) -> String {
    let mut lines = shared.to_vec();
    for question in questions {
        lines.extend(question.block.iter().cloned());
        lines.push(String::new());
    }
    lines.push("Answer format (one line per question):".to_owned());
    lines.extend(questions.iter().map(|question| question.hint.clone()));
    lines.join("\n")
}
