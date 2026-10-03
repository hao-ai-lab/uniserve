//! Answer-token vocabulary of the readout canvas, verified at startup.
//!
//! A readout canvas answers question `i` on the line `"{i}:" <slot> "\n"`,
//! where the slot is one masked token (or two for a two-letter label) whose
//! predicted distribution over answer tokens is the answer. The answer tokens
//! must therefore be single vocabulary tokens that tokenize the same inside
//! the scaffold as on their own; [`Vocabulary::resolve`] checks this for the
//! served tokenizer before any request is planned.

use crate::profile::tokenizer::HuggingFaceTokenizer;
use crate::serving::systemone::prompt::{LABELS, SECOND_LETTERS};

/// Spellings whose probabilities sum into a yes answer, in the reference order.
const YES_SPELLINGS: [&str; 7] = [" yes", "yes", " Yes", "Yes", " YES", " true", " True"];
/// Spellings whose probabilities sum into a no answer, in the reference order.
const NO_SPELLINGS: [&str; 7] = [" no", "no", " No", "No", " NO", " false", " False"];
/// First letters of two-letter labels: A-J, enough for 260 options.
const FIRST_LETTERS: usize = 10;

/// Verified answer tokens of the served tokenizer.
#[derive(Debug, Clone)]
pub(super) struct Vocabulary {
    /// Token of `" X"` for each label `X` of [`LABELS`].
    pub(super) labels: Vec<u32>,
    /// Per label, the distinct single-token spellings of `" X"` and `"X"`,
    /// whose probabilities sum into the label's answer.
    pub(super) label_variants: Vec<Vec<u32>>,
    /// Distinct single-token yes spellings.
    pub(super) yes_variants: Vec<u32>,
    /// Distinct single-token no spellings.
    pub(super) no_variants: Vec<u32>,
    /// Tokens of the line break closing a scaffold line.
    pub(super) newline: Vec<u32>,
    /// Tokens of the empty thinking channel `<|channel>thought\n<channel|>`
    /// that opens a Gemma-4 reply when thinking is off. It ends every prompt,
    /// so the first canvas token is the first answer line.
    pub(super) thought_prefix: Vec<u32>,
}

impl Vocabulary {
    /// Resolves and verifies the answer tokens.
    ///
    /// Every label `" X"` (A-Z, a-z), `" yes"`, and `" no"` must be one token.
    /// The joint scaffold `"1: A\n2: no\n"` must tokenize as its pieces, and
    /// every two-letter label `" X Y"` (X in A-J, Y in A-Z) must tokenize as
    /// `[" X", " Y"]`, also inside `"1: X Y\n"`.
    ///
    /// # Errors
    ///
    /// Returns a message naming the first violated condition.
    pub(super) fn resolve(tokenizer: &HuggingFaceTokenizer) -> Result<Self, String> {
        let encode = |text: &str| {
            tokenizer
                .encode(text, false)
                .map_err(|error| format!("cannot tokenize {text:?}: {error}"))
        };
        let single = |text: &str| -> Result<u32, String> {
            match encode(text)?.as_slice() {
                [token] => Ok(*token),
                tokens => Err(format!("{text:?} is not a single token: {tokens:?}")),
            }
        };
        // Distinct single-token ids among the spellings, in spelling order.
        let variants = |spellings: &[&str]| -> Result<Vec<u32>, String> {
            let mut tokens = Vec::new();
            for spelling in spellings {
                if let [token] = encode(spelling)?.as_slice()
                    && !tokens.contains(token)
                {
                    tokens.push(*token);
                }
            }
            if tokens.is_empty() {
                return Err(format!("no single-token spelling among {spellings:?}"));
            }
            Ok(tokens)
        };

        let labels = LABELS
            .iter()
            .map(|label| single(&format!(" {label}")))
            .collect::<Result<Vec<_>, _>>()?;
        let label_variants = LABELS
            .iter()
            .map(|label| variants(&[&format!(" {label}"), &label.to_string()]))
            .collect::<Result<Vec<_>, _>>()?;
        let no = single(" no")?;
        single(" yes")?;
        let vocabulary = Self {
            labels,
            label_variants,
            yes_variants: variants(&YES_SPELLINGS)?,
            no_variants: variants(&NO_SPELLINGS)?,
            newline: encode("\n")?,
            thought_prefix: encode("<|channel>thought\n<channel|>")?,
        };

        let number = |i: usize| encode(&format!("{i}:"));
        let joint = encode("1: A\n2: no\n")?;
        let pieces = [
            number(1)?,
            vec![vocabulary.labels[0]],
            vocabulary.newline.clone(),
            number(2)?,
            vec![no],
            vocabulary.newline.clone(),
        ]
        .concat();
        if joint != pieces {
            return Err(format!(
                "scaffold tokenization mismatch: {joint:?} != {pieces:?}"
            ));
        }

        let letters = LABELS.iter().zip(&vocabulary.labels);
        for (x, first) in letters.clone().take(FIRST_LETTERS) {
            for (y, second) in letters.clone().take(SECOND_LETTERS) {
                let expected = [*first, *second];
                let pair = encode(&format!(" {x} {y}"))?;
                if pair != expected {
                    return Err(format!(
                        "label \" {x} {y}\" tokenizes as {pair:?}, not {expected:?}"
                    ));
                }
                let line = encode(&format!("1: {x} {y}\n"))?;
                let pieces = [number(1)?, expected.to_vec(), vocabulary.newline.clone()].concat();
                if line != pieces {
                    return Err(format!(
                        "scaffold line \"1: {x} {y}\" tokenizes as {line:?}, not {pieces:?}"
                    ));
                }
            }
        }
        Ok(vocabulary)
    }
}
