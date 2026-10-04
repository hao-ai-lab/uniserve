//! `POST /v1/systemone` request types and validation.
//!
//! [`SystemOneRequest::from_json`] accepts exactly the official System One
//! 0.2.0 request schema plus the `x_images` extension, and reports every
//! violation the way the official FastAPI service does: pydantic v2 error
//! types, messages, and locations, collected in schema order.
//!
//! The official schema leaves several limits to prose; UniServe states them
//! with pydantic's own constructs:
//!
//! - unknown top-level fields are ignored, but unknown keys inside a question
//!   or inside noul criteria are `extra_forbidden`, since a misspelled key
//!   would silently ask a different question;
//! - a choice has 1-255 options (`too_short`/`too_long`) whose names are not
//!   blank (a `value_error` on `criteria`); JSON object keys are distinct by
//!   construction, a repeated key keeping its last value as Python does;
//! - a score has 1-10 levels (`too_short`/`too_long`);
//! - a noul question needs non-null `instructions` or a non-null `true` or
//!   `false` criterion (a `value_error` on the question);
//! - `x_images` holds 1-8 strings, each an `http(s)` URL or a
//!   `data:image/...;base64,` URI (a `value_error` per item).
//!
//! Errors inside a list stop at its maximum length, as pydantic's list
//! validator does: a list longer than its bound reports only `too_long`. A map
//! reports its items' errors before its length bounds.

use serde_json::{Map, Value, json};

use crate::serving::systemone::error::{LocItem, SystemOneError, ValidationIssue};
use crate::serving::systemone::python::{python_space, python_str};

/// Most options of one choice question.
pub const MAX_CHOICE_OPTIONS: usize = 255;
/// Most levels of one score question.
pub const MAX_SCORE_LEVELS: usize = 10;
/// Most images of one request (`x_images`).
pub const MAX_IMAGES: usize = 8;

/// A validated System One request.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemOneRequest {
    /// The content every question refers to: a JSON string, object, or array.
    pub state: Value,
    /// Requested model name.
    pub model: String,
    /// Questions in request order.
    pub questions: Vec<Question>,
    /// Image sources (`x_images`) in prompt order: `http(s)` URLs or
    /// `data:image/...;base64,` URIs. Empty when the request carries none.
    pub images: Vec<String>,
}

/// One named question.
#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    /// The caller's name for the question; its answer is returned under it.
    pub id: String,
    /// What to decide: a JSON string, object, or array, or absent.
    pub instructions: Option<Value>,
    /// The question type and its criteria.
    pub criteria: Criteria,
}

/// A question type with its criteria.
#[derive(Debug, Clone, PartialEq)]
pub enum Criteria {
    /// A yes/no question answered with the probability of yes.
    Noul {
        /// What counts as a yes answer, when described.
        yes: Option<Value>,
        /// What counts as a no answer, when described.
        no: Option<Value>,
    },
    /// Selection of one of 1-255 options, in request order.
    Choice(Vec<ChoiceOption>),
    /// A rating on 1-10 ordered levels, lowest first; each level is a JSON
    /// string, object, or array.
    Score(Vec<Value>),
}

/// One option of a choice question.
#[derive(Debug, Clone, PartialEq)]
pub struct ChoiceOption {
    /// Option name, returned as the chosen answer.
    pub name: String,
    /// When the option applies, or absent to rely on the name alone.
    pub description: Option<Value>,
}

impl SystemOneRequest {
    /// Parses and validates a JSON request body.
    ///
    /// An empty body is a missing body. Malformed JSON is `json_invalid` at
    /// the character position `serde_json` reports, with its message as the
    /// context error (Python's decoder words its messages differently).
    ///
    /// # Errors
    ///
    /// Returns [`SystemOneError::Validation`] with every violation found.
    pub fn from_json(body: &[u8]) -> Result<Self, SystemOneError> {
        if body.is_empty() {
            return Err(SystemOneError::Validation(vec![ValidationIssue {
                kind: "missing",
                loc: vec!["body".into()],
                msg: "Field required".to_owned(),
                input: Some(Value::Null),
                ctx: None,
            }]));
        }
        let body: Value =
            serde_json::from_slice(body).map_err(|error| json_invalid(body, &error))?;
        Self::from_value(&body)
    }

    /// Validates a decoded JSON request body.
    ///
    /// # Errors
    ///
    /// Returns [`SystemOneError::Validation`] with every violation found.
    pub fn from_value(body: &Value) -> Result<Self, SystemOneError> {
        let mut issues = Issues::default();
        let root = vec![LocItem::from("body")];
        let Value::Object(fields) = body else {
            issues.push(
                "model_attributes_type",
                root,
                "Input should be a valid dictionary or object to extract fields from",
                body,
            );
            return Err(SystemOneError::Validation(issues.0));
        };

        let state = match fields.get("state") {
            None => {
                issues.missing(at(&root, "state"), body);
                None
            }
            Some(state) => issues
                .text(state, &at(&root, "state"))
                .then(|| state.clone()),
        };

        let model = match fields.get("model") {
            None => {
                issues.missing(at(&root, "model"), body);
                None
            }
            Some(Value::String(model)) => Some(model.clone()),
            Some(other) => {
                issues.push(
                    "string_type",
                    at(&root, "model"),
                    "Input should be a valid string",
                    other,
                );
                None
            }
        };

        let questions = match fields.get("questions") {
            None => {
                issues.missing(at(&root, "questions"), body);
                Vec::new()
            }
            Some(Value::Object(entries)) => {
                let loc = at(&root, "questions");
                let before = issues.len();
                let questions: Vec<Question> = entries
                    .iter()
                    .filter_map(|(id, question)| parse_question(id, question, &loc, &mut issues))
                    .collect();
                if issues.len() == before && entries.is_empty() {
                    issues.too_short(loc, "Dictionary", 1, &Value::Object(entries.clone()));
                }
                questions
            }
            Some(other) => {
                issues.push(
                    "dict_type",
                    at(&root, "questions"),
                    "Input should be a valid dictionary",
                    other,
                );
                Vec::new()
            }
        };

        let images = match fields.get("x_images") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(items)) => parse_images(items, &at(&root, "x_images"), &mut issues),
            Some(other) => {
                issues.push(
                    "list_type",
                    at(&root, "x_images"),
                    "Input should be a valid list",
                    other,
                );
                Vec::new()
            }
        };

        match (state, model) {
            (Some(state), Some(model)) if issues.is_empty() => Ok(Self {
                state,
                model,
                questions,
                images,
            }),
            _ => Err(SystemOneError::Validation(issues.0)),
        }
    }

    /// Checks that the request names the served model.
    ///
    /// The comparison is exact; the server declares no model aliases.
    ///
    /// # Errors
    ///
    /// Returns [`SystemOneError::ModelNotFound`] for any other name.
    pub fn check_model(&self, served: &str) -> Result<(), SystemOneError> {
        if self.model == served {
            Ok(())
        } else {
            Err(SystemOneError::ModelNotFound {
                requested: self.model.clone(),
                served: served.to_owned(),
            })
        }
    }
}

/// Validates one entry of `questions`, a union discriminated by `type`.
fn parse_question(
    id: &str,
    value: &Value,
    parent: &[LocItem],
    issues: &mut Issues,
) -> Option<Question> {
    let loc = at(parent, id);
    let Value::Object(fields) = value else {
        issues.push(
            "model_attributes_type",
            loc,
            "Input should be a valid dictionary or object to extract fields from",
            value,
        );
        return None;
    };
    let tag = match fields.get("type") {
        Some(Value::String(tag)) if matches!(tag.as_str(), "noul" | "choice" | "score") => {
            tag.as_str()
        }
        None => {
            issues.0.push(ValidationIssue {
                kind: "union_tag_not_found",
                loc,
                msg: "Unable to extract tag using discriminator 'type'".to_owned(),
                input: Some(value.clone()),
                ctx: Some(json!({ "discriminator": "'type'" })),
            });
            return None;
        }
        Some(other) => {
            let tag = python_str(other);
            issues.0.push(ValidationIssue {
                kind: "union_tag_invalid",
                loc,
                msg: format!(
                    "Input tag '{tag}' found using 'type' does not match any of the expected \
                     tags: 'noul', 'choice', 'score'"
                ),
                input: Some(value.clone()),
                ctx: Some(json!({
                    "discriminator": "'type'",
                    "tag": tag,
                    "expected_tags": "'noul', 'choice', 'score'",
                })),
            });
            return None;
        }
    };

    // Fields validate in schema order (`type`, `instructions`, `criteria`),
    // then unknown keys in request order.
    let loc = at(&loc, tag);
    let before = issues.len();
    let instructions = issues.optional_text(fields.get("instructions"), &at(&loc, "instructions"));
    let criteria = match tag {
        "noul" => parse_noul_criteria(fields.get("criteria"), &at(&loc, "criteria"), issues),
        "choice" => parse_choice_criteria(fields.get("criteria"), value, &loc, issues),
        _ => parse_score_criteria(fields.get("criteria"), value, &loc, issues),
    };
    issues.forbid_extra(fields, &["type", "instructions", "criteria"], &loc);
    if issues.len() > before {
        return None;
    }
    let criteria = criteria?;

    // The question-level rule runs only on an otherwise valid question.
    if let Criteria::Noul {
        yes: None,
        no: None,
    } = criteria
        && instructions.is_none()
    {
        issues.0.push(ValidationIssue::value_error(
            loc,
            "a noul question needs instructions or a true or false criterion",
            Some(value.clone()),
        ));
        return None;
    }

    Some(Question {
        id: id.to_owned(),
        instructions,
        criteria,
    })
}

/// Validates noul criteria: absent, `null`, or an object with optional
/// `true` and `false` descriptions.
fn parse_noul_criteria(
    value: Option<&Value>,
    loc: &[LocItem],
    issues: &mut Issues,
) -> Option<Criteria> {
    match value {
        None | Some(Value::Null) => Some(Criteria::Noul {
            yes: None,
            no: None,
        }),
        Some(Value::Object(fields)) => {
            let yes = issues.optional_text(fields.get("true"), &at(loc, "true"));
            let no = issues.optional_text(fields.get("false"), &at(loc, "false"));
            issues.forbid_extra(fields, &["true", "false"], loc);
            Some(Criteria::Noul { yes, no })
        }
        Some(other) => {
            issues.push(
                "model_attributes_type",
                loc.to_vec(),
                "Input should be a valid dictionary or object to extract fields from",
                other,
            );
            None
        }
    }
}

/// Validates choice criteria: a map of 1-255 non-blank option names to
/// optional descriptions.
fn parse_choice_criteria(
    value: Option<&Value>,
    question: &Value,
    parent: &[LocItem],
    issues: &mut Issues,
) -> Option<Criteria> {
    let loc = at(parent, "criteria");
    let options = match value {
        None => {
            issues.missing(loc, question);
            return None;
        }
        Some(Value::Object(options)) => options,
        Some(other) => {
            issues.push(
                "dict_type",
                loc,
                "Input should be a valid dictionary",
                other,
            );
            return None;
        }
    };

    let before = issues.len();
    let parsed: Vec<ChoiceOption> = options
        .iter()
        .map(|(name, description)| ChoiceOption {
            name: name.clone(),
            description: issues.optional_text(Some(description), &at(&loc, name.as_str())),
        })
        .collect();
    if issues.len() > before {
        return None;
    }

    let criteria = Value::Object(options.clone());
    if options.is_empty() {
        issues.too_short(loc, "Dictionary", 1, &criteria);
        None
    } else if options.len() > MAX_CHOICE_OPTIONS {
        issues.too_long(
            loc,
            "Dictionary",
            MAX_CHOICE_OPTIONS,
            options.len(),
            &criteria,
        );
        None
    } else if options.keys().any(|name| name.chars().all(python_space)) {
        issues.0.push(ValidationIssue::value_error(
            loc,
            "option names must not be blank",
            Some(criteria),
        ));
        None
    } else {
        Some(Criteria::Choice(parsed))
    }
}

/// Validates score criteria: a list of 1-10 levels, each a JSON string,
/// object, or array.
fn parse_score_criteria(
    value: Option<&Value>,
    question: &Value,
    parent: &[LocItem],
    issues: &mut Issues,
) -> Option<Criteria> {
    let loc = at(parent, "criteria");
    let levels = match value {
        None => {
            issues.missing(loc, question);
            return None;
        }
        Some(Value::Array(levels)) => levels,
        Some(other) => {
            issues.push("list_type", loc, "Input should be a valid list", other);
            return None;
        }
    };
    let checked = issues.bounded_list(levels, &loc, MAX_SCORE_LEVELS, |issues, index, level| {
        issues.text(level, &at(&loc, index))
    });
    checked.then(|| Criteria::Score(levels.clone()))
}

/// Validates `x_images`: 1-8 image source strings.
fn parse_images(items: &[Value], loc: &[LocItem], issues: &mut Issues) -> Vec<String> {
    let checked = issues.bounded_list(items, loc, MAX_IMAGES, |issues, index, item| {
        let loc = at(loc, index);
        match item {
            Value::String(source) if is_image_source(source) => true,
            Value::String(_) => {
                issues.0.push(ValidationIssue::value_error(
                    loc,
                    "image must be an http(s) URL or a data:image/...;base64 URI",
                    Some(item.clone()),
                ));
                false
            }
            other => {
                issues.push("string_type", loc, "Input should be a valid string", other);
                false
            }
        }
    });
    if !checked {
        return Vec::new();
    }
    items
        .iter()
        .filter_map(|item| item.as_str().map(str::to_owned))
        .collect()
}

/// Returns whether a string names an image the server can fetch or decode:
/// an `http(s)` URL, or a `data:image/<subtype>;base64,<payload>` URI with a
/// non-empty payload. Scheme and media-type prefixes compare ASCII
/// case-insensitively.
fn is_image_source(source: &str) -> bool {
    let starts_with = |prefix: &str| {
        source
            .get(..prefix.len())
            .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
    };
    if starts_with("http://") || starts_with("https://") {
        return true;
    }
    let Some((header, payload)) = source.split_once(',') else {
        return false;
    };
    starts_with("data:image/")
        && !payload.is_empty()
        && header.len() >= ";base64".len()
        && header[header.len() - ";base64".len()..].eq_ignore_ascii_case(";base64")
}

/// Builds the `json_invalid` failure for an undecodable body.
fn json_invalid(body: &[u8], error: &serde_json::Error) -> SystemOneError {
    // serde_json reports a 1-based line and the byte count before the error
    // within that line; the location is the error's character offset.
    let line_start: usize = body
        .split_inclusive(|byte| *byte == b'\n')
        .take(error.line().saturating_sub(1))
        .map(<[u8]>::len)
        .sum();
    let end = (line_start + error.column()).min(body.len());
    let position = String::from_utf8_lossy(&body[..end]).chars().count();
    let message = error.to_string();
    let suffix = format!(" at line {} column {}", error.line(), error.column());
    let message = message.strip_suffix(&suffix).unwrap_or(&message);

    SystemOneError::Validation(vec![ValidationIssue {
        kind: "json_invalid",
        loc: vec!["body".into(), position.into()],
        msg: "JSON decode error".to_owned(),
        input: Some(json!({})),
        ctx: Some(json!({ "error": message })),
    }])
}

/// Returns `parent` extended by one location item.
fn at(parent: &[LocItem], item: impl Into<LocItem>) -> Vec<LocItem> {
    let mut loc = parent.to_vec();
    loc.push(item.into());
    loc
}

/// Validation issues collected in schema order.
#[derive(Default)]
struct Issues(Vec<ValidationIssue>);

impl Issues {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn push(&mut self, kind: &'static str, loc: Vec<LocItem>, msg: &str, input: &Value) {
        self.0.push(ValidationIssue {
            kind,
            loc,
            msg: msg.to_owned(),
            input: Some(input.clone()),
            ctx: None,
        });
    }

    /// Records a missing required field; the input is the enclosing object.
    fn missing(&mut self, loc: Vec<LocItem>, parent: &Value) {
        self.push("missing", loc, "Field required", parent);
    }

    /// Checks a `string | object | array` value.
    ///
    /// Any other value fails each union member, which pydantic reports under
    /// the member's label.
    fn text(&mut self, value: &Value, loc: &[LocItem]) -> bool {
        if matches!(value, Value::String(_) | Value::Object(_) | Value::Array(_)) {
            return true;
        }
        for (member, kind, msg) in [
            ("str", "string_type", "Input should be a valid string"),
            (
                "dict[str,any]",
                "dict_type",
                "Input should be a valid dictionary",
            ),
            ("list[any]", "list_type", "Input should be a valid list"),
        ] {
            self.push(kind, at(loc, member), msg, value);
        }
        false
    }

    /// Checks an optional `string | object | array | null` value; absent and
    /// `null` both yield `None`.
    fn optional_text(&mut self, value: Option<&Value>, loc: &[LocItem]) -> Option<Value> {
        match value {
            None | Some(Value::Null) => None,
            Some(value) => self.text(value, loc).then(|| value.clone()),
        }
    }

    /// Records every key of `fields` outside `declared`, in request order.
    fn forbid_extra(&mut self, fields: &Map<String, Value>, declared: &[&str], loc: &[LocItem]) {
        for (key, value) in fields {
            if !declared.contains(&key.as_str()) {
                self.push(
                    "extra_forbidden",
                    at(loc, key.as_str()),
                    "Extra inputs are not permitted",
                    value,
                );
            }
        }
    }

    /// Validates a list of at least one and at most `max` items with
    /// `check`, returning whether it is valid.
    ///
    /// As in pydantic, once the item count exceeds `max` the list reports
    /// only `too_long` with its full length, discarding its item errors.
    fn bounded_list(
        &mut self,
        items: &[Value],
        loc: &[LocItem],
        max: usize,
        mut check: impl FnMut(&mut Self, usize, &Value) -> bool,
    ) -> bool {
        let before = self.len();
        let mut valid = true;
        for (index, item) in items.iter().enumerate() {
            valid &= check(self, index, item);
            if index + 1 > max {
                self.0.truncate(before);
                self.too_long(loc.to_vec(), "List", max, items.len(), &Value::from(items));
                return false;
            }
        }
        if valid && items.is_empty() {
            self.too_short(loc.to_vec(), "List", 1, &Value::from(items));
            return false;
        }
        valid
    }

    fn too_short(&mut self, loc: Vec<LocItem>, field_type: &str, min: usize, input: &Value) {
        let actual = collection_len(input);
        self.0.push(ValidationIssue {
            kind: "too_short",
            loc,
            msg: format!(
                "{field_type} should have at least {min} {} after validation, not {actual}",
                items_word(min)
            ),
            input: Some(input.clone()),
            ctx: Some(json!({
                "field_type": field_type,
                "min_length": min,
                "actual_length": actual,
            })),
        });
    }

    fn too_long(
        &mut self,
        loc: Vec<LocItem>,
        field_type: &str,
        max: usize,
        actual: usize,
        input: &Value,
    ) {
        self.0.push(ValidationIssue {
            kind: "too_long",
            loc,
            msg: format!(
                "{field_type} should have at most {max} {} after validation, not {actual}",
                items_word(max)
            ),
            input: Some(input.clone()),
            ctx: Some(json!({
                "field_type": field_type,
                "max_length": max,
                "actual_length": actual,
            })),
        });
    }
}

/// Pydantic's noun for a length bound.
fn items_word(count: usize) -> &'static str {
    if count == 1 { "item" } else { "items" }
}

fn collection_len(value: &Value) -> usize {
    match value {
        Value::Array(items) => items.len(),
        Value::Object(fields) => fields.len(),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{ChoiceOption, Criteria, Question, SystemOneRequest};
    use crate::serving::systemone::SystemOneError;

    #[derive(serde::Deserialize)]
    struct Fixture {
        cases: Vec<Case>,
    }

    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        body: Option<Value>,
        raw_body: Option<String>,
        status: u16,
        response: Value,
    }

    /// Every rejected body produces the 422 body of a FastAPI/pydantic v2
    /// service declaring the same request model, and the accepted body passes.
    #[test]
    fn validation_errors_match_the_fastapi_contract() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../../../tests/python/fixtures/systemone_request_validation.json"
        ))
        .unwrap();

        for case in fixture.cases {
            let body = match (&case.body, &case.raw_body) {
                (Some(body), _) => serde_json::to_vec(body).unwrap(),
                (None, Some(raw)) => raw.clone().into_bytes(),
                (None, None) => unreachable!("fixture case {} has no body", case.name),
            };
            match SystemOneRequest::from_json(&body) {
                Ok(_) => assert_eq!(case.status, 200, "{} was accepted", case.name),
                Err(error) => {
                    assert_eq!(error.status_code().as_u16(), case.status, "{}", case.name);
                    assert_eq!(error.body(), case.response, "{}", case.name);
                }
            }
        }
    }

    #[test]
    fn a_valid_request_keeps_question_and_option_order() {
        let request = SystemOneRequest::from_json(
            br#"{"model": "m", "state": {"b": 1, "a": 2}, "extra": 1,
                 "questions": {
                    "z": {"type": "score", "criteria": ["low", {"k": 1}]},
                    "a": {"type": "choice", "instructions": "Pick", "criteria": {"y": null, "x": "desc"}},
                    "m": {"type": "noul", "criteria": {"false": ["no"]}}},
                 "x_images": ["https://example.com/a.png", "data:image/png;base64,AAAA"]}"#,
        )
        .unwrap();

        assert_eq!(request.model, "m");
        assert_eq!(
            serde_json::to_string(&request.state).unwrap(),
            r#"{"b":1,"a":2}"#
        );
        assert_eq!(
            request.questions,
            vec![
                Question {
                    id: "z".to_owned(),
                    instructions: None,
                    criteria: Criteria::Score(vec![json!("low"), json!({"k": 1})]),
                },
                Question {
                    id: "a".to_owned(),
                    instructions: Some(json!("Pick")),
                    criteria: Criteria::Choice(vec![
                        ChoiceOption {
                            name: "y".to_owned(),
                            description: None,
                        },
                        ChoiceOption {
                            name: "x".to_owned(),
                            description: Some(json!("desc")),
                        },
                    ]),
                },
                Question {
                    id: "m".to_owned(),
                    instructions: None,
                    criteria: Criteria::Noul {
                        yes: None,
                        no: Some(json!(["no"])),
                    },
                },
            ]
        );
        assert_eq!(
            request.images,
            ["https://example.com/a.png", "data:image/png;base64,AAAA"]
        );
    }

    /// The location counts characters, not bytes: serde_json stops after the
    /// closing brace, 17 bytes and 16 characters into the body.
    #[test]
    fn malformed_json_is_a_decode_error_at_its_character_position() {
        let error = SystemOneRequest::from_json("{\"state\": \"é\", }".as_bytes()).unwrap_err();

        let SystemOneError::Validation(issues) = &error else {
            panic!("unexpected {error:?}");
        };
        assert_eq!(error.status_code().as_u16(), 422);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].kind, "json_invalid");
        assert_eq!(
            serde_json::to_value(&issues[0].loc).unwrap(),
            json!(["body", 16])
        );
        assert_eq!(issues[0].msg, "JSON decode error");
    }

    #[test]
    fn only_the_served_model_name_is_accepted() {
        let request = SystemOneRequest::from_json(
            br#"{"model": "jev-latest", "state": "s", "questions": {"a": {"type": "noul", "instructions": "x"}}}"#,
        )
        .unwrap();

        assert!(request.check_model("jev-latest").is_ok());
        let error = request.check_model("diffusion-gemma").unwrap_err();
        assert_eq!(error.status_code().as_u16(), 404);
        assert_eq!(
            error.body(),
            json!({"detail": "The model 'jev-latest' does not exist; this server serves 'diffusion-gemma'."})
        );
    }
}
