use uniserve_engine_client::GenMode;
use uniserve_text::tokenizer::DynTokenizer;

use super::defaults;
use super::resolution::{BAGEL_BUCKETS, ResolutionBucket, ResolutionPolicy, SENSENOVA_BUCKETS};
use super::schema::NativeGenerateBody;

const GEN_THINK_SYSTEM_PROMPT: &str = "You should first think about the planning process in the mind and then generate the image. \n\
     The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here";

const SENSENOVA_INTERLEAVE_SYSTEM_PROMPT: &str = "You are a multimodal assistant capable of reasoning with both text and images. \
You support two modes:\n\n\
Think Mode: When reasoning is needed, you MUST start with a <think></think> block \
and place all reasoning inside it. You MUST interleave text with generated images \
using tags like <image1>, <image2>. Images can ONLY be generated between <think> \
and </think>, and may be referenced in the final answer.\n\n\
Non-Think Mode: When no reasoning is needed, directly provide the answer without \
reasoning. Do not use tags like <image1>, <image2>; present any images naturally \
alongside the text.\n\n\
After the think block, always provide a concise, user-facing final answer. The \
answer may include text, images, or both. Match the user's language in both \
reasoning and the final answer.";

const SENSENOVA_GENERATION_SYSTEM_PROMPT: &str = "You are an image generation and editing assistant that accurately understands \
and executes user intent.\n\n\
You support two modes:\n\n\
1. Think Mode:\n\
If the task requires reasoning, you MUST start with a <think></think> block. Put \
all reasoning inside the block using plain text. DO NOT include any image tags. \
Keep it reasonable and directly useful for producing the final image.\n\n\
2. Non-Think Mode:\n\
If no reasoning is needed, directly produce the final image.\n\n\
Task Types:\n\n\
A. Text-to-Image Generation:\n\
- Generate a high-quality image based on the user's description.\n\
- Ensure visual clarity, semantic consistency, and completeness.\n\
- DO NOT introduce elements that contradict or override the user's intent.\n\n\
B. Image Editing:\n\
- Use the provided image(s) as input or reference for modification or transformation.\n\
- The result can be an edited image or a new image based on the reference(s).\n\
- Preserve all unspecified attributes unless explicitly changed.\n\n\
General Rules:\n\
- For any visible text in the image, follow the language specified for the \
rendered text in the user's description, not the language of the prompt. If no \
language is specified, use the user's input language.";

pub const IU_SYSTEM_PROMPT: &str = "\nLet's think step by step to answer the question. For text-based thinking, \
enclose the process within <think> </think>, e.g. <think> thinking process here \
</think>. For visual thinking, enclose the content within <image_start> \
</image_end>, e.g. <image_start> thinking image here </image_end>. Finally \
conclude with the final answer wrapped in <answer></answer> tags, i.e.\
<answer> answer here </answer>.\n";

const SENSENOVA_IMAGE_PREFIX: &str = "<think>\n\n</think>\n\n<img>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeModelFamily {
    Bagel,
    SenseNovaU1,
}

#[derive(Debug, Clone, Default)]
pub struct NativeControls {
    pub bos: u32,
    pub eos: u32,
    pub start_of_image: u32,
    pub end_of_image: u32,
    pub image_start_ids: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct NativeImageDefaults {
    pub resolution: &'static str,
    pub steps: u16,
    pub cfg_text_scale: f32,
    pub cfg_img_scale: f32,
    pub cfg_renorm_type: &'static str,
    pub cfg_renorm_min: f32,
    pub cfg_interval: (f32, f32),
    pub timestep_shift: f32,
    pub seed: Option<u64>,
    pub max_images: u16,
    pub max_images_limit: u16,
}

#[derive(Debug, Clone)]
pub struct NativeModelProfile {
    pub family: NativeModelFamily,
    pub controls: NativeControls,
    pub image_defaults: NativeImageDefaults,
    pub resolution_policy: ResolutionPolicy,
}

impl Default for NativeModelProfile {
    fn default() -> Self {
        Self::bagel(NativeControls::default())
    }
}

impl NativeModelProfile {
    fn bagel(controls: NativeControls) -> Self {
        Self {
            family: NativeModelFamily::Bagel,
            controls,
            image_defaults: NativeImageDefaults {
                resolution: "1:1",
                steps: defaults::DEFAULT_STEPS,
                cfg_text_scale: defaults::DEFAULT_CFG_TEXT_SCALE,
                cfg_img_scale: defaults::DEFAULT_CFG_IMG_SCALE,
                cfg_renorm_type: defaults::BAGEL_DEFAULT_CFG_RENORM_TYPE,
                cfg_renorm_min: defaults::DEFAULT_CFG_RENORM_MIN,
                cfg_interval: defaults::DEFAULT_CFG_INTERVAL,
                timestep_shift: defaults::BAGEL_DEFAULT_TIMESTEP_SHIFT,
                seed: None,
                max_images: defaults::DEFAULT_MAX_IMAGES,
                max_images_limit: defaults::BAGEL_MAX_IMAGES,
            },
            resolution_policy: ResolutionPolicy {
                default: ResolutionBucket {
                    name: "1:1",
                    width: defaults::BAGEL_DEFAULT_WIDTH,
                    height: defaults::BAGEL_DEFAULT_HEIGHT,
                },
                buckets: BAGEL_BUCKETS,
                allow_custom: true,
            },
        }
    }

    fn sensenova(controls: NativeControls) -> Self {
        Self {
            family: NativeModelFamily::SenseNovaU1,
            controls,
            image_defaults: NativeImageDefaults {
                resolution: defaults::SENSENOVA_DEFAULT_RESOLUTION,
                steps: defaults::DEFAULT_STEPS,
                cfg_text_scale: defaults::DEFAULT_CFG_TEXT_SCALE,
                cfg_img_scale: defaults::DEFAULT_CFG_IMG_SCALE,
                cfg_renorm_type: defaults::SENSENOVA_DEFAULT_CFG_RENORM_TYPE,
                cfg_renorm_min: defaults::DEFAULT_CFG_RENORM_MIN,
                cfg_interval: defaults::DEFAULT_CFG_INTERVAL,
                timestep_shift: defaults::SENSENOVA_DEFAULT_TIMESTEP_SHIFT,
                seed: Some(defaults::SENSENOVA_DEFAULT_SEED),
                max_images: 4,
                max_images_limit: defaults::SENSENOVA_MAX_IMAGES,
            },
            resolution_policy: ResolutionPolicy {
                default: ResolutionBucket {
                    name: defaults::SENSENOVA_DEFAULT_RESOLUTION,
                    width: defaults::SENSENOVA_DEFAULT_WIDTH,
                    height: defaults::SENSENOVA_DEFAULT_HEIGHT,
                },
                buckets: SENSENOVA_BUCKETS,
                allow_custom: false,
            },
        }
    }

    pub fn default_mode_name(&self) -> &'static str {
        match self.family {
            NativeModelFamily::Bagel => "text",
            NativeModelFamily::SenseNovaU1 => "interleave",
        }
    }

    pub fn supports_mode(&self, mode: GenMode) -> bool {
        !matches!(
            (self.family, mode),
            (NativeModelFamily::SenseNovaU1, GenMode::InterleaveUnd)
        )
    }

    pub fn build_prompt_ids(
        &self,
        tok: &DynTokenizer,
        body: &NativeGenerateBody,
        mode: GenMode,
    ) -> Vec<u32> {
        match self.family {
            NativeModelFamily::SenseNovaU1 => {
                let text = match mode {
                    GenMode::Text => chatml(
                        body.system_prompt.as_deref(),
                        &body.prompt,
                        body.assistant_prefix.as_deref().unwrap_or(""),
                    ),
                    GenMode::Image => chatml(
                        Some(
                            body.system_prompt
                                .as_deref()
                                .unwrap_or(SENSENOVA_GENERATION_SYSTEM_PROMPT),
                        ),
                        &body.prompt,
                        body.assistant_prefix
                            .as_deref()
                            .unwrap_or(SENSENOVA_IMAGE_PREFIX),
                    ),
                    GenMode::AutoInterleave => chatml(
                        Some(
                            body.system_prompt
                                .as_deref()
                                .unwrap_or(SENSENOVA_INTERLEAVE_SYSTEM_PROMPT),
                        ),
                        &body.prompt,
                        body.assistant_prefix.as_deref().unwrap_or(""),
                    ),
                    GenMode::InterleaveUnd => body.prompt.clone(),
                };
                encode(tok, &text)
            }
            NativeModelFamily::Bagel => match mode {
                GenMode::Text => encode(
                    tok,
                    &format!(
                        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}",
                        body.prompt,
                        body.assistant_prefix.as_deref().unwrap_or("")
                    ),
                ),
                GenMode::Image => {
                    let mut ids = vec![self.controls.bos];
                    ids.extend(encode(tok, &body.prompt));
                    ids.push(self.controls.eos);
                    ids
                }
                GenMode::AutoInterleave => encode(
                    tok,
                    &format!(
                        "<|im_start|>{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}",
                        body.system_prompt
                            .as_deref()
                            .unwrap_or(GEN_THINK_SYSTEM_PROMPT),
                        body.prompt,
                        body.assistant_prefix.as_deref().unwrap_or("")
                    ),
                ),
                GenMode::InterleaveUnd => encode(tok, &body.prompt),
            },
        }
    }

    pub fn build_negative_prompt_ids(&self, tok: &DynTokenizer, negative_prompt: &str) -> Vec<u32> {
        if negative_prompt.is_empty() {
            return Vec::new();
        }
        match self.family {
            NativeModelFamily::SenseNovaU1 => encode(
                tok,
                &chatml(
                    Some(SENSENOVA_GENERATION_SYSTEM_PROMPT),
                    negative_prompt,
                    "<img>",
                ),
            ),
            NativeModelFamily::Bagel => {
                let mut ids = vec![self.controls.bos];
                ids.extend(encode(tok, negative_prompt));
                ids.push(self.controls.eos);
                ids
            }
        }
    }

    pub fn wrap_understanding_text(&self, tok: &DynTokenizer, text: &str) -> Vec<u32> {
        let mut ids = vec![self.controls.bos];
        ids.extend(encode(tok, text));
        ids.push(self.controls.eos);
        ids
    }
}

pub fn resolve_native_profile(
    tokenizer: &dyn uniserve_text::tokenizer::Tokenizer,
) -> NativeModelProfile {
    let id = |token: &str| tokenizer.token_to_id(token);
    let vision_start = id("<|vision_start|>");
    let vision_end = id("<|vision_end|>");
    let img_start = id("<img>");
    let img_end = id("</img>");
    let img_context = id("<IMG_CONTEXT>");
    let sensenova = img_start.is_some() && img_end.is_some() && img_context.is_some();

    let controls = NativeControls {
        bos: id("<|im_start|>")
            .or_else(|| id("<|begin_of_sentence|>"))
            .unwrap_or(0),
        eos: id("<|im_end|>")
            .or_else(|| id("<|end_of_sentence|>"))
            .or_else(|| id("<|endoftext|>"))
            .unwrap_or(0),
        start_of_image: if sensenova {
            img_start
        } else {
            vision_start.or(img_start)
        }
        .unwrap_or(0),
        end_of_image: if sensenova {
            img_end
        } else {
            vision_end.or(img_end)
        }
        .unwrap_or(0),
        image_start_ids: if sensenova {
            Vec::new()
        } else {
            tokenizer.encode("image_start", false).unwrap_or_default()
        },
    };
    if sensenova {
        NativeModelProfile::sensenova(controls)
    } else {
        NativeModelProfile::bagel(controls)
    }
}

fn encode(tok: &DynTokenizer, text: &str) -> Vec<u32> {
    tok.encode(text, false).unwrap_or_default()
}

fn chatml(system: Option<&str>, user: &str, assistant_suffix: &str) -> String {
    let mut out = String::new();
    if let Some(system) = system {
        out.push_str("<|im_start|>system\n");
        out.push_str(system);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>user\n");
    out.push_str(user);
    out.push_str("<|im_end|>\n<|im_start|>assistant\n");
    out.push_str(assistant_suffix);
    out
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use uniserve_text::tokenizer::{DynTokenizer, Tokenizer};

    use super::*;

    #[derive(Debug)]
    struct ByteTokenizer;

    impl Tokenizer for ByteTokenizer {
        fn encode(
            &self,
            text: &str,
            _add_special_tokens: bool,
        ) -> uniserve_text::tokenizer::Result<Vec<u32>> {
            Ok(text.bytes().map(u32::from).collect())
        }

        fn decode(
            &self,
            token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> uniserve_text::tokenizer::Result<String> {
            Ok(
                String::from_utf8_lossy(&token_ids.iter().map(|id| *id as u8).collect::<Vec<_>>())
                    .into_owned(),
            )
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            match token {
                "<img>" => Some(10),
                "</img>" => Some(11),
                "<IMG_CONTEXT>" => Some(12),
                "<|im_start|>" => Some(13),
                "<|im_end|>" => Some(14),
                _ => None,
            }
        }
    }

    #[test]
    fn resolves_sensenova_profile() {
        let tok = ByteTokenizer;
        let profile = resolve_native_profile(&tok);
        assert_eq!(profile.family, NativeModelFamily::SenseNovaU1);
        assert_eq!(profile.resolution_policy.default.width, 2048);
        assert_eq!(profile.resolution_policy.default.height, 1152);
    }

    #[test]
    fn sensenova_image_prompt_uses_official_generation_prompt() {
        let tok: DynTokenizer = Arc::new(ByteTokenizer);
        let profile = resolve_native_profile(&*tok);
        let body = NativeGenerateBody {
            prompt: "paint a lake".into(),
            mode: Some("image".into()),
            ..Default::default()
        };
        let ids = profile.build_prompt_ids(&tok, &body, GenMode::Image);
        let text = tok.decode(&ids, false).unwrap();
        assert!(text.contains("image generation and editing assistant"));
        assert!(text.ends_with("<think>\n\n</think>\n\n<img>"));
    }
}
