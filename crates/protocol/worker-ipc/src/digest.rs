use sha2::{Digest as _, Sha256};

use super::*;

pub(crate) fn is_digest(value: &Digest) -> bool {
    Digest::validate(value.as_str()).is_ok()
}

/// A little-endian, length-prefixed SHA-256 builder. Each digest is domain
/// separated by a neutral tag; the byte layout is mirrored exactly by the Python
/// worker so both sides compute identical digests.
pub(crate) struct CanonicalDigest(Sha256);

impl CanonicalDigest {
    pub(crate) fn new(domain: &[u8]) -> Self {
        let mut digest = Sha256::new();
        digest.update(domain);
        Self(digest)
    }

    pub(crate) fn finish(self) -> Digest {
        Digest::from_sha256_bytes(self.0.finalize().into())
    }
    pub(crate) fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }
    pub(crate) fn u8(&mut self, value: u8) {
        self.0.update([value]);
    }
    pub(crate) fn u16(&mut self, value: u16) {
        self.0.update(value.to_le_bytes());
    }
    pub(crate) fn u32(&mut self, value: u32) {
        self.0.update(value.to_le_bytes());
    }
    pub(crate) fn u64(&mut self, value: u64) {
        self.0.update(value.to_le_bytes());
    }
    pub(crate) fn f32(&mut self, value: f32) {
        self.u32(value.to_bits());
    }
    pub(crate) fn string(&mut self, value: &str) {
        self.u64(value.len() as u64);
        self.0.update(value.as_bytes());
    }
    pub(crate) fn u32s(&mut self, values: impl IntoIterator<Item = u32>) {
        let values: Vec<_> = values.into_iter().collect();
        self.u64(values.len() as u64);
        for value in values {
            self.u32(value);
        }
    }
    pub(crate) fn option<T>(&mut self, value: Option<T>, encode: impl FnOnce(&mut Self, T)) {
        match value {
            Some(value) => {
                self.u8(1);
                encode(self, value);
            }
            None => self.u8(0),
        }
    }

    pub(crate) fn request_key(&mut self, value: RequestKey) {
        self.u64(value.authority_id);
        self.u64(value.session_id.0);
        self.u64(value.epoch);
    }

    pub(crate) fn op_id(&mut self, value: OpId) {
        self.u64(value.0);
    }

    pub(crate) fn shape_bound(&mut self, value: &ShapeBound) {
        self.u64(value.dims.len() as u64);
        for dim in &value.dims {
            match dim {
                DimBound::Static(extent) => {
                    self.u8(0);
                    self.u32(*extent);
                }
                DimBound::Device { max } => {
                    self.u8(1);
                    self.u32(*max);
                }
            }
        }
    }

    pub(crate) fn product_ref(&mut self, value: &ProductRef) {
        self.request_key(value.request_key);
        self.op_id(value.producer_op_id);
        self.u16(value.output_index);
        self.u32(value.generation);
        self.u8(value.kind as u8);
        self.u8(value.storage_class as u8);
        self.u8(value.dtype as u8);
        self.shape_bound(&value.shape_bound);
        self.u32(value.point_range.base_point);
        self.u32(value.point_range.max_points);
    }

    pub(crate) fn version_ref(&mut self, value: &VersionRef) {
        self.request_key(value.request_key);
        self.op_id(value.producer_op_id);
        match &value.point {
            Point::Fixed {
                point_index,
                semantic_digest,
            } => {
                self.u8(0);
                self.u32(*point_index);
                self.string(semantic_digest);
            }
            Point::Device {
                point_index,
                selected_point,
                producer_plan_digest,
            } => {
                self.u8(1);
                self.u32(*point_index);
                self.u8(u8::from(selected_point.is_some()));
                if let Some(selected_point) = selected_point {
                    self.product_ref(selected_point);
                }
                self.string(producer_plan_digest);
            }
        }
    }

    pub(crate) fn bounds(&mut self, value: &Bounds) {
        self.u32(value.max_points);
        self.u32(value.max_tokens);
        self.u32(value.max_kv_pages);
        self.u64(value.max_latent_bytes);
        self.u64(value.max_completion_bytes);
        self.u64(value.max_transfer_bytes);
    }

    pub(crate) fn rng(&mut self, value: &Rng) {
        self.u64(value.seed);
        self.u64(value.semantic_index_base);
        self.u8(value.draw_layout as u8);
    }

    pub(crate) fn sampling(&mut self, value: &SamplingParams) {
        self.f32(value.temperature);
        self.u32(value.top_k);
        self.f32(value.top_p);
        self.bool(value.ignore_eos);
        self.option(value.seed, Self::u64);
        self.f32(value.min_p);
        self.f32(value.repetition_penalty);
        self.f32(value.frequency_penalty);
        self.f32(value.presence_penalty);
        self.u64(value.logit_bias.len() as u64);
        for (token, bias) in &value.logit_bias {
            self.u32(*token);
            self.f32(*bias);
        }
        self.u64(value.min_tokens as u64);
        self.bool(value.return_logprobs);
        self.u32(value.n_logprobs);
        self.bool(value.return_prompt_logprobs);
        self.u32(value.n_prompt_logprobs);
        self.u32s(value.logprob_token_ids.iter().copied());
        self.u64(value.bad_words_ids.len() as u64);
        for tokens in &value.bad_words_ids {
            self.u32s(tokens.iter().copied());
        }
        self.option(value.allowed_token_ids.as_deref(), |digest, tokens| {
            digest.u32s(tokens.iter().copied())
        });
        self.f32(value.typical_p);
        self.u32s(value.forced_token_ids.iter().copied());
    }

    pub(crate) fn image(&mut self, value: &ImageParams) {
        self.u16(value.steps);
        self.f32(value.cfg_text_scale);
        self.f32(value.cfg_img_scale);
        self.string(value.cfg_renorm_type.as_str());
        self.f32(value.cfg_renorm_min);
        self.f32(value.cfg_interval.0);
        self.f32(value.cfg_interval.1);
        self.f32(value.timestep_shift);
        self.u32(value.height);
        self.u32(value.width);
        self.option(value.seed, Self::u64);
        self.string(&value.negative_prompt);
        self.u16(value.max_images);
        self.u64(value.image_prompts.len() as u64);
        for prompt in &value.image_prompts {
            self.string(prompt);
        }
        self.bool(value.retain_images);
    }
}
