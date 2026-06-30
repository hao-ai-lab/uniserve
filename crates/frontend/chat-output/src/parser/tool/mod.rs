//! Tool parser registration and selection boundary for `chat`.

use std::sync::LazyLock;

pub use uniserve_tool_parser::{
    DeepSeekV3ToolParser, DeepSeekV4ToolParser, DeepSeekV31ToolParser, DeepSeekV32ToolParser,
    Gemma4ToolParser, Glm45MoeToolParser, Glm47MoeToolParser, HermesToolParser, HyV3ToolParser,
    Internlm2ToolParser, KimiK2ToolParser, Llama3JsonToolParser, MinimaxM2ToolParser,
    MistralToolParser, Phi4MiniJsonToolParser, Qwen3CoderToolParser, Qwen3XmlToolParser,
    ToolCallDelta, ToolParser, ToolParserOutput,
};

use crate::parser::ParserFactory;
use crate::request::ChatTool;

/// Canonical public names for registered tool parsers.
pub mod names {
    pub const DEEPSEEK_V3: &str = "deepseek_v3";
    pub const DEEPSEEK_V31: &str = "deepseek_v31";
    pub const DEEPSEEK_V32: &str = "deepseek_v32";
    pub const DEEPSEEK_V4: &str = "deepseek_v4";
    pub const GLM45: &str = "glm45";
    pub const GLM47: &str = "glm47";
    pub const GEMMA4: &str = "gemma4";
    pub const HERMES: &str = "hermes";
    pub const HY_V3: &str = "hy_v3";
    // Matches the Python CLI name `--tool-call-parser internlm`, which Python
    // also routes to `Internlm2ToolParser` despite the version-agnostic name.
    pub const INTERNLM: &str = "internlm";
    pub const KIMI_K2: &str = "kimi_k2";
    pub const LLAMA3_JSON: &str = "llama3_json";
    pub const LLAMA4_JSON: &str = "llama4_json";
    pub const MINIMAX_M2: &str = "minimax_m2";
    pub const MISTRAL: &str = "mistral";
    pub const PHI4_MINI_JSON: &str = "phi4_mini_json";
    pub const QWEN3_CODER: &str = "qwen3_coder";
    pub const QWEN3_XML: &str = "qwen3_xml";
}

/// Constructor signature for one registered tool parser implementation.
type ToolParserCreator = fn(&[ChatTool]) -> uniserve_tool_parser::Result<Box<dyn ToolParser>>;

/// Registry and model matcher for tool parsers.
pub type ToolParserFactory = ParserFactory<ToolParserCreator>;

impl ToolParserFactory {
    /// Get the global tool parser factory with built-in registrations and model
    /// mappings.
    pub fn global() -> &'static Self {
        static INSTANCE: LazyLock<ToolParserFactory> = LazyLock::new(ToolParserFactory::new);
        &INSTANCE
    }

    /// Create the default registry with built-in parser names and model
    /// mappings.
    pub fn new() -> Self {
        let mut factory = Self::default();

        factory
            .register_parser::<DeepSeekV3ToolParser>(names::DEEPSEEK_V3)
            .register_parser::<DeepSeekV31ToolParser>(names::DEEPSEEK_V31)
            .register_parser::<DeepSeekV32ToolParser>(names::DEEPSEEK_V32)
            .register_parser::<DeepSeekV4ToolParser>(names::DEEPSEEK_V4)
            .register_parser::<Glm45MoeToolParser>(names::GLM45)
            .register_parser::<Glm47MoeToolParser>(names::GLM47)
            .register_parser::<Gemma4ToolParser>(names::GEMMA4)
            .register_parser::<HermesToolParser>(names::HERMES)
            .register_parser::<HyV3ToolParser>(names::HY_V3)
            .register_parser::<Internlm2ToolParser>(names::INTERNLM)
            .register_parser::<KimiK2ToolParser>(names::KIMI_K2)
            .register_parser::<Llama3JsonToolParser>(names::LLAMA3_JSON)
            .register_parser::<Llama3JsonToolParser>(names::LLAMA4_JSON)
            .register_parser::<MinimaxM2ToolParser>(names::MINIMAX_M2)
            .register_parser::<MistralToolParser>(names::MISTRAL)
            .register_parser::<Phi4MiniJsonToolParser>(names::PHI4_MINI_JSON)
            .register_parser::<Qwen3XmlToolParser>(names::QWEN3_XML)
            .register_parser::<Qwen3CoderToolParser>(names::QWEN3_CODER);

        factory
            .register_pattern("mistral-", names::MISTRAL)
            .register_pattern("mixtral-", names::MISTRAL)
            .register_pattern("qwen3-coder", names::QWEN3_CODER)
            .register_pattern("qwen2.5-coder", names::QWEN3_CODER)
            .register_pattern("qwen3.5", names::QWEN3_CODER)
            .register_pattern("qwen", names::QWEN3_XML)
            .register_pattern("hermes", names::HERMES)
            .register_pattern("hy3", names::HY_V3)
            .register_pattern("hy_v3", names::HY_V3)
            // Narrow to `internlm2` substring so it matches `internlm2-chat-7b`
            // and `internlm2_5-7b-chat` but NOT `internlm-chat-7b` (InternLM v1,
            // routes to Llama), `internlm3-*` (also Llama-architecture), or `Intern-S1` /
            // `Intern-S1-Pro` (separate intern-s1 parser, see PR #40115).
            .register_pattern("internlm2", names::INTERNLM)
            .register_pattern("llama-4", names::LLAMA4_JSON)
            .register_pattern("llama-3.2", names::LLAMA3_JSON)
            .register_pattern("llama-3.1", names::LLAMA3_JSON)
            .register_pattern("deepseek-r1", names::DEEPSEEK_V3)
            .register_pattern("deepseek-v4", names::DEEPSEEK_V4)
            .register_pattern("deepseek_v4", names::DEEPSEEK_V4)
            .register_pattern("deepseek-v3.2", names::DEEPSEEK_V32)
            .register_pattern("deepseek-v3.1", names::DEEPSEEK_V31)
            .register_pattern("deepseek-v3", names::DEEPSEEK_V3)
            .register_pattern("glm-5", names::GLM47)
            .register_pattern("glm-4.7", names::GLM47)
            .register_pattern("glm-4.6", names::GLM45)
            .register_pattern("glm-4.5", names::GLM45)
            .register_pattern("gemma4", names::GEMMA4)
            .register_pattern("gemma-4", names::GEMMA4)
            .register_pattern("kimi-k2", names::KIMI_K2)
            .register_pattern("minimax", names::MINIMAX_M2)
            .register_pattern("mm-m2", names::MINIMAX_M2);

        factory
    }

    /// Register one parser type that exposes a static `create` constructor.
    pub fn register_parser<T>(&mut self, name: &str) -> &mut Self
    where
        T: ToolParser + 'static,
    {
        self.register_creator(name, T::create)
    }

    /// Construct a parser from an exact name.
    pub fn create(&self, name: &str, tools: &[ChatTool]) -> crate::Result<Box<dyn ToolParser>> {
        let creator = self
            .creator(name)
            .ok_or_else(|| crate::Error::ParserUnavailableByName {
                kind: "tool",
                name: name.to_string(),
                available_names: self.list(),
            })?;

        creator(tools).map_err(|error| crate::Error::ParserInitialization {
            kind: "tool",
            name: name.to_string(),
            error: error.into(),
        })
    }

    /// Resolve a parser from model ID and then construct it.
    pub fn create_for_model(
        &self,
        model_id: &str,
        tools: &[ChatTool],
    ) -> crate::Result<Box<dyn ToolParser>> {
        let name = self.resolve_name_for_model(model_id).ok_or_else(|| {
            crate::Error::ParserUnavailableForModel {
                kind: "tool",
                model_id: model_id.to_string(),
            }
        })?;
        self.create(name, tools)
    }
}

#[cfg(test)]
mod tests;

/// Cross-registry routing guards for the hand-maintained tool/reasoning pattern
/// tables, which have no shared source of truth.
#[cfg(test)]
mod cross_registry_tests {
    use super::names;
    use crate::parser::reasoning::{ReasoningParserFactory, names as reasoning_names};
    use crate::parser::tool::ToolParserFactory;

    #[test]
    fn glm_tool_vs_reasoning_routing_divergence_is_pinned() {
        // The tool and reasoning factories each carry their own hand-maintained
        // (model-substring -> parser-name) table, so GLM routing can drift
        // between them. The divergence below is *intentional and forced*: a
        // dedicated GLM47 tool parser exists (Separator::Flexible) distinct from
        // GLM45 (Separator::Newline), but there is no GLM47 reasoning parser, so
        // every GLM variant shares the one GLM45 reasoning impl (an alias to the
        // Qwen3 reasoning parser).

        // This pins both sides so a future reordering of either pattern table —
        // or adding a GLM47-specific reasoning parser without updating both
        // registries — surfaces here instead of silently mis-routing one side.
        let tool = ToolParserFactory::new();
        let reasoning = ReasoningParserFactory::new();

        // Newest GLM family: dedicated GLM47 tool parser, shared GLM45 reasoning.
        for model_id in ["zai-org/GLM-5-32B-Chat", "glm-4.7"] {
            assert_eq!(
                tool.resolve_name_for_model(model_id),
                Some(names::GLM47),
                "tool routing for {model_id} changed",
            );
            assert_eq!(
                reasoning.resolve_name_for_model(model_id),
                Some(reasoning_names::GLM45),
                "reasoning routing for {model_id} changed",
            );
        }

        // Older GLM variants resolve to GLM45 on both sides.
        for model_id in ["glm-4.6", "glm-4.5"] {
            assert_eq!(
                tool.resolve_name_for_model(model_id),
                Some(names::GLM45),
                "tool routing for {model_id} changed",
            );
            assert_eq!(
                reasoning.resolve_name_for_model(model_id),
                Some(reasoning_names::GLM45),
                "reasoning routing for {model_id} changed",
            );
        }
    }

    #[test]
    fn deepseek_tool_vs_reasoning_routing_divergence_is_pinned() {
        // DeepSeek routing is intentionally divergent between the two
        // hand-maintained tables. The tool table carries dedicated
        // `deepseek-v3.1`/`deepseek-v3.2` parsers and routes `deepseek-r1` to the
        // V3 *tool* parser, while the reasoning table keeps a distinct
        // `deepseek_r1` reasoning parser and has no V3.1/V3.2 specialization (both
        // collapse onto the shared `deepseek_v3` reasoning parser). Pin both sides
        // so reordering or specializing one table without the other surfaces here.
        let tool = ToolParserFactory::new();
        let reasoning = ReasoningParserFactory::new();

        // R1 diverges: tool routes to the V3 tool parser, reasoning to its own R1.
        assert_eq!(
            tool.resolve_name_for_model("deepseek-ai/DeepSeek-R1-0528"),
            Some(names::DEEPSEEK_V3),
            "deepseek-r1 tool routing changed",
        );
        assert_eq!(
            reasoning.resolve_name_for_model("deepseek-ai/DeepSeek-R1-0528"),
            Some(reasoning_names::DEEPSEEK_R1),
            "deepseek-r1 reasoning routing changed",
        );

        // V3.1 / V3.2 diverge: tool has dedicated parsers, reasoning collapses to V3.
        assert_eq!(
            tool.resolve_name_for_model("deepseek-ai/DeepSeek-V3.1"),
            Some(names::DEEPSEEK_V31),
            "deepseek-v3.1 tool routing changed",
        );
        assert_eq!(
            reasoning.resolve_name_for_model("deepseek-ai/DeepSeek-V3.1"),
            Some(reasoning_names::DEEPSEEK_V3),
            "deepseek-v3.1 reasoning routing changed",
        );
        assert_eq!(
            tool.resolve_name_for_model("deepseek-ai/DeepSeek-V3.2-Exp"),
            Some(names::DEEPSEEK_V32),
            "deepseek-v3.2 tool routing changed",
        );
        assert_eq!(
            reasoning.resolve_name_for_model("deepseek-ai/DeepSeek-V3.2-Exp"),
            Some(reasoning_names::DEEPSEEK_V3),
            "deepseek-v3.2 reasoning routing changed",
        );

        // V4 stays in sync: both tables route to their own deepseek_v4 entry.
        assert_eq!(
            tool.resolve_name_for_model("deepseek-ai/DeepSeek-V4"),
            Some(names::DEEPSEEK_V4),
            "deepseek-v4 tool routing changed",
        );
        assert_eq!(
            reasoning.resolve_name_for_model("deepseek-ai/DeepSeek-V4"),
            Some(reasoning_names::DEEPSEEK_V4),
            "deepseek-v4 reasoning routing changed",
        );
    }

    #[test]
    fn internlm_tool_routing_has_no_reasoning_counterpart() {
        // InternLM2 has a dedicated tool parser pattern but the reasoning table
        // carries no InternLM pattern at all, so a versioned InternLM2 model that
        // resolves on the tool side resolves to nothing on the reasoning side.
        // Pin this asymmetry so adding/removing an InternLM reasoning pattern is a
        // deliberate, visible change.
        let tool = ToolParserFactory::new();
        let reasoning = ReasoningParserFactory::new();

        for model_id in ["internlm/internlm2-chat-7b", "internlm/internlm2_5-7b-chat"] {
            assert_eq!(
                tool.resolve_name_for_model(model_id),
                Some(names::INTERNLM),
                "internlm2 tool routing changed",
            );
            assert_eq!(
                reasoning.resolve_name_for_model(model_id),
                None,
                "internlm2 unexpectedly gained a reasoning pattern",
            );
        }
    }
}
