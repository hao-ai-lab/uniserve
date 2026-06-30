//! `InferenceProgram`: a typed, host-internal op-graph.
//!
//! A model-aware but runtime-independent plan compiled from a `GenerateRequest`.
//! It is host-internal control-plane IR — it never crosses the host↔worker
//! wire. The FSM remains authoritative for execution; the program is the typed
//! representation alongside it.
//!
//! **What the scheduler consumes today vs. forward-looking scaffolding.** This
//! IR is deliberately richer than the FSM currently reads, so lifecycle
//! tracking / preemption / retry / fairness can build on it without re-deriving
//! the op graph. As of today the scheduler only consumes
//! [`InferenceProgram::program_id`] (trace/debug keying) and
//! [`InferenceProgram::wire_kinds`] (the FSM↔program op-kind consistency check,
//! which reads `ProgramOp::kind` only). The remaining structure —
//! [`ProgramOp::deps`]/[`ProgramOp::recurrence`]/[`ProgramOp::modality`],
//! [`OpState`], [`Recurrence`], [`LatencyClass`], and the
//! `priority`/`reserving`/`latency_class`/`fairness_group` policy-hint fields —
//! is compiled but **not yet read** by the scheduler (which keys off the
//! equivalent `ReqState`/`GenerateRequest` fields). It is intentionally retained
//! for future consumers; do not assume populating it changes runtime behavior
//! until a consumer is wired in.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use uniserve_core::{GenMode, Modality, ProgramId, RequestId};
use uniserve_engine_api::GenerateRequest;

/// Typed model-lifecycle op kinds — not generic prefill/decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramOpKind {
    Encode,
    Prefill,
    Decode,
    Denoise,
    Commit,
}

impl ProgramOpKind {
    pub fn modality(&self) -> Modality {
        match self {
            ProgramOpKind::Denoise | ProgramOpKind::Commit => Modality::Gen,
            _ => Modality::Und,
        }
    }
    /// The worker op-kind label this lifecycle op lowers to.
    pub fn wire_kinds(&self) -> &'static [&'static str] {
        match self {
            ProgramOpKind::Encode => &["vit_encode", "vae_encode"],
            ProgramOpKind::Prefill => &["prefill_und"],
            ProgramOpKind::Decode => &["decode_und"],
            ProgramOpKind::Denoise => &["denoise_gen"],
            ProgramOpKind::Commit => &["commit_gen", "commit_writeback"],
        }
    }
}

/// How many times an op recurs (the loop/condition structure).

/// Forward-looking: compiled per op but not yet read by the scheduler, whose
/// loop bounds come from the FSM and `GenerateRequest` (see module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recurrence {
    /// Runs once.
    Once,
    /// A fixed-count loop (e.g. `denoise` × steps).
    Fixed(u32),
    /// Data-dependent, bounded by a budget (e.g. `decode` until EOS/max_tokens,
    /// or the image sub-graph repeated up to `max_images`).
    Bounded(u32),
}

/// One typed op node with its dependencies.

/// Only `kind` is consumed today (via [`InferenceProgram::wire_kinds`]); `deps`,
/// `recurrence`, and `modality` are forward-looking scaffolding for moving
/// scheduling/lifecycle off the FSM (see module docs).
#[derive(Debug, Clone)]
pub struct ProgramOp {
    pub op_id: u32,
    pub kind: ProgramOpKind,
    pub modality: Modality,
    pub deps: Vec<u32>,
    pub recurrence: Recurrence,
}

/// Latency class — a policy hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyClass {
    Interactive,
    Batch,
}

/// Explicit op lifecycle states — the states an op moves through
/// for tracing/preemption/retry/cleanup. The FSM phases map onto these.

/// Op-level lifecycle
/// tracking; no scheduler code transitions through these yet (see module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpState {
    Created,
    Ready,
    Admitted,
    Leased,
    Batched,
    Submitted,
    Running,
    Succeeded,
    Failed,
    Cancelled,
    Released,
}

/// A typed program compiled from a request.
#[derive(Debug, Clone)]
pub struct InferenceProgram {
    pub request_id: RequestId,
    /// Assigned by the scheduler at enqueue (compile leaves it 0).
    pub program_id: ProgramId,
    pub mode: GenMode,
    pub ops: Vec<ProgramOp>,
    // Policy hints. Forward-looking: compiled from the request but
    // not yet read by the scheduler, which keys admission/preemption off the
    // equivalent `ReqState`/`GenerateRequest` fields (see module docs).
    pub priority: i32,
    pub latency_class: LatencyClass,
    pub reserving: bool,
    pub fairness_group: u32,
}

impl InferenceProgram {
    /// The distinct typed op kinds in this program.
    pub fn op_kinds(&self) -> Vec<ProgramOpKind> {
        let mut v: Vec<ProgramOpKind> = self.ops.iter().map(|o| o.kind).collect();
        v.dedup();
        v
    }

    /// All worker op-kind labels this program may lower to (the consistency set).
    pub fn wire_kinds(&self) -> std::collections::BTreeSet<&'static str> {
        self.ops
            .iter()
            .flat_map(|o| o.kind.wire_kinds().iter().copied())
            .collect()
    }
}

/// Compile a request into its typed program. Represents the
/// four current workflows exactly:
/// - Text: prefill → decode*
/// - Image: prefill → denoise×steps → commit
/// - AutoInterleave: prefill → decode* → (denoise×steps → commit → decode*) up to max_images
/// - InterleaveUnd: encode → prefill → decode* → (denoise×steps → commit → decode*) up to max_images
pub fn compile(req: &GenerateRequest) -> InferenceProgram {
    let mut ops = Vec::new();
    let mut next = 0u32;
    let mut id = || {
        let v = next;
        next += 1;
        v
    };
    let steps = req.image.steps.max(1) as u32;
    let max_images = req.image.max_images.max(1) as u32;
    let max_tokens = req.max_tokens.max(1) as u32;

    // Staged input images are dual-encoded (ViT ⊕ VAE) before prefill, for ANY
    // mode — as is understanding-interleave (its input + reasoning images).
    let has_encode = !req.mm_items.is_empty() || req.mode == GenMode::InterleaveUnd;
    let mut prefill_deps = Vec::new();
    if has_encode {
        let e = id();
        ops.push(ProgramOp {
            op_id: e,
            kind: ProgramOpKind::Encode,
            modality: Modality::Und,
            deps: vec![],
            recurrence: Recurrence::Once,
        });
        prefill_deps.push(e);
    }
    let p = id();
    ops.push(ProgramOp {
        op_id: p,
        kind: ProgramOpKind::Prefill,
        modality: Modality::Und,
        deps: prefill_deps,
        recurrence: Recurrence::Once,
    });

    match req.mode {
        GenMode::Text => {
            let d = id();
            ops.push(ProgramOp {
                op_id: d,
                kind: ProgramOpKind::Decode,
                modality: Modality::Und,
                deps: vec![p],
                recurrence: Recurrence::Bounded(max_tokens),
            });
        }
        GenMode::Image => {
            let dn = id();
            ops.push(ProgramOp {
                op_id: dn,
                kind: ProgramOpKind::Denoise,
                modality: Modality::Gen,
                deps: vec![p],
                recurrence: Recurrence::Fixed(steps),
            });
            let c = id();
            ops.push(ProgramOp {
                op_id: c,
                kind: ProgramOpKind::Commit,
                modality: Modality::Gen,
                deps: vec![dn],
                recurrence: Recurrence::Once,
            });
        }
        GenMode::AutoInterleave | GenMode::InterleaveUnd => {
            let d = id();
            ops.push(ProgramOp {
                op_id: d,
                kind: ProgramOpKind::Decode,
                modality: Modality::Und,
                deps: vec![p],
                recurrence: Recurrence::Bounded(max_tokens),
            });
            // The image sub-graph, repeated up to max_images, each fed by decode
            // and feeding back into decode (the interleave loop).
            let dn = id();
            ops.push(ProgramOp {
                op_id: dn,
                kind: ProgramOpKind::Denoise,
                modality: Modality::Gen,
                deps: vec![d],
                recurrence: Recurrence::Fixed(steps),
            });
            let c = id();
            ops.push(ProgramOp {
                op_id: c,
                kind: ProgramOpKind::Commit,
                modality: Modality::Gen,
                deps: vec![dn],
                recurrence: Recurrence::Bounded(max_images),
            });
            // commit feeds back into a continuation decode.
            let d2 = id();
            ops.push(ProgramOp {
                op_id: d2,
                kind: ProgramOpKind::Decode,
                modality: Modality::Und,
                deps: vec![c],
                recurrence: Recurrence::Bounded(max_tokens),
            });
        }
    }

    InferenceProgram {
        request_id: req.request_id,
        program_id: ProgramId(0), // assigned by the scheduler at enqueue
        mode: req.mode,
        ops,
        priority: req.priority,
        latency_class: if req.mode == GenMode::Text {
            LatencyClass::Interactive
        } else {
            LatencyClass::Batch
        },
        reserving: req.mode != GenMode::Text,
        fairness_group: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_core::{ImageParams, SamplingParams};

    fn req(mode: GenMode) -> GenerateRequest {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        GenerateRequest::new(
            RequestId(1),
            vec![1, 2, 3],
            SamplingParams::default(),
            ImageParams {
                steps: 8,
                max_images: 2,
                ..Default::default()
            },
            mode,
            32,
            tx,
        )
    }

    #[test]
    fn text_compiles_to_prefill_decode() {
        let p = compile(&req(GenMode::Text));
        assert_eq!(
            p.op_kinds(),
            vec![ProgramOpKind::Prefill, ProgramOpKind::Decode]
        );
        assert!(!p.reserving);
        assert_eq!(p.latency_class, LatencyClass::Interactive);
        assert_eq!(
            p.wire_kinds().into_iter().collect::<Vec<_>>(),
            vec!["decode_und", "prefill_und"]
        );
    }

    #[test]
    fn image_compiles_to_prefill_denoise_commit() {
        let p = compile(&req(GenMode::Image));
        assert_eq!(
            p.op_kinds(),
            vec![
                ProgramOpKind::Prefill,
                ProgramOpKind::Denoise,
                ProgramOpKind::Commit
            ]
        );
        assert!(p.reserving);
        // denoise recurs `steps` times.
        let dn = p
            .ops
            .iter()
            .find(|o| o.kind == ProgramOpKind::Denoise)
            .unwrap();
        assert_eq!(dn.recurrence, Recurrence::Fixed(8));
    }

    #[test]
    fn auto_interleave_has_text_image_text_graph() {
        let p = compile(&req(GenMode::AutoInterleave));
        let kinds = p.op_kinds();
        assert!(kinds.contains(&ProgramOpKind::Prefill));
        assert!(kinds.contains(&ProgramOpKind::Decode));
        assert!(kinds.contains(&ProgramOpKind::Denoise));
        assert!(kinds.contains(&ProgramOpKind::Commit));
        // the image commit is bounded by max_images.
        let c = p
            .ops
            .iter()
            .find(|o| o.kind == ProgramOpKind::Commit)
            .unwrap();
        assert_eq!(c.recurrence, Recurrence::Bounded(2));
        // no encode (that's understanding-interleave only).
        assert!(!kinds.contains(&ProgramOpKind::Encode));
    }

    #[test]
    fn understanding_interleave_starts_with_encode() {
        let p = compile(&req(GenMode::InterleaveUnd));
        assert_eq!(p.ops[0].kind, ProgramOpKind::Encode);
        assert!(p.op_kinds().contains(&ProgramOpKind::Denoise));
        assert!(p.wire_kinds().contains("vit_encode"));
    }

    #[test]
    fn staged_input_images_prepend_encode_in_any_mode() {
        let mut r = req(GenMode::Text);
        r.mm_items = vec![uniserve_engine_api::MmItem {
            hash: 1,
            position: 0,
            num_tokens: 4,
            b64: "x".into(),
        }];
        let p = compile(&r);
        assert_eq!(
            p.ops[0].kind,
            ProgramOpKind::Encode,
            "input images encode before prefill"
        );
        assert!(p.wire_kinds().contains("vit_encode") && p.wire_kinds().contains("vae_encode"));
    }

    #[test]
    fn all_modes_are_representable() {
        for mode in [
            GenMode::Text,
            GenMode::Image,
            GenMode::AutoInterleave,
            GenMode::InterleaveUnd,
        ] {
            let p = compile(&req(mode));
            assert!(
                !p.ops.is_empty(),
                "{mode:?} must compile to a non-empty program"
            );
            // deps reference valid op ids (a well-formed DAG).
            let ids: std::collections::BTreeSet<u32> = p.ops.iter().map(|o| o.op_id).collect();
            for op in &p.ops {
                for d in &op.deps {
                    assert!(
                        ids.contains(d),
                        "{mode:?} op {} dep {d} out of range",
                        op.op_id
                    );
                }
            }
        }
    }
}
