//! Dormant bounded typed inference program.
//!
//! Target contract from `specs/unified_inference_runtime.md` (Slice 2:
//! "Program, Product, Resource, And Scheduler Foundation"). An
//! [`InferenceProgram`] is a deterministic, bounded, typed state machine
//! compiled from one semantic request: typed input and product slots, a sealed
//! model-backed operation algebra
//! (`SequenceStep | FlowStep | EncodeStep | MaterializeStep`), finite
//! continuations, and declared output intents. It contains no protocol DTOs,
//! callbacks, media bytes, or model objects, and public media names never
//! appear below this seam.
//!
//! Nothing in production submits these programs yet; per the migration plan
//! the stack stays unreachable from production traffic until the whole-slice
//! cutover. This module owns the value shapes, the validation obligations that
//! are provable at the contract level, and the canonical fingerprint, so the
//! scheduler runtime and the Python worker can later negotiate one exact
//! schema.
//!
//! Boundedness is a graph property here, not a runtime hope: every cycle in
//! the continuation graph must pass through a [`Continuation::Repeat`] node,
//! each `Repeat` increments one declared monotone counter with a finite limit,
//! and no two `Repeat` nodes share a counter. Validation removes the `Repeat`
//! nodes and rejects any remaining cycle, which proves every execution
//! terminates within [`InferenceProgram::worst_case_transitions`].

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Program schema epoch; bumped only by an intentional schema-major review.
pub const PROGRAM_SCHEMA_VERSION: u32 = 1;

macro_rules! id_type {
    ($(#[$doc:meta])* $name:ident($inner:ty)) => {
        $(#[$doc])*
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        pub struct $name(pub $inner);
    };
}

id_type!(
    /// One node in the program's continuation graph.
    NodeId(u32)
);
id_type!(
    /// One declared monotone loop counter.
    CounterId(u16)
);
id_type!(
    /// One immutable compiled input slot.
    InputSlotId(u16)
);
id_type!(
    /// One typed internal product slot.
    ProductSlotId(u16)
);
id_type!(
    /// One model-backed request session slot.
    SessionSlotId(u16)
);
id_type!(
    /// One requested output intent.
    OutputId(u16)
);
id_type!(
    /// One family-registered resident route.
    RouteId(u32)
);
id_type!(
    /// One registered flow schedule.
    ScheduleId(u32)
);

/// Typed representation space of one product slot. Text tokens, audio codec
/// tokens, and video tokens are different registered discrete spaces; the
/// scheduler validates schema identity without branching on media names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepresentationSpace {
    DiscreteSequence(u32),
    FeatureSequence(u32),
    LatentTensor(u32),
    SampleTensor(u32),
    ScoreTensor(u32),
    RecurrentState(u32),
}

/// One immutable compiled input (already tokenized / preprocessed / ingested).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputSlotSpec {
    pub slot: InputSlotId,
    pub space: RepresentationSpace,
    /// Exact element extent of the compiled input.
    pub elements: u64,
}

/// One typed internal product slot written and read by program nodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProductSlotSpec {
    pub slot: ProductSlotId,
    pub space: RepresentationSpace,
    /// Finite worst-case element extent used for resource derivation.
    pub max_elements: u64,
}

/// One model-backed request session slot (logical state only).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSlotSpec {
    pub slot: SessionSlotId,
}

/// One declared monotone counter; every backedge increments exactly one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CounterSpec {
    pub counter: CounterId,
}

/// One requested output with a finite bound, compiled from an output intent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledOutputSpec {
    pub output: OutputId,
    pub space: RepresentationSpace,
    /// Finite bound on produced elements (tokens, samples, frames, ...).
    pub max_elements: u64,
}

/// A typed operand read by one node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotRead {
    Input(InputSlotId),
    Product(ProductSlotId),
}

/// A typed product written by one node (a new immutable product version).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotWrite {
    pub product: ProductSlotId,
}

/// One discrete sequence-model iteration (prefill, decode, verification —
/// all one operation; phase never selects a worker entry point).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SequenceStepTemplate {
    pub route: RouteId,
    pub session: SessionSlotId,
    /// Finite bound on discrete positions evaluated in this iteration.
    pub max_positions: NonZeroU32,
}

/// One exact diffusion / flow schedule coordinate; no internal timestep loop.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlowStepTemplate {
    pub route: RouteId,
    pub schedule: ScheduleId,
    /// Bounded CFG / guidance branch table size.
    pub branch_count: NonZeroU32,
}

/// One resident neural transform from typed products into typed products
/// (vision, audio-feature, video, or latent encoders).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncodeStepTemplate {
    pub route: RouteId,
}

/// One resident neural transform from latents/features into a durable
/// sample-space product (image decode, vocoding, video decode). Non-neural
/// codecs and muxing stay in media egress, outside the engine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterializeStepTemplate {
    pub route: RouteId,
}

/// The sealed model-backed operation algebra. A genuinely new neural
/// iteration semantic is a schema-major change, never an opaque custom op.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationTemplate {
    Sequence(SequenceStepTemplate),
    Flow(FlowStepTemplate),
    Encode(EncodeStepTemplate),
    Materialize(MaterializeStepTemplate),
}

/// One compact-result selector case.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectCase {
    /// Closed result tag declared by the operation's result contract.
    pub result_tag: u32,
    pub next: NodeId,
}

/// Terminal projection: which declared outputs this terminal path closes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminalProjection {
    pub close_outputs: Vec<OutputId>,
}

/// The finite continuation algebra. Every backedge is a `Repeat` with one
/// declared counter and a compile-time maximum.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Continuation {
    Next(NodeId),
    Repeat {
        counter: CounterId,
        limit: NonZeroU32,
        body: NodeId,
        exit: NodeId,
    },
    Select {
        cases: Vec<SelectCase>,
        default: NodeId,
    },
    Finish(TerminalProjection),
}

/// One program node: a sealed operation template plus typed dependencies and
/// one finite continuation. Operands are closed typed references — no
/// callbacks, string method names, or untyped extension records.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgramNode {
    pub node_id: NodeId,
    pub operation: OperationTemplate,
    pub reads: Vec<SlotRead>,
    pub writes: Vec<SlotWrite>,
    pub continuation: Continuation,
}

/// A deterministic bounded typed state machine compiled from one semantic
/// request. See the module docs for the boundedness proof obligations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceProgram {
    pub schema_version: u32,
    pub inputs: Vec<InputSlotSpec>,
    pub products: Vec<ProductSlotSpec>,
    pub sessions: Vec<SessionSlotSpec>,
    pub counters: Vec<CounterSpec>,
    pub nodes: Vec<ProgramNode>,
    pub entry: NodeId,
    pub outputs: Vec<CompiledOutputSpec>,
}

/// A program shape that violates the closed contract.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProgramValidationError {
    #[error("program declares no nodes")]
    NoNodes,
    #[error("program declares no outputs")]
    NoOutputs,
    #[error("duplicate node id {0:?}")]
    DuplicateNode(NodeId),
    #[error("duplicate slot or counter declaration")]
    DuplicateDeclaration,
    #[error("entry node {0:?} is not declared")]
    UnknownEntry(NodeId),
    #[error("node {0:?} continues to undeclared node {1:?}")]
    UnknownTarget(NodeId, NodeId),
    #[error("node {0:?} references undeclared slot")]
    UnknownSlot(NodeId),
    #[error("repeat at node {0:?} uses undeclared counter {1:?}")]
    UnknownCounter(NodeId, CounterId),
    #[error("counter {0:?} is incremented by more than one repeat")]
    SharedCounter(CounterId),
    #[error("select at node {0:?} has no cases")]
    EmptySelect(NodeId),
    #[error("node {0:?} is unreachable from the entry")]
    Unreachable(NodeId),
    #[error("program has no terminal node")]
    NoTerminal,
    #[error("cycle through node {0:?} is not bounded by a repeat counter")]
    UnboundedCycle(NodeId),
    #[error("product slot {0:?} is read but never written")]
    ReadWithoutProducer(ProductSlotId),
    #[error("terminal at node {0:?} closes undeclared output {1:?}")]
    UnknownOutput(NodeId, OutputId),
    #[error("output {0:?} is not closed by any terminal path")]
    UnclosedOutput(OutputId),
}

impl InferenceProgram {
    /// Prove every contract-level obligation from the migration plan's
    /// Slice 2 acceptance list that is decidable on the program value alone:
    /// invalid loops, unknown references, illegal product edges, unreachable
    /// or non-terminating shapes, and unclosed outputs all fail here, before
    /// admission. Resource-envelope and placement validation need catalog
    /// context and live in the scheduler runtime.
    pub fn validate(&self) -> Result<(), ProgramValidationError> {
        if self.nodes.is_empty() {
            return Err(ProgramValidationError::NoNodes);
        }
        if self.outputs.is_empty() {
            return Err(ProgramValidationError::NoOutputs);
        }
        let nodes = self.node_index()?;
        let inputs: BTreeSet<_> = self.inputs.iter().map(|input| input.slot).collect();
        let products: BTreeSet<_> = self.products.iter().map(|product| product.slot).collect();
        let sessions: BTreeSet<_> = self.sessions.iter().map(|session| session.slot).collect();
        let counters: BTreeSet<_> = self
            .counters
            .iter()
            .map(|counter| counter.counter)
            .collect();
        let outputs: BTreeSet<_> = self.outputs.iter().map(|output| output.output).collect();
        if inputs.len() != self.inputs.len()
            || products.len() != self.products.len()
            || sessions.len() != self.sessions.len()
            || counters.len() != self.counters.len()
            || outputs.len() != self.outputs.len()
        {
            return Err(ProgramValidationError::DuplicateDeclaration);
        }
        if !nodes.contains_key(&self.entry) {
            return Err(ProgramValidationError::UnknownEntry(self.entry));
        }
        self.validate_nodes(&nodes, &inputs, &products, &sessions, &counters, &outputs)?;
        self.validate_reachability(&nodes)?;
        self.validate_bounded_cycles(&nodes)?;
        self.validate_products(&products)?;
        self.validate_outputs(&outputs)?;
        Ok(())
    }

    /// Upper bound on scheduler transitions for one incarnation: each node
    /// runs at most once per surrounding repeat iteration, so the product of
    /// declared repeat limits times the node count dominates every execution.
    /// Saturates instead of overflowing.
    pub fn worst_case_transitions(&self) -> u64 {
        let mut bound: u64 = self.nodes.len() as u64;
        for node in &self.nodes {
            if let Continuation::Repeat { limit, .. } = &node.continuation {
                bound = bound.saturating_mul(u64::from(limit.get()));
            }
        }
        bound
    }

    /// Canonical deterministic digest over the schema version and every field
    /// in declaration order (via the canonical serde encoding). Equal programs
    /// fingerprint equally on every peer; any field change is a new identity.
    pub fn fingerprint(&self) -> ProgramFingerprint {
        let mut hasher = Sha256::new();
        let encoded = serde_json::to_vec(self).expect("program values always serialize");
        hasher.update(&encoded);
        let digest = hasher.finalize();
        let mut bytes = [0_u8; 32];
        bytes.copy_from_slice(&digest);
        ProgramFingerprint(bytes)
    }

    fn node_index(&self) -> Result<BTreeMap<NodeId, &ProgramNode>, ProgramValidationError> {
        let mut nodes = BTreeMap::new();
        for node in &self.nodes {
            if nodes.insert(node.node_id, node).is_some() {
                return Err(ProgramValidationError::DuplicateNode(node.node_id));
            }
        }
        Ok(nodes)
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_nodes(
        &self,
        nodes: &BTreeMap<NodeId, &ProgramNode>,
        inputs: &BTreeSet<InputSlotId>,
        products: &BTreeSet<ProductSlotId>,
        sessions: &BTreeSet<SessionSlotId>,
        counters: &BTreeSet<CounterId>,
        outputs: &BTreeSet<OutputId>,
    ) -> Result<(), ProgramValidationError> {
        let mut used_counters = BTreeSet::new();
        for node in &self.nodes {
            for read in &node.reads {
                let known = match read {
                    SlotRead::Input(slot) => inputs.contains(slot),
                    SlotRead::Product(slot) => products.contains(slot),
                };
                if !known {
                    return Err(ProgramValidationError::UnknownSlot(node.node_id));
                }
            }
            for write in &node.writes {
                if !products.contains(&write.product) {
                    return Err(ProgramValidationError::UnknownSlot(node.node_id));
                }
            }
            if let OperationTemplate::Sequence(step) = &node.operation
                && !sessions.contains(&step.session)
            {
                return Err(ProgramValidationError::UnknownSlot(node.node_id));
            }
            for target in continuation_targets(&node.continuation) {
                if !nodes.contains_key(&target) {
                    return Err(ProgramValidationError::UnknownTarget(node.node_id, target));
                }
            }
            match &node.continuation {
                Continuation::Repeat { counter, .. } => {
                    if !counters.contains(counter) {
                        return Err(ProgramValidationError::UnknownCounter(
                            node.node_id,
                            *counter,
                        ));
                    }
                    if !used_counters.insert(*counter) {
                        return Err(ProgramValidationError::SharedCounter(*counter));
                    }
                }
                Continuation::Select { cases, .. } if cases.is_empty() => {
                    return Err(ProgramValidationError::EmptySelect(node.node_id));
                }
                Continuation::Finish(projection) => {
                    for output in &projection.close_outputs {
                        if !outputs.contains(output) {
                            return Err(ProgramValidationError::UnknownOutput(
                                node.node_id,
                                *output,
                            ));
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn validate_reachability(
        &self,
        nodes: &BTreeMap<NodeId, &ProgramNode>,
    ) -> Result<(), ProgramValidationError> {
        let mut seen = BTreeSet::new();
        let mut stack = vec![self.entry];
        let mut terminal_reachable = false;
        while let Some(node_id) = stack.pop() {
            if !seen.insert(node_id) {
                continue;
            }
            let node = nodes[&node_id];
            if matches!(node.continuation, Continuation::Finish(_)) {
                terminal_reachable = true;
            }
            stack.extend(continuation_targets(&node.continuation));
        }
        for node in &self.nodes {
            if !seen.contains(&node.node_id) {
                return Err(ProgramValidationError::Unreachable(node.node_id));
            }
        }
        if !terminal_reachable {
            return Err(ProgramValidationError::NoTerminal);
        }
        Ok(())
    }

    /// Every cycle must pass through a `Repeat`: with all `Repeat` nodes
    /// removed, the continuation graph must be acyclic. Combined with finite
    /// per-counter limits this proves termination.
    fn validate_bounded_cycles(
        &self,
        nodes: &BTreeMap<NodeId, &ProgramNode>,
    ) -> Result<(), ProgramValidationError> {
        #[derive(Clone, Copy, PartialEq)]
        enum Mark {
            Visiting,
            Done,
        }
        let mut marks: BTreeMap<NodeId, Mark> = BTreeMap::new();
        for &start in nodes.keys() {
            if marks.contains_key(&start) {
                continue;
            }
            // Iterative DFS with an explicit re-entry frame for post-marking.
            let mut stack = vec![(start, false)];
            while let Some((node_id, expanded)) = stack.pop() {
                if expanded {
                    marks.insert(node_id, Mark::Done);
                    continue;
                }
                match marks.get(&node_id) {
                    Some(Mark::Done) => continue,
                    Some(Mark::Visiting) => {
                        return Err(ProgramValidationError::UnboundedCycle(node_id));
                    }
                    None => {}
                }
                let node = nodes[&node_id];
                if matches!(node.continuation, Continuation::Repeat { .. }) {
                    // Repeat nodes are the sanctioned backedge owners; cutting
                    // them out of the walk is exactly "remove Repeat nodes".
                    marks.insert(node_id, Mark::Done);
                    continue;
                }
                marks.insert(node_id, Mark::Visiting);
                stack.push((node_id, true));
                for target in continuation_targets(&node.continuation) {
                    match marks.get(&target) {
                        Some(Mark::Visiting) => {
                            return Err(ProgramValidationError::UnboundedCycle(target));
                        }
                        Some(Mark::Done) => {}
                        None => stack.push((target, false)),
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_products(
        &self,
        products: &BTreeSet<ProductSlotId>,
    ) -> Result<(), ProgramValidationError> {
        let mut written = BTreeSet::new();
        for node in &self.nodes {
            for write in &node.writes {
                written.insert(write.product);
            }
        }
        for node in &self.nodes {
            for read in &node.reads {
                if let SlotRead::Product(slot) = read
                    && products.contains(slot)
                    && !written.contains(slot)
                {
                    return Err(ProgramValidationError::ReadWithoutProducer(*slot));
                }
            }
        }
        Ok(())
    }

    fn validate_outputs(&self, outputs: &BTreeSet<OutputId>) -> Result<(), ProgramValidationError> {
        let mut closed = BTreeSet::new();
        for node in &self.nodes {
            if let Continuation::Finish(projection) = &node.continuation {
                closed.extend(projection.close_outputs.iter().copied());
            }
        }
        for output in outputs {
            if !closed.contains(output) {
                return Err(ProgramValidationError::UnclosedOutput(*output));
            }
        }
        Ok(())
    }
}

/// Deterministic content identity of one validated program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProgramFingerprint(pub [u8; 32]);

impl ProgramFingerprint {
    pub fn to_hex(self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

fn continuation_targets(continuation: &Continuation) -> Vec<NodeId> {
    match continuation {
        Continuation::Next(next) => vec![*next],
        Continuation::Repeat { body, exit, .. } => vec![*body, *exit],
        Continuation::Select { cases, default } => {
            let mut targets: Vec<NodeId> = cases.iter().map(|case| case.next).collect();
            targets.push(*default);
            targets
        }
        Continuation::Finish(_) => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limit(value: u32) -> NonZeroU32 {
        NonZeroU32::new(value).expect("nonzero")
    }

    fn sequence_op() -> OperationTemplate {
        OperationTemplate::Sequence(SequenceStepTemplate {
            route: RouteId(0),
            session: SessionSlotId(0),
            max_positions: limit(1),
        })
    }

    /// Text decode: one sequence node looping under a bounded counter, then a
    /// terminal that closes the text output.
    fn text_program() -> InferenceProgram {
        InferenceProgram {
            schema_version: PROGRAM_SCHEMA_VERSION,
            inputs: vec![InputSlotSpec {
                slot: InputSlotId(0),
                space: RepresentationSpace::DiscreteSequence(1),
                elements: 128,
            }],
            products: vec![ProductSlotSpec {
                slot: ProductSlotId(0),
                space: RepresentationSpace::DiscreteSequence(1),
                max_elements: 256,
            }],
            sessions: vec![SessionSlotSpec {
                slot: SessionSlotId(0),
            }],
            counters: vec![CounterSpec {
                counter: CounterId(0),
            }],
            nodes: vec![
                ProgramNode {
                    node_id: NodeId(0),
                    operation: sequence_op(),
                    reads: vec![SlotRead::Input(InputSlotId(0))],
                    writes: vec![SlotWrite {
                        product: ProductSlotId(0),
                    }],
                    continuation: Continuation::Next(NodeId(1)),
                },
                ProgramNode {
                    node_id: NodeId(1),
                    operation: sequence_op(),
                    reads: vec![SlotRead::Product(ProductSlotId(0))],
                    writes: vec![SlotWrite {
                        product: ProductSlotId(0),
                    }],
                    continuation: Continuation::Repeat {
                        counter: CounterId(0),
                        limit: limit(255),
                        body: NodeId(1),
                        exit: NodeId(2),
                    },
                },
                ProgramNode {
                    node_id: NodeId(2),
                    operation: sequence_op(),
                    reads: vec![],
                    writes: vec![],
                    continuation: Continuation::Finish(TerminalProjection {
                        close_outputs: vec![OutputId(0)],
                    }),
                },
            ],
            entry: NodeId(0),
            outputs: vec![CompiledOutputSpec {
                output: OutputId(0),
                space: RepresentationSpace::DiscreteSequence(1),
                max_elements: 256,
            }],
        }
    }

    /// Text-to-image: encode conditioning, repeat one flow step per schedule
    /// coordinate, materialize samples — the same shapes with no image mode.
    fn t2i_program() -> InferenceProgram {
        let latent = ProductSlotId(0);
        let sample = ProductSlotId(1);
        InferenceProgram {
            schema_version: PROGRAM_SCHEMA_VERSION,
            inputs: vec![InputSlotSpec {
                slot: InputSlotId(0),
                space: RepresentationSpace::DiscreteSequence(1),
                elements: 64,
            }],
            products: vec![
                ProductSlotSpec {
                    slot: latent,
                    space: RepresentationSpace::LatentTensor(7),
                    max_elements: 1 << 20,
                },
                ProductSlotSpec {
                    slot: sample,
                    space: RepresentationSpace::SampleTensor(3),
                    max_elements: 1 << 24,
                },
            ],
            sessions: vec![],
            counters: vec![CounterSpec {
                counter: CounterId(0),
            }],
            nodes: vec![
                ProgramNode {
                    node_id: NodeId(0),
                    operation: OperationTemplate::Encode(EncodeStepTemplate { route: RouteId(1) }),
                    reads: vec![SlotRead::Input(InputSlotId(0))],
                    writes: vec![SlotWrite { product: latent }],
                    continuation: Continuation::Next(NodeId(1)),
                },
                ProgramNode {
                    node_id: NodeId(1),
                    operation: OperationTemplate::Flow(FlowStepTemplate {
                        route: RouteId(2),
                        schedule: ScheduleId(1),
                        branch_count: limit(3),
                    }),
                    reads: vec![SlotRead::Product(latent)],
                    writes: vec![SlotWrite { product: latent }],
                    continuation: Continuation::Repeat {
                        counter: CounterId(0),
                        limit: limit(50),
                        body: NodeId(1),
                        exit: NodeId(2),
                    },
                },
                ProgramNode {
                    node_id: NodeId(2),
                    operation: OperationTemplate::Materialize(MaterializeStepTemplate {
                        route: RouteId(3),
                    }),
                    reads: vec![SlotRead::Product(latent)],
                    writes: vec![SlotWrite { product: sample }],
                    continuation: Continuation::Finish(TerminalProjection {
                        close_outputs: vec![OutputId(0)],
                    }),
                },
            ],
            entry: NodeId(0),
            outputs: vec![CompiledOutputSpec {
                output: OutputId(0),
                space: RepresentationSpace::SampleTensor(3),
                max_elements: 1 << 24,
            }],
        }
    }

    #[test]
    fn text_and_t2i_programs_validate_with_one_shape() {
        text_program().validate().expect("text program is valid");
        t2i_program().validate().expect("t2i program is valid");
    }

    #[test]
    fn worst_case_transitions_multiply_repeat_limits() {
        assert_eq!(text_program().worst_case_transitions(), 3 * 255);
        assert_eq!(t2i_program().worst_case_transitions(), 3 * 50);
    }

    #[test]
    fn fingerprints_are_deterministic_and_content_sensitive() {
        let base = text_program().fingerprint();
        assert_eq!(base, text_program().fingerprint());
        let mut changed = text_program();
        changed.outputs[0].max_elements = 512;
        assert_ne!(base, changed.fingerprint());
        assert_eq!(base.to_hex().len(), 64);
    }

    #[test]
    fn cycles_without_a_repeat_are_rejected() {
        let mut program = text_program();
        // Rewire the loop as a raw Select backedge (node 1 -> node 0) while the
        // terminal stays reachable through the default arm.
        program.nodes[1].continuation = Continuation::Select {
            cases: vec![SelectCase {
                result_tag: 0,
                next: NodeId(0),
            }],
            default: NodeId(2),
        };
        assert!(matches!(
            program.validate(),
            Err(ProgramValidationError::UnboundedCycle(_))
        ));
    }

    #[test]
    fn repeat_counters_must_be_declared_and_exclusive() {
        let mut program = text_program();
        program.counters.clear();
        assert!(matches!(
            program.validate(),
            Err(ProgramValidationError::UnknownCounter(_, _))
        ));

        let mut shared = t2i_program();
        shared.nodes[0].continuation = Continuation::Repeat {
            counter: CounterId(0),
            limit: limit(2),
            body: NodeId(0),
            exit: NodeId(1),
        };
        assert!(matches!(
            shared.validate(),
            Err(ProgramValidationError::SharedCounter(CounterId(0)))
        ));
    }

    #[test]
    fn product_reads_need_a_producer() {
        let mut program = t2i_program();
        program.nodes[0].writes.clear();
        program.nodes[1].writes.clear();
        assert!(matches!(
            program.validate(),
            Err(ProgramValidationError::ReadWithoutProducer(_))
        ));
    }

    #[test]
    fn unreachable_nodes_and_missing_terminals_are_rejected() {
        let mut program = text_program();
        program.nodes.push(ProgramNode {
            node_id: NodeId(9),
            operation: sequence_op(),
            reads: vec![],
            writes: vec![],
            continuation: Continuation::Finish(TerminalProjection {
                close_outputs: vec![],
            }),
        });
        assert!(matches!(
            program.validate(),
            Err(ProgramValidationError::Unreachable(NodeId(9)))
        ));
    }

    #[test]
    fn every_output_needs_a_closing_terminal() {
        let mut program = text_program();
        program.outputs.push(CompiledOutputSpec {
            output: OutputId(7),
            space: RepresentationSpace::SampleTensor(3),
            max_elements: 1,
        });
        assert!(matches!(
            program.validate(),
            Err(ProgramValidationError::UnclosedOutput(OutputId(7)))
        ));
    }

    #[test]
    fn unknown_references_fail_closed() {
        let mut program = text_program();
        program.nodes[0].continuation = Continuation::Next(NodeId(99));
        assert!(matches!(
            program.validate(),
            Err(ProgramValidationError::UnknownTarget(NodeId(0), NodeId(99)))
        ));

        let mut bad_read = text_program();
        bad_read.nodes[0].reads = vec![SlotRead::Product(ProductSlotId(9))];
        assert!(matches!(
            bad_read.validate(),
            Err(ProgramValidationError::UnknownSlot(NodeId(0)))
        ));
    }

    #[test]
    fn select_requires_cases() {
        let mut program = text_program();
        program.nodes[0].continuation = Continuation::Select {
            cases: vec![],
            default: NodeId(1),
        };
        assert!(matches!(
            program.validate(),
            Err(ProgramValidationError::EmptySelect(NodeId(0)))
        ));
    }
}
