//! Dormant program-cursor semantics for the bounded inference program.
//!
//! Value-level foundation for the scheduler's future `ProgramRuntime`
//! (`specs/unified_inference_runtime.md`, Slice 2). This module owns the pure
//! state-machine semantics — admission, side-effect-free transition planning,
//! result-validated resolution, deterministic invocation identity, and
//! duplicate-resolution rejection — over a validated
//! [`InferenceProgram`](crate::program::InferenceProgram). Placement binding,
//! resource ledgers, batching, and worker submission stay in the scheduler
//! crate when the target stack activates; nothing here touches production.
//!
//! The laws enforced here come straight from the migration plan:
//!
//! * Planning is side-effect-free — [`ProgramInstance::plan_next`] borrows
//!   immutably and returns a value; cursor state changes only in
//!   [`ProgramInstance::resolve`] after exact identity validation.
//! * Program transition replay is deterministic — a [`PlannedTransition`]
//!   carries the invocation identity (cursor version, node, ordinal) and
//!   resolving with a stale or mismatched identity is rejected, so applying a
//!   committed result twice cannot double-advance state.
//! * Every backedge increments its declared monotone counter; when the
//!   declared limit is reached the `Repeat` exits, so terminal resolution is
//!   guaranteed within `worst_case_transitions`.

use serde::{Deserialize, Serialize};

use crate::program::{
    Continuation, CounterId, InferenceProgram, NodeId, OperationTemplate, ProgramFingerprint,
    ProgramValidationError,
};

/// One planned (not yet executed) model-backed transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedTransition {
    pub program: ProgramFingerprint,
    pub cursor_version: u64,
    pub invocation_ordinal: u64,
    pub node: NodeId,
    pub operation: OperationTemplate,
}

/// Compact validated result of one committed transition. `result_tag` feeds
/// `Select` continuations; the closed tag set is declared by the operation's
/// result contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommittedResult {
    pub result_tag: u32,
}

/// What one committed resolution did to the program.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgramOutcome {
    /// The cursor advanced to another model-backed node.
    Advanced { next: NodeId },
    /// A terminal node committed; the listed outputs close.
    Finished { closed_outputs: Vec<u16> },
}

/// A resolution that cannot be applied.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ResolutionError {
    #[error("planned transition names program {planned:?} but instance runs {actual:?}")]
    ForeignProgram {
        planned: Box<ProgramFingerprint>,
        actual: Box<ProgramFingerprint>,
    },
    #[error("planned cursor version {planned} does not match instance version {actual}")]
    StaleCursor { planned: u64, actual: u64 },
    #[error("planned node {planned:?} does not match cursor node {actual:?}")]
    NodeMismatch { planned: NodeId, actual: NodeId },
    #[error("instance already reached a terminal state")]
    AlreadyTerminal,
}

/// One admitted program incarnation: the validated program plus its cursor.
///
/// The cursor is exactly the spec's logical truth — node, version, ordinal,
/// bounded counters, terminal state — with no tokenizer, parser, tensor, or
/// transport state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgramInstance {
    program: InferenceProgram,
    fingerprint: ProgramFingerprint,
    cursor_version: u64,
    invocation_ordinal: u64,
    node: NodeId,
    counters: Vec<(CounterId, u32)>,
    terminal: Option<Vec<u16>>,
}

impl ProgramInstance {
    /// Admit one program: full contract validation, then a cursor at the
    /// entry node with zeroed counters.
    pub fn admit(program: InferenceProgram) -> Result<Self, ProgramValidationError> {
        program.validate()?;
        let fingerprint = program.fingerprint();
        let counters = program
            .counters
            .iter()
            .map(|spec| (spec.counter, 0_u32))
            .collect();
        let entry = program.entry;
        Ok(Self {
            program,
            fingerprint,
            cursor_version: 0,
            invocation_ordinal: 0,
            node: entry,
            counters,
            terminal: None,
        })
    }

    pub fn is_terminal(&self) -> bool {
        self.terminal.is_some()
    }

    pub fn cursor_version(&self) -> u64 {
        self.cursor_version
    }

    /// Plan the next model-backed transition. Pure: borrows immutably and
    /// changes nothing; calling it twice returns the identical value.
    pub fn plan_next(&self) -> Option<PlannedTransition> {
        if self.terminal.is_some() {
            return None;
        }
        let node = self.node_ref(self.node);
        Some(PlannedTransition {
            program: self.fingerprint,
            cursor_version: self.cursor_version,
            invocation_ordinal: self.invocation_ordinal,
            node: node.node_id,
            operation: node.operation.clone(),
        })
    }

    /// Apply one committed result to the planned transition. Validates the
    /// exact invocation identity before any mutation, then advances the
    /// cursor along the node's continuation.
    pub fn resolve(
        &mut self,
        planned: &PlannedTransition,
        result: CommittedResult,
    ) -> Result<ProgramOutcome, ResolutionError> {
        if self.terminal.is_some() {
            return Err(ResolutionError::AlreadyTerminal);
        }
        if planned.program != self.fingerprint {
            return Err(ResolutionError::ForeignProgram {
                planned: Box::new(planned.program),
                actual: Box::new(self.fingerprint),
            });
        }
        if planned.cursor_version != self.cursor_version {
            return Err(ResolutionError::StaleCursor {
                planned: planned.cursor_version,
                actual: self.cursor_version,
            });
        }
        if planned.node != self.node {
            return Err(ResolutionError::NodeMismatch {
                planned: planned.node,
                actual: self.node,
            });
        }
        let continuation = self.node_ref(self.node).continuation.clone();
        let outcome = match continuation {
            Continuation::Next(next) => self.advance_to(next),
            Continuation::Repeat {
                counter,
                limit,
                body,
                exit,
            } => {
                let count = self.counter_mut(counter);
                if *count < limit.get() {
                    *count += 1;
                    let target = body;
                    self.advance_to(target)
                } else {
                    self.advance_to(exit)
                }
            }
            Continuation::Select { cases, default } => {
                let next = cases
                    .iter()
                    .find(|case| case.result_tag == result.result_tag)
                    .map(|case| case.next)
                    .unwrap_or(default);
                self.advance_to(next)
            }
            Continuation::Finish(projection) => {
                let closed: Vec<u16> = projection
                    .close_outputs
                    .iter()
                    .map(|output| output.0)
                    .collect();
                self.terminal = Some(closed.clone());
                ProgramOutcome::Finished {
                    closed_outputs: closed,
                }
            }
        };
        self.cursor_version += 1;
        self.invocation_ordinal += 1;
        Ok(outcome)
    }

    fn advance_to(&mut self, next: NodeId) -> ProgramOutcome {
        self.node = next;
        ProgramOutcome::Advanced { next }
    }

    fn node_ref(&self, node_id: NodeId) -> &crate::program::ProgramNode {
        self.program
            .nodes
            .iter()
            .find(|node| node.node_id == node_id)
            .expect("validated programs contain every continuation target")
    }

    fn counter_mut(&mut self, counter: CounterId) -> &mut u32 {
        &mut self
            .counters
            .iter_mut()
            .find(|(id, _)| *id == counter)
            .expect("validated repeats use declared counters")
            .1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::{
        CompiledOutputSpec, CounterSpec, InputSlotId, InputSlotSpec, OutputId,
        PROGRAM_SCHEMA_VERSION, ProgramNode, RepresentationSpace, RouteId, SelectCase,
        SequenceStepTemplate, SessionSlotId, SessionSlotSpec, SlotRead, TerminalProjection,
    };
    use std::num::NonZeroU32;

    fn seq_op() -> OperationTemplate {
        OperationTemplate::Sequence(SequenceStepTemplate {
            route: RouteId(0),
            session: SessionSlotId(0),
            max_positions: NonZeroU32::new(1).expect("nonzero"),
        })
    }

    /// prefill -> decode loop (limit 3) -> finish
    fn looped_program() -> InferenceProgram {
        InferenceProgram {
            schema_version: PROGRAM_SCHEMA_VERSION,
            inputs: vec![InputSlotSpec {
                slot: InputSlotId(0),
                space: RepresentationSpace::DiscreteSequence(1),
                elements: 16,
            }],
            products: vec![],
            sessions: vec![SessionSlotSpec {
                slot: SessionSlotId(0),
            }],
            counters: vec![CounterSpec {
                counter: CounterId(0),
            }],
            nodes: vec![
                ProgramNode {
                    node_id: NodeId(0),
                    operation: seq_op(),
                    reads: vec![SlotRead::Input(InputSlotId(0))],
                    writes: vec![],
                    continuation: Continuation::Next(NodeId(1)),
                },
                ProgramNode {
                    node_id: NodeId(1),
                    operation: seq_op(),
                    reads: vec![],
                    writes: vec![],
                    continuation: Continuation::Repeat {
                        counter: CounterId(0),
                        limit: NonZeroU32::new(3).expect("nonzero"),
                        body: NodeId(1),
                        exit: NodeId(2),
                    },
                },
                ProgramNode {
                    node_id: NodeId(2),
                    operation: seq_op(),
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
                max_elements: 8,
            }],
        }
    }

    fn drive_to_terminal(instance: &mut ProgramInstance) -> (u32, Vec<u16>) {
        let mut transitions = 0_u32;
        loop {
            let planned = match instance.plan_next() {
                Some(planned) => planned,
                None => panic!("no terminal reached"),
            };
            transitions += 1;
            match instance
                .resolve(&planned, CommittedResult { result_tag: 0 })
                .expect("resolution applies")
            {
                ProgramOutcome::Advanced { .. } => continue,
                ProgramOutcome::Finished { closed_outputs } => {
                    return (transitions, closed_outputs);
                }
            }
        }
    }

    #[test]
    fn loop_runs_exactly_its_declared_limit_then_finishes() {
        let mut instance = ProgramInstance::admit(looped_program()).expect("admits");
        let (transitions, closed) = drive_to_terminal(&mut instance);
        // entry + (limit=3 body re-entries + 1 exhausted exit) at the repeat
        // node + terminal = 6 resolutions, bounded by worst_case_transitions.
        assert_eq!(transitions, 6);
        assert!(u64::from(transitions) <= looped_program().worst_case_transitions());
        assert_eq!(closed, vec![0]);
        assert!(instance.is_terminal());
        assert!(instance.plan_next().is_none());
    }

    #[test]
    fn planning_is_side_effect_free_and_deterministic() {
        let instance = ProgramInstance::admit(looped_program()).expect("admits");
        let first = instance.plan_next().expect("plans");
        let second = instance.plan_next().expect("plans");
        assert_eq!(first, second);
        assert_eq!(instance.cursor_version(), 0);
    }

    #[test]
    fn duplicate_and_stale_resolutions_are_rejected() {
        let mut instance = ProgramInstance::admit(looped_program()).expect("admits");
        let planned = instance.plan_next().expect("plans");
        instance
            .resolve(&planned, CommittedResult { result_tag: 0 })
            .expect("first resolution applies");
        // Replaying the same committed transition cannot double-advance.
        assert_eq!(
            instance.resolve(&planned, CommittedResult { result_tag: 0 }),
            Err(ResolutionError::StaleCursor {
                planned: 0,
                actual: 1
            })
        );
    }

    #[test]
    fn select_routes_by_result_tag_with_default() {
        let mut program = looped_program();
        program.nodes[0].continuation = Continuation::Select {
            cases: vec![SelectCase {
                result_tag: 7,
                next: NodeId(2),
            }],
            default: NodeId(1),
        };
        let mut by_case = ProgramInstance::admit(program.clone()).expect("admits");
        let planned = by_case.plan_next().expect("plans");
        assert_eq!(
            by_case.resolve(&planned, CommittedResult { result_tag: 7 }),
            Ok(ProgramOutcome::Advanced { next: NodeId(2) })
        );
        let mut by_default = ProgramInstance::admit(program).expect("admits");
        let planned = by_default.plan_next().expect("plans");
        assert_eq!(
            by_default.resolve(&planned, CommittedResult { result_tag: 9 }),
            Ok(ProgramOutcome::Advanced { next: NodeId(1) })
        );
    }

    #[test]
    fn foreign_transitions_are_rejected() {
        let mut instance = ProgramInstance::admit(looped_program()).expect("admits");
        let mut other_program = looped_program();
        other_program.outputs[0].max_elements = 99;
        let other = ProgramInstance::admit(other_program).expect("admits");
        let foreign = other.plan_next().expect("plans");
        assert!(matches!(
            instance.resolve(&foreign, CommittedResult { result_tag: 0 }),
            Err(ResolutionError::ForeignProgram { .. })
        ));
    }
}
