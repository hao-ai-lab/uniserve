//! Bounded per-lineage CPU continuations for grammar and custom token masks.

use std::sync::Arc;

use uniserve_core::{RequestId, SamplingParams};

use crate::grammar::{GrammarMatcher, grammar_allowed_tokens};
use crate::logits::{LogitsProcessor, ProcCtx, run_pipeline_checked};

const CPU_TASK_CAPACITY: usize = 256;
const CPU_WORKERS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CpuTaskKey {
    pub request_id: RequestId,
    pub epoch: u64,
    pub point: u64,
    pub generation: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct CpuMasks {
    pub allowed: Option<Vec<u32>>,
    pub suppress: Option<Vec<u32>>,
}

pub(crate) struct CpuTask {
    pub key: CpuTaskKey,
    pub matcher: Option<GrammarMatcher>,
    pub tokens_to_advance: Vec<u32>,
    pub n_generated: usize,
    pub eos: Vec<u32>,
    pub generated: Vec<u32>,
    pub sampling: SamplingParams,
    pub grammar_stops: Vec<u32>,
    pub pipeline: Vec<Arc<dyn LogitsProcessor>>,
}

pub(crate) struct CpuResult {
    pub key: CpuTaskKey,
    pub matcher: Option<GrammarMatcher>,
    pub outcome: Result<CpuMasks, String>,
}

impl CpuTask {
    fn execute(mut self) -> CpuResult {
        let outcome = (|| {
            if let Some(matcher) = self.matcher.as_mut() {
                for token in self.tokens_to_advance.iter().copied() {
                    matcher.advance(token)?;
                }
            }
            let ctx = ProcCtx {
                n_generated: self.n_generated,
                eos: &self.eos,
                generated: &self.generated,
                sampling: &self.sampling,
            };
            let (mut allowed, suppress) = run_pipeline_checked(&self.pipeline, &ctx)?;
            if let Some(matcher) = self.matcher.as_mut() {
                let (complete, continuations) = matcher.next_mask()?;
                let grammar = grammar_allowed_tokens(complete, continuations, &self.grammar_stops);
                allowed = Some(match allowed {
                    Some(tokens) => tokens
                        .into_iter()
                        .filter(|token| grammar.contains(token))
                        .collect(),
                    None => grammar,
                });
            }
            Ok(CpuMasks { allowed, suppress })
        })();
        CpuResult {
            key: self.key,
            matcher: self.matcher,
            outcome,
        }
    }
}

pub(crate) struct CpuContinuationPool {
    tx: crossbeam_channel::Sender<Box<CpuTask>>,
    rx: crossbeam_channel::Receiver<CpuResult>,
    _workers: Vec<std::thread::JoinHandle<()>>,
}

impl CpuContinuationPool {
    pub(crate) fn new(waker: uniserve_core::CommandWaker) -> Self {
        let (tx, task_rx) = crossbeam_channel::bounded::<Box<CpuTask>>(CPU_TASK_CAPACITY);
        let (result_tx, rx) = crossbeam_channel::bounded::<CpuResult>(CPU_TASK_CAPACITY);
        let mut workers = Vec::with_capacity(CPU_WORKERS);
        for index in 0..CPU_WORKERS {
            let task_rx = task_rx.clone();
            let result_tx = result_tx.clone();
            let waker = waker.clone();
            workers.push(
                std::thread::Builder::new()
                    .name(format!("cpu-continuation-{index}"))
                    .spawn(move || {
                        while let Ok(task) = task_rx.recv() {
                            if result_tx.send((*task).execute()).is_err() {
                                break;
                            }
                            waker.wake();
                        }
                    })
                    .unwrap_or_else(|error| {
                        panic!("failed to spawn CPU continuation worker: {error}")
                    }),
            );
        }
        Self {
            tx,
            rx,
            _workers: workers,
        }
    }

    pub(crate) fn try_submit(&self, task: CpuTask) -> Result<(), Box<CpuTask>> {
        self.tx
            .try_send(Box::new(task))
            .map_err(|error| error.into_inner())
    }

    pub(crate) fn drain_ready(&self) -> Vec<CpuResult> {
        let mut ready = Vec::new();
        while let Ok(result) = self.rx.try_recv() {
            ready.push(result);
        }
        ready
    }
}
