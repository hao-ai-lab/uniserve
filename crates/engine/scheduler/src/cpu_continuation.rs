//! Bounded per-lineage CPU continuations for configured token processors.

use std::sync::Arc;

use uniserve_core::{RequestId, SamplingParams};

use crate::logits::{LogitsProcessor, ProcCtx, run_pipeline_checked};

const CPU_TASK_CAPACITY: usize = 256;
const CPU_WORKERS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    pub n_generated: usize,
    pub eos: Vec<u32>,
    pub generated: Vec<u32>,
    pub sampling: SamplingParams,
    pub pipeline: Vec<Arc<dyn LogitsProcessor>>,
}

pub(crate) struct CpuResult {
    pub key: CpuTaskKey,
    pub outcome: Result<CpuMasks, String>,
}

impl CpuTask {
    fn execute(self) -> CpuResult {
        let outcome = (|| {
            let ctx = ProcCtx {
                n_generated: self.n_generated,
                eos: &self.eos,
                generated: &self.generated,
                sampling: &self.sampling,
            };
            let (allowed, suppress) = run_pipeline_checked(&self.pipeline, &ctx)?;
            Ok(CpuMasks { allowed, suppress })
        })();
        CpuResult {
            key: self.key,
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
