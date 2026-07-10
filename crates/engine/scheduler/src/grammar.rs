//! Scheduler-side structured-output enforcement.
//!
//! Choices use a compact token trie. Rich constraints arrive as tokenizer-
//! specific XGrammar artifacts compiled by the serving runtime; admission
//! validates the artifact off the scheduler loop, and each request owns only
//! its mutable matcher state.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use uniserve_core::RequestId;
use uniserve_engine_api::GrammarSpec;
use xgrammar::{
    DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLTensor,
    GrammarMatcher as XGrammarMatcher, TokenizerInfo, allocate_token_bitmask, get_bitmask_shape,
};

#[derive(Debug)]
pub enum CompiledGrammar {
    Choice {
        sequences: Vec<Vec<u32>>,
    },
    XGrammar {
        token_bytes: Vec<Vec<u8>>,
        compiled_grammar_json: String,
        stop_token_ids: Vec<u32>,
    },
}

impl CompiledGrammar {
    fn compile(spec: GrammarSpec) -> Result<Self, String> {
        match spec {
            GrammarSpec::Choice {
                token_sequences: sequences,
            } => {
                let mut sequences: Vec<Vec<u32>> = sequences
                    .into_iter()
                    .filter(|sequence| !sequence.is_empty())
                    .collect();
                sequences.sort();
                sequences.dedup();
                if sequences.is_empty() {
                    return Err("choice grammar has no non-empty alternatives".to_string());
                }
                Ok(Self::Choice { sequences })
            }
            GrammarSpec::Compiled {
                token_bytes,
                compiled_grammar_json,
                stop_token_ids,
            } => {
                validate_xgrammar_artifact(&token_bytes, &compiled_grammar_json, &stop_token_ids)?;
                Ok(Self::XGrammar {
                    token_bytes,
                    compiled_grammar_json,
                    stop_token_ids,
                })
            }
        }
    }
}

pub enum GrammarMatcher {
    Choice {
        sequences: Vec<Vec<u32>>,
        progress: Vec<u32>,
    },
    XGrammar {
        session: XGrammarSession,
    },
}

impl GrammarMatcher {
    fn new(grammar: CompiledGrammar, runtime: &GrammarRuntime) -> Result<Self, String> {
        match grammar {
            CompiledGrammar::Choice { sequences } => Ok(Self::Choice {
                sequences,
                progress: Vec::new(),
            }),
            CompiledGrammar::XGrammar {
                token_bytes,
                compiled_grammar_json,
                stop_token_ids,
            } => Ok(Self::XGrammar {
                session: runtime.create(token_bytes, compiled_grammar_json, stop_token_ids)?,
            }),
        }
    }

    pub fn next_mask(&mut self) -> Result<(bool, Vec<u32>), String> {
        match self {
            Self::Choice {
                sequences,
                progress,
            } => {
                let complete = sequences.iter().any(|sequence| sequence == progress);
                let mut allowed = sequences
                    .iter()
                    .filter(|sequence| {
                        sequence.len() > progress.len()
                            && sequence[..progress.len()] == progress[..]
                    })
                    .map(|sequence| sequence[progress.len()])
                    .collect::<Vec<_>>();
                allowed.sort_unstable();
                allowed.dedup();
                Ok((complete, allowed))
            }
            Self::XGrammar { session } => session.next_mask(),
        }
    }

    pub fn advance(&mut self, token: u32) -> Result<(), String> {
        match self {
            Self::Choice { progress, .. } => {
                progress.push(token);
                Ok(())
            }
            Self::XGrammar { session } => session.advance(token),
        }
    }
}

pub(crate) fn grammar_allowed_tokens(
    complete: bool,
    mut continuations: Vec<u32>,
    stop_token_ids: &[u32],
) -> Vec<u32> {
    if complete {
        continuations.extend_from_slice(stop_token_ids);
    }
    continuations.sort_unstable();
    continuations.dedup();
    continuations
}

#[derive(Clone)]
struct GrammarRuntime {
    tx: crossbeam_channel::Sender<XGrammarCommand>,
    next_session_id: Arc<AtomicU64>,
}

impl GrammarRuntime {
    fn create(
        &self,
        token_bytes: Vec<Vec<u8>>,
        compiled_grammar_json: String,
        stop_token_ids: Vec<u32>,
    ) -> Result<XGrammarSession, String> {
        let id = self.next_session_id.fetch_add(1, Ordering::Relaxed);
        let (response_tx, response_rx) = crossbeam_channel::bounded(1);
        self.tx
            .send(XGrammarCommand::Create {
                id,
                token_bytes,
                compiled_grammar_json,
                stop_token_ids,
                response: response_tx,
            })
            .map_err(|_| "grammar runtime thread is unavailable".to_string())?;
        response_rx
            .recv()
            .map_err(|_| "grammar runtime dropped a create response".to_string())??;
        Ok(XGrammarSession {
            id,
            runtime: self.clone(),
        })
    }
}

pub struct XGrammarSession {
    id: u64,
    runtime: GrammarRuntime,
}

impl XGrammarSession {
    fn next_mask(&self) -> Result<(bool, Vec<u32>), String> {
        let (response_tx, response_rx) = crossbeam_channel::bounded(1);
        self.runtime
            .tx
            .send(XGrammarCommand::NextMask {
                id: self.id,
                response: response_tx,
            })
            .map_err(|_| "grammar runtime thread is unavailable".to_string())?;
        response_rx
            .recv()
            .map_err(|_| "grammar runtime dropped a mask response".to_string())?
    }

    fn advance(&self, token: u32) -> Result<(), String> {
        let (response_tx, response_rx) = crossbeam_channel::bounded(1);
        self.runtime
            .tx
            .send(XGrammarCommand::Advance {
                id: self.id,
                token,
                response: response_tx,
            })
            .map_err(|_| "grammar runtime thread is unavailable".to_string())?;
        response_rx
            .recv()
            .map_err(|_| "grammar runtime dropped an advance response".to_string())?
    }
}

impl Drop for XGrammarSession {
    fn drop(&mut self) {
        let _ = self.runtime.tx.send(XGrammarCommand::Drop { id: self.id });
    }
}

enum XGrammarCommand {
    Create {
        id: u64,
        token_bytes: Vec<Vec<u8>>,
        compiled_grammar_json: String,
        stop_token_ids: Vec<u32>,
        response: crossbeam_channel::Sender<Result<(), String>>,
    },
    NextMask {
        id: u64,
        response: crossbeam_channel::Sender<Result<(bool, Vec<u32>), String>>,
    },
    Advance {
        id: u64,
        token: u32,
        response: crossbeam_channel::Sender<Result<(), String>>,
    },
    Drop {
        id: u64,
    },
}

struct XGrammarState {
    matcher: XGrammarMatcher,
    vocab_size: usize,
    stop_token_ids: Vec<u32>,
    cached_allowed: Option<Vec<u32>>,
}

fn run_grammar_runtime(rx: crossbeam_channel::Receiver<XGrammarCommand>) {
    let mut sessions = HashMap::<u64, XGrammarState>::new();
    while let Ok(command) = rx.recv() {
        match command {
            XGrammarCommand::Create {
                id,
                token_bytes,
                compiled_grammar_json,
                stop_token_ids,
                response,
            } => {
                let result =
                    create_xgrammar_state(token_bytes, compiled_grammar_json, stop_token_ids).map(
                        |state| {
                            sessions.insert(id, state);
                        },
                    );
                let _ = response.send(result);
            }
            XGrammarCommand::NextMask { id, response } => {
                let result = sessions
                    .get_mut(&id)
                    .ok_or_else(|| format!("grammar session {id} does not exist"))
                    .map(|state| {
                        let allowed = state
                            .cached_allowed
                            .get_or_insert_with(|| {
                                xgrammar_allowed_tokens(&mut state.matcher, state.vocab_size)
                            })
                            .clone();
                        let complete = !allowed.is_empty()
                            && allowed
                                .iter()
                                .all(|token_id| state.stop_token_ids.contains(token_id));
                        (complete, allowed)
                    });
                let _ = response.send(result);
            }
            XGrammarCommand::Advance {
                id,
                token,
                response,
            } => {
                let result = sessions
                    .get_mut(&id)
                    .ok_or_else(|| format!("grammar session {id} does not exist"))
                    .and_then(|state| {
                        let token = i32::try_from(token)
                            .map_err(|_| format!("sampled token id {token} exceeds i32"))?;
                        if !state.matcher.accept_token(token) {
                            return Err(format!(
                                "sampled token {token} was rejected by the structured-output matcher"
                            ));
                        }
                        state.cached_allowed = None;
                        Ok(())
                    });
                let _ = response.send(result);
            }
            XGrammarCommand::Drop { id } => {
                sessions.remove(&id);
            }
        }
    }
}

fn create_xgrammar_state(
    token_bytes: Vec<Vec<u8>>,
    compiled_grammar_json: String,
    stop_token_ids: Vec<u32>,
) -> Result<XGrammarState, String> {
    let tokenizer_info = tokenizer_info(&token_bytes, &stop_token_ids)?;
    let compiled =
        xgrammar::CompiledGrammar::deserialize_json(&compiled_grammar_json, &tokenizer_info)?;
    let stop_ids = stop_token_ids
        .iter()
        .copied()
        .map(|id| i32::try_from(id).map_err(|_| format!("grammar stop token id {id} exceeds i32")))
        .collect::<Result<Vec<_>, _>>()?;
    let matcher = XGrammarMatcher::new(&compiled, Some(&stop_ids), false, -1)?;
    Ok(XGrammarState {
        matcher,
        vocab_size: token_bytes.len(),
        stop_token_ids,
        cached_allowed: None,
    })
}

fn validate_xgrammar_artifact(
    token_bytes: &[Vec<u8>],
    compiled_grammar_json: &str,
    stop_token_ids: &[u32],
) -> Result<(), String> {
    let tokenizer_info = tokenizer_info(token_bytes, stop_token_ids)?;
    xgrammar::CompiledGrammar::deserialize_json(compiled_grammar_json, &tokenizer_info).map(|_| ())
}

fn tokenizer_info(
    token_bytes: &[Vec<u8>],
    stop_token_ids: &[u32],
) -> Result<TokenizerInfo, String> {
    if token_bytes.is_empty() {
        return Err("compiled grammar has an empty tokenizer vocabulary".to_string());
    }
    if let Some(id) = stop_token_ids
        .iter()
        .find(|id| **id as usize >= token_bytes.len())
    {
        return Err(format!(
            "grammar stop token id {id} is outside vocabulary size {}",
            token_bytes.len()
        ));
    }
    let stops = stop_token_ids
        .iter()
        .copied()
        .map(|id| i32::try_from(id).map_err(|_| format!("grammar stop token id {id} exceeds i32")))
        .collect::<Result<Vec<_>, _>>()?;
    let metadata = serde_json::json!({
        "vocab_type": 0,
        "vocab_size": token_bytes.len(),
        "add_prefix_space": false,
        "stop_token_ids": stops,
    })
    .to_string();
    Ok(TokenizerInfo::from_vocab_and_metadata_bytes(
        token_bytes.iter(),
        &metadata,
    ))
}

fn xgrammar_allowed_tokens(matcher: &mut XGrammarMatcher, vocab_size: usize) -> Vec<u32> {
    let mut bitmask = allocate_token_bitmask(1, vocab_size);
    let (_, bitmask_size) = get_bitmask_shape(1, vocab_size);
    let mut shape = [1i64, bitmask_size as i64];
    let mut strides = [bitmask_size as i64, 1];
    let mut tensor = DLTensor {
        data: bitmask.as_mut_ptr().cast(),
        device: DLDevice {
            device_type: DLDeviceType::kDLCPU,
            device_id: 0,
        },
        ndim: 2,
        dtype: DLDataType {
            code: DLDataTypeCode::kDLInt as u8,
            bits: 32,
            lanes: 1,
        },
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    };
    let constrained = matcher.fill_next_token_bitmask(&mut tensor, 0, false);
    if !constrained {
        return (0..vocab_size.min(u32::MAX as usize))
            .map(|id| id as u32)
            .collect();
    }
    (0..vocab_size)
        .filter(|id| {
            let word = id / 32;
            let bit = id % 32;
            bitmask
                .get(word)
                .is_some_and(|mask| (*mask & (1i32 << bit)) != 0)
        })
        .map(|id| id as u32)
        .collect()
}

pub struct GrammarCompiler {
    tx: crossbeam_channel::Sender<(RequestId, GrammarSpec)>,
    rx: crossbeam_channel::Receiver<(RequestId, Result<CompiledGrammar, String>)>,
    runtime: GrammarRuntime,
    _thread: std::thread::JoinHandle<()>,
    _runtime_thread: std::thread::JoinHandle<()>,
}

impl GrammarCompiler {
    pub fn new() -> Self {
        let (tx, job_rx) = crossbeam_channel::unbounded::<(RequestId, GrammarSpec)>();
        let (done_tx, rx) = crossbeam_channel::unbounded();
        let thread = std::thread::Builder::new()
            .name("grammar-compiler".into())
            .spawn(move || {
                while let Ok((id, spec)) = job_rx.recv() {
                    if done_tx.send((id, CompiledGrammar::compile(spec))).is_err() {
                        break;
                    }
                }
            })
            .unwrap_or_else(|error| panic!("failed to spawn grammar compiler: {error}"));
        let (runtime_tx, runtime_rx) = crossbeam_channel::unbounded();
        let runtime_thread = std::thread::Builder::new()
            .name("grammar-runtime".into())
            .spawn(move || run_grammar_runtime(runtime_rx))
            .unwrap_or_else(|error| panic!("failed to spawn grammar runtime: {error}"));
        Self {
            tx,
            rx,
            runtime: GrammarRuntime {
                tx: runtime_tx,
                next_session_id: Arc::new(AtomicU64::new(1)),
            },
            _thread: thread,
            _runtime_thread: runtime_thread,
        }
    }

    pub fn create_matcher(&self, grammar: CompiledGrammar) -> Result<GrammarMatcher, String> {
        GrammarMatcher::new(grammar, &self.runtime)
    }

    pub fn submit(&self, id: RequestId, spec: GrammarSpec) {
        if self.tx.send((id, spec)).is_err() {
            tracing::error!(
                ?id,
                "grammar compiler thread is gone; request will never be admitted from the structured-output gate"
            );
        }
    }

    pub fn drain_ready(&self) -> Vec<(RequestId, Result<CompiledGrammar, String>)> {
        let mut out = Vec::new();
        while let Ok(item) = self.rx.try_recv() {
            out.push(item);
        }
        out
    }
}

impl Default for GrammarCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choice_trie_masks_each_step() {
        let grammar = CompiledGrammar::compile(GrammarSpec::Choice {
            token_sequences: vec![vec![10, 20, 30], vec![10, 25], vec![40]],
        })
        .unwrap();
        let compiler = GrammarCompiler::new();
        let mut matcher = compiler.create_matcher(grammar).unwrap();
        assert_eq!(matcher.next_mask().unwrap(), (false, vec![10, 40]));
        matcher.advance(10).unwrap();
        assert_eq!(matcher.next_mask().unwrap(), (false, vec![20, 25]));
        matcher.advance(25).unwrap();
        assert_eq!(matcher.next_mask().unwrap(), (true, Vec::new()));
    }

    #[test]
    fn completed_choice_keeps_longer_alternative_reachable() {
        let grammar = CompiledGrammar::compile(GrammarSpec::Choice {
            token_sequences: vec![vec![10], vec![10, 20]],
        })
        .unwrap();
        let compiler = GrammarCompiler::new();
        let mut matcher = compiler.create_matcher(grammar).unwrap();

        matcher.advance(10).unwrap();
        let (complete, continuations) = matcher.next_mask().unwrap();
        assert!(complete);
        assert_eq!(continuations, vec![20]);
        assert_eq!(
            grammar_allowed_tokens(complete, continuations, &[2, 3]),
            vec![2, 3, 20]
        );
    }

    #[test]
    fn compiler_gate_roundtrip() {
        let compiler = GrammarCompiler::new();
        compiler.submit(
            RequestId(7),
            GrammarSpec::Choice {
                token_sequences: vec![vec![1, 2]],
            },
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let ready = compiler.drain_ready();
            if !ready.is_empty() {
                assert_eq!(ready[0].0, RequestId(7));
                assert!(ready[0].1.is_ok());
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "compiler never answered"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn xgrammar_runtime_enforces_compiled_regex_masks() {
        let mut token_bytes = (0_u8..=126).map(|byte| vec![byte]).collect::<Vec<_>>();
        token_bytes.push(vec![0xff]);
        let tokenizer_info = tokenizer_info(&token_bytes, &[127]).unwrap();
        let mut xcompiler =
            xgrammar::GrammarCompiler::new(&tokenizer_info, 1, true, 64 * 1024 * 1024).unwrap();
        let compiled = xcompiler.compile_regex("a(b|c)").unwrap();
        let grammar = CompiledGrammar::compile(GrammarSpec::Compiled {
            token_bytes,
            compiled_grammar_json: compiled.serialize_json(),
            stop_token_ids: vec![127],
        })
        .unwrap();
        let compiler = GrammarCompiler::new();
        let mut matcher = compiler.create_matcher(grammar).unwrap();

        let (complete, allowed) = matcher.next_mask().unwrap();
        assert!(!complete);
        assert!(allowed.contains(&u32::from(b'a')));
        matcher.advance(u32::from(b'a')).unwrap();
        let (_, allowed) = matcher.next_mask().unwrap();
        assert!(allowed.contains(&u32::from(b'b')));
        assert!(allowed.contains(&u32::from(b'c')));
    }
}
