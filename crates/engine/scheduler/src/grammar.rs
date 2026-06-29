//! Engine-side structured-output grammars.
//!
//! Grammars compile on a background thread (requests whose grammar is not ready
//! sit in a `skipped_waiting` gate instead of the admission queue). The compiled
//! grammar contributes a per-step token mask shipped to the worker over the
//! existing `allowed_tokens` descriptor (bitmask framing is deferred until the
//! list form is too large).
//!
//! One real grammar is implemented: the guided-choice token trie (the
//! frontend tokenizes the choice strings — the engine has no tokenizer). The
//! compiler seam is where heavier grammars (regex/json-schema automata) plug
//! in; compilation is asynchronous precisely so a slow compiler cannot stall
//! admission.

use uniserve_core::RequestId;
use uniserve_engine_api::GrammarSpec;

/// A compiled grammar: a token trie over the allowed completions.
#[derive(Debug, Clone)]
pub struct CompiledGrammar {
 /// The allowed token-id sequences (deduplicated, empty sequences removed).
    sequences: Vec<Vec<u32>>,
}

impl CompiledGrammar {
    fn compile(spec: &GrammarSpec) -> Self {
        match spec {
            GrammarSpec::Choice(seqs) => {
                let mut sequences: Vec<Vec<u32>> =
                    seqs.iter().filter(|s| !s.is_empty()).cloned().collect();
                sequences.sort();
                sequences.dedup();
                Self { sequences }
            }
        }
    }
}

/// Per-request matcher state over a [`CompiledGrammar`].
#[derive(Debug, Clone)]
pub struct GrammarMatcher {
    grammar: CompiledGrammar,
 /// Tokens accepted so far.
    progress: Vec<u32>,
}

impl GrammarMatcher {
    pub fn new(grammar: CompiledGrammar) -> Self {
        Self {
            grammar,
            progress: Vec::new(),
        }
    }

 /// Whether one allowed sequence has been fully produced.
    pub fn is_complete(&self) -> bool {
        self.grammar.sequences.iter().any(|s| s == &self.progress)
    }

 /// The token ids allowed at the current position: the next token of every
 /// sequence the progress is a strict prefix of. Empty means no
 /// continuation is allowed (the request must terminate).
    pub fn allowed_next(&self) -> Vec<u32> {
        let mut allowed: Vec<u32> = self
            .grammar
            .sequences
            .iter()
            .filter(|s| s.len() > self.progress.len() && s[..self.progress.len()] == self.progress)
            .map(|s| s[self.progress.len()])
            .collect();
        allowed.sort();
        allowed.dedup();
        allowed
    }

 /// Record one accepted token.
    pub fn advance(&mut self, token: u32) {
        self.progress.push(token);
    }
}

/// Asynchronous grammar compiler (the `skipped_waiting` gate's other half):
/// requests submit their spec at enqueue; admission collects finished
/// compilations and only then queues the request.
pub struct GrammarCompiler {
    tx: crossbeam_channel::Sender<(RequestId, GrammarSpec)>,
    rx: crossbeam_channel::Receiver<(RequestId, CompiledGrammar)>,
    _thread: std::thread::JoinHandle<()>,
}

impl GrammarCompiler {
    pub fn new() -> Self {
        let (tx, job_rx) = crossbeam_channel::unbounded::<(RequestId, GrammarSpec)>();
        let (done_tx, rx) = crossbeam_channel::unbounded();
        let thread = std::thread::Builder::new()
            .name("grammar-compiler".into())
            .spawn(move || {
                while let Ok((id, spec)) = job_rx.recv() {
                    let compiled = CompiledGrammar::compile(&spec);
                    if done_tx.send((id, compiled)).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn grammar compiler thread");
        Self {
            tx,
            rx,
            _thread: thread,
        }
    }

 /// Submit one grammar for compilation.

 /// The send only fails if the background compiler thread has gone away
 /// (panicked, or the receiver was dropped). That is unrecoverable for this
 /// request — its compilation will never land in [`drain_ready`], so the
 /// request would otherwise sit in the `skipped_waiting` gate forever. We
 /// cannot fail the request from here (the gate is the scheduler's, not the
 /// compiler's), so surface the dead-compiler condition loudly instead of
 /// swallowing it silently.
    pub fn submit(&self, id: RequestId, spec: GrammarSpec) {
        if self.tx.send((id, spec)).is_err() {
            tracing::error!(
                ?id,
                "grammar compiler thread is gone; request will never be admitted from the structured-output gate"
            );
        }
    }

 /// Collect every compilation finished so far (non-blocking).
    pub fn drain_ready(&self) -> Vec<(RequestId, CompiledGrammar)> {
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
        let g = CompiledGrammar::compile(&GrammarSpec::Choice(vec![
            vec![10, 20, 30],
            vec![10, 25],
            vec![40],
        ]));
        let mut m = GrammarMatcher::new(g);
        assert_eq!(m.allowed_next(), vec![10, 40]);
        m.advance(10);
        assert_eq!(m.allowed_next(), vec![20, 25]);
        m.advance(25);
        assert!(m.is_complete());
        assert!(m.allowed_next().is_empty());
    }

    #[test]
    fn compiler_gate_roundtrip() {
        let c = GrammarCompiler::new();
        c.submit(RequestId(7), GrammarSpec::Choice(vec![vec![1, 2]]));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let ready = c.drain_ready();
            if !ready.is_empty() {
                assert_eq!(ready[0].0, RequestId(7));
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "compiler never answered"
            );
            std::thread::yield_now();
        }
    }
}
