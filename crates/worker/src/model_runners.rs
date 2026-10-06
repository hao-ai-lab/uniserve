//! Routing and homogeneous batches for the worker's numerical runners.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use uniserve_worker_ipc::{Call, CallKind};

use crate::{Error, Result};

/// One numerical invocation and the rows whose outputs it produces.
pub struct ModelBatch<'a, R> {
    pub runner: &'a R,
    pub rows: Vec<usize>,
    /// A later batch reuses this runner's graph or output buffers.
    pub preserve_output: bool,
}

/// Component operations bound to numerical runners at worker startup.
/// Runners own their device, buffers, and graphs; this table owns routing and
/// preserves first-appearance order when combining compatible input rows.
pub struct ModelRunners<R> {
    routes: HashMap<String, HashMap<CallKind, usize>>,
    runners: Vec<R>,
}

impl<R> Default for ModelRunners<R> {
    fn default() -> Self {
        Self {
            routes: HashMap::new(),
            runners: Vec::new(),
        }
    }
}

impl<R> ModelRunners<R> {
    pub fn clear(&mut self) {
        self.routes.clear();
        self.runners.clear();
    }

    /// Bind every operation of one runner, rejecting ambiguous dispatch.
    pub fn bind(&mut self, component: &str, kinds: &[CallKind], runner: R) -> Result<()> {
        let routes = self.routes.entry(component.into()).or_default();
        if let Some(kind) = kinds.iter().find(|kind| routes.contains_key(kind)) {
            return Err(Error::Invalid(format!(
                "computation {component}.{} has multiple lane bindings",
                kind.as_str()
            )));
        }

        let index = self.runners.len();
        self.runners.push(runner);
        routes.extend(kinds.iter().map(|&kind| (kind, index)));
        Ok(())
    }

    pub fn get(&self, component: &str, kind: CallKind) -> Option<&R> {
        self.route(component, kind)
            .map(|index| &self.runners[index])
    }

    /// First runner of an operation in startup binding order.
    pub fn first(&self, kind: CallKind) -> Option<&R> {
        self.routes
            .values()
            .filter_map(|routes| routes.get(&kind))
            .min()
            .map(|&index| &self.runners[index])
    }

    /// Group rows by their runner, operation, and numerical input format.
    /// The backend supplies `K` (row type and shape); native calls determine
    /// whether context segments require per-row attention causality.
    /// Unbound rows are reported separately so their errors precede execution.
    pub fn group<'a, K: Eq + Hash>(
        &self,
        rows: impl IntoIterator<Item = (&'a Call, CallKind, K, Option<bool>)>,
    ) -> (Vec<usize>, Vec<ModelBatch<'_, R>>) {
        let mut missing = Vec::new();
        let mut positions = HashMap::new();
        let mut batches: Vec<ModelBatch<'_, R>> = Vec::new();
        let mut runners = Vec::new();

        for (index, (call, kind, format, causal)) in rows.into_iter().enumerate() {
            let Some(runner) = self.route(&call.component, kind) else {
                missing.push(index);
                continue;
            };
            // Interdependent text and vision segments write all their K/V
            // before attending. Keep them together with device causality flags.
            let causal = if !call.vision_inputs.is_empty() && call.completion_output.is_none() {
                None
            } else {
                causal
            };
            let position = *positions
                .entry((runner, kind, format, causal))
                .or_insert_with(|| {
                    let position = batches.len();
                    batches.push(ModelBatch {
                        runner: &self.runners[runner],
                        rows: Vec::new(),
                        preserve_output: false,
                    });
                    runners.push(runner);
                    position
                });
            batches[position].rows.push(index);
        }

        // Only the final invocation can lend its output directly. Earlier
        // invocations must survive reuse of the same backing by later batches.
        let mut seen = HashSet::new();
        for (batch, runner) in batches.iter_mut().zip(runners).rev() {
            batch.preserve_output = !seen.insert(runner);
        }
        (missing, batches)
    }

    fn route(&self, component: &str, kind: CallKind) -> Option<usize> {
        self.routes.get(component)?.get(&kind).copied()
    }
}
