//! Request slots, call dependencies, and accepted execution progress.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard};

use indexmap::IndexMap;
use uniserve_core::CallId;
use uniserve_worker_ipc::{CallStatus, CanvasStep, NewRequest, RequestKey};

use crate::{Error, Result};

/// Accepted coordinates of a state-consuming call.
///
/// KV lengths and logical positions count tokens. `flow_step` counts solver
/// steps; `rng_counter` counts draws in the request's sampling stream.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RequestProgress {
    pub logical_position: u64,
    pub rng_counter: u64,
    pub flow_step: u64,
    pub kv_visible_len: u64,
    pub kv_computed_len: u64,
    pub prompt_logits_ready: bool,
}

#[derive(Debug, Default)]
struct RequestState {
    progress: RequestProgress,
    // Last submitted canvas step, independent of accepted device results.
    canvas: Option<CanvasStep>,
    accepted_call: CallId,
    state_call: CallId,
    // Insertion order is submission order, which need not equal call-id order.
    pending: IndexMap<CallId, bool>,
    closed: bool,
    retired: bool,
}

/// One admitted epoch, retained by result observers even after slot reuse.
///
/// Only its pool changes execution state. Borrowers may inspect progress from
/// completion threads; no foreign callback runs while the state is locked.
#[derive(Debug)]
pub struct Request {
    admission: NewRequest,
    state: Mutex<RequestState>,
}

impl Request {
    fn new(admission: NewRequest) -> Self {
        let prefix = admission
            .ar
            .as_ref()
            .map_or(0, |ar| u64::from(ar.initial_position));

        Self {
            admission,
            state: Mutex::new(RequestState {
                progress: RequestProgress {
                    logical_position: prefix,
                    kv_visible_len: prefix,
                    kv_computed_len: prefix,
                    ..RequestProgress::default()
                },
                ..RequestState::default()
            }),
        }
    }

    pub fn admission(&self) -> &NewRequest {
        &self.admission
    }

    pub fn key(&self) -> RequestKey {
        self.admission.request_key
    }

    pub fn slot(&self) -> usize {
        self.admission.request_pool_idx as usize
    }

    pub fn progress(&self) -> Result<RequestProgress> {
        Ok(self.state()?.progress)
    }

    pub fn closed(&self) -> Result<bool> {
        Ok(self.state()?.closed)
    }

    pub fn retired(&self) -> Result<bool> {
        Ok(self.state()?.retired)
    }

    fn state(&self) -> Result<MutexGuard<'_, RequestState>> {
        self.state
            .lock()
            .map_err(|_| Error::State("request state lock is poisoned"))
    }
}

/// Own scheduler-assigned slots and advance their admitted request epochs.
pub struct RequestPool {
    // Slot zero belongs to inactive graph rows.
    slots: Vec<Option<u64>>,
    requests: HashMap<u64, Arc<Request>>,
    closed: bool,
}

impl RequestPool {
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::Invalid(
                "request-pool capacity must be positive".into(),
            ));
        }

        Ok(Self {
            slots: vec![None; capacity + 1],
            requests: HashMap::new(),
            closed: false,
        })
    }

    pub fn capacity(&self) -> usize {
        self.slots.len() - 1
    }

    pub fn get(&self, request_id: u64) -> Result<&Arc<Request>> {
        self.peek(request_id)
            .ok_or_else(|| Error::Invalid(format!("unknown request {request_id}")))
    }

    pub fn peek(&self, request_id: u64) -> Option<&Arc<Request>> {
        self.requests.get(&request_id)
    }

    pub fn request_ids(&self) -> Vec<u64> {
        let mut ids: Vec<_> = self.requests.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    pub fn has_open_requests(&self) -> Result<bool> {
        for request in self.requests.values() {
            let state = request.state()?;
            if !state.closed && !state.retired {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Install an admission at its assigned slot; identical retries retain
    /// progress. A different epoch may replace only a retired occupant.
    pub fn start(&mut self, admission: NewRequest) -> Result<Option<usize>> {
        let key = admission.request_key;
        let slot = self.validate_slot(admission.request_pool_idx as usize)?;

        if let Some(resident) = self.peek(key.request_id.0) {
            if resident.key() == key {
                if resident.admission() != &admission {
                    return Err(Error::Invalid(
                        "request admission conflicts with resident state".into(),
                    ));
                }
                return Ok(None);
            }
            if !resident.retired()? {
                return Err(Error::Invalid(
                    "request admission conflicts with resident state".into(),
                ));
            }
        }
        if let Some(id) = self.slots[slot]
            && !self.get(id)?.retired()?
        {
            return Err(Error::Invalid(format!(
                "request-pool index {slot} is occupied"
            )));
        }

        // Check both occupants before eviction: a failed admission must leave
        // the previous request available to its remaining result observers.
        self.remove(key.request_id.0);
        if let Some(id) = self.slots[slot] {
            self.remove(id);
        }
        self.requests
            .insert(key.request_id.0, Arc::new(Request::new(admission)));
        self.slots[slot] = Some(key.request_id.0);
        Ok(Some(slot))
    }

    /// Bind numerical rows to distinct, live request epochs and slots.
    pub fn bind_calls(&self, calls: &[(RequestKey, CallId, usize)]) -> Result<Vec<Arc<Request>>> {
        let mut keys = HashSet::new();
        let mut slots = HashSet::new();
        let mut requests = Vec::with_capacity(calls.len());

        for &(key, id, slot) in calls {
            let slot = self.validate_slot(slot)?;
            if !keys.insert(key) {
                return Err(Error::Invalid("a batch repeats a request".into()));
            }
            if !slots.insert(slot) {
                return Err(Error::Invalid(
                    "a batch repeats a request-pool index".into(),
                ));
            }

            let request = self.get(key.request_id.0)?;
            if request.key() != key {
                return Err(Error::Invalid(format!(
                    "call {id:?} has a stale request key"
                )));
            }
            if request.slot() != slot {
                return Err(Error::Invalid(format!(
                    "call {id:?} has a stale request slot"
                )));
            }
            let state = request.state()?;
            if state.closed {
                return Err(Error::Invalid(format!(
                    "call {id:?} targets a closed request"
                )));
            }
            if state.pending.contains_key(&id) {
                return Err(Error::Invalid("request call is already executing".into()));
            }
            requests.push(Arc::clone(request));
        }
        Ok(requests)
    }

    /// Check the complete submission before reserving any pending call.
    /// Each tuple names the request, call, and whether it advances request state.
    pub fn validate_pending(&self, calls: &[(RequestKey, CallId, bool)]) -> Result<()> {
        let mut seen = HashSet::new();
        for &(key, id, _) in calls {
            let request = self.get(key.request_id.0)?;
            if request.key() != key {
                return Err(Error::State("request commit lost its admitted slot"));
            }
            if request.state()?.pending.contains_key(&id) || !seen.insert((key, id)) {
                return Err(Error::State("request commit repeats an executing call"));
            }
        }
        Ok(())
    }

    /// Reserve all calls together in their submission order.
    pub fn add_pending(&mut self, calls: &[(RequestKey, CallId, bool)]) -> Result<()> {
        self.validate_pending(calls)?;
        for &(key, id, advances) in calls {
            self.get(key.request_id.0)?
                .state()?
                .pending
                .insert(id, advances);
        }
        Ok(())
    }

    /// Follow the last submitted state call, or the accepted admission root.
    /// Independent video encoders and transfers have no state predecessor.
    pub fn predecessor(&self, key: RequestKey, advances: bool) -> Result<Option<CallId>> {
        let Some(request) = self.peek(key.request_id.0) else {
            return Ok(None);
        };
        if request.key() != key || (request.admission.diffusion.is_some() && !advances) {
            return Ok(None);
        }

        let state = request.state()?;
        Ok(Some(
            state
                .pending
                .iter()
                .rev()
                .find_map(|(&id, &advances)| advances.then_some(id))
                .unwrap_or(state.state_call),
        ))
    }

    /// Submit the next canvas step of a retained request epoch.
    ///
    /// Queued steps can precede device completion, so these coordinates are
    /// distinct from accepted progress. A new epoch starts without a canvas;
    /// its device banks are initialized by step zero before their first read.
    pub fn advance_canvas(&self, key: RequestKey, step: CanvasStep) -> Result<()> {
        let request = self.get(key.request_id.0)?;
        let mut state = request.state()?;
        if request.key() != key || state.retired {
            return Err(Error::Invalid(
                "canvas step requires the retained request epoch".into(),
            ));
        }

        let follows = if step.step == 0 {
            state
                .canvas
                .is_none_or(|last| last.block.checked_add(1) == Some(step.block))
        } else {
            state.canvas
                == Some(CanvasStep {
                    block: step.block,
                    step: step.step - 1,
                })
        };
        if !follows {
            return Err(Error::Invalid(format!(
                "canvas step {step:?} does not continue {:?}",
                state.canvas
            )));
        }

        state.canvas = Some(step);
        Ok(())
    }

    /// Retire one call's pending state. Stale epochs and duplicate results
    /// cannot affect a reused slot; late results cannot regress progress.
    pub fn apply_result(
        &mut self,
        key: RequestKey,
        id: CallId,
        status: CallStatus,
        progress: Option<RequestProgress>,
    ) -> Result<()> {
        let Some(request) = self
            .peek(key.request_id.0)
            .filter(|request| request.key() == key)
        else {
            return Ok(());
        };
        let mut state = request.state()?;
        let Some(advances) = state.pending.shift_remove(&id) else {
            return Ok(());
        };

        if let Some(progress) = progress.filter(|_| id > state.accepted_call) {
            state.progress = progress;
            state.accepted_call = id;
        }
        if status == CallStatus::Ok && advances && id > state.state_call {
            state.state_call = id;
        }
        if status == CallStatus::Error {
            state.closed = true;
        }
        Ok(())
    }

    pub fn cancel_calls(&mut self, calls: &[(RequestKey, CallId)]) -> Result<()> {
        for &(key, id) in calls {
            if let Some(request) = self
                .peek(key.request_id.0)
                .filter(|request| request.key() == key)
            {
                let mut state = request.state()?;
                state.pending.shift_remove(&id);
                state.closed = true;
            }
        }
        Ok(())
    }

    pub fn finish(&mut self, key: RequestKey) -> Result<()> {
        if let Some(request) = self
            .peek(key.request_id.0)
            .filter(|request| request.key() == key)
        {
            request.state()?.closed = true;
        }
        Ok(())
    }

    pub fn retirement_ready(&self, key: RequestKey) -> Result<bool> {
        match self
            .peek(key.request_id.0)
            .filter(|request| request.key() == key)
        {
            Some(request) => Ok(request.state()?.pending.is_empty()),
            None => Ok(true),
        }
    }

    /// Mark a closed epoch reusable after its physical readers have drained.
    /// The execution owner must finish device and host writes before calling.
    pub fn retire(&mut self, request_id: u64) -> Result<()> {
        let mut state = self.get(request_id)?.state()?;
        if !state.closed || !state.pending.is_empty() {
            return Err(Error::State(
                "request retirement requires closed, completed execution",
            ));
        }
        state.retired = true;
        Ok(())
    }

    /// Remove the pool's reference after the execution owner releases a request.
    pub fn remove(&mut self, request_id: u64) {
        if let Some(request) = self.requests.remove(&request_id) {
            self.slots[request.slot()] = None;
        }
    }

    /// Release request references after the execution owner has drained work.
    pub fn close(&mut self) {
        self.closed = true;
        self.requests.clear();
        self.slots.fill(None);
    }

    fn validate_slot(&self, slot: usize) -> Result<usize> {
        if self.closed {
            return Err(Error::State("request pool is closed"));
        }
        if slot == 0 || slot > self.capacity() {
            return Err(Error::Invalid(format!(
                "request-pool index {slot} exceeds capacity"
            )));
        }
        Ok(slot)
    }
}
