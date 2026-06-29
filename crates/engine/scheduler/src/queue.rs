//! The waiting-request queue abstraction, mirroring vLLM's `request_queue.py`:
//! admission code consumes the queue; ordering policy lives in the queue
//! implementation.

use std::collections::VecDeque;

use uniserve_core::RequestId;

use crate::scheduler::ReqState;

/// Ordering policy for waiting requests.
pub trait RequestQueue: Send {
 /// Append a newly arrived request.
    fn add_request(&mut self, st: ReqState);
 /// Re-queue a preempted request so it restarts promptly: at the front
 /// under FCFS, in priority order (original arrival time) under Priority.
    fn prepend_request(&mut self, st: ReqState);
 /// The next request admission would consider.
    fn peek_request(&self) -> Option<&ReqState>;
 /// Remove and return the next request.
    fn pop_request(&mut self) -> Option<ReqState>;
 /// Remove a specific waiting request (cancellation before admission).
    fn remove_request(&mut self, id: RequestId) -> Option<ReqState>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// First-come-first-served: a plain deque.
#[derive(Default)]
pub struct FcfsRequestQueue {
    queue: VecDeque<ReqState>,
}

impl RequestQueue for FcfsRequestQueue {
    fn add_request(&mut self, st: ReqState) {
        self.queue.push_back(st);
    }

    fn prepend_request(&mut self, st: ReqState) {
        self.queue.push_front(st);
    }

    fn peek_request(&self) -> Option<&ReqState> {
        self.queue.front()
    }

    fn pop_request(&mut self) -> Option<ReqState> {
        self.queue.pop_front()
    }

    fn remove_request(&mut self, id: RequestId) -> Option<ReqState> {
        let pos = self.queue.iter().position(|s| s.req.request_id == id)?;
        self.queue.remove(pos)
    }

    fn len(&self) -> usize {
        self.queue.len()
    }
}

/// Priority ordering: lower `priority` value first, ties broken by arrival
/// time (vLLM's `(priority, arrival_time)` heap). Kept as an ordered vector
/// rather than a binary heap because removal-by-id must be supported anyway;
/// the sorted invariant lets the insertion point be located by binary search.
#[derive(Default)]
pub struct PriorityRequestQueue {
 /// Sorted ascending by `(priority, queued_at)`.
    queue: VecDeque<ReqState>,
}

impl PriorityRequestQueue {
 /// Index at which `st` should be inserted to preserve the ascending
 /// `(priority, queued_at)` order. `O(log n)` comparisons via binary search
 /// over the sorted queue (binary search; insertion is still `O(n)` shifts).
    fn insertion_index(&self, st: &ReqState) -> usize {
        let key = (st.req.priority, st.queued_at);
        self.queue
            .partition_point(|s| (s.req.priority, s.queued_at) <= key)
    }
}

impl RequestQueue for PriorityRequestQueue {
    fn add_request(&mut self, st: ReqState) {
        let idx = self.insertion_index(&st);
        self.queue.insert(idx, st);
    }

    fn prepend_request(&mut self, st: ReqState) {
 // A preempted request keeps its original `queued_at`, so ordered
 // insertion already places it ahead of same-priority later arrivals.
        self.add_request(st);
    }

    fn peek_request(&self) -> Option<&ReqState> {
        self.queue.front()
    }

    fn pop_request(&mut self) -> Option<ReqState> {
        self.queue.pop_front()
    }

    fn remove_request(&mut self, id: RequestId) -> Option<ReqState> {
        let pos = self.queue.iter().position(|s| s.req.request_id == id)?;
        self.queue.remove(pos)
    }

    fn len(&self) -> usize {
        self.queue.len()
    }
}
