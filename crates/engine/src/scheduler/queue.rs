//! Waiting-request queues ordered by the configured scheduling policy.

use std::collections::VecDeque;

use uniserve_core::RequestId;

use crate::scheduler::{QueueItem, SchedulingPolicy};

/// Queue representation selected by the configured ordering policy.
pub(crate) enum RequestQueue<T> {
    Fcfs(VecDeque<T>),
    Priority(VecDeque<T>),
}

impl<T: QueueItem> RequestQueue<T> {
    /// Creates a scheduling queue using the requested policy.
    pub(crate) fn new(policy: SchedulingPolicy) -> Self {
        match policy {
            SchedulingPolicy::Fcfs => Self::Fcfs(VecDeque::new()),
            SchedulingPolicy::Priority => Self::Priority(VecDeque::new()),
        }
    }

    /// Returns shared access to the scheduling queue.
    fn queue(&self) -> &VecDeque<T> {
        match self {
            Self::Fcfs(queue) | Self::Priority(queue) => queue,
        }
    }

    /// Returns mutable access to the scheduling queue.
    fn queue_mut(&mut self) -> &mut VecDeque<T> {
        match self {
            Self::Fcfs(queue) | Self::Priority(queue) => queue,
        }
    }

    /// Adds the request.
    pub(crate) fn add_request(&mut self, state: T) {
        match self {
            Self::Fcfs(queue) => queue.push_back(state),
            Self::Priority(queue) => {
                let key = (state.priority(), state.queued_at());
                let index =
                    queue.partition_point(|item| (item.priority(), item.queued_at()) <= key);
                queue.insert(index, state);
            }
        }
    }

    /// Returns the next queued request without removing it.
    pub(crate) fn peek_request(&self) -> Option<&T> {
        self.queue().front()
    }

    /// Removes and returns the next queued request.
    pub(crate) fn pop_request(&mut self) -> Option<T> {
        self.queue_mut().pop_front()
    }

    /// Removes and returns a request by identifier.
    pub(crate) fn remove_request(&mut self, id: RequestId) -> Option<T> {
        let position = self
            .queue()
            .iter()
            .position(|state| state.request_id() == id)?;
        self.queue_mut().remove(position)
    }

    /// Returns the number of entries.
    pub(crate) fn len(&self) -> usize {
        self.queue().len()
    }
}
