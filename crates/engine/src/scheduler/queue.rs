//! Waiting requests ordered by the configured closed scheduling policy.

use std::collections::VecDeque;

use uniserve_core::RequestId;

use crate::scheduler::{ReqState, SchedulingPolicy};

pub(crate) enum RequestQueue {
    Fcfs(VecDeque<ReqState>),
    Priority(VecDeque<ReqState>),
}

impl RequestQueue {
    pub(crate) fn new(policy: SchedulingPolicy) -> Self {
        match policy {
            SchedulingPolicy::Fcfs => Self::Fcfs(VecDeque::new()),
            SchedulingPolicy::Priority => Self::Priority(VecDeque::new()),
        }
    }

    fn queue(&self) -> &VecDeque<ReqState> {
        match self {
            Self::Fcfs(queue) | Self::Priority(queue) => queue,
        }
    }

    fn queue_mut(&mut self) -> &mut VecDeque<ReqState> {
        match self {
            Self::Fcfs(queue) | Self::Priority(queue) => queue,
        }
    }

    pub(crate) fn add_request(&mut self, state: ReqState) {
        match self {
            Self::Fcfs(queue) => queue.push_back(state),
            Self::Priority(queue) => {
                let key = (state.req.priority, state.queued_at);
                let index =
                    queue.partition_point(|item| (item.req.priority, item.queued_at) <= key);
                queue.insert(index, state);
            }
        }
    }

    pub(crate) fn peek_request(&self) -> Option<&ReqState> {
        self.queue().front()
    }

    pub(crate) fn pop_request(&mut self) -> Option<ReqState> {
        self.queue_mut().pop_front()
    }

    pub(crate) fn remove_request(&mut self, id: RequestId) -> Option<ReqState> {
        let position = self
            .queue()
            .iter()
            .position(|state| state.req.request_id == id)?;
        self.queue_mut().remove(position)
    }

    pub(crate) fn len(&self) -> usize {
        self.queue().len()
    }
}
