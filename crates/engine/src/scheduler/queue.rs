//! Waiting requests ordered by the configured closed scheduling policy.

use std::collections::VecDeque;

use uniserve_core::RequestId;

use crate::scheduler::{QueueItem, SchedulingPolicy};

pub(crate) enum RequestQueue<T> {
    Fcfs(VecDeque<T>),
    Priority(VecDeque<T>),
}

impl<T: QueueItem> RequestQueue<T> {
    pub(crate) fn new(policy: SchedulingPolicy) -> Self {
        match policy {
            SchedulingPolicy::Fcfs => Self::Fcfs(VecDeque::new()),
            SchedulingPolicy::Priority => Self::Priority(VecDeque::new()),
        }
    }

    fn queue(&self) -> &VecDeque<T> {
        match self {
            Self::Fcfs(queue) | Self::Priority(queue) => queue,
        }
    }

    fn queue_mut(&mut self) -> &mut VecDeque<T> {
        match self {
            Self::Fcfs(queue) | Self::Priority(queue) => queue,
        }
    }

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

    pub(crate) fn peek_request(&self) -> Option<&T> {
        self.queue().front()
    }

    pub(crate) fn pop_request(&mut self) -> Option<T> {
        self.queue_mut().pop_front()
    }

    pub(crate) fn remove_request(&mut self, id: RequestId) -> Option<T> {
        let position = self
            .queue()
            .iter()
            .position(|state| state.request_id() == id)?;
        self.queue_mut().remove(position)
    }

    pub(crate) fn len(&self) -> usize {
        self.queue().len()
    }
}
