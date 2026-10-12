//! Where each live rank incarnation sits in the deployment.

use std::collections::HashMap;
use std::sync::{Arc, PoisonError, RwLock};

use uniserve_worker_ipc::WorkerEndpoint;

use crate::WorkerId;

/// The worker and rank each live rank incarnation was launched as.
///
/// A rank reports only its own incarnation: the engine chose which worker and
/// rank that process serves when it launched it, so the engine is the only
/// owner of that fact. Groups register their ranks when the deployment is
/// bound and again when a replacement group is installed, which retires the
/// incarnations it replaces. Readers resolve the incarnation a publication
/// names, and a publication from a retired incarnation resolves to nothing.
///
/// Clones share one table.
#[derive(Debug, Clone, Default)]
pub(crate) struct IncarnationRegistry {
    positions: Arc<RwLock<HashMap<String, (WorkerId, u32)>>>,
}

impl IncarnationRegistry {
    /// Makes `endpoints`, in rank order, the live incarnations of `worker`.
    ///
    /// Every incarnation `worker` held before is retired.
    pub(crate) fn assign<'a>(
        &self,
        worker: &WorkerId,
        endpoints: impl IntoIterator<Item = &'a WorkerEndpoint>,
    ) {
        let mut positions = self
            .positions
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        positions.retain(|_, (owner, _)| owner != worker);
        for (rank, endpoint) in endpoints.into_iter().enumerate() {
            positions.insert(endpoint.incarnation.clone(), (worker.clone(), rank as u32));
        }
    }

    /// Returns the worker and rank `endpoint`'s incarnation serves, or `None`
    /// when it is not a live incarnation of this deployment.
    pub(crate) fn position(&self, endpoint: &WorkerEndpoint) -> Option<(WorkerId, u32)> {
        self.positions
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&endpoint.incarnation)
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint(incarnation: &str) -> WorkerEndpoint {
        WorkerEndpoint {
            node: "host".to_owned(),
            address_space: format!("{incarnation}-process"),
            incarnation: incarnation.to_owned(),
        }
    }

    #[test]
    fn replacement_retires_the_replaced_incarnations_of_that_worker_only() {
        let registry = IncarnationRegistry::default();
        let text = WorkerId("text".to_owned());
        let decoder = WorkerId("decoder".to_owned());
        let before = [endpoint("text-0"), endpoint("text-1")];
        let other = endpoint("decoder-0");
        registry.assign(&text, &before);
        registry.assign(&decoder, [&other]);
        assert_eq!(registry.position(&before[1]), Some((text.clone(), 1)));

        let after = [endpoint("text-0b"), endpoint("text-1b")];
        registry.assign(&text, &after);

        assert_eq!(registry.position(&before[0]), None);
        assert_eq!(registry.position(&before[1]), None);
        assert_eq!(registry.position(&after[1]), Some((text, 1)));
        assert_eq!(registry.position(&other), Some((decoder, 0)));
    }
}
