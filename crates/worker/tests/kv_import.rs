//! Physical KV copy failure must not return memory to its allocator.

use std::sync::Arc;

use uniserve_core::{CallId, RequestId};
use uniserve_worker::{
    Completion, Error, HostLane, ImportBackend, ImportCopy, KVImport, KVImporter, Outcome,
};
use uniserve_worker_ipc::{BufferId, RequestKey};

type Read = Arc<Completion<String, ()>>;
type Import = KVImport<Read, ()>;

/// External numerical backend whose reset submitted work that cannot be drained.
struct FailedCopy {
    _storage: Arc<Vec<u8>>,
}

impl ImportBackend for FailedCopy {
    type Error = String;
    type Callback = ();
    type Read = Read;
    type Workspace = ();
    type Import = Arc<Import>;

    fn reset(&self, _: &()) -> Result<(), String> {
        Ok(())
    }

    fn copy(&self, _: &()) -> Result<(), String> {
        Ok(())
    }

    fn drain(&self, _: &()) -> Result<(), String> {
        Err("device access could not be drained".into())
    }

    fn retired(&self, _: Vec<Arc<Import>>) -> Result<(), String> {
        Ok(())
    }

    fn error(error: Error) -> String {
        error.to_string()
    }

    fn report(error: String) {
        panic!("unexpected cleanup failure: {error}");
    }

    fn note_cleanup(error: &mut String, cleanup: String) {
        error.push_str(&format!("; {cleanup}"));
    }
}

#[test]
fn unknown_completion_keeps_backing_after_importer_and_task_drop()
-> Result<(), Box<dyn std::error::Error>> {
    let lane = HostLane::new(1, 1, "kv-copy-failure")?;
    let importer = Arc::new(KVImporter::new(vec![Arc::new(())]));
    let write = Arc::new(Import::new(
        BufferId {
            owner: RequestKey::new(1, RequestId(1), 1),
            producer_call_id: CallId::new(1, 0),
            output_index: 0,
            generation: 1,
        },
        1,
        true,
    ));
    importer.reserve(Arc::clone(&write), || Ok(()))?;
    let storage = Arc::new(vec![1, 2, 3, 4]);
    let retained = Arc::downgrade(&storage);
    let task = lane.reserve()?;
    task.configure(ImportCopy::new(
        Arc::clone(&importer),
        Arc::clone(&write),
        FailedCopy { _storage: storage },
    ))?;
    task.submit()?;
    assert!(matches!(
        task.completion.wait(None),
        Some(Outcome::Failed(_))
    ));

    importer.abandon(&write);
    importer.reap();
    assert!(importer.require_retired().is_err());
    lane.close();
    drop(task);
    drop(write);
    drop(importer);
    drop(lane);
    assert!(retained.upgrade().is_some());
    Ok(())
}
