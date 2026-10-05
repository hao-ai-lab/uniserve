//! KV copy actions on the shared native host executor.

use std::ops::Deref;
use std::sync::Arc;

use crate::{Completion, Error, HostAction, HostTask, Outcome};

use super::{KVImport, KVImporter};

/// Numerical copies and completion observers supplied by a language binding.
/// Workspace admission, cancellation and physical retirement remain native.
pub trait ImportBackend: Send + Sync + 'static {
    type Error: Send + Sync;
    type Callback: Send + Sync;
    type Read: Deref<Target = Completion<Self::Error, Self::Callback>> + Send + Sync;
    type Workspace: Send + Sync;
    type Import: Deref<Target = KVImport<Self::Read, Self::Workspace>> + Send + Sync;

    /// Reset newly allocated destination units on the workspace's stream.
    fn reset(&self, workspace: &Self::Workspace) -> Result<(), Self::Error>;

    /// Copy KV values and retain any physical transport reads on the import.
    fn copy(&self, workspace: &Self::Workspace) -> Result<(), Self::Error>;

    /// Wait for numerical accesses. An error leaves their completion unknown.
    fn drain(&self, workspace: &Self::Workspace) -> Result<(), Self::Error>;

    /// Notify observers of physical retirement after the importer unlocks.
    fn retired(&self, imports: Vec<Self::Import>) -> Result<(), Self::Error>;
    fn error(error: Error) -> Self::Error;
    fn report(error: Self::Error);
    fn note_cleanup(error: &mut Self::Error, cleanup: Self::Error);
}

/// One KV import's copy task. The importer retains its workspace and reads
/// independently of the task result until their physical accesses end.
pub struct ImportCopy<B: ImportBackend> {
    importer: Arc<KVImporter<B::Import, B::Workspace>>,
    write: Arc<KVImport<B::Read, B::Workspace>>,
    backend: B,
}

impl<B: ImportBackend> ImportCopy<B> {
    pub fn new(
        importer: Arc<KVImporter<B::Import, B::Workspace>>,
        write: Arc<KVImport<B::Read, B::Workspace>>,
        backend: B,
    ) -> Self {
        Self {
            importer,
            write,
            backend,
        }
    }
}

impl<B: ImportBackend> HostAction for ImportCopy<B> {
    type Output = ();
    type Error = B::Error;
    type Callback = Box<dyn FnOnce() + Send>;
    type Wake = Box<dyn Fn() + Send + Sync>;

    fn ready(&self) -> Result<bool, Self::Error> {
        Ok(true)
    }

    fn run(&self) -> Result<(), Self::Error> {
        let _range = crate::profiling::range(c"uniserve.kv_import", None);
        self.write.start();
        let workspace = match self.importer.acquire(&self.write) {
            Ok(workspace) => workspace,
            Err(error) => {
                self.write.finish(true);
                return Err(B::error(error));
            }
        };

        let copied = (|| {
            self.importer
                .require_active(&self.write)
                .map_err(B::error)?;
            self.backend.reset(&workspace)?;

            // Transport reads use other streams. Reset the destination before
            // those streams can start writing it.
            self.backend.drain(&workspace)?;
            self.backend.copy(&workspace)
        })();

        let drained = self.backend.drain(&workspace);
        self.write.finish(drained.is_ok());
        if let Err(mut error) = drained {
            if let Err(cause) = copied {
                B::note_cleanup(&mut error, cause);
            }
            return Err(error);
        }

        copied
    }

    fn input_outcome(&self) -> Option<Outcome<Self::Error>> {
        // Queued cancellation has no numerical access to drain. Failed DMA
        // retains this action and its backing through the host lane's common
        // unknown-completion handling.
        self.write.task_done();
        Some(if self.write.drained() {
            Outcome::Success(())
        } else {
            Outcome::Cancelled
        })
    }

    fn release(&self) -> Result<(), Self::Error> {
        self.backend.retired(self.importer.reap())
    }

    fn defer_release(_: Arc<HostTask<Self>>) -> Result<(), Self::Error> {
        Err(B::error(Error::Invariant(
            "KV copy input completion is known after its action".into(),
        )))
    }

    fn notify(callbacks: Vec<Self::Callback>) {
        for callback in callbacks {
            callback();
        }
    }

    fn wake(wake: &Self::Wake) {
        wake();
    }

    fn error(error: Error) -> Self::Error {
        B::error(error)
    }

    fn report(error: Self::Error) {
        B::report(error);
    }

    fn note_cleanup(error: &mut Self::Error, cleanup: Self::Error) {
        B::note_cleanup(error, cleanup);
    }
}
