//! Numerical callbacks and tensor lifetime for the shared worker executor.

use std::collections::HashSet;
use std::sync::{Arc, Mutex, MutexGuard};

use tvm_ffi::{Error, Function, Result, Tensor, RUNTIME_ERROR};
use uniserve_worker::cuda::Event;
use uniserve_worker::{Backend, Batch as ScheduledBatch, Executor as NativeExecutor, Submission};

pub fn failure(message: impl AsRef<str>) -> Error {
    Error::new(RUNTIME_ERROR, message.as_ref(), "")
}

pub fn lock<T>(value: &Mutex<T>) -> Result<MutexGuard<'_, T>> {
    value
        .lock()
        .map_err(|_| failure("execution state is poisoned"))
}

/// Tensor references and the fence covering one numerical invocation.
pub struct Batch {
    input: Option<Tensor>,
    output: Option<Result<Tensor>>,
    event: Result<Option<Arc<Event>>>,
    cancelled: bool,
}

impl Batch {
    fn new(input: Tensor) -> Self {
        Self {
            input: Some(input),
            output: None,
            event: Ok(None),
            cancelled: false,
        }
    }

    fn run(&mut self, forward: &Function) {
        let input = self.input.as_ref().expect("unexecuted batch input");
        let device = input.device();
        let stream = unsafe {
            tvm_ffi::tvm_ffi_sys::TVMFFIEnvGetStream(device.device_type as i32, device.device_id)
        };
        self.output = Some(forward.call_tuple((input,)).and_then(Tensor::try_from));

        // A callback may fail after launching kernels. Both outcomes retain
        // their inputs until the submitting stream has finished those accesses.
        self.event = if device.device_type as i32 == 2 {
            let event = Event::new(device.device_id, false, false);
            event
                .record(stream as usize)
                .map(|()| Some(Arc::new(event)))
                .map_err(failure)
        } else {
            Ok(None)
        };
    }

    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    pub fn retired(&self) -> Result<bool> {
        match &self.event {
            Ok(Some(event)) => event.ready().map_err(failure),
            Ok(None) => Ok(true),
            Err(error) => Err(error.clone()),
        }
    }

    pub fn wait(batch: &Mutex<Self>) -> Result<()> {
        // Waiting must not hold the batch lock: another caller may query or
        // cancel it while the executor delivers an independent result.
        let event = lock(batch)?.event.clone();
        wait(event)
    }

    fn result(&mut self) -> Result<Tensor> {
        // The executor delivers results only after poll observed completion.
        // Release the fence here so normal delivery needs no host wait.
        self.event = Ok(None);
        if self.cancelled {
            return Err(failure("batch was cancelled"));
        }

        self.output.take().expect("completed numerical batch")
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        if wait(self.event.clone()).is_err() {
            // Unknown physical completion cannot return DLPack storage for reuse.
            std::mem::forget(self.input.take());
            std::mem::forget(self.output.take());
        }
    }
}

struct NumericalBackend {
    forward: Function,
}

impl Backend for NumericalBackend {
    type Batch = Arc<Mutex<Batch>>;
    type Output = Tensor;
    type Error = Error;

    fn error(&self, error: uniserve_worker::Error) -> Error {
        failure(error.to_string())
    }

    fn classify(&self, error: Error, _batch: &Self::Batch, _context: &str) -> Error {
        error
    }

    fn note_cleanup(&self, error: &mut Error, cleanup: Error) {
        let message = format!(
            "{}\nBatch cleanup also failed: {}",
            error.message(),
            cleanup.message()
        );
        *error = Error::new(error.kind(), &message, error.backtrace());
    }

    // This backend borrows an already prepared tensor. It has no request
    // commands, external input transfers or deferred storage retirement.
    fn admit(&mut self, _batch: &mut Self::Batch) -> Result<()> {
        Ok(())
    }

    fn prepare(&mut self, _batch: &mut Self::Batch) -> Result<()> {
        Ok(())
    }

    fn prepare_inputs(
        &mut self,
        _batch: &mut Self::Batch,
        _submission: &Arc<Submission>,
    ) -> Result<bool> {
        Ok(true)
    }

    fn await_inputs(
        &mut self,
        _batch: &mut Self::Batch,
        _submission: &Arc<Submission>,
    ) -> Result<()> {
        unreachable!("numerical inputs are supplied at submission")
    }

    fn execute(&mut self, batch: &mut Self::Batch) -> Result<()> {
        lock(batch)?.run(&self.forward);
        Ok(())
    }

    fn begin_retirement(&mut self, _batch: &mut Self::Batch) -> Result<()> {
        Ok(())
    }

    fn poll(&mut self, batch: &mut Self::Batch) -> Result<(bool, bool)> {
        Ok((false, lock(batch)?.retired()?))
    }

    fn result(&mut self, batch: &mut Self::Batch) -> Result<Tensor> {
        lock(batch)?.result()
    }

    fn close(&mut self, batch: &mut Self::Batch) -> Result<()> {
        Batch::wait(batch)?;
        let tensors = {
            let mut batch = lock(batch)?;
            batch.event = Ok(None);
            (batch.input.take(), batch.output.take())
        };

        // DLPack owners may have foreign destructors; release them unlocked.
        drop(tensors);
        Ok(())
    }

    fn reap(&mut self) -> Result<()> {
        Ok(())
    }
}

pub struct Executor {
    executor: Mutex<Option<NativeExecutor<NumericalBackend>>>,
}

impl Executor {
    pub fn new(forward: Function, capacity: i64) -> Result<Self> {
        let capacity = usize::try_from(capacity).map_err(|error| failure(error.to_string()))?;
        let backend = NumericalBackend { forward };
        let executor = NativeExecutor::new(backend, capacity, false, false)?;

        Ok(Self {
            executor: Mutex::new(Some(executor)),
        })
    }

    /// The submitting thread owns numerical execution. Recursive callbacks and
    /// concurrent callers must not borrow its workspace during another call.
    fn borrow(&self) -> Result<MutexGuard<'_, Option<NativeExecutor<NumericalBackend>>>> {
        self.executor.try_lock().map_err(|error| match error {
            std::sync::TryLockError::WouldBlock => failure("executor is busy"),
            std::sync::TryLockError::Poisoned(_) => failure("executor state is poisoned"),
        })
    }

    pub fn submit(&self, id: i64, input: Tensor) -> Result<(Arc<Submission>, Arc<Mutex<Batch>>)> {
        let id = u64::try_from(id).map_err(|error| failure(error.to_string()))?;
        let mut owner = self.borrow()?;
        let executor = owner
            .as_mut()
            .ok_or_else(|| failure("executor is closed"))?;
        if !matches!(input.device().device_type as i32, 1 | 2) {
            return Err(failure("numerical execution requires CPU or CUDA tensors"));
        }

        // Foreign references can be released by different Python threads. Keep
        // atomic ownership without adding Send/Sync to the SDK's tensor handle.
        #[allow(clippy::arc_with_non_send_sync)]
        let batch = Arc::new(Mutex::new(Batch::new(input)));
        let scheduled =
            ScheduledBatch::new(id, None, HashSet::new(), HashSet::new(), batch.clone());
        let submission = executor.submit(scheduled, false)?;
        Ok((submission, batch))
    }

    pub fn poll(&self, submission: &Arc<Submission>) -> Result<Option<Tensor>> {
        let mut owner = self.borrow()?;
        let executor = owner
            .as_mut()
            .ok_or_else(|| failure("executor is closed"))?;
        executor.advance()?;
        executor.poll(submission)
    }

    pub fn close(&self) -> Result<()> {
        let executor = self.borrow()?.take();
        if let Some(executor) = executor {
            drain(executor)?;
        }
        Ok(())
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        let owner = self
            .executor
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(executor) = owner.take() {
            let _ = drain(executor);
        }
    }
}

fn wait(event: Result<Option<Arc<Event>>>) -> Result<()> {
    if let Some(event) = event? {
        if !event.ready().map_err(failure)? {
            event.wait().map_err(failure)?;
        }
    }
    Ok(())
}

fn drain(mut executor: NativeExecutor<NumericalBackend>) -> Result<()> {
    let result = executor.close();
    if result.is_err() {
        // The callable may retain model weights still accessed by the GPU.
        std::mem::forget(executor);
    }
    result
}
