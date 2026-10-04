//! Batch ownership and completion, with no Python object or Future access.

use std::sync::{Arc, Mutex, MutexGuard};

use tvm_ffi::{Error, Function, Result, Tensor, RUNTIME_ERROR};

use crate::cuda::Event;

pub fn failure(message: impl AsRef<str>) -> Error {
    Error::new(RUNTIME_ERROR, message.as_ref(), "")
}

pub fn lock<T>(value: &Mutex<T>) -> Result<MutexGuard<'_, T>> {
    value
        .lock()
        .map_err(|_| failure("execution state is poisoned"))
}

pub struct Execution {
    input: Option<Tensor>,
    output: Result<Tensor>,
    event: Result<Option<Event>>,
    cancelled: bool,
}

impl Execution {
    pub fn run(forward: &Function, input: Tensor) -> Self {
        let device = input.device();
        let stream = unsafe {
            tvm_ffi::tvm_ffi_sys::TVMFFIEnvGetStream(device.device_type as i32, device.device_id)
        };
        let output = forward.call_tuple((&input,)).and_then(Tensor::try_from);

        // A callback may raise after launching kernels. Record the fence on
        // both paths and retain the input until those kernels have drained.
        let event = if device.device_type as i32 == 2 {
            Event::record(stream).map(Some).map_err(failure)
        } else {
            Ok(None)
        };
        Self {
            input: Some(input),
            output,
            event,
            cancelled: false,
        }
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

    pub fn wait(&self) -> Result<()> {
        match &self.event {
            Ok(Some(event)) => event.wait().map_err(failure),
            Ok(None) => Ok(()),
            Err(error) => Err(error.clone()),
        }
    }

    pub fn result(&self) -> Result<Tensor> {
        if self.cancelled {
            return Err(failure("batch was cancelled"));
        }
        let output = self.output.as_ref().map_err(Clone::clone)?;
        if let Some(event) = self.event.as_ref().map_err(Clone::clone)? {
            let device = output.device();
            let stream = unsafe {
                tvm_ffi::tvm_ffi_sys::TVMFFIEnvGetStream(
                    device.device_type as i32,
                    device.device_id,
                )
            };
            if device.device_type as i32 == 1 {
                event.wait().map_err(failure)?;
            } else {
                event.wait_on(stream).map_err(failure)?;
            }
        }
        Ok(output.clone())
    }
}

impl Drop for Execution {
    fn drop(&mut self) {
        // The executor normally reaps completed work. Dropping its last owner
        // early drains only this batch, without calling Python while waiting.
        // Unknown physical completion must not return DLPack storage for reuse.
        if self.wait().is_err() {
            std::mem::forget(self.input.take());
            let output = std::mem::replace(&mut self.output, Err(failure("completion unknown")));
            std::mem::forget(output);
        }
    }
}

pub struct Executor {
    capacity: usize,
    state: Mutex<ExecutorState>,
    submitting: Mutex<()>,
}

struct ExecutorState {
    forward: Option<Function>,
    active: Vec<Arc<Mutex<Execution>>>,
}

impl Executor {
    pub fn new(forward: Function, capacity: i64) -> Result<Self> {
        if capacity < 1 {
            return Err(failure("execution capacity must be positive"));
        }
        Ok(Self {
            capacity: capacity as usize,
            state: Mutex::new(ExecutorState {
                forward: Some(forward),
                active: Vec::new(),
            }),
            submitting: Mutex::new(()),
        })
    }

    pub fn submit(&self, input: Tensor) -> Result<Arc<Mutex<Execution>>> {
        // A numerical callback may invoke the public API again. It cannot
        // recursively use the same runner's workspace during a forward.
        let _submitting = self
            .submitting
            .try_lock()
            .map_err(|_| failure("executor is already submitting"))?;
        self.reap()?;
        let forward = {
            let state = lock(&self.state)?;
            if state.active.len() >= self.capacity {
                return Err(failure("execution capacity is occupied"));
            }
            state
                .forward
                .clone()
                .ok_or_else(|| failure("executor is closed"))?
        };
        if !matches!(input.device().device_type as i32, 1 | 2) {
            return Err(failure("numerical execution requires CPU or CUDA tensors"));
        }

        // FFI objects can be released by different Python threads, so their
        // shared owner needs atomic refcounts. The upstream tensor handle is
        // not Send; keep that restriction for Rust tasks instead of asserting
        // thread safety for arbitrary foreign tensor allocators.
        #[allow(clippy::arc_with_non_send_sync)]
        let batch = Arc::new(Mutex::new(Execution::run(&forward, input)));
        lock(&self.state)?.active.push(batch.clone());
        Ok(batch)
    }

    pub fn reap(&self) -> Result<()> {
        let mut state = lock(&self.state)?;
        let active = &mut state.active;
        let mut index = 0;
        while index < active.len() {
            if lock(&active[index])?.retired()? {
                active.swap_remove(index);
            } else {
                index += 1;
            }
        }
        Ok(())
    }

    pub fn close(&self) -> Result<()> {
        let _submitting = self
            .submitting
            .try_lock()
            .map_err(|_| failure("executor is already submitting"))?;
        let (active, forward) = {
            let mut state = lock(&self.state)?;
            for batch in &state.active {
                lock(batch)?.wait()?;
            }
            (std::mem::take(&mut state.active), state.forward.take())
        };
        drop(active);
        drop(forward);
        Ok(())
    }
}
