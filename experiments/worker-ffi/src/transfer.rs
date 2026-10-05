//! Native read tasks with DLPack results and numerical FFI callbacks.

use std::sync::{Arc, Mutex, Weak};

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::{Function, Object, ObjectArc, Result, Tensor};
use uniserve_worker::cuda::DeviceGuard;
use uniserve_worker::{
    EventPool, HostAction, HostTask, Outcome, ReadBackend, TransferCapacity,
    TransferPool as NativePool, TransferRead, TransferTicket as NativeTicket,
};

use crate::execution::{failure, lock};
use crate::host::{Action, CallbackError};
use crate::{method, object};

type Ticket = NativeTicket<(), CallbackError, Function>;
type Task = HostTask<TransferRead<Read>>;

struct ReadState {
    ticket: Mutex<Ticket>,
    events: Arc<Mutex<EventPool<()>>>,
}

impl Drop for ReadState {
    fn drop(&mut self) {
        let ticket = self
            .ticket
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());
        let events = ticket.take_events();
        let mut pool = self
            .events
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Err(error) = pool.defer_release(events, ()).and_then(|_| pool.reap().1) {
            Read::report(error.into());
        }
    }
}

struct Read {
    state: Arc<ReadState>,
    pool: Weak<NativePool<Self>>,
    callback: Function,
    device: Option<i32>,
}

impl ReadBackend for Read {
    type Value = ();
    type Error = CallbackError;
    type Callback = Function;

    fn with_ticket<T>(&self, action: impl FnOnce(&mut Ticket) -> T) -> T {
        let mut ticket = self
            .state
            .ticket
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        action(&mut ticket)
    }

    fn read(&self) -> std::result::Result<(), CallbackError> {
        let Some(device) = self.device else {
            self.callback.call_tuple((0_usize,))?;
            Self::notify(self.with_ticket(|ticket| ticket.complete((), None)));
            return Ok(());
        };

        // The task's completion callback retains the native pool through read.
        let pool = self.pool.upgrade().expect("active read pool");
        let _device = DeviceGuard::new(device).map_err(failure)?;
        let stream = pool.stream(device)?;
        self.with_ticket(|ticket| ticket.wait_for_copy(stream.handle()))?;

        let submitted = (|| -> std::result::Result<(), CallbackError> {
            self.callback.call_tuple((stream.handle(),))?;
            let event = {
                let mut events = lock(&self.state.events)?;
                let event = events.acquire(device, false, false)?;
                events.record(&event, device, stream.handle())?;
                events.retain(&event, device, 1)?;
                event
            };
            Self::notify(self.with_ticket(|ticket| ticket.complete((), Some(event))));
            Ok(())
        })();

        // Even a failing numerical callback may have enqueued a copy. Publish
        // readiness before draining, but retain its callback and credits until
        // all physical accesses end. Consumer streams join the readiness event.
        let drained = stream.wait().map_err(failure).map_err(CallbackError::from);
        if drained.is_err() {
            self.with_ticket(|ticket| ticket.mark_undrained(Some(stream), None));
        }
        submitted.and(drained)
    }

    fn notify(callbacks: Vec<Function>) {
        for callback in callbacks {
            Self::wake(&callback);
        }
    }

    fn wake(callback: &Function) {
        if let Err(error) = callback.call_tuple(()) {
            Self::report(error.into());
        }
    }

    fn error(error: uniserve_worker::Error) -> CallbackError {
        error.into()
    }

    fn report(error: CallbackError) {
        Action::report(error);
    }
}

#[repr(C)]
#[derive(Object)]
#[type_key = "uniserve.ffi.TransferPool"]
pub struct TransferPoolObj {
    object: Object,
    pool: Arc<NativePool<Read>>,
    capacity: Arc<TransferCapacity<Function>>,
    events: Arc<Mutex<EventPool<()>>>,
}

#[derive(Clone, ObjectRef)]
pub struct TransferPool {
    data: ObjectArc<TransferPoolObj>,
}

impl TransferPool {
    fn new(bytes: u64, reads: usize, workers: usize) -> Result<Self> {
        let capacity = Arc::new(
            TransferCapacity::new(bytes, reads, Read::notify)
                .map_err(|error| failure(error.to_string()))?,
        );
        let pool = NativePool::new(Arc::clone(&capacity), workers, "ffi-transfer")
            .map_err(|error| failure(error.to_string()))?;
        Ok(Self {
            data: ObjectArc::new(TransferPoolObj {
                object: Object::new(),
                pool: Arc::new(pool),
                capacity,
                events: Arc::default(),
            }),
        })
    }

    fn submit(
        &self,
        callback: Function,
        destination: Tensor,
        bytes: u64,
    ) -> Result<TransferTicket> {
        let device = match destination.device().device_type as i32 {
            1 => None,
            2 => Some(destination.device().device_id),
            _ => return Err(failure("transfer destination requires CPU or CUDA storage")),
        };
        let state = Arc::new(ReadState {
            ticket: Mutex::new(NativeTicket::new(false)),
            events: Arc::clone(&self.data.events),
        });
        let read = Read {
            state: Arc::clone(&state),
            pool: Arc::downgrade(&self.data.pool),
            callback,
            device,
        };
        let task = self
            .data
            .pool
            .reserve(read, bytes, None)
            .map_err(|error| failure(error.to_string()))?;
        let prepared = (|| -> Result<()> {
            if let Some(device) = device {
                let stream =
                    unsafe { tvm_ffi::tvm_ffi_sys::TVMFFIEnvGetStream(2, device) as usize };
                lock(&state.ticket)?
                    .prepare_copy(&mut *lock(&state.events)?, device, stream)
                    .map_err(|error| failure(error.to_string()))?;
            }
            task.submit().map_err(|error| failure(error.to_string()))
        })();
        if let Err(error) = prepared {
            if let Err(cleanup) = task.cancel(true) {
                Read::report(cleanup);
            }
            return Err(error);
        }

        Ok(TransferTicket {
            data: ObjectArc::new(TransferTicketObj {
                object: Object::new(),
                state,
                task: Arc::downgrade(&task),
                destination,
            }),
        })
    }
}

impl Drop for TransferPoolObj {
    fn drop(&mut self) {
        if let Some(error) = self.pool.close() {
            eprintln!("Transfer close failed: {}", error.to_ffi());
        }
    }
}

#[repr(C)]
#[derive(Object)]
#[type_key = "uniserve.ffi.TransferTicket"]
pub struct TransferTicketObj {
    object: Object,
    state: Arc<ReadState>,
    task: Weak<Task>,
    // SDK tensors stay on the observing thread. The numerical Function retains
    // source and destination allocations while executing on a native thread.
    destination: Tensor,
}

#[derive(Clone, ObjectRef)]
pub struct TransferTicket {
    data: ObjectArc<TransferTicketObj>,
}

impl TransferTicket {
    fn result(&self) -> Result<Tensor> {
        let mut ticket = lock(&self.data.state.ticket)?;
        match ticket
            .result()
            .map_err(|error| failure(error.to_string()))?
        {
            Outcome::Failed(error) => return Err(error.to_ffi()),
            Outcome::Cancelled => return Err(failure("transfer read was cancelled")),
            Outcome::Success(_) => {}
        }
        let device = self.data.destination.device();
        if device.device_type as i32 == 2 {
            let stream =
                unsafe { tvm_ffi::tvm_ffi_sys::TVMFFIEnvGetStream(2, device.device_id) as usize };
            ticket
                .consume(device.device_id, stream)
                .map_err(|error| failure(error.to_string()))?;
        }
        Ok(self.data.destination.clone())
    }

    fn cancel(&self) -> Result<()> {
        let callbacks =
            lock(&self.data.state.ticket)?.cancel(failure("transfer read was cancelled").into());
        Read::notify(callbacks);
        if let Some(task) = self.data.task.upgrade() {
            task.cancel(false).map_err(|error| error.to_ffi())?;
        }
        Ok(())
    }
}

pub fn register() -> Result<()> {
    object::<TransferPoolObj>();
    object::<TransferTicketObj>();

    method::<TransferPoolObj>(
        "__ffi_init__",
        Function::from_typed(TransferPool::new),
        "Create a native read pool with byte and read capacities.",
    )?;
    method::<TransferPoolObj>(
        "submit",
        Function::from_typed(
            |pool: TransferPool, callback: Function, destination: Tensor, bytes: u64| {
                pool.submit(callback, destination, bytes)
            },
        ),
        "Read into destination on the supplied stream; the callback retains its tensors.",
    )?;
    method::<TransferPoolObj>(
        "used",
        Function::from_typed(|pool: TransferPool| -> Result<i64> {
            i64::try_from(pool.data.capacity.used()).map_err(|error| failure(error.to_string()))
        }),
        "Return bytes held by reads whose physical access has not ended.",
    )?;
    method::<TransferPoolObj>(
        "close",
        Function::from_typed(|pool: TransferPool| -> Result<()> {
            match pool.data.pool.close() {
                Some(error) => Err(error.to_ffi()),
                None => Ok(()),
            }
        }),
        "Join reads and close admission outside the Python GIL.",
    )?;
    method::<TransferTicketObj>(
        "ready",
        Function::from_typed(|ticket: TransferTicket| -> Result<bool> {
            Ok(lock(&ticket.data.state.ticket)?.ready())
        }),
        "Query result readiness without waiting for device completion.",
    )?;
    method::<TransferTicketObj>(
        "retirement_ready",
        Function::from_typed(|ticket: TransferTicket| {
            lock(&ticket.data.state.ticket)?
                .retirement_ready()
                .map_err(|error| failure(error.to_string()))
        }),
        "Query whether physical access ended and its credits returned.",
    )?;
    method::<TransferTicketObj>(
        "result",
        Function::from_typed(|ticket: TransferTicket| ticket.result()),
        "Join the readable result onto the current FFI stream.",
    )?;
    method::<TransferTicketObj>(
        "cancel",
        Function::from_typed(|ticket: TransferTicket| ticket.cancel()),
        "Cancel consumption and withdraw a queued read; running access retains its credits.",
    )?;
    method::<TransferTicketObj>(
        "add_done_callback",
        Function::from_typed(|ticket: TransferTicket, callback: Function| -> Result<()> {
            let immediate = lock(&ticket.data.state.ticket)?.add_done_callback(callback);
            if let Some(callback) = immediate {
                Read::wake(&callback);
            }
            Ok(())
        }),
        "Observe result readiness outside the native ticket lock.",
    )?;
    method::<TransferTicketObj>(
        "add_retirement_callback",
        Function::from_typed(|ticket: TransferTicket, callback: Function| -> Result<()> {
            let immediate = lock(&ticket.data.state.ticket)?
                .retirement
                .subscribe(callback);
            if let Some(callback) = immediate {
                Read::wake(&callback);
            }
            Ok(())
        }),
        "Observe physical completion and returned read credits.",
    )?;
    Ok(())
}
