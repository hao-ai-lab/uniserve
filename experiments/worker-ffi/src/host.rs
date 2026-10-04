//! Python host callbacks on the production native task executor.

use std::sync::Arc;

use tvm_ffi::object::ObjectRefCore;
use tvm_ffi::tvm_ffi_sys::{TVMFFIByteArray, TVMFFIErrorCreate};
use tvm_ffi::{Bytes, Error, Function, ObjectArc};
use uniserve_worker::{HostAction, HostTask, Outcome};

/// The same error value representation TVM-FFI uses for serialization. The
/// SDK's Error handle is not Send; native completions retain its three strings.
pub struct CallbackError {
    kind: String,
    message: String,
    backtrace: String,
}

impl From<Error> for CallbackError {
    fn from(error: Error) -> Self {
        Self {
            kind: error.kind().as_str().to_owned(),
            message: error.message().to_owned(),
            backtrace: error.backtrace().to_owned(),
        }
    }
}

impl CallbackError {
    pub fn to_ffi(&self) -> Error {
        // ErrorKind has no public string constructor in the pinned Rust SDK.
        // The C API copies these views and returns one owning error reference.
        unsafe {
            let kind = TVMFFIByteArray::from_str(&self.kind);
            let message = TVMFFIByteArray::from_str(&self.message);
            let backtrace = TVMFFIByteArray::from_str(&self.backtrace);
            let mut error = std::ptr::null_mut();

            if TVMFFIErrorCreate(&kind, &message, &backtrace, &mut error) != 0 {
                return Error::from_raised();
            }

            Error::from_data(ObjectArc::from_raw(error.cast()))
        }
    }
}

pub struct Action(pub Function);

impl HostAction for Action {
    type Output = Option<Vec<u8>>;
    type Error = CallbackError;
    type Callback = (Function, crate::HostTask);
    type Wake = ();

    fn ready(&self) -> Result<bool, CallbackError> {
        Ok(true)
    }

    fn run(&self) -> Result<Self::Output, CallbackError> {
        // FFI values stay on the calling thread. Host outputs are completed
        // bytes (encoded media) or None (in-place staging), owned by Rust here.
        let output = self.0.call_tuple(()).and_then(Option::<Bytes>::try_from)?;
        Ok(output.map(|bytes| bytes.as_slice().to_vec()))
    }

    fn release(&self) -> Result<(), CallbackError> {
        Ok(())
    }

    fn input_outcome(&self) -> Option<Outcome<CallbackError>> {
        Some(Outcome::Success(()))
    }

    fn defer_release(_task: Arc<HostTask<Self>>) -> Result<(), CallbackError> {
        unreachable!("host callbacks have no asynchronous input lease")
    }

    fn notify(callbacks: Vec<Self::Callback>) {
        for (callback, task) in callbacks {
            if let Err(error) = callback.call_tuple((task,)) {
                Self::report(error.into());
            }
        }
    }

    fn wake(_: &()) {}

    fn report(error: CallbackError) {
        eprintln!(
            "Host completion callback failed: {}: {}",
            error.kind, error.message
        );
    }

    fn error(error: uniserve_worker::Error) -> CallbackError {
        CallbackError {
            kind: "RuntimeError".into(),
            message: error.to_string(),
            backtrace: String::new(),
        }
    }

    fn note_cleanup(error: &mut CallbackError, cleanup: CallbackError) {
        error
            .message
            .push_str(&format!("\nHost cleanup also failed: {}", cleanup.message));
    }
}
