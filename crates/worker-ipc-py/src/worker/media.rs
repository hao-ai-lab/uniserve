//! Request-owned containers and codec work on the shared host lane.

mod container;
mod units;

pub(super) use container::MuxSession;
pub(super) use units::{frame_encoded_unit, host_array, read_encoded_unit};

use units::parse_units;

use std::sync::Arc;

use pyo3::buffer::PyBuffer;
use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PySlice, PyTuple};
use uniserve_worker_ipc::{TensorTransfer, TransferTransport};

use super::completion::Completion;
use super::error::invalid;
use super::locator::Locator;
use super::output::OutputBuffer;
use super::pending::PendingOutput;
use super::shared_buffer::SharedRead;
use super::storage::TensorStore;
use super::transfer::TransferTicket;
use super::transport::Transport;

pub(super) enum MediaTask {
    Image {
        pixels: Py<PyAny>,
        buffer: Py<OutputBuffer>,
        maximum: usize,
    },
    Video {
        config: Py<PyAny>,
        source: Py<PyAny>,
    },
    Audio {
        session: Arc<MuxSession>,
        pcm: Py<PyAny>,
    },
    Append {
        session: Arc<MuxSession>,
        units: Vec<Py<PyBytes>>,
    },
    Finalize {
        session: Arc<MuxSession>,
    },
}

impl MediaTask {
    pub(super) fn ready(&self, py: Python<'_>) -> PyResult<bool> {
        match self {
            Self::Image { buffer, .. } => buffer.get().ready(py),
            _ => Ok(true),
        }
    }

    pub(super) fn completion(&self, py: Python<'_>) -> PyResult<Option<Py<Completion>>> {
        match self {
            Self::Image { buffer, .. } => buffer.get().completion(py).map(Some),
            _ => Ok(None),
        }
    }

    pub(super) fn run(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            Self::Image {
                pixels, maximum, ..
            } => {
                let encoded = py
                    .import("uniserve_worker.media.codec")?
                    .call_method1("uint8_image_to_png_base64_bytes", (pixels,))?;
                let size = encoded.len()?;
                if size == 0 || size > *maximum {
                    return Err(PyRuntimeError::new_err(
                        "encoded image is empty or exceeds its completion byte bound",
                    ));
                }
                Ok(encoded.unbind())
            }
            Self::Video { config, source } => py
                .import("uniserve_worker.media.mux")?
                .call_method1("encode_unit", (config, source))
                .map(Bound::unbind),
            Self::Audio { session, pcm } => {
                session.encode_audio(py, pcm.bind(py))?;
                Ok(py.None())
            }
            Self::Append { session, units } => {
                session.append(py, units.iter().map(|unit| unit.clone_ref(py)).collect())?;
                Ok(py.None())
            }
            Self::Finalize { session } => session.finish(py),
        }
    }

    pub(super) fn release(&self, py: Python<'_>) -> PyResult<()> {
        if let Self::Image { buffer, .. } = self {
            return buffer.get().release_reader(py);
        }
        if let Self::Video { source, .. } = self
            && let Ok(borrow) = source.bind(py).cast::<SharedRead>()
        {
            borrow.get().release(py)?;
        }
        Ok(())
    }

    pub(super) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        match self {
            Self::Image { pixels, buffer, .. } => {
                visit.call(pixels)?;
                visit.call(buffer)?;
            }
            Self::Video { config, source } => {
                visit.call(config)?;
                visit.call(source)?;
            }
            Self::Audio { pcm, .. } => visit.call(pcm)?,
            Self::Append { units, .. } => {
                for unit in units {
                    visit.call(unit)?;
                }
            }
            Self::Finalize { .. } => {}
        }
        // Containers only reference codec data, never request or task owners.
        Ok(())
    }
}

/// Tensors filled by host work remain deferred until all its tasks succeed.
pub(super) struct HostTensors {
    pub(super) values: Vec<Py<PyAny>>,
    pub(super) encoded: bool,
    pub(super) store: Py<TensorStore>,
    pub(super) transports: Py<PyAny>,
}

impl HostTensors {
    pub(super) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.store)?;
        visit.call(&self.transports)?;
        for value in &self.values {
            visit.call(value)?;
        }
        Ok(())
    }

    pub(super) fn finish(
        self,
        output: &Bound<'_, PendingOutput>,
        results: Vec<Py<PyAny>>,
    ) -> PyResult<()> {
        let py = output.py();
        let regions = if self.encoded {
            let rows = self.values[0].bind(py).call_method1("unbind", (0,))?;
            if rows.len()? != results.len() {
                return Err(invalid(
                    py,
                    "encoded results do not fill their reserved rows",
                ));
            }
            let mut regions = Vec::with_capacity(results.len());
            for (index, result) in results.iter().enumerate() {
                let framed =
                    frame_encoded_unit(result.bind(py).cast::<PyBytes>()?, &rows.get_item(index)?)?;
                let length = framed.call_method0("numel")?.extract::<isize>()?;
                regions.push(PyTuple::new(
                    py,
                    [
                        PySlice::new(py, index as isize, index as isize + 1, 1),
                        PySlice::new(py, 0, length, 1),
                    ],
                )?);
            }
            Some(vec![regions])
        } else {
            None
        };
        let call = Arc::clone(
            &output
                .borrow()
                .call
                .bind(py)
                .cast::<crate::calls::Call>()?
                .get()
                .inner,
        );
        PendingOutput::export_tensors(
            output,
            &call,
            self.values
                .iter()
                .map(|value| value.bind(py).clone())
                .collect(),
            self.store.get(),
            self.transports.bind(py),
            true,
            regions,
        )
    }
}

/// Read framed byte rows without touching their uninitialized capacity.
pub(super) fn read_units(
    py: Python<'_>,
    tensor: &TensorTransfer,
    transports: &Bound<'_, PyDict>,
) -> PyResult<Vec<Py<PyBytes>>> {
    if tensor.shape.len() != 2
        || tensor
            .locations
            .iter()
            .any(|location| location.dtype != "uint8")
    {
        return Err(invalid(py, "encoded units require byte rows"));
    }
    let mut units: Vec<Option<Py<PyBytes>>> = (0..tensor.shape[0]).map(|_| None).collect();
    for location in &tensor.locations {
        let backend = match &location.transport {
            TransferTransport::PosixShm { .. } => "shm",
            TransferTransport::Local { .. } => "local",
            TransferTransport::Channel { .. } => "channel",
            _ => continue,
        };
        let transport = transports.get_item(backend)?;
        let transport = transport
            .as_ref()
            .map(|value| value.cast::<Transport>())
            .transpose()?;
        if backend != "channel"
            && !transport.is_some_and(|transport| {
                let source = &transport.get().source;
                if backend == "shm" {
                    source.node == location.source.node
                } else {
                    source.address_space == location.source.address_space
                }
            })
        {
            continue;
        }
        if location.offset[1] != 0 {
            return Err(invalid(py, "encoded unit location lacks its length prefix"));
        }
        let first = location.offset[0] as usize;
        let count = location.shape[0] as usize;
        if units[first..first + count].iter().all(Option::is_some) {
            continue;
        }
        let locator = Locator::wrap(py, location.clone())?;
        let rows = match (&location.transport, transport) {
            (TransferTransport::Channel { payload, .. }, _) => {
                parse_units(py, payload, location.shape[1] as usize)
            }
            (TransferTransport::PosixShm { .. }, Some(transport)) => {
                let read = Bound::new(py, transport.get().borrow(py, &locator, None)?)?;
                let result = PyBuffer::<u8>::get(read.as_any()).and_then(|buffer| {
                    let raw = buffer.to_vec(py)?;
                    parse_units(py, &raw, location.shape[1] as usize)
                });
                read.get().release(py)?;
                result
            }
            (TransferTransport::Local { .. }, Some(transport)) => {
                let device = py
                    .import("torch")?
                    .call_method1("device", ("cpu",))?
                    .unbind();
                let ticket = transport
                    .get()
                    .fetch(py, locator, device, None, None, None)?;
                let result = (|| {
                    let value = ticket.get().result(py, None)?;
                    let array = value.bind(py).call_method0("numpy")?;
                    let raw = PyBuffer::<u8>::get(&array)?.to_vec(py)?;
                    parse_units(py, &raw, location.shape[1] as usize)
                })();
                TransferTicket::close(ticket.into_bound(py))?;
                result
            }
            _ => continue,
        }?;
        for (index, value) in (first..first + count).zip(rows) {
            units[index] = Some(value);
        }
    }
    units
        .into_iter()
        .map(|unit| {
            unit.ok_or_else(|| invalid(py, "artifact assembly requires every encoded media unit"))
        })
        .collect()
}
