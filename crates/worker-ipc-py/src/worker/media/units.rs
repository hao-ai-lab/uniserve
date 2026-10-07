//! Length-prefixed encoded units in host tensor rows.

use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PySlice};

use crate::worker::error::invalid;
use crate::worker::host_buffers::with_host;

const LENGTH_BYTES: usize = 8;

/// Write one native-endian uint64 length and its bytes. Only the returned
/// initialized span may be exported; the unused row capacity stays untouched.
#[pyfunction]
pub(in crate::worker) fn frame_encoded_unit<'py>(
    payload: &Bound<'py, PyBytes>,
    destination: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = payload.py();
    let bytes = payload.as_bytes();
    let capacity = destination.call_method0("numel")?.extract::<usize>()?;
    if capacity < LENGTH_BYTES || bytes.len() > capacity - LENGTH_BYTES {
        return Err(invalid(
            py,
            format!(
                "encoded media unit of {} bytes exceeds the {} bytes reserved for it",
                bytes.len(),
                capacity.saturating_sub(LENGTH_BYTES),
            ),
        ));
    }

    let length = LENGTH_BYTES + bytes.len();
    let header = (bytes.len() as u64).to_ne_bytes();
    with_host::<u8>(destination, length, |cells| {
        for (cell, value) in cells.iter().zip(header.iter().chain(bytes)) {
            cell.set(*value);
        }
        Ok(())
    })?;
    destination.get_item(PySlice::new(py, 0, length as isize, 1))
}

/// Copy only the initialized payload from a contiguous CPU byte row.
#[pyfunction]
pub(in crate::worker) fn read_encoded_unit(row: &Bound<'_, PyAny>) -> PyResult<Py<PyBytes>> {
    let py = row.py();
    let array = row.call_method0("numpy")?;
    let buffer = PyBuffer::<u8>::get(&array)?;
    let cells = buffer
        .as_slice(py)
        .ok_or_else(|| invalid(py, "encoded media unit requires contiguous host bytes"))?;
    decode(py, cells.iter().map(|cell| cell.get()))
}

pub(super) fn parse_units(
    py: Python<'_>,
    bytes: &[u8],
    stride: usize,
) -> PyResult<Vec<Py<PyBytes>>> {
    if stride < LENGTH_BYTES {
        return Err(invalid(py, "encoded media unit has no length prefix"));
    }
    bytes
        .chunks_exact(stride)
        .map(|row| decode(py, row.iter().copied()))
        .collect()
}

fn decode(py: Python<'_>, mut row: impl ExactSizeIterator<Item = u8>) -> PyResult<Py<PyBytes>> {
    if row.len() < LENGTH_BYTES {
        return Err(invalid(py, "encoded media unit has no length prefix"));
    }
    let mut header = [0; LENGTH_BYTES];
    for (byte, value) in header.iter_mut().zip(row.by_ref()) {
        *byte = value;
    }
    let length = u64::from_ne_bytes(header) as usize;
    if length > row.len() {
        return Err(invalid(py, "encoded media unit names an invalid length"));
    }

    // Borrowed tensor storage remains held while PyBytes copies the payload.
    // No temporary ndarray or tensor is allocated for either header or body.
    Ok(PyBytes::new_with(py, length, |bytes| {
        for (destination, source) in bytes.iter_mut().zip(row) {
            *destination = source;
        }
        Ok(())
    })?
    .unbind())
}

/// Imported storage may retire before a codec runs on its host lane. Keep an
/// independent CPU array for that task, including its full element layout.
pub(in crate::worker) fn host_array<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    value
        .call_method0("detach")?
        .call_method0("contiguous")?
        .call_method1("view", (value.py().import("torch")?.getattr("uint8")?,))?
        .call_method0("numpy")?
        .call_method0("copy")
}
