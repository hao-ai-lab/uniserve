//! Plan tensor reads and submit their shared credit reservation.

use std::ops::Range;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};

use uniserve_worker_ipc::{TensorTransfer, TransferTransport};

use super::error::invalid;
use super::transfer::{ReadReservation, TransferTicket};
use super::transport::Transport;
use crate::convert;

pub(super) struct PlannedRead<'py> {
    transport: Bound<'py, Transport>,
    location: Bound<'py, PyAny>,
    source_region: Bound<'py, PyTuple>,
    target: Bound<'py, PyAny>,
}

/// Deliver the requested logical region through bound physical transports.
/// Reserve the complete fan-out before submitting any read. A backend or
/// observer error cancels submitted tickets without releasing physical accesses.
#[pyfunction]
#[pyo3(signature = (tensor, destination, *, bindings, region=None, retain=None))]
pub(super) fn fetch_tensor(
    py: Python<'_>,
    tensor: &Bound<'_, PyAny>,
    destination: &Bound<'_, PyAny>,
    bindings: &Bound<'_, PyAny>,
    region: Option<&Bound<'_, PyTuple>>,
    retain: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyTuple>> {
    let shape: Vec<u64> = tensor.getattr("shape")?.extract()?;
    let region = requested_region(py, region, &shape)?;
    let reads = plan_reads(py, tensor, destination, bindings, &region)?;
    let tickets = submit_reads(py, &reads, |ticket| {
        if let Some(retain) = retain {
            retain.call1((ticket,))?;
        }
        Ok(())
    })?;
    PyTuple::new(py, tickets).map(Bound::unbind)
}

pub(super) fn plan_reads<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyAny>,
    destination: &Bound<'py, PyAny>,
    bindings: &Bound<'py, PyAny>,
    region: &[Range<u64>],
) -> PyResult<Vec<PlannedRead<'py>>> {
    // Prefer a process-local replica; other locations keep export order.
    let mut ordered = tensor
        .getattr("locations")?
        .try_iter()?
        .map(|location| {
            let location = location?;
            let backend: String = location.getattr("backend")?.extract()?;
            Ok((location, backend))
        })
        .collect::<PyResult<Vec<_>>>()?;
    ordered.sort_by_key(|(_, backend)| backend != "local");

    // Resolve bounds only until coverage is complete. Keep handles borrowed,
    // without copying channel payloads or inspecting unused replica metadata.
    let locations = ordered
        .into_iter()
        .map(|(location, backend)| {
            let transport =
                bindings.call_method1("get", ((location.getattr("source")?, &backend),))?;
            if transport.is_none() {
                return Ok(None);
            }
            let transport = transport.cast_into::<Transport>()?;
            if transport.get().name() != backend {
                return Err(invalid(
                    py,
                    "bound transport disagrees with the physical edge",
                ));
            }

            let offset: Vec<u64> = location.getattr("offset")?.extract()?;
            let shape: Vec<u64> = location.getattr("shape")?.extract()?;
            let covered = offset
                .into_iter()
                .zip(shape)
                .map(|(start, extent)| start..start + extent)
                .collect::<Vec<_>>();
            Ok(Some((transport, location, covered)))
        })
        .filter_map(Result::transpose);
    let dtype: String = tensor.getattr("dtype")?.extract()?;
    plan_views(py, destination, &dtype, region, locations, |location| {
        Ok(location.clone())
    })
}

/// Native callers retain their descriptor and convert only selected physical
/// locators for the numerical transport. Channel payloads are not round-tripped.
pub(super) fn plan_native_reads<'py>(
    py: Python<'py>,
    tensor: &TensorTransfer,
    destination: &Bound<'py, PyAny>,
    transports: &Bound<'py, PyAny>,
    region: &[Range<u64>],
    locators: &mut [Option<Bound<'py, PyAny>>],
) -> PyResult<Vec<PlannedRead<'py>>> {
    let mut ordered: Vec<_> = tensor.locations.iter().enumerate().collect();
    ordered.sort_by_key(|(_, location)| {
        !matches!(location.transport, TransferTransport::Local { .. })
    });
    let locations = ordered
        .into_iter()
        .map(|(index, location)| {
            let name = match location.transport {
                TransferTransport::Local { .. } => "local",
                TransferTransport::PosixShm { .. } => "shm",
                TransferTransport::CudaVmm { .. } => "cuda_vmm",
                TransferTransport::Channel { .. } => "channel",
            };
            let transport = transports.call_method1("get", (name,))?;
            if transport.is_none() {
                return Ok(None);
            }
            let transport = transport.cast_into::<Transport>()?;
            let covered = location
                .offset
                .iter()
                .zip(&location.shape)
                .map(|(&start, &extent)| start..start + extent)
                .collect();
            Ok(Some((transport, (index, location), covered)))
        })
        .filter_map(Result::transpose);
    plan_views(
        py,
        destination,
        &tensor.locations[0].dtype,
        region,
        locations,
        |&(index, location)| {
            // A KV import can read many layers from the same descriptor.
            // Retain one Python locator so channel bytes cross only once.
            if let Some(location) = &locators[index] {
                return Ok(location.clone());
            }
            let location = convert::transfer_locator_to_py(py, location)?;
            locators[index] = Some(location.clone());
            Ok(location)
        },
    )
}

fn plan_views<'py, L>(
    py: Python<'py>,
    destination: &Bound<'py, PyAny>,
    dtype: &str,
    region: &[Range<u64>],
    locations: impl Iterator<Item = PyResult<(Bound<'py, Transport>, L, Vec<Range<u64>>)>>,
    mut convert: impl FnMut(&L) -> PyResult<Bound<'py, PyAny>>,
) -> PyResult<Vec<PlannedRead<'py>>> {
    let device = target_device(destination)?;

    let layout = py.import("uniserve_worker.transport.layout")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item(
        "shape",
        PyTuple::new(py, region.iter().map(|axis| axis.end - axis.start))?,
    )?;
    kwargs.set_item("dtype", dtype)?;
    kwargs.set_item("device", &device)?;
    layout
        .getattr("validate_destination")?
        .call((destination,), Some(&kwargs))?;

    let mut bound = Vec::new();
    let locations = locations.map(|location| -> PyResult<_> {
        let (transport, location, covered) = location?;
        let index = bound.len();
        bound.push((transport, location));
        Ok((index, covered))
    });
    let reads = uniserve_worker::plan_reads(region, locations)?.ok_or_else(|| {
        invalid(
            py,
            "bound product locations do not cover the consumer region",
        )
    })?;

    let region_view = layout.getattr("region_view")?;
    let mut planned = Vec::with_capacity(reads.len());
    for read in reads {
        let (transport, location) = &bound[read.location];
        let target_region = slices(py, &read.destination)?;
        planned.push(PlannedRead {
            transport: transport.clone(),
            location: convert(location)?,
            source_region: slices(py, &read.source)?,
            target: region_view.call1((destination, target_region))?,
        });
    }

    Ok(planned)
}

pub(super) fn submit_reads(
    py: Python<'_>,
    reads: &[PlannedRead<'_>],
    mut retain: impl FnMut(&Py<TransferTicket>) -> PyResult<()>,
) -> PyResult<Vec<Py<TransferTicket>>> {
    let Some(first) = reads.first() else {
        return Ok(Vec::new());
    };
    let capacity = first.transport.get().capacity.clone_ref(py);
    for read in &reads[1..] {
        if !read.transport.get().capacity.is(&capacity) {
            return Err(invalid(
                py,
                "a fetch's transports must share the rank's transfer capacity",
            ));
        }
    }

    let reservation = Py::new(
        py,
        ReadReservation::new(py, capacity, reads.len() as isize)?,
    )?;
    let mut tickets: Vec<Py<TransferTicket>> = Vec::with_capacity(reads.len());
    let result = (|| -> PyResult<()> {
        for read in reads {
            let ticket = read.transport.get().fetch(
                py,
                read.location.clone(),
                target_device(&read.target)?.unbind(),
                Some(read.target.clone()),
                Some(read.source_region.clone().into_any().unbind()),
                Some(reservation.clone_ref(py)),
            )?;
            tickets.push(ticket.clone_ref(py));
            retain(&ticket)?;
        }
        Ok(())
    })();

    if let Err(error) = &result {
        for ticket in &tickets {
            if let Err(cleanup) = ticket.get().cancel(py) {
                let _ = error
                    .value(py)
                    .call_method1("add_note", (cleanup.to_string(),));
            }
        }
    }

    let cleanup = reservation.get().close(py);
    if let Err(error) = result {
        if let Err(cleanup) = cleanup {
            let _ = error
                .value(py)
                .call_method1("add_note", (cleanup.to_string(),));
        }
        return Err(error);
    }
    cleanup?;

    Ok(tickets)
}

pub(super) fn requested_region(
    py: Python<'_>,
    region: Option<&Bound<'_, PyTuple>>,
    shape: &[u64],
) -> PyResult<Vec<Range<u64>>> {
    let Some(region) = region else {
        return Ok(shape.iter().map(|&extent| 0..extent).collect());
    };
    if region.len() != shape.len() {
        return Err(invalid(py, "consumer region exceeds the logical tensor"));
    }

    region
        .iter()
        .zip(shape)
        .map(|(axis, &extent)| {
            let axis = axis.cast::<PySlice>()?;
            let start: u64 = axis.getattr("start")?.extract()?;
            let end: u64 = axis.getattr("stop")?.extract()?;
            let step: Option<i64> = axis.getattr("step")?.extract()?;
            if start > end || step.is_some_and(|step| step != 1) {
                return Err(PyValueError::new_err(
                    "tensor slices require nonnegative explicit bounds and unit steps",
                ));
            }
            if end > extent {
                return Err(invalid(py, "consumer region exceeds the logical tensor"));
            }
            Ok(start..end)
        })
        .collect()
}

pub(super) fn slices<'py>(py: Python<'py>, region: &[Range<u64>]) -> PyResult<Bound<'py, PyTuple>> {
    let axes = region
        .iter()
        .map(|axis| {
            Ok(PySlice::new(
                py,
                isize::try_from(axis.start)?,
                isize::try_from(axis.end)?,
                1,
            ))
        })
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, axes)
}

fn target_device<'py>(target: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(spans) = target.cast::<PyTuple>() {
        if spans.is_empty() {
            return Err(invalid(target.py(), "tensor destination has no spans"));
        }
        spans.get_item(0)?.getattr("device")
    } else {
        target.getattr("device")
    }
}
