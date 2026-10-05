//! Plan tensor reads and submit their shared credit reservation.

use std::ops::Range;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};

use super::error::invalid;
use super::transfer::{ReadReservation, TransferCapacity, TransferTicket};

pub(super) struct PlannedRead<'py> {
    transport: Bound<'py, PyAny>,
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
    let device = target_device(destination)?;

    let layout = py.import("uniserve_worker.transport.layout")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item(
        "shape",
        PyTuple::new(py, region.iter().map(|axis| axis.end - axis.start))?,
    )?;
    kwargs.set_item("dtype", tensor.getattr("dtype")?)?;
    kwargs.set_item("device", &device)?;
    layout
        .getattr("validate_destination")?
        .call((destination,), Some(&kwargs))?;

    // Prefer a process-local replica; other locations keep publication order.
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
    let mut bound = Vec::new();
    let locations = ordered
        .into_iter()
        .map(|(location, backend)| {
            let transport =
                bindings.call_method1("get", ((location.getattr("source")?, &backend),))?;
            if transport.is_none() {
                return Ok(None);
            }
            if transport.getattr("name")?.extract::<String>()? != backend {
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
            let index = bound.len();
            bound.push((transport, location));
            Ok(Some((index, covered)))
        })
        .filter_map(Result::transpose);
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
            location: location.clone(),
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
    let capacity: Py<TransferCapacity> = first.transport.getattr("capacity")?.extract()?;
    for read in &reads[1..] {
        if !read.transport.getattr("capacity")?.is(capacity.bind(py)) {
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
            let kwargs = PyDict::new(py);
            kwargs.set_item("device", target_device(&read.target)?)?;
            kwargs.set_item("destination", &read.target)?;
            kwargs.set_item("region", &read.source_region)?;
            kwargs.set_item("reservation", &reservation)?;
            let ticket: Py<TransferTicket> = read
                .transport
                .call_method("fetch", (&read.location,), Some(&kwargs))?
                .extract()?;
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
