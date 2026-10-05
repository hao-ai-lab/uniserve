//! Completion predicates use one readback buffer per batch. Canvas continuation
//! stays on the device; imported predicates join their source fences before copy.

use indexmap::IndexMap;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use uniserve_worker::InputReady;
use uniserve_worker_ipc::{CallKind, DType, ForwardMode};

use super::{BatchState, PythonBackend};
use crate::worker::inputs::Input;
use crate::worker::output::OutputBuffer;
use crate::worker::storage::TensorRead;

impl PythonBackend {
    pub(super) fn prepare_predicates(&self, py: Python<'_>, batch: &BatchState) -> PyResult<()> {
        // Canvas steps use the request slot's continuation flag. Reading it
        // here would serialize successors that can already run on the device.
        let selected: Vec<_> = batch
            .plan
            .calls
            .iter()
            .enumerate()
            .filter(|(_, call)| {
                call.predicate
                    .as_ref()
                    .is_some_and(|value| value.dtype == DType::U8)
                    && !(call.code == CallKind::Forward(ForwardMode::TokenDenoising)
                        && call.canvas.is_some())
            })
            .map(|(index, _)| index)
            .collect();
        if selected.is_empty() {
            return Ok(());
        }

        let numerical = batch.numerical.borrow(py).batch.clone_ref(py);
        let calls = numerical.bind(py).getattr("calls")?;
        let mut devices = Vec::with_capacity(selected.len());
        for &index in &selected {
            devices.push(
                self.model_runner
                    .bind(py)
                    .call_method1("call_devices", (calls.get_item(index)?,))?
                    .get_item(0)?,
            );
        }
        let buffer =
            self.output_pool
                .get()
                .acquire(py, selected.len(), selected.len(), devices.clone())?;
        let mut captures: Vec<_> = selected.iter().map(|&index| (index, None)).collect();
        let mut recorded: Vec<Bound<'_, TensorRead>> = Vec::new();

        let prepared: PyResult<()> = (|| {
            let mut groups: IndexMap<String, (Bound<'_, PyAny>, Vec<usize>)> = IndexMap::new();
            for (row, &index) in selected.iter().enumerate() {
                let call = &batch.plan.calls[index];
                let Some(predicate) = &call.predicate else {
                    continue;
                };
                if matches!(
                    batch.inputs.borrow(py).get(py, predicate.buffer_id()),
                    Some(Input::Tensor(_))
                ) {
                    continue;
                }

                let device = &devices[row];
                groups
                    .entry(device.str()?.to_str()?.to_owned())
                    .or_insert_with(|| (device.clone(), Vec::new()))
                    .1
                    .push(row);
            }

            // Acquire local reads by device, then fence every accepted lease,
            // including leases whose copy fails before the group completes.
            for (_, (device, rows)) in groups {
                let requests = rows
                    .iter()
                    .map(|&row| {
                        let call = calls.get_item(selected[row])?;
                        Ok((call.getattr("predicate")?, call.getattr("call_id")?, None))
                    })
                    .collect::<PyResult<_>>()?;
                let reads = self
                    .tensors
                    .get()
                    .consume_batch(py, requests, Some(device.clone()))?;
                let reads: Vec<Bound<'_, TensorRead>> = reads.extract()?;
                recorded.extend(reads.iter().cloned());
                for (&row, read) in rows.iter().zip(&reads) {
                    let tensor = read.borrow().tensor.clone_ref(py);
                    captures[row].1 = Some(buffer.get().capture(py, tensor.bind(py))?);
                }
                self.tensors
                    .get()
                    .complete_reads(py, reads, Some(device), Vec::new())?;
            }

            if captures.iter().all(|(_, span)| span.is_some()) {
                OutputBuffer::seal(buffer.bind(py))?;
            }
            Ok(())
        })();
        if let Err(error) = prepared {
            let reads = self
                .tensors
                .get()
                .complete_reads(py, recorded, None, Vec::new());
            let abandoned = OutputBuffer::abandon(buffer.bind(py));
            for failure in [reads, abandoned].into_iter().filter_map(Result::err) {
                let _ = error.value(py).call_method1(
                    "add_note",
                    (format!("predicate cleanup failed: {failure}"),),
                );
            }
            return Err(error);
        }

        batch.numerical.borrow_mut(py).predicate_captures = captures;
        batch.inputs.borrow_mut(py).set_predicate(Some(buffer));
        Ok(())
    }

    pub(super) fn capture_predicates(&self, py: Python<'_>, batch: &BatchState) -> PyResult<()> {
        let Some(buffer) = batch.inputs.borrow(py).predicate(py) else {
            return Ok(());
        };
        if buffer.get().sealed(py)? {
            return Ok(());
        }

        let captures = batch.numerical.borrow(py).predicate_captures.clone();
        let mut pending = Vec::new();
        for (row, (index, span)) in captures.into_iter().enumerate() {
            if span.is_some() {
                continue;
            }
            let Some(predicate) = &batch.plan.calls[index].predicate else {
                continue;
            };
            let input = batch
                .inputs
                .borrow(py)
                .get(py, predicate.buffer_id())
                .ok_or_else(|| PyRuntimeError::new_err("predicate import has no input owner"))?;
            if !input.ready()? {
                return Ok(());
            }
            let Input::Tensor(read) = input else {
                return Err(PyRuntimeError::new_err(
                    "predicate import has no tensor view",
                ));
            };
            pending.push((row, read));
        }

        // Readiness exposes the tensor before physical transfer completion.
        // wait_import orders its producer fence on the copy stream without a host wait.
        let captured = (|| {
            for (row, read) in pending {
                self.tensors.get().wait_import(py, read.bind(py))?;
                let tensor = read.borrow(py).tensor.clone_ref(py);
                let span = buffer.get().capture(py, tensor.bind(py))?;
                batch.numerical.borrow_mut(py).predicate_captures[row].1 = Some(span);
            }
            OutputBuffer::seal(buffer.bind(py))
        })();
        if let Err(error) = captured {
            OutputBuffer::abandon(buffer.bind(py))?;
            return Err(error);
        }
        Ok(())
    }
}
