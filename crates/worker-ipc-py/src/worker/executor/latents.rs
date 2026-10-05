//! Export completed latent writes and resident trajectories without a model call.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use uniserve_worker_ipc::{
    CallKind, MediaCall, TensorExport, TensorTransfer, TransferHandle, TransferMode,
};

use super::{BatchState, PythonBackend};
use crate::convert;
use crate::worker::error::{invalid, native_error, unsupported};
use crate::worker::exports;

impl PythonBackend {
    pub(super) fn export_latents(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        active: &[usize],
    ) -> PyResult<()> {
        let transports = self.worker.bind(py).getattr("export_transports")?;
        let mut remote = false;
        for name in transports.try_iter()? {
            if name?.extract::<String>()? != "local" {
                remote = true;
                break;
            }
        }
        let outputs = batch.pending_outputs(py);

        for &index in active {
            let call = &batch.plan.calls[index];
            let Some(product) = &call.latent_output else {
                continue;
            };
            let pending = outputs[index].borrow();
            let slot = pending.request.borrow(py).request.slot();
            let (params, generation, step) = if call.code
                == CallKind::Transfer(TransferMode::Tensor)
            {
                if call.tensor_inputs().count() != 1 || call.tensor_outputs().count() != 1 {
                    return Err(invalid(
                        py,
                        "product transfer requires one physical input and one output",
                    ));
                }
                let input = call.latent_input.as_ref().ok_or_else(|| {
                    invalid(py, "product transfer changes the physical product kind")
                })?;
                let params = batch
                    .plan
                    .latent_params
                    .iter()
                    .find(|params| {
                        params.request_key == call.request_key && params.call_id == call.call_id
                    })
                    .ok_or_else(|| invalid(py, "latent transfer has no bound parameters"))?;
                (params.clone(), Some(input.generation), params.start_step)
            } else {
                // A rank without a numerical result has no bank to export.
                // Standalone denoisers do not declare a latent output here.
                let update = pending.lock(py)?.latent_update.clone();
                let Some(update) = update else {
                    continue;
                };
                if update.release
                    || !remote
                    || self.info.endpoint.rank as usize != self.output_rank(py, &call.component)?
                {
                    continue;
                }
                (update.params, None, update.step as u32)
            };
            drop(pending);

            let pool = self
                .latents
                .as_ref()
                .ok_or_else(|| unsupported(py, "latent export requires a physical latent pool"))?;
            let source = pool
                .borrow_mut(py)
                .reserve(py, product, slot, &params, generation)?;
            let locations = exports::export(
                &transports,
                source.get().spans.bind(py).as_any(),
                None,
                &call.consumer_slots,
                false,
                |retirement| {
                    pool.borrow(py)
                        .retain_export(py, source.bind(py), retirement)
                },
            )?;

            // Register every accepted location before converting the result.
            // Batch failure then revokes exports even if conversion fails.
            let pending = outputs[index].borrow();
            pending
                .exported_locators
                .bind(py)
                .call_method1("extend", (&locations,))?;
            let registrations = locations
                .iter()
                .map(|location| Ok((transports.get_item(location.getattr("backend")?)?, location)))
                .collect::<PyResult<Vec<_>>>()?;
            pending.latent_exports.bind(py).set_item(
                convert::buffer_id_to_py(py, &product.buffer_id())?,
                PyTuple::new(py, registrations)?,
            )?;

            let locations = locations
                .iter()
                .map(|location| {
                    convert::transfer_locator_from_py(&location.call_method0("to_mapping")?)
                        .ok_or_else(|| invalid(py, "transport returned an invalid latent location"))
                })
                .collect::<PyResult<Vec<_>>>()?;
            let export = TensorExport {
                product: product.clone(),
                value: TransferHandle::Latent {
                    height: params.height,
                    width: params.width,
                    latent_units: params.latent_units,
                    step,
                    tensor: TensorTransfer {
                        shape: vec![
                            u64::from(params.latent_units),
                            pool.borrow(py).latent_width as u64,
                        ],
                        locations,
                    },
                },
            };
            pending
                .lock(py)?
                .set_products(vec![export])
                .map_err(|error| native_error(py, error))?;
        }
        self.retire_prefixes(py, batch, active)
    }

    fn retire_prefixes(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        active: &[usize],
    ) -> PyResult<()> {
        let outputs = batch.pending_outputs(py);
        for &index in active {
            let call = &batch.plan.calls[index];
            if call.code != CallKind::Media(MediaCall::Denoising) {
                continue;
            }
            let pending = outputs[index].borrow();
            let request = pending.request.borrow(py);
            let Some(image) = &request.request.admission().image else {
                continue;
            };
            let finished = pending
                .lock(py)?
                .latent_update
                .as_ref()
                .is_some_and(|update| !update.release && update.step >= i64::from(image.steps));
            if !finished {
                continue;
            }

            // Prefix forwards and their guidance branches share these bound
            // slots. Keep them through every export so a refusal can retry.
            let mut slots: Vec<_> = batch
                .plan
                .forward
                .call_indices
                .iter()
                .enumerate()
                .filter(|&(_, &owner)| owner as usize == index)
                .map(|(row, _)| batch.plan.forward.request_pool_indices[row])
                .filter(|&slot| slot as usize != request.request.slot())
                .collect();
            slots.sort_unstable();
            slots.dedup();
            drop(request);
            drop(pending);
            if !slots.is_empty() {
                let tables = self.tables.as_ref().ok_or_else(|| {
                    invalid(py, "flow prefix retirement lost its request page tables")
                })?;
                let copy = self
                    .worker
                    .bind(py)
                    .getattr("block_tables")?
                    .getattr("_clear_slots")?;
                tables
                    .borrow_mut(py)
                    .clear_prefixes(call.request_key, &slots, &copy)?;
            }
        }
        Ok(())
    }
}
