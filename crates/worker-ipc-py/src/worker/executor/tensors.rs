//! Move resident tensor values without entering the numerical dispatcher.

use pyo3::prelude::*;
use uniserve_worker_ipc::{CallKind, MediaCall};

use super::{BatchState, PythonBackend};
use crate::worker::error::{invalid, native_error};
use crate::worker::storage::Buffer;

impl PythonBackend {
    pub(super) fn execute_tensors(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        active: &[usize],
    ) -> PyResult<()> {
        let outputs = batch.pending_outputs(py);
        let transports = self.worker.bind(py).getattr("export_transports")?;
        let store = self.tensors.get();

        for &index in active {
            let call = &batch.plan.calls[index];
            if call.latent_input.is_some() {
                if call.latent_output.is_none() {
                    return Err(invalid(
                        py,
                        "product transfer changes the physical product kind",
                    ));
                }
                continue;
            }
            let mut products = call.tensor_outputs();
            let product = match (
                call.tensor_inputs().count(),
                products.next(),
                products.next(),
            ) {
                (1, Some(product), None) => product,
                _ => {
                    return Err(invalid(
                        py,
                        "product transfer requires one physical input and one output",
                    ));
                }
            };
            let pending = outputs[index].borrow();
            let numerical = pending.call.bind(py);
            let device = self
                .model_runner
                .borrow(py)
                .call_devices(py, &*numerical.extract::<PyRef<crate::calls::Call>>()?)?
                .into_bound(py)
                .get_item(0)?;
            let read = batch.numerical.borrow(py).consume_tensor(
                py,
                call.request_key.request_id.0,
                store,
                device,
            )?;
            let read = read.borrow(py);
            let metadata = read.metadata.as_ref().map(|metadata| metadata.bind(py));

            let write = pending
                .writes
                .bind(py)
                .iter()
                .map(|write| write.cast_into::<Buffer>())
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .find(|write| write.get().id(py) == product.buffer_id())
                .ok_or_else(|| invalid(py, "tensor transfer has no bound output"))?;
            // Feature writes may convert model output into pool dtype. A
            // transfer must instead preserve its declared representation.
            let dtype = read.tensor.bind(py).getattr("dtype")?.str()?;
            if dtype.to_str()?.trim_start_matches("torch.") != product.dtype.storage_name() {
                return Err(invalid(
                    py,
                    "product transfer changes its declared representation",
                ));
            }
            store.write(
                py,
                write.clone(),
                read.tensor.bind(py).clone(),
                None,
                metadata.cloned(),
            )?;
            let export = store.export_buffer(
                py,
                call,
                product,
                &write,
                &transports,
                false,
                None,
                &pending,
            )?;
            pending
                .lock(py)?
                .set_products(vec![export])
                .map_err(|error| native_error(py, error))?;
        }
        Ok(())
    }

    pub(super) fn export_features(
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
        if !remote {
            return Ok(());
        }
        let outputs = batch.pending_outputs(py);
        for &index in active {
            let call = &batch.plan.calls[index];
            if !matches!(
                call.code,
                CallKind::Media(MediaCall::VisionEncoding | MediaCall::LatentEncoding)
            ) || self.info.endpoint.rank as usize != self.output_rank(py, &call.component)?
            {
                continue;
            }
            let Some(product) = &call.encoder_output else {
                continue;
            };
            let pending = outputs[index].borrow();
            for write in pending.writes.bind(py) {
                let write = write.cast_into::<Buffer>()?;
                if write.get().id(py) != product.buffer_id() || !write.get().producer_recorded(py) {
                    continue;
                }
                let export = self.tensors.get().export_buffer(
                    py,
                    call,
                    product,
                    &write,
                    &transports,
                    false,
                    None,
                    &pending,
                )?;
                pending
                    .lock(py)?
                    .set_products(vec![export])
                    .map_err(|error| native_error(py, error))?;
            }
        }
        Ok(())
    }
}
