//! Export numerical writes through the same storage and transport owners.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use uniserve_worker_ipc::{
    Call, CallKind, FeatureKind, MediaCall, TensorExport, TensorRef, TensorTransfer, TransferHandle,
};

use super::{Buffer, TensorStore, lock, numerical, value_view};
use crate::convert;
use crate::worker::error::invalid;
use crate::worker::exports;
use crate::worker::fetch;
use crate::worker::pending::PendingOutput;

impl TensorStore {
    /// Export a completed write and retain each export before converting its
    /// location. The pending output owns revocation until storage commit.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::worker) fn export_buffer<'py>(
        &self,
        py: Python<'py>,
        call: &Call,
        product: &TensorRef,
        write: &Bound<'py, Buffer>,
        transports: &Bound<'py, PyAny>,
        host: bool,
        regions: Option<&[Bound<'py, PyTuple>]>,
        output: &PendingOutput,
    ) -> PyResult<TensorExport> {
        let (feature, logical_shape, region, metadata, tensor, value_shape, extent) = {
            let buffer = lock(py, &write.get().inner);
            let views = lock(py, &write.get().views);
            (
                buffer.feature,
                buffer
                    .logical_shape
                    .iter()
                    .map(|&dim| dim as u64)
                    .collect::<Vec<_>>(),
                views.region.as_ref().map(|region| region.clone_ref(py)),
                views
                    .metadata
                    .as_ref()
                    .map(|metadata| metadata.clone_ref(py)),
                views.tensor.clone_ref(py),
                buffer.value_shape.clone(),
                buffer.extent,
            )
        };
        let value = value_view(py, tensor.bind(py), &value_shape, extent)?;

        let mut height = 0;
        let mut width = 0;
        let mut value_range = String::new();
        if let Some(metadata) = metadata.as_ref().map(|metadata| metadata.bind(py)) {
            height = metadata.getattr("height")?.extract()?;
            width = metadata.getattr("width")?.extract()?;
            let types = numerical(py)?;
            if metadata.is_instance(&types.getattr("ImageMetadata")?)? {
                let range: Option<(f64, f64)> = metadata.getattr("value_range")?.extract()?;
                if let Some(range) = range {
                    value_range = if range == (-1.0, 1.0) {
                        "signed_unit"
                    } else {
                        "unit"
                    }
                    .into();
                }
            }
        }
        let source_kind = if feature {
            if call.code == CallKind::Media(MediaCall::VisionEncoding)
                || !call.vision_inputs.is_empty()
            {
                Some(FeatureKind::Vision)
            } else if call.code == CallKind::Media(MediaCall::LatentEncoding)
                || call.latent_feature_input.is_some()
            {
                Some(FeatureKind::Latent)
            } else {
                return Err(invalid(py, "encoder transfer has no feature source"));
            }
        } else {
            if height == 0 && !value_range.is_empty() {
                return Err(invalid(py, "non-image tensor carries an image range"));
            }
            None
        };

        // Storage write already checked the physical shape and metadata.
        // A shard exports its logical shape and the bound region's offset.
        let value_shape: Vec<_> = value_shape.into_iter().map(|dim| dim as u64).collect();
        let (shape, origin) = if let Some(region) = &region {
            let origin = region
                .bind(py)
                .try_iter()?
                .map(|axis| axis?.getattr("start")?.extract::<u64>())
                .collect::<PyResult<Vec<_>>>()?;
            (logical_shape, origin)
        } else {
            (value_shape.clone(), vec![0; value_shape.len()])
        };
        if !product.shape_bound.contains_shape(&shape) {
            return Err(invalid(
                py,
                "product transfer changes its declared representation",
            ));
        }

        let mut locations = Vec::new();
        let mut export_view = |view: Bound<'py, PyAny>, offset: Vec<u64>| -> PyResult<()> {
            let offset = PyTuple::new(py, offset)?;
            let accepted = exports::export(
                transports,
                &view,
                Some(offset.as_any()),
                &call.consumer_slots,
                host,
                |retirement| self.retain_export(py, write, retirement),
            )?;
            // Record accepted locations before parsing them, so any later
            // view or descriptor failure can revoke the entire call's output.
            output
                .exported_locators
                .bind(py)
                .call_method1("extend", (&accepted,))?;
            locations.extend(accepted.iter());
            Ok(())
        };
        if let Some(regions) = regions {
            for region in regions {
                let bounds = fetch::requested_region(py, Some(region), &value_shape)?;
                let offset = origin
                    .iter()
                    .zip(bounds)
                    .map(|(&base, axis)| base + axis.start)
                    .collect();
                export_view(value.get_item(region)?, offset)?;
            }
        } else {
            export_view(value, origin)?;
        }

        let registrations = locations
            .iter()
            .map(|location| Ok((transports.get_item(location.getattr("backend")?)?, location)))
            .collect::<PyResult<Vec<_>>>()?;
        output.tensor_exports.bind(py).set_item(
            convert::buffer_id_to_py(py, &product.buffer_id())?,
            PyTuple::new(py, registrations)?,
        )?;
        let locations = locations
            .iter()
            .map(|location| {
                convert::transfer_locator_from_py(&location.call_method0("to_mapping")?)
                    .ok_or_else(|| invalid(py, "transport returned an invalid tensor location"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let tensor = TensorTransfer { shape, locations };
        let value = match source_kind {
            Some(payload_kind) => TransferHandle::Encoder {
                height,
                width,
                payload_kind,
                tensor,
            },
            None => TransferHandle::DeviceProduct {
                height,
                width,
                value_range,
                tensor,
            },
        };
        Ok(TensorExport {
            product: product.clone(),
            value,
        })
    }
}
