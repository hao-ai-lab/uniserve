//! Read immutable identities at the Python numerical backend boundary.

use pyo3::prelude::*;
use uniserve_core::{CallId, RequestId};
use uniserve_worker_ipc::{BufferId, RequestKey};

pub(crate) fn request_key(value: &Bound<'_, PyAny>) -> PyResult<RequestKey> {
    Ok(RequestKey::new(
        value.getattr("engine_id")?.extract()?,
        RequestId(value.getattr("request_id")?.extract()?),
        value.getattr("request_epoch")?.extract()?,
    ))
}

pub(crate) fn call_id(value: &Bound<'_, PyAny>) -> PyResult<CallId> {
    Ok(CallId::new(
        value.getattr("batch_id")?.extract()?,
        value.getattr("request_index")?.extract()?,
    ))
}

pub(crate) fn buffer_id(value: &Bound<'_, PyAny>) -> PyResult<BufferId> {
    Ok(BufferId {
        owner: request_key(&value.getattr("owner")?)?,
        producer_call_id: call_id(&value.getattr("producer_call_id")?)?,
        output_index: value.getattr("output_index")?.extract()?,
        generation: value.getattr("generation")?.extract()?,
    })
}
