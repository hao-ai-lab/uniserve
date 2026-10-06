//! Native sampling controls with borrowed numerical tensors.

use indexmap::IndexMap;
use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker::{SamplingMetadata as NativeMetadata, SamplingPath};

use crate::calls::Call;
use crate::sampling::SamplingParams;

use super::error::{invalid, native_error, unsupported};
use super::host::with_context;
use super::pending::PendingOutput;

/// One sampled call's candidate logits, native controls and numerical views.
/// Speculative rows carry the penalties and constraints of each draft prefix.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct SamplingMetadata {
    pub(super) inner: NativeMetadata,
    // [rows, vocab], with one row per draft token and one bonus row for verify.
    #[pyo3(get)]
    logits: Py<PyAny>,
    #[pyo3(get)]
    parameters: Py<SamplingParams>,
    // One optional [vocab] generated-token count view per candidate row.
    #[pyo3(get)]
    penalty_counts: Py<PyTuple>,
    // [rows] uniforms and [rows, 3] (temperature, top_p, min_p), both absent
    // for device-greedy calls. They retain their numerical device backing.
    #[pyo3(get)]
    draws: Option<Py<PyAny>>,
    #[pyo3(get)]
    parameter_values: Option<Py<PyAny>>,
    // A flag or tagged token relay gates the call without a host observation.
    #[pyo3(get)]
    predicate: Option<Py<PyAny>>,
    #[pyo3(get)]
    tagged_predicate: bool,
    #[pyo3(get)]
    request_pool_index: Option<Py<PyAny>>,
    // Committed counts are only updated by the executor after selection.
    #[pyo3(get)]
    pub(super) penalty_base: Option<Py<PyAny>>,
}

#[pymethods]
impl SamplingMetadata {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        *,
        logits,
        parameters,
        penalty_counts,
        allowed,
        suppress,
        finish_token_ids,
        transition_token_ids,
        force_finish,
        draws,
        parameter_values,
        draft_token_ids=Vec::new(),
        terminal_draft_prefix=None,
        return_transition=false,
        predicate=None,
        tagged_predicate=false,
        request_pool_index=None,
        penalty_base=None,
    ))]
    fn new(
        logits: Py<PyAny>,
        parameters: Py<SamplingParams>,
        penalty_counts: Py<PyTuple>,
        allowed: Vec<Option<Vec<u32>>>,
        suppress: Vec<u32>,
        finish_token_ids: Vec<u32>,
        transition_token_ids: Vec<u32>,
        force_finish: bool,
        draws: Option<Py<PyAny>>,
        parameter_values: Option<Py<PyAny>>,
        draft_token_ids: Vec<u32>,
        terminal_draft_prefix: Option<usize>,
        return_transition: bool,
        predicate: Option<Py<PyAny>>,
        tagged_predicate: bool,
        request_pool_index: Option<Py<PyAny>>,
        penalty_base: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            inner: NativeMetadata {
                allowed,
                suppress,
                finish_token_ids,
                transition_token_ids,
                force_finish,
                draft_token_ids,
                terminal_draft_prefix,
                return_transition,
            },
            logits,
            parameters,
            penalty_counts,
            draws,
            parameter_values,
            predicate,
            tagged_predicate,
            request_pool_index,
            penalty_base,
        }
    }

    /// Prepare one call's host controls, then materialize its numerical inputs.
    /// Device penalty histories are borrowed or copied by the numerical backend.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (call, logits, request, *, positions, request_pool_index, decode_state, draft_token_ids=Vec::new()))]
    pub(super) fn for_call(
        py: Python<'_>,
        call: &Call,
        logits: &Bound<'_, PyAny>,
        request: &PendingOutput,
        positions: Vec<u64>,
        request_pool_index: Py<PyAny>,
        decode_state: &Bound<'_, PyAny>,
        draft_token_ids: Vec<u32>,
    ) -> PyResult<Self> {
        let admitted = request.request.borrow(py);
        let parameters = admitted.sampling(py)?;
        if parameters.is_none() {
            return Err(invalid(
                py,
                "sequence execution requires admitted sampling parameters",
            ));
        }
        let parameters = parameters.extract::<Py<SamplingParams>>()?;
        let finish = admitted
            .request
            .admission()
            .ar
            .as_ref()
            .map_or(&[][..], |ar| ar.finish_token_ids.as_slice());
        let (mut inner, uniforms) = NativeMetadata::for_call(
            &call.inner,
            &parameters.borrow(py).inner,
            finish,
            &positions,
            draft_token_ids,
        )
        .map_err(|error| native_error(py, error))?;
        inner.return_transition = request.transition_write.is_some();
        let slot = admitted.request.slot();
        drop(admitted);

        let sampled = request
            .token_update
            .sampled
            .as_ref()
            .map(|value| value.clone_ref(py));
        let tensors = py
            .import("uniserve_worker.sampling.metadata")?
            .call_method1(
                "prepare_inputs",
                (
                    logits,
                    &parameters,
                    slot,
                    decode_state,
                    sampled,
                    &inner.draft_token_ids,
                    uniforms,
                ),
            )?;
        let (rows, penalty_counts, draws, parameter_values, penalty_base) = tensors.extract()?;
        let (predicate, tagged_predicate) = match &request.predicate {
            Some(predicate) => {
                let predicate = predicate.bind(py);
                (
                    Some(predicate.get_item(0)?.unbind()),
                    predicate.get_item(1)?.extract()?,
                )
            }
            None => (None, false),
        };
        Ok(Self {
            inner,
            logits: rows,
            parameters,
            penalty_counts,
            draws,
            parameter_values,
            predicate,
            tagged_predicate,
            request_pool_index: Some(request_pool_index),
            penalty_base,
        })
    }

    #[getter]
    fn allowed<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            self.inner
                .allowed
                .iter()
                .map(|ids| ids.as_ref().map(|ids| PyTuple::new(py, ids)).transpose())
                .collect::<PyResult<Vec<_>>>()?,
        )
    }

    #[getter]
    fn suppress<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.suppress)
    }

    #[getter]
    fn finish_token_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.finish_token_ids)
    }

    #[getter]
    fn transition_token_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.transition_token_ids)
    }

    #[getter]
    fn force_finish(&self) -> bool {
        self.inner.force_finish
    }

    #[getter]
    fn draft_token_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.draft_token_ids)
    }

    #[getter]
    fn terminal_draft_prefix(&self) -> Option<usize> {
        self.inner.terminal_draft_prefix
    }

    #[getter]
    fn return_transition(&self) -> bool {
        self.inner.return_transition
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.logits)?;
        visit.call(&self.parameters)?;
        visit.call(&self.penalty_counts)?;
        visit.call(&self.draws)?;
        visit.call(&self.parameter_values)?;
        visit.call(&self.predicate)?;
        visit.call(&self.request_pool_index)?;
        visit.call(&self.penalty_base)
    }

    fn __clear__(&mut self, py: Python<'_>) {
        self.logits = py.None();
        self.penalty_counts = PyTuple::empty(py).unbind();
        self.draws = None;
        self.parameter_values = None;
        self.predicate = None;
        self.request_pool_index = None;
        self.penalty_base = None;
    }
}

/// Group homogeneous numerical selectors and restore their original row order.
/// The optional broadcast replaces the selection tensor in place on every
/// tensor-parallel rank. Results within each group share one output buffer.
/// Malformed row metadata raises a descriptor error before numerical dispatch.
#[pyfunction]
#[pyo3(signature = (tasks, *, selection_broadcast=None))]
pub(super) fn sample<'py>(
    py: Python<'py>,
    tasks: Vec<Py<SamplingMetadata>>,
    selection_broadcast: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyTuple>> {
    let mut groups = IndexMap::<(String, usize, SamplingPath), Vec<usize>>::new();
    for (index, task) in tasks.iter().enumerate() {
        let task = task.borrow(py);
        let (device, vocab, path) = validate(py, &task)?;
        groups.entry((device, vocab, path)).or_default().push(index);
    }

    let backend = py.import("uniserve_worker.sampling.sampler")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("selection_broadcast", selection_broadcast)?;
    let mut results: Vec<Option<Py<PyAny>>> = (0..tasks.len()).map(|_| None).collect();
    with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
        for ((_, _, path), indices) in groups {
            let group = PyTuple::new(py, indices.iter().map(|index| tasks[*index].bind(py)))?;
            let sampled = match path {
                SamplingPath::Greedy { suppressed } => {
                    let kwargs = kwargs.copy()?;
                    kwargs.set_item("apply_suppression", suppressed)?;
                    backend.call_method("sample_device_greedy_group", (group,), Some(&kwargs))?
                }
                SamplingPath::TopK(width) => backend.call_method(
                    "_sample_fused_top_k_group",
                    (group, width),
                    Some(&kwargs),
                )?,
                SamplingPath::Categorical => {
                    backend.call_method("_sample_task_group", (group,), Some(&kwargs))?
                }
            };
            for (row, index) in indices.into_iter().enumerate() {
                results[index] = Some(sampled.get_item(row)?.unbind());
            }
        }
        PyTuple::new(py, results.into_iter().flatten().collect::<Vec<_>>())
    })
}

/// Tensor metadata is host-resident; these shape checks never observe a device value.
fn validate(py: Python<'_>, task: &SamplingMetadata) -> PyResult<(String, usize, SamplingPath)> {
    let logits = task.logits.bind(py);
    let shape: Vec<usize> = logits.getattr("shape")?.extract()?;
    let rows = task.inner.allowed.len();
    if shape.len() != 2
        || shape[0] == 0
        || shape[1] == 0
        || !logits
            .call_method0("is_floating_point")?
            .extract::<bool>()?
        || shape[0] != rows
        || task.penalty_counts.bind(py).len() != rows
    {
        return Err(invalid(
            py,
            "sampling task logits must be shaped [rows, vocab]",
        ));
    }
    let device = logits.getattr("device")?;
    let parameters = task.parameters.borrow(py);
    if task.inner.device_greedy(&parameters.inner) {
        if task.draws.is_some() || task.parameter_values.is_some() || rows != 1 {
            return Err(invalid(py, "greedy sampling task has shaped metadata"));
        }
    } else {
        let aligned = |value: Option<&Py<PyAny>>, shape: &[usize]| -> PyResult<bool> {
            let Some(value) = value else { return Ok(false) };
            let value = value.bind(py);
            Ok(value.getattr("device")?.eq(&device)?
                && value.getattr("shape")?.extract::<Vec<usize>>()? == shape)
        };
        let floating = match &task.draws {
            Some(draws) => draws
                .bind(py)
                .call_method0("is_floating_point")?
                .extract::<bool>()?,
            None => false,
        };
        if !floating || !aligned(task.draws.as_ref(), &[rows])? {
            return Err(invalid(py, "sampling task draws must align with its rows"));
        }
        if !aligned(task.parameter_values.as_ref(), &[rows, 3])? {
            return Err(invalid(py, "sampling task parameter vectors do not align"));
        }
    }
    if rows != task.inner.draft_token_ids.len() + 1 {
        return Err(invalid(
            py,
            "sampling rows must cover the draft chain and bonus token",
        ));
    }
    let vocab = shape[1];
    if vocab > uniserve_worker::TOKEN_VALUE_MASK as usize {
        return Err(unsupported(
            py,
            "vocabulary exceeds the device token decision range",
        ));
    }
    if task
        .inner
        .draft_token_ids
        .iter()
        .any(|token| *token as usize >= vocab)
    {
        return Err(invalid(
            py,
            "speculative draft token is outside the model vocabulary",
        ));
    }
    let cuda = device.getattr("type")?.extract::<String>()? == "cuda";
    Ok((
        device.str()?.to_string(),
        vocab,
        task.inner.path(&parameters.inner, vocab, cuda),
    ))
}

/// Accept captured greedy selections only when every call can use them.
/// Richer per-call controls return None so the caller samples retained logits.
#[pyfunction]
#[pyo3(signature = (calls, requests, tasks, output, *, sampling_group, request_pool_indices))]
pub(super) fn sample_graph<'py>(
    py: Python<'py>,
    calls: Vec<Py<Call>>,
    requests: Vec<Py<PendingOutput>>,
    tasks: &Bound<'py, PyTuple>,
    output: &Bound<'py, PyAny>,
    sampling_group: &Bound<'py, PyAny>,
    request_pool_indices: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    use uniserve_worker_ipc::{CallKind, ForwardMode};

    let count = calls.len();
    if output.is_none() || count == 0 || requests.len() != count || tasks.len() != count {
        return Ok(py.None().into_bound(py));
    }
    if request_pool_indices
        .call_method0("numel")?
        .extract::<usize>()?
        != count
    {
        return Ok(py.None().into_bound(py));
    }
    for name in [
        "tokens",
        "valid",
        "active",
        "finish",
        "continuation",
        "tagged_tokens",
    ] {
        let value = output.getattr(name)?;
        if value.is_none() || value.call_method0("numel")?.extract::<usize>()? != count {
            return Ok(py.None().into_bound(py));
        }
    }
    let backend = py.import("uniserve_worker.sampling.sampler")?;
    let columns: usize = backend.getattr("SAMPLING_COMPLETION_FIELDS")?.extract()?;
    if output
        .getattr("completion")?
        .call_method0("numel")?
        .extract::<usize>()?
        != columns * count
    {
        return Ok(py.None().into_bound(py));
    }

    let row_type = py
        .import("uniserve_worker.model_executor.input_batch")?
        .getattr("TokenRow")?;
    let mut finishes = Vec::with_capacity(count);
    let mut forced = Vec::with_capacity(count);
    for ((call, request), task) in calls.iter().zip(&requests).zip(tasks) {
        let call = call.borrow(py);
        let request = request.borrow(py);
        let admitted = request.request.borrow(py);
        let Some(ar) = &admitted.request.admission().ar else {
            return Ok(py.None().into_bound(py));
        };
        let state = call.inner.sampling_state.as_ref();
        let force_finish = state.is_some_and(|state| state.force_finish);
        if call.inner.code != CallKind::Forward(ForwardMode::Decode)
            || !ar.sampling.device_greedy()
            || ar.sampling.allowed_token_ids.is_some()
            || !ar.sampling.forced_token_ids.is_empty()
            || state.is_some_and(|state| {
                state.allowed_token_ids.is_some()
                    || !state.suppressed_token_ids.is_empty()
                    || !state.transition_token_ids.is_empty()
            })
            || request.transition_write.is_some()
            || !task.is_instance(&row_type)?
            || task.getattr("decode_predicate")?.is_none()
            || !task.getattr("decode_predicate_tagged")?.extract::<bool>()?
            || task.getattr("decode_force_finish")?.extract::<bool>()? != force_finish
            || request.token_write.is_none()
        {
            return Ok(py.None().into_bound(py));
        }
        finishes.push(PyTuple::new(
            py,
            uniserve_worker::finish_token_ids(&call.inner, &ar.finish_token_ids),
        )?);
        forced.push(force_finish);
    }
    backend.call_method1(
        "finish_graph",
        (
            output,
            PyTuple::new(py, finishes)?,
            PyTuple::new(py, forced)?,
            sampling_group,
            request_pool_indices,
        ),
    )
}
