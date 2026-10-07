//! Tensor arguments and process-group bootstrap for the native communicator.

use std::sync::{Arc, Mutex, MutexGuard};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyBytes, PyDict, PyList};
use uniserve_worker::CUDAStream as NativeStream;
use uniserve_worker::cuda::Event;
use uniserve_worker::nccl::NcclCommunicator as NativeCommunicator;

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct NcclCommunicator {
    inner: Mutex<NativeCommunicator<Py<PyAny>>>,
    device: Py<PyAny>,
    transfer: Py<PyAny>,
    rank: usize,
    ranks: Vec<i32>,
}

impl NcclCommunicator {
    pub(super) fn new(
        py: Python<'_>,
        group: &Bound<'_, PyAny>,
        stream: &Bound<'_, PyAny>,
        owner: &Arc<Mutex<NativeStream>>,
    ) -> PyResult<Self> {
        py.import("uniserve.runtime._collectives")?
            .call_method0("_forbid_implicit_registration")?;
        let dist = py.import("torch.distributed")?;
        let rank: usize = dist.call_method1("get_rank", (group,))?.extract()?;
        let ranks: Vec<i32> = dist
            .call_method1("get_process_group_ranks", (group,))?
            .extract()?;
        let id = if rank == 0 {
            let bytes =
                NativeCommunicator::<Py<PyAny>>::unique_id().map_err(PyRuntimeError::new_err)?;
            PyBytes::new(py, &bytes).into_any()
        } else {
            py.None().into_bound(py)
        };
        let values = PyList::new(py, [id])?;

        let options = PyDict::new(py);
        options.set_item("src", ranks[0])?;
        options.set_item("group", group)?;
        dist.call_method("broadcast_object_list", (&values,), Some(&options))?;
        let bytes: Vec<u8> = values.get_item(0)?.extract()?;
        let id = bytes
            .try_into()
            .map_err(|_| PyValueError::new_err("NCCL unique ID must contain 128 bytes"))?;

        let native = owner
            .lock_py_attached(py)
            .map_err(|_| PyRuntimeError::new_err("CUDA stream lock is poisoned"))?;
        let native = &*native;
        let inner = py
            .detach(|| NativeCommunicator::new(native, ranks.len() as i32, rank as i32, id))
            .map_err(PyRuntimeError::new_err)?;

        let device = stream.getattr("device")?;
        let transfer = py.import("torch.cuda")?.call_method1(
            "ExternalStream",
            (
                inner.transfer_stream().map_err(PyRuntimeError::new_err)?,
                &device,
            ),
        )?;

        Ok(Self {
            inner: Mutex::new(inner),
            device: device.unbind(),
            transfer: transfer.unbind(),
            rank,
            ranks,
        })
    }

    fn lock(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, NativeCommunicator<Py<PyAny>>>> {
        self.inner
            .lock_py_attached(py)
            .map_err(|_| PyRuntimeError::new_err("NCCL communicator lock is poisoned"))
    }

    fn peer(&self, rank: i32) -> PyResult<i32> {
        self.ranks
            .iter()
            .position(|candidate| *candidate == rank)
            .map(|index| index as i32)
            .ok_or_else(|| PyValueError::new_err("collective peer is outside the process group"))
    }

    fn validate(
        &self,
        py: Python<'_>,
        value: &Bound<'_, PyAny>,
        output: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        if !value.getattr("device")?.eq(self.device.bind(py))?
            || !value.call_method0("is_contiguous")?.is_truthy()?
        {
            return Err(PyValueError::new_err(
                "computation collectives require contiguous tensors on their device",
            ));
        }
        if let Some(output) = output
            && (!output.getattr("device")?.eq(self.device.bind(py))?
                || !output.getattr("dtype")?.eq(value.getattr("dtype")?)?
                || !output.call_method0("is_contiguous")?.is_truthy()?)
        {
            return Err(PyValueError::new_err(
                "collective output must match input dtype, device, and layout",
            ));
        }
        Ok(())
    }

    fn gather(
        slf: &Bound<'_, Self>,
        output: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
        asynchronous: bool,
    ) -> PyResult<Option<CollectiveWork>> {
        let py = slf.py();
        let owner = slf.get();
        owner.validate(py, value, Some(output))?;
        let count = count(value)?;
        if self::count(output)? != count * owner.ranks.len() {
            return Err(PyValueError::new_err(
                "collective gather output must hold every rank's contribution",
            ));
        }

        let (input, output, dtype) = (pointer(value)?, pointer(output)?, dtype(value)?);

        let mut inner = owner.lock(py)?;
        let inner = &mut *inner;
        let completed = py
            .detach(|| unsafe { inner.all_gather(output, input, count, dtype, asynchronous) })
            .map_err(PyRuntimeError::new_err)?;
        Ok(completed.map(|event| CollectiveWork {
            event,
            owner: slf.clone().unbind(),
        }))
    }

    fn exchange(
        slf: &Bound<'_, Self>,
        output: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
        asynchronous: bool,
    ) -> PyResult<Option<CollectiveWork>> {
        let py = slf.py();
        let owner = slf.get();
        owner.validate(py, value, Some(output))?;
        let count = count(value)?;
        if self::count(output)? != count || count % owner.ranks.len() != 0 {
            return Err(PyValueError::new_err(
                "asynchronous exchange requires equal peer payloads",
            ));
        }

        let (input, output, dtype) = (pointer(value)?, pointer(output)?, dtype(value)?);
        let count = count / owner.ranks.len();

        let mut inner = owner.lock(py)?;
        let inner = &mut *inner;
        let completed = py
            .detach(|| unsafe { inner.all_to_all(output, input, count, dtype, asynchronous) })
            .map_err(PyRuntimeError::new_err)?;
        Ok(completed.map(|event| CollectiveWork {
            event,
            owner: slf.clone().unbind(),
        }))
    }
}

#[pymethods]
impl NcclCommunicator {
    #[getter]
    fn transfer_stream(&self, py: Python<'_>) -> Py<PyAny> {
        self.transfer.clone_ref(py)
    }

    #[pyo3(signature = (value, op="sum"))]
    fn all_reduce(&self, py: Python<'_>, value: &Bound<'_, PyAny>, op: &str) -> PyResult<()> {
        self.validate(py, value, None)?;

        let op = match op {
            "sum" => 0,
            "max" => 2,
            "min" => 3,
            _ => return Err(PyValueError::new_err("unsupported NCCL reduction")),
        };

        let (data, count, dtype) = (pointer(value)?, count(value)?, dtype(value)?);

        let mut inner = self.lock(py)?;
        let inner = &mut *inner;
        py.detach(|| unsafe { inner.all_reduce(data, count, dtype, op) })
            .map_err(PyRuntimeError::new_err)
    }

    fn all_gather(
        slf: &Bound<'_, Self>,
        output: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        Self::gather(slf, output, value, false).map(drop)
    }

    fn start_all_gather(
        slf: &Bound<'_, Self>,
        output: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Option<CollectiveWork>> {
        Self::gather(slf, output, value, true)
    }

    fn start_all_to_all(
        slf: &Bound<'_, Self>,
        output: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Option<CollectiveWork>> {
        Self::exchange(slf, output, value, true)
    }

    fn all_to_all(
        slf: &Bound<'_, Self>,
        output: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
        output_splits: Vec<usize>,
        input_splits: Vec<usize>,
    ) -> PyResult<()> {
        let py = slf.py();
        let owner = slf.get();
        owner.validate(py, value, Some(output))?;
        if input_splits.len() != owner.ranks.len() || output_splits.len() != owner.ranks.len() {
            return Err(PyValueError::new_err(
                "collective exchange requires one split per rank",
            ));
        }
        if input_splits
            .iter()
            .chain(&output_splits)
            .all(|size| *size == input_splits[0])
        {
            return Self::exchange(slf, output, value, false).map(drop);
        }

        let inputs = spans(value, &input_splits)?;
        let outputs = spans(output, &output_splits)?;
        let dtype = dtype(value)?;

        let mut inner = owner.lock(py)?;
        let inner = &mut *inner;
        py.detach(|| {
            inner.grouped(|inner| {
                for (peer, ((input, inputs), (output, outputs))) in
                    inputs.into_iter().zip(outputs).enumerate()
                {
                    if inputs != 0 {
                        unsafe {
                            inner.send(input, inputs, dtype, peer as i32)?;
                        }
                    }
                    if outputs != 0 {
                        unsafe {
                            inner.recv(output, outputs, dtype, peer as i32)?;
                        }
                    }
                }
                Ok(())
            })
        })
        .map_err(PyRuntimeError::new_err)
    }

    #[pyo3(name = "gather")]
    fn gather_to_root(
        &self,
        py: Python<'_>,
        outputs: Option<Vec<Bound<'_, PyAny>>>,
        value: &Bound<'_, PyAny>,
        root: i32,
    ) -> PyResult<()> {
        self.validate(py, value, None)?;
        let root = self.peer(root)?;
        let (input, count, dtype) = (pointer(value)?, count(value)?, dtype(value)?);
        let destinations = if root as usize == self.rank {
            let outputs = outputs
                .filter(|outputs| outputs.len() == self.ranks.len())
                .ok_or_else(|| {
                    PyValueError::new_err("collective gather requires one destination per rank")
                })?;
            outputs
                .iter()
                .map(|output| {
                    self.validate(py, value, Some(output))?;
                    if self::count(output)? != count {
                        return Err(PyValueError::new_err(
                            "gather destination must match the contribution size",
                        ));
                    }
                    pointer(output)
                })
                .collect::<PyResult<Vec<_>>>()?
        } else {
            Vec::new()
        };

        let mut inner = self.lock(py)?;
        let inner = &mut *inner;
        py.detach(|| {
            inner.grouped(|inner| unsafe {
                inner.send(input, count, dtype, root)?;
                for (peer, output) in destinations.into_iter().enumerate() {
                    inner.recv(output, count, dtype, peer as i32)?;
                }
                Ok(())
            })
        })
        .map_err(PyRuntimeError::new_err)
    }

    fn broadcast(&self, py: Python<'_>, value: &Bound<'_, PyAny>, root: i32) -> PyResult<()> {
        self.validate(py, value, None)?;
        let root = self.peer(root)?;

        let (data, count, dtype) = (pointer(value)?, count(value)?, dtype(value)?);

        let mut inner = self.lock(py)?;
        let inner = &mut *inner;
        py.detach(|| unsafe { inner.broadcast(data, count, dtype, root) })
            .map_err(PyRuntimeError::new_err)
    }

    fn reduce_scatter(
        &self,
        py: Python<'_>,
        output: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.validate(py, value, Some(output))?;
        let count = count(output)?;
        if self::count(value)? != count * self.ranks.len() {
            return Err(PyValueError::new_err(
                "collective reduction requires one output-sized partition per rank",
            ));
        }

        let (input, output, dtype) = (pointer(value)?, pointer(output)?, dtype(value)?);

        let mut inner = self.lock(py)?;
        let inner = &mut *inner;
        py.detach(|| unsafe { inner.reduce_scatter(output, input, count, dtype) })
            .map_err(PyRuntimeError::new_err)
    }

    fn send(&self, py: Python<'_>, value: &Bound<'_, PyAny>, peer: i32) -> PyResult<()> {
        self.validate(py, value, None)?;
        let peer = self.peer(peer)?;

        let (data, count, dtype) = (pointer(value)?, count(value)?, dtype(value)?);

        let mut inner = self.lock(py)?;
        let inner = &mut *inner;
        py.detach(|| unsafe { inner.send(data, count, dtype, peer) })
            .map_err(PyRuntimeError::new_err)
    }

    fn recv(&self, py: Python<'_>, value: &Bound<'_, PyAny>, peer: i32) -> PyResult<()> {
        self.validate(py, value, None)?;
        let peer = self.peer(peer)?;

        let (data, count, dtype) = (pointer(value)?, count(value)?, dtype(value)?);

        let mut inner = self.lock(py)?;
        let inner = &mut *inner;
        py.detach(|| unsafe { inner.recv(data, count, dtype, peer) })
            .map_err(PyRuntimeError::new_err)
    }

    fn send_recv(
        &self,
        py: Python<'_>,
        output: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
        dst: i32,
        src: i32,
    ) -> PyResult<()> {
        // Paired sends use bytes so the logical communicator may exchange
        // different numerical shapes and dtypes in either direction.
        self.validate(py, value, None)?;
        self.validate(py, output, None)?;
        let (dst, src) = (self.peer(dst)?, self.peer(src)?);
        let (input, output_ptr) = (pointer(value)?, pointer(output)?);
        let input_bytes = count(value)? * value.call_method0("element_size")?.extract::<usize>()?;
        let output_bytes =
            count(output)? * output.call_method0("element_size")?.extract::<usize>()?;

        let mut inner = self.lock(py)?;
        let inner = &mut *inner;
        py.detach(|| {
            inner.grouped(|inner| unsafe {
                inner.send(input, input_bytes, 1, dst)?;
                inner.recv(output_ptr, output_bytes, 1, src)
            })
        })
        .map_err(PyRuntimeError::new_err)
    }

    #[pyo3(signature = (*buffers))]
    pub(super) fn register_buffers(
        &self,
        py: Python<'_>,
        buffers: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        if py
            .import("torch.cuda")?
            .call_method0("is_current_stream_capturing")?
            .is_truthy()?
        {
            return Err(PyRuntimeError::new_err(
                "register communication buffers before CUDA graph capture",
            ));
        }
        for value in buffers {
            self.validate(py, &value, None)?;
            let data = pointer(&value)?;
            let bytes = count(&value)? * value.call_method0("element_size")?.extract::<usize>()?;

            let mut inner = self.lock(py)?;
            let inner = &mut *inner;
            let backing = value.unbind();
            py.detach(|| unsafe { inner.register(data, bytes, backing) })
                .map_err(PyRuntimeError::new_err)?;
        }
        Ok(())
    }

    pub(super) fn close(&self, py: Python<'_>) -> PyResult<()> {
        let mut inner = self.lock(py)?;
        let inner = &mut *inner;
        py.detach(|| inner.close(false))
            .map_err(PyRuntimeError::new_err)
    }

    pub(super) fn abort(&self, py: Python<'_>) -> PyResult<()> {
        self.lock(py)?.close(true).map_err(PyRuntimeError::new_err)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.device)?;
        visit.call(&self.transfer)?;
        // Native calls can release the GIL while holding the communicator.
        // Their live method reference keeps the object reachable during GC.
        if let Ok(inner) = self.inner.try_lock() {
            for backing in inner.backings() {
                visit.call(backing)?;
            }
        }
        Ok(())
    }
}

/// Contiguous row partitions need byte offsets, not temporary tensor views.
fn spans(value: &Bound<'_, PyAny>, splits: &[usize]) -> PyResult<Vec<(usize, usize)>> {
    let rows = value.call_method1("size", (0,))?.extract::<usize>()?;
    if splits.iter().sum::<usize>() != rows {
        return Err(PyValueError::new_err(
            "collective exchange splits must cover every row",
        ));
    }

    let width = count(value)?.checked_div(rows).unwrap_or(0);
    let element_size = value.call_method0("element_size")?.extract::<usize>()?;
    let mut data = pointer(value)?;

    Ok(splits
        .iter()
        .map(|rows| {
            let count = rows * width;
            let span = (data, count);
            data += count * element_size;
            span
        })
        .collect())
}

#[pyclass(frozen)]
pub(super) struct CollectiveWork {
    event: Arc<Event>,
    owner: Py<NcclCommunicator>,
}

#[pymethods]
impl CollectiveWork {
    fn block_current_stream(&self, py: Python<'_>) -> PyResult<()> {
        let owner = self.owner.get();
        let stream = py
            .import("torch.cuda")?
            .call_method1("current_stream", (&owner.device,))?;
        let handle = stream.getattr("cuda_stream")?.extract()?;

        let mut inner = owner.lock(py)?;
        let inner = &mut *inner;
        py.detach(|| inner.join(&self.event, handle))
            .map_err(PyRuntimeError::new_err)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.owner)
    }
}

fn pointer(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    value.call_method0("data_ptr")?.extract()
}

fn count(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    value.call_method0("numel")?.extract()
}

fn dtype(value: &Bound<'_, PyAny>) -> PyResult<i32> {
    match value.getattr("dtype")?.str()?.to_str()? {
        "torch.int8" => Ok(0),
        "torch.uint8" | "torch.bool" => Ok(1),
        "torch.int32" => Ok(2),
        "torch.int64" => Ok(4),
        "torch.float16" => Ok(6),
        "torch.float32" => Ok(7),
        "torch.float64" => Ok(8),
        "torch.bfloat16" => Ok(9),
        dtype => Err(PyValueError::new_err(format!(
            "unsupported NCCL tensor dtype {dtype}"
        ))),
    }
}
