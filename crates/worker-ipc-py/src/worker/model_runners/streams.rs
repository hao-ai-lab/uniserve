//! Execution lanes, their forks and the input-copy stream of one executor.

use std::collections::HashSet;
use std::sync::{Mutex, MutexGuard, PoisonError};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker_ipc::CallKind;

use super::modules::input_error;
use crate::worker::execution::close_all;

struct Lane {
    config: Py<PyAny>,
    device: i32,
    kinds: Option<HashSet<CallKind>>,
    stream: Py<PyAny>,
}

#[derive(Default)]
struct State {
    initialized: bool,
    lanes: Vec<Lane>,
    microbatches: Vec<Py<PyAny>>,
    modules: Vec<Py<PyAny>>,
    preparation: Option<Py<PyAny>>,
}

pub(super) struct Streams {
    event_slots: usize,
    state: Mutex<State>,
}

impl Default for Streams {
    fn default() -> Self {
        Self::new(2)
    }
}

impl Streams {
    pub(super) fn new(event_slots: usize) -> Self {
        Self {
            event_slots,
            state: Mutex::new(State::default()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub(super) fn initialize(
        &self,
        owner: &Bound<'_, PyAny>,
        event_slots: Option<usize>,
    ) -> PyResult<()> {
        if self.lock().initialized {
            return Ok(());
        }

        let py = owner.py();
        let config = owner.getattr("worker_config")?;
        let device = py
            .import("uniserve.runtime.device")?
            .call_method1("canonical_device", (config.getattr("device")?,))?;
        let others = owner.call_method1("_capture_devices", (&device,))?;
        let lanes = config.getattr("lanes")?;
        let event_slots = event_slots.unwrap_or(self.event_slots);
        let runtime = py.import("uniserve.runtime")?;

        if lanes.is_truthy()? {
            if others.is_truthy()? {
                return Err(PyValueError::new_err(
                    "Green Context lanes require one physical device",
                ));
            }
            let mut configs = Vec::new();
            let mut kinds = Vec::new();
            let mut budgets = Vec::new();
            let mut slots = Vec::new();
            for lane in lanes.try_iter()? {
                let lane = lane?;
                let inflight: Option<usize> = lane.getattr("max_inflight")?.extract()?;
                slots.push(
                    inflight
                        .filter(|value| *value != 0)
                        .unwrap_or(event_slots - 1)
                        + 1,
                );
                budgets.push(lane.getattr("sm_budget")?.extract::<usize>()?);
                kinds.push(
                    lane.getattr("call_kinds")?
                        .try_iter()?
                        .map(|kind| Ok(pythonize::depythonize(&kind?)?))
                        .collect::<PyResult<HashSet<CallKind>>>()?,
                );
                configs.push(lane.unbind());
            }
            let options = PyDict::new(py);
            options.set_item("event_slots", PyTuple::new(py, slots)?)?;
            let streams = runtime
                .getattr("partition_streams")?
                .call((&device, PyTuple::new(py, budgets)?), Some(&options))?;
            let device = device.getattr("index")?.extract()?;
            for ((config, kinds), stream) in configs.into_iter().zip(kinds).zip(streams.try_iter()?)
            {
                self.lock().lanes.push(Lane {
                    config,
                    device,
                    kinds: Some(kinds),
                    stream: stream?.unbind(),
                });
            }
        } else {
            let mut devices = vec![device];
            devices.extend(others.try_iter()?.collect::<PyResult<Vec<_>>>()?);
            for device in devices {
                if let Some(index) = cuda_index(&device)? {
                    let stream = external(&device, event_slots)?;
                    self.lock().lanes.push(Lane {
                        config: py.None(),
                        device: index,
                        kinds: None,
                        stream: stream.unbind(),
                    });
                }
            }
        }

        self.lock().initialized = true;
        Ok(())
    }

    pub(super) fn module<'py>(
        &self,
        owner: &Bound<'py, PyAny>,
        device: &Bound<'py, PyAny>,
        kinds: &HashSet<CallKind>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = owner.py();
        self.initialize(owner, None)?;
        let device_index = cuda_index(device)?;
        let parents: Vec<_> = self
            .lock()
            .lanes
            .iter()
            .filter(|lane| {
                Some(lane.device) == device_index
                    && lane
                        .kinds
                        .as_ref()
                        .is_none_or(|covered| !covered.is_disjoint(kinds))
            })
            .map(|lane| lane.stream.clone_ref(py))
            .collect();

        let stream = match parents.as_slice() {
            [parent] => parent.bind(py).call_method0("fork")?,
            [] if !owner
                .getattr("worker_config")?
                .getattr("lanes")?
                .is_truthy()? =>
            {
                external(device, self.event_slots)?
            }
            [] => {
                return Err(input_error(
                    py,
                    "numerical module has no initialized execution partition",
                ));
            }
            _ => {
                return Err(input_error(
                    py,
                    "one numerical module requires an unambiguous execution partition",
                ));
            }
        };
        // Weights and backing are initialized on the caller's stream. Later
        // invocations order only their actual input and output dependencies.
        let ready = py
            .import("torch.cuda")?
            .call_method1("current_stream", (device,))?;
        if let Err(error) = stream.call_method1("wait", (ready,)) {
            close_all(py, [Err(error), stream.call_method0("close").map(drop)])?;
        }
        self.lock().modules.push(stream.clone().unbind());
        Ok(stream)
    }

    pub(super) fn lane_count(&self) -> usize {
        self.lock().lanes.len()
    }

    pub(super) fn batch_streams<'py>(
        &self,
        device: &Bound<'py, PyAny>,
        microbatches: bool,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let py = device.py();
        let index = cuda_index(device)?;
        let streams = {
            let state = self.lock();
            let mut streams: Vec<_> = state
                .lanes
                .iter()
                .filter(|lane| Some(lane.device) == index)
                .map(|lane| (lane.config.clone_ref(py), lane.stream.clone_ref(py), 0))
                .collect();
            if microbatches && let Some((lane, _, _)) = streams.first() {
                let lane = lane.clone_ref(py);
                streams.extend(
                    state
                        .microbatches
                        .iter()
                        .enumerate()
                        .map(|(index, stream)| {
                            (lane.clone_ref(py), stream.clone_ref(py), index + 1)
                        }),
                );
            }
            streams
        };
        PyTuple::new(py, streams)
    }

    pub(super) fn fork_microbatch(&self, py: Python<'_>) -> PyResult<()> {
        let parent = self
            .lock()
            .lanes
            .first()
            .map(|lane| lane.stream.clone_ref(py))
            .ok_or_else(|| PyRuntimeError::new_err("microbatch execution requires a CUDA lane"))?;
        let stream = parent.bind(py).call_method0("fork")?;
        self.lock().microbatches.push(stream.unbind());
        Ok(())
    }

    pub(super) fn expert_streams<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let streams: Vec<_> = {
            let state = self.lock();
            state
                .lanes
                .first()
                .map(|lane| &lane.stream)
                .into_iter()
                .chain(&state.microbatches)
                .map(|stream| stream.clone_ref(py))
                .collect()
        };
        PyTuple::new(py, streams)
    }

    pub(super) fn preparation<'py>(
        &self,
        device: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = device.py();
        if let Some(stream) = self.lock().preparation.as_ref() {
            return Ok(stream.bind(py).clone());
        }
        let stream = external(device, self.event_slots)?;
        self.lock().preparation = Some(stream.clone().unbind());
        Ok(stream)
    }

    /// Dependency order: consumers close before the lanes they borrow. Take
    /// references under the lock, but never call Python while holding it.
    pub(super) fn owned(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        let state = self.lock();
        state
            .preparation
            .iter()
            .chain(state.modules.iter().rev())
            .chain(state.microbatches.iter().rev())
            .chain(state.lanes.iter().rev().map(|lane| &lane.stream))
            .map(|stream| stream.clone_ref(py))
            .collect()
    }

    pub(super) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        let state = self.lock();
        for lane in &state.lanes {
            visit.call(&lane.config)?;
            visit.call(&lane.stream)?;
        }
        for stream in state
            .modules
            .iter()
            .chain(&state.microbatches)
            .chain(&state.preparation)
        {
            visit.call(stream)?;
        }
        Ok(())
    }

    pub(super) fn clear(&self) {
        let state = std::mem::take(&mut *self.lock());
        drop(state);
    }
}

pub(super) fn cuda_index(device: &Bound<'_, PyAny>) -> PyResult<Option<i32>> {
    if device.getattr("type")?.extract::<String>()? == "cuda" {
        device.getattr("index")?.extract().map(Some)
    } else {
        Ok(None)
    }
}

fn external<'py>(device: &Bound<'py, PyAny>, event_slots: usize) -> PyResult<Bound<'py, PyAny>> {
    let py = device.py();
    let options = PyDict::new(py);
    options.set_item("device", device)?;
    let stream = py
        .import("torch.cuda")?
        .getattr("Stream")?
        .call((), Some(&options))?;
    let options = PyDict::new(py);
    options.set_item("event_slots", event_slots)?;
    py.import("uniserve.runtime")?
        .getattr("CUDAStream")?
        .call_method("external", (stream,), Some(&options))
}
