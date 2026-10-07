//! Scoped delivery for ordinary modules placed on another device.

use indexmap::IndexMap;
use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet};
use uniserve_worker::cuda::{DeviceGuard, Event, Stream};

use crate::worker::execution::close_all;

/// Placement and streams belong to the execution context. Hooks borrow them
/// only during an activation, so shared modules keep independent invocations.
#[pyclass]
pub(super) struct DeviceTransfers {
    modules: IndexMap<usize, (Py<PyAny>, Py<PyAny>)>,
    // (origin device, origin stream, destination device) -> numerical view.
    streams: IndexMap<(i32, usize, i32), Py<PyAny>>,
    owned: Vec<Stream>,
}

impl DeviceTransfers {
    pub(super) fn new(module: &Bound<'_, PyAny>, device: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = module.py();
        let options = PyDict::new(py);
        options.set_item("remove_duplicate", false)?;
        let children = module.call_method("named_modules", (), Some(&options))?;
        let mut inherited = IndexMap::from([(String::new(), device.clone().unbind())]);
        let mut modules = IndexMap::new();

        for child in children.try_iter()? {
            let (path, child): (String, Bound<'_, PyAny>) = child?.extract()?;
            let parent = inherited[path.rsplit_once('.').map_or("", |(parent, _)| parent)].bind(py);
            let mut placements = devices(&child, "parameters")?;
            if placements.is_empty() {
                placements = devices(&child, "buffers")?;
            }
            let mut candidates = placements.iter();
            let target = match (candidates.next(), candidates.next()) {
                (Some(device), None) => device,
                _ => parent.clone(),
            };
            if !path.is_empty() && !target.eq(parent)? {
                modules.insert(
                    child.as_ptr() as usize,
                    (child.unbind(), target.clone().unbind()),
                );
            }
            inherited.insert(path, target.unbind());
        }

        Ok(Self {
            modules,
            streams: IndexMap::new(),
            owned: Vec::new(),
        })
    }

    pub(super) fn scope(owner: &Bound<'_, Self>) -> Option<TransferScope> {
        (!owner.borrow().modules.is_empty()).then(|| TransferScope {
            owner: owner.clone().unbind(),
            token: None,
            hooks: Vec::new(),
            calls: Vec::new(),
        })
    }

    pub(super) fn streams(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.streams
            .values()
            .map(|stream| stream.clone_ref(py))
            .collect()
    }

    pub(super) fn reset(&mut self) {
        // Graphs and readers retire before preparation changes. Borrowed origin
        // views do not own a stream and must never synchronize during capture.
        self.streams.clear();
        self.owned.clear();
    }

    pub(super) fn close(&mut self) {
        self.reset();
        self.modules.clear();
    }

    fn stream(
        &mut self,
        origin: &Bound<'_, PyAny>,
        device: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = origin.py();
        let source = origin
            .getattr("device")?
            .getattr("index")?
            .extract::<i32>()?;
        let target = device.getattr("index")?.extract::<i32>()?;
        let handle = origin.getattr("cuda_stream")?.extract::<usize>()?;
        let route = (source, handle, target);
        if let Some(stream) = self.streams.get(&route) {
            return Ok(stream.clone_ref(py));
        }
        if source == target {
            self.streams.insert(route, origin.clone().unbind());
            return Ok(origin.clone().unbind());
        }

        // Capture can use another origin stream than eager warmup. Its target
        // keeps the same serialized delivery stream and allocation domain.
        let stream = self
            .streams
            .iter()
            .find(|((_, _, destination), _)| *destination == target)
            .map(|(_, stream)| stream.clone_ref(py));
        let stream = match stream {
            Some(stream) => stream,
            None => {
                let cuda = py.import("torch.cuda")?;
                if cuda
                    .call_method0("is_current_stream_capturing")?
                    .is_truthy()?
                {
                    return Err(PyRuntimeError::new_err(
                        "warm cross-device numerical calls before capture",
                    ));
                }
                let owner = py
                    .detach(|| Stream::new(target))
                    .map_err(PyRuntimeError::new_err)?;
                let stream = cuda
                    .call_method1("ExternalStream", (owner.handle(), device))?
                    .unbind();
                self.owned.push(owner);
                stream
            }
        };
        let destination = stream.getattr(py, "cuda_stream")?.extract::<usize>(py)?;
        self.streams.insert(route, stream.clone_ref(py));
        self.streams
            .insert((target, destination, source), origin.clone().unbind());
        Ok(stream)
    }
}

#[pymethods]
impl DeviceTransfers {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for (module, device) in self.modules.values() {
            visit.call(module)?;
            visit.call(device)?;
        }
        for stream in self.streams.values() {
            visit.call(stream)?;
        }
        Ok(())
    }
}

fn devices<'py>(module: &Bound<'py, PyAny>, kind: &str) -> PyResult<Bound<'py, PySet>> {
    let devices = PySet::empty(module.py())?;
    for value in module.call_method0(kind)?.try_iter()? {
        let value = value?;
        if !value.getattr("is_meta")?.is_truthy()? {
            devices.add(value.getattr("device")?)?;
        }
    }
    Ok(devices)
}

/// One synchronous module invocation. Frames nest independently for each
/// activation; Python ContextVar selection isolates hooks on shared modules.
struct Call {
    module: usize,
    target: Py<PyAny>,
    out: Py<PyAny>,
    origin: Option<Py<PyAny>>,
    stream: Option<Py<PyAny>>,
    scopes: Vec<Py<PyAny>>,
}

impl Call {
    fn close(self, py: Python<'_>) -> PyResult<()> {
        let mut releases = Vec::new();
        if let (Some(origin), Some(stream)) = (&self.origin, &self.stream) {
            releases.push(order(stream.bind(py), origin.bind(py)));
        }
        for scope in self.scopes.into_iter().rev() {
            releases.push(
                scope
                    .call_method1(py, "__exit__", (py.None(), py.None(), py.None()))
                    .map(drop),
            );
        }
        close_all(py, releases)
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.target)?;
        visit.call(&self.out)?;
        visit.call(&self.origin)?;
        visit.call(&self.stream)?;
        for scope in &self.scopes {
            visit.call(scope)?;
        }
        Ok(())
    }
}

#[pyclass]
pub(super) struct TransferScope {
    owner: Py<DeviceTransfers>,
    token: Option<Py<PyAny>>,
    hooks: Vec<Py<PyAny>>,
    calls: Vec<Call>,
}

impl TransferScope {
    fn call_mut(&mut self) -> PyResult<&mut Call> {
        self.calls
            .last_mut()
            .ok_or_else(|| PyRuntimeError::new_err("device transfer call is not active"))
    }

    fn active(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(variable(slf.py())?.call_method0("get")?.is(slf))
    }

    fn enter(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let token = variable(py)?.call_method1("set", (slf,))?.unbind();
        let modules = {
            let mut this = slf.borrow_mut();
            this.token = Some(token);
            this.owner
                .borrow(py)
                .modules
                .values()
                .map(|(module, _)| module.clone_ref(py))
                .collect::<Vec<_>>()
        };
        let result: PyResult<()> = (|| {
            let prepare = slf.getattr("prepare")?;
            let finish = slf.getattr("finish")?;
            for module in modules {
                let options = PyDict::new(py);
                options.set_item("prepend", true)?;
                options.set_item("with_kwargs", true)?;
                let hook = module.bind(py).call_method(
                    "register_forward_pre_hook",
                    (&prepare,),
                    Some(&options),
                )?;
                slf.borrow_mut().hooks.push(hook.unbind());

                options.del_item("prepend")?;
                options.set_item("always_call", true)?;
                let hook = module.bind(py).call_method(
                    "register_forward_hook",
                    (&finish,),
                    Some(&options),
                )?;
                slf.borrow_mut().hooks.push(hook.unbind());
            }
            Ok(())
        })();
        if let Err(error) = result {
            if let Err(cleanup) = Self::exit(slf) {
                let _ = error.value(py).call_method1(
                    "add_note",
                    (format!("transfer hook cleanup failed: {cleanup}"),),
                );
            }
            return Err(error);
        }
        Ok(())
    }

    fn exit(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let (calls, hooks, token) = {
            let mut this = slf.borrow_mut();
            (
                std::mem::take(&mut this.calls),
                std::mem::take(&mut this.hooks),
                this.token.take(),
            )
        };
        let mut releases = calls
            .into_iter()
            .rev()
            .map(|call| call.close(py))
            .collect::<Vec<_>>();
        for hook in hooks.into_iter().rev() {
            releases.push(hook.call_method0(py, "remove").map(drop));
        }
        if let Some(token) = token {
            releases.push(variable(py)?.call_method1("reset", (token,)).map(drop));
        }
        close_all(py, releases)
    }

    fn enter_scope(slf: &Bound<'_, Self>, scope: Bound<'_, PyAny>) -> PyResult<()> {
        scope.call_method0("__enter__")?;
        slf.borrow_mut().call_mut()?.scopes.push(scope.unbind());
        Ok(())
    }
}

#[pymethods]
impl TransferScope {
    fn __enter__(slf: &Bound<'_, Self>) -> PyResult<()> {
        Self::enter(slf)
    }

    fn __exit__(
        slf: &Bound<'_, Self>,
        _kind: &Bound<'_, PyAny>,
        _error: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        Self::exit(slf)
    }

    fn prepare(
        slf: &Bound<'_, Self>,
        module: &Bound<'_, PyAny>,
        args: &Bound<'_, PyAny>,
        kwargs: &Bound<'_, PyDict>,
    ) -> PyResult<Option<Py<PyAny>>> {
        if !Self::active(slf)? {
            return Ok(None);
        }
        let py = slf.py();
        let id = module.as_ptr() as usize;
        let owner = slf.borrow().owner.clone_ref(py);
        let device = owner
            .borrow(py)
            .modules
            .get(&id)
            .ok_or_else(|| PyRuntimeError::new_err("module device binding is closed"))?
            .1
            .clone_ref(py);
        let out = kwargs
            .get_item("out")?
            .map(Bound::unbind)
            .unwrap_or_else(|| py.None());
        slf.borrow_mut().calls.push(Call {
            module: id,
            target: py.None(),
            out,
            origin: None,
            stream: None,
            scopes: Vec::new(),
        });

        if device.bind(py).getattr("type")?.extract::<String>()? == "cuda" {
            let cuda = py.import("torch.cuda")?;
            let origin = cuda.call_method0("current_stream")?;
            let stream = owner.borrow_mut(py).stream(&origin, device.bind(py))?;
            {
                let mut this = slf.borrow_mut();
                let call = this.call_mut()?;
                call.origin = Some(origin.clone().unbind());
                call.stream = Some(stream.clone_ref(py));
            }
            order(&origin, stream.bind(py))?;
            Self::enter_scope(slf, cuda.call_method1("device", (&device,))?)?;
            Self::enter_scope(slf, cuda.call_method1("stream", (stream,))?)?;
        }

        // Numerical traversal returns the origin while moving the inputs, so
        // discovery does not reconstruct every dataclass and container twice.
        let (inputs, target): (Py<PyAny>, Py<PyAny>) = py
            .import("uniserve.runtime._transfers")?
            .call_method1("_copy_inputs", (args, kwargs, device))?
            .extract()?;
        slf.borrow_mut().call_mut()?.target = target;
        Ok(Some(inputs))
    }

    fn finish(
        slf: &Bound<'_, Self>,
        module: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyAny>,
        _kwargs: &Bound<'_, PyAny>,
        result: Py<PyAny>,
    ) -> PyResult<Option<Py<PyAny>>> {
        if !Self::active(slf)? {
            return Ok(None);
        }
        let py = slf.py();
        let call = {
            let mut this = slf.borrow_mut();
            let Some(call) = this.calls.pop() else {
                return Ok(None);
            };
            if call.module != module.as_ptr() as usize {
                this.calls.push(call);
                return Ok(None);
            }
            call
        };
        let copied = if call.target.is_none(py) || result.is_none(py) {
            Ok(result)
        } else {
            py.import("uniserve.runtime._transfers")?
                .call_method1("_copy_outputs", (result, &call.target, &call.out))
                .map(Bound::unbind)
        };
        let closed = call.close(py);
        match copied {
            Ok(result) => {
                closed?;
                Ok(Some(result))
            }
            Err(error) => {
                if let Err(cleanup) = closed {
                    let _ = error.value(py).call_method1(
                        "add_note",
                        (format!("device transfer cleanup failed: {cleanup}"),),
                    );
                }
                Err(error)
            }
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.owner)?;
        visit.call(&self.token)?;
        for hook in &self.hooks {
            visit.call(hook)?;
        }
        for call in &self.calls {
            call.traverse(&visit)?;
        }
        Ok(())
    }
}

fn variable(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    py.import("uniserve.runtime._transfers")?.getattr("_ACTIVE")
}

/// Use the common CUDA event implementation for cross-device fork/join.
/// Default streams need their device selected separately for record and wait.
fn order(producer: &Bound<'_, PyAny>, consumer: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = producer.py();
    if producer.eq(consumer)? {
        return Ok(());
    }
    let source = producer
        .getattr("device")?
        .getattr("index")?
        .extract::<i32>()?;
    let target = consumer
        .getattr("device")?
        .getattr("index")?
        .extract::<i32>()?;
    let producer = producer.getattr("cuda_stream")?.extract::<usize>()?;
    let consumer = consumer.getattr("cuda_stream")?.extract::<usize>()?;
    py.detach(|| {
        let event = Event::new(source, false, false);
        {
            let _device = DeviceGuard::new(source)?;
            event.record(producer)?;
        }
        let _device = DeviceGuard::new(target)?;
        event.wait_on(consumer)
    })
    .map_err(PyRuntimeError::new_err)
}
