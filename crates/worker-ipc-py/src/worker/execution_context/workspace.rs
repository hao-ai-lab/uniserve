//! Call-site backing and stream-ordered communication workspaces.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyString};

use super::ExecutionContext;
use crate::worker::tensor_buffers::{TensorBuffers, mapping};

pub(super) fn allocate(
    owner: &Bound<'_, ExecutionContext>,
    requirements: &Bound<'_, PyAny>,
    device: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let py = owner.py();
    let allocation = Py::new(
        py,
        TensorBuffers::allocate(py, requirements, device, false, None)?,
    )?;
    let views = allocation.borrow(py).view(py, requirements)?;
    owner.borrow_mut().open_mut()?.allocations.push(allocation);
    Ok(views)
}

pub(super) fn attention(
    owner: &Bound<'_, ExecutionContext>,
    requirements: &Bound<'_, PyAny>,
    device: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let py = owner.py();
    let requirements = mapping(requirements)?;
    let persistent = PyDict::new(py);
    for (name, config) in requirements.iter() {
        if name.extract::<String>()? != "scratch" {
            persistent.set_item(name, config)?;
        }
    }
    let views = mapping(allocate(owner, &persistent, device)?.bind(py))?;
    if let Some(scratch) = requirements.get_item("scratch")? {
        let fields = PyDict::new(py);
        fields.set_item("scratch", scratch)?;
        let shared = owner.borrow().scratch(
            py,
            &PyString::new(py, "attention").into_any(),
            &fields,
            device,
        )?;
        views.set_item("scratch", shared.bind(py).get_item("scratch")?)?;
    }
    Ok(views.into_any().unbind())
}

pub(super) fn prepare_exchange(
    owner: &Bound<'_, ExecutionContext>,
    layer: &Bound<'_, PyAny>,
    tokens: usize,
    device: &Bound<'_, PyAny>,
    dtype: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let py = owner.py();
    let size: usize = layer
        .getattr("exchange")?
        .getattr("group")?
        .getattr("size")?
        .extract()?;
    if size == 1 {
        return Ok(());
    }
    let rows = tokens.div_ceil(size) * size;
    let heads: usize = layer.getattr("local_heads")?.extract()?;
    let kv_heads: usize = layer.getattr("local_kv_heads")?.extract()?;
    let head_dim: usize = layer.getattr("head_dim")?.extract()?;
    let itemsize: usize = dtype.getattr("itemsize")?.extract()?;
    let widths = [
        ("heads_send", heads + 2 * kv_heads),
        ("heads_receive", heads + 2 * kv_heads),
        ("output_receive", heads),
    ];
    let exchange = owner.borrow().open()?.exchange.clone_ref(py);
    if let Some(previous) = exchange.bind(py).get_item(layer.as_ptr() as usize)? {
        let tensors = previous.getattr("tensors")?;
        let mut fits = true;
        for (name, heads) in widths {
            fits &= tensors
                .get_item(name)?
                .call_method0("numel")?
                .extract::<usize>()?
                >= rows * heads * head_dim * itemsize;
        }
        if fits {
            return Ok(());
        }
    }

    // Flat byte fields carry Q/K/V heads out and local query heads back.
    let config = py.import("uniserve.tensors")?.getattr("BufferConfig")?;
    let dtype = py.import("torch")?.getattr("uint8")?;
    let requirements = PyDict::new(py);
    for (name, heads) in widths {
        requirements.set_item(
            name,
            config.call1(((rows * heads * head_dim * itemsize,), &dtype))?,
        )?;
    }
    let views = allocate(owner, &requirements, device)?;
    let buffers = py
        .import("uniserve.runtime.bindings.attention")?
        .call_method1("ExchangeBuffers", (views,))?;
    exchange.bind(py).set_item(layer.as_ptr() as usize, buffers)
}

pub(super) fn context_transport(
    owner: &Bound<'_, ExecutionContext>,
    layer: &Bound<'_, PyAny>,
    mut rows: usize,
    dtype: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let py = owner.py();
    let parallel = layer.getattr("context_parallel")?;
    let group = parallel.getattr("context_group")?;
    let context_size: usize = group.getattr("size")?.extract()?;
    let exchange_size: usize = layer
        .getattr("exchange")?
        .getattr("group")?
        .getattr("size")?
        .extract()?;
    let (max_tokens, backing, contexts) = {
        let context = owner.borrow();
        let state = context.open()?;
        (
            state.max_tokens,
            state.context_backing.clone_ref(py),
            state.vsa_context.clone_ref(py),
        )
    };
    if let Some(tokens) = max_tokens {
        rows = rows.max(tokens.div_ceil(context_size * exchange_size) * exchange_size);
    }
    prepare_exchange(
        owner,
        layer,
        rows * context_size,
        &group.getattr("device")?,
        dtype,
    )?;
    if context_size == 1 || rows == 0 {
        return Ok(py.None());
    }
    let heads = layer.getattr("local_kv_heads")?;
    let head_dim = layer.getattr("head_dim")?;
    let key = (
        parallel.getattr("key_group")?,
        rows,
        &heads,
        &head_dim,
        dtype,
    );
    let buffers = match backing.bind(py).get_item(&key)? {
        Some(buffers) => buffers,
        None => {
            let options = PyDict::new(py);
            options.set_item("rows", rows)?;
            options.set_item("heads", &heads)?;
            options.set_item("head_dim", &head_dim)?;
            options.set_item("dtype", dtype)?;
            options.set_item("block_size", 1)?;
            let buffers = py
                .import("uniserve.runtime.attention_storage")?
                .call_method("allocate_context_storage", ((&parallel,),), Some(&options))?
                .get_item(&parallel)?;
            backing.bind(py).set_item(&key, &buffers)?;
            buffers
        }
    };
    contexts.bind(py).set_item(parallel, &buffers)?;
    Ok(buffers.unbind())
}

pub(super) fn vsa_exchange(
    owner: &Bound<'_, ExecutionContext>,
    layer: &Bound<'_, PyAny>,
    rows: usize,
    heads: usize,
    head_dim: usize,
    dtype: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let py = owner.py();
    let Some(parallel) = layer
        .getattr_opt("parallel")?
        .filter(|value| !value.is_none())
    else {
        return Ok(py.None());
    };
    let (transport, outputs, contexts, stream, device) = {
        let context = owner.borrow();
        let state = context.open()?;
        (
            state.vsa_transport.clone_ref(py),
            state.vsa_output.clone_ref(py),
            state.vsa_context.clone_ref(py),
            state.stream.clone_ref(py),
            state.device.clone_ref(py),
        )
    };
    let ulysses = parallel.getattr("ulysses_group")?;
    let context_group = parallel.getattr("context_group")?;
    let groups = (&ulysses, &context_group, parallel.getattr("key_group")?).into_pyobject(py)?;
    let dimensions = (heads, head_dim, dtype).into_pyobject(py)?;
    let mut selected = None;
    for (key, _) in transport.bind(py).iter() {
        if key.get_item(PySlice::new(py, 0, 3, 1))?.eq(&groups)?
            && key.get_item(3)?.extract::<usize>()? >= rows
            && key.get_item(PySlice::new(py, 4, 7, 1))?.eq(&dimensions)?
        {
            selected = Some(key);
            break;
        }
    }
    let key = match selected {
        Some(key) => key,
        None => (
            &ulysses,
            &context_group,
            parallel.getattr("key_group")?,
            rows,
            heads,
            head_dim,
            dtype,
        )
            .into_pyobject(py)?
            .into_any(),
    };
    if !transport.bind(py).contains(&key)? {
        if py
            .import("uniserve.runtime.bindings")?
            .call_method1("capturing", (&device,))?
            .is_truthy()?
        {
            return Err(PyRuntimeError::new_err(
                "prepare VSA communication storage before capture",
            ));
        }
        let options = PyDict::new(py);
        options.set_item("rows", rows)?;
        options.set_item("heads", heads)?;
        options.set_item("head_dim", head_dim)?;
        options.set_item("dtype", dtype)?;
        let stream = stream.bind(py);
        let registered = !stream.is_none()
            && ulysses.getattr("size")?.extract::<usize>()? > 1
            && context_group.getattr("size")?.extract::<usize>()? > 1
            && stream
                .getattr("communication")?
                .getattr("communicators")?
                .contains(ulysses.call_method0("_require")?.getattr("group_name")?)?;

        let storage = py.import("uniserve.runtime.attention_storage")?;
        let output = if registered {
            // The stream's communicator owns registered symmetric windows;
            // independent context buffers use the ordinary collective path.
            let callback = py.import("functools")?.getattr("partial")?.call(
                (
                    py.import("uniserve.runtime.execution")?
                        .getattr("_registered_output")?,
                    &parallel,
                ),
                Some(&options),
            )?;
            stream
                .getattr("communication")?
                .call_method1("windows", (("vsa_output", &key), &ulysses, callback))?
        } else {
            let allocated =
                storage.call_method("allocate_output_storage", ((&parallel,),), Some(&options))?;
            owner.borrow_mut().open_mut()?.allocations.extend(
                allocated
                    .getattr("allocations")?
                    .extract::<Vec<Py<TensorBuffers>>>()?,
            );
            allocated.getattr("views")?.get_item(&parallel)?
        };
        options.set_item("block_size", 64)?;
        let context = storage
            .call_method("allocate_context_storage", ((&parallel,),), Some(&options))?
            .call_method1("get", (&parallel,))?;
        transport.bind(py).set_item(&key, (output, context))?;
    }
    let (output, context): (Py<PyAny>, Py<PyAny>) =
        transport.bind(py).as_any().get_item(key)?.extract()?;
    outputs.bind(py).set_item(&parallel, output)?;
    if !context.is_none(py) {
        contexts.bind(py).set_item(&parallel, context)?;
    }
    Ok(parallel.unbind())
}
