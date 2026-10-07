//! Discover numerical call sites and prepare their independently owned plans.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple};

use super::{ExecutionContext, workspace};
use crate::worker::host::with_context;
use crate::worker::tensor_buffers::TensorBuffers;

pub(super) fn prepare(
    owner: &Bound<'_, ExecutionContext>,
    size: &Bound<'_, PyAny>,
    constants: Option<&Bound<'_, TensorBuffers>>,
    workspace: Option<&Bound<'_, TensorBuffers>>,
) -> PyResult<()> {
    let py = owner.py();
    let text = size.is_instance(&py.import("uniserve.model.inputs")?.getattr("TextSize")?)?;
    let max_rows = if text {
        Some(size.getattr("num_tokens")?.extract::<usize>()?)
    } else {
        None
    };
    let (module, device, dtype, stream, groups) = {
        let mut context = owner.borrow_mut();
        let state = context.open_mut()?;
        state.max_tokens = max_rows;
        (
            state.module.clone_ref(py),
            state.device.clone_ref(py),
            state.dtype.clone_ref(py),
            state.stream.clone_ref(py),
            state.groups.clone_ref(py),
        )
    };

    // Communicators and registered windows belong to the stream. Preparing
    // another size reuses them without a new collective bootstrap.
    if !stream.is_none(py) {
        stream
            .bind(py)
            .getattr("communication")?
            .call_method1("bind", (groups,))?;
    }
    ExecutionContext::with_active(owner, || {
        for (name, query, supplied) in [
            ("constants", "constant_buffers", constants),
            ("workspace", "workspace_buffers", workspace),
        ] {
            let requirements = match module.bind(py).getattr_opt(query)? {
                Some(query) => query.call1((size,))?,
                None => PyDict::new(py).into_any(),
            };
            let views = match supplied {
                Some(buffers) => buffers.borrow().view(py, &requirements)?,
                None => workspace::allocate(owner, &requirements, device.bind(py))?,
            };
            let mut context = owner.borrow_mut();
            let state = context.open_mut()?;
            if name == "constants" {
                state.constants = views;
            } else {
                state.workspace = views;
            }
        }
        if let Some(prepare) = module.bind(py).getattr_opt("prepare_constants")? {
            let options = PyDict::new(py);
            options.set_item("out", owner.borrow().open()?.constants.bind(py))?;
            prepare.call((size,), Some(&options))?;
        }

        let representations = PyDict::new(py);
        representations.set_item("", (&device, &dtype))?;
        let representation = py
            .import("uniserve.runtime.execution")?
            .getattr("_layer_representation")?;
        let linear = py.import("uniserve.nn.linear")?;
        let plain = linear.getattr("Linear")?;
        let merged = linear.getattr("MergedColumnParallelLinear")?;
        let column = linear.getattr("ColumnParallelLinear")?;
        let moe = py.import("uniserve.nn.moe")?.getattr("FusedMoE")?;
        let vsa = py
            .import("uniserve.nn.attention.vsa")?
            .getattr("BlockAttention")?;
        let attention = py.import("uniserve.nn.attention")?.getattr("Attention")?;
        let mut vsa_slot = 0;

        for pair in module.bind(py).call_method0("named_modules")?.try_iter()? {
            let (path, child): (String, Bound<'_, PyAny>) = pair?.extract()?;
            let parent = path.rsplit_once('.').map_or("", |(parent, _)| parent);
            let inherited = representations.as_any().get_item(parent)?;
            let (device, dtype): (Bound<'_, PyAny>, Bound<'_, PyAny>) =
                representation.call1((&child, inherited))?.extract()?;
            representations.set_item(&path, (&device, &dtype))?;

            let is_merged = child.is_instance(&merged)?;
            if is_merged || child.is_instance(&plain)? {
                bind_matmul(owner, &child, &dtype, max_rows, is_merged)?;
            }
            if child.is_instance(&column)? {
                bind_gather(owner, &child, &device, max_rows)?;
            }
            if child.is_instance(&moe)? {
                bind_moe(owner, &child, &device, text.then_some(size))?;
            }
            if child.is_instance(&vsa)? {
                bind_vsa(owner, &child, vsa_slot % 2)?;
                vsa_slot += 1;
            }
            if child.is_instance(&attention)? {
                bind_attention(owner, &child, &device, &dtype, text.then_some(size))?;
            }
        }
        Ok(())
    })
}

fn bind_matmul(
    owner: &Bound<'_, ExecutionContext>,
    layer: &Bound<'_, PyAny>,
    dtype: &Bound<'_, PyAny>,
    rows: Option<usize>,
    merged: bool,
) -> PyResult<()> {
    let py = owner.py();
    let (backend, bindings) = {
        let context = owner.borrow();
        let state = context.open()?;
        (
            state.matmul_backend.clone_ref(py),
            if merged {
                &state.merged
            } else {
                &state.operators
            }
            .clone_ref(py),
        )
    };
    let binding = py
        .import("uniserve.runtime.bindings.matmul")?
        .call_method1(
            "MatmulBinding",
            (layer, backend, rows, owner.getattr("_matmul_workspace")?),
        )?;
    bindings
        .bind(py)
        .set_item(layer.as_ptr() as usize, &binding)?;
    let quantizers = PySet::empty(py)?;
    let quantizer = if merged {
        let mut weights = Vec::new();
        let mut quantizer = py.None().into_bound(py);
        for pair in layer
            .getattr("projections")?
            .call_method0("items")?
            .try_iter()?
        {
            let (name, branch): (Bound<'_, PyAny>, Bound<'_, PyAny>) = pair?.extract()?;
            weights.push((name, branch.getattr("weight")?.as_ptr() as usize));
            quantizer = branch.getattr("input_quantizer")?;
            quantizers.add(&quantizer)?;
        }
        bindings.bind(py).set_item(
            (PyTuple::new(py, weights)?, layer.getattr("branch_width")?),
            &binding,
        )?;
        quantizer
    } else {
        bindings
            .bind(py)
            .set_item(layer.getattr("weight")?.as_ptr() as usize, &binding)?;
        let quantizer = layer.getattr("input_quantizer")?;
        quantizers.add(&quantizer)?;
        quantizer
    };
    if let Some(rows) = rows.filter(|_| quantizers.len() == 1) {
        binding.call_method1("_prepare", (dtype, dtype, quantizer, rows))?;
    }
    Ok(())
}

fn bind_gather(
    owner: &Bound<'_, ExecutionContext>,
    layer: &Bound<'_, PyAny>,
    device: &Bound<'_, PyAny>,
    rows: Option<usize>,
) -> PyResult<()> {
    let py = owner.py();
    let distribution = layer.getattr("input_distribution")?;
    let axes = match layer.getattr_opt("gather_axes")? {
        Some(axes) => axes,
        None => distribution.call_method1("shard_axes", (0,))?,
    };
    let group = distribution
        .getattr("mesh")?
        .call_method1("get_group", (axes,))?;
    let members: usize = group.getattr("size")?.extract()?;
    if members == 1 {
        return Ok(());
    }

    // Two complete gather slots cover all chunks. FP32 capacity also fits
    // encoded values and their scales, independent of the selected kernel.
    let width: usize = layer
        .getattr("weight")?
        .getattr("shape")?
        .get_item(1)?
        .extract()?;
    let capacity = rows.map(|rows| 2 * rows.div_ceil(members) * members * width * 4);
    let (stream, pools, chunks) = {
        let context = owner.borrow();
        let state = context.open()?;
        (
            state.stream.clone_ref(py),
            state.gather_pools.clone_ref(py),
            state.chunks.clone_ref(py),
        )
    };
    let pool = if stream.is_none(py) {
        match pools.bind(py).get_item(&group)? {
            Some(pool) => pool,
            None => {
                let pool = py
                    .import("uniserve.runtime._collectives")?
                    .call_method1("GatherPool", (&group,))?;
                pools.bind(py).set_item(&group, &pool)?;
                pool
            }
        }
    } else {
        stream
            .bind(py)
            .getattr("communication")?
            .call_method1("gather_pool", (group,))?
    };
    let options = PyDict::new(py);
    options.set_item("capacity", capacity)?;
    let borrow = py
        .import("functools")?
        .getattr("partial")?
        .call((pool.getattr("borrow")?,), Some(&options))?;
    chunks.bind(py).set_item(layer.as_ptr() as usize, borrow)?;
    if let Some(capacity) = capacity.filter(|capacity| *capacity > 0) {
        with_context(&pool.call_method1("borrow", (capacity, device))?, || Ok(()))?;
    }
    Ok(())
}

fn bind_moe(
    owner: &Bound<'_, ExecutionContext>,
    layer: &Bound<'_, PyAny>,
    device: &Bound<'_, PyAny>,
    size: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let py = owner.py();
    let (backend, experts, weights, bindings) = {
        let context = owner.borrow();
        let state = context.open()?;
        (
            state.moe_backend.clone_ref(py),
            state.experts.clone_ref(py),
            state.weights.clone_ref(py),
            state.moe.clone_ref(py),
        )
    };
    let binding = py.import("uniserve.runtime.bindings.moe")?.call_method1(
        "MoEBinding",
        (
            layer,
            backend,
            size,
            device,
            owner.getattr("_moe_workspace")?,
            experts,
            weights,
        ),
    )?;
    bindings
        .bind(py)
        .set_item(layer.as_ptr() as usize, &binding)?;
    if let Some(size) = size {
        binding.call_method1("prepare", (size,))?;
    }
    Ok(())
}

fn bind_vsa(
    owner: &Bound<'_, ExecutionContext>,
    layer: &Bound<'_, PyAny>,
    slot: usize,
) -> PyResult<()> {
    let py = owner.py();
    let (backend, operators, bindings) = {
        let context = owner.borrow();
        let state = context.open()?;
        (
            state.vsa_backend.clone_ref(py),
            state.vsa_operators.clone_ref(py),
            state.vsa.clone_ref(py),
        )
    };
    // Alternate projection buffers so output consumption can overlap the
    // next layer's input preparation on the same stream.
    let buffers = py
        .import("functools")?
        .call_method1("partial", (owner.getattr("_vsa_buffers")?, slot))?;
    let binding = py.import("uniserve.runtime.bindings.vsa")?.call_method1(
        "VsaBinding",
        (
            backend,
            owner.getattr("scratch")?,
            owner.getattr("_attention_workspace")?,
            buffers,
            owner.getattr("_vsa_exchange")?,
            operators,
        ),
    )?;
    bindings.bind(py).set_item(layer.as_ptr() as usize, binding)
}

fn bind_attention(
    owner: &Bound<'_, ExecutionContext>,
    layer: &Bound<'_, PyAny>,
    device: &Bound<'_, PyAny>,
    dtype: &Bound<'_, PyAny>,
    size: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let py = owner.py();
    let (backend, cache, bindings, derive) = {
        let context = owner.borrow();
        let state = context.open()?;
        (
            state.attention_backend.clone_ref(py),
            state.cache.clone_ref(py),
            state.attention.clone_ref(py),
            state.derive_host_lengths,
        )
    };
    let name = layer.getattr("cache_name")?;
    let (state, table, device, dtype) = if !name.is_none() && !cache.is_none(py) {
        let state = cache.bind(py).call_method1("state", (&name,))?;
        let key = state.getattr("key")?;
        (
            state,
            cache.bind(py).call_method1("table", (name,))?,
            key.getattr("device")?,
            key.getattr("dtype")?,
        )
    } else {
        (
            py.None().into_bound(py),
            py.None().into_bound(py),
            device.clone(),
            dtype.clone(),
        )
    };
    let options = PyDict::new(py);
    options.set_item("derive_host_lengths", derive)?;
    let binding = py
        .import("uniserve.runtime.bindings.attention")?
        .getattr("AttentionBinding")?
        .call(
            (
                layer,
                backend,
                state,
                size,
                &device,
                &dtype,
                owner.getattr("_attention_workspace")?,
                owner.getattr("_context_transport")?,
                table,
            ),
            Some(&options),
        )?;
    bindings
        .bind(py)
        .set_item(layer.as_ptr() as usize, &binding)?;
    if let Some(size) = size {
        binding.call_method1("prepare", (&dtype, size))?;
        workspace::prepare_exchange(
            owner,
            layer,
            size.getattr("num_tokens")?.extract()?,
            &device,
            &dtype,
        )?;
    }
    Ok(())
}

pub(super) fn kernels(owner: &Bound<'_, ExecutionContext>) -> PyResult<Py<PyAny>> {
    let py = owner.py();
    let records = PyList::empty(py);
    let context = owner.borrow();
    let Some(state) = &context.state else {
        return Ok(records.into_any().unbind());
    };
    let merged = py
        .import("uniserve.nn.linear")?
        .getattr("MergedColumnParallelLinear")?;
    for pair in state
        .module
        .bind(py)
        .call_method0("named_modules")?
        .try_iter()?
    {
        let (path, child): (Bound<'_, PyAny>, Bound<'_, PyAny>) = pair?.extract()?;
        let matmul = if child.is_instance(&merged)? {
            &state.merged
        } else {
            &state.operators
        };
        for bindings in [matmul, &state.moe, &state.vsa, &state.attention] {
            if let Some(binding) = bindings.bind(py).get_item(child.as_ptr() as usize)? {
                for record in binding.call_method0("kernels")?.try_iter()? {
                    let record = record?;
                    let output = PyDict::new(py);
                    output.set_item("path", &path)?;
                    output.call_method1("update", (record,))?;
                    records.append(output)?;
                }
            }
        }
    }
    Ok(records.into_any().unbind())
}
