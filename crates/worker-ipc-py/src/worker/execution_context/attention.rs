//! Attention metadata binding and empty expert participation.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::ExecutionContext;

pub(super) fn bind(
    owner: &Bound<'_, ExecutionContext>,
    batch: &Bound<'_, PyAny>,
    replay: bool,
) -> PyResult<()> {
    let py = owner.py();
    let (bindings, derive) = {
        let context = owner.borrow();
        let state = context.open()?;
        (state.attention.clone_ref(py), state.derive_host_lengths)
    };
    ExecutionContext::with_active(owner, || {
        let entries = batch.getattr("entries")?;
        let mut readers = Vec::new();
        for (_, binding) in bindings.bind(py).iter() {
            let table = binding.getattr("table")?;
            if if table.is_none() {
                entries.len()? == 1
            } else {
                entries.contains(&table)?
            } {
                readers.push(binding);
            }
        }
        if replay {
            let mut plans = false;
            for reader in &readers {
                plans |= reader.getattr("builds_launch_plan")?.is_truthy()?;
            }
            if !plans {
                // Device-driven launches share one numerical input per table.
                // Host-planned launches bind every reading layer independently.
                let first = PyDict::new(py);
                for reader in &readers {
                    let table = reader.getattr("table")?;
                    if !first.contains(&table)? {
                        first.set_item(table, reader)?;
                    }
                }
                readers = first.values().iter().collect();
            }
        }
        let mut lengths = false;
        for reader in &readers {
            if reader
                .call_method1(
                    "reads_host_lengths",
                    (batch.call_method1("entry", (reader.getattr("table")?,))?,),
                )?
                .is_truthy()?
            {
                lengths = true;
                break;
            }
        }
        let mirrored = if lengths {
            let options = PyDict::new(py);
            options.set_item("derive", derive)?;
            py.import("uniserve.runtime.backends.attention._sequences")?
                .call_method("batch_host_lengths", (batch,), Some(&options))?
        } else {
            batch.clone()
        };
        let options = PyDict::new(py);
        options.set_item("source", batch)?;
        for reader in readers {
            reader.call_method("bind", (&mirrored,), Some(&options))?;
        }
        Ok(())
    })
}

pub(super) fn join(owner: &Bound<'_, ExecutionContext>) -> PyResult<()> {
    let py = owner.py();
    let (exchange, bindings, dtype) = {
        let context = owner.borrow();
        let state = context.open()?;
        (
            state.experts.clone_ref(py),
            state.moe.clone_ref(py),
            state.dtype.clone_ref(py),
        )
    };
    let exchange = exchange.bind(py);
    if exchange.is_none() || !exchange.getattr("capacity")?.is_truthy()? {
        return Ok(());
    }
    let modules = PyList::empty(py);
    for (module, binding) in bindings.bind(py).iter() {
        if binding.getattr("exchange")?.is(exchange) {
            modules.append(module)?;
        }
    }
    for module in exchange
        .call_method1("pending_layers", (modules,))?
        .try_iter()?
    {
        let binding = bindings.bind(py).as_any().get_item(module?)?;
        binding.call_method1(
            "join",
            (binding.getattr("module")?.getattr("hidden_size")?, &dtype),
        )?;
    }
    Ok(())
}
