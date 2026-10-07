//! Resolve a launch descriptor once, before model loading and collective setup.

use std::collections::HashSet;

use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyInt, PyTuple, PyType};

use super::{WorkerConfig, device, optional, required};
use crate::worker::placement;
use crate::worker::process_groups::Rendezvous;

/// Immutable Python records are the checkpoint loader's inputs. All deployment
/// decisions are made here; the numerical loader does not parse launch values.
#[pyfunction]
pub(in crate::worker) fn prepare_worker_launch<'py>(
    cls: &Bound<'py, PyType>,
    fields: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    prepare(cls, fields).map_err(|error| {
        let py = fields.py();
        if error.is_instance_of::<PyTypeError>(py) || error.is_instance_of::<PyOverflowError>(py) {
            PyValueError::new_err(error.to_string())
        } else {
            error
        }
    })
}

fn prepare<'py>(
    cls: &Bound<'py, PyType>,
    fields: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = fields.py();
    let records = py.import("uniserve_worker.config.deployment")?;
    let role = optional_text(fields, "role")?.unwrap_or_else(|| "model".into());
    let experts = expert_parallel(fields, &role)?;
    let calls = if role == "experts" {
        if experts
            .as_ref()
            .is_none_or(|world| world.1 == 0 || world.0 < world.1)
        {
            return Err(PyValueError::new_err(
                "expert workers require a disaggregated expert union",
            ));
        }
        PyFrozenSet::empty(py)?
    } else {
        supported_calls(&required(fields, "supported_calls")?)?
    };
    let device = device::normalize(
        py,
        &required(fields, "device")?.str()?.extract::<String>()?,
        "--device",
    )?;
    let generation = device::generation(
        py,
        &optional_text(fields, "mesh")?.unwrap_or_default(),
        &device,
    )?;
    let backends = transfer_backends(&required(fields, "transfer_backends")?)?;
    let exports = transfer_backends(&required(fields, "export_backends")?)?;
    if exports.iter().any(|name| !backends.contains(name)) {
        return Err(PyValueError::new_err(
            "export backends must be bound transports",
        ));
    }
    if backends.iter().any(|name| name == "cuda_vmm") && !device.starts_with("cuda:") {
        return Err(PyValueError::new_err(
            "CUDA VMM requires a CUDA worker device",
        ));
    }
    if device.starts_with("cuda")
        || generation
            .as_deref()
            .is_some_and(|device| device.starts_with("cuda"))
    {
        cuda_representation(fields)?;
    }

    let queue_depth: i64 = required(fields, "queue_depth")?.extract()?;
    let payload: i64 = required(fields, "ipc_payload_cap")?.extract()?;
    if queue_depth <= 0 || payload <= 0 {
        return Err(PyValueError::new_err(if queue_depth <= 0 {
            "--queue-depth must be positive"
        } else {
            "--ipc-payload-cap must be positive"
        }));
    }
    let stub = required(fields, "no_model")?.is_truthy()?;
    let model = optional_text(fields, "model")?;
    if stub && !required(fields, "allow_stub")?.is_truthy()? {
        return Err(PyValueError::new_err(
            "--no-model loads synthetic outputs and requires --allow-stub",
        ));
    }
    if !stub && model.is_none() {
        return Err(PyValueError::new_err(
            "--model is required for a model worker",
        ));
    }

    let ipc = PyDict::new(py);
    for name in [
        "registration_address",
        "channel_transport",
        "acknowledgment_slot",
        "products_cross_hosts",
    ] {
        ipc.set_item(name, required(fields, name)?)?;
    }
    let slots: Vec<usize> = required(fields, "host_slots")?.extract()?;
    ipc.set_item("host_slots", PyTuple::new(py, slots)?)?;
    ipc.set_item("queue_depth", queue_depth)?;
    ipc.set_item("max_payload_bytes", payload)?;

    let args = PyDict::new(py);
    args.set_item("worker_id", required(fields, "worker_id")?)?;
    args.set_item("supported_calls", calls)?;
    args.set_item(
        "ipc",
        records.getattr("WorkerIpcConfig")?.call((), Some(&ipc))?,
    )?;
    args.set_item("local_rank", required(fields, "local_rank")?)?;
    args.set_item(
        "distributed_backend",
        optional_text(fields, "distributed_backend")?,
    )?;
    args.set_item("rendezvous", rendezvous(fields)?)?;
    args.set_item("expert_parallel", experts.map(|(_, _, value)| value))?;
    args.set_item(
        "data_plane",
        records.call_method1(
            "DataPlaneConfig",
            (PyTuple::new(py, backends)?, PyTuple::new(py, exports)?),
        )?,
    )?;
    args.set_item(
        "model",
        match model {
            Some(path) => records.call_method1(
                "ModelLaunchConfig",
                (
                    path,
                    required(fields, "quantization_config")?
                        .cast::<PyDict>()?
                        .copy()?,
                    optional_text(fields, "base_model")?,
                ),
            )?,
            None => py.None().into_bound(py),
        },
    )?;
    args.set_item(
        "execution",
        WorkerConfig::from_launch(fields, device, generation)?,
    )?;
    args.set_item("load", load_config(fields)?)?;
    args.set_item("use_stub_model", stub)?;
    args.set_item(
        "components",
        placement::parse_components(
            required(fields, "components")?.cast()?,
            required(fields, "world_size")?.extract()?,
            &role,
        )?,
    )?;
    cls.call((), Some(&args))
}

fn optional_text(fields: &Bound<'_, PyDict>, name: &str) -> PyResult<Option<String>> {
    Ok(optional(fields, name)?
        .map(|value| {
            value
                .str()?
                .extract::<String>()
                .map(|text| text.trim().to_owned())
        })
        .transpose()?
        .filter(|value| !value.is_empty()))
}

fn load_config<'py>(fields: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyAny>> {
    let py = fields.py();
    let selected: String = required(fields, "load_format")?.extract()?;
    let (format, mode) = match selected.as_str() {
        "auto" | "safetensors" | "pt" => (selected.as_str(), "eager"),
        "dummy" | "layered" => ("auto", selected.as_str()),
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown checkpoint load format {selected:?}"
            )));
        }
    };
    let options = PyDict::new(py);
    options.set_item("format", format)?;
    options.set_item("mode", mode)?;
    // The library IO config converts the caller's optional paths itself.
    options.set_item("download_dir", optional_text(fields, "download_dir")?)?;
    options.set_item(
        "checksum_manifest",
        optional_text(fields, "checksum_manifest")?,
    )?;
    options.set_item("num_threads", optional(fields, "load_threads")?)?;
    py.import("uniserve.loading")?
        .getattr("Config")?
        .call((), Some(&options))
}

fn cuda_representation(fields: &Bound<'_, PyDict>) -> PyResult<()> {
    if required(fields, "model_dtype")?.extract::<String>()? == "float32" {
        return Err(PyValueError::new_err(
            "--dtype float32 has no native CUDA attention kernel: CUDA attention computes in bfloat16 or float16",
        ));
    }
    let quantization = required(fields, "quantization_config")?;
    let override_dtype = quantization
        .cast::<PyDict>()?
        .get_item("kv_cache_dtype")?
        .filter(|value| !value.is_none());
    let dtype = match &override_dtype {
        Some(value) => Some(value.clone()),
        None => optional(fields, "kv_cache_dtype")?,
    };
    if let Some(dtype) = dtype
        && dtype.extract::<String>()? == "float8_e4m3fn"
    {
        let option = if override_dtype.is_some() {
            "--quantization-config kv_cache_dtype"
        } else {
            "--kv-cache-dtype"
        };
        return Err(PyValueError::new_err(format!(
            "{option} float8_e4m3fn has no native CUDA attention kernel: the cache keeps one FP8 scale per block, which no native kernel reads (the FP8-KV TensorRT-LLM kernels take FP8 queries with per-tensor BMM1/BMM2 scales); use a bfloat16 or float16 KV cache"
        )));
    }
    Ok(())
}

fn transfer_backends(value: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let text: String = value.str()?.extract()?;
    let names: Vec<_> = text.split(',').map(|name| name.trim().to_owned()).collect();
    if names.iter().collect::<HashSet<_>>().len() != names.len() {
        return Err(PyValueError::new_err(
            "transfer backends must be nonempty and unique",
        ));
    }
    if names
        .iter()
        .any(|name| !matches!(name.as_str(), "local" | "shm" | "cuda_vmm" | "channel"))
    {
        return Err(PyValueError::new_err(
            "transfer backends must name local, shm, cuda_vmm, or channel",
        ));
    }
    Ok(names)
}

fn rendezvous(fields: &Bound<'_, PyDict>) -> PyResult<Option<Rendezvous>> {
    let address = optional_text(fields, "rendezvous_address")?;
    let fd: Option<i32> = optional(fields, "rendezvous_listen_fd")?
        .map(|value| value.extract())
        .transpose()?;
    let Some(address) = address else {
        if fd.is_some() {
            return Err(PyValueError::new_err(
                "a rendezvous socket requires its address",
            ));
        }
        return Ok(None);
    };
    let first = required(fields, "rank")?.extract::<usize>()? == 0;
    if first != fd.is_some() {
        return Err(PyValueError::new_err(if first {
            "the first rank must inherit its rendezvous socket"
        } else {
            "only the first rank inherits a rendezvous socket"
        }));
    }
    store(&address, fd).map(Some)
}

fn store(address: &str, fd: Option<i32>) -> PyResult<Rendezvous> {
    let (host, port) = address
        .rsplit_once(':')
        .ok_or_else(|| PyValueError::new_err("store address must be host:port"))?;
    Ok(Rendezvous::new(
        host.into(),
        port.parse()
            .map_err(|error| PyValueError::new_err(format!("invalid store port: {error}")))?,
        fd,
    ))
}

// Return role coordinates with the immutable loader view; no second world
// representation is retained after the descriptor has been normalized.
fn expert_parallel<'py>(
    fields: &Bound<'py, PyDict>,
    role: &str,
) -> PyResult<Option<(i64, i64, Bound<'py, PyAny>)>> {
    let Some(value) = optional(fields, "expert_parallel")? else {
        return Ok(None);
    };
    let value = value.cast::<PyDict>()?;
    let rank: i64 = required(value, "rank")?.cast_exact::<PyInt>()?.extract()?;
    let size: i64 = required(value, "size")?.cast_exact::<PyInt>()?.extract()?;
    let attention: i64 = required(value, "attention_ranks")?
        .cast_exact::<PyInt>()?
        .extract()?;
    let exchange: String = required(value, "exchange")?.extract()?;
    if !matches!(
        exchange.as_str(),
        "alltoall" | "megamoe" | "dwdp" | "deepep"
    ) {
        return Err(PyValueError::new_err(format!(
            "unknown expert exchange {exchange:?}; expected alltoall, megamoe, dwdp or deepep"
        )));
    }
    if size < 2 {
        return Err(PyValueError::new_err(
            "an expert-parallel world spans at least two ranks",
        ));
    }
    if rank < 0 || rank >= size {
        return Err(PyValueError::new_err(
            "expert-parallel rank must satisfy 0 <= rank < size",
        ));
    }
    if attention < 0 || attention >= size {
        return Err(PyValueError::new_err(
            "attention ranks must leave at least one expert rank",
        ));
    }
    let start = rank - required(fields, "rank")?.extract::<i64>()?;
    let world_size: i64 = required(fields, "world_size")?.extract()?;
    let stop = start + world_size;
    if start < 0 || stop > size || (start < attention && attention < stop) {
        return Err(PyValueError::new_err(
            "worker ranks must fit within one role in the expert union",
        ));
    }
    if attention != 0
        && (!matches!(exchange.as_str(), "deepep" | "megamoe")
            || ((rank >= attention) != (role == "experts")))
    {
        return Err(PyValueError::new_err(
            "disaggregated role or expert exchange disagrees with placement",
        ));
    }
    if attention == 0 && (world_size != 1 || exchange == "deepep") {
        return Err(PyValueError::new_err(
            "an expert-parallel replica joins its world with its only rank",
        ));
    }
    let address: String = required(value, "address")?.extract()?;
    let fd: Option<i32> = required(value, "listen_fd")?.extract()?;
    if (rank == 0) != fd.is_some() {
        return Err(PyValueError::new_err(
            "exactly the expert-parallel world's rank 0 serves its store",
        ));
    }
    let record = fields
        .py()
        .import("uniserve_worker.config.deployment")?
        .call_method1(
            "ExpertParallelLaunch",
            (rank, size, attention, store(&address, fd)?),
        )?;
    Ok(Some((rank, attention, record)))
}

fn supported_calls<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyFrozenSet>> {
    let groups: &[(&str, &str, &[&str])] = &[
        ("ar_extend", "ForwardMode", &["prefill"]),
        ("ar_decode", "ForwardMode", &["decode"]),
        ("ar_verify", "ForwardMode", &["verify"]),
        ("token_denoising", "ForwardMode", &["token_denoising"]),
        ("encoder_vision", "MediaCall", &["vision_encoding"]),
        ("encoder_latent", "MediaCall", &["latent_encoding"]),
        ("encoder_text", "MediaCall", &["text_encoding"]),
        ("transfer_product", "TransferMode", &["tensor"]),
        ("transfer_kv_export", "TransferMode", &["kv_export"]),
        ("transfer_kv_install", "TransferMode", &["kv_install"]),
        ("diffusion_prepare", "MediaCall", &["latent_preparation"]),
        ("diffusion_step", "MediaCall", &["denoising"]),
        (
            "diffusion_finalize",
            "MediaCall",
            &["image_decoding", "muxing"],
        ),
        (
            "diffusion_decode",
            "MediaCall",
            &["video_decoding", "audio_decoding"],
        ),
        ("media_read", "MediaCall", &["media_reading"]),
        (
            "media_append",
            "MediaCall",
            &["video_encoding", "audio_encoding"],
        ),
    ];
    let selected = if value.is_none() {
        groups
            .iter()
            .map(|(name, _, _)| *name)
            .collect::<Vec<_>>()
            .join(",")
    } else {
        value.str()?.extract::<String>()?
    };
    let module = value.py().import("uniserve_worker.protocol.call")?;
    let mut seen = HashSet::new();
    let mut calls = Vec::new();
    for name in selected
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
    {
        if !seen.insert(name) {
            return Err(PyValueError::new_err(
                "supported calls contain duplicate entries",
            ));
        }
        let (_, class, kinds) = groups
            .iter()
            .find(|(group, _, _)| *group == name)
            .ok_or_else(|| {
                PyValueError::new_err(format!("unknown call in supported calls {selected:?}"))
            })?;
        for kind in *kinds {
            calls.push(module.getattr(*class)?.call1((kind,))?);
        }
    }
    if calls.is_empty() {
        return Err(PyValueError::new_err(
            "supported calls must list at least one group",
        ));
    }
    PyFrozenSet::new(value.py(), calls)
}
