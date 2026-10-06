//! Representative token, canvas and image requests before admission begins.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use uniserve_core::{ImageParams, SamplingParams};
use uniserve_worker_ipc::{
    ArRequestParams, CanvasSampling, CanvasStep, DType, DimBound, DrawLayout, Rng, ShapeBound,
};

use super::*;

impl Warmup {
    pub(super) fn tokens(&mut self, py: Python<'_>) -> PyResult<()> {
        if !self.supports(CallKind::Forward(ForwardMode::Prefill)) {
            return Ok(());
        }

        let key = key(1);
        let admission = admission(
            key,
            1,
            SamplingParams {
                temperature: 0.0,
                ignore_eos: true,
                ..SamplingParams::default()
            },
            None,
        );

        let mut prompt = self.call(key, 0, CallKind::Forward(ForwardMode::Prefill));
        prompt.bounds.max_tokens = 1;
        prompt.input_token_ids.push(0);
        let token = tensor(key, prompt.call_id, 0, 1, DType::I64, Vec::new());
        prompt.token_output = Some(token.clone());
        let decode = self.supports(CallKind::Forward(ForwardMode::Decode));
        let batch = self.plan(py, vec![admission], vec![prompt], None)?;
        self.submit(py, batch, decode)?;

        if decode {
            let mut call = self.call(key, 0, CallKind::Forward(ForwardMode::Decode));
            call.coordinates.logical_position = 1;
            call.coordinates.kv_visible_len = 1;
            call.coordinates.kv_computed_len = 1;
            call.bounds.max_tokens = 1;
            call.token_output = Some(tensor(key, call.call_id, 0, 2, DType::I64, Vec::new()));
            call.predicate = Some(token.clone());
            let batch = self.plan(py, Vec::new(), vec![call], None)?;
            self.submit(py, batch, true)?;
            self.free_products(py, vec![token.buffer_id()])?;
        }

        self.synchronize(py)?;
        self.finish(py, key)?;
        log(py, "completed token runtime warmup")
    }

    pub(super) fn canvas(&mut self, py: Python<'_>) -> PyResult<()> {
        let Some(runner) = self.runner.borrow(py).canvas_runner(py) else {
            return Ok(());
        };
        let runner = runner.bind(py);

        let slots = runner.getattr("canvas_slots")?;
        if slots.is_none() {
            return Ok(());
        }

        let served: Py<crate::batches::CanvasSampling> = slots.getattr("served")?.extract()?;
        let sampling = served.get().inner;
        let length = sampling.canvas_length;
        let rows: usize = runner.getattr("step_rows")?.extract()?;
        let keys: Vec<_> = (1..=rows).map(|row| key(row as u64)).collect();
        let admissions = keys
            .iter()
            .enumerate()
            .map(|(row, &key)| {
                admission(
                    key,
                    row as u32 + 1,
                    SamplingParams {
                        seed: Some(row as u64 + 1),
                        ..SamplingParams::default()
                    },
                    Some(sampling),
                )
            })
            .collect();

        let calls = keys
            .iter()
            .enumerate()
            .map(|(row, &key)| {
                let mut call = self.call(key, row, CallKind::Forward(ForwardMode::Prefill));
                call.bounds.max_tokens = 1;
                call.input_token_ids.push(0);
                call
            })
            .collect();
        let batch = self.plan(py, admissions, calls, None)?;
        self.submit(py, batch, false)?;

        // Sampling handles whole chunks plus a remainder. Prepare every row
        // count it can see while preserving each request's canvas position.
        let mut progress = vec![(0, 0); rows];
        for count in 1..=rows {
            let mut calls = Vec::with_capacity(count);
            for (row, &key) in keys[..count].iter().enumerate() {
                let (block, step) = progress[row];
                let mut call = self.call(key, row, CallKind::Forward(ForwardMode::TokenDenoising));
                call.coordinates.logical_position = 1;
                call.coordinates.kv_visible_len = 1;
                call.coordinates.kv_computed_len = 1;
                call.bounds.max_tokens = length;
                call.bounds.max_completion_bytes = 4 * u64::from(length);
                call.canvas = Some(CanvasStep { block, step });
                calls.push(call);
                progress[row] = if step + 1 == sampling.max_steps {
                    (block + 1, 0)
                } else {
                    (block, step + 1)
                };
            }
            let batch = self.plan(py, Vec::new(), calls, None)?;
            self.submit(py, batch, false)?;
        }

        let mut commit = self.call(keys[0], 0, CallKind::Forward(ForwardMode::Prefill));
        commit.coordinates.logical_position = 1;
        commit.coordinates.kv_visible_len = 1;
        commit.coordinates.kv_computed_len = 1;
        commit.bounds.max_tokens = length;
        commit.input_token_ids = vec![0; length as usize];
        let batch = self.plan(py, Vec::new(), vec![commit], None)?;
        self.submit(py, batch, false)?;
        self.synchronize(py)?;
        for key in keys {
            self.finish(py, key)?;
        }
        log(py, "completed canvas generation warmup")
    }

    pub(super) fn flow(&mut self, py: Python<'_>) -> PyResult<()> {
        if !self.supports(CallKind::Media(MediaCall::LatentPreparation))
            || !self.supports(CallKind::Media(MediaCall::Denoising))
            || self.runner.bind(py).getattr("image_builder")?.is_none()
        {
            return Ok(());
        }

        let branches: Vec<u32> = self
            .runner
            .bind(py)
            .getattr("flow_cfg_branches")?
            .extract()?;
        let configured: Vec<Py<PyAny>> =
            self.runner.bind(py).getattr("flow_captures")?.extract()?;
        let mut request_id = 1;
        let mut generation = 1;

        for branches in branches {
            let (mut height, mut width) = self.image_size(py)?;
            let mut rows = 1;
            for shape in configured.iter().rev() {
                let shape = shape.bind(py);
                if shape.getattr("cfg_branches")?.extract::<u32>()? == branches {
                    height = shape.getattr("height")?.extract()?;
                    width = shape.getattr("width")?.extract()?;
                    rows = shape.getattr("rows")?.extract()?;
                    break;
                }
            }

            // Guided requests also occupy one alternative-prefix slot. Real
            // rows grow upward and their prefix slots downward without overlap.
            rows = rows.min(self.info.request_slots / if branches > 1 { 2 } else { 1 });
            if rows == 0 {
                continue;
            }

            let keys: Vec<_> = (request_id..request_id + u64::from(rows))
                .map(key)
                .collect();
            request_id += u64::from(rows);

            let options = PyDict::new(py);
            options.set_item("steps", 2)?;
            options.set_item("height", height)?;
            options.set_item("width", width)?;
            let image = py
                .import("uniserve_worker.model_executor.startup")?
                .call_method("capture_image_parameters", (branches,), Some(&options))?;
            let image: ImageParams = pythonize::depythonize(&image.call_method0("to_mapping")?)?;
            let admissions = keys
                .iter()
                .enumerate()
                .map(|(row, &key)| NewRequest {
                    request_key: key,
                    request_pool_idx: row as u32 + 1,
                    ar: None,
                    image: Some(image.clone()),
                    diffusion: None,
                    video: None,
                    prompt_token_ids: Vec::new(),
                    input_images: 0,
                })
                .collect();

            let mut exports = Vec::new();
            let mut conditionings = Vec::new();
            for (row, &key) in keys.iter().enumerate() {
                let mut call = self.call(key, row, CallKind::Transfer(TransferMode::KvExport));
                call.bounds.max_transfer_bytes = 1 << 20;
                let buffer = BufferId {
                    owner: key,
                    producer_call_id: call.call_id,
                    output_index: 0,
                    generation,
                };
                generation += 1;
                call.kv_output = Some(buffer);
                conditionings.push(buffer);
                exports.push(call);
            }
            let batch = self.plan(py, admissions, exports, Some((height, width)))?;
            self.submit(py, batch, false)?;

            let elements = self
                .latent_shape(py, height, width)?
                .into_iter()
                .product::<u32>()
                .max(1);
            let dims = vec![DimBound::Device { max: elements }];
            let mut latents = Vec::new();
            let mut preparation = Vec::new();
            for (row, &key) in keys.iter().enumerate() {
                let mut call = self.call(key, row, CallKind::Media(MediaCall::LatentPreparation));
                let latent = tensor(key, call.call_id, 0, generation, DType::BF16, dims.clone());
                generation += 1;
                let ready = tensor(
                    key,
                    call.call_id,
                    1,
                    generation,
                    DType::U8,
                    vec![DimBound::Static(1)],
                );
                generation += 1;
                call.bounds.max_tokens = 1;
                call.bounds.max_latent_bytes = 2 * u64::from(elements);
                call.kv_input = Some(conditionings[row]);
                call.latent_output = Some(latent.clone());
                call.completion_output = Some(ready);
                call.rng = Some(Rng {
                    seed: 0,
                    semantic_index_base: 1,
                    draw_layout: DrawLayout::FlowNoise,
                });
                latents.push(latent);
                preparation.push(call);
            }
            let batch = self.plan(py, Vec::new(), preparation, Some((height, width)))?;
            self.submit(py, batch, false)?;

            for step in 0..2 {
                let mut calls = Vec::new();
                let mut outputs = Vec::new();
                for (row, &key) in keys.iter().enumerate() {
                    let mut call = self.call(key, row, CallKind::Media(MediaCall::Denoising));
                    let output =
                        tensor(key, call.call_id, 0, generation, DType::BF16, dims.clone());
                    generation += 1;
                    call.coordinates.flow_step = step;
                    call.bounds.max_tokens = 1;
                    call.bounds.max_latent_bytes = 2 * u64::from(elements);
                    call.kv_input = Some(conditionings[row]);
                    call.latent_input = Some(latents[row].clone());
                    call.latent_output = Some(output.clone());
                    calls.push(call);
                    outputs.push(output);
                }
                let batch = self.plan(py, Vec::new(), calls, Some((height, width)))?;
                self.submit(py, batch, false)?;
                self.free_products(py, latents.iter().map(TensorRef::buffer_id).collect())?;
                latents = outputs;
            }
            for key in keys {
                self.finish(py, key)?;
            }
        }
        log(py, "completed flow runtime warmup")
    }

    pub(super) fn image_size(&self, py: Python<'_>) -> PyResult<(u32, u32)> {
        let builder = self.runner.bind(py).getattr("image_builder")?;
        let mut capacity = self.info.latent_capacity_units().max(1);
        let downsample = if builder.is_none() {
            1
        } else {
            capacity = capacity.min(
                builder.getattr("max_tokens")?.extract::<u64>()?
                    + builder.getattr("framing")?.extract::<u64>()?,
            );
            builder
                .getattr("denoiser")?
                .getattr("downsample")?
                .extract::<u32>()?
        };
        let side = capacity.isqrt().max(1) as u32 * downsample;
        Ok((side, side))
    }

    pub(super) fn latent_shape(
        &self,
        py: Python<'_>,
        height: u32,
        width: u32,
    ) -> PyResult<Vec<u32>> {
        self.runner
            .bind(py)
            .getattr("image_builder")?
            .getattr("denoiser")?
            .call_method1("latent_shape", ("image", image_size(py, height, width)?))?
            .extract()
    }

    fn synchronize(&self, py: Python<'_>) -> PyResult<()> {
        py.import("torch.cuda")?
            .call_method1("synchronize", (&self.info.device,))
            .map(drop)
    }
}

fn admission(
    key: RequestKey,
    slot: u32,
    sampling: SamplingParams,
    canvas: Option<CanvasSampling>,
) -> NewRequest {
    NewRequest {
        request_key: key,
        request_pool_idx: slot,
        ar: Some(ArRequestParams {
            sampling,
            canvas,
            initial_position: 0,
            finish_token_ids: Vec::new(),
            negative_token_ids: Vec::new(),
        }),
        image: None,
        diffusion: None,
        video: None,
        prompt_token_ids: Vec::new(),
        input_images: 0,
    }
}

fn tensor(
    key: RequestKey,
    call: CallId,
    output_index: u16,
    generation: u32,
    dtype: DType,
    dims: Vec<DimBound>,
) -> TensorRef {
    TensorRef {
        request_key: key,
        producer_call_id: call,
        output_index,
        generation,
        dtype,
        shape_bound: ShapeBound { dims },
    }
}

fn log(py: Python<'_>, message: &str) -> PyResult<()> {
    py.import("logging")?
        .call_method1("getLogger", ("uniserve_worker.worker",))?
        .call_method1("info", (message,))
        .map(drop)
}
