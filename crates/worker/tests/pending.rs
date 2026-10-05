//! Tensor and media results stay owned until delivery or abandonment.

use uniserve_core::{CallId, MediaSource, RequestId, SharedMedia};
use uniserve_worker::{
    BatchResult, Completion, LatentExport, LatentImport, LatentPool, LatentUpdate, PendingOutput,
    RequestProgress,
};
use uniserve_worker_ipc::{
    BatchOutput, Call, CallKind, DType, LatentParams, Locator, MAX_TRANSFER_HANDLE_BYTES,
    MediaCall, RequestKey, ShapeBound, TensorExport, TensorRef, TensorTransfer, TransferHandle,
    TransferTransport, WorkerInfo,
};

fn call() -> Call {
    Call {
        request_key: RequestKey::new(1, RequestId(7), 1),
        call_id: CallId::new(1, 0),
        coordinates: Default::default(),
        component: "image".into(),
        code: CallKind::Media(MediaCall::ImageDecoding),
        bounds: Default::default(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        consumer_slots: Vec::new(),
        token_input: None,
        token_output: None,
        vision_inputs: Vec::new(),
        latent_feature_input: None,
        encoder_output: None,
        latent_input: None,
        latent_output: None,
        image_input: None,
        image_output: None,
        completion_output: None,
        transition_output: None,
        predicate: None,
        rng: None,
        sampling_state: None,
        input_token_ids: Vec::new(),
        readout: None,
        canvas: None,
        input_image: None,
        kv_input: None,
        kv_output: None,
    }
}

fn pending() -> uniserve_worker::Result<PendingOutput> {
    PendingOutput::new(&call(), RequestProgress::default(), 0)
}

#[test]
fn completed_latent_calls_advance_pool_visibility_and_request_progress()
-> uniserve_worker::Result<()> {
    use std::sync::Arc;

    type Signal = Arc<Completion<(), ()>>;
    type Pool = LatentPool<Arc<LatentImport<Signal>>, Arc<LatentExport<Signal>>>;

    for transferred in [false, true] {
        let mut pool = Pool::new(1, 2, 4)?;
        let mut call = call();
        let mut params = LatentParams {
            request_key: call.request_key,
            call_id: call.call_id,
            page_table: vec![1],
            latent_units: 4,
            height: 16,
            width: 64,
            start_step: 0,
            step_count: 0,
        };
        let mut reference = TensorRef {
            request_key: call.request_key,
            producer_call_id: call.call_id,
            output_index: 0,
            generation: 7,
            dtype: DType::F32,
            shape_bound: ShapeBound::default(),
        };

        call.code = CallKind::Media(MediaCall::LatentPreparation);
        call.latent_output = transferred.then(|| reference.clone());
        let initial = LatentUpdate::for_call(1, &call, &params)?;
        pool.validate_updates(std::slice::from_ref(&initial))?;
        pool.apply_updates(&[initial]);
        let generation = if transferred { 7 } else { 1 };
        assert_eq!(pool.current_bank(1, 0, generation, 4, 16, 64, &[1])?, 1);

        call.code = CallKind::Media(MediaCall::Denoising);
        call.latent_input = call.latent_output.take();
        reference.generation = 19;
        call.latent_output = transferred.then(|| reference.clone());
        params.step_count = 2;
        let update = LatentUpdate::for_call(1, &call, &params)?;
        let mut output = PendingOutput::new(&call, RequestProgress::default(), 0)?;
        pool.validate_updates(std::slice::from_ref(&update))?;
        output.apply_latent_update(&update);
        pool.apply_updates(&[update]);
        let generation = if transferred { 19 } else { 3 };
        assert_eq!(pool.current_bank(1, 2, generation, 4, 16, 64, &[1])?, 0);
        assert_eq!(output.progress.flow_step, 2);

        if transferred {
            call.code = CallKind::Media(MediaCall::ImageDecoding);
            call.latent_input = call.latent_output.take();
            params.start_step = 2;
            params.step_count = 0;
            let release = LatentUpdate::for_call(1, &call, &params)?;
            pool.validate_updates(std::slice::from_ref(&release))?;
            output.apply_latent_update(&release);
            pool.apply_updates(&[release]);
            assert_eq!(output.progress.flow_step, 0);
        } else {
            pool.release_slots(&[1])?;
        }
        pool.reap();
        pool.require_initial(1, &[1])?;
    }
    Ok(())
}

#[test]
fn rejected_tensor_results_preserve_the_accepted_exports() -> Result<(), Box<dyn std::error::Error>>
{
    let mut call = call();
    let reference = TensorRef {
        request_key: call.request_key,
        producer_call_id: call.call_id,
        output_index: 0,
        generation: 1,
        dtype: DType::U8,
        shape_bound: ShapeBound::default(),
    };
    call.outputs.push(reference.clone());
    let mut output = PendingOutput::new(&call, RequestProgress::default(), 0)?;
    let export = TensorExport {
        product: reference,
        value: TransferHandle::DeviceProduct {
            height: 0,
            width: 0,
            value_range: String::new(),
            tensor: TensorTransfer {
                shape: vec![1],
                locations: vec![Locator {
                    source: WorkerInfo::default().endpoint,
                    transport: TransferTransport::Channel {
                        endpoint: "reader".into(),
                        payload: vec![7],
                    },
                    nbytes: 1,
                    dtype: "uint8".into(),
                    shape: vec![1],
                    offset: vec![0],
                    device: "cpu".into(),
                }],
            },
        },
    };
    output.set_products(vec![export.clone()])?;

    let mut undeclared = export.clone();
    undeclared.product.output_index = 1;
    assert!(
        output
            .set_products(vec![export.clone(), undeclared])
            .is_err()
    );
    assert_eq!(output.products(), std::slice::from_ref(&export));

    // Locator metadata has a separate budget from the transported bytes.
    let mut oversized = export.clone();
    let tensor = &mut oversized.value.tensors_mut()[0];
    let location = &mut tensor.locations[0];
    location.transport = TransferTransport::Channel {
        endpoint: "r".repeat(MAX_TRANSFER_HANDLE_BYTES + 1),
        payload: vec![7],
    };
    assert!(output.set_products(vec![oversized]).is_err());
    assert_eq!(output.products(), &[export]);
    Ok(())
}

#[test]
fn undelivered_media_is_unlinked_on_abandonment_or_drop() -> Result<(), Box<dyn std::error::Error>>
{
    for abandon in [false, true] {
        let mut output = pending()?;
        let source = MediaSource::publish(b"encoded image")?;
        let locator = source.locator();
        output.set_media(source)?;
        if abandon {
            output.discard_media();
            assert!(output.output.media_output.is_none());
        }
        drop(output);

        // SAFETY: the publisher completed all writes before relinquishing
        // the source; this claim must fail because no result was delivered.
        assert!(unsafe { SharedMedia::open(&locator.name, locator.bytes) }.is_err());
    }
    Ok(())
}

#[test]
fn a_second_media_result_is_rejected_without_leaking_its_storage()
-> Result<(), Box<dyn std::error::Error>> {
    let mut output = pending()?;
    let source = MediaSource::publish(b"first")?;
    let first = source.locator();
    output.set_media(source)?;
    let source = MediaSource::publish(b"second")?;
    let second = source.locator();
    assert!(output.set_media(source).is_err());

    // SAFETY: both exports are immutable; the refused source was dropped.
    assert!(unsafe { SharedMedia::open(&second.name, second.bytes) }.is_err());
    output.handoff_media();
    drop(output);
    let media = unsafe { SharedMedia::open(&first.name, first.bytes) }?;
    assert_eq!(media.as_bytes(), b"first");
    Ok(())
}

#[test]
fn handed_off_media_survives_output_retirement() -> Result<(), Box<dyn std::error::Error>> {
    let mut output = pending()?;
    let source = MediaSource::publish(b"encoded image")?;
    let locator = source.locator();
    output.set_media(source)?;
    output.handoff_media();
    output.discard_media();
    assert_eq!(
        output
            .output
            .media_output
            .as_ref()
            .ok_or("media output missing")?
            .bytes,
        13
    );
    drop(output);

    // SAFETY: result delivery relinquished the immutable export.
    let media = unsafe { SharedMedia::open(&locator.name, locator.bytes) }?;
    assert_eq!(media.as_bytes(), b"encoded image");
    Ok(())
}

#[test]
fn batch_results_retain_media_until_delivery_or_failure() -> Result<(), Box<dyn std::error::Error>>
{
    for delivered in [false, true] {
        let mut output = pending()?;
        let source = MediaSource::publish(b"encoded image")?;
        let locator = source.locator();
        output.set_media(source)?;
        let mut result = BatchResult {
            output: BatchOutput {
                batch_id: 1,
                completions: vec![output.output.clone()],
                products: Vec::new(),
                worker_exec_us: None,
                forward_stats: None,
            },
            media: output.take_media().into_iter().collect(),
        };
        drop(output);

        if delivered {
            for source in result.media.drain(..) {
                source.into_locator();
            }
        }
        drop(result);

        // SAFETY: the encoded bytes are immutable. Delivery hands them to
        // this reader; a failed or discarded response must have unlinked them.
        let received = unsafe { SharedMedia::open(&locator.name, locator.bytes) };
        if delivered {
            assert_eq!(received?.as_bytes(), b"encoded image");
        } else {
            assert!(received.is_err());
        }
    }
    Ok(())
}
