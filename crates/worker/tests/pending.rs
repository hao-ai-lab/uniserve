//! Generated media stays owned until its result is handed to the receiver.

use uniserve_core::{CallId, MediaSource, RequestId, SharedMedia};
use uniserve_worker::{BatchResult, PendingOutput, RequestProgress};
use uniserve_worker_ipc::{BatchOutput, Call, CallKind, MediaCall, RequestKey};

fn pending() -> uniserve_worker::Result<PendingOutput> {
    let call = Call {
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
    };
    PendingOutput::new(&call, RequestProgress::default(), 0)
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

    // SAFETY: both publications are immutable; the refused source was dropped.
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

    // SAFETY: result delivery relinquished the immutable publication.
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
