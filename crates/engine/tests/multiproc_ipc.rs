//! Multiprocess framing, single-submit delivery, and physical-rank recovery.

#![cfg(target_os = "linux")]

use std::fs::{OpenOptions, remove_file};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uniserve_worker_ipc::{CallCoordinates, ForwardMode, MediaCall, TransferMode};

use anyhow::Context as _;
use uniserve_core::{
    BlockId, DiffusionRequest, DiffusionSamplingParams, EngineCoreOutput, Request, RequestId,
    RuntimeFamily, SamplingParams,
};
use uniserve_engine::{
    EngineConfig, EngineCore, Executor, WorkerConfig, WorkerFailure, WorkerGroup, WorkerProcessArgs,
};
use uniserve_worker_ipc::{
    ArRequestParams, Batch, BatchCommand, BlockTable, Bounds, CachePageAllocation, Call, CallId,
    CallKind, CallStatus, DType, DimBound, ErrorCode, ForwardBatch, Locator, NewRequest,
    RequestKey, ShapeBound, TensorPublication, TensorRef, TransferHandle, TransferTransport,
};

const WORLD_SIZE: usize = 2;
const QUEUE_DEPTH: usize = 2;
static CHILD_LAUNCH_ENV_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn independent_components_complete_on_their_assigned_ranks() -> anyhow::Result<()> {
    let mut args = rank_group_args(1 << 20, 8 << 20);
    // The language backbone and the patch encoder hold a rank each, which is
    // the arrangement a model with several components is served in.
    args.components = [
        (
            "model".into(),
            uniserve_core::ComponentConfig::parallel(vec![1], Default::default()),
        ),
        (
            "vision_encoder".into(),
            uniserve_core::ComponentConfig::parallel(vec![0], Default::default()),
        ),
    ]
    .into_iter()
    .collect();
    let mut worker = WorkerGroup::spawn(args)?;
    let supported = worker.info().supported_calls.clone();
    assert!(supported.contains(&CallKind::Forward(ForwardMode::Prefill)));
    assert!(supported.contains(&CallKind::Media(MediaCall::VisionEncoding)));
    assert_eq!(
        worker.info().media_components[&MediaCall::VisionEncoding],
        "vision_encoder"
    );

    let first = text_admission(51, 1, 1)?;
    let second = text_admission(52, 1, 2)?;
    let first_key = first.request_key;
    let second_key = second.request_key;
    let mut extend = token_batch(
        1,
        1,
        first_key,
        Some(first),
        CallId::new(2, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[7, 8],
        BlockId(1),
        0,
    );
    let second_admission = token_batch(
        1,
        1,
        second_key,
        Some(second),
        CallId::new(3, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[9, 10],
        BlockId(2),
        0,
    );
    let mut starts = std::mem::take(&mut extend.commands);
    starts.extend(second_admission.commands);
    let admission = execute(
        &mut worker,
        Batch::new(1, vec![], vec![]).with_commands(starts),
    )?;
    assert!(admission.results.is_empty());

    // A batch carries one call kind for one component, so each component's work
    // travels in its own batch and retires on the ranks that component holds.
    extend.batch_id = 2;
    extend.collective_seq = 2;
    let extend_report = execute(&mut worker, extend)?;
    assert_eq!(extend_report.results.len(), 1);
    assert_eq!(extend_report.results[0].output.call_id, CallId::new(2, 0));
    assert_eq!(extend_report.results[0].output.status, CallStatus::Ok);
    assert_eq!(
        extend_report.results[0]
            .output
            .committed_tokens
            .as_slice()
            .len(),
        1
    );

    let image_bytes = b"iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAGUlEQVR4nGN0SGhgIAUwkaR6VMOohiGlAQCjvQFA6eri4wAAAABJRU5ErkJggg==".to_vec();
    let feature = TensorRef {
        request_key: second_key,
        producer_call_id: CallId::new(3, 0),
        output_index: 0,
        generation: 1,
        dtype: DType::BF16,
        shape_bound: ShapeBound {
            dims: vec![DimBound::Device { max: 4096 }],
        },
    };
    let encode = Call {
        consumer_slots: Vec::new(),
        coordinates: CallCoordinates::default(),
        token_input: None,

        token_output: None,
        vision_input: None,
        latent_feature_input: None,
        encoder_output: Some(feature.clone()),
        latent_input: None,
        latent_output: None,
        image_input: None,
        image_output: None,
        completion_output: None,
        transition_output: None,

        kv_input: None,
        kv_output: None,
        input_image: Some(String::from_utf8(image_bytes).unwrap().into()),
        input_token_ids: Vec::new(),
        sampling_state: None,
        request_key: second_key,
        call_id: CallId::new(3, 0),
        component: "vision_encoder".into(),
        code: CallKind::Media(MediaCall::VisionEncoding),
        bounds: Bounds {
            max_tokens: 64,
            max_latent_bytes: 8192,
            ..Bounds::default()
        },
        inputs: Vec::new(),
        outputs: Vec::new(),
        predicate: None,
        rng: None,
    };
    let mut encode = Batch::new(3, Vec::new(), vec![encode]);
    encode.collective_seq = 3;
    encode
        .buffer_allocations
        .push(uniserve_worker_ipc::BufferAllocation {
            buffer: feature.buffer_id(),
            offset: 0,
            bytes: feature.max_bytes(),
        });
    let encode_report = execute(&mut worker, encode)?;
    assert_eq!(encode_report.results.len(), 1);
    assert_eq!(encode_report.results[0].output.call_id, CallId::new(3, 0));
    assert_eq!(encode_report.results[0].output.status, CallStatus::Ok);

    // Closing a request releases every participating rank's physical storage.
    worker.submit_batch(
        Batch::new(4, vec![], vec![]).with_commands(
            [first_key, second_key]
                .into_iter()
                .map(|request_key| BatchCommand::Finish {
                    request_key,
                    retained_buffers: vec![],
                })
                .collect(),
        ),
    )?;
    let report = worker
        .poll_batch(Duration::from_secs(5))?
        .context("request release did not retire")?;
    assert!(report.results.is_empty());
    worker.close()?;
    Ok(())
}

/// Reads one rank's registration report and returns the endpoint it names.
fn accept_reported_endpoint(listener: &std::net::TcpListener) -> anyhow::Result<String> {
    use std::io::BufRead as _;

    let (stream, _) = listener.accept()?;
    stream.set_read_timeout(Some(Duration::from_secs(60)))?;
    let mut line = String::new();
    std::io::BufReader::new(stream).read_line(&mut line)?;
    let report: serde_json::Value = serde_json::from_str(line.trim())?;
    Ok(report["endpoint"]
        .as_str()
        .context("rank registration named no endpoint")?
        .to_owned())
}

#[test]
fn launch_refuses_a_rank_that_offers_an_unsupported_channel() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;

    // A rank names its own endpoint, so the mechanism it offers is data the
    // head has to check rather than a value it chose.
    let directory = tempfile::tempdir()?;
    let wrapper = directory.path().join("worker");
    std::fs::write(
        &wrapper,
        r#"#!/usr/bin/env python3
import json, socket, sys

descriptor = json.load(open(sys.argv[sys.argv.index("--launch-descriptor") + 1]))
host, _, port = descriptor["registration_address"].rpartition(":")
report = {
    "worker_id": descriptor["worker_id"],
    "rank": descriptor["rank"],
    "transport": "socket",
    "endpoint": "unreachable",
}
with socket.create_connection((host, int(port)), timeout=60) as connection:
    connection.sendall(json.dumps(report).encode() + b"\n")
"#,
    )?;
    std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))?;
    let mut args = rank_group_args(1 << 20, 8 << 20);
    args.python = wrapper;

    let reported = match WorkerGroup::spawn(args) {
        Ok(_) => anyhow::bail!("an unsupported channel transport was accepted"),
        Err(error) => format!("{error:#}"),
    };
    assert!(
        reported.contains("unsupported channel transport socket"),
        "launch failure did not name the offered transport: {reported}"
    );
    Ok(())
}

#[test]
fn native_close_drains_accepted_results_on_each_launch() -> anyhow::Result<()> {
    use uniserve_worker_ipc::{ClientEndpoint, WorkerRequest, WorkerResponse};

    // Each launch names its own endpoint and reports it here, so a graceful
    // exit is observable as the next launch registering and serving again.
    for _ in 0..2 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let descriptor_directory = tempfile::tempdir()?;
        let descriptor_path = descriptor_directory.path().join("launch.json");
        std::fs::write(
            &descriptor_path,
            serde_json::to_vec(&stub_launch_descriptor(&listener.local_addr()?.to_string()))?,
        )?;
        let mut child = std::process::Command::new(worker_python())
            .args(["-m", "uniserve_worker.main"])
            .arg("--launch-descriptor")
            .arg(&descriptor_path)
            .spawn()?;
        let result = (|| -> anyhow::Result<()> {
            let client =
                ClientEndpoint::connect(&accept_reported_endpoint(&listener)?, 1 << 20, 8)?;
            let admission = text_admission(51, 1, 1)?;
            let batch = token_batch(
                1,
                1,
                admission.request_key,
                Some(admission),
                CallId::new(1, 0),
                CallKind::Forward(ForwardMode::Prefill),
                &[7, 8],
                BlockId(1),
                0,
            );
            let request = |mut request: WorkerRequest, message_id| {
                request.set_call_id(Some(message_id));
                request
            };
            let initial = client.send_request(&request(WorkerRequest::submit(batch.clone()), 1))?;
            let first = client
                .recv_response_timeout(&initial, Duration::from_secs(30))?
                .context("initial submission did not complete")?
                .decode_response()?;
            let WorkerResponse::Result { result: first, .. } = first else {
                anyhow::bail!("initial submission failed: {first:?}");
            };
            anyhow::ensure!(
                first.completions.len() == 1,
                "initial completion count changed"
            );
            anyhow::ensure!(
                first.completions[0].status == CallStatus::Ok,
                "initial call failed"
            );
            anyhow::ensure!(
                first.completions[0].committed_tokens.as_slice().len() == 1,
                "initial token missing"
            );
            drop(initial);

            let other = text_admission(52, 1, 2)?;
            let other_run = token_batch(
                2,
                2,
                other.request_key,
                Some(other),
                CallId::new(2, 0),
                CallKind::Forward(ForwardMode::Prefill),
                &[9, 10],
                BlockId(2),
                0,
            );
            let accepted = client.send_request(&request(WorkerRequest::submit(other_run), 2))?;
            let rejected =
                client.send_request(&request(WorkerRequest::submit(batch.clone()), 3))?;
            let duplicate = client.send_request(&request(WorkerRequest::submit(batch), 4))?;
            let close = client.send_request(&request(WorkerRequest::close(), 5))?;
            let closed = client
                .recv_response_timeout(&close, Duration::from_secs(30))?
                .context("Close did not drain accepted work")?
                .decode_response()?;
            anyhow::ensure!(
                closed
                    == WorkerResponse::Ok {
                        message_id: Some(5)
                    },
                "Close failed: {closed:?}"
            );

            // Receiving Close first must still leave every accepted response available.
            let receive = |pending| -> anyhow::Result<WorkerResponse> {
                Ok(client
                    .try_recv_response(pending)?
                    .context("Close acknowledged before accepted response delivery")?
                    .decode_response()?)
            };
            let WorkerResponse::Result { result: other, .. } = receive(&accepted)? else {
                anyhow::bail!("independent submission did not return a result");
            };
            anyhow::ensure!(
                other.completions.len() == 1,
                "independent batch did not complete"
            );
            anyhow::ensure!(
                other.completions[0].call_id == CallId::new(2, 0),
                "independent call identity changed"
            );
            for pending in [&rejected, &duplicate] {
                let WorkerResponse::Error { error, .. } = receive(pending)? else {
                    anyhow::bail!("duplicate submission was accepted");
                };
                anyhow::ensure!(
                    error.code.as_deref() == Some("InvalidDescriptor"),
                    "duplicate must be a protocol error"
                );
            }
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            loop {
                if let Some(status) = child.try_wait()? {
                    anyhow::ensure!(status.success(), "worker shutdown failed: {status}");
                    break;
                }
                anyhow::ensure!(
                    std::time::Instant::now() < deadline,
                    "worker did not exit after Close"
                );
                thread::sleep(Duration::from_millis(20));
            }
            Ok(())
        })();
        // Clean up the external process even when an assertion or transport call fails.
        if result.is_err() {
            let _ = child.kill();
            let _ = child.wait();
        }
        result?;
    }
    Ok(())
}

#[test]
fn components_transfer_published_values_within_one_worker() -> anyhow::Result<()> {
    use uniserve_engine::{ExecutionBatch, RequestPlacement, WorkerExecutor, WorkerId};

    for transfer in [
        uniserve_engine::TransferConfig::parse("worker:1->worker:0=shm")?,
        uniserve_engine::TransferConfig::default(),
    ] {
        let mut args = rank_group_args(1 << 20, 8 << 20);
        args.components = [
            (
                "model".into(),
                uniserve_core::ComponentConfig::parallel(vec![1], Default::default()),
            ),
            (
                "vision_encoder".into(),
                uniserve_core::ComponentConfig::parallel(vec![0], Default::default()),
            ),
        ]
        .into_iter()
        .collect();
        let transfer = transfer.with_worker_defaults(&[WorkerConfig {
            id: WorkerId("worker".into()),
            ranks: args.ranks.clone(),
            components: args.components.clone(),
            queue_depth: args.queue_depth,
            memory_fraction: None,
        }])?;
        args.transfer = transfer.clone();
        let mut executor = WorkerExecutor::try_new(
            vec![(WorkerId("worker".into()), WorkerGroup::spawn(args)?)],
            transfer,
        )?;
        let bind = |mut batch: Batch, entry: &str| {
            let mut call = batch.calls.remove(0);
            call.component = entry.into();
            ExecutionBatch::new(
                batch.batch_id,
                vec![(
                    call,
                    RequestPlacement {
                        worker: WorkerId("worker".into()),
                        request_pool_idx: None,
                        block_tables: batch.block_tables,
                        new_cache_pages: batch.new_cache_pages,
                        forward: batch.forward,
                        latent: batch.latent_params.into_iter().next(),
                        decode: batch.decode_ranges.into_iter().next(),
                        buffers: batch.buffer_allocations,
                    },
                )],
                batch.commands,
                batch.input_products,
            )
        };
        let admission = text_admission(61, 1, 1)?;
        let source = token_batch(
            1,
            1,
            admission.request_key,
            Some(admission.clone()),
            CallId::new(1, 0),
            CallKind::Forward(ForwardMode::Prefill),
            &[7],
            BlockId(1),
            0,
        );
        let value = source.calls[0].token_output.clone().unwrap();
        let logical = bind(source, "model");
        let publication = TensorRef {
            producer_call_id: CallId::new(2, 0),
            generation: 2,
            ..value.clone()
        };
        let publish = Call {
            consumer_slots: Vec::new(),
            coordinates: CallCoordinates::default(),
            token_input: Some(value.clone()),

            token_output: Some(publication.clone()),
            vision_input: None,
            latent_feature_input: None,
            encoder_output: None,
            latent_input: None,
            latent_output: None,
            image_input: None,
            image_output: None,
            completion_output: None,
            transition_output: None,

            input_image: None,
            kv_input: None,
            kv_output: None,
            input_token_ids: Vec::new(),
            sampling_state: None,
            request_key: admission.request_key,
            call_id: CallId::new(2, 0),
            component: "model".into(),
            code: CallKind::Transfer(TransferMode::Tensor),
            bounds: Bounds {
                max_transfer_bytes: value.max_bytes(),
                ..Bounds::default()
            },
            inputs: Vec::new(),
            outputs: Vec::new(),
            predicate: None,
            rng: None,
        };

        let copy = TensorRef {
            producer_call_id: CallId::new(3, 0),
            generation: 3,
            ..publication.clone()
        };
        let consume = Call {
            consumer_slots: Vec::new(),
            coordinates: CallCoordinates::default(),
            token_input: Some(publication.clone()),

            token_output: Some(copy.clone()),
            vision_input: None,
            latent_feature_input: None,
            encoder_output: None,
            latent_input: None,
            latent_output: None,
            image_input: None,
            image_output: None,
            completion_output: None,
            transition_output: None,

            input_image: None,
            kv_input: None,
            kv_output: None,
            input_token_ids: Vec::new(),
            sampling_state: None,
            request_key: admission.request_key,
            call_id: CallId::new(3, 0),
            component: "vision_encoder".into(),
            code: CallKind::Transfer(TransferMode::Tensor),
            bounds: Bounds {
                max_transfer_bytes: publication.max_bytes(),
                ..Bounds::default()
            },
            inputs: Vec::new(),
            outputs: Vec::new(),
            predicate: None,
            rng: None,
        };
        let mut completions = std::collections::BTreeMap::new();
        let drain = |executor: &mut WorkerExecutor,
                     completions: &mut std::collections::BTreeMap<_, _>|
         -> anyhow::Result<()> {
            loop {
                let result = poll_logical(executor)?.context("a submitted batch did not retire")?;
                for result in result.results {
                    assert_eq!(result.output.status, CallStatus::Ok);
                    assert!(
                        completions
                            .insert(result.output.call_id, result.output)
                            .is_none()
                    );
                }
                if result.done {
                    return Ok(());
                }
            }
        };

        // Each batch carries one call kind for one component, and a batch that
        // reads another's product follows it: a rank executes batches in
        // channel order and refuses an identifier that does not advance, which
        // is what carries the dependency now that no ledger does.
        executor.submit(logical)?;
        drain(&mut executor, &mut completions)?;

        executor.submit(bind(Batch::new(2, vec![], vec![publish]), "model"))?;
        drain(&mut executor, &mut completions)?;

        executor.submit(bind(Batch::new(3, vec![], vec![consume]), "vision_encoder"))?;
        drain(&mut executor, &mut completions)?;
        assert_eq!(
            completions.keys().copied().collect::<Vec<_>>(),
            vec![CallId::new(1, 0), CallId::new(2, 0), CallId::new(3, 0),]
        );
        assert_eq!(completions[&CallId::new(1, 0)].committed_tokens, vec![1000]);

        executor.close()?;
    }
    Ok(())
}

#[test]
fn same_batch_successor_consumes_the_unobserved_device_token() -> anyhow::Result<()> {
    use uniserve_engine::{ExecutionBatch, RequestPlacement, WorkerExecutor, WorkerId};

    let transfer = uniserve_engine::TransferConfig::default();
    // Four calls of one request travel as four batches, all submitted before
    // any result is read, so each successor starts from its predecessor's
    // device product rather than from a host observation.
    let mut args = rank_group_args(1 << 20, 8 << 20);
    args.queue_depth = 4;
    let mut executor = WorkerExecutor::try_new(
        vec![(WorkerId("worker".into()), WorkerGroup::spawn(args)?)],
        transfer,
    )?;
    let admission = text_admission(63, 1, 1)?;
    let key = admission.request_key;
    let first_id = CallId::new(9, 0);
    let second_id = CallId::new(10, 0);
    let verify_id = CallId::new(11, 0);
    let resumed_id = CallId::new(12, 0);
    let first = token_batch(
        9,
        1,
        key,
        Some(admission),
        first_id,
        CallKind::Forward(ForwardMode::Prefill),
        &[7],
        BlockId(1),
        0,
    );
    let mut successor = token_batch(
        10,
        2,
        key,
        None,
        second_id,
        CallKind::Forward(ForwardMode::Decode),
        &[0],
        BlockId(1),
        1,
    );
    // Reserve the next row's shape, but obtain its token only from the device
    // relay. No host result is read between the two logical call kinds.
    successor.calls[0].input_token_ids.clear();
    successor.calls[0].predicate = Some(first.calls[0].token_output.clone().unwrap());
    let mut verifier = token_batch(
        11,
        3,
        key,
        None,
        verify_id,
        CallKind::Forward(ForwardMode::Verify),
        &[0, 900, 901],
        BlockId(1),
        2,
    );
    verifier.calls[0].input_token_ids = vec![900, 901];
    verifier.calls[0].predicate = Some(successor.calls[0].token_output.clone().unwrap());
    let mut resumed = token_batch(
        12,
        4,
        key,
        None,
        resumed_id,
        CallKind::Forward(ForwardMode::Decode),
        &[0],
        BlockId(1),
        5,
    );
    resumed.calls[0].input_token_ids.clear();
    resumed.calls[0].predicate = Some(verifier.calls[0].token_output.clone().unwrap());
    // Five initialized cache positions are the verifier's maximum. Its first
    // rejected draft must leave only three positions visible to the next decode.
    resumed.calls[0].coordinates = CallCoordinates {
        logical_position: 3,
        kv_visible_len: 3,
        kv_computed_len: 5,
        flow_step: 0,
    };
    for mut batch in [first, successor, verifier, resumed] {
        let batch_id = batch.batch_id;
        let commands = std::mem::take(&mut batch.commands);
        let call = batch.calls.remove(0);
        let placement = RequestPlacement {
            worker: WorkerId("worker".into()),
            request_pool_idx: None,
            block_tables: batch.block_tables,
            new_cache_pages: batch.new_cache_pages,
            forward: batch.forward,
            latent: None,
            decode: None,
            buffers: batch.buffer_allocations,
        };
        executor.submit(ExecutionBatch::new(
            batch_id,
            vec![(call, placement)],
            commands,
            vec![],
        ))?;
    }
    let mut outputs = std::collections::BTreeMap::new();
    while outputs.len() < 4 {
        let result = poll_logical(&mut executor)?.context("device successor did not complete")?;
        for result in result.results {
            assert_eq!(result.output.status, CallStatus::Ok);
            assert!(
                outputs
                    .insert(result.output.call_id, result.output)
                    .is_none()
            );
        }
    }
    assert_eq!(
        outputs.keys().copied().collect::<Vec<_>>(),
        vec![first_id, second_id, verify_id, resumed_id]
    );
    assert_eq!(outputs[&first_id].committed_tokens, vec![1000]);
    assert_eq!(outputs[&second_id].committed_tokens, vec![1001]);
    assert_eq!(outputs[&second_id].kv_visible_len, 2);
    assert_eq!(outputs[&verify_id].committed_tokens, vec![151_670]);
    assert_eq!(outputs[&verify_id].kv_visible_len, 3);
    assert_eq!(outputs[&verify_id].kv_computed_len, 5);
    assert_eq!(outputs[&resumed_id].committed_tokens, vec![1002]);
    assert_eq!(outputs[&resumed_id].kv_visible_len, 4);
    assert_eq!(outputs[&resumed_id].position, 4);
    executor.close()?;
    Ok(())
}

#[test]
fn failed_producer_retires_waiting_consumers_and_preserves_independent_work() -> anyhow::Result<()>
{
    use uniserve_engine::{ExecutionBatch, RequestPlacement, WorkerExecutor, WorkerId};
    use uniserve_worker_ipc::{BufferAllocation, DiffusionSamplingParams};

    let transfer = uniserve_engine::TransferConfig::parse("worker:1->worker:0=shm")?;
    let mut args = rank_group_args(1 << 20, 8 << 20);
    args.queue_depth = 3;
    args.transfer = transfer.clone();
    args.components = [
        (
            "model".into(),
            uniserve_core::ComponentConfig::parallel(vec![1], Default::default()),
        ),
        (
            "vision_encoder".into(),
            uniserve_core::ComponentConfig::parallel(vec![0], Default::default()),
        ),
    ]
    .into_iter()
    .collect();
    let mut executor = WorkerExecutor::try_new(
        vec![(WorkerId("worker".into()), WorkerGroup::spawn(args)?)],
        transfer,
    )?;
    let bind = |mut batch: Batch, entry: &str| {
        let mut call = batch.calls.remove(0);
        call.component = entry.into();
        ExecutionBatch::new(
            batch.batch_id,
            vec![(
                call,
                RequestPlacement {
                    worker: WorkerId("worker".into()),
                    request_pool_idx: None,
                    block_tables: batch.block_tables,
                    new_cache_pages: batch.new_cache_pages,
                    forward: batch.forward,
                    latent: batch.latent_params.into_iter().next(),
                    decode: batch.decode_ranges.into_iter().next(),
                    buffers: batch.buffer_allocations,
                },
            )],
            batch.commands,
            batch.input_products,
        )
    };
    let key = RequestKey::new(1, RequestId(71), 1);
    let admission = NewRequest::new_media(
        key,
        1,
        vec![7],
        DiffusionSamplingParams {
            num_frames: 22,
            video_units: 1,
            num_inference_steps: 4,
            seed: 1000,
        },
    )?;
    let value = TensorRef {
        request_key: key,
        producer_call_id: CallId::new(1, 0),
        output_index: 0,
        generation: 1,
        dtype: DType::F32,
        shape_bound: ShapeBound {
            dims: vec![DimBound::Static(1)],
        },
    };
    // The component holding the language backbone does not serve vision encoding,
    // which the patch encoder's own component does. Registration succeeds, then
    // the call reports its actual execution error without a product.
    let produce = Call {
        consumer_slots: Vec::new(),
        coordinates: CallCoordinates::default(),
        token_input: None,

        token_output: None,
        vision_input: None,
        latent_feature_input: None,
        encoder_output: None,
        latent_input: None,
        latent_output: None,
        image_input: None,
        image_output: None,
        completion_output: None,
        transition_output: None,

        input_image: None,
        kv_input: None,
        kv_output: None,
        input_token_ids: Vec::new(),
        sampling_state: None,
        request_key: key,
        call_id: CallId::new(1, 0),
        component: "model".into(),
        code: CallKind::Media(MediaCall::VisionEncoding),
        bounds: Bounds::default(),
        inputs: vec![],
        outputs: vec![value.clone()],
        predicate: None,
        rng: None,
    };
    let mut source = Batch::new(1, vec![admission], vec![produce]);
    source.buffer_allocations.push(BufferAllocation {
        buffer: value.buffer_id(),
        offset: 0,
        bytes: value.max_bytes(),
    });
    let logical = bind(source, "model");
    let copied = TensorRef {
        producer_call_id: CallId::new(2, 0),
        generation: 2,
        ..value.clone()
    };
    let consume = Call {
        consumer_slots: Vec::new(),
        coordinates: CallCoordinates::default(),
        token_input: None,

        token_output: None,
        vision_input: None,
        latent_feature_input: None,
        encoder_output: None,
        latent_input: None,
        latent_output: None,
        image_input: None,
        image_output: None,
        completion_output: None,
        transition_output: None,

        input_image: None,
        kv_input: None,
        kv_output: None,
        input_token_ids: Vec::new(),
        sampling_state: None,
        request_key: key,
        call_id: CallId::new(2, 0),
        component: "vision_encoder".into(),
        code: CallKind::Transfer(TransferMode::Tensor),
        bounds: Bounds {
            max_transfer_bytes: value.max_bytes(),
            ..Bounds::default()
        },
        inputs: vec![value.clone()],
        outputs: vec![copied.clone()],
        predicate: None,
        rng: None,
    };
    let mut consumer = Batch::new(2, vec![], vec![consume]);
    consumer.buffer_allocations.push(BufferAllocation {
        buffer: copied.buffer_id(),
        offset: 256,
        bytes: copied.max_bytes(),
    });
    let consumer = bind(consumer, "vision_encoder");
    let independent = text_admission(72, 1, 2)?;
    let independent_key = independent.request_key;
    let independent = bind(
        token_batch(
            3,
            3,
            independent_key,
            Some(independent),
            CallId::new(3, 0),
            CallKind::Forward(ForwardMode::Prefill),
            &[9],
            BlockId(2),
            0,
        ),
        "model",
    );

    executor.submit(logical)?;
    executor.submit(consumer)?;
    executor.submit(independent)?;

    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let mut failed = false;
    let mut source_returned = false;
    let mut batch_done = false;
    let mut independent_returned = false;
    while std::time::Instant::now() < deadline
        && !(failed && source_returned && batch_done && independent_returned)
    {
        match executor.poll(Duration::from_millis(100)) {
            Err(error) => {
                let loss = error.downcast::<WorkerFailure>()?;
                assert!(
                    !failed,
                    "one producer failure must retire its dependents once"
                );
                assert!(loss.endpoints.is_empty());
                assert_eq!(loss.requests, vec![key]);
                assert_eq!(loss.retired, vec![(2, key, CallId::new(2, 0))]);
                assert!(loss.buffers.contains(&value.buffer_id()));
                failed = true;
            }
            Ok(Some(result)) => {
                batch_done |= result.done && result.batch_id == 3;
                for result in result.results {
                    match result.output.call_id {
                        CallId {
                            batch_id: 1,
                            request_index: 0,
                        } => {
                            assert_eq!(result.output.status, CallStatus::Error);
                            assert!(!source_returned);
                            source_returned = true;
                        }
                        CallId {
                            batch_id: 3,
                            request_index: 0,
                        } => {
                            assert_eq!(result.output.status, CallStatus::Ok);
                            assert_eq!(result.output.committed_tokens, vec![1000]);
                            assert!(!independent_returned);
                            independent_returned = true;
                        }
                        computation => panic!("unexpected completed computation {computation:?}"),
                    }
                }
            }
            Ok(None) => {}
        }
    }
    assert!(failed && source_returned && batch_done && independent_returned);
    executor.close()?;
    Ok(())
}

#[test]
fn media_storage_is_owned_through_rank_result_validation() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;

    for case in ["retained", "unknown-call", "rank-output", "short-storage"] {
        let directory = tempfile::tempdir()?;
        let wrapper = directory.path().join("worker");
        let name_path = directory.path().join("media-name");
        let python = serde_json::to_string(&worker_python())?;
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let fixture = serde_json::to_string(&root.join("tests/python/fixtures/rank_media.py"))?;
        // The fixture is run by path, so its own directory leads the import
        // path; the package it shares with the rest of the suite is named here.
        let import_root = serde_json::to_string(&root.canonicalize()?)?;
        let name = serde_json::to_string(&name_path)?;
        std::fs::write(
            &wrapper,
            format!(
                "#!/usr/bin/env python3\nimport os, sys\nenv = dict(os.environ, PYTHONPATH={import_root}, UNISERVE_TEST_MEDIA_RESPONSE={case:?}, UNISERVE_TEST_MEDIA_NAME={name})\nos.execve({python}, [{python}, {fixture}, *sys.argv[3:]], env)\n"
            ),
        )?;
        std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))?;
        let mut args = rank_group_args(1 << 20, 8 << 20);
        args.python = wrapper;
        let mut worker = WorkerGroup::spawn(args)?;
        let admission = text_admission(81, 1, 1)?;
        let batch = token_batch(
            1,
            1,
            admission.request_key,
            Some(admission),
            CallId::new(1, 0),
            CallKind::Forward(ForwardMode::Prefill),
            &[7, 8],
            BlockId(1),
            0,
        );
        worker.submit_batch(batch)?;
        let rejected = matches!(case, "unknown-call" | "rank-output");
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        let mut terminal = false;
        let mut results = Vec::new();
        while std::time::Instant::now() < deadline && !terminal {
            match worker.poll_batch(Duration::from_millis(100)) {
                Ok(Some(report)) => {
                    terminal = true;
                    results.extend(report.results);
                }
                Ok(None) => {}
                Err(error) => {
                    anyhow::ensure!(
                        rejected,
                        "unexpected media result failure in {case}: {error:#}"
                    );
                    terminal = true;
                }
            }
        }
        assert!(terminal, "media result did not retire in {case}");
        let published_name = std::fs::read_to_string(&name_path)?;
        // The receiver claims the POSIX name even when correlation, rank ownership,
        // or extent validation rejects the result. Retained mappings keep the bytes.
        assert!(
            !Path::new("/dev/shm").join(published_name.trim()).exists(),
            "unclaimed media in {case}"
        );
        if rejected {
            assert!(
                results.is_empty(),
                "invalid rank output was delivered in {case}"
            );
        } else {
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].output.status, CallStatus::Ok);
            if case == "short-storage" {
                assert!(results[0].media.is_err());
            } else {
                let media = results[0]
                    .media
                    .as_ref()
                    .map_err(|error| anyhow::anyhow!(error.clone()))?
                    .as_ref()
                    .context("media result has no storage")?
                    .clone();
                worker.close()?;
                drop(results);
                assert_eq!(media.as_bytes(), b"generated media content");
                continue;
            }
        }
        worker.close()?;
    }
    Ok(())
}

#[test]
fn multiprocess_topology_handles_rank_failure_and_capacity_limits() -> anyhow::Result<()> {
    check_rank_ipc()?;
    qualify_slow_transfer()?;
    qualify_peer_replacement()?;
    Ok(())
}

#[test]
fn unsupported_media_is_rejected_without_stopping_the_engine() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::ERROR)
        .try_init();
    let mut config = EngineConfig::sim("stub");
    config.runtime_family = RuntimeFamily::Diffusion;
    config.generation_limits = uniserve_core::GenerationLimits {
        latent_downsample: 1,
        max_cfg_branches: 1,
        ..Default::default()
    };
    config.max_batch = QUEUE_DEPTH * 8;
    config.workers = vec![WorkerConfig::placed(
        &["localhost".to_owned()],
        "cpu",
        WORLD_SIZE,
        2,
        WorkerConfig::single_component("model", WORLD_SIZE),
    )];
    config.worker_process = rank_group_args(128 << 10, 128 << 10);
    let engine = {
        let _launch_guard = CHILD_LAUNCH_ENV_LOCK
            .lock()
            .map_err(|_| anyhow::anyhow!("child-launch environment lock is poisoned"))?;
        EngineCore::new(config)?
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()?;

    let result = (|| -> anyhow::Result<()> {
        let mut requests = Vec::with_capacity(2);
        for index in 0..2 {
            let request_id = RequestId(10_000 + index as u64);
            let prompt_token_ids = vec![100_000 + index as u32];
            let events = engine.submit(Request::Diffusion(DiffusionRequest {
                request_id,
                prompt_token_ids,
                priority: 0,
                sampling: DiffusionSamplingParams {
                    num_frames: 22,
                    video_units: 3,
                    num_inference_steps: 4,
                    seed: index as u64,
                },
            }))?;
            requests.push((request_id, events));
        }
        for (request_id, mut events) in requests {
            let event = runtime
                .block_on(async {
                    tokio::time::timeout(Duration::from_secs(10), events.recv()).await
                })
                .with_context(|| format!("timed out waiting for media request {request_id:?}"))?;
            let event = event.ok_or_else(|| {
                anyhow::anyhow!("media event stream closed for request {request_id:?}")
            })?;
            assert!(
                matches!(event, EngineCoreOutput::Rejected { .. }),
                "unsupported request {request_id:?} was not rejected: {event:?}"
            );
        }
        Ok(())
    })();

    engine.shutdown();
    result?;
    assert!(!engine.is_dead(), "media rejection killed the engine");
    Ok(())
}

fn check_rank_ipc() -> anyhow::Result<()> {
    let mut executor = spawn_rank_group()?;
    let info = executor.info();
    assert_eq!(info.endpoint.rank, 0);
    assert_eq!(info.world_size, WORLD_SIZE as u32);
    assert_eq!(info.max_batch_calls, 256);
    assert_eq!(info.max_batch_tokens, 256);
    assert_eq!(info.queue_depth as usize, QUEUE_DEPTH);

    let mut admission = text_admission(11, 1, 1)?;
    let sampling = &mut admission.ar.as_mut().unwrap().sampling;
    sampling.return_logprobs = true;
    sampling.n_logprobs = 1;
    sampling.return_prompt_logprobs = true;
    sampling.n_prompt_logprobs = 1;
    let mut initial = token_batch(
        1,
        1,
        admission.request_key,
        Some(admission.clone()),
        CallId::new(1, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[7, 8],
        BlockId(1),
        0,
    );
    // One sampled score and two bounded candidate sets: generated and prompt.
    initial.calls[0].bounds.max_completion_bytes = 4 + 2 * 12 + 4 + 2 * 12;
    let first = execute(&mut executor, initial.clone())?;
    let first_record = &first.results[0].output;
    assert_eq!(first_record.status, CallStatus::Ok);
    assert_eq!(first_record.committed_tokens.as_slice().len(), 1);

    assert!(
        first_record
            .sampled_logprob
            .is_some_and(|score| score.is_finite() && score <= 0.0)
    );
    assert_eq!(
        first_record.top_logprobs[0].token_id,
        first_record.committed_tokens[0]
    );
    assert_eq!(first_record.prompt_logprobs.len(), 1);
    assert_eq!(first_record.prompt_logprobs[0][0].token_id, 8);

    assert!(executor.submit_batch(initial).is_err());

    let conflicting = token_batch(
        1,
        1,
        admission.request_key,
        Some(admission.clone()),
        CallId::new(1, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[7, 8, 9],
        BlockId(1),
        0,
    );
    assert!(executor.submit_batch(conflicting).is_err());

    let stale_key = RequestKey::new(
        admission.request_key.engine_id,
        admission.request_key.request_id,
        admission.request_key.request_epoch + 1,
    );
    let stale_finish = BatchCommand::Finish {
        request_key: stale_key,
        retained_buffers: Vec::new(),
    };
    assert!(
        execute(&mut executor, command_batch(2, stale_finish))?
            .results
            .is_empty()
    );
    let mut continuation = token_batch(
        3,
        2,
        admission.request_key,
        None,
        CallId::new(3, 0),
        CallKind::Forward(ForwardMode::Decode),
        &[first_record.committed_tokens.as_slice()[0]],
        BlockId(1),
        2,
    );
    continuation.calls[0].bounds.max_completion_bytes = 4 + 2 * 12;
    let continued = execute(&mut executor, continuation)?;
    assert_eq!(continued.results[0].output.status, CallStatus::Ok);
    assert_eq!(continued.results[0].output.position, 3);
    assert!(continued.results[0].output.sampled_logprob.is_some());
    assert_eq!(
        continued.results[0].output.top_logprobs[0].token_id,
        continued.results[0].output.committed_tokens[0]
    );
    assert!(continued.results[0].output.prompt_logprobs.is_empty());
    let close = BatchCommand::Finish {
        request_key: admission.request_key,
        retained_buffers: Vec::new(),
    };
    let independent = text_admission(12, 1, 2)?;
    let mut close_batch = token_batch(
        6,
        3,
        independent.request_key,
        Some(independent),
        CallId::new(6, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[9, 10],
        BlockId(2),
        0,
    );
    close_batch.commands.push(close.clone());
    executor.submit_batch(close_batch.clone())?;
    // A batch returns one result: the call's completion and the retirement
    // it carries arrive together.
    let result = executor
        .poll_batch(Duration::from_secs(30))?
        .ok_or_else(|| anyhow::anyhow!("independent call did not complete"))?;
    assert_eq!(result.results.len(), 1);
    assert_eq!(result.results[0].output.status, CallStatus::Ok);
    assert!(executor.poll_batch(Duration::from_millis(50))?.is_none());
    assert!(matches!(
        executor.submit_batch(close_batch),
        Err(uniserve_engine::BatchSubmitError::Failed(_))
    ));
    let descendant = token_batch(
        8,
        4,
        admission.request_key,
        None,
        CallId::new(8, 0),
        CallKind::Forward(ForwardMode::Decode),
        &[first_record.committed_tokens.as_slice()[0]],
        BlockId(1),
        2,
    );
    let closed = execute(&mut executor, descendant)?;
    let closed_record = &closed.results[0].output;
    assert_eq!(closed_record.status, CallStatus::Error);
    assert_eq!(closed_record.error_code, Some(ErrorCode::InvalidCall));

    qualify_kv_rank_locations(&mut executor)?;
    executor.close()?;
    assert!(!executor.is_ready());
    assert!(executor.poll_batch(Duration::ZERO)?.is_none());
    assert!(matches!(
        executor.submit_batch(command_batch(
            100,
            BatchCommand::Finish {
                request_key: admission.request_key,
                retained_buffers: Vec::new(),
            }
        )),
        Err(uniserve_engine::BatchSubmitError::Failed(_))
    ));
    executor.close()?;
    Ok(())
}

fn qualify_kv_rank_locations(executor: &mut WorkerGroup) -> anyhow::Result<()> {
    let admission = text_admission(13, 1, 3)?;
    let request_key = admission.request_key;
    let mut initial = token_batch(
        9,
        5,
        request_key,
        Some(admission),
        CallId::new(9, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[7, 8],
        BlockId(3),
        0,
    );
    initial.calls[0].token_output = None;
    let tables = initial.block_tables.clone();
    let first = execute(executor, initial)?;
    assert_eq!(first.results[0].output.status, CallStatus::Ok);
    let buffer = uniserve_worker_ipc::BufferId {
        owner: request_key,
        producer_call_id: CallId::new(10, 0),
        output_index: 0,
        generation: 2,
    };
    let call = Call {
        consumer_slots: Vec::new(),
        coordinates: coordinates_after(&first.results[0].output),
        token_input: None,

        token_output: None,
        vision_input: None,
        latent_feature_input: None,
        encoder_output: None,
        latent_input: None,
        latent_output: None,
        image_input: None,
        image_output: None,
        completion_output: None,
        transition_output: None,

        input_image: None,
        kv_input: None,
        kv_output: Some(buffer),
        input_token_ids: Vec::new(),
        sampling_state: None,
        request_key,
        call_id: CallId::new(10, 0),
        component: "model".into(),
        code: CallKind::Transfer(TransferMode::KvPublish),
        bounds: Bounds {
            max_transfer_bytes: 4096,
            ..Bounds::default()
        },
        inputs: Vec::new(),
        outputs: Vec::new(),
        predicate: None,
        rng: None,
    };
    let mut publish = Batch::new(10, Vec::new(), vec![call]);
    publish.collective_seq = 6;
    publish.block_tables = tables;

    let report = execute(executor, publish.clone())?;
    assert_eq!(report.results[0].output.status, CallStatus::Ok);
    let publication = report.results[0]
        .output
        .kv_output
        .as_ref()
        .context("KV publication has no physical locations")?;
    assert_eq!(publication.source, buffer);
    let tensors = &publication.tensors;
    for tensor in tensors {
        let ranks = tensor
            .locations
            .iter()
            .map(|location| location.source.rank)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(ranks, (0..WORLD_SIZE as u32).collect());
    }
    assert!(execute(executor, publish).is_err());
    Ok(())
}

fn qualify_peer_replacement() -> anyhow::Result<()> {
    // Identify only this test's rank children while the shared launch lock is held.
    let launch_guard = CHILD_LAUNCH_ENV_LOCK
        .lock()
        .map_err(|_| anyhow::anyhow!("child launch lock poisoned"))?;
    let child_ids = || -> anyhow::Result<Vec<u32>> {
        Ok(std::fs::read_to_string("/proc/thread-self/children")?
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()?)
    };
    let before = child_ids()?;
    let mut executor = spawn_rank_group()?;
    let victim = child_ids()?
        .into_iter()
        .find(|pid| {
            if before.contains(pid) {
                return false;
            }
            let args = std::fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
            let args = args.split(|byte| *byte == 0).collect::<Vec<_>>();
            args.windows(2)
                .any(|pair| pair == [b"--rank".as_slice(), b"1".as_slice()])
        })
        .context("rank child was not found")?;
    drop(launch_guard);

    let initial_endpoint = executor.info().endpoint.clone();
    let first_admission = text_admission(21, 1, 1)?;
    let mut first = token_batch(
        1,
        1,
        first_admission.request_key,
        Some(first_admission),
        CallId::new(1, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[3],
        BlockId(1),
        0,
    );
    let finished_admission = text_admission(20, 1, 2)?;
    let finished_key = finished_admission.request_key;
    let mut finished = token_batch(
        1,
        1,
        finished_key,
        Some(finished_admission),
        CallId::new(1, 1),
        CallKind::Forward(ForwardMode::Prefill),
        &[2],
        BlockId(2),
        0,
    );
    finished.forward.call_indices[0] = 1;
    first.calls.extend(finished.calls);
    first.commands.extend(finished.commands);
    first.block_tables.extend(finished.block_tables);
    first.new_cache_pages.extend(finished.new_cache_pages);
    first.forward.append(finished.forward, 0);
    first.input_products.extend(finished.input_products);
    execute(&mut executor, first)?;
    let mut close = Batch::new(2, Vec::new(), Vec::new());
    close.collective_seq = 2;
    close.commands.push(BatchCommand::Finish {
        request_key: finished_key,
        retained_buffers: Vec::new(),
    });
    execute(&mut executor, close)?;

    let lost_admission = text_admission(22, 1, 2)?;
    // Keep this rank from completing the batch before the test terminates it.
    let paused = PausedProcess::new(victim.try_into()?)?;
    executor.submit_batch(token_batch(
        3,
        3,
        lost_admission.request_key,
        Some(lost_admission),
        CallId::new(3, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[4],
        BlockId(2),
        0,
    ))?;
    paused.terminate()?;
    let loss = executor
        .poll_batch(Duration::from_secs(30))
        .expect_err("rank loss must be reported");
    let loss = loss
        .downcast_ref::<WorkerFailure>()
        .context("expected scoped WorkerGroup loss")?;
    assert_eq!(loss.worker_id.0, initial_endpoint.worker_id);
    assert_eq!(loss.endpoints.len(), WORLD_SIZE);
    assert!(loss.endpoints.contains(&initial_endpoint));
    let requests = loss
        .requests
        .iter()
        .map(|request| request.request_id.0)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(requests, [21, 22].into_iter().collect());

    let ready_deadline = std::time::Instant::now() + Duration::from_secs(30);
    while !executor.is_ready() {
        anyhow::ensure!(
            std::time::Instant::now() < ready_deadline,
            "replacement did not become ready"
        );
        executor.poll_batch(Duration::from_millis(100))?;
    }

    let recovered_admission = text_admission(23, 1, 1)?;
    let recovered = execute(
        &mut executor,
        token_batch(
            4,
            4,
            recovered_admission.request_key,
            Some(recovered_admission),
            CallId::new(4, 0),
            CallKind::Forward(ForwardMode::Prefill),
            &[5],
            BlockId(1),
            0,
        ),
    )?;
    assert_eq!(recovered.results[0].output.status, CallStatus::Ok);
    assert_eq!(
        executor.info().endpoint.worker_id,
        initial_endpoint.worker_id
    );
    assert_ne!(
        executor.info().endpoint.incarnation,
        initial_endpoint.incarnation
    );
    assert_ne!(
        executor.info().endpoint.address_space,
        initial_endpoint.address_space
    );
    executor.close()?;

    use std::os::unix::fs::PermissionsExt as _;
    let wrapper = std::env::temp_dir().join(format!("uniserve-member-load-{}", std::process::id()));
    let python = serde_json::to_string(&worker_python())?;
    std::fs::write(
        &wrapper,
        format!(
            "#!/usr/bin/env python3\nimport os, sys\nif sys.argv[sys.argv.index('--rank') + 1] == '1':\n    sys.exit(47)\nos.execv({python}, [{python}, *sys.argv[1:]])\n"
        ),
    )?;
    std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))?;
    let mut args = rank_group_args(1 << 20, 8 << 20);
    args.python = wrapper.clone();
    let startup = WorkerGroup::spawn(args);
    remove_file(wrapper)?;
    anyhow::ensure!(startup.is_err(), "an incomplete rank group became ready");
    Ok(())
}

fn qualify_slow_transfer() -> anyhow::Result<()> {
    let worker = worker_python();
    let config = WorkerProcessArgs {
        stub: true,
        prefill_cuda_graph: false,
        ..WorkerProcessArgs::default()
    };
    let mut executor = WorkerGroup::spawn(WorkerProcessArgs {
        python: worker,
        model: String::new(),
        ranks: WorkerConfig::placed(
            &["localhost".to_owned()],
            "cpu",
            WORLD_SIZE,
            2,
            WorkerConfig::single_component("model", WORLD_SIZE),
        )
        .ranks,
        components: WorkerConfig::single_component("model", WORLD_SIZE),
        queue_depth: QUEUE_DEPTH,
        req_slot_cap: 1 << 20,
        resp_slot_cap: 8 << 20,
        kv_token_capacity: Some(4096),
        block_size: 16,
        max_batch_calls: 256,
        max_batch_tokens: 256,
        attention_backend: uniserve_worker_ipc::AttentionBackend::TorchSdpa,
        transfer: uniserve_engine::TransferConfig::parse("publisher->worker=shm")?,
        ..config
    })?;

    let slow_admission = text_admission(31, 1, 1)?;
    let mut slow = token_batch(
        1,
        1,
        slow_admission.request_key,
        Some(slow_admission.clone()),
        CallId::new(1, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[6],
        BlockId(1),
        0,
    );
    let predicate = TensorRef {
        request_key: slow_admission.request_key,
        producer_call_id: CallId::new(91, 0),
        output_index: 0,
        generation: 91,
        dtype: DType::U8,
        shape_bound: ShapeBound::default(),
    };
    let publication = SlowShmPublication::start()?;
    slow.calls[0].predicate = Some(predicate.clone());
    slow.input_products.push(TensorPublication {
        product: predicate,
        value: publication.descriptor()?,
    });

    let fast_admission = text_admission(32, 1, 2)?;
    let fast = token_batch(
        2,
        2,
        fast_admission.request_key,
        Some(fast_admission),
        CallId::new(2, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[9],
        BlockId(2),
        0,
    );

    executor.submit_batch(slow.clone())?;
    assert!(matches!(
        executor.submit_batch(slow),
        Err(uniserve_engine::BatchSubmitError::Failed(_))
    ));
    executor.submit_batch(fast)?;
    // This worker's component spans both ranks, so its ranks stay inside the
    // same collective and hold one batch in flight. The blocked batch is at the
    // head of the channel, so nothing behind it runs until its dependency is
    // readable, and neither batch completes meanwhile.
    assert!(executor.poll_batch(Duration::from_millis(50))?.is_none());
    assert!(!publication.published.load(Ordering::Acquire));

    // Waiting does not submit the slow batch again. Its original result becomes
    // available once the external publisher makes the dependency readable, and
    // the batch behind it follows in the order the channel delivered them.
    publication.publish()?;
    let first = executor
        .poll_batch(Duration::from_secs(30))?
        .ok_or_else(|| anyhow::anyhow!("blocked submission did not complete"))?;
    assert_eq!(first.batch_id, 1);
    assert_eq!(first.results[0].output.status, CallStatus::Ok);

    let second = executor
        .poll_batch(Duration::from_secs(30))?
        .ok_or_else(|| anyhow::anyhow!("following submission did not complete"))?;
    assert_eq!(second.batch_id, 2);
    assert_eq!(second.results[0].output.status, CallStatus::Ok);

    let admission = text_admission(33, 1, 3)?;
    let input = TensorRef {
        request_key: admission.request_key,
        producer_call_id: CallId::new(92, 0),
        output_index: 0,
        generation: 92,
        dtype: DType::U8,
        shape_bound: ShapeBound {
            dims: vec![DimBound::Static(1)],
        },
    };
    let output = TensorRef {
        producer_call_id: CallId::new(3, 0),
        generation: 1,
        ..input.clone()
    };
    let call = Call {
        consumer_slots: Vec::new(),
        coordinates: CallCoordinates::default(),
        token_input: None,

        token_output: None,
        vision_input: None,
        latent_feature_input: None,
        encoder_output: None,
        latent_input: None,
        latent_output: None,
        image_input: None,
        image_output: None,
        completion_output: None,
        transition_output: None,

        input_image: None,
        kv_input: None,
        kv_output: None,
        input_token_ids: Vec::new(),
        sampling_state: None,
        request_key: admission.request_key,
        call_id: CallId::new(3, 0),
        component: "model".into(),
        code: CallKind::Transfer(TransferMode::Tensor),
        bounds: Bounds {
            max_transfer_bytes: input.max_bytes(),
            ..Bounds::default()
        },
        inputs: vec![input.clone()],
        outputs: vec![output.clone()],
        predicate: None,
        rng: None,
    };
    let mut batch = Batch::new(3, vec![admission], vec![call]);
    batch.buffer_allocations = [(&input, 0), (&output, 256)]
        .into_iter()
        .map(|(product, offset)| uniserve_worker_ipc::BufferAllocation {
            buffer: product.buffer_id(),
            offset,
            bytes: product.max_bytes(),
        })
        .collect();
    batch.input_products.push(TensorPublication {
        product: input,
        value: publication.descriptor()?,
    });
    executor.submit_batch(batch)?;
    let transferred = executor
        .poll_batch(Duration::from_secs(30))?
        .context("tensor input did not reach its component")?;
    assert_eq!(transferred.results[0].output.status, CallStatus::Ok);
    assert_eq!(transferred.products[0].product, output);
    let TransferHandle::DeviceProduct { tensor, .. } = &transferred.products[0].value else {
        anyhow::bail!("tensor output has no physical publication");
    };
    assert_eq!(tensor.shape, vec![1]);
    assert_eq!(tensor.locations.len(), WORLD_SIZE);
    executor.close()?;
    assert!(publication.path.is_file());
    Ok(())
}

/// Independent instances accepting the same computation must retain separate capacity.
#[test]
fn independent_workers_preserve_capacity_retirement_and_failed_work() -> anyhow::Result<()> {
    use uniserve_engine::{
        ExecutionBatch, ExecutorSubmitError, RequestPlacement, WorkerExecutor, WorkerId,
    };

    // Control process scheduling at the OS boundary. This keeps the occupied
    // instance deterministic without forging a product from an unbound source.
    use std::os::unix::fs::PermissionsExt as _;
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let wrapper =
        std::env::temp_dir().join(format!("uniserve-worker-{}-{nonce}", std::process::id()));
    let pid_file = wrapper.with_extension("pid");
    let python = serde_json::to_string(&worker_python())?;
    std::fs::write(
        &wrapper,
        format!(
            "#!/usr/bin/env python3\nimport os, sys\nwith open({}, 'w') as output:\n    output.write(str(os.getpid()))\nos.execv({python}, [{python}, *sys.argv[1:]])\n",
            serde_json::to_string(&pid_file)?,
        ),
    )?;
    std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))?;
    let spawn = |worker_id: &str, depth| -> anyhow::Result<WorkerGroup> {
        let mut args = rank_group_args(1 << 20, 8 << 20);
        let binding = WorkerConfig::placed(
            &["localhost".to_owned()],
            "cpu",
            1,
            depth,
            stub_components(1),
        );
        args.ranks = binding.ranks.clone();
        args.components = binding.components;
        args.queue_depth = depth;
        args.worker_id = worker_id.into();
        if worker_id == "encoder-0" {
            args.python = wrapper.clone();
        }

        args.transfer = uniserve_engine::TransferConfig::parse(
            "encoder-0->encoder-1=shm,encoder-1->encoder-0=shm",
        )?;
        WorkerGroup::spawn(args)
    };
    let mut executor = WorkerExecutor::try_new(
        vec![
            (WorkerId("encoder-0".into()), spawn("encoder-0", 1)?),
            (WorkerId("encoder-1".into()), spawn("encoder-1", 2)?),
        ],
        uniserve_engine::TransferConfig::parse(
            "encoder-0->encoder-1=shm,encoder-1->encoder-0=shm",
        )?,
    )?;
    let pid = std::fs::read_to_string(&pid_file)?.parse::<i32>()?;
    let mut paused = PausedProcess::new(pid)?;
    let bind = |batch: Batch, worker: &str| {
        assert_eq!(batch.calls.len(), 1);
        let call = batch.calls.into_iter().next().unwrap();
        ExecutionBatch::new(
            batch.batch_id,
            vec![(
                call,
                RequestPlacement {
                    worker: WorkerId(worker.into()),
                    request_pool_idx: None,
                    block_tables: batch.block_tables,
                    new_cache_pages: batch.new_cache_pages,
                    forward: batch.forward,
                    latent: batch.latent_params.into_iter().next(),
                    decode: batch.decode_ranges.into_iter().next(),
                    buffers: batch.buffer_allocations,
                },
            )],
            batch.commands,
            batch.input_products,
        )
    };
    let make_batch = |batch_id, request_index, request_id, page| -> anyhow::Result<Batch> {
        let admission = text_admission(request_id, 1, page)?;
        Ok(token_batch(
            batch_id,
            batch_id,
            admission.request_key,
            Some(admission),
            CallId::new(batch_id, request_index),
            CallKind::Forward(ForwardMode::Prefill),
            &[7],
            BlockId(page),
            0,
        ))
    };
    let slow = make_batch(1, 0, 51, 1)?;
    executor.submit(bind(slow, "encoder-0"))?;
    let blocked = bind(make_batch(2, 0, 52, 2)?, "encoder-0");
    let blocked = match executor.submit(blocked) {
        Err(ExecutorSubmitError::WouldBlock(batch)) => batch,
        other => anyhow::bail!("occupied instance did not apply its capacity bound: {other:?}"),
    };
    let independent = make_batch(3, 0, 53, 1)?;
    let token = independent.calls[0].token_output.clone().unwrap();
    executor.submit(bind(independent, "encoder-1"))?;
    let first = poll_logical(&mut executor)?.context("independent worker did not complete")?;
    assert_eq!(first.batch_id, 3);
    assert_eq!(first.results[0].output.status, CallStatus::Ok);
    paused.resume()?;
    let slow = poll_logical(&mut executor)?.context("released worker did not complete")?;
    assert_eq!(slow.batch_id, 1);
    assert_eq!(slow.results[0].output.status, CallStatus::Ok);
    executor.submit(blocked)?;
    let resumed = poll_logical(&mut executor)?.context("capacity was not reusable")?;
    assert_eq!(resumed.batch_id, 2);
    assert_eq!(resumed.results[0].output.status, CallStatus::Ok);
    // Request-relay products have no arena params, but their release must
    // still reach the rank holding the published generation.
    let release = ExecutionBatch::new(
        4,
        Vec::new(),
        vec![BatchCommand::Free {
            buffer: token.buffer_id(),
        }],
        Vec::new(),
    );
    executor.submit(release)?;
    let retired = poll_logical(&mut executor)?.context("product release did not complete")?;
    assert_eq!(retired.batch_id, 4);
    assert!(retired.results.is_empty());

    // The same request owns numerical state on one instance and a transported
    // product on another; Finish must retire both physical owners.
    let shared = make_batch(5, 0, 54, 3)?;
    let admission = shared.admissions().next().unwrap().clone();
    let source = shared.calls[0].token_output.clone().unwrap();
    executor.submit(bind(shared, "encoder-0"))?;
    let produced = poll_logical(&mut executor)?.context("source did not complete")?;
    assert!(produced.done);
    let publication = TensorRef {
        producer_call_id: CallId::new(6, 0),
        generation: 2,
        ..source.clone()
    };
    let publish = Call {
        consumer_slots: Vec::new(),
        coordinates: coordinates_after(&produced.results[0].output),
        token_input: Some(source.clone()),

        token_output: Some(publication.clone()),
        vision_input: None,
        latent_feature_input: None,
        encoder_output: None,
        latent_input: None,
        latent_output: None,
        image_input: None,
        image_output: None,
        completion_output: None,
        transition_output: None,

        input_image: None,
        kv_input: None,
        kv_output: None,
        input_token_ids: Vec::new(),
        sampling_state: None,
        request_key: admission.request_key,
        call_id: CallId::new(6, 0),
        component: "model".into(),
        code: CallKind::Transfer(TransferMode::Tensor),
        bounds: Bounds {
            max_transfer_bytes: source.max_bytes(),
            ..Bounds::default()
        },
        inputs: Vec::new(),
        outputs: Vec::new(),
        predicate: None,
        rng: None,
    };
    let publish = Batch::new(6, Vec::new(), vec![publish]);

    executor.submit(bind(publish, "encoder-0"))?;
    let published = poll_logical(&mut executor)?.context("source publication did not complete")?;
    assert!(published.done);
    assert_eq!(published.results[0].output.status, CallStatus::Ok);
    let copy = TensorRef {
        producer_call_id: CallId::new(7, 0),
        generation: 3,
        ..publication.clone()
    };
    let transfer = Call {
        consumer_slots: Vec::new(),
        coordinates: CallCoordinates::default(),
        token_input: Some(publication.clone()),

        token_output: Some(copy.clone()),
        vision_input: None,
        latent_feature_input: None,
        encoder_output: None,
        latent_input: None,
        latent_output: None,
        image_input: None,
        image_output: None,
        completion_output: None,
        transition_output: None,

        input_image: None,
        kv_input: None,
        kv_output: None,
        input_token_ids: Vec::new(),
        sampling_state: None,
        request_key: admission.request_key,
        call_id: CallId::new(7, 0),
        component: "model".into(),
        code: CallKind::Transfer(TransferMode::Tensor),
        bounds: Bounds {
            max_transfer_bytes: publication.max_bytes(),
            ..Bounds::default()
        },
        inputs: Vec::new(),
        outputs: Vec::new(),
        predicate: None,
        rng: None,
    };
    executor.submit(bind(
        Batch::new(7, vec![admission.clone()], vec![transfer]),
        "encoder-1",
    ))?;
    let copied = poll_logical(&mut executor)?.context("auxiliary transfer did not complete")?;
    assert!(copied.done);
    assert_eq!(copied.results[0].output.status, CallStatus::Ok);
    let image_bytes = b"iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAGUlEQVR4nGN0SGhgIAUwkaR6VMOohiGlAQCjvQFA6eri4wAAAABJRU5ErkJggg==".to_vec();
    let feature = TensorRef {
        request_key: admission.request_key,
        producer_call_id: CallId::new(8, 0),
        output_index: 0,
        generation: 4,
        dtype: DType::BF16,
        shape_bound: ShapeBound {
            dims: vec![DimBound::Device { max: 4096 }],
        },
    };
    let encode = Call {
        consumer_slots: Vec::new(),
        coordinates: CallCoordinates::default(),
        token_input: None,

        token_output: None,
        vision_input: None,
        latent_feature_input: None,
        encoder_output: Some(feature.clone()),
        latent_input: None,
        latent_output: None,
        image_input: None,
        image_output: None,
        completion_output: None,
        transition_output: None,

        kv_input: None,
        kv_output: None,
        input_image: Some(String::from_utf8(image_bytes).unwrap().into()),
        input_token_ids: Vec::new(),
        sampling_state: None,
        request_key: admission.request_key,
        call_id: CallId::new(8, 0),
        component: "vision_encoder".into(),
        code: CallKind::Media(MediaCall::VisionEncoding),
        bounds: Bounds {
            max_tokens: 64,
            max_latent_bytes: 8192,
            ..Bounds::default()
        },
        inputs: Vec::new(),
        outputs: Vec::new(),
        predicate: None,
        rng: None,
    };
    let mut encode = Batch::new(8, Vec::new(), vec![encode]);
    encode
        .buffer_allocations
        .push(uniserve_worker_ipc::BufferAllocation {
            buffer: feature.buffer_id(),
            offset: 0,
            bytes: 8192,
        });
    executor.submit(bind(encode, "encoder-1"))?;
    let encoded = poll_logical(&mut executor)?.context("encoder product did not complete")?;
    assert_eq!(encoded.results[0].output.status, CallStatus::Ok);
    let finish = BatchCommand::Finish {
        request_key: admission.request_key,
        retained_buffers: vec![feature.buffer_id()],
    };
    executor.submit(ExecutionBatch::new(9, Vec::new(), vec![finish], Vec::new()))?;
    let closed = poll_logical(&mut executor)?.context("shared request did not retire")?;
    assert!(closed.done);
    for (batch_id, request_id, worker) in [(10, 55, "encoder-0"), (11, 56, "encoder-1")] {
        executor.submit(bind(make_batch(batch_id, 0, request_id, 3)?, worker))?;
        let reused = poll_logical(&mut executor)?.context("retired slot was not reusable")?;
        assert!(reused.done);
        assert_eq!(reused.results[0].output.status, CallStatus::Ok);
    }

    // Closing one request leaves admitted work on both instances independent.
    let closing_key = RequestKey::new(1, RequestId(56), 1);
    let active_key = RequestKey::new(1, RequestId(61), 1);
    let mut mixed = bind(make_batch(12, 0, 57, 4)?, "encoder-0");
    let affected = bind(make_batch(12, 1, 61, 6)?, "encoder-1");
    mixed.requests.extend(affected.requests);
    mixed.commands.extend(affected.commands);
    mixed.input_transfers.extend(affected.input_transfers);
    mixed.commands.push(BatchCommand::Finish {
        request_key: closing_key,
        retained_buffers: Vec::new(),
    });
    executor.submit(mixed)?;
    let mut completed = false;
    let mut tokens = 0;
    while !completed {
        let report = poll_logical(&mut executor)?.context("mixed closure did not complete")?;
        assert_eq!(report.batch_id, 12);
        for result in report.results {
            assert_eq!(result.output.status, CallStatus::Ok);
            tokens += result.output.committed_tokens.as_slice().len();
        }
        if report.done {
            assert_eq!(report.command_results.len(), 1);
            assert_eq!(
                report.command_results[0].outcome,
                uniserve_engine::CommandOutcome::Applied
            );
            completed = true;
        }
    }
    assert_eq!(tokens, 2);
    executor.submit(ExecutionBatch::new(
        13,
        Vec::new(),
        vec![
            BatchCommand::Finish {
                request_key: closing_key,
                retained_buffers: Vec::new(),
            },
            BatchCommand::Finish {
                request_key: active_key,
                retained_buffers: Vec::new(),
            },
        ],
        Vec::new(),
    ))?;
    assert!(
        poll_logical(&mut executor)?
            .context("request close did not retire")?
            .done
    );
    executor.submit(bind(make_batch(14, 0, 58, 3)?, "encoder-1"))?;
    let reused = poll_logical(&mut executor)?.context("closed request slot was not reusable")?;
    assert_eq!(reused.results[0].output.status, CallStatus::Ok);

    // The retained product's separate lifetime survives request slot reuse.
    let next = text_admission(59, 1, 5)?;
    let retained = Call {
        consumer_slots: Vec::new(),
        coordinates: CallCoordinates::default(),
        token_input: None,
        token_output: None,
        vision_input: Some(feature.clone()),
        latent_feature_input: None,
        encoder_output: Some(TensorRef {
            request_key: next.request_key,
            producer_call_id: CallId::new(15, 0),
            ..feature.clone()
        }),
        latent_input: None,
        latent_output: None,
        image_input: None,
        image_output: None,
        completion_output: None,
        transition_output: None,

        input_image: None,
        kv_input: None,
        kv_output: None,
        input_token_ids: Vec::new(),
        sampling_state: None,
        request_key: next.request_key,
        call_id: CallId::new(15, 0),
        component: "model".into(),
        code: CallKind::Transfer(TransferMode::Tensor),
        bounds: Bounds {
            max_transfer_bytes: feature.max_bytes(),
            max_latent_bytes: feature.max_bytes(),
            ..Bounds::default()
        },
        inputs: Vec::new(),
        outputs: Vec::new(),
        predicate: None,
        rng: None,
    };
    let mut retained = Batch::new(15, vec![next], vec![retained]);
    // The source product lives at offset zero on encoder-1. Its destination
    // address on encoder-0 is independent and must not collide with the new
    // output that this call writes at offset zero in encoder-0's address space.
    retained.buffer_allocations.extend([
        uniserve_worker_ipc::BufferAllocation {
            buffer: retained.calls[0]
                .encoder_output
                .as_ref()
                .unwrap()
                .buffer_id(),
            offset: 0,
            bytes: 8192,
        },
        uniserve_worker_ipc::BufferAllocation {
            buffer: feature.buffer_id(),
            offset: 8192,
            bytes: 8192,
        },
    ]);
    executor.submit(bind(retained, "encoder-0"))?;
    let retained = poll_logical(&mut executor)?.context("retained product was not readable")?;
    assert_eq!(retained.results[0].output.status, CallStatus::Ok);
    executor.submit(bind(make_batch(17, 0, 62, 6)?, "encoder-1"))?;
    let fresh =
        poll_logical(&mut executor)?.context("WorkerGroup did not accept work after closure")?;
    assert_eq!(fresh.results[0].output.status, CallStatus::Ok);

    executor.close()?;
    remove_file(&wrapper)?;
    remove_file(&pid_file)?;
    Ok(())
}

/// Progress wakes may precede the last destination's result; preserve one deadline.
fn poll_logical(
    executor: &mut impl Executor,
) -> anyhow::Result<Option<uniserve_engine::BatchResult>> {
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        let Some(remaining) = deadline.checked_duration_since(std::time::Instant::now()) else {
            return Ok(None);
        };
        if let Some(result) = executor.poll(remaining)? {
            return Ok(Some(result));
        }
    }
}

struct PausedProcess(Option<i32>);

impl PausedProcess {
    fn new(pid: i32) -> anyhow::Result<Self> {
        anyhow::ensure!(pid > 0, "worker process id must be positive");
        anyhow::ensure!(
            unsafe { libc::kill(pid, libc::SIGSTOP) } == 0,
            "failed to pause the selected worker: {}",
            std::io::Error::last_os_error()
        );
        Ok(Self(Some(pid)))
    }

    fn resume(&mut self) -> anyhow::Result<()> {
        if let Some(pid) = self.0 {
            anyhow::ensure!(
                unsafe { libc::kill(pid, libc::SIGCONT) } == 0,
                "failed to resume the selected worker: {}",
                std::io::Error::last_os_error()
            );
            self.0 = None;
        }
        Ok(())
    }

    fn terminate(mut self) -> anyhow::Result<()> {
        let pid = self.0.context("selected worker is no longer paused")?;
        anyhow::ensure!(
            unsafe { libc::kill(pid, libc::SIGKILL) } == 0,
            "failed to terminate the selected worker: {}",
            std::io::Error::last_os_error()
        );
        self.0 = None;
        Ok(())
    }
}

impl Drop for PausedProcess {
    fn drop(&mut self) {
        let _ = self.resume();
    }
}

/// An external shared-memory publisher the test controls.
///
/// The segment carries what a consumer needs in its header: the digest of the
/// locator that names it, and a readiness word the test writes after the
/// payload. Until then a rank that reads it waits on that word.
struct SlowShmPublication {
    name: String,
    endpoint: String,
    path: PathBuf,
    published: Arc<AtomicBool>,
}

/// Byte layout of a shared-memory publication's header, as the worker's
/// `transfer.segment` module lays it out.
const SEGMENT_HEADER_BYTES: u64 = 512;
const SEGMENT_STATE_OFFSET: u64 = 32;
const SEGMENT_READY: u32 = 1;

impl SlowShmPublication {
    fn start() -> anyhow::Result<Self> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let name = format!("uniserve-transfer-{}-{nonce}", std::process::id());
        let path = Path::new("/dev/shm").join(&name);
        let endpoint = format!("{name}-readers");
        let publication = Self {
            name,
            endpoint,
            path,
            published: Arc::new(AtomicBool::new(false)),
        };
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&publication.path)?;
        file.set_len(SEGMENT_HEADER_BYTES + 1)?;
        // The header names the exact locator the rank will be handed, the way
        // the worker's own publications do, so the rank accepts the segment.
        file.write_all_at(&publication.digest()?, 0)?;
        file.write_all_at(&[0], SEGMENT_HEADER_BYTES)?;
        Ok(publication)
    }

    /// The digest the worker computes over a locator: SHA-256 of its
    /// canonical JSON, keys sorted, no whitespace.
    fn digest(&self) -> anyhow::Result<[u8; 32]> {
        use sha2::Digest as _;

        let canonical = format!(
            concat!(
                "{{\"device\":\"cpu\",\"dtype\":\"uint8\",\"endpoint\":\"{endpoint}\",",
                "\"name\":\"{name}\",\"nbytes\":1,\"offset\":[0],\"shape\":[1],",
                "\"source\":{{\"address_space\":\"{address_space}\",",
                "\"incarnation\":\"{incarnation}\",\"node\":\"{node}\",\"rank\":0,",
                "\"worker_id\":\"publisher\"}},\"transport\":\"posix_shm\"}}"
            ),
            endpoint = self.endpoint,
            name = self.name,
            address_space = self.address_space(),
            incarnation = self.endpoint,
            node = self.node()?,
        );
        Ok(sha2::Sha256::digest(canonical.as_bytes()).into())
    }

    fn address_space(&self) -> String {
        format!("rust:{}", std::process::id())
    }

    fn node(&self) -> anyhow::Result<String> {
        Ok(std::fs::read_to_string("/etc/hostname")?.trim().to_owned())
    }

    fn publish(&self) -> anyhow::Result<()> {
        let file = OpenOptions::new().write(true).open(&self.path)?;
        // The payload lands before the readiness word; each write is a system
        // call, which orders them for a reader on another core.
        file.write_all_at(&[1], SEGMENT_HEADER_BYTES)?;
        file.write_all_at(&SEGMENT_READY.to_ne_bytes(), SEGMENT_STATE_OFFSET)?;
        self.published.store(true, Ordering::Release);
        Ok(())
    }

    fn descriptor(&self) -> anyhow::Result<TransferHandle> {
        Ok(TransferHandle::DeviceProduct {
            height: 0,
            width: 0,
            value_range: String::new(),
            tensor: uniserve_worker_ipc::TensorTransfer {
                shape: vec![1],
                locations: vec![Locator {
                    source: uniserve_worker_ipc::WorkerEndpoint {
                        worker_id: "publisher".into(),
                        rank: 0,
                        node: self.node()?,
                        address_space: self.address_space(),
                        incarnation: self.endpoint.clone(),
                    },
                    transport: TransferTransport::PosixShm {
                        endpoint: self.endpoint.clone(),
                        name: self.name.clone(),
                    },
                    nbytes: 1,
                    dtype: "uint8".to_owned(),
                    shape: vec![1],
                    offset: vec![0],
                    device: "cpu".to_owned(),
                }],
            },
        })
    }
}

impl Drop for SlowShmPublication {
    fn drop(&mut self) {
        let _ = remove_file(&self.path);
    }
}

fn spawn_rank_group() -> anyhow::Result<WorkerGroup> {
    spawn_rank_group_with_capacities(1 << 20, 8 << 20)
}

fn spawn_rank_group_with_capacities(
    request_slot_capacity: usize,
    response_slot_capacity: usize,
) -> anyhow::Result<WorkerGroup> {
    WorkerGroup::spawn(rank_group_args(
        request_slot_capacity,
        response_slot_capacity,
    ))
}

/// The components the stub model declares, over `rank_count` ranks.
///
/// A placement names what a worker holds, and a worker that serves a model's
/// whole computation holds every component it declares.
fn stub_components(
    rank_count: usize,
) -> std::collections::BTreeMap<String, uniserve_core::ComponentConfig> {
    let members: Vec<_> = (0..rank_count).collect();
    std::collections::BTreeMap::from([
        (
            "model".into(),
            uniserve_core::ComponentConfig::parallel(
                members.clone(),
                uniserve_core::ParallelConfig {
                    tensor_parallel_size: rank_count,
                    ..Default::default()
                },
            ),
        ),
        (
            "vision_encoder".into(),
            uniserve_core::ComponentConfig::parallel(members, Default::default()),
        ),
    ])
}

fn rank_group_args(
    request_slot_capacity: usize,
    response_slot_capacity: usize,
) -> WorkerProcessArgs {
    let worker = worker_python();
    let config = WorkerProcessArgs {
        stub: true,
        prefill_cuda_graph: false,
        ..WorkerProcessArgs::default()
    };
    WorkerProcessArgs {
        python: worker,
        model: String::new(),
        ranks: WorkerConfig::placed(
            &["localhost".to_owned()],
            "cpu",
            WORLD_SIZE,
            2,
            WorkerConfig::single_component("model", WORLD_SIZE),
        )
        .ranks,
        components: WorkerConfig::single_component("model", WORLD_SIZE),
        queue_depth: QUEUE_DEPTH,
        req_slot_cap: request_slot_capacity,
        resp_slot_cap: response_slot_capacity,
        kv_token_capacity: Some(4096),
        block_size: 16,
        max_batch_calls: 256,
        max_batch_tokens: 256,
        attention_backend: uniserve_worker_ipc::AttentionBackend::TorchSdpa,
        transfer: Default::default(),
        ..config
    }
}

/// Builds the launch descriptor for a weightless CPU worker used by IPC tests.
///
/// Production launches receive the same descriptor from the engine; these tests
/// state one directly because they drive the worker process without a group.
fn stub_launch_descriptor(registration: &str) -> serde_json::Value {
    const TEMPLATE: &str = r#"{
    "registration_address": "__REGISTRATION__",
    "channel_transport": "iceoryx2",
    "acknowledgment_slot": 0,
    "host_slots": [0],
    "product_consumers": [],
    "products_cross_hosts": false,
    "worker_id": "worker",
    "queue_depth": 2,
    "ipc_payload_cap": 1048576,
    "queue_depth": 8,
    "model": "",
    "device": "cpu",
    "rank": 0,
    "local_rank": 0,
    "world_size": 1,
    "components": {
        "model": {
            "ranks": [
                0
            ]
        }
    },
    "supported_calls": null,
    "transfer_backends": "local",
    "publish_backends": "local",
    "distributed_init_method": null,
    "distributed_backend": null,
    "mesh": null,
    "lane": [],
    "no_model": true,
    "allow_stub": true,
    "load_format": "auto",
    "download_dir": null,
    "load_threads": null,
    "checksum_manifest": null,
    "model_dtype": "bfloat16",
    "quantization_config": {},
    "kv_cache_dtype": null,
    "kv_memory_fraction": 0.7,
    "kv_token_capacity": 4096,
    "attention_backend": "torch_sdpa",
    "block_size": 16,
    "max_batch_calls": 8,
    "max_batch_tokens": 256,
    "max_model_len": 8192,
    "max_video_seconds": 15.0,
    "graph_policy": "off",
    "decode_graph_batch_sizes": null,
    "prefill_cuda_graph": false,
    "prefill_graph_token_sizes": null,
    "flow_graph_batch_sizes": null,
    "flow_graph_shapes": null,
    "video_graph_shapes": null,
    "flashinfer_workspace_size": 536870912,
    "flashinfer_use_tensor_core": null,
    "flashinfer_decode_backend": "fa2",
    "flashinfer_prefill_backend": "auto",
    "flashinfer_decode_split_tile_size": null,
    "flashinfer_prefill_split_tile_size": null,
    "flashinfer_disable_split_kv": false
}"#;
    let mut value: serde_json::Value =
        serde_json::from_str(TEMPLATE).expect("stub launch descriptor template");
    value["registration_address"] = serde_json::Value::String(registration.to_owned());
    value
}

fn worker_python() -> PathBuf {
    std::env::var_os("UNISERVE_WORKER_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(".venv/bin/python")
        })
}

fn execute(
    executor: &mut WorkerGroup,
    batch: Batch,
) -> anyhow::Result<uniserve_engine::WorkerResult> {
    executor.submit_batch(batch)?;
    executor
        .poll_batch(Duration::from_secs(30))?
        .ok_or_else(|| anyhow::anyhow!("submission did not complete"))
}

fn text_admission(
    request_id: u64,
    request_epoch: u64,
    request_pool_idx: u32,
) -> anyhow::Result<NewRequest> {
    Ok(NewRequest::new(
        RequestKey::new(1, RequestId(request_id), request_epoch),
        request_pool_idx,
        Some(ArRequestParams {
            sampling: SamplingParams {
                temperature: 0.0,
                ignore_eos: true,
                ..SamplingParams::default()
            },
            negative_token_ids: Vec::new(),
            finish_token_ids: Vec::new(),
            initial_position: 0,
        }),
        None,
    )?)
}

#[allow(clippy::too_many_arguments)]
/// The coordinates a call entering after `output` executes at, as the engine
/// derives them from the completion it observed.
fn coordinates_after(output: &uniserve_worker_ipc::RequestOutput) -> CallCoordinates {
    CallCoordinates {
        logical_position: output.position,
        kv_visible_len: output.kv_visible_len,
        kv_computed_len: output.kv_computed_len,
        flow_step: output.num_completed_steps,
    }
}

fn token_batch(
    batch_id: u64,
    collective_seq: u64,
    request_key: RequestKey,
    admission: Option<NewRequest>,
    call_id: CallId,
    mode: CallKind,
    tokens: &[u32],
    page: BlockId,
    prefix_length: u32,
) -> Batch {
    let request_pool_idx = admission.as_ref().map_or(1, |value| value.request_pool_idx);
    let token_output = TensorRef {
        request_key,
        producer_call_id: call_id,
        output_index: 0,
        generation: (call_id.batch_id as u32)
            .saturating_mul(4)
            .saturating_add(1),
        dtype: DType::I64,
        shape_bound: ShapeBound::default(),
    };
    let call = Call {
        consumer_slots: Vec::new(),
        // These requests hold no cached prefix and no image positions, so the
        // logical sequence and the initialized cache advance together.
        coordinates: CallCoordinates {
            logical_position: prefix_length,
            kv_visible_len: prefix_length,
            kv_computed_len: prefix_length,
            flow_step: 0,
        },
        token_input: None,

        token_output: Some(token_output),
        vision_input: None,
        latent_feature_input: None,
        encoder_output: None,
        latent_input: None,
        latent_output: None,
        image_input: None,
        image_output: None,
        completion_output: None,
        transition_output: None,

        input_image: None,
        kv_input: None,
        kv_output: None,
        input_token_ids: tokens.to_vec(),
        sampling_state: None,
        request_key,
        call_id,
        component: "model".into(),
        code: mode,
        bounds: Bounds {
            max_tokens: tokens.len().max(1) as u32,
            max_kv_pages: u32::from(prefix_length == 0),
            ..Bounds::default()
        },
        inputs: Vec::new(),
        outputs: Vec::new(),
        predicate: None,
        rng: None,
    };
    let input_length = tokens.len() as u32;
    let mut batch = Batch::new(batch_id, admission.into_iter().collect(), vec![call]);
    batch.collective_seq = collective_seq;
    batch.block_tables = vec![BlockTable {
        request_pool_idx,
        group_id: 0,
        page_ids: vec![page],
        allocated_tokens: prefix_length + input_length,
    }];
    batch.new_cache_pages = if prefix_length == 0 {
        vec![CachePageAllocation {
            request_pool_idx,
            group_id: 0,
            page_ids: vec![page],
        }]
    } else {
        Vec::new()
    };
    batch.forward = ForwardBatch {
        call_indices: vec![0],
        request_pool_indices: vec![request_pool_idx],
        seq_lens: vec![prefix_length + input_length.max(1)],
        query_lens: vec![input_length.max(1)],
        write_kv: vec![true],
    };
    batch
}

fn command_batch(batch_id: u64, command: BatchCommand) -> Batch {
    Batch::new(batch_id, Vec::new(), Vec::new()).with_commands(vec![command])
}

#[test]
fn placement_binds_ranks_to_the_engine_host_by_name() {
    let named = WorkerConfig::placed(
        &["compute-0".to_owned()],
        "cpu",
        WORLD_SIZE,
        2,
        WorkerConfig::single_component("model", WORLD_SIZE),
    );
    assert!(
        named.ranks.iter().all(|rank| rank.node == "compute-0"),
        "the placement shorthand must place ranks on the host it is given"
    );

    // A rank placed on another host is started by that host's launcher, which
    // connects to the head and presents the host it owns. No launcher presents
    // here, so the launch fails naming the host whose ranks never started
    // rather than waiting for them or starting them in the wrong place.
    let spawned = WorkerGroup::spawn(WorkerProcessArgs {
        python: worker_python(),
        model: String::new(),
        host: "compute-0".into(),
        ranks: WorkerConfig::placed(
            &["compute-1".to_owned()],
            "cpu",
            WORLD_SIZE,
            2,
            WorkerConfig::single_component("model", WORLD_SIZE),
        )
        .ranks,
        components: named.components.clone(),
        stub: true,
        launcher_timeout: std::time::Duration::from_secs(2),
        ..WorkerProcessArgs::default()
    });
    let Err(error) = spawned else {
        panic!("a rank whose host never presented a launcher must not launch");
    };
    let message = format!("{error:#}");
    assert!(
        message.contains("compute-1"),
        "the refusal must name the host whose launcher never presented: {message}"
    );
}
