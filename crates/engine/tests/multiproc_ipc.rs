//! Multiprocess framing, replay idempotency, and physical-rank recovery.

#![cfg(target_os = "linux")]

use std::fs::{OpenOptions, remove_file};
use std::io::Write as _;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::fs::FileExt;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Context as _;
use uniserve_core::{
    BlockId, DiffusionRequest, Event, MediaGeometry, Request, RequestId, RuntimeFamily,
    SamplingParams,
};
use uniserve_engine::{
    EngineConfig, EngineCore, Executor, RuntimeProfile, Worker, WorkerConfig, WorkerFailure,
    WorkerProcessArgs,
};
use uniserve_worker_ipc::{
    ArRequestParams, BatchCommand, BlockTable, Bounds, CachePageAllocation, Checkpoint,
    CheckpointPoint, CloseReason, DType, DimBound, Disposition, ErrorCode, InlineValue, Locator,
    NewRequest, OpCode, OpId, OpPayload, OpStatus, Operation, PointRange, ProductKind,
    ProductPayload, ProductRef, RequestKey, RowGeometry, Run as Batch, ShapeBound, StorageClass,
    TransferHandle, TransferTransport, encode_token_product_bytes,
};

const WORLD_SIZE: usize = 2;
const PIPELINE_DEPTH: usize = 2;
static CHILD_LAUNCH_ENV_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn independent_entries_complete_on_their_assigned_ranks() -> anyhow::Result<()> {
    let mut args = rank_group_args(1 << 20, 8 << 20);
    args.entries = [
        (
            "model".into(),
            uniserve_core::EntryConfig::parallel(vec![1], Default::default()),
        ),
        (
            "output".into(),
            uniserve_core::EntryConfig::parallel(vec![0], Default::default()),
        ),
    ]
    .into_iter()
    .collect();
    let mut worker = Worker::spawn(args)?;
    let first = text_admission(51, 1, 1)?;
    let second = text_admission(52, 1, 2)?;
    let first_key = first.request_key;
    let second_key = second.request_key;
    let mut run = token_batch(
        1,
        1,
        first_key,
        Some(first),
        OpId(1),
        Checkpoint::admission_root(OpId(0)),
        OpCode::ArExtend,
        &[7, 8],
        0,
        BlockId(1),
        0,
    );
    let mut other = token_batch(
        1,
        1,
        second_key,
        Some(second),
        OpId(2),
        Checkpoint::admission_root(OpId(0)),
        OpCode::ArExtend,
        &[9, 10],
        0,
        BlockId(2),
        0,
    );
    other.operations[0].entry = "output".into();
    for row in &mut other.forward_rows {
        row.operation_index += 1;
    }
    run.operations.extend(other.operations);
    run.commands.extend(other.commands);
    run.input_products.extend(other.input_products);
    run.forward_rows.extend(other.forward_rows);
    run.block_tables.extend(other.block_tables);
    run.new_cache_pages.extend(other.new_cache_pages);
    let starts = std::mem::take(&mut run.commands);
    let admission = execute(
        &mut worker,
        Batch::new(1, vec![], vec![]).with_commands(starts),
    )?;
    assert!(admission.done && admission.completions.is_empty());
    run.batch_id = 2;
    run.run_id = 2;
    run.collective_seq = 2;
    worker.submit_run(run)?;
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    let mut completed = std::collections::BTreeSet::new();
    let mut checkpoints = std::collections::HashMap::new();
    let mut terminal = false;
    while std::time::Instant::now() < deadline && !terminal {
        if let Some(report) = worker.poll_run(Duration::from_millis(100))? {
            for completion in report.completions {
                assert_eq!(completion.status, OpStatus::Ok);
                assert_eq!(completion.committed_tokens().len(), 1);
                assert!(completed.insert(completion.op_id.0));
                checkpoints.insert(completion.request_key, fixed_completion(&completion));
            }
            terminal = report.done;
        }
    }
    assert!(terminal, "independent entry work did not retire");
    assert_eq!(completed, [1, 2].into_iter().collect());

    // Semantic completion selects the entry's state; other ranks release their request storage.
    worker.submit_run(
        Batch::new(3, vec![], vec![]).with_commands(
            [first_key, second_key]
                .into_iter()
                .map(|request_key| BatchCommand::Finish {
                    request_key,
                    control_seq: 1,
                    cutoff: checkpoints[&request_key].clone(),
                    reason: CloseReason::Completed,
                    retained_buffers: vec![],
                })
                .collect(),
        ),
    )?;
    let report = worker
        .poll_run(Duration::from_secs(5))?
        .context("request release did not retire")?;
    assert!(report.done);
    assert!(report.completions.is_empty());
    worker.close()?;
    Ok(())
}

#[test]
fn entries_transfer_published_values_within_one_worker() -> anyhow::Result<()> {
    use uniserve_engine::{Batch as LogicalBatch, Op, WorkerExecutor, WorkerId};

    for transfer in [
        uniserve_engine::TransportMap::parse("worker:1->worker:0=shm")?,
        uniserve_engine::TransportMap::default(),
    ] {
        let mut args = rank_group_args(1 << 20, 8 << 20);
        args.entries = [
            (
                "model".into(),
                uniserve_core::EntryConfig::parallel(vec![1], Default::default()),
            ),
            (
                "output".into(),
                uniserve_core::EntryConfig::parallel(vec![0], Default::default()),
            ),
        ]
        .into_iter()
        .collect();
        let transfer = transfer.with_worker_defaults(&[WorkerConfig {
            id: WorkerId("worker".into()),
            ranks: args.ranks.clone(),
            entries: args.entries.clone(),
            queue_depth: args.pipeline_depth,
        }])?;
        args.transfer = transfer.clone();
        let mut executor = WorkerExecutor::try_new(
            vec![(WorkerId("worker".into()), Worker::spawn(args)?)],
            transfer,
        )?;
        let bind = |batch: Batch, entry: &str| {
            LogicalBatch::new(
                batch.batch_id,
                vec![Op {
                    block_tables: batch.block_tables,
                    new_cache_pages: batch.new_cache_pages,
                    forward_rows: batch.forward_rows,
                    latent: batch.latent_params.into_iter().next(),
                    decode: batch.decode_ranges.into_iter().next(),
                    buffers: batch.buffer_allocations,
                    ..Op::new(
                        batch.operations[0].clone(),
                        (WorkerId("worker".into()), entry.into()),
                    )
                }],
                batch.commands,
                batch.input_products,
            )
        };
        let admission = text_admission(61, 1, 1)?;
        let root = Checkpoint::admission_root(OpId(0));
        let source = token_batch(
            1,
            1,
            admission.request_key,
            Some(admission.clone()),
            OpId(1),
            root.clone(),
            OpCode::ArExtend,
            &[7],
            0,
            BlockId(1),
            0,
        );
        let value = source.operations[0].outputs()[0].clone();
        executor.submit(bind(source, "model"))?;
        let result = poll_logical(&mut executor)?.context("source entry did not complete")?;
        assert_eq!(result.results[0].output.committed_tokens(), &[1000]);
        let cutoff = fixed_completion(&result.results[0].output);
        let publication = ProductRef {
            producer_op_id: OpId(2),
            generation: 2,
            ..value.clone()
        };
        let publish = Operation {
            request_key: admission.request_key,
            op_id: OpId(2),
            parent: None,
            entry: "model".into(),
            payload: OpPayload::new(
                OpCode::TransferProduct,
                Bounds {
                    max_transfer_bytes: value.max_bytes(),
                    ..Bounds::default()
                },
                vec![value],
                vec![publication.clone()],
                None,
                None,
                0,
            ),
        };
        let publish =
            Batch::new(2, vec![], vec![publish]).with_commands(vec![BatchCommand::Commit {
                request_key: admission.request_key,
                control_seq: 1,
                expected_parent: root.clone(),
                selected: cutoff,
                public_event_limit: 0,
                disposition: Disposition::Retain,
            }]);
        executor.submit(bind(publish, "model"))?;
        let published =
            poll_logical(&mut executor)?.context("source publication did not complete")?;
        assert_eq!(published.results[0].output.status, OpStatus::Ok);

        let copy = ProductRef {
            producer_op_id: OpId(3),
            generation: 3,
            ..publication.clone()
        };
        let consume = Operation {
            request_key: admission.request_key,
            op_id: OpId(3),
            parent: None,
            entry: "output".into(),
            payload: OpPayload::new(
                OpCode::TransferProduct,
                Bounds {
                    max_transfer_bytes: publication.max_bytes(),
                    ..Bounds::default()
                },
                vec![publication],
                vec![copy.clone()],
                None,
                None,
                0,
            ),
        };
        executor.submit(bind(Batch::new(3, vec![], vec![consume]), "output"))?;
        let copied = poll_logical(&mut executor)?
            .context("destination entry did not receive the product")?;
        assert_eq!(copied.results[0].output.status, OpStatus::Ok);

        executor.close()?;
    }
    Ok(())
}

#[test]
fn failed_producer_retires_waiting_consumers_and_preserves_independent_work() -> anyhow::Result<()>
{
    use uniserve_engine::{Batch as LogicalBatch, Op, WorkerExecutor, WorkerId};
    use uniserve_worker_ipc::{BufferAllocation, DiffusionRequestParams};

    let transfer = uniserve_engine::TransportMap::parse("worker:1->worker:0=shm")?;
    let mut args = rank_group_args(1 << 20, 8 << 20);
    args.pipeline_depth = 3;
    args.transfer = transfer.clone();
    args.entries = [
        (
            "model".into(),
            uniserve_core::EntryConfig::parallel(vec![1], Default::default()),
        ),
        (
            "output".into(),
            uniserve_core::EntryConfig::parallel(vec![0], Default::default()),
        ),
    ]
    .into_iter()
    .collect();
    let mut executor = WorkerExecutor::try_new(
        vec![(WorkerId("worker".into()), Worker::spawn(args)?)],
        transfer,
    )?;
    let bind = |batch: Batch, entry: &str| {
        LogicalBatch::new(
            batch.batch_id,
            vec![Op {
                block_tables: batch.block_tables,
                new_cache_pages: batch.new_cache_pages,
                forward_rows: batch.forward_rows,
                latent: batch.latent_params.into_iter().next(),
                decode: batch.decode_ranges.into_iter().next(),
                buffers: batch.buffer_allocations,
                ..Op::new(
                    batch.operations[0].clone(),
                    (WorkerId("worker".into()), entry.into()),
                )
            }],
            batch.commands,
            batch.input_products,
        )
    };
    let key = RequestKey::new(1, RequestId(71), 1);
    let admission = NewRequest::new_media(
        key,
        1,
        DiffusionRequestParams {
            prompt_token_ids: vec![7],
            seed: 1000,
            geometry: uniserve_worker_ipc::MediaGeometry {
                frame_count: 22,
                video_units: 1,
                prompt_tokens: 1,
                denoise_steps: 4,
            },
        },
    )?;
    let value = ProductRef {
        request_key: key,
        producer_op_id: OpId(1),
        output_index: 0,
        generation: 1,
        kind: ProductKind::Tensor,
        storage_class: StorageClass::DeviceTensor,
        dtype: DType::F32,
        shape_bound: ShapeBound {
            dims: vec![DimBound::Static(1)],
        },
        point_range: PointRange::default(),
    };
    // The stub has no text conditioning computation. Registration succeeds,
    // then the operation reports its actual execution error without a product.
    let produce = Operation {
        request_key: key,
        op_id: OpId(1),
        parent: None,
        entry: "model".into(),
        payload: OpPayload::new(
            OpCode::EncoderText,
            Bounds::default(),
            vec![],
            vec![value.clone()],
            None,
            None,
            0,
        ),
    };
    let mut source = Batch::new(1, vec![admission], vec![produce]);
    source.buffer_allocations.push(BufferAllocation {
        buffer: value.buffer_id(),
        offset: 0,
        bytes: value.max_bytes(),
    });
    executor.submit(bind(source, "model"))?;
    let copied = ProductRef {
        producer_op_id: OpId(2),
        generation: 2,
        ..value.clone()
    };
    let consume = Operation {
        request_key: key,
        op_id: OpId(2),
        parent: None,
        entry: "output".into(),
        payload: OpPayload::new(
            OpCode::TransferProduct,
            Bounds {
                max_transfer_bytes: value.max_bytes(),
                ..Bounds::default()
            },
            vec![value.clone()],
            vec![copied.clone()],
            None,
            None,
            0,
        ),
    };
    let mut consumer = Batch::new(2, vec![], vec![consume]);
    consumer.buffer_allocations.push(BufferAllocation {
        buffer: copied.buffer_id(),
        offset: 256,
        bytes: copied.max_bytes(),
    });
    executor.submit(bind(consumer, "output"))?;
    let independent = text_admission(72, 1, 2)?;
    let independent_key = independent.request_key;
    executor.submit(bind(
        token_batch(
            3,
            3,
            independent_key,
            Some(independent),
            OpId(1),
            Checkpoint::admission_root(OpId(0)),
            OpCode::ArExtend,
            &[9],
            0,
            BlockId(2),
            0,
        ),
        "model",
    ))?;

    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let mut failed = false;
    let mut source_returned = false;
    let mut consumer_retired = false;
    let mut independent_returned = false;
    while std::time::Instant::now() < deadline
        && !(failed && source_returned && consumer_retired && independent_returned)
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
                assert_eq!(loss.retired, vec![(2, key, OpId(2))]);
                assert!(loss.products.contains(&value));
                failed = true;
            }
            Ok(Some(result)) => match result.batch_id {
                1 => {
                    assert_eq!(result.results.len(), 1);
                    assert_eq!(result.results[0].output.status, OpStatus::Error);
                    source_returned = result.done;
                }
                2 => {
                    assert!(result.results.is_empty());
                    consumer_retired = result.done;
                }
                3 => {
                    assert_eq!(result.results.len(), 1);
                    assert_eq!(result.results[0].output.status, OpStatus::Ok);
                    assert_eq!(result.results[0].output.committed_tokens(), &[1000]);
                    independent_returned = result.done;
                }
                batch => panic!("unexpected batch {batch}"),
            },
            Ok(None) => {}
        }
    }
    assert!(failed && source_returned && consumer_retired && independent_returned);
    executor.close()?;
    Ok(())
}

#[test]
fn rank_result_fragmentation_preserves_independent_completions() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;

    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let wrapper =
        std::env::temp_dir().join(format!("uniserve-framing-{}-{nonce}", std::process::id()));
    let python = serde_json::to_string(&worker_python())?;
    let fixture = serde_json::to_string(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/python/fixtures/rank_framing.py"),
    )?;
    std::fs::write(
        &wrapper,
        format!(
            "#!/usr/bin/env python3\nimport os, sys\nos.execv({python}, [{python}, {fixture}, *sys.argv[3:]])\n"
        ),
    )?;
    std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))?;
    let mut args = rank_group_args(1 << 20, 8 << 20);
    args.python = wrapper.clone();
    let mut worker = Worker::spawn(args)?;
    let root = Checkpoint::admission_root(OpId(0));
    let first = text_admission(41, 1, 1)?;
    let second = text_admission(42, 1, 2)?;
    let mut run = token_batch(
        1,
        1,
        first.request_key,
        Some(first),
        OpId(1),
        root.clone(),
        OpCode::ArExtend,
        &[7, 8],
        0,
        BlockId(1),
        0,
    );
    let mut other = token_batch(
        1,
        1,
        second.request_key,
        Some(second),
        OpId(2),
        root,
        OpCode::ArExtend,
        &[9, 10],
        0,
        BlockId(2),
        0,
    );
    for row in &mut other.forward_rows {
        row.operation_index += run.operations.len() as u32;
    }
    run.operations.extend(other.operations);
    run.commands.extend(other.commands);
    run.input_products.extend(other.input_products);
    run.forward_rows.extend(other.forward_rows);
    run.block_tables.extend(other.block_tables);
    run.new_cache_pages.extend(other.new_cache_pages);
    worker.submit_run(run)?;
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    let mut completed = std::collections::BTreeSet::new();
    let mut terminal = false;
    while std::time::Instant::now() < deadline && !terminal {
        if let Some(report) = worker.poll_run(Duration::from_millis(100))? {
            for completion in report.completions {
                assert_eq!(completion.status, OpStatus::Ok);
                assert_eq!(completion.committed_tokens().len(), 1);
                assert!(completed.insert(completion.op_id.0));
            }
            terminal = report.done;
        }
    }
    let _ = remove_file(&wrapper);
    assert!(
        terminal,
        "rank-local report grouping prevented terminal completion"
    );
    assert_eq!(completed, [1, 2].into_iter().collect());
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
    config.runtime_profile = RuntimeProfile::diffusion(uniserve_core::ModelDtype::BFloat16);
    config.max_batch = PIPELINE_DEPTH * 8;
    config.workers = vec![WorkerConfig::model("cpu", WORLD_SIZE, 2)];
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
            let prompt_tokens = u32::try_from(prompt_token_ids.len())?;
            let events = engine.submit(Request::Diffusion(DiffusionRequest {
                request_id,
                prompt_token_ids,
                seed: index as u64,
                priority: 0,
                geometry: MediaGeometry {
                    frame_count: 22,
                    video_units: 3,
                    prompt_tokens,
                    denoise_steps: 4,
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
                matches!(event, Event::Rejected { .. }),
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
    assert_eq!(info.max_batch_ops, 256);
    assert_eq!(info.max_batch_tokens, 256);
    assert_eq!(info.queue_depth as usize, PIPELINE_DEPTH);

    let admission = text_admission(11, 1, 1)?;
    let root = Checkpoint::admission_root(OpId(0));
    let initial = token_batch(
        1,
        1,
        admission.request_key,
        Some(admission.clone()),
        OpId(1),
        root.clone(),
        OpCode::ArExtend,
        &[7, 8],
        0,
        BlockId(1),
        0,
    );
    let first = execute(&mut executor, initial.clone())?;
    let first_record = &first.completions[0];
    assert_eq!(first_record.status, OpStatus::Ok);
    assert_eq!(first_record.committed_tokens().len(), 1);

    let replayed = execute(&mut executor, initial.clone())?;
    assert_eq!(replayed, first);

    let conflicting = token_batch(
        1,
        1,
        admission.request_key,
        Some(admission.clone()),
        OpId(1),
        root.clone(),
        OpCode::ArExtend,
        &[7, 8, 9],
        0,
        BlockId(1),
        0,
    );
    assert_execution_error(&mut executor, conflicting, "submitted run")?;

    let selected = fixed_completion(first_record);
    let commit = BatchCommand::Commit {
        request_key: admission.request_key,
        control_seq: 1,
        expected_parent: root.clone(),
        selected: selected.clone(),
        public_event_limit: 1,
        disposition: Disposition::Publish,
    };
    let commit_batch = command_batch(2, commit.clone());
    assert!(
        execute(&mut executor, commit_batch.clone())?
            .completions
            .is_empty()
    );
    assert!(execute(&mut executor, commit_batch)?.completions.is_empty());

    let gap = BatchCommand::Finish {
        request_key: admission.request_key,
        control_seq: 3,
        cutoff: selected.clone(),
        reason: CloseReason::Cancelled,
        retained_buffers: Vec::new(),
    };
    assert_execution_error(&mut executor, command_batch(3, gap), "does not follow")?;

    let stale_key = RequestKey::new(
        admission.request_key.authority_id,
        admission.request_key.request_id,
        admission.request_key.epoch + 1,
    );
    let stale = BatchCommand::Commit {
        request_key: stale_key,
        control_seq: 2,
        expected_parent: selected.clone(),
        selected: selected.clone(),
        public_event_limit: 1,
        disposition: Disposition::Publish,
    };
    assert_execution_error(&mut executor, command_batch(4, stale), "stale request key")?;

    let cutoff = Checkpoint {
        op_id: OpId(99),
        point: CheckpointPoint::Fixed(1),
    };
    let unreachable = BatchCommand::Finish {
        request_key: admission.request_key,
        control_seq: 2,
        cutoff,
        reason: CloseReason::Cancelled,
        retained_buffers: Vec::new(),
    };
    assert_execution_error(
        &mut executor,
        command_batch(5, unreachable),
        "resolved lineage",
    )?;

    let close = BatchCommand::Finish {
        request_key: admission.request_key,
        control_seq: 2,
        cutoff: selected.clone(),
        reason: CloseReason::Cancelled,
        retained_buffers: Vec::new(),
    };
    let independent = text_admission(12, 1, 2)?;
    let mut close_batch = token_batch(
        6,
        2,
        independent.request_key,
        Some(independent),
        OpId(1),
        root.clone(),
        OpCode::ArExtend,
        &[9, 10],
        0,
        BlockId(2),
        0,
    );
    close_batch.commands.push(close.clone());
    for _ in 0..2 {
        executor.submit_run(close_batch.clone())?;
        let result = executor
            .poll_run(Duration::from_secs(30))?
            .ok_or_else(|| anyhow::anyhow!("independent operation did not complete"))?;
        assert!(!result.done);
        assert_eq!(result.completions.len(), 1);
        assert_eq!(result.completions[0].status, OpStatus::Ok);
        let acknowledgement = executor
            .poll_run(Duration::from_secs(30))?
            .ok_or_else(|| anyhow::anyhow!("Finish acknowledgement did not arrive"))?;
        assert!(acknowledgement.done);
        assert!(acknowledgement.completions.is_empty());
    }
    let conflicting_close = BatchCommand::Finish {
        request_key: admission.request_key,
        control_seq: 2,
        cutoff: selected.clone(),
        reason: CloseReason::Error,
        retained_buffers: Vec::new(),
    };
    assert_execution_error(
        &mut executor,
        command_batch(7, conflicting_close),
        "conflicts with its committed content",
    )?;

    let descendant = token_batch(
        8,
        3,
        admission.request_key,
        None,
        OpId(2),
        selected,
        OpCode::ArDecode,
        &[first_record.committed_tokens()[0]],
        2,
        BlockId(1),
        2,
    );
    let closed = execute(&mut executor, descendant)?;
    let closed_record = &closed.completions[0];
    assert_eq!(closed_record.status, OpStatus::Error);
    assert_eq!(closed_record.error_code, Some(ErrorCode::InvalidOperation));

    qualify_kv_rank_locations(&mut executor)?;
    executor.close()?;
    assert!(!executor.is_ready());
    assert!(executor.poll_run(Duration::ZERO)?.is_none());
    assert!(matches!(
        executor.submit_run(command_batch(
            100,
            BatchCommand::Retire {
                request_key: admission.request_key,
                retained_buffers: Vec::new(),
            }
        )),
        Err(uniserve_engine::RunSubmitError::Failed(_))
    ));
    executor.close()?;
    Ok(())
}

fn qualify_kv_rank_locations(executor: &mut Worker) -> anyhow::Result<()> {
    let admission = text_admission(13, 1, 3)?;
    let request_key = admission.request_key;
    let root = Checkpoint::admission_root(OpId(0));
    let mut initial = token_batch(
        9,
        4,
        request_key,
        Some(admission),
        OpId(1),
        root.clone(),
        OpCode::ArExtend,
        &[7, 8],
        0,
        BlockId(3),
        0,
    );
    initial.operations[0].outputs_mut().clear();
    let tables = initial.block_tables.clone();
    let first = execute(executor, initial)?;
    assert_eq!(first.completions[0].status, OpStatus::Ok);
    let parent = fixed_completion(&first.completions[0]);
    let product = ProductRef {
        request_key,
        producer_op_id: OpId(2),
        output_index: 0,
        generation: 2,
        kind: ProductKind::Kv,
        storage_class: StorageClass::PagedKv,
        dtype: DType::U8,
        shape_bound: ShapeBound {
            dims: vec![DimBound::Static(4096)],
        },
        point_range: PointRange::default(),
    };
    let operation = Operation {
        request_key,
        op_id: OpId(2),
        parent: Some(parent.clone()),
        entry: "model".into(),
        payload: OpPayload::new(
            OpCode::TransferKvPublish,
            Bounds {
                max_points: 1,
                max_transfer_bytes: 4096,
                ..Bounds::default()
            },
            Vec::new(),
            vec![product.clone()],
            None,
            None,
            1,
        ),
    }
    .sealed();
    let mut publish = Batch::new(10, Vec::new(), vec![operation]);
    publish.collective_seq = 5;
    publish.block_tables = tables;
    publish.commands.push(BatchCommand::Commit {
        request_key,
        control_seq: 1,
        expected_parent: root,
        selected: parent,
        public_event_limit: 0,
        disposition: Disposition::Publish,
    });
    let report = execute(executor, publish.clone())?;
    assert_eq!(report.completions[0].status, OpStatus::Ok);
    assert_eq!(report.products.len(), 1);
    assert_eq!(report.products[0].product, product);
    let InlineValue::Transfer(TransferHandle::Kv { tensors, .. }) = &report.products[0].value
    else {
        anyhow::bail!("KV publication did not expose its physical locations");
    };
    for tensor in tensors {
        let ranks = tensor
            .locations
            .iter()
            .map(|location| location.source.rank)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(ranks, (0..WORLD_SIZE as u32).collect());
    }
    assert_eq!(execute(executor, publish)?, report);
    Ok(())
}

fn qualify_peer_replacement() -> anyhow::Result<()> {
    // Cargo may execute the media test in this binary concurrently. Serialize
    // child launch while the replacement fault is injected so unrelated
    // workers cannot inherit the process-wide test environment.
    let launch_guard = CHILD_LAUNCH_ENV_LOCK
        .lock()
        .map_err(|_| anyhow::anyhow!("child-launch environment lock is poisoned"))?;
    unsafe {
        std::env::set_var("UNISERVE_STUB_DIE_AFTER", "2");
        std::env::set_var("UNISERVE_STUB_DIE_RANK", "1");
    }
    let mut executor = spawn_rank_group()?;
    unsafe {
        std::env::remove_var("UNISERVE_STUB_DIE_AFTER");
        std::env::remove_var("UNISERVE_STUB_DIE_RANK");
    }
    drop(launch_guard);

    let initial_endpoint = executor.info().endpoint.clone();
    let first_admission = text_admission(21, 1, 1)?;
    let first_root = Checkpoint::admission_root(OpId(0));
    let mut first = token_batch(
        1,
        1,
        first_admission.request_key,
        Some(first_admission),
        OpId(1),
        first_root.clone(),
        OpCode::ArExtend,
        &[3],
        0,
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
        OpId(1),
        first_root.clone(),
        OpCode::ArExtend,
        &[2],
        0,
        BlockId(2),
        0,
    );
    finished.forward_rows[0].operation_index = 1;
    first.operations.extend(finished.operations);
    first.commands.extend(finished.commands);
    first.block_tables.extend(finished.block_tables);
    first.new_cache_pages.extend(finished.new_cache_pages);
    first.forward_rows.extend(finished.forward_rows);
    first.input_products.extend(finished.input_products);
    execute(&mut executor, first)?;
    let mut close = Batch::new(2, Vec::new(), Vec::new());
    close.collective_seq = 2;
    close.commands.push(BatchCommand::Finish {
        request_key: finished_key,
        control_seq: 1,
        cutoff: first_root,
        reason: CloseReason::Cancelled,
        retained_buffers: Vec::new(),
    });
    let mut report = execute(&mut executor, close)?;
    while !report.done {
        report = executor
            .poll_run(Duration::from_secs(30))?
            .context("request Finish did not acknowledge physical retirement")?;
    }

    let lost_admission = text_admission(22, 1, 2)?;
    let lost_root = Checkpoint::admission_root(OpId(0));
    executor.submit_run(token_batch(
        3,
        3,
        lost_admission.request_key,
        Some(lost_admission),
        OpId(2),
        lost_root,
        OpCode::ArExtend,
        &[4],
        0,
        BlockId(2),
        0,
    ))?;
    let loss = executor
        .poll_run(Duration::from_secs(30))
        .expect_err("rank loss must be reported");
    let loss = loss
        .downcast_ref::<WorkerFailure>()
        .context("expected scoped Worker loss")?;
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
        executor.poll_run(Duration::from_millis(100))?;
    }

    let recovered_admission = text_admission(23, 1, 1)?;
    let recovered_root = Checkpoint::admission_root(OpId(0));
    let recovered = execute(
        &mut executor,
        token_batch(
            4,
            4,
            recovered_admission.request_key,
            Some(recovered_admission),
            OpId(3),
            recovered_root,
            OpCode::ArExtend,
            &[5],
            0,
            BlockId(1),
            0,
        ),
    )?;
    assert_eq!(recovered.completions[0].status, OpStatus::Ok);
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
    let startup = Worker::spawn(args);
    remove_file(wrapper)?;
    anyhow::ensure!(startup.is_err(), "an incomplete rank group became ready");
    Ok(())
}

fn qualify_slow_transfer() -> anyhow::Result<()> {
    let worker = worker_python();
    let config = WorkerProcessArgs {
        stub: true,
        cuda_graph: false,
        prefill_cuda_graph: false,
        ..WorkerProcessArgs::default()
    };
    let mut executor = Worker::spawn(WorkerProcessArgs {
        python: worker,
        model: String::new(),
        ranks: WorkerConfig::model("cpu", WORLD_SIZE, 2).ranks,
        entries: WorkerConfig::model("cpu", WORLD_SIZE, 2).entries,
        pipeline_depth: PIPELINE_DEPTH,
        req_slot_cap: 1 << 20,
        resp_slot_cap: 8 << 20,
        kv_token_capacity: Some(4096),
        block_size: 16,
        max_batch_operations: 256,
        max_batch_tokens: 256,
        attention_backend: uniserve_worker_ipc::AttentionBackend::TorchSdpa,
        supported_ops: uniserve_worker_ipc::OpCode::ALL.to_vec(),
        transfer: uniserve_engine::TransportMap::parse("publisher->worker=shm")?,
        ..config
    })?;

    let slow_admission = text_admission(31, 1, 1)?;
    let slow_root = Checkpoint::admission_root(OpId(0));
    let mut slow = token_batch(
        1,
        2,
        slow_admission.request_key,
        Some(slow_admission.clone()),
        OpId(1),
        slow_root,
        OpCode::ArExtend,
        &[6],
        0,
        BlockId(1),
        0,
    );
    let predicate = ProductRef {
        request_key: slow_admission.request_key,
        producer_op_id: OpId(91),
        output_index: 0,
        generation: 91,
        kind: ProductKind::Completion,
        storage_class: StorageClass::DeviceTensor,
        dtype: DType::U8,
        shape_bound: ShapeBound::default(),
        point_range: PointRange::default(),
    };
    let mut publication = SlowShmPublication::start()?;
    slow.operations[0].set_predicate(Some(predicate.clone()));
    slow.input_products.push(ProductPayload {
        product: predicate,
        value: InlineValue::Transfer(publication.descriptor(91)?),
    });

    let fast_admission = text_admission(32, 1, 2)?;
    let fast_root = Checkpoint::admission_root(OpId(0));
    let fast = token_batch(
        2,
        1,
        fast_admission.request_key,
        Some(fast_admission),
        OpId(1),
        fast_root,
        OpCode::ArExtend,
        &[9],
        0,
        BlockId(2),
        0,
    );

    executor.submit_run(slow)?;
    executor.submit_run(fast)?;
    let first = executor
        .poll_run(Duration::from_secs(30))?
        .ok_or_else(|| anyhow::anyhow!("fast submission did not complete"))?;
    assert_eq!(first.run_id, 2);
    assert!(!publication.published.load(Ordering::Acquire));
    assert_eq!(first.completions[0].status, OpStatus::Ok);

    publication.publish()?;
    let second = executor
        .poll_run(Duration::from_secs(30))?
        .ok_or_else(|| anyhow::anyhow!("slow submission did not complete"))?;
    assert_eq!(second.run_id, 1);
    assert_eq!(second.completions[0].status, OpStatus::Ok);

    let admission = text_admission(33, 1, 3)?;
    let input = ProductRef {
        request_key: admission.request_key,
        producer_op_id: OpId(92),
        output_index: 0,
        generation: 92,
        kind: ProductKind::Tensor,
        storage_class: StorageClass::DeviceTensor,
        dtype: DType::U8,
        shape_bound: ShapeBound {
            dims: vec![DimBound::Static(1)],
        },
        point_range: PointRange::default(),
    };
    let output = ProductRef {
        producer_op_id: OpId(1),
        generation: 1,
        ..input.clone()
    };
    let operation = Operation {
        request_key: admission.request_key,
        op_id: OpId(1),
        parent: Some(Checkpoint::admission_root(OpId(0))),
        entry: "model".into(),
        payload: OpPayload::new(
            OpCode::TransferProduct,
            Bounds {
                max_transfer_bytes: input.max_bytes(),
                ..Bounds::default()
            },
            vec![input.clone()],
            vec![output.clone()],
            None,
            None,
            0,
        ),
    };
    let mut batch = Batch::new(3, vec![admission], vec![operation]);
    batch.buffer_allocations = [(&input, 0), (&output, 256)]
        .into_iter()
        .map(|(product, offset)| uniserve_worker_ipc::BufferAllocation {
            buffer: product.buffer_id(),
            offset,
            bytes: product.max_bytes(),
        })
        .collect();
    batch.input_products.push(ProductPayload {
        product: input,
        value: InlineValue::Transfer(publication.descriptor(92)?),
    });
    executor.submit_run(batch)?;
    let transferred = executor
        .poll_run(Duration::from_secs(30))?
        .context("tensor input did not reach its computation entry")?;
    assert_eq!(transferred.completions[0].status, OpStatus::Ok);
    assert_eq!(transferred.products[0].product, output);
    let InlineValue::Transfer(TransferHandle::DeviceProduct { tensor, .. }) =
        &transferred.products[0].value
    else {
        anyhow::bail!("tensor output has no physical publication");
    };
    assert_eq!(tensor.shape, vec![1]);
    assert_eq!(tensor.locations.len(), WORLD_SIZE);
    publication.finish()?;
    executor.close()?;
    assert!(publication.path.is_file());
    Ok(())
}

/// Independent instances accepting the same computation must retain separate capacity.
#[test]
fn independent_workers_preserve_capacity_retirement_and_failed_work() -> anyhow::Result<()> {
    use uniserve_engine::{
        Batch as LogicalBatch, ExecutorSubmitError, Op, WorkerExecutor, WorkerId,
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
    let spawn = |worker_id: &str, depth| -> anyhow::Result<Worker> {
        let mut args = rank_group_args(1 << 20, 8 << 20);
        let binding = WorkerConfig::model("cpu", 1, depth);
        args.ranks = binding.ranks.clone();
        args.entries = binding.entries;
        args.pipeline_depth = depth;
        args.worker_id = worker_id.into();
        if worker_id == "encoder-0" {
            args.python = wrapper.clone();
        }

        args.transfer = uniserve_engine::TransportMap::parse(
            "encoder-0->encoder-1=shm,encoder-1->encoder-0=shm",
        )?;
        Worker::spawn(args)
    };
    let mut executor = WorkerExecutor::try_new(
        vec![
            (WorkerId("encoder-0".into()), spawn("encoder-0", 1)?),
            (WorkerId("encoder-1".into()), spawn("encoder-1", 2)?),
        ],
        uniserve_engine::TransportMap::parse("encoder-0->encoder-1=shm,encoder-1->encoder-0=shm")?,
    )?;
    let pid = std::fs::read_to_string(&pid_file)?.parse::<i32>()?;
    let mut paused = PausedProcess::new(pid)?;
    let bind = |batch: Batch, worker: &str| {
        assert_eq!(batch.operations.len(), 1);
        LogicalBatch::new(
            batch.batch_id,
            vec![Op {
                block_tables: batch.block_tables,
                new_cache_pages: batch.new_cache_pages,
                forward_rows: batch.forward_rows,
                latent: batch.latent_params.into_iter().next(),
                decode: batch.decode_ranges.into_iter().next(),
                buffers: batch.buffer_allocations,
                ..Op::new(
                    batch.operations[0].clone(),
                    (WorkerId(worker.into()), "model".into()),
                )
            }],
            batch.commands,
            batch.input_products,
        )
    };
    let make_run = |run_id, request_id, page| -> anyhow::Result<Batch> {
        let admission = text_admission(request_id, 1, page)?;
        Ok(token_batch(
            run_id,
            run_id,
            admission.request_key,
            Some(admission),
            OpId(1),
            Checkpoint::admission_root(OpId(0)),
            OpCode::ArExtend,
            &[7],
            0,
            BlockId(page),
            0,
        ))
    };
    let slow = make_run(1, 51, 1)?;
    executor.submit(bind(slow, "encoder-0"))?;
    let blocked = bind(make_run(2, 52, 2)?, "encoder-0");
    let blocked = match executor.submit(blocked) {
        Err(ExecutorSubmitError::WouldBlock(batch)) => batch,
        other => anyhow::bail!("occupied instance did not apply its capacity bound: {other:?}"),
    };
    let independent = make_run(3, 53, 1)?;
    let token = independent.operations[0].outputs()[0].clone();
    executor.submit(bind(independent, "encoder-1"))?;
    let first = poll_logical(&mut executor)?.context("independent worker did not complete")?;
    assert_eq!(first.batch_id, 3);
    assert_eq!(first.results[0].output.status, OpStatus::Ok);
    paused.resume()?;
    let slow = poll_logical(&mut executor)?.context("released worker did not complete")?;
    assert_eq!(slow.batch_id, 1);
    assert_eq!(slow.results[0].output.status, OpStatus::Ok);
    executor.submit(blocked)?;
    let resumed = poll_logical(&mut executor)?.context("capacity was not reusable")?;
    assert_eq!(resumed.batch_id, 2);
    assert_eq!(resumed.results[0].output.status, OpStatus::Ok);
    // Request-relay products have no arena params, but their release must
    // still reach the rank holding the published generation and support replay.
    let release = LogicalBatch::new(
        4,
        Vec::new(),
        vec![BatchCommand::Free {
            buffer: token.buffer_id(),
        }],
        Vec::new(),
    );
    executor.submit(release.clone())?;
    let retired = poll_logical(&mut executor)?.context("product release did not complete")?;
    assert_eq!(retired.batch_id, 4);
    assert!(retired.results.is_empty());
    executor.submit(release)?;
    let replay = poll_logical(&mut executor)?.context("product release replay did not complete")?;
    assert_eq!(replay.batch_id, retired.batch_id);
    assert!(replay.results.is_empty());

    // The second instance holds a transported product and an admission root,
    // while the first instance owns the request's semantic Finish checkpoint.
    let shared = make_run(5, 54, 3)?;
    let admission = shared.admissions().next().unwrap().clone();
    let source = shared.operations[0].outputs()[0].clone();
    executor.submit(bind(shared, "encoder-0"))?;
    let produced = poll_logical(&mut executor)?.context("source did not complete")?;
    assert!(produced.done);
    let cutoff = fixed_completion(&produced.results[0].output);
    let publication = ProductRef {
        producer_op_id: OpId(2),
        generation: 2,
        ..source.clone()
    };
    let publish = Operation {
        request_key: admission.request_key,
        op_id: OpId(2),
        parent: Some(cutoff.clone()),
        entry: "model".into(),
        payload: OpPayload::new(
            OpCode::TransferProduct,
            Bounds {
                max_transfer_bytes: source.max_bytes(),
                ..Bounds::default()
            },
            vec![source],
            vec![publication.clone()],
            None,
            None,
            1,
        ),
    };
    let mut publish = Batch::new(6, Vec::new(), vec![publish]);
    publish.commands.push(BatchCommand::Commit {
        request_key: admission.request_key,
        control_seq: 1,
        expected_parent: Checkpoint::admission_root(OpId(0)),
        selected: cutoff.clone(),
        public_event_limit: 0,
        disposition: Disposition::Retain,
    });
    executor.submit(bind(publish, "encoder-0"))?;
    let published = poll_logical(&mut executor)?.context("source publication did not complete")?;
    assert!(published.done);
    assert_eq!(published.results[0].output.status, OpStatus::Ok);
    let copy = ProductRef {
        producer_op_id: OpId(3),
        generation: 3,
        ..publication.clone()
    };
    let transfer = Operation {
        request_key: admission.request_key,
        op_id: OpId(3),
        parent: Some(Checkpoint::admission_root(OpId(0))),
        entry: "model".into(),
        payload: OpPayload::new(
            OpCode::TransferProduct,
            Bounds {
                max_transfer_bytes: publication.max_bytes(),
                ..Bounds::default()
            },
            vec![publication],
            vec![copy.clone()],
            None,
            None,
            0,
        ),
    };
    executor.submit(bind(
        Batch::new(7, vec![admission.clone()], vec![transfer]),
        "encoder-1",
    ))?;
    let copied = poll_logical(&mut executor)?.context("auxiliary transfer did not complete")?;
    assert!(copied.done);
    assert_eq!(copied.results[0].output.status, OpStatus::Ok);
    let image_bytes = b"iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAGUlEQVR4nGN0SGhgIAUwkaR6VMOohiGlAQCjvQFA6eri4wAAAABJRU5ErkJggg==".to_vec();
    let image = ProductRef {
        request_key: admission.request_key,
        producer_op_id: OpId(4),
        output_index: u16::MAX,
        generation: 4,
        kind: ProductKind::Artifact,
        storage_class: StorageClass::HostStaging,
        dtype: DType::U8,
        shape_bound: ShapeBound {
            dims: vec![DimBound::Static(image_bytes.len() as u32)],
        },
        point_range: PointRange::default(),
    };
    let feature = ProductRef {
        request_key: admission.request_key,
        producer_op_id: OpId(4),
        output_index: 0,
        generation: 4,
        kind: ProductKind::VisionFeature,
        storage_class: StorageClass::LatentArena,
        dtype: DType::BF16,
        shape_bound: ShapeBound {
            dims: vec![DimBound::Device { max: 4096 }],
        },
        point_range: PointRange::default(),
    };
    let encode = Operation {
        request_key: admission.request_key,
        op_id: OpId(4),
        parent: Some(Checkpoint::admission_root(OpId(0))),
        entry: "model".into(),
        payload: OpPayload::new(
            OpCode::EncoderVision,
            Bounds {
                max_points: 1,
                max_tokens: 64,
                max_latent_bytes: 8192,
                ..Bounds::default()
            },
            vec![image.clone()],
            vec![feature.clone()],
            None,
            None,
            0,
        ),
    };
    let mut encode = Batch::new(16, Vec::new(), vec![encode]);
    encode.input_products.push(ProductPayload {
        product: image,
        value: InlineValue::Bytes(image_bytes),
    });
    encode
        .buffer_allocations
        .push(uniserve_worker_ipc::BufferAllocation {
            buffer: feature.buffer_id(),
            offset: 0,
            bytes: 8192,
        });
    executor.submit(bind(encode, "encoder-1"))?;
    let encoded = poll_logical(&mut executor)?.context("encoder product did not complete")?;
    assert_eq!(encoded.results[0].output.status, OpStatus::Ok);
    let finish = BatchCommand::Finish {
        request_key: admission.request_key,
        control_seq: 2,
        cutoff,
        reason: CloseReason::Completed,
        retained_buffers: vec![feature.buffer_id()],
    };
    executor.submit(LogicalBatch::new(
        8,
        Vec::new(),
        vec![finish.clone()],
        Vec::new(),
    ))?;
    let closed = poll_logical(&mut executor)?.context("shared request did not retire")?;
    assert!(closed.done);
    let mut last_cutoff = None;
    for (batch_id, request_id, worker) in [(9, 55, "encoder-0"), (10, 56, "encoder-1")] {
        executor.submit(bind(make_run(batch_id, request_id, 3)?, worker))?;
        let reused = poll_logical(&mut executor)?.context("retired slot was not reusable")?;
        assert!(reused.done);
        assert_eq!(reused.results[0].output.status, OpStatus::Ok);
        last_cutoff = Some(fixed_completion(&reused.results[0].output));
    }
    executor.submit(LogicalBatch::new(11, Vec::new(), vec![finish], Vec::new()))?;
    let replay = poll_logical(&mut executor)?.context("Finish replay did not complete")?;
    assert!(replay.done);

    // A rejected close has no physical retirement acknowledgement. The other
    // instance's operation in the same logical batch still completes normally.
    let failed_key = RequestKey::new(1, RequestId(56), 1);
    let rejected_key = RequestKey::new(1, RequestId(61), 1);
    let mut mixed = bind(make_run(12, 57, 4)?, "encoder-0");
    let affected = bind(make_run(12, 61, 6)?, "encoder-1");
    mixed.ops.extend(affected.ops);
    mixed.commands.extend(affected.commands);
    mixed.inline.extend(affected.inline);
    mixed.commands.push(BatchCommand::Finish {
        request_key: failed_key,
        control_seq: 2,
        cutoff: last_cutoff.context("missing completed request checkpoint")?,
        reason: CloseReason::Completed,
        retained_buffers: Vec::new(),
    });
    executor.submit(mixed)?;
    let mut failure_observed = false;
    let mut completed = false;
    let mut tokens = 0;
    while !completed {
        match poll_logical(&mut executor) {
            Err(error) => {
                let failure = error
                    .downcast_ref::<WorkerFailure>()
                    .context("expected scoped command failure")?;
                assert_eq!(
                    failure
                        .requests
                        .iter()
                        .copied()
                        .collect::<std::collections::HashSet<_>>(),
                    [failed_key, rejected_key].into_iter().collect()
                );
                assert_eq!(failure.retired, vec![(12, rejected_key, OpId(1))]);
                assert!(failure.endpoints.is_empty());
                failure_observed = true;
            }
            Ok(Some(report)) => {
                assert_eq!(report.batch_id, 12);
                for result in report.results {
                    assert_eq!(result.output.status, OpStatus::Ok);
                    tokens += result.output.committed_tokens().len();
                }
                if report.done {
                    assert_eq!(report.command_results.len(), 1);
                    assert_eq!(
                        report.command_results[0].outcome,
                        uniserve_engine::CommandOutcome::Failed
                    );
                    completed = true;
                }
            }
            Ok(None) => anyhow::bail!("mixed failure did not retire its logical batch"),
        }
    }
    assert!(failure_observed);
    assert_eq!(tokens, 1);
    executor.submit(LogicalBatch::new(
        13,
        Vec::new(),
        vec![
            BatchCommand::Retire {
                request_key: failed_key,
                retained_buffers: Vec::new(),
            },
            BatchCommand::Retire {
                request_key: rejected_key,
                retained_buffers: Vec::new(),
            },
        ],
        Vec::new(),
    ))?;
    assert!(
        poll_logical(&mut executor)?
            .context("failed close did not retire")?
            .done
    );
    executor.submit(bind(make_run(14, 58, 3)?, "encoder-1"))?;
    let reused = poll_logical(&mut executor)?.context("failed request slot was not reusable")?;
    assert_eq!(reused.results[0].output.status, OpStatus::Ok);

    // The retained product's separate lifetime survives both request slot reuse
    // and replay of the producer's Finish command.
    let next = text_admission(59, 1, 5)?;
    let retained = Operation {
        request_key: next.request_key,
        op_id: OpId(1),
        parent: Some(Checkpoint::admission_root(OpId(0))),
        entry: "model".into(),
        payload: OpPayload::new(
            OpCode::TransferProduct,
            Bounds {
                max_transfer_bytes: feature.max_bytes(),
                max_latent_bytes: feature.max_bytes(),
                ..Bounds::default()
            },
            vec![feature.clone()],
            vec![ProductRef {
                request_key: next.request_key,
                producer_op_id: OpId(1),
                ..feature
            }],
            None,
            None,
            0,
        ),
    };
    let mut retained = Batch::new(15, vec![next], vec![retained]);
    retained
        .buffer_allocations
        .push(uniserve_worker_ipc::BufferAllocation {
            buffer: retained.operations[0].outputs()[0].buffer_id(),
            offset: 8192,
            bytes: 8192,
        });
    executor.submit(bind(retained, "encoder-0"))?;
    let retained = poll_logical(&mut executor)?.context("retained product was not readable")?;
    assert_eq!(retained.results[0].output.status, OpStatus::Ok);
    executor.submit(bind(make_run(17, 62, 6)?, "encoder-1"))?;
    let fresh =
        poll_logical(&mut executor)?.context("Worker did not accept work after rejection")?;
    assert_eq!(fresh.results[0].output.status, OpStatus::Ok);

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
}

impl Drop for PausedProcess {
    fn drop(&mut self) {
        let _ = self.resume();
    }
}

struct SlowShmPublication {
    name: String,
    endpoint: String,
    path: PathBuf,
    published: Arc<AtomicBool>,
    stop_readers: Arc<AtomicBool>,
    notify: UnixStream,
    readers: Option<thread::JoinHandle<anyhow::Result<()>>>,
}

impl SlowShmPublication {
    fn start() -> anyhow::Result<Self> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let name = format!("uniserve-transfer-{}-{nonce}", std::process::id());
        let path = Path::new("/dev/shm").join(&name);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        file.set_len(1)?;
        file.write_all_at(&[0], 0)?;
        let endpoint = format!("{name}-readers");
        let published = Arc::new(AtomicBool::new(false));
        let stop_readers = Arc::new(AtomicBool::new(false));
        let (notify, notifications) = UnixStream::pair()?;
        let readers = start_shm_readers(
            &endpoint,
            Arc::clone(&published),
            Arc::clone(&stop_readers),
            notifications,
        )?;
        Ok(Self {
            name,
            endpoint,
            path,
            published,
            stop_readers,
            notify,
            readers: Some(readers),
        })
    }

    fn publish(&self) -> anyhow::Result<()> {
        OpenOptions::new()
            .write(true)
            .open(&self.path)?
            .write_all_at(&[1], 0)?;
        self.published.store(true, Ordering::Release);
        (&self.notify).write_all(b"R")?;
        Ok(())
    }

    fn descriptor(&self, generation: u32) -> anyhow::Result<TransferHandle> {
        Ok(TransferHandle::DeviceProduct {
            generation,
            height: 0,
            width: 0,
            value_range: String::new(),
            tensor: uniserve_worker_ipc::TensorTransfer {
                shape: vec![1],
                locations: vec![Locator {
                    source: uniserve_worker_ipc::WorkerEndpoint {
                        worker_id: "publisher".into(),
                        rank: 0,
                        node: std::fs::read_to_string("/etc/hostname")?.trim().into(),
                        address_space: format!("rust:{}", std::process::id()),
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

    fn finish(&mut self) -> anyhow::Result<()> {
        self.stop_readers.store(true, Ordering::Release);
        (&self.notify).write_all(b"C")?;
        if let Some(readers) = self.readers.take() {
            readers
                .join()
                .map_err(|_| anyhow::anyhow!("shared-memory reader endpoint panicked"))??;
        }
        Ok(())
    }
}

impl Drop for SlowShmPublication {
    fn drop(&mut self) {
        if self.readers.is_some() {
            let _ = self.finish();
        }
        let _ = remove_file(&self.path);
    }
}

fn spawn_rank_group() -> anyhow::Result<Worker> {
    spawn_rank_group_with_capacities(1 << 20, 8 << 20)
}

fn spawn_rank_group_with_capacities(
    request_slot_capacity: usize,
    response_slot_capacity: usize,
) -> anyhow::Result<Worker> {
    Worker::spawn(rank_group_args(
        request_slot_capacity,
        response_slot_capacity,
    ))
}

fn rank_group_args(
    request_slot_capacity: usize,
    response_slot_capacity: usize,
) -> WorkerProcessArgs {
    let worker = worker_python();
    let config = WorkerProcessArgs {
        stub: true,
        cuda_graph: false,
        prefill_cuda_graph: false,
        ..WorkerProcessArgs::default()
    };
    WorkerProcessArgs {
        python: worker,
        model: String::new(),
        ranks: WorkerConfig::model("cpu", WORLD_SIZE, 2).ranks,
        entries: WorkerConfig::model("cpu", WORLD_SIZE, 2).entries,
        pipeline_depth: PIPELINE_DEPTH,
        req_slot_cap: request_slot_capacity,
        resp_slot_cap: response_slot_capacity,
        kv_token_capacity: Some(4096),
        block_size: 16,
        max_batch_operations: 256,
        max_batch_tokens: 256,
        attention_backend: uniserve_worker_ipc::AttentionBackend::TorchSdpa,
        supported_ops: uniserve_worker_ipc::OpCode::ALL.to_vec(),
        transfer: Default::default(),
        ..config
    }
}

fn worker_python() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(".venv/bin/python")
}

fn execute(executor: &mut Worker, batch: Batch) -> anyhow::Result<uniserve_worker_ipc::RunResult> {
    executor.submit_run(batch)?;
    executor
        .poll_run(Duration::from_secs(30))?
        .ok_or_else(|| anyhow::anyhow!("submission did not complete"))
}

fn assert_execution_error(
    executor: &mut Worker,
    batch: Batch,
    message: &str,
) -> anyhow::Result<()> {
    executor.submit_run(batch)?;
    let error = executor
        .poll_run(Duration::from_secs(30))
        .expect_err("submission must be rejected");
    let execution = error
        .downcast_ref::<WorkerFailure>()
        .and_then(|failure| failure.execution.as_ref())
        .ok_or_else(|| anyhow::anyhow!("expected a typed worker execution error: {error:#}"))?;
    assert_eq!(execution.code.as_deref(), Some("InvalidDescriptor"));
    assert!(execution.message.contains(message), "{execution}");
    Ok(())
}

fn text_admission(
    request_id: u64,
    epoch: u64,
    request_pool_idx: u32,
) -> anyhow::Result<NewRequest> {
    Ok(NewRequest::new(
        RequestKey::new(1, RequestId(request_id), epoch),
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

fn token_batch(
    run_id: u64,
    collective_seq: u64,
    request_key: RequestKey,
    admission: Option<NewRequest>,
    op_id: OpId,
    parent: Checkpoint,
    mode: OpCode,
    tokens: &[u32],
    control_seq: u64,
    page: BlockId,
    prefix_length: u32,
) -> Batch {
    let request_pool_idx = admission.as_ref().map_or(1, |value| value.request_pool_idx);
    let input = ProductRef {
        request_key,
        producer_op_id: op_id,
        output_index: u16::MAX,
        generation: (op_id.0 as u32).saturating_mul(3).max(1),
        kind: ProductKind::Token,
        storage_class: StorageClass::HostStaging,
        dtype: DType::U32,
        shape_bound: ShapeBound {
            dims: vec![DimBound::Static(tokens.len().max(1) as u32)],
        },
        point_range: PointRange::default(),
    };
    let token_output = ProductRef {
        request_key,
        producer_op_id: op_id,
        output_index: 0,
        generation: (op_id.0 as u32).saturating_mul(4).saturating_add(1),
        kind: ProductKind::Token,
        storage_class: StorageClass::RequestRelay,
        dtype: DType::U32,
        shape_bound: ShapeBound::default(),
        point_range: PointRange {
            base_point: 0,
            max_points: 1,
        },
    };
    let operation = Operation {
        request_key,
        op_id,
        parent: Some(parent),
        entry: "model".into(),
        payload: OpPayload::new(
            mode,
            Bounds {
                max_points: 1,
                max_tokens: tokens.len().max(1) as u32,
                max_kv_pages: u32::from(prefix_length == 0),
                ..Bounds::default()
            },
            vec![input.clone()],
            vec![token_output],
            None,
            None,
            control_seq,
        ),
    }
    .sealed();
    let input_length = tokens.len() as u32;
    let mut batch = Batch::new(run_id, admission.into_iter().collect(), vec![operation]);
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
    batch.forward_rows = vec![RowGeometry {
        operation_index: 0,
        request_pool_index: request_pool_idx,
        seq_len: prefix_length,
        query_len: input_length.max(1),
        write_kv: true,
    }];
    batch.with_input_products(vec![ProductPayload {
        product: input,
        value: InlineValue::Bytes(encode_token_product_bytes(tokens)),
    }])
}

fn command_batch(run_id: u64, command: BatchCommand) -> Batch {
    Batch::new(run_id, Vec::new(), Vec::new()).with_commands(vec![command])
}

fn fixed_completion(record: &uniserve_worker_ipc::ModelOutput) -> Checkpoint {
    Checkpoint {
        op_id: record.op_id,
        point: CheckpointPoint::Fixed(record.selected_point),
    }
}

/// External shared-memory peer used to hold readiness while real ranks make progress.
fn start_shm_readers(
    endpoint: &str,
    published: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    notifications: UnixStream,
) -> anyhow::Result<thread::JoinHandle<anyhow::Result<()>>> {
    let descriptor = unsafe {
        libc::socket(
            libc::AF_UNIX,
            libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC | libc::SOCK_NONBLOCK,
            0,
        )
    };
    anyhow::ensure!(
        descriptor >= 0,
        "failed to create shared-memory reader socket"
    );
    let listener = unsafe { OwnedFd::from_raw_fd(descriptor) };
    let mut address: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    address.sun_family = libc::AF_UNIX as libc::sa_family_t;
    anyhow::ensure!(
        endpoint.len() + 1 < address.sun_path.len(),
        "reader endpoint is too long"
    );
    for (target, byte) in address.sun_path[1..].iter_mut().zip(endpoint.bytes()) {
        *target = byte as libc::c_char;
    }
    let address_length = std::mem::offset_of!(libc::sockaddr_un, sun_path) + 1 + endpoint.len();
    let bound = unsafe {
        libc::bind(
            listener.as_raw_fd(),
            (&raw const address).cast(),
            address_length as libc::socklen_t,
        )
    };
    anyhow::ensure!(
        bound == 0,
        "failed to bind reader endpoint: {}",
        std::io::Error::last_os_error()
    );
    anyhow::ensure!(
        unsafe { libc::listen(listener.as_raw_fd(), 8) } == 0,
        "failed to listen for readers"
    );
    Ok(thread::spawn(move || {
        let mut readers: Vec<(OwnedFd, bool, bool)> = Vec::new();
        let send = |reader: &OwnedFd, response: u8| -> anyhow::Result<()> {
            anyhow::ensure!(
                unsafe {
                    libc::send(
                        reader.as_raw_fd(),
                        (&raw const response).cast(),
                        1,
                        libc::MSG_NOSIGNAL,
                    )
                } == 1,
                "reader endpoint response failed"
            );
            Ok(())
        };
        while !stop.load(Ordering::Acquire) {
            let mut descriptors = vec![
                libc::pollfd {
                    fd: listener.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                },
                libc::pollfd {
                    fd: notifications.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                },
            ];
            descriptors.extend(readers.iter().map(|(reader, _, _)| libc::pollfd {
                fd: reader.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            }));
            let polled = unsafe {
                libc::poll(
                    descriptors.as_mut_ptr(),
                    descriptors.len() as libc::nfds_t,
                    -1,
                )
            };
            anyhow::ensure!(polled >= 0, "reader endpoint poll failed");
            if descriptors[1].revents != 0 {
                let mut messages = [0_u8; 64];
                let count = unsafe {
                    libc::recv(
                        notifications.as_raw_fd(),
                        messages.as_mut_ptr().cast(),
                        messages.len(),
                        0,
                    )
                };
                anyhow::ensure!(count > 0, "publication notification endpoint closed");
                if stop.load(Ordering::Acquire) {
                    break;
                }
                if published.load(Ordering::Acquire) {
                    for (reader, requested, granted) in &mut readers {
                        if *requested && !*granted {
                            send(reader, b'G')?;
                            *granted = true;
                        }
                    }
                }
            }
            for index in (2..descriptors.len()).rev() {
                if descriptors[index].revents == 0 {
                    continue;
                }
                let (reader, requested, granted) = &mut readers[index - 2];
                let mut packet = [0_u8; 65];
                let received = unsafe {
                    libc::recv(
                        reader.as_raw_fd(),
                        packet.as_mut_ptr().cast(),
                        packet.len(),
                        0,
                    )
                };
                if received == 0 {
                    readers.remove(index - 2);
                    continue;
                }
                anyhow::ensure!(received > 0, "reader endpoint receive failed");
                if *granted {
                    anyhow::ensure!(
                        received == 1 && packet[0] == b'A',
                        "reader did not acknowledge completion"
                    );
                    send(reader, b'D')?;
                    readers.remove(index - 2);
                } else {
                    anyhow::ensure!(
                        !*requested && received == 64,
                        "reader grant has invalid identity length"
                    );
                    *requested = true;
                    if published.load(Ordering::Acquire) {
                        send(reader, b'G')?;
                        *granted = true;
                    }
                }
            }
            if descriptors[0].revents != 0 {
                let client = unsafe {
                    libc::accept4(
                        listener.as_raw_fd(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        libc::SOCK_CLOEXEC | libc::SOCK_NONBLOCK,
                    )
                };
                anyhow::ensure!(client >= 0, "reader endpoint accept failed");
                readers.push((unsafe { OwnedFd::from_raw_fd(client) }, false, false));
            }
        }
        Ok(())
    }))
}
