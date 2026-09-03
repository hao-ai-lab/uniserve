//! Multiprocess IPC framing, replay/control idempotency, and rank-respawn qualification.

#![cfg(target_os = "linux")]

use std::ffi::CString;
use std::fs::{OpenOptions, remove_file};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use uniserve_core::{
    BlockId, DiffusionRequest, Event, MediaGeometry, Request, RequestId, RuntimeFamily,
    SamplingParams,
};
use uniserve_engine::{
    ControlTokens, EngineHandle, EngineLoop, Executor, MultiprocExecutor, PhysicalExecutor,
    SchedulerConfig, TransferBackend, WorkerExecError, WorkerLossError, WorkerProcessArgs,
};
use uniserve_worker_ipc::{
    ArRequestParams, BatchCommand, BlockTable, Bounds, CachePageAllocation, Checkpoint,
    CheckpointPoint, CloseReason, DType, DimBound, Disposition, ErrorCode, InlineValue, NewRequest,
    OpId, OpPayload, OpStatus, Operation, PointRange, ProductKind, ProductPayload, ProductRef,
    RequestKey, RowGeometry, Run as Batch, RunKind, ShapeBound, StorageClass, TransferHandle,
    TransferLocator, TransferTransport, encode_token_product_bytes,
};

const WORLD_SIZE: usize = 2;
const PIPELINE_DEPTH: usize = 2;

#[test]
fn multiprocess_topology_handles_rank_failure_and_capacity_limits() -> anyhow::Result<()> {
    check_rank_ipc()?;
    qualify_failed_media_admission_reclamation()?;
    qualify_slow_transfer()?;
    qualify_peer_replacement()?;
    Ok(())
}

fn qualify_failed_media_admission_reclamation() -> anyhow::Result<()> {
    let executor = spawn_rank_group_with_slot_capacity(128 << 10)?;
    let request_capacity = usize::try_from(executor.info().single_pool().request_slots)?;
    let scheduler = EngineLoop::with_config_for_family(
        Box::new(executor),
        ControlTokens::default(),
        SchedulerConfig {
            max_batch: PIPELINE_DEPTH * 8,
            ..SchedulerConfig::default()
        },
        RuntimeFamily::Diffusion,
    );
    let (command_tx, command_rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(command_tx);
    let scheduler_thread = thread::spawn(move || scheduler.run(command_rx));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()?;

    let result = (|| -> anyhow::Result<()> {
        let mut requests = Vec::with_capacity(request_capacity + 1);
        for index in 0..=request_capacity {
            let request_id = RequestId(10_000 + index as u64);
            let prompt_token_ids = if index == 0 {
                (0..16_384_u32)
                    .map(|token| token.saturating_add(100_000))
                    .collect()
            } else {
                vec![100_000 + index as u32]
            };
            let prompt_tokens = u32::try_from(prompt_token_ids.len())?;
            let events = handle.submit(Request::Diffusion(DiffusionRequest {
                request_id,
                prompt_token_ids,
                seed: index as u64,
                priority: 0,
                geometry: MediaGeometry {
                    frame_count: 22,
                    decode_units: 3,
                    prompt_tokens,
                    denoise_steps: 4,
                },
            }))?;
            requests.push((request_id, events));
        }
        for (request_id, mut events) in requests {
            let event = runtime.block_on(async {
                tokio::time::timeout(Duration::from_secs(10), events.recv()).await
            })?;
            let event = event.ok_or_else(|| {
                anyhow::anyhow!("media event stream closed for request {request_id:?}")
            })?;
            assert!(
                matches!(event, Event::Error { .. }),
                "request {request_id:?} did not reach terminal failure: {event:?}"
            );
        }
        Ok(())
    })();

    handle.shutdown();
    let engine_died = scheduler_thread
        .join()
        .map_err(|_| anyhow::anyhow!("scheduler thread panicked"))?;
    result?;
    assert!(!engine_died, "media admission failures killed the engine");
    Ok(())
}

fn check_rank_ipc() -> anyhow::Result<()> {
    let mut executor = spawn_rank_group()?;
    let info = executor.info().single_pool();
    assert_eq!(info.rank.tp_rank, 0);
    assert_eq!(info.rank.tp_size, WORLD_SIZE as u32);
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
        RunKind::ArExtend,
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
        RunKind::ArExtend,
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
    };
    let close_batch = command_batch(6, close.clone());
    assert!(
        execute(&mut executor, close_batch.clone())?
            .completions
            .is_empty()
    );
    assert!(execute(&mut executor, close_batch)?.completions.is_empty());
    let conflicting_close = BatchCommand::Finish {
        request_key: admission.request_key,
        control_seq: 2,
        cutoff: selected.clone(),
        reason: CloseReason::Error,
    };
    assert_execution_error(
        &mut executor,
        command_batch(7, conflicting_close),
        "conflicts with its committed content",
    )?;

    let descendant = token_batch(
        8,
        2,
        admission.request_key,
        None,
        OpId(2),
        selected,
        RunKind::ArDecode,
        &[first_record.committed_tokens()[0]],
        2,
        BlockId(1),
        2,
    );
    let closed = execute(&mut executor, descendant)?;
    let closed_record = &closed.completions[0];
    assert_eq!(closed_record.status, OpStatus::Error);
    assert_eq!(closed_record.error_code, Some(ErrorCode::InvalidOperation));

    executor.close_physical()?;
    Ok(())
}

fn qualify_peer_replacement() -> anyhow::Result<()> {
    // This integration-test process has one test thread and no live worker
    // group at this point, so changing its child-launch environment cannot
    // race another environment access in this process.
    unsafe {
        std::env::set_var("UNISERVE_STUB_DIE_AFTER", "1");
        std::env::set_var("UNISERVE_STUB_DIE_RANK", "1");
    }
    let mut executor = spawn_rank_group()?;
    unsafe {
        std::env::remove_var("UNISERVE_STUB_DIE_AFTER");
        std::env::remove_var("UNISERVE_STUB_DIE_RANK");
    }

    let first_admission = text_admission(21, 1, 1)?;
    let first_root = Checkpoint::admission_root(OpId(0));
    execute(
        &mut executor,
        token_batch(
            1,
            1,
            first_admission.request_key,
            Some(first_admission),
            OpId(1),
            first_root,
            RunKind::ArExtend,
            &[3],
            0,
            BlockId(1),
            0,
        ),
    )?;

    let lost_admission = text_admission(22, 1, 2)?;
    let lost_root = Checkpoint::admission_root(OpId(0));
    executor.submit_run(token_batch(
        2,
        2,
        lost_admission.request_key,
        Some(lost_admission),
        OpId(2),
        lost_root,
        RunKind::ArExtend,
        &[4],
        0,
        BlockId(2),
        0,
    ))?;
    let loss = executor
        .poll_run(Duration::from_secs(30))
        .expect_err("rank loss must be reported");
    assert!(loss.downcast_ref::<WorkerLossError>().is_some(), "{loss:#}");

    let recovered_admission = text_admission(23, 1, 1)?;
    let recovered_root = Checkpoint::admission_root(OpId(0));
    let recovered = execute(
        &mut executor,
        token_batch(
            3,
            3,
            recovered_admission.request_key,
            Some(recovered_admission),
            OpId(3),
            recovered_root,
            RunKind::ArExtend,
            &[5],
            0,
            BlockId(1),
            0,
        ),
    )?;
    assert_eq!(recovered.completions[0].status, OpStatus::Ok);
    executor.close_physical()?;
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
    let mut executor = MultiprocExecutor::spawn(WorkerProcessArgs {
        python: worker,
        model: String::new(),
        device: "cpu".into(),
        world_size: WORLD_SIZE,
        pipeline_depth: PIPELINE_DEPTH,
        req_slot_cap: 1 << 20,
        resp_slot_cap: 8 << 20,
        kv_token_capacity: Some(4096),
        block_size: 16,
        max_batch_operations: 256,
        max_batch_tokens: 256,
        attention_backend: uniserve_worker_ipc::AttentionBackend::TorchSdpa,
        supported_ops: uniserve_worker_ipc::OpKind::ALL.to_vec(),
        transfer_backend: TransferBackend::Shm,
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
        RunKind::ArExtend,
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
        RunKind::ArExtend,
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
    publication.finish()?;
    executor.close_physical()?;
    assert!(publication.path.is_file());
    Ok(())
}

struct SlowShmPublication {
    name: String,
    semaphore_name: CString,
    path: PathBuf,
    published: Arc<AtomicBool>,
    release: Arc<(Mutex<bool>, Condvar)>,
    publisher: Option<thread::JoinHandle<anyhow::Result<()>>>,
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
        file.set_len(2)?;
        file.write_all_at(&[0, 0], 0)?;

        let semaphore_name = CString::new(format!("/{name}"))?;
        let semaphore = unsafe {
            libc::sem_open(
                semaphore_name.as_ptr(),
                libc::O_CREAT | libc::O_EXCL,
                0o600,
                0,
            )
        };
        if semaphore == libc::SEM_FAILED {
            let error = std::io::Error::last_os_error();
            let _ = remove_file(&path);
            return Err(error.into());
        }
        if unsafe { libc::sem_close(semaphore) } != 0 {
            let error = std::io::Error::last_os_error();
            unsafe {
                libc::sem_unlink(semaphore_name.as_ptr());
            }
            let _ = remove_file(&path);
            return Err(error.into());
        }

        let published = Arc::new(AtomicBool::new(false));
        let publisher_path = path.clone();
        let publisher_semaphore = semaphore_name.clone();
        let publisher_state = Arc::clone(&published);
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let publisher_release = Arc::clone(&release);
        let publisher = thread::spawn(move || {
            let publish_result = (|| -> anyhow::Result<()> {
                let (lock, changed) = &*publisher_release;
                let released = lock
                    .lock()
                    .map_err(|_| anyhow::anyhow!("shared-memory publication gate is poisoned"))?;
                let released = changed
                    .wait_while(released, |value| !*value)
                    .map_err(|_| anyhow::anyhow!("shared-memory publication gate is poisoned"))?;
                anyhow::ensure!(
                    *released,
                    "shared-memory publication gate closed without release"
                );
                drop(released);
                let file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&publisher_path)?;
                file.write_all_at(&[1], 1)?;
                file.write_all_at(&[1], 0)?;
                publisher_state.store(true, Ordering::Release);
                Ok(())
            })();
            if publish_result.is_err()
                && let Ok(file) = OpenOptions::new().write(true).open(&publisher_path)
            {
                let _ = file.write_all_at(&[2], 0);
            }
            let signal_result = post_named_semaphore(&publisher_semaphore);
            publish_result.and(signal_result)
        });
        Ok(Self {
            name,
            semaphore_name,
            path,
            published,
            release,
            publisher: Some(publisher),
        })
    }

    fn publish(&self) -> anyhow::Result<()> {
        let (lock, changed) = &*self.release;
        let mut released = lock
            .lock()
            .map_err(|_| anyhow::anyhow!("shared-memory publication gate is poisoned"))?;
        *released = true;
        changed.notify_all();
        Ok(())
    }

    fn descriptor(&self, generation: u32) -> anyhow::Result<TransferHandle> {
        Ok(TransferHandle::DeviceProduct {
            generation,
            height: 0,
            width: 0,
            value_range: String::new(),
            locator: TransferLocator {
                transport: TransferTransport::PosixShm {
                    name: self.name.clone(),
                    ready_header_bytes: 1,
                    ready_semaphore: Some(self.semaphore_name.to_string_lossy().into_owned()),
                },
                nbytes: 1,
                dtype: "uint8".to_owned(),
                shape: vec![1],
                device: "cpu".to_owned(),
            },
        })
    }

    fn finish(&mut self) -> anyhow::Result<()> {
        let publisher = self
            .publisher
            .take()
            .ok_or_else(|| anyhow::anyhow!("shared-memory publication already joined"))?;
        publisher
            .join()
            .map_err(|_| anyhow::anyhow!("shared-memory publisher panicked"))?
    }
}

impl Drop for SlowShmPublication {
    fn drop(&mut self) {
        let _ = self.publish();
        if let Some(publisher) = self.publisher.take() {
            let _ = publisher.join();
        }
        unsafe {
            libc::sem_unlink(self.semaphore_name.as_ptr());
        }
        let _ = remove_file(&self.path);
    }
}

fn post_named_semaphore(name: &CString) -> anyhow::Result<()> {
    let semaphore = unsafe { libc::sem_open(name.as_ptr(), 0) };
    if semaphore == libc::SEM_FAILED {
        return Err(std::io::Error::last_os_error().into());
    }
    let post_result = if unsafe { libc::sem_post(semaphore) } == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error().into())
    };
    let close_result = if unsafe { libc::sem_close(semaphore) } == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error().into())
    };
    post_result.and(close_result)
}

fn spawn_rank_group() -> anyhow::Result<MultiprocExecutor> {
    spawn_rank_group_with_capacities(1 << 20, 8 << 20)
}

fn spawn_rank_group_with_slot_capacity(slot_capacity: usize) -> anyhow::Result<MultiprocExecutor> {
    spawn_rank_group_with_capacities(slot_capacity, slot_capacity)
}

fn spawn_rank_group_with_capacities(
    request_slot_capacity: usize,
    response_slot_capacity: usize,
) -> anyhow::Result<MultiprocExecutor> {
    let worker = worker_python();
    let config = WorkerProcessArgs {
        stub: true,
        cuda_graph: false,
        prefill_cuda_graph: false,
        ..WorkerProcessArgs::default()
    };
    MultiprocExecutor::spawn(WorkerProcessArgs {
        python: worker,
        model: String::new(),
        device: "cpu".into(),
        world_size: WORLD_SIZE,
        pipeline_depth: PIPELINE_DEPTH,
        req_slot_cap: request_slot_capacity,
        resp_slot_cap: response_slot_capacity,
        kv_token_capacity: Some(4096),
        block_size: 16,
        max_batch_operations: 256,
        max_batch_tokens: 256,
        attention_backend: uniserve_worker_ipc::AttentionBackend::TorchSdpa,
        supported_ops: uniserve_worker_ipc::OpKind::ALL.to_vec(),
        transfer_backend: TransferBackend::Inproc,
        ..config
    })
}

fn worker_python() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(".venv/bin/python")
}

fn execute(
    executor: &mut MultiprocExecutor,
    batch: Batch,
) -> anyhow::Result<uniserve_worker_ipc::RunResult> {
    executor.submit_run(batch)?;
    executor
        .poll_run(Duration::from_secs(30))?
        .ok_or_else(|| anyhow::anyhow!("submission did not complete"))
}

fn assert_execution_error(
    executor: &mut MultiprocExecutor,
    batch: Batch,
    message: &str,
) -> anyhow::Result<()> {
    executor.submit_run(batch)?;
    let error = executor
        .poll_run(Duration::from_secs(30))
        .expect_err("submission must be rejected");
    let execution = error
        .downcast_ref::<WorkerExecError>()
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
    mode: RunKind,
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
        parent,
        kind: mode,
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
