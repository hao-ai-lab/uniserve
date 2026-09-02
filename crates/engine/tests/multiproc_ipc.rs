//! Multiprocess IPC framing, replay/control idempotency, and rank-respawn qualification.

#![cfg(target_os = "linux")]

use std::collections::BTreeMap;
use std::ffi::CString;
use std::fs::{OpenOptions, remove_file};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use uniserve_core::{BlockId, MediaEvent, MediaGeometry, MediaRequest, RequestId, SamplingParams};
use uniserve_engine::{
    ControlOp, ControlTokens, EngineHandle, Executor, MultiprocExecutor, Scheduler,
    TransferBackend, WorkerExecError, WorkerLossError, WorkerProcessArgs, WorkerRole,
};
use uniserve_worker_ipc::{
    AttentionRegime, Batch, BatchPartition, BlockTable, Bounds, CachePageAllocation, CloseReason,
    Control, DType, DimBound, Disposition, ErrorCode, ForwardMode, NewRequest, OpId, OpStatus,
    Operation, Point, PointRange, ProductKind, ProductPayload, ProductRef, RequestKey, RouteId,
    RowGeometry, SamplingOwnership, ShapeBound, StorageClass, TRANSFER_DESCRIPTOR_PREFIX,
    UndAdmission, VersionRef, encode_token_product_bytes,
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
    let request_capacity = usize::try_from(executor.info().max_request_pool_size)?;
    let scheduler = Scheduler::new(
        Box::new(executor),
        ControlTokens::default(),
        PIPELINE_DEPTH * 8,
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
            let events = handle.submit_media(MediaRequest {
                request_id,
                prompt_token_ids,
                seed: index as u64,
                priority: 0,
                geometry: MediaGeometry {
                    frame_count: 22,
                    video_reconstruction_units: 1,
                    audio_latent_frames: 37,
                    prompt_tokens,
                    denoise_steps: 4,
                },
            })?;
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
                matches!(event, MediaEvent::Failed { .. }),
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
    let info = executor.info();
    assert_eq!(info.rank.tp_rank, 0);
    assert_eq!(info.rank.tp_size, WORLD_SIZE as u32);
    assert_eq!(info.sampling_ownership, SamplingOwnership::DesignatedRank);
    assert_eq!(info.max_batch_operations, 256);
    assert_eq!(info.max_batch_tokens, 256);
    assert_eq!(executor.pipeline_depth(), PIPELINE_DEPTH);

    let admission = text_admission(11, 1, 1)?;
    let root = VersionRef::admission_root(admission.request_key, OpId(0));
    let initial = token_batch(
        1,
        1,
        Some(admission.clone()),
        OpId(1),
        root.clone(),
        ForwardMode::TokenExtend,
        &[7, 8],
        0,
        BlockId(1),
        0,
    );
    let first = execute(&mut executor, initial.clone())?;
    let first_record = &first.partitions[0].completions[0];
    assert_eq!(first_record.status, OpStatus::Ok);
    assert_eq!(first_record.committed_tokens.len(), 1);

    let replayed = execute(&mut executor, initial.clone())?;
    assert_eq!(replayed, first);

    let conflicting = token_batch(
        1,
        1,
        Some(admission.clone()),
        OpId(1),
        root.clone(),
        ForwardMode::TokenExtend,
        &[7, 8, 9],
        0,
        BlockId(1),
        0,
    );
    assert_execution_error(&mut executor, conflicting, "submitted batch")?;

    let selected = fixed_completion(first_record);
    let commit = Control::Commit {
        request_key: admission.request_key,
        control_seq: 1,
        expected_parent: root.clone(),
        selected: selected.clone(),
        public_event_limit: 1,
        disposition: Disposition::Publish,
    };
    let commit_batch = control_batch(2, commit.clone());
    assert!(
        execute(&mut executor, commit_batch.clone())?
            .partitions
            .is_empty()
    );
    assert!(execute(&mut executor, commit_batch)?.partitions.is_empty());

    let gap = Control::Close {
        request_key: admission.request_key,
        control_seq: 3,
        cutoff: selected.clone(),
        reason: CloseReason::Cancelled,
    };
    assert_execution_error(&mut executor, control_batch(3, gap), "does not follow")?;

    let stale_key = RequestKey::new(
        admission.request_key.authority_id,
        admission.request_key.session_id,
        admission.request_key.epoch + 1,
    );
    let stale = Control::Commit {
        request_key: stale_key,
        control_seq: 2,
        expected_parent: with_request_key(&selected, stale_key),
        selected: with_request_key(&selected, stale_key),
        public_event_limit: 1,
        disposition: Disposition::Publish,
    };
    assert_execution_error(&mut executor, control_batch(4, stale), "stale request key")?;

    let cutoff = VersionRef {
        request_key: admission.request_key,
        producer_op_id: OpId(99),
        point: Point::Fixed { point_index: 1 },
    };
    let unreachable = Control::Close {
        request_key: admission.request_key,
        control_seq: 2,
        cutoff,
        reason: CloseReason::Cancelled,
    };
    assert_execution_error(
        &mut executor,
        control_batch(5, unreachable),
        "resolved lineage",
    )?;

    let close = Control::Close {
        request_key: admission.request_key,
        control_seq: 2,
        cutoff: selected.clone(),
        reason: CloseReason::Cancelled,
    };
    let close_batch = control_batch(6, close.clone());
    assert!(
        execute(&mut executor, close_batch.clone())?
            .partitions
            .is_empty()
    );
    assert!(execute(&mut executor, close_batch)?.partitions.is_empty());
    let conflicting_close = Control::Close {
        request_key: admission.request_key,
        control_seq: 2,
        cutoff: selected.clone(),
        reason: CloseReason::Error,
    };
    assert_execution_error(
        &mut executor,
        control_batch(7, conflicting_close),
        "conflicts with its committed content",
    )?;

    let descendant = token_batch(
        8,
        2,
        None,
        OpId(2),
        selected,
        ForwardMode::TokenDecode,
        &[first_record.committed_tokens[0]],
        2,
        BlockId(1),
        2,
    );
    let closed = execute(&mut executor, descendant)?;
    let closed_record = &closed.partitions[0].completions[0];
    assert_eq!(closed_record.status, OpStatus::Error);
    assert_eq!(closed_record.error_code, Some(ErrorCode::InvalidOperation));

    let acknowledgements = executor.control_wait(
        ControlOp::DropSession(admission.request_key.session_id),
        None,
    )?;
    assert_eq!(acknowledgements.len(), WORLD_SIZE);
    assert!(
        acknowledgements
            .iter()
            .all(|acknowledgement| acknowledgement.result.is_ok())
    );
    assert_eq!(executor.in_flight(), 0);
    executor.shutdown();
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
    let first_root = VersionRef::admission_root(first_admission.request_key, OpId(0));
    execute(
        &mut executor,
        token_batch(
            1,
            1,
            Some(first_admission),
            OpId(1),
            first_root,
            ForwardMode::TokenExtend,
            &[3],
            0,
            BlockId(1),
            0,
        ),
    )?;

    let lost_admission = text_admission(22, 1, 2)?;
    let lost_root = VersionRef::admission_root(lost_admission.request_key, OpId(0));
    executor.submit(token_batch(
        2,
        2,
        Some(lost_admission),
        OpId(2),
        lost_root,
        ForwardMode::TokenExtend,
        &[4],
        0,
        BlockId(2),
        0,
    ))?;
    let loss = executor
        .next_result()
        .expect_err("rank loss must be reported");
    assert!(loss.downcast_ref::<WorkerLossError>().is_some(), "{loss:#}");
    assert_eq!(executor.in_flight(), 0);

    let recovered_admission = text_admission(23, 1, 1)?;
    let recovered_root = VersionRef::admission_root(recovered_admission.request_key, OpId(0));
    let recovered = execute(
        &mut executor,
        token_batch(
            3,
            3,
            Some(recovered_admission),
            OpId(3),
            recovered_root,
            ForwardMode::TokenExtend,
            &[5],
            0,
            BlockId(1),
            0,
        ),
    )?;
    assert_eq!(recovered.partitions[0].completions[0].status, OpStatus::Ok);
    executor.shutdown();
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
        worker_role: Some(WorkerRole::Full),
        transfer_backend: TransferBackend::Shm,
        ..config
    })?;

    let slow_admission = text_admission(31, 1, 1)?;
    let slow_root = VersionRef::admission_root(slow_admission.request_key, OpId(0));
    let mut slow = token_batch(
        1,
        2,
        Some(slow_admission.clone()),
        OpId(1),
        slow_root,
        ForwardMode::TokenExtend,
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
    slow.partitions[0].operations[0].predicate = Some(predicate.clone());
    slow.input_products.push(ProductPayload {
        product: predicate,
        bytes: publication.descriptor(91)?,
    });

    let fast_admission = text_admission(32, 1, 2)?;
    let fast_root = VersionRef::admission_root(fast_admission.request_key, OpId(0));
    let fast = token_batch(
        2,
        1,
        Some(fast_admission),
        OpId(1),
        fast_root,
        ForwardMode::TokenExtend,
        &[9],
        0,
        BlockId(2),
        0,
    );

    executor.submit(slow)?;
    assert!(executor.can_submit());
    executor.submit(fast)?;
    let first = executor.next_result()?;
    assert_eq!(first.step_id, 2);
    assert!(!publication.published.load(Ordering::Acquire));
    assert_eq!(first.partitions[0].completions[0].status, OpStatus::Ok);

    publication.publish()?;
    let second = executor.next_result()?;
    assert_eq!(second.step_id, 1);
    assert_eq!(second.partitions[0].completions[0].status, OpStatus::Ok);
    publication.finish()?;
    executor.shutdown();
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

    fn descriptor(&self, generation: u32) -> anyhow::Result<Vec<u8>> {
        let mut metadata = BTreeMap::new();
        metadata.insert("generation", serde_json::json!(generation));
        metadata.insert("height", serde_json::json!(0));
        metadata.insert("ready_header_bytes", serde_json::json!(1));
        metadata.insert(
            "ready_semaphore",
            serde_json::json!(self.semaphore_name.to_string_lossy()),
        );
        metadata.insert("value_range", serde_json::json!(""));
        metadata.insert("width", serde_json::json!(0));

        let mut locator = BTreeMap::new();
        locator.insert("device", serde_json::json!("cpu"));
        locator.insert("dtype", serde_json::json!("uint8"));
        locator.insert(
            "handle_b64",
            serde_json::json!(base64::engine::general_purpose::STANDARD.encode(&self.name)),
        );
        locator.insert("meta", serde_json::to_value(metadata)?);
        locator.insert("nbytes", serde_json::json!(1));
        locator.insert("session", serde_json::json!("shm"));
        locator.insert("shape", serde_json::json!([1]));
        locator.insert("transport", serde_json::json!("shm"));
        locator.insert("version", serde_json::json!(1));

        let mut value = BTreeMap::new();
        value.insert("generation", serde_json::json!(generation));
        value.insert("height", serde_json::json!(0));
        value.insert("locator", serde_json::to_value(locator)?);
        value.insert("value_range", serde_json::json!(""));
        value.insert("width", serde_json::json!(0));

        let mut envelope = BTreeMap::new();
        envelope.insert("kind", serde_json::json!("device_product"));
        envelope.insert("value", serde_json::to_value(value)?);

        let mut descriptor = TRANSFER_DESCRIPTOR_PREFIX.to_vec();
        descriptor.extend(serde_json::to_vec(&envelope)?);
        Ok(descriptor)
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
        worker_role: None,
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
) -> anyhow::Result<uniserve_worker_ipc::CompletionReport> {
    assert!(executor.can_submit());
    executor.submit(batch)?;
    executor.next_result()
}

fn assert_execution_error(
    executor: &mut MultiprocExecutor,
    batch: Batch,
    message: &str,
) -> anyhow::Result<()> {
    assert!(executor.can_submit());
    executor.submit(batch)?;
    let error = executor
        .next_result()
        .expect_err("submission must be rejected");
    let execution = error
        .downcast_ref::<WorkerExecError>()
        .ok_or_else(|| anyhow::anyhow!("expected a typed worker execution error: {error:#}"))?;
    assert_eq!(execution.code.as_deref(), Some("InvalidDescriptor"));
    assert!(execution.message.contains(message), "{execution}");
    assert_eq!(executor.in_flight(), 0);
    Ok(())
}

fn text_admission(
    session_id: u64,
    epoch: u64,
    request_pool_idx: u32,
) -> anyhow::Result<NewRequest> {
    Ok(NewRequest::new(
        RequestKey::new(1, RequestId(session_id), epoch),
        request_pool_idx,
        Some(UndAdmission {
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
    step_id: u64,
    collective_seq: u64,
    admission: Option<NewRequest>,
    op_id: OpId,
    parent: VersionRef,
    mode: ForwardMode,
    tokens: &[u32],
    control_seq: u64,
    page: BlockId,
    prefix_length: u32,
) -> Batch {
    let request_key = parent.request_key;
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
        work: mode,
        route: RouteId(0),
        domain: mode.domain(),
        advances_state: false,
        bounds: Bounds {
            max_points: 1,
            max_tokens: tokens.len().max(1) as u32,
            max_kv_pages: u32::from(prefix_length == 0),
            ..Bounds::default()
        },
        inputs: vec![input.clone()],
        outputs: vec![token_output],
        predicate: None,
        rng: None,
        control_seq,
    }
    .sealed();
    let input_length = tokens.len() as u32;
    let partition = BatchPartition {
        partition_id: 1,
        submission_group: 1,
        collective_seq,
        domain: operation.domain,
        route: operation.route,
        attention: AttentionRegime::Causal,
        shape_class: 0,
        operations: vec![operation],
        block_tables: vec![BlockTable {
            request_pool_idx,
            group_id: 0,
            page_ids: vec![page],
            allocated_tokens: prefix_length + input_length,
        }],
        new_cache_pages: if prefix_length == 0 {
            vec![CachePageAllocation {
                request_pool_idx,
                group_id: 0,
                page_ids: vec![page],
            }]
        } else {
            Vec::new()
        },
        forward_rows: vec![RowGeometry {
            operation_index: 0,
            request_pool_index: request_pool_idx,
            seq_len: prefix_length,
            query_len: input_length.max(1),
        }],
        latent_placements: Vec::new(),
        reconstruction_placements: Vec::new(),
    };
    Batch::new(step_id, admission.into_iter().collect(), vec![partition]).with_input_products(vec![
        ProductPayload {
            product: input,
            bytes: encode_token_product_bytes(tokens),
        },
    ])
}

fn control_batch(step_id: u64, control: Control) -> Batch {
    Batch::new(step_id, Vec::new(), Vec::new()).with_controls(vec![control])
}

fn fixed_completion(record: &uniserve_worker_ipc::ModelOutput) -> VersionRef {
    VersionRef {
        request_key: record.request_key,
        producer_op_id: record.op_id,
        point: Point::Fixed {
            point_index: record.selected_point,
        },
    }
}

fn with_request_key(version: &VersionRef, request_key: RequestKey) -> VersionRef {
    VersionRef {
        request_key,
        producer_op_id: version.producer_op_id,
        point: version.point.clone(),
    }
}
