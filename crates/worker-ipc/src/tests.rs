//! Protocol round trips and validation behavior for every message family.

use std::collections::BTreeMap;

use uniserve_core::{BlockId, CfgRenorm, KvGroupKind, RequestId};

use super::*;
use crate::codec::{decode_request, decode_response, encode_request, encode_response};

fn request_key() -> RequestKey {
    RequestKey::new(4, RequestId(7), 2)
}

fn fixed_parent() -> Checkpoint {
    Checkpoint::admission_root(OpId(1))
}

fn output_product(op: OpId) -> ProductRef {
    ProductRef {
        request_key: request_key(),
        producer_op_id: op,
        output_index: 0,
        generation: 3,
        kind: ProductKind::Token,
        storage_class: StorageClass::RequestRelay,
        dtype: DType::U32,
        shape_bound: ShapeBound::default(),
        point_range: PointRange {
            base_point: 0,
            max_points: 1,
        },
    }
}

fn selected_point_product(op: OpId) -> ProductRef {
    ProductRef {
        request_key: request_key(),
        producer_op_id: op,
        output_index: 1,
        generation: 4,
        kind: ProductKind::SelectedPoint,
        storage_class: StorageClass::RequestRelay,
        dtype: DType::U32,
        shape_bound: ShapeBound::default(),
        point_range: PointRange {
            base_point: 0,
            max_points: 1,
        },
    }
}

fn ar_decode_operation() -> Operation {
    let kind = OpCode::ArDecode;
    Operation {
        request_key: request_key(),
        op_id: OpId(11),
        parent: Some(fixed_parent()),
        entry: "model".into(),
        payload: OpPayload::new(
            kind,
            Bounds {
                max_points: 1,
                max_tokens: 1,
                max_kv_pages: 1,
                ..Bounds::default()
            },
            Vec::new(),
            vec![output_product(OpId(11)), selected_point_product(OpId(11))],
            None,
            Some(Rng {
                seed: 99,
                semantic_index_base: 4,
                draw_layout: DrawLayout::TargetSampling,
            }),
            0,
        ),
    }
    .sealed()
}

fn operation_for(kind: OpCode, op_id: OpId, advances: bool) -> Operation {
    Operation {
        request_key: request_key(),
        op_id,
        parent: Some(fixed_parent()),
        entry: "model".into(),
        payload: OpPayload::new(
            kind,
            Bounds {
                max_points: if advances { 1 } else { 0 },
                ..Bounds::default()
            },
            Vec::new(),
            Vec::new(),
            None,
            None,
            0,
        ),
    }
    .sealed()
}

fn completion_record() -> ModelOutput {
    ModelOutput {
        request_key: request_key(),
        op_id: OpId(11),
        completion_slot_generation: 2,
        status: OpStatus::Ok,
        selected_point: 1,
        product_generations: vec![3, 4],
        error_code: None,
        timing_counters: TimingCounters::default(),
        payload: ResultPayload::Ar(ResultData {
            logical_lengths: LogicalLengths {
                token_len: 5,
                kv_visible_len: 5,
                kv_computed_len: 5,
                latent_len: 0,
            },
            token_span: TokenSpan { base: 4, len: 1 },
            committed_tokens: vec![271],
            finish_flags: FinishFlags::default(),
            media_output: None,
        }),
    }
}

fn admission() -> NewRequest {
    NewRequest::new(
        request_key(),
        u32::try_from(request_key().request_id.0).unwrap(),
        Some(ArRequestParams {
            sampling: SamplingParams::default(),
            negative_token_ids: Vec::new(),
            finish_token_ids: vec![2, 7],
            initial_position: 0,
        }),
        None,
    )
    .unwrap()
}

fn media_admission(prompt_token_ids: Vec<u32>) -> NewRequest {
    NewRequest::new_media(
        request_key(),
        u32::try_from(request_key().request_id.0).unwrap(),
        DiffusionRequestParams {
            geometry: MediaGeometry {
                frame_count: 22,
                video_units: 3,
                prompt_tokens: u32::try_from(prompt_token_ids.len()).unwrap(),
                denoise_steps: 4,
            },
            prompt_token_ids,
            seed: 17,
        },
    )
    .unwrap()
}

fn execute_round_trip(batch: Run) -> Run {
    let request = WorkerRequest::submit(batch);
    let decoded = decode_request(&encode_request(&request).unwrap()).unwrap();
    decoded.run().unwrap().clone()
}

fn batch_with_operations(
    run_id: u64,
    admissions: Vec<NewRequest>,
    operations: Vec<Operation>,
) -> Run {
    let mut run = Run::new(run_id, admissions, operations);
    let mut next_buffer_offset = 0_u64;
    for (operation_index, operation) in run.operations.iter().enumerate() {
        for output in operation
            .outputs()
            .iter()
            .filter(|output| output.uses_persistent_buffer())
        {
            next_buffer_offset = next_buffer_offset.div_ceil(256) * 256;
            let bytes = output.max_bytes();
            run.buffer_allocations.push(BufferAllocation {
                buffer: output.buffer_id(),
                offset: next_buffer_offset,
                bytes,
            });
            next_buffer_offset += bytes;
        }
        let capacity_pages = operation.bounds().max_kv_pages;
        if capacity_pages > 0 {
            let request_pool_idx = u32::try_from(operation.request_key.request_id.0).unwrap();
            let page_ids = (1..=capacity_pages).map(BlockId).collect::<Vec<_>>();
            run.block_tables.push(BlockTable {
                request_pool_idx,
                group_id: 0,
                page_ids: page_ids.clone(),
                allocated_tokens: operation.bounds().max_tokens.max(1),
            });
            run.new_cache_pages.push(CachePageAllocation {
                request_pool_idx,
                group_id: 0,
                page_ids,
            });
            run.forward_rows.push(RowGeometry {
                operation_index: operation_index as u32,
                request_pool_index: request_pool_idx,
                seq_len: 0,
                query_len: operation.bounds().max_tokens.max(1),
                write_kv: true,
            });
        }
        if matches!(
            operation.kind(),
            OpCode::DiffusionPrepare | OpCode::DiffusionStep
        ) || operation
            .inputs()
            .iter()
            .any(|reference| reference.kind == ProductKind::Latent)
        {
            run.latent_params.push(LatentParams {
                request_key: operation.request_key,
                op_id: operation.op_id,
                page_table: vec![u32::try_from(operation.op_id.0).unwrap()],
                latent_units: 1,
                height: 1,
                width: 1,
                start_step: 0,
                step_count: u32::from(operation.kind() == OpCode::DiffusionStep),
            });
        }
        if matches!(
            operation.kind(),
            OpCode::DiffusionDecode | OpCode::MediaAppend
        ) {
            run.decode_ranges.push(DecodeRange {
                track: MediaTrack::Video,
                request_key: operation.request_key,
                op_id: operation.op_id,
                cursor: 0,
                max_units: 1,
            });
        }
    }
    run
}

fn lane_report(
    run_id: u64,
    completions: Vec<ModelOutput>,
    products: Vec<ProductPayload>,
    visible: bool,
    worker_exec_us: Option<u64>,
    forward_stats: Option<WorkerForwardStats>,
) -> RunResult {
    RunResult {
        batch_id: run_id,
        run_id,
        completions,
        products,
        registration: RegistrationAck { visible },
        worker_exec_us,
        forward_stats,
        done: true,
    }
}

#[test]
fn every_work_variant_round_trips_through_ipc() {
    let variants = [
        (OpCode::ArExtend, true, Domain::Prefill),
        (OpCode::ArDecode, true, Domain::Decode),
        (OpCode::ArVerify, true, Domain::Decode),
        (OpCode::EncoderVision, false, Domain::Prefill),
        (OpCode::EncoderLatent, false, Domain::Prefill),
        (OpCode::TransferProduct, false, Domain::Prefill),
        (OpCode::TransferKvPublish, false, Domain::Prefill),
        (OpCode::TransferKvInstall, false, Domain::Prefill),
        (OpCode::DiffusionPrepare, true, Domain::Flow),
        (OpCode::DiffusionStep, true, Domain::Flow),
        (OpCode::DiffusionDecode, false, Domain::Flow),
        (OpCode::DiffusionFinalize, false, Domain::Flow),
        (OpCode::MediaAppend, false, Domain::Flow),
    ];
    for (index, (work, advances, domain)) in variants.into_iter().enumerate() {
        assert_eq!(
            work.advances_state(),
            advances,
            "work table effect mismatch"
        );
        assert_eq!(work.domain(), domain, "work table domain mismatch");
        let operation = operation_for(work, OpId(100 + index as u64), advances);
        let batch = execute_round_trip(batch_with_operations(
            1,
            Vec::new(),
            vec![operation.clone()],
        ));
        assert_eq!(batch.operations().next().unwrap(), &operation);
        assert_eq!(batch.operations().next().unwrap().kind(), work);
    }
}

#[test]
fn solver_parameters_round_trip_with_request_or_paged_storage() {
    let operation = operation_for(OpCode::DiffusionStep, OpId(101), true);
    let mut run = batch_with_operations(1, Vec::new(), vec![operation]);
    assert_eq!(execute_round_trip(run.clone()), run);
    run.latent_params[0].page_table.clear();
    run.latent_params[0].latent_units = 0;
    assert_eq!(execute_round_trip(run.clone()), run);
    run.latent_params[0].latent_units = 1;
    assert!(run.validate().is_err());
    run.latent_params[0].latent_units = 0;
    run.latent_params[0].page_table.push(1);
    assert!(run.validate().is_err());
}

#[test]
fn media_tracks_preserve_independent_ranges_and_tensor_dependencies() {
    let mut latent = output_product(OpId(55));
    latent.kind = ProductKind::Tensor;
    latent.storage_class = StorageClass::DeviceTensor;
    latent.dtype = DType::F32;
    latent.shape_bound = ShapeBound {
        dims: vec![DimBound::Static(8), DimBound::Static(32)],
    };
    latent.point_range = PointRange::default();
    for (index, entry) in ["video_decoder", "audio_decoder"].iter().enumerate() {
        let operation = Operation {
            request_key: request_key(),
            op_id: OpId(56 + index as u64),
            parent: None,
            entry: (*entry).into(),
            payload: OpPayload::new(
                OpCode::DiffusionDecode,
                Bounds::default(),
                vec![latent.clone()],
                Vec::new(),
                None,
                None,
                0,
            ),
        }
        .sealed();
        let mut run = batch_with_operations(index as u64 + 1, Vec::new(), vec![operation]);
        if index == 0 {
            run.decode_ranges[0].cursor = 3;
            run.decode_ranges[0].max_units = 2;
        } else {
            run.decode_ranges[0].track = MediaTrack::Audio;
        }
        assert_eq!(execute_round_trip(run.clone()), run);
        if index == 1 {
            run.decode_ranges[0].cursor = 1;
            assert!(run.validate().is_err());
        }
        run.decode_ranges.clear();
        assert!(run.validate().is_err());
    }
}

#[test]
fn device_selected_checkpoint_round_trips() {
    let device_parent = Checkpoint {
        op_id: OpId(9),
        point: CheckpointPoint::DeviceSelected,
    };
    let operation = Operation {
        request_key: request_key(),
        op_id: OpId(12),
        parent: Some(device_parent.clone()),
        entry: "model".into(),
        payload: OpPayload::new(
            OpCode::ArDecode,
            Bounds {
                max_points: 1,
                ..Bounds::default()
            },
            Vec::new(),
            vec![output_product(OpId(12))],
            None,
            None,
            0,
        ),
    }
    .sealed();
    let batch = execute_round_trip(batch_with_operations(2, Vec::new(), vec![operation]));
    assert_eq!(
        batch.operations().next().unwrap().parent,
        Some(device_parent)
    );
}

#[test]
fn block_table_allocation_round_trips_with_the_operation() {
    let base = ar_decode_operation();
    let batch = execute_round_trip(batch_with_operations(9, Vec::new(), vec![base.clone()]));
    assert_eq!(batch.operations().next().unwrap(), &base);
    assert_eq!(batch.block_tables[0].page_ids, vec![BlockId(1)]);
}

#[test]
fn run_result_round_trips_records_and_product_payloads() {
    let mut logprob = output_product(OpId(11));
    logprob.output_index = 2;
    logprob.kind = ProductKind::Logprob;
    logprob.storage_class = StorageClass::HostStaging;
    logprob.dtype = DType::F32;
    logprob.shape_bound = ShapeBound::default();
    let report = lane_report(
        5,
        vec![completion_record()],
        vec![ProductPayload {
            product: logprob,
            value: InlineValue::Bytes(vec![1, 2, 3, 4]),
        }],
        true,
        Some(10),
        None,
    );
    let response = WorkerResponse::result(report.clone());
    let decoded = decode_response(&encode_response(&response).unwrap()).unwrap();
    assert_eq!(decoded.report().unwrap(), &report);
}

#[test]
fn publication_round_trips_its_registered_view_and_endpoint() {
    for kind in [ProductKind::Artifact, ProductKind::Tensor] {
        let mut product = output_product(OpId(11));
        product.kind = kind;
        product.storage_class = if kind == ProductKind::Tensor {
            StorageClass::DeviceTensor
        } else {
            StorageClass::LatentArena
        };
        product.dtype = DType::F32;
        product.shape_bound = ShapeBound {
            dims: vec![DimBound::Static(4)],
        };
        for transport in [
            TransferTransport::CudaIpc {
                endpoint: "uniserve-cuda-physical-incarnation".into(),
                publication_id: "0123456789abcdef0123456789abcdef".into(),
                storage_handle: vec![7; 64],
                storage_size_bytes: 4096,
                storage_offsets_bytes: vec![128, 64],
                span_lengths: vec![2],
                span_counts: vec![2],
                tensor_stride: vec![2],
                ready_event_handle: vec![9; 64],
            },
            TransferTransport::PosixShm {
                endpoint: "uniserve-shm-physical-incarnation".into(),
                name: "uniserve-publication".into(),
            },
        ] {
            let mut report = lane_report(
                5,
                vec![completion_record()],
                vec![ProductPayload {
                    product: product.clone(),
                    value: InlineValue::Transfer(TransferHandle::DeviceProduct {
                        generation: 3,
                        height: 0,
                        width: 0,
                        value_range: String::new(),
                        tensor: TensorTransfer {
                            shape: vec![4],
                            locations: vec![Locator {
                                source: WorkerInfo::default().endpoint,
                                transport,
                                nbytes: 16,
                                dtype: "float32".into(),
                                shape: vec![4],
                                offset: vec![0],
                                device: "cuda:1".into(),
                            }],
                        },
                    }),
                }],
                true,
                None,
                None,
            );
            let InlineValue::Transfer(original) = &mut report.products[0].value else {
                unreachable!()
            };
            let mut replica = original.clone();
            if let TransferHandle::DeviceProduct { tensor, .. } = &mut replica {
                tensor.locations[0].source.rank = 1;
            }
            original.merge_locations(&replica).unwrap();
            let decoded =
                decode_response(&encode_response(&WorkerResponse::result(report.clone())).unwrap())
                    .unwrap();
            assert_eq!(decoded.report().unwrap(), &report);
            for (dtype, storage_dtype, nbytes) in
                [(DType::U32, "int64", 32), (DType::I16, "int16", 8)]
            {
                let mut typed_report = report.clone();
                typed_report.products[0].product.dtype = dtype;
                let InlineValue::Transfer(TransferHandle::DeviceProduct { tensor, .. }) =
                    &mut typed_report.products[0].value
                else {
                    unreachable!()
                };
                for location in &mut tensor.locations {
                    location.dtype = storage_dtype.into();
                    location.nbytes = nbytes;
                }
                let decoded = decode_response(
                    &encode_response(&WorkerResponse::result(typed_report.clone())).unwrap(),
                )
                .unwrap();
                assert_eq!(decoded.report().unwrap(), &typed_report);
            }
        }
    }
}

#[test]
fn unchanged_kv_publication_round_trips_without_physical_tensors() {
    let mut product = output_product(OpId(11));
    product.kind = ProductKind::Kv;
    product.storage_class = StorageClass::PagedKv;
    product.dtype = DType::U8;
    product.shape_bound = ShapeBound {
        dims: vec![DimBound::Static(4096)],
    };
    let mut report = lane_report(
        5,
        vec![completion_record()],
        vec![ProductPayload {
            product,
            value: InlineValue::Transfer(TransferHandle::Kv {
                generation: 3,
                tensors: Vec::new(),
                source: fixed_parent(),
                destination: "decoder".into(),
                base: Some(fixed_parent()),
                base_extent: 16,
                published_extent: 16,
                group_id: 0,
                compute_dtype: "bfloat16".into(),
                page_size: 16,
            }),
        }],
        true,
        None,
        None,
    );
    let decoded =
        decode_response(&encode_response(&WorkerResponse::result(report.clone())).unwrap())
            .unwrap();
    assert_eq!(decoded.report().unwrap(), &report);

    let InlineValue::Transfer(TransferHandle::Kv {
        published_extent, ..
    }) = &mut report.products[0].value
    else {
        unreachable!()
    };
    *published_extent = 17;
    assert!(encode_response(&WorkerResponse::result(report)).is_err());
}

#[test]
fn tensor_coverage_preserves_replicas_and_detects_missing_regions() {
    let shard = |offset: Vec<u64>, shape: Vec<u64>| Locator {
        source: WorkerInfo::default().endpoint,
        transport: TransferTransport::PosixShm {
            endpoint: "tensor-publisher".into(),
            name: "tensor-shard".into(),
        },
        nbytes: shape.iter().product::<u64>() * 4,
        dtype: "float32".into(),
        device: "cpu".into(),
        offset,
        shape,
    };
    let mut tensor = TensorTransfer {
        shape: vec![4, 4],
        locations: vec![shard(vec![0, 0], vec![4, 2]), shard(vec![0, 2], vec![2, 2])],
    };
    assert!(!tensor.has_complete_coverage());
    tensor.locations.push(shard(vec![2, 2], vec![2, 2]));
    assert!(tensor.has_complete_coverage());
    tensor.locations.push(shard(vec![0, 0], vec![4, 4]));
    tensor.locations.remove(0);
    assert!(tensor.has_complete_coverage());
    tensor.locations.pop();
    assert!(!tensor.has_complete_coverage());
}

#[test]
fn raw_kv_publication_round_trips_page_representation_and_exact_lineage() {
    let mut product = output_product(OpId(11));
    product.kind = ProductKind::Kv;
    product.storage_class = StorageClass::PagedKv;
    product.dtype = DType::U8;
    product.shape_bound = ShapeBound {
        dims: vec![DimBound::Static(4096)],
    };
    let tensor = |name: &str, dtype: &str, shape: Vec<u64>, itemsize: u64| TensorTransfer {
        shape: shape.clone(),
        locations: vec![Locator {
            source: WorkerInfo::default().endpoint,
            transport: TransferTransport::PosixShm {
                endpoint: "uniserve-kv-publisher".into(),
                name: name.into(),
            },
            nbytes: shape.iter().product::<u64>() * itemsize,
            dtype: dtype.into(),
            offset: vec![0; shape.len()],
            shape,
            device: "cpu".into(),
        }],
    };
    for (dtype, itemsize) in [("bfloat16", 2), ("float8_e4m3fn", 1)] {
        let mut tensors = vec![
            tensor("keys", dtype, vec![5, 2, 3, 4], itemsize),
            tensor("values", dtype, vec![5, 2, 3, 4], itemsize),
        ];
        for field in &mut tensors {
            field.shape[2] = 6;
            field.locations[0].offset[2] = 3;
        }
        if dtype == "float8_e4m3fn" {
            let mut scales = tensor("scales", "float32", vec![2, 2, 2, 1], 4);
            scales.shape[3] = 2;
            scales.locations[0].offset[3] = 1;
            tensors.push(scales);
        }
        let report = lane_report(
            5,
            vec![completion_record()],
            vec![ProductPayload {
                product: product.clone(),
                value: InlineValue::Transfer(TransferHandle::Kv {
                    generation: 3,
                    tensors,
                    source: fixed_parent(),
                    destination: "decoder".into(),
                    base: Some(fixed_parent()),
                    base_extent: 3,
                    published_extent: 8,
                    group_id: 0,
                    compute_dtype: "bfloat16".into(),
                    page_size: 4,
                }),
            }],
            true,
            None,
            None,
        );
        let decoded =
            decode_response(&encode_response(&WorkerResponse::result(report.clone())).unwrap())
                .unwrap();
        assert_eq!(decoded.report().unwrap(), &report);

        for invalid_field in ["source", "base", "page_size", "scales"] {
            let mut invalid = report.clone();
            let InlineValue::Transfer(TransferHandle::Kv {
                source,
                base,
                page_size,
                tensors,
                ..
            }) = &mut invalid.products[0].value
            else {
                unreachable!()
            };
            match invalid_field {
                "source" => source.point = CheckpointPoint::DeviceSelected,
                "base" => base.as_mut().unwrap().point = CheckpointPoint::DeviceSelected,
                "page_size" => *page_size = 0,
                "scales" if dtype == "float8_e4m3fn" => {
                    tensors[2] = tensor("scales", "float32", vec![1, 2, 2], 4);
                }
                "scales" => tensors.push(tensor("scales", "float32", vec![2, 2, 2], 4)),
                _ => unreachable!(),
            }
            assert!(encode_response(&WorkerResponse::result(invalid)).is_err());
        }
    }
}

#[test]
fn completion_product_value_respects_its_registered_bound() {
    let mut logprob = output_product(OpId(11));
    logprob.output_index = 2;
    logprob.kind = ProductKind::Logprob;
    logprob.storage_class = StorageClass::HostStaging;
    logprob.dtype = DType::U8;
    logprob.shape_bound = ShapeBound {
        dims: vec![DimBound::Static(3)],
    };
    let report = lane_report(
        5,
        vec![completion_record()],
        vec![ProductPayload {
            product: logprob,
            value: InlineValue::Bytes(vec![1, 2, 3, 4]),
        }],
        true,
        Some(10),
        None,
    );

    assert!(report.validate().is_err());
}

#[test]
fn error_completion_round_trips_with_its_error_code() {
    let mut record = completion_record();
    record.status = OpStatus::Error;
    record.error_code = Some(ErrorCode::ComputeError);
    let report = lane_report(6, vec![record.clone()], Vec::new(), false, None, None);
    let decoded =
        decode_response(&encode_response(&WorkerResponse::result(report)).unwrap()).unwrap();
    assert_eq!(
        decoded.report().unwrap().completions().next().unwrap(),
        &record
    );
}

#[test]
fn request_retirement_requires_unique_buffers_from_its_lineage() {
    let product = product_for(request_key(), OpId(7), 0, ProductKind::VisionFeature);
    for retained_buffers in [
        vec![product.buffer_id(), product.buffer_id()],
        vec![BufferId {
            owner: key_for_request(900),
            ..product.buffer_id()
        }],
    ] {
        let command = BatchCommand::Finish {
            request_key: request_key(),
            control_seq: 1,
            cutoff: fixed_parent(),
            reason: CloseReason::Completed,
            retained_buffers: retained_buffers.clone(),
        };
        for command in [
            command,
            BatchCommand::Retire {
                request_key: request_key(),
                retained_buffers,
            },
        ] {
            assert!(
                encode_request(&WorkerRequest::submit(
                    batch_with_operations(3, vec![admission()], vec![ar_decode_operation()])
                        .with_commands(vec![command]),
                ))
                .is_err()
            );
        }
    }
}

#[test]
fn every_batch_command_variant_round_trips_through_ipc() {
    let commit = BatchCommand::Commit {
        request_key: request_key(),
        control_seq: 1,
        expected_parent: fixed_parent(),
        selected: fixed_parent(),
        public_event_limit: 7,
        disposition: Disposition::Publish,
    };
    let close = BatchCommand::Finish {
        request_key: request_key(),
        control_seq: 2,
        cutoff: fixed_parent(),
        reason: CloseReason::Completed,
        retained_buffers: vec![
            product_for(request_key(), OpId(7), 0, ProductKind::VisionFeature).buffer_id(),
        ],
    };
    let retire = BatchCommand::Retire {
        request_key: request_key(),
        retained_buffers: vec![
            product_for(request_key(), OpId(7), 0, ProductKind::VisionFeature).buffer_id(),
        ],
    };
    let batch = batch_with_operations(3, vec![admission()], vec![ar_decode_operation()])
        .with_commands(vec![commit.clone(), close.clone(), retire.clone()]);
    let decoded = execute_round_trip(batch);
    assert_eq!(
        decoded.commands,
        vec![
            BatchCommand::Start {
                request: admission(),
            },
            commit,
            close,
            retire,
        ]
    );
}

#[test]
fn new_request_round_trips() {
    let batch = execute_round_trip(batch_with_operations(
        4,
        vec![admission()],
        vec![ar_decode_operation()],
    ));
    assert_eq!(batch.admissions().next(), Some(&admission()));
}

#[test]
fn maximum_media_prompt_round_trips() {
    let prompt_token_ids = (0..16_384_u32).map(|index| 100_000 + index).collect();
    let admission = media_admission(prompt_token_ids);
    let request = WorkerRequest::submit(batch_with_operations(
        5,
        vec![admission.clone()],
        vec![operation_for(OpCode::DiffusionPrepare, OpId(12), true)],
    ));

    let decoded = decode_request(&encode_request(&request).unwrap()).unwrap();
    assert_eq!(decoded.run().unwrap().admissions().next(), Some(&admission));
}

#[test]
fn kv_publication_requires_a_fixed_semantic_parent() {
    let mut operation = operation_for(OpCode::TransferKvPublish, OpId(12), false);
    operation.parent = Some(Checkpoint {
        op_id: OpId(9),
        point: CheckpointPoint::DeviceSelected,
    });

    assert!(operation.validate().is_err());
}

#[test]
fn validation_rejects_an_output_owned_by_another_operation() {
    let mut operation = ar_decode_operation();
    operation.outputs_mut()[0].producer_op_id = OpId(999);
    assert!(operation.validate().is_err());
}

#[test]
fn validation_allows_shared_encoder_features_and_rejects_foreign_lineage_state() {
    let foreign_key = RequestKey::new(4, RequestId(8), 2);
    let mut feature = output_product(OpId(3));
    feature.request_key = foreign_key;
    feature.kind = ProductKind::VisionFeature;
    let mut operation = ar_decode_operation();
    operation.inputs_mut().push(feature);
    operation.validate().unwrap();

    operation.inputs_mut()[0].kind = ProductKind::Token;
    assert!(operation.validate().is_err());
}

#[test]
fn product_validation_enforces_generation_and_shape_bounds() {
    let mut product = output_product(OpId(11));
    product.generation = 0;
    assert!(product.validate().is_err());
    product.generation = 1;
    product.shape_bound.dims = vec![DimBound::Device { max: 2 }, DimBound::Device { max: 4 }];
    assert!(product.validate().is_err());
}

#[test]
fn batch_rejects_two_operations_for_one_request() {
    let batch = Run {
        batch_id: 1,
        run_id: 1,
        collective_seq: 1,
        operations: vec![ar_decode_operation(), ar_decode_operation()],
        block_tables: Vec::new(),
        new_cache_pages: Vec::new(),
        forward_rows: Vec::new(),
        latent_params: Vec::new(),
        decode_ranges: Vec::new(),
        buffer_allocations: Vec::new(),
        commands: Vec::new(),
        input_products: Vec::new(),
    };
    assert!(batch.validate().is_err());
}

#[test]
fn token_product_bytes_round_trip() {
    for tokens in [vec![], vec![42], vec![1, 2, 3, 4, 5], vec![u32::MAX, 0, 7]] {
        let bytes = encode_token_product_bytes(&tokens);
        assert_eq!(decode_token_product_bytes(&bytes).unwrap(), tokens);
    }
    // The count prefix must occupy a complete little-endian word.
    assert!(decode_token_product_bytes(&[0, 0]).is_err());
    // A length that disagrees with the declared count is a fault.
    assert!(decode_token_product_bytes(&[2, 0, 0, 0, 9, 0, 0, 0]).is_err());
}

#[test]
fn sampling_state_bytes_preserve_empty_allowed_and_canonical_sets() {
    let state = SamplingState {
        allowed_token_ids: Some(Vec::new()),
        suppressed_token_ids: vec![7, 2, 7],
        finish_token_ids: vec![11, 5, 11],
        transition_token_ids: vec![29, 13, 29],
        force_finish: true,
    };
    let decoded =
        decode_sampling_state_bytes(&encode_sampling_state_bytes(&state)).expect("decode state");
    assert_eq!(
        decoded,
        SamplingState {
            allowed_token_ids: Some(Vec::new()),
            suppressed_token_ids: vec![2, 7],
            finish_token_ids: vec![5, 11],
            transition_token_ids: vec![13, 29],
            force_finish: true,
        }
    );
}

#[test]
fn batch_carries_host_supplied_input_product_values() {
    let mut token_input = output_product(OpId(11));
    token_input.output_index = 0;
    token_input.kind = ProductKind::Token;
    token_input.storage_class = StorageClass::HostStaging;
    token_input.shape_bound = ShapeBound {
        dims: vec![DimBound::Static(3)],
    };
    let payload = ProductPayload {
        product: token_input.clone(),
        value: InlineValue::Bytes(encode_token_product_bytes(&[7, 8, 9])),
    };
    let mut operation = ar_decode_operation();
    operation.inputs_mut().push(token_input);
    let batch = batch_with_operations(1, vec![admission()], vec![operation])
        .with_input_products(vec![payload.clone()]);
    let decoded = execute_round_trip(batch);
    assert_eq!(decoded.input_products, vec![payload.clone()]);
    assert_eq!(
        decode_token_product_bytes(decoded.input_products[0].value.bytes().unwrap()).unwrap(),
        vec![7, 8, 9]
    );
}

#[test]
fn batch_rejects_a_conflicting_command_identity() {
    let commit = |limit| BatchCommand::Commit {
        request_key: request_key(),
        control_seq: 1,
        expected_parent: fixed_parent(),
        selected: fixed_parent(),
        public_event_limit: limit,
        disposition: Disposition::Publish,
    };
    let batch = batch_with_operations(1, vec![admission()], vec![ar_decode_operation()])
        .with_commands(vec![commit(1), commit(2)]);
    assert!(batch.validate().is_err());
}

#[test]
fn batch_allows_a_duplicate_identical_command() {
    let commit = BatchCommand::Commit {
        request_key: request_key(),
        control_seq: 1,
        expected_parent: fixed_parent(),
        selected: fixed_parent(),
        public_event_limit: 1,
        disposition: Disposition::Publish,
    };
    let batch = batch_with_operations(1, vec![admission()], vec![ar_decode_operation()])
        .with_commands(vec![commit.clone(), commit]);
    assert!(batch.validate().is_ok());
}

#[test]
fn commit_command_requires_a_fixed_selected_version() {
    let device_selected = Checkpoint {
        op_id: OpId(9),
        point: CheckpointPoint::DeviceSelected,
    };
    let command = BatchCommand::Commit {
        request_key: request_key(),
        control_seq: 1,
        expected_parent: fixed_parent(),
        selected: device_selected,
        public_event_limit: 1,
        disposition: Disposition::Publish,
    };
    assert!(command.validate().is_err());
}

#[test]
fn worker_info_round_trips() {
    use uniserve_core::{EntryConfig, ParallelConfig, SequenceParallel};
    let strategies = [
        SequenceParallel::Local,
        SequenceParallel::Ulysses { ulysses_degree: 2 },
        SequenceParallel::Ring { ring_degree: 2 },
        SequenceParallel::Hybrid {
            ulysses_degree: 2,
            ring_degree: 2,
        },
        SequenceParallel::Allgather {
            allgather_degree: 2,
        },
        SequenceParallel::Attention2d {
            attn2d_row_size: 2,
            attn2d_col_size: 2,
            ulysses_degree: 2,
        },
    ];
    for sequence_parallel in strategies {
        let config = ParallelConfig {
            sequence_parallel,
            ..Default::default()
        };
        let count = config.world_size().unwrap();
        let info = WorkerInfo {
            world_size: count as u32,
            configuration_id: "a".repeat(64),
            kv_cache: Some(KvCacheConfig {
                num_layers: 9,
                total_layers: 28,
                layer_offset: 11,
                ..WorkerInfo::default().kv_cache.unwrap()
            }),
            components: vec![EntryInfo {
                name: "denoiser".into(),
                config: EntryConfig::parallel((0..count).rev().collect(), config),
                outputs: vec![TensorSpec {
                    name: "conditioning".into(),
                    dtype: DType::BF16,
                    shape_bound: ShapeBound {
                        dims: vec![
                            DimBound::Static(1),
                            DimBound::Device { max: 16384 },
                            DimBound::Static(2560),
                        ],
                    },
                }],
            }],
            ..Default::default()
        };
        // Python startup metadata crosses the serde mapping boundary before
        // the binary response is delivered to the native host.
        let mapped = serde_json::to_value(&info).unwrap();
        assert_eq!(serde_json::from_value::<WorkerInfo>(mapped).unwrap(), info);
        let response = WorkerResponse::info(info.clone());
        let decoded = decode_response(&encode_response(&response).unwrap()).unwrap();
        let WorkerResponse::Info { info: decoded, .. } = decoded else {
            panic!("decoded response must preserve its info variant");
        };
        assert_eq!(decoded, info);
    }
}

#[test]
fn kv_free_worker_info_round_trips() {
    let info = WorkerInfo {
        media_plan: Some(terminal_media_plan()),
        supported_ops: vec![
            OpCode::EncoderText,
            OpCode::DiffusionPrepare,
            OpCode::DiffusionStep,
            OpCode::DiffusionDecode,
            OpCode::DiffusionFinalize,
            OpCode::MediaAppend,
        ],
        kv_cache: None,
        latent_page_units: 64,
        latent_pages: 3,
        ..WorkerInfo::default()
    };
    let response = WorkerResponse::info(info.clone());
    let decoded = decode_response(&encode_response(&response).unwrap()).unwrap();
    let WorkerResponse::Info { info: decoded, .. } = decoded else {
        panic!("decoded response must preserve its info variant");
    };
    assert_eq!(decoded, info);
    assert_eq!(decoded.denoise_steps(), 4);
}

#[test]
fn worker_info_rejects_duplicate_set_members() {
    let info = WorkerInfo {
        supported_ops: vec![OpCode::ArExtend, OpCode::ArDecode, OpCode::ArExtend],
        ..Default::default()
    };
    assert!(encode_response(&WorkerResponse::info(info)).is_err());
}

#[test]
fn worker_info_requires_group_totals_to_match_the_cache() {
    let mut info = full_caps();
    info.kv_cache.as_mut().unwrap().groups[1].num_blocks = 2047;
    assert!(encode_response(&WorkerResponse::info(info)).is_err());
}

fn key_for_request(request: u64) -> RequestKey {
    RequestKey::new(4, RequestId(request), 2)
}

fn product_for(key: RequestKey, op: OpId, output_index: u16, kind: ProductKind) -> ProductRef {
    ProductRef {
        request_key: key,
        producer_op_id: op,
        output_index,
        generation: 3 + u32::from(output_index),
        kind,
        storage_class: match kind {
            ProductKind::Tensor => StorageClass::DeviceTensor,
            ProductKind::Kv => StorageClass::PagedKv,
            ProductKind::Latent | ProductKind::LatentFeature | ProductKind::VisionFeature => {
                StorageClass::LatentArena
            }
            ProductKind::Token | ProductKind::Completion | ProductKind::SelectedPoint => {
                StorageClass::RequestRelay
            }
            ProductKind::Logprob | ProductKind::SamplingState | ProductKind::Artifact => {
                StorageClass::HostStaging
            }
        },
        dtype: match kind {
            ProductKind::Token | ProductKind::SelectedPoint => DType::U32,
            ProductKind::Kv => DType::BF16,
            ProductKind::Logprob => DType::F32,
            ProductKind::Completion | ProductKind::SamplingState | ProductKind::Artifact => {
                DType::U8
            }
            _ => DType::F16,
        },
        shape_bound: ShapeBound {
            dims: match kind {
                ProductKind::Token | ProductKind::Completion | ProductKind::SelectedPoint => {
                    Vec::new()
                }
                ProductKind::Logprob | ProductKind::SamplingState | ProductKind::Artifact => {
                    vec![DimBound::Device { max: 32 }]
                }
                _ => vec![DimBound::Static(2), DimBound::Device { max: 16 }],
            },
        },
        point_range: PointRange {
            base_point: 0,
            max_points: 1,
        },
    }
}

fn full_sampling() -> SamplingParams {
    SamplingParams {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        ignore_eos: true,
        seed: Some(0xdead_beef),
        min_p: 0.05,
        repetition_penalty: 1.1,
        frequency_penalty: 0.25,
        presence_penalty: -0.5,
        logit_bias: vec![(5, 1.5), (11, -2.0)],
        min_tokens: 3,
        n_logprobs: 4,
        return_logprobs: true,
        return_prompt_logprobs: true,
        n_prompt_logprobs: 2,
        logprob_token_ids: vec![7, 8, 9],
        bad_words_ids: vec![vec![1, 2, 3], vec![42]],
        allowed_token_ids: Some(vec![10, 11, 12]),
        typical_p: 0.8,
        forced_token_ids: vec![13, 14],
    }
}

fn full_image() -> ImageParams {
    ImageParams {
        steps: 20,
        cfg_text_scale: 5.0,
        cfg_img_scale: 1.5,
        cfg_renorm_type: CfgRenorm::Global,
        cfg_renorm_min: 0.1,
        cfg_interval: (0.2, 0.9),
        timestep_shift: 3.0,
        height: 1024,
        width: 768,
        seed: Some(77),
        negative_prompt: "blurry".into(),
        max_images: 2,
        image_prompts: vec!["p0".into(), "p1".into()],
        retain_images: false,
    }
}

/// One operation per closed `OpCode` variant, each on its own request key so the
/// batch admits them together; the first two keys also carry admissions.
fn comprehensive_batch() -> Run {
    let variants = OpCode::ALL;
    let mut operations = Vec::new();
    for (index, kind) in variants.into_iter().enumerate() {
        let key = key_for_request(100 + index as u64);
        let op_id = OpId(11 + index as u64);
        let parent = if kind.requires_fixed_parent() || index % 2 == 0 {
            Checkpoint::admission_root(OpId(1))
        } else {
            Checkpoint {
                op_id: OpId(9),
                point: CheckpointPoint::DeviceSelected,
            }
        };
        operations.push(
            Operation {
                request_key: key,
                op_id,
                parent: Some(parent),
                entry: "model".into(),
                payload: OpPayload::new(
                    kind,
                    Bounds {
                        max_points: if kind.advances_state() { 1 } else { 0 },
                        max_tokens: 7 + index as u32,
                        max_kv_pages: 3,
                        max_latent_bytes: 1 << 20,
                        max_completion_bytes: 4096,
                        max_transfer_bytes: 1 << 16,
                    },
                    vec![product_for(key, OpId(2), 0, ProductKind::VisionFeature)],
                    vec![
                        product_for(key, op_id, 0, ProductKind::Token),
                        product_for(key, op_id, 1, ProductKind::Kv),
                    ],
                    Some(product_for(key, OpId(3), 0, ProductKind::Completion)),
                    Some(Rng {
                        seed: 99 + index as u64,
                        semantic_index_base: 4,
                        draw_layout: match index % 3 {
                            0 => DrawLayout::TargetSampling,
                            1 => DrawLayout::SpeculativeProposal,
                            _ => DrawLayout::FlowNoise,
                        },
                    }),
                    index as u64,
                ),
            }
            .sealed(),
        );
    }
    let ar_params = NewRequest::new(
        key_for_request(100),
        100,
        Some(ArRequestParams {
            sampling: full_sampling(),
            negative_token_ids: vec![100, 101],
            finish_token_ids: vec![2, 7],
            initial_position: 128,
        }),
        None,
    )
    .unwrap();
    let umm_params = NewRequest::new(
        key_for_request(110),
        110,
        None,
        Some(UmmRequestParams {
            image: full_image(),
        }),
    )
    .unwrap();
    let commands = vec![
        BatchCommand::Commit {
            request_key: key_for_request(200),
            control_seq: 1,
            expected_parent: Checkpoint::admission_root(OpId(1)),
            selected: Checkpoint::admission_root(OpId(2)),
            public_event_limit: 7,
            disposition: Disposition::Retain,
        },
        BatchCommand::Finish {
            request_key: key_for_request(201),
            control_seq: 2,
            cutoff: Checkpoint::admission_root(OpId(1)),
            reason: CloseReason::Preempted,
            retained_buffers: Vec::new(),
        },
    ];
    let mut input_product = product_for(
        operations[0].request_key,
        operations[0].op_id,
        2,
        ProductKind::Token,
    );
    input_product.storage_class = StorageClass::HostStaging;
    input_product.shape_bound = ShapeBound {
        dims: vec![DimBound::Static(4)],
    };
    operations[0].inputs_mut().push(input_product.clone());
    let input_products = vec![ProductPayload {
        product: input_product,
        value: InlineValue::Bytes(encode_token_product_bytes(&[7, 8, 9, 10])),
    }];
    batch_with_operations(42, vec![ar_params, umm_params], operations)
        .with_commands(commands)
        .with_input_products(input_products)
}

/// One fixture per `RequestKind`, plus call-id coverage on the submit frame.
fn request_fixtures() -> Vec<WorkerRequest> {
    let mut submit = WorkerRequest::submit(comprehensive_batch());
    submit.set_call_id(Some(91));
    vec![
        WorkerRequest::info(),
        submit,
        WorkerRequest::poll(42),
        WorkerRequest::close(),
    ]
}

fn full_caps() -> WorkerInfo {
    WorkerInfo {
        media_plan: Some(terminal_media_plan()),
        supported_ops: OpCode::ALL.to_vec(),
        kv_cache: Some(KvCacheConfig {
            groups: vec![
                KvCacheGroup {
                    num_blocks: 2048,
                    kind: KvGroupKind::Full,
                },
                KvCacheGroup {
                    num_blocks: 2048,
                    kind: KvGroupKind::SlidingWindow {
                        window: 4096,
                        sink: 64,
                    },
                },
            ],
            ..WorkerInfo::default().kv_cache.unwrap()
        }),
        endpoint: WorkerEndpoint {
            rank: 1,
            ..WorkerInfo::default().endpoint
        },
        world_size: 2,
        queue_depth: 2,
        max_batch_ops: 64,
        max_batch_tokens: 4096,
        request_slots: 96,
        max_unresolved_ops: 3,
        latent_page_units: 64,
        latent_pages: 17,
        model_name: "test-model".into(),
        ..WorkerInfo::default()
    }
}

fn terminal_media_plan() -> MediaExecutionPlan {
    let stage = |name: &str,
                 operation: OpCode,
                 entry: &str,
                 dependencies: &[&str],
                 input_from: Option<&str>,
                 repeat: MediaPlanRepeat,
                 count: u32| MediaPlanStage {
        name: name.into(),
        operation,
        entry: entry.into(),
        dependencies: dependencies.iter().map(|value| (*value).into()).collect(),
        input_from: input_from.map(str::to_owned),
        repeat,
        count,
    };
    MediaExecutionPlan {
        stages: vec![
            stage(
                "encode",
                OpCode::EncoderText,
                "text_encoder",
                &[],
                None,
                MediaPlanRepeat::Once,
                1,
            ),
            stage(
                "prepare",
                OpCode::DiffusionPrepare,
                "denoiser",
                &["encode"],
                Some("encode"),
                MediaPlanRepeat::Once,
                1,
            ),
            stage(
                "denoise",
                OpCode::DiffusionStep,
                "denoiser",
                &["prepare"],
                None,
                MediaPlanRepeat::Fixed,
                4,
            ),
            stage(
                "video_decode",
                OpCode::DiffusionDecode,
                "video_decoder",
                &["denoise"],
                Some("denoise"),
                MediaPlanRepeat::VideoUnits,
                1,
            ),
            stage(
                "audio_decode",
                OpCode::DiffusionDecode,
                "audio_decoder",
                &["denoise"],
                Some("denoise"),
                MediaPlanRepeat::Once,
                1,
            ),
            stage(
                "video_append",
                OpCode::MediaAppend,
                "output",
                &["denoise"],
                Some("video_decode"),
                MediaPlanRepeat::VideoUnits,
                1,
            ),
            stage(
                "audio_append",
                OpCode::MediaAppend,
                "output",
                &["denoise"],
                Some("audio_decode"),
                MediaPlanRepeat::Once,
                1,
            ),
            stage(
                "finalize",
                OpCode::DiffusionFinalize,
                "output",
                &["video_append", "audio_append"],
                None,
                MediaPlanRepeat::Once,
                1,
            ),
        ],
    }
}

fn full_forward_stats() -> WorkerForwardStats {
    let map = |prefix: &str, base: u64| {
        BTreeMap::from([
            (format!("{prefix}.a"), base),
            (format!("{prefix}.b"), base + 1),
        ])
    };
    WorkerForwardStats {
        mode_counts: map("mode_counts", 1),
        mode_tokens: map("mode_tokens", 3),
        mode_us: map("mode_us", 5),
        component_us: map("component_us", 7),
        attention_launches: 11,
        attention_us: 12,
        attention_backend_counts: map("backend", 13),
        cuda_graph_captures: 15,
        cuda_graph_replays: 16,
        cuda_graph_misses: 17,
        cuda_graph_fallbacks: 18,
        cuda_graph_unpadded_tokens: 19,
        cuda_graph_padded_tokens: 20,
        cuda_graph_runtime_mode_counts: map("runtime_mode", 21),
        text_decode_token_relay_hits: 23,
        text_decode_token_relay_misses: 24,
        text_decode_position_relay_hits: 25,
        text_decode_position_relay_misses: 26,
        flashinfer_decode_plan_calls: 27,
        flashinfer_decode_plan_reuses: 28,
        flashinfer_decode_plan_rows: 29,
        flashinfer_decode_plan_indices: 30,
        flashinfer_decode_graph_plan_calls: 31,
        flashinfer_decode_graph_plan_reuses: 32,
        spec_verify_rows: 33,
        spec_verify_draft_tokens: 34,
        spec_verify_accepted_tokens: 35,
        spec_verify_rejected_tokens: 36,
        spec_verify_committed_tokens: 37,
        spec_verify_path_counts: map("spec_path", 38),
    }
}

fn full_run_result() -> RunResult {
    let ok_record = ModelOutput {
        request_key: key_for_request(100),
        op_id: OpId(11),
        completion_slot_generation: 2,
        status: OpStatus::Ok,
        selected_point: 1,
        product_generations: vec![3, 5],
        error_code: None,
        timing_counters: TimingCounters {
            queued_us: 41,
            device_us: 42,
            copy_us: 43,
            host_us: 44,
        },
        payload: ResultPayload::Ar(ResultData {
            logical_lengths: LogicalLengths {
                token_len: 5,
                kv_visible_len: 6,
                latent_len: 7,
                kv_computed_len: 8,
            },
            token_span: TokenSpan { base: 4, len: 2 },
            committed_tokens: vec![271, 272],
            finish_flags: FinishFlags {
                eos: true,
                length: false,
                stop: false,
            },
            media_output: None,
        }),
    };
    let mut predicated_record = ok_record.clone();
    predicated_record.request_key = key_for_request(101);
    predicated_record.op_id = OpId(12);
    predicated_record.status = OpStatus::Predicated;
    predicated_record.token_span_mut().len = 0;
    predicated_record.committed_tokens_mut().clear();
    *predicated_record.finish_flags_mut() = FinishFlags::default();
    predicated_record.product_generations = Vec::new();
    let mut error_record = ok_record.clone();
    error_record.request_key = key_for_request(102);
    error_record.op_id = OpId(13);
    error_record.status = OpStatus::Error;
    error_record.error_code = Some(ErrorCode::ResourceExhausted);
    *error_record.finish_flags_mut() = FinishFlags {
        eos: false,
        length: false,
        stop: true,
    };
    let mut artifact = product_for(key_for_request(102), OpId(13), 3, ProductKind::Artifact);
    artifact.storage_class = StorageClass::PinnedOutput;
    artifact.dtype = DType::U8;
    artifact.shape_bound = ShapeBound {
        dims: vec![DimBound::Static(256)],
    };
    let products = vec![
        ProductPayload {
            product: product_for(key_for_request(100), OpId(11), 2, ProductKind::Logprob),
            value: InlineValue::Bytes(vec![1, 2, 3, 4, 5]),
        },
        ProductPayload {
            product: artifact,
            value: InlineValue::Bytes((0..=255).collect()),
        },
    ];
    lane_report(
        5,
        vec![ok_record, predicated_record, error_record],
        products,
        true,
        Some(1234),
        Some(full_forward_stats()),
    )
}

/// One fixture per `ResponseKind`, plus a second info frame that
/// exercises non-default worker information.
fn response_fixtures() -> Vec<WorkerResponse> {
    vec![
        WorkerResponse::Info {
            call_id: None,
            info: WorkerInfo::default(),
        },
        WorkerResponse::Info {
            call_id: Some(17),
            info: full_caps(),
        },
        WorkerResponse::Result {
            call_id: Some(17),
            result: full_run_result(),
        },
        WorkerResponse::Ok { call_id: Some(17) },
        WorkerResponse::Error {
            call_id: Some(17),
            error: WorkerResponseError {
                message: "device fault on decode".into(),
                code: Some("compute_error".into()),
                retryable: true,
                fatal: false,
                phase: Some("execute".into()),
                route: Some("und.decode".into()),
                operations: vec![
                    ErrorOperationIdentity {
                        request_key: key_for_request(100),
                        op_id: OpId(11),
                    },
                    ErrorOperationIdentity {
                        request_key: key_for_request(101),
                        op_id: OpId(12),
                    },
                ],
            },
        },
    ]
}

#[test]
fn every_request_kind_round_trips_through_ipc() {
    let fixtures = request_fixtures();
    for kind in RequestKind::ALL {
        assert!(
            fixtures.iter().any(|request| request.kind() == kind),
            "no fixture covers request kind {kind:?}"
        );
    }
    for request in fixtures {
        let bytes = encode_request(&request).unwrap();
        assert_eq!(
            decode_request(&bytes).unwrap(),
            request,
            "round trip for {:?}",
            request.kind()
        );
    }
}

#[test]
fn every_response_kind_round_trips_through_ipc() {
    let fixtures = response_fixtures();
    for kind in [
        ResponseKind::Info,
        ResponseKind::Result,
        ResponseKind::Ok,
        ResponseKind::Error,
    ] {
        assert!(
            fixtures.iter().any(|response| response.kind() == kind),
            "no fixture covers response kind {kind:?}"
        );
    }
    for response in fixtures {
        let bytes = encode_response(&response).unwrap();
        assert_eq!(
            decode_response(&bytes).unwrap(),
            response,
            "round trip for {:?}",
            response.kind()
        );
    }
}

#[test]
fn ipc_decode_rejects_malformed_frames() {
    let garbage: &[u8] = &[0x01, 0x02, 0x03];
    assert!(decode_request(garbage).is_err());
    assert!(decode_response(garbage).is_err());

    let mut request = encode_request(&WorkerRequest::info()).unwrap();
    request.truncate(request.len() / 2);
    assert!(decode_request(&request).is_err());

    let mut response = encode_response(&WorkerResponse::ok()).unwrap();
    response.truncate(response.len() / 2);
    assert!(decode_response(&response).is_err());
}
