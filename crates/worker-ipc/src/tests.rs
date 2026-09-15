//! Protocol round trips and validation behavior for every message family.

use std::collections::BTreeMap;

use uniserve_core::{BlockId, CfgRenorm, KvGroupKind, RequestId};

use super::*;
use crate::codec::{decode_request, decode_response, encode_request, encode_response};

fn request_key() -> RequestKey {
    RequestKey::new(4, RequestId(7), 2)
}

fn admission_predecessor() -> ComputationId {
    ComputationId::new(0, 0)
}

fn output_product(op: ComputationId) -> TensorRef {
    TensorRef {
        request_key: request_key(),
        producer_op_id: op,
        output_index: 0,
        generation: 3,
        dtype: DType::I64,
        shape_bound: ShapeBound::default(),
    }
}

fn ar_decode_operation() -> ScheduledRequest {
    let kind = Computation::Forward(ForwardMode::Decode);
    ScheduledRequest {
        token_input: None,

        token_output: Some(output_product(ComputationId::new(11, 0))),
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
        request_key: request_key(),
        op_id: ComputationId::new(11, 0),
        predecessor: Some(admission_predecessor()),
        entry: "model".into(),
        code: kind,
        bounds: Bounds {
            max_tokens: 1,
            max_kv_pages: 1,
            ..Bounds::default()
        },
        inputs: Vec::new(),
        outputs: Vec::new(),
        predicate: None,
        rng: Some(Rng {
            seed: 99,
            semantic_index_base: 4,
            draw_layout: DrawLayout::TargetSampling,
        }),
    }
}

fn operation_for(kind: Computation, op_id: ComputationId) -> ScheduledRequest {
    ScheduledRequest {
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
        kv_input: (kind == Computation::Transfer(TransferMode::KvInstall)).then_some(BufferId {
            owner: request_key(),
            producer_op_id: ComputationId::new(1, 0),
            output_index: 0,
            generation: 1,
        }),
        kv_output: matches!(
            kind,
            Computation::Transfer(TransferMode::KvPublish | TransferMode::KvInstall)
        )
        .then_some(BufferId {
            owner: request_key(),
            producer_op_id: op_id,
            output_index: 0,
            generation: 2,
        }),
        input_token_ids: Vec::new(),
        sampling_state: None,
        request_key: request_key(),
        op_id,
        predecessor: Some(admission_predecessor()),
        entry: "model".into(),
        code: kind,
        bounds: Bounds::default(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        predicate: None,
        rng: None,
    }
}

fn completion_record() -> RequestOutput {
    RequestOutput {
        sampled_logprob: Some(-0.25),
        top_logprobs: vec![
            TokenLogprob {
                token_id: 271,
                logprob: -0.25,
                rank: 1,
            },
            TokenLogprob {
                token_id: 42,
                logprob: f32::NEG_INFINITY,
                rank: 1000,
            },
        ],
        prompt_logprobs: vec![vec![TokenLogprob {
            token_id: 7,
            logprob: -0.5,
            rank: 2,
        }]],
        request_key: request_key(),
        op_id: ComputationId::new(11, 0),

        status: OpStatus::Ok,

        product_generations: vec![3, 4],
        error_code: None,
        timing_counters: TimingCounters::default(),
        code: Computation::Forward(ForwardMode::Decode),
        position: 5,
        kv_visible_len: 5,
        kv_computed_len: 5,
        num_completed_steps: 0,

        committed_tokens: vec![271],
        finish_flags: FinishFlags::default(),
        media_output: None,
        kv_output: None,
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
        prompt_token_ids,
        DiffusionSamplingParams {
            num_frames: 22,
            num_decode_chunks: 3,
            num_inference_steps: 4,
            seed: 17,
        },
    )
    .unwrap()
}

fn execute_round_trip(batch: ScheduleBatch) -> ScheduleBatch {
    let request = WorkerRequest::submit(batch);
    let decoded = decode_request(&encode_request(&request).unwrap()).unwrap();
    decoded.run().unwrap().clone()
}

#[test]
fn forward_columns_preserve_cfg_rows_and_reject_misalignment() {
    let operation = operation_for(
        Computation::Pipeline(PipelineStage::Denoising),
        ComputationId::new(11, 0),
    );
    let mut run = batch_with_operations(3, vec![], vec![operation]);
    // One alternative-prefix write followed by two denoising rows. Prefix
    // plus query lengths form total attention lengths even for read-only rows.
    run.forward = ForwardBatch {
        operation_indices: vec![0, 0, 0],
        request_pool_indices: vec![2, 1, 2],
        seq_lens: vec![2, 17, 6],
        query_lens: vec![2, 4, 4],
        write_kv: vec![true, false, false],
    };
    assert_eq!(execute_round_trip(run.clone()), run);

    let mut missing_length = run.clone();
    missing_length.forward.seq_lens.pop();
    assert!(encode_request(&WorkerRequest::submit(missing_length)).is_err());

    let mut short_sequence = run.clone();
    short_sequence.forward.seq_lens[2] = 3;
    assert!(encode_request(&WorkerRequest::submit(short_sequence)).is_err());

    run.forward.operation_indices[2] = 1;
    assert!(encode_request(&WorkerRequest::submit(run)).is_err());
}

#[test]
fn computation_coordinates_survive_physical_dispatch_and_reject_collisions() {
    let batch_id = u64::MAX - 7;
    let first = operation_for(
        Computation::Pipeline(PipelineStage::TextEncoding),
        ComputationId::new(batch_id, u32::MAX - 1),
    );
    let mut second = operation_for(
        Computation::Pipeline(PipelineStage::TextEncoding),
        ComputationId::new(batch_id, u32::MAX),
    );
    second.request_key = key_for_request(101);
    let run = batch_with_operations(3, vec![], vec![first.clone(), second]);
    assert_eq!(execute_round_trip(run.clone()), run);

    // A physical fragment can contain sparse logical indices. Dispatch neither
    // renumbers them nor narrows their unsigned coordinate range.
    let fragment = batch_with_operations(4, vec![], vec![first.clone()]);
    assert_eq!(
        execute_round_trip(fragment).operations[0].op_id,
        first.op_id
    );

    let mut collision = run;
    collision.operations[1].op_id = first.op_id;
    assert!(encode_request(&WorkerRequest::submit(collision)).is_err());

    let mut invalid_parent = first;
    invalid_parent.predecessor = Some(ComputationId::new(0, 1));
    assert!(invalid_parent.validate().is_err());
}

fn batch_with_operations(
    run_id: u64,
    admissions: Vec<NewRequest>,
    operations: Vec<ScheduledRequest>,
) -> ScheduleBatch {
    // Logical producer coordinates stay fixed when the physical run is numbered.
    let batch_id = operations
        .first()
        .map_or(run_id, |operation| operation.op_id.batch_id);
    let mut run = ScheduleBatch::new(batch_id, admissions, operations);
    run.run_id = run_id;
    run.collective_seq = run_id.max(1);
    let mut next_buffer_offset = 0_u64;
    for (operation_index, operation) in run.operations.iter().enumerate() {
        for output in operation.buffer_outputs() {
            next_buffer_offset = next_buffer_offset.div_ceil(256) * 256;
            let bytes = output.max_bytes();
            run.buffer_allocations.push(BufferAllocation {
                buffer: output.buffer_id(),
                offset: next_buffer_offset,
                bytes,
            });
            next_buffer_offset += bytes;
        }
        let capacity_pages = operation.bounds.max_kv_pages;
        if capacity_pages > 0 {
            let request_pool_idx = u32::try_from(operation.request_key.request_id.0).unwrap();
            let page_ids = (1..=capacity_pages).map(BlockId).collect::<Vec<_>>();
            run.block_tables.push(BlockTable {
                request_pool_idx,
                group_id: 0,
                page_ids: page_ids.clone(),
                allocated_tokens: operation.bounds.max_tokens.max(1),
            });
            run.new_cache_pages.push(CachePageAllocation {
                request_pool_idx,
                group_id: 0,
                page_ids,
            });
            run.forward.push(
                operation_index as u32,
                request_pool_idx,
                operation.bounds.max_tokens.max(1),
                operation.bounds.max_tokens.max(1),
                true,
            );
        }
        if matches!(
            operation.code,
            Computation::Pipeline(PipelineStage::LatentPreparation)
                | Computation::Pipeline(PipelineStage::Denoising)
        ) || operation.latent_input.is_some()
        {
            run.latent_params.push(LatentParams {
                request_key: operation.request_key,
                op_id: operation.op_id,
                page_table: vec![u32::try_from(operation_index + 1).unwrap()],
                latent_units: 1,
                height: 1,
                width: 1,
                start_step: 0,
                step_count: u32::from(
                    operation.code == Computation::Pipeline(PipelineStage::Denoising),
                ),
            });
        }
        if matches!(
            operation.code,
            Computation::Pipeline(
                PipelineStage::VideoDecoding
                    | PipelineStage::VideoEncoding
                    | PipelineStage::AudioDecoding
                    | PipelineStage::AudioEncoding
            )
        ) {
            run.decode_ranges.push(DecodeRange {
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
    completions: Vec<RequestOutput>,
    products: Vec<TensorPublication>,
    visible: bool,
    worker_exec_us: Option<u64>,
    forward_stats: Option<ForwardStats>,
) -> BatchOutput {
    BatchOutput {
        batch_id: completions
            .first()
            .map_or(run_id, |record| record.op_id.batch_id),
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
    let variants = Computation::ALL;
    for (index, work) in variants.into_iter().enumerate() {
        let operation = operation_for(work, ComputationId::new(100 + index as u64, 0));
        let batch = execute_round_trip(batch_with_operations(
            1,
            Vec::new(),
            vec![operation.clone()],
        ));
        assert_eq!(batch.operations().next().unwrap(), &operation);
    }
}

#[test]
fn decoder_requires_one_concrete_computation() {
    use crate::schema::uniserve::ipc as fbs;

    let run = batch_with_operations(1, Vec::new(), vec![ar_decode_operation()]);
    let bytes = encode_request(&WorkerRequest::submit(run)).unwrap();
    let frame = fbs::root_as_worker_request(&bytes).unwrap().unpack();
    let invalid = [
        fbs::ComputationT::default(),
        fbs::ComputationT {
            forward_mode: fbs::ForwardMode::Decode,
            stage: fbs::PipelineStage::Denoising,
            transfer: fbs::TransferMode::None,
        },
        fbs::ComputationT {
            forward_mode: fbs::ForwardMode(255),
            ..Default::default()
        },
        fbs::ComputationT {
            stage: fbs::PipelineStage(255),
            ..Default::default()
        },
    ];
    for code in invalid {
        let mut malformed = frame.clone();
        malformed.run.as_mut().unwrap().operations.as_mut().unwrap()[0].code = code;
        let mut builder = flatbuffers::FlatBufferBuilder::new();
        let root = malformed.pack(&mut builder);
        builder.finish(root, None);
        assert!(decode_request(builder.finished_data()).is_err());
    }
}

#[test]
fn solver_parameters_round_trip_with_request_or_paged_storage() {
    let operation = operation_for(
        Computation::Pipeline(PipelineStage::Denoising),
        ComputationId::new(101, 0),
    );
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
    let mut latent = output_product(ComputationId::new(55, 0));
    latent.dtype = DType::F32;
    latent.shape_bound = ShapeBound {
        dims: vec![DimBound::Static(8), DimBound::Static(32)],
    };
    for (index, (entry, stage)) in [
        ("video_decoder", PipelineStage::VideoDecoding),
        ("audio_decoder", PipelineStage::AudioDecoding),
    ]
    .iter()
    .enumerate()
    {
        let operation = ScheduledRequest {
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
            request_key: request_key(),
            op_id: ComputationId::new(56 + index as u64, 0),
            predecessor: None,
            entry: (*entry).into(),
            code: Computation::Pipeline(*stage),
            bounds: Bounds::default(),
            inputs: vec![latent.clone()],
            outputs: Vec::new(),
            predicate: None,
            rng: None,
        };
        let mut run = batch_with_operations(index as u64 + 1, Vec::new(), vec![operation]);
        if index == 0 {
            run.decode_ranges[0].cursor = 3;
            run.decode_ranges[0].max_units = 2;
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
fn block_table_allocation_round_trips_with_the_operation() {
    let base = ar_decode_operation();
    let batch = execute_round_trip(batch_with_operations(9, Vec::new(), vec![base.clone()]));
    assert_eq!(batch.operations().next().unwrap(), &base);
    assert_eq!(batch.block_tables[0].page_ids, vec![BlockId(1)]);
}

#[test]
fn publication_round_trips_its_registered_view_and_endpoint() {
    let mut product = output_product(ComputationId::new(11, 0));
    product.dtype = DType::F32;
    product.shape_bound = ShapeBound {
        dims: vec![DimBound::Static(4)],
    };
    for transport in [
        TransferTransport::CudaIpc {
            endpoint: "uniserve-cuda-physical-incarnation".into(),
            publication_id: "0123456789abcdef0123456789abcdef".into(),
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
            vec![TensorPublication {
                product: product.clone(),
                value: TransferHandle::DeviceProduct {
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
                },
            }],
            true,
            None,
            None,
        );
        let original = &mut report.products[0];
        let mut replica = original.clone();
        if let TransferHandle::DeviceProduct { tensor, .. } = &mut replica.value {
            tensor.locations[0].source.rank = 1;
        }
        let mut stale = replica.clone();
        stale.product.generation += 1;
        let before = original.clone();
        assert!(original.merge_locations(&stale).is_err());
        assert_eq!(*original, before);
        original.merge_locations(&replica).unwrap();
        let decoded =
            decode_response(&encode_response(&WorkerResponse::result(report.clone())).unwrap())
                .unwrap();
        assert_eq!(decoded.report().unwrap(), &report);
        for (dtype, storage_dtype, nbytes) in [(DType::I64, "int64", 32), (DType::I16, "int16", 8)]
        {
            let mut typed_report = report.clone();
            typed_report.products[0].product.dtype = dtype;
            let TransferHandle::DeviceProduct { tensor, .. } = &mut typed_report.products[0].value
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

#[test]
fn tensor_publication_preserves_static_axes_within_dynamic_capacity() {
    let mut reference = output_product(ComputationId::new(11, 0));
    reference.dtype = DType::F32;
    reference.shape_bound.dims = vec![DimBound::Static(2), DimBound::Device { max: 4 }];
    for (shape, valid) in [
        (vec![2, 3], true),
        (vec![2, 4], true),
        (vec![1, 4], false),
        (vec![2, 5], false),
        (vec![8], false),
    ] {
        let publication = TensorPublication {
            product: reference.clone(),
            value: TransferHandle::DeviceProduct {
                height: 0,
                width: 0,
                value_range: String::new(),
                tensor: TensorTransfer {
                    shape: shape.clone(),
                    locations: vec![Locator {
                        source: WorkerInfo::default().endpoint,
                        transport: TransferTransport::PosixShm {
                            endpoint: "tensor-publisher".into(),
                            name: "bounded-tensor".into(),
                        },
                        nbytes: shape.iter().product::<u64>() * 4,
                        dtype: "float32".into(),
                        offset: vec![0; shape.len()],
                        shape,
                        device: "cpu".into(),
                    }],
                },
            },
        };
        assert_eq!(publication.validate().is_ok(), valid);
        let mut flat = publication;
        flat.product.shape_bound.dims = vec![DimBound::Device { max: 8 }];
        assert_eq!(
            flat.validate().is_ok(),
            flat.value.tensors()[0].shape.iter().product::<u64>() <= 8
        );
    }
}

#[test]
fn unchanged_kv_publication_round_trips_without_physical_tensors() {
    let source = BufferId {
        owner: request_key(),
        producer_op_id: ComputationId::new(11, 0),
        output_index: 0,
        generation: 3,
    };
    let mut report = lane_report(
        5,
        vec![RequestOutput {
            code: Computation::Transfer(TransferMode::KvPublish),
            kv_output: Some(KvTransfer {
                tensors: Vec::new(),
                source: source,
                destination: "decoder".into(),
                base: Some(source),
                base_extent: 16,
                published_extent: 16,
                group_id: 0,
                compute_dtype: "bfloat16".into(),
                page_size: 16,
            }),
            ..completion_record()
        }],
        Vec::new(),
        true,
        None,
        None,
    );
    let decoded =
        decode_response(&encode_response(&WorkerResponse::result(report.clone())).unwrap())
            .unwrap();
    assert_eq!(decoded.report().unwrap(), &report);

    let KvTransfer {
        published_extent, ..
    } = report.completions[0].kv_output.as_mut().unwrap();
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
    let source = BufferId {
        owner: request_key(),
        producer_op_id: ComputationId::new(11, 0),
        output_index: 0,
        generation: 3,
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
            vec![RequestOutput {
                code: Computation::Transfer(TransferMode::KvPublish),
                kv_output: Some(KvTransfer {
                    tensors,
                    source: source,
                    destination: "decoder".into(),
                    base: Some(source),
                    base_extent: 3,
                    published_extent: 8,
                    group_id: 0,
                    compute_dtype: "bfloat16".into(),
                    page_size: 4,
                }),
                ..completion_record()
            }],
            Vec::new(),
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
            let KvTransfer {
                source,
                base,
                page_size,
                tensors,
                ..
            } = invalid.completions[0].kv_output.as_mut().unwrap();
            match invalid_field {
                "source" => source.generation = 0,
                "base" => base.as_mut().unwrap().generation = 0,
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
    let product = tensor_for(
        request_key(),
        ComputationId::new(7, 0),
        0,
        DType::F16,
        ShapeBound {
            dims: vec![DimBound::Static(2), DimBound::Device { max: 16 }],
        },
    );
    for retained_buffers in [
        vec![product.buffer_id(), product.buffer_id()],
        vec![BufferId {
            owner: key_for_request(900),
            ..product.buffer_id()
        }],
    ] {
        let command = BatchCommand::Finish {
            request_key: request_key(),
            retained_buffers,
        };
        assert!(
            encode_request(&WorkerRequest::submit(
                batch_with_operations(3, vec![admission()], vec![ar_decode_operation()])
                    .with_commands(vec![command]),
            ))
            .is_err()
        );
    }
}

#[test]
fn every_batch_command_variant_round_trips_through_ipc() {
    let finish = BatchCommand::Finish {
        request_key: request_key(),
        retained_buffers: vec![
            tensor_for(
                request_key(),
                ComputationId::new(7, 0),
                0,
                DType::F16,
                ShapeBound {
                    dims: vec![DimBound::Static(2), DimBound::Device { max: 16 }],
                },
            )
            .buffer_id(),
        ],
    };
    let free = BatchCommand::Free {
        buffer: tensor_for(
            request_key(),
            ComputationId::new(8, 0),
            0,
            DType::F16,
            ShapeBound {
                dims: vec![DimBound::Static(2), DimBound::Device { max: 16 }],
            },
        )
        .buffer_id(),
    };
    let batch = batch_with_operations(3, vec![admission()], vec![ar_decode_operation()])
        .with_commands(vec![finish.clone(), free.clone()]);
    let decoded = execute_round_trip(batch);
    assert_eq!(
        decoded.commands,
        vec![
            BatchCommand::Start {
                request: admission()
            },
            finish,
            free
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
        vec![operation_for(
            Computation::Pipeline(PipelineStage::LatentPreparation),
            ComputationId::new(12, 0),
        )],
    ));

    let decoded = decode_request(&encode_request(&request).unwrap()).unwrap();
    assert_eq!(decoded.run().unwrap().admissions().next(), Some(&admission));
}

#[test]
fn validation_rejects_an_output_owned_by_another_operation() {
    let mut operation = ar_decode_operation();
    operation.token_output.as_mut().unwrap().producer_op_id = ComputationId::new(999, 0);
    assert!(operation.validate().is_err());
}

#[test]
fn validation_allows_shared_encoder_features_and_rejects_foreign_lineage_state() {
    let foreign_key = RequestKey::new(4, RequestId(8), 2);
    let mut feature = output_product(ComputationId::new(3, 0));
    feature.request_key = foreign_key;
    let mut operation = ar_decode_operation();
    operation.vision_input = Some(feature);
    operation.validate().unwrap();

    operation
        .inputs
        .push(operation.vision_input.take().unwrap());
    assert!(operation.validate().is_err());
}

#[test]
fn product_validation_enforces_generation_and_shape_bounds() {
    let mut product = output_product(ComputationId::new(11, 0));
    product.generation = 0;
    assert!(product.validate().is_err());
    product.generation = 1;
    product.shape_bound.dims = vec![DimBound::Device { max: 2 }, DimBound::Device { max: 4 }];
    assert!(product.validate().is_err());
}

#[test]
fn batch_rejects_two_operations_for_one_request() {
    let batch = ScheduleBatch {
        batch_id: 1,
        run_id: 1,
        collective_seq: 1,
        operations: vec![ar_decode_operation(), ar_decode_operation()],
        block_tables: Vec::new(),
        new_cache_pages: Vec::new(),
        forward: ForwardBatch::default(),
        latent_params: Vec::new(),
        decode_ranges: Vec::new(),
        buffer_allocations: Vec::new(),
        commands: Vec::new(),
        input_products: Vec::new(),
        kv_inputs: Vec::new(),
    };
    assert!(batch.validate().is_err());
}

#[test]
fn token_inputs_preserve_order_and_enforce_capacity() {
    for tokens in [vec![], vec![42], vec![1, 2, 3, 4, 5], vec![u32::MAX, 0, 7]] {
        let mut operation = ar_decode_operation();
        operation.bounds.max_tokens = 5;
        operation.input_token_ids = tokens.clone();
        let decoded = execute_round_trip(batch_with_operations(1, Vec::new(), vec![operation]));
        assert_eq!(decoded.operations[0].input_token_ids, tokens);
    }
    let mut operation = ar_decode_operation();
    operation.bounds.max_tokens = 1;
    operation.input_token_ids = vec![7, 8];
    assert!(operation.validate().is_err());
}

#[test]
fn operation_sampling_state_preserves_whitelist_presence() {
    for allowed_token_ids in [None, Some(Vec::new()), Some(vec![2, 7])] {
        let mut operation = ar_decode_operation();
        operation.sampling_state = Some(SamplingState {
            allowed_token_ids,
            suppressed_token_ids: vec![3, 9],
            finish_token_ids: vec![5, 11],
            transition_token_ids: vec![13, 29],
            force_finish: true,
        });
        let request = WorkerRequest::submit(batch_with_operations(1, Vec::new(), vec![operation]));
        assert_eq!(
            decode_request(&encode_request(&request).unwrap()).unwrap(),
            request
        );
    }
}

#[test]
fn image_encoder_input_round_trips_and_rejects_an_incompatible_computation() {
    let mut operation = ar_decode_operation();
    operation.code = Computation::Pipeline(PipelineStage::VisionEncoding);
    operation.input_image = Some("iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAGUlEQVR4nGN0SGhgIAUwkaR6VMOohiGlAQCjvQFA6eri4wAAAABJRU5ErkJggg==".into());
    operation.token_output = None;
    let batch = batch_with_operations(1, vec![admission()], vec![operation.clone()]);
    let decoded = execute_round_trip(batch);
    assert_eq!(
        decoded.operations().next().unwrap().input_image,
        operation.input_image
    );

    for (code, image) in [
        (
            Computation::Forward(ForwardMode::Decode),
            operation.input_image.clone(),
        ),
        (
            Computation::Pipeline(PipelineStage::VisionEncoding),
            Some(String::new().into()),
        ),
    ] {
        let mut invalid = operation.clone();
        invalid.code = code;
        invalid.input_image = image;
        let request =
            WorkerRequest::submit(batch_with_operations(1, vec![admission()], vec![invalid]));
        assert!(encode_request(&request).is_err());
    }
}

#[test]
fn batch_rejects_a_conflicting_command_identity() {
    let finish = |retained_buffers| BatchCommand::Finish {
        request_key: request_key(),
        retained_buffers,
    };
    let buffer = tensor_for(
        request_key(),
        ComputationId::new(7, 0),
        0,
        DType::F16,
        ShapeBound {
            dims: vec![DimBound::Static(2), DimBound::Device { max: 16 }],
        },
    )
    .buffer_id();
    let batch = batch_with_operations(1, vec![admission()], vec![ar_decode_operation()])
        .with_commands(vec![finish(vec![]), finish(vec![buffer])]);
    assert!(batch.validate().is_err());
}

#[test]
fn batch_allows_a_duplicate_identical_command() {
    let finish = BatchCommand::Finish {
        request_key: request_key(),
        retained_buffers: vec![],
    };
    let batch = batch_with_operations(1, vec![admission()], vec![ar_decode_operation()])
        .with_commands(vec![finish.clone(), finish]);
    assert!(batch.validate().is_ok());
}

#[test]
fn worker_info_round_trips() {
    use uniserve_core::{ComponentConfig, ParallelConfig, SequenceParallel};
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
            kv_cache: Some(KvCacheInfo {
                num_layers: 9,
                total_layers: 28,
                layer_offset: 11,
                ..WorkerInfo::default().kv_cache.unwrap()
            }),
            components: vec![EntryInfo {
                name: "denoiser".into(),
                config: ComponentConfig::parallel((0..count).rev().collect(), config),
                outputs: vec![OutputInfo {
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
        pipeline_components: video_components(),
        num_inference_steps: 4,
        supported_ops: PipelineStage::VIDEO
            .into_iter()
            .map(Computation::Pipeline)
            .collect(),
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
        supported_ops: vec![
            Computation::Forward(ForwardMode::Prefill),
            Computation::Forward(ForwardMode::Decode),
            Computation::Forward(ForwardMode::Prefill),
        ],
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

fn tensor_for(
    key: RequestKey,
    op: ComputationId,
    output_index: u16,
    dtype: DType,
    shape_bound: ShapeBound,
) -> TensorRef {
    TensorRef {
        request_key: key,
        producer_op_id: op,
        output_index,
        generation: 3 + u32::from(output_index),
        dtype,
        shape_bound,
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

/// One operation per closed `Computation` variant, each on its own request key so the
/// batch admits them together; the first two keys also carry admissions.
fn comprehensive_batch() -> ScheduleBatch {
    let variants = Computation::ALL;
    let mut operations = Vec::new();
    for (index, kind) in variants.into_iter().enumerate() {
        let key = key_for_request(100 + index as u64);
        let op_id = ComputationId::new(11, index as u32);
        let predecessor = if index % 2 == 0 {
            ComputationId::new(0, 0)
        } else {
            ComputationId::new(9, 0)
        };
        operations.push(ScheduledRequest {
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
            kv_input: (kind == Computation::Transfer(TransferMode::KvInstall)).then_some(
                BufferId {
                    owner: key,
                    producer_op_id: ComputationId::new(2, 0),
                    output_index: 0,
                    generation: 1,
                },
            ),
            kv_output: matches!(
                kind,
                Computation::Transfer(TransferMode::KvPublish | TransferMode::KvInstall)
            )
            .then_some(BufferId {
                owner: key,
                producer_op_id: op_id,
                output_index: 1,
                generation: 2,
            }),
            input_token_ids: Vec::new(),
            sampling_state: None,
            request_key: key,
            op_id,
            predecessor: Some(predecessor),
            entry: "model".into(),
            code: kind,
            bounds: Bounds {
                max_tokens: 7 + index as u32,
                max_kv_pages: 3,
                max_latent_bytes: 1 << 20,
                max_completion_bytes: 4096,
                max_transfer_bytes: 1 << 16,
            },
            inputs: vec![tensor_for(
                key,
                ComputationId::new(2, 0),
                0,
                DType::F16,
                ShapeBound {
                    dims: vec![DimBound::Static(2), DimBound::Device { max: 16 }],
                },
            )],
            outputs: Vec::new(),
            predicate: Some(tensor_for(
                key,
                ComputationId::new(3, 0),
                0,
                DType::U8,
                ShapeBound::default(),
            )),
            rng: Some(Rng {
                seed: 99 + index as u64,
                semantic_index_base: 4,
                draw_layout: match index % 3 {
                    0 => DrawLayout::TargetSampling,
                    1 => DrawLayout::SpeculativeProposal,
                    _ => DrawLayout::FlowNoise,
                },
            }),
        });
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
    let commands = vec![BatchCommand::Finish {
        request_key: key_for_request(201),
        retained_buffers: Vec::new(),
    }];
    operations[0].input_token_ids = vec![7, 8, 9, 10];
    batch_with_operations(42, vec![ar_params, umm_params], operations).with_commands(commands)
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
        encoder_cache_entries: 64,
        encoder_entry_bytes: 128 << 20,
        pipeline_components: video_components(),
        num_inference_steps: 4,
        supported_ops: Computation::ALL.to_vec(),
        kv_cache: Some(KvCacheInfo {
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

fn video_components() -> std::collections::BTreeMap<PipelineStage, String> {
    [
        (PipelineStage::TextEncoding, "text_encoder"),
        (PipelineStage::LatentPreparation, "denoiser"),
        (PipelineStage::Denoising, "denoiser"),
        (PipelineStage::VideoDecoding, "video_decoder"),
        (PipelineStage::AudioDecoding, "audio_decoder"),
        (PipelineStage::VideoEncoding, "output"),
        (PipelineStage::AudioEncoding, "output"),
        (PipelineStage::Muxing, "output"),
    ]
    .into_iter()
    .map(|(stage, entry)| (stage, entry.to_owned()))
    .collect()
}

fn full_forward_stats() -> ForwardStats {
    let map = |prefix: &str, base: u64| {
        BTreeMap::from([
            (format!("{prefix}.a"), base),
            (format!("{prefix}.b"), base + 1),
        ])
    };
    ForwardStats {
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

fn full_run_result() -> BatchOutput {
    let mut ok_record = RequestOutput {
        sampled_logprob: None,
        top_logprobs: Vec::new(),
        prompt_logprobs: Vec::new(),
        request_key: key_for_request(100),
        op_id: ComputationId::new(11, 0),

        status: OpStatus::Ok,

        product_generations: vec![3, 5],
        error_code: None,
        timing_counters: TimingCounters {
            queued_us: 41,
            device_us: 42,
            copy_us: 43,
            host_us: 44,
        },
        code: Computation::Forward(ForwardMode::Decode),
        position: 5,
        kv_visible_len: 6,
        num_completed_steps: 7,
        kv_computed_len: 8,

        committed_tokens: vec![271, 272],
        finish_flags: FinishFlags {
            eos: true,
            length: false,
            stop: false,
        },
        media_output: None,
        kv_output: None,
    };
    let mut predicated_record = ok_record.clone();
    predicated_record.request_key = key_for_request(101);
    predicated_record.op_id = ComputationId::new(11, 1);
    predicated_record.status = OpStatus::Predicated;
    predicated_record.committed_tokens.clear();
    predicated_record.finish_flags = FinishFlags::default();
    predicated_record.product_generations = Vec::new();
    let mut error_record = ok_record.clone();
    error_record.request_key = key_for_request(102);
    error_record.op_id = ComputationId::new(11, 2);
    error_record.status = OpStatus::Error;
    error_record.error_code = Some(ErrorCode::ResourceExhausted);
    error_record.finish_flags = FinishFlags {
        eos: false,
        length: false,
        stop: true,
    };
    ok_record.media_output = Some(MediaOutput {
        handle: ArtifactHandle::PosixShm {
            name: "rendered-image".into(),
        },
        bytes: 256,
    });
    let products = Vec::new();
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
                        op_id: ComputationId::new(11, 0),
                    },
                    ErrorOperationIdentity {
                        request_key: key_for_request(101),
                        op_id: ComputationId::new(12, 0),
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
