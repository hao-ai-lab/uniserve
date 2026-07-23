//! Cross-language wire-format check for protocol v3.

use uniserve_core::{BlockId, ImageParams, RequestId, SamplingParams};
use uniserve_worker_wire::*;

fn digest() -> String {
    "ab".repeat(32)
}

fn seal(
    session_id: RequestId,
    admission_digest: String,
    operation: Operation,
    op_id: u64,
) -> OperationEnvelope {
    let mut envelope = OperationEnvelope::unsealed(session_id, operation);
    envelope.admission_digest = admission_digest;
    envelope.model_spec_digest = digest();
    envelope.weight_digest = digest();
    envelope.seal(1, op_id, 0);
    envelope
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/wire_test.bin".into());

    let sequence_admission = Admission::new(
        RequestId(42),
        Some(SequenceAdmission {
            sampling: SamplingParams::default(),
            negative_token_ids: Vec::new(),
            kv: KvAllocation {
                block_ids: vec![BlockId(0), BlockId(1)],
                prefix_len: 0,
                group_id: 0,
            },
        }),
        None,
        None,
    )
    .unwrap();
    let flow_admission = Admission::new(
        RequestId(43),
        None,
        Some(FlowAdmission {
            image: ImageParams::default(),
        }),
        None,
    )
    .unwrap();

    let operations = vec![
        seal(
            RequestId(42),
            sequence_admission.digest.clone(),
            Operation::Sequence(SequenceOperation {
                mode: SequenceMode::Extend,
                lease: KvLeaseDelta::default(),
                position: (0, 5),
                policy: TokenPolicy::default(),
                input: SequenceInput::Tokens(TokenInput {
                    token_ids: vec![151_644, 100, 200, 300, 151_645],
                    source: TokenSource::Wire,
                    draft_token_ids: Vec::new(),
                    burst_tokens: 1,
                    stop_token_ids: Vec::new(),
                    stop_terminal: false,
                    return_all_logits: false,
                }),
            }),
            1,
        ),
        seal(
            RequestId(43),
            flow_admission.digest.clone(),
            Operation::Flow(FlowOperation {
                latent_handle: 1,
                position: 5,
                start_step: 3,
                step_count: 1,
                conditioning_position: 5,
                conditioning: None,
                guidance: Guidance {
                    branch_count: 1,
                    text_scale: 1.0,
                    image_scale: 1.0,
                    renorm_type: "none".into(),
                    renorm_min: 0.0,
                    interval: (0.0, 1.0),
                },
                image_prompt: String::new(),
            }),
            2,
        ),
    ];
    let batch = Batch::new(7, vec![sequence_admission, flow_admission], operations);
    let request = WorkerRequest::execute(batch);
    let bytes = flat::encode_request(&request).unwrap();
    std::fs::write(&path, &bytes).unwrap();
    println!("wrote {} bytes to {}", bytes.len(), path);

    let response = WorkerResponse::capabilities(EngineCaps {
        num_blocks: 1_000,
        scratch_capacity_tokens: 100_000,
        supported_operation_types: OperationType::ALL.to_vec(),
        model_spec_digest: digest(),
        weight_digest: digest(),
        ..EngineCaps::default()
    });
    std::fs::write(
        format!("{path}.resp"),
        flat::encode_response(&response).unwrap(),
    )
    .unwrap();

    let decoded_request = flat::decode_request(&bytes).unwrap();
    println!(
        "request round-trip ok: {}",
        decoded_request.kind == RequestKind::Execute
    );
    let decoded_response =
        flat::decode_response(&flat::encode_response(&response).unwrap()).unwrap();
    println!(
        "response round-trip ok: {:?} {}",
        decoded_response.kind,
        decoded_response.capabilities.is_some()
    );
}
