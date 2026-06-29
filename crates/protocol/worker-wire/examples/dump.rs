// Cross-language wire-format check: encode sample FlatBuffers messages and
// write them to files for external inspection.
use uniserve_core::{BlockId, Modality, RequestId, SamplingParams};
use uniserve_worker_wire::*;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or("/tmp/wire_test.bin".into());
    let batch = ForwardBatch {
        step_id: 7,
        new_reqs: vec![NewRequestData {
            sampling: Some(SamplingParams::default()),
            block_ids: vec![BlockId(0), BlockId(1)],
            ..NewRequestData::new(RequestId(42))
        }],
        ops: vec![
            ForwardOp {
                req_id: RequestId(42),
                kind: OpKind::PrefillUnd,
                modality: Modality::Und,
                pos_range: (0, 5),
                token_ids: Some(vec![151644, 100, 200, 300, 151645]),
                ..Default::default()
            },
            ForwardOp {
                req_id: RequestId(43),
                kind: OpKind::DenoiseGen,
                modality: Modality::Gen,
                new_block_ids: vec![BlockId(2)],
                pos_range: (5, 6),
                timestep_idx: Some(3),
                cond_pos: Some(5),
                ..Default::default()
            },
        ],
    };
    let req = WorkerRequest::execute(batch);
    let bytes = flat::encode_request(&req).unwrap();
    std::fs::write(&path, &bytes).unwrap();
    println!("wrote {} bytes to {}", bytes.len(), path);

 // also round-trip a response
    let resp = WorkerResponse {
        kind: "caps".into(),
        call_id: None,
        caps: Some(EngineCaps {
            num_blocks: 1000,
            scratch_capacity_tokens: 100000,
            supported_ops: vec!["prefill_und".into()],
            ..Default::default()
        }),
        result: None,
        metrics: None,
        pressure: None,
        message: None,
        code: None,
        retryable: None,
        fatal: None,
    };
    std::fs::write(
        format!("{path}.resp"),
        flat::encode_response(&resp).unwrap(),
    )
    .unwrap();

 // self round-trip check
    let back = flat::decode_request(&bytes).unwrap();
    println!(
        "rust round-trip ok: {:?}",
        back.kind == RequestKind::Execute
    );

    let rt = flat::decode_response(&flat::encode_response(&resp).unwrap()).unwrap();
    println!("resp decode ok: {} {}", rt.kind, rt.caps.is_some());
}
