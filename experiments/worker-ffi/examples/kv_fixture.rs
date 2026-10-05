//! Emit two KV installation inputs for native importer interface tests.

use std::io::Write;

use uniserve_core::RequestId;
use uniserve_worker_ipc::{
    codec, Batch, Bounds, BufferId, Call, CallCoordinates, CallId, CallKind, KvGroupTransfer,
    KvTransfer, Locator, RequestKey, TensorTransfer, TransferMode, TransferTransport,
    WorkerEndpoint, WorkerRequest,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut batch = Batch::new(17, Vec::new(), Vec::new());
    for index in 0..2 {
        let key = RequestKey::new(1, RequestId(index + 1), 1);
        let call_id = CallId::new(17, index as u32);
        let source = BufferId {
            owner: key,
            producer_call_id: CallId::new(1, 0),
            output_index: 0,
            generation: 1,
        };
        batch.calls.push(Call {
            request_key: key,
            call_id,
            coordinates: CallCoordinates::default(),
            component: "model".into(),
            code: CallKind::Transfer(TransferMode::KvInstall),
            bounds: Bounds {
                max_transfer_bytes: 32,
                ..Bounds::default()
            },
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
            kv_input: Some(source),
            kv_output: Some(BufferId {
                producer_call_id: call_id,
                generation: 2,
                ..source
            }),
        });
        let tensor = TensorTransfer {
            shape: vec![2, 1, 1, 2],
            locations: vec![Locator {
                source: WorkerEndpoint {
                    worker_id: "producer".into(),
                    rank: 0,
                    node: "host".into(),
                    address_space: "process".into(),
                    incarnation: "rank".into(),
                },
                transport: TransferTransport::Channel {
                    endpoint: "producer".into(),
                    payload: [1.0_f32, 2.0, 3.0, 4.0]
                        .into_iter()
                        .flat_map(f32::to_le_bytes)
                        .collect(),
                },
                nbytes: 16,
                dtype: "float32".into(),
                shape: vec![2, 1, 1, 2],
                offset: vec![0; 4],
                device: "cpu".into(),
            }],
        };
        batch.kv_inputs.push(KvTransfer {
            groups: vec![KvGroupTransfer {
                start: 0,
                page_tokens: 2,
                tensors: vec![tensor.clone(), tensor],
            }],
            source,
            destination: "consumer".into(),
            base: None,
            base_extent: 0,
            exported_extent: 2,
            compute_dtype: "float32".into(),
        });
    }
    let request = WorkerRequest::Submit {
        message_id: Some(23),
        batch: Box::new(batch),
    };
    std::io::stdout().write_all(&codec::encode_request(&request)?)?;
    Ok(())
}
