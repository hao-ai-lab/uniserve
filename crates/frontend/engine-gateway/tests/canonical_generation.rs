#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;

use uniserve_core::{
    ContextSegment, GenerationBehaviorDescriptor, GenerationConstraint, GenerationPolicyDescriptor,
    GenerationRequest, GenerationResourceBounds, ImageParams, RequestId, SamplingParams,
    UndVisibility,
};
use uniserve_engine_gateway::protocol::generation::{GenerationFinish, GenerationOutput};
use uniserve_engine_gateway::protocol::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs,
};
use uniserve_engine_gateway::{
    EngineCoreClient, EngineGateway, GenEvent, GenerationFinishReason, GenerationSubmission,
    MockClientMessage, StreamCancelCause,
};

fn text_request() -> GenerationRequest {
    let constraint = GenerationConstraint::UndOnly;
    let policy = GenerationPolicyDescriptor::default();
    GenerationRequest {
        request_id: RequestId(17),
        context: vec![ContextSegment::UndTokens {
            token_ids: vec![11, 22],
            visibility: UndVisibility::Internal,
        }],
        negative_context: Vec::new(),
        constraint,
        behavior: GenerationBehaviorDescriptor::resolve(constraint, &policy),
        sampling: SamplingParams::default(),
        image: ImageParams::default(),
        max_und_tokens: 3,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 4,
        lora_id: None,
        grammar: None,
        cache: Default::default(),
        policy,
        resources: GenerationResourceBounds {
            context_tokens: 2,
            max_kv_tokens: 5,
            ..GenerationResourceBounds::default()
        },
    }
}

#[tokio::test]
async fn canonical_submission_preserves_request_and_transport_metadata() {
    let (client, mut engine) = EngineCoreClient::connect_mock("test-model");
    let gateway = EngineGateway::new(client);
    let mut submission = GenerationSubmission::new("external-17", text_request());
    submission.arrival_time = Some(42.5);
    submission.data_parallel_rank = Some(3);
    submission.trace_headers = Some(BTreeMap::from([(
        "traceparent".to_string(),
        "00-abc-def-01".to_string(),
    )]));

    let mut events = gateway.submit_generation(submission).await.unwrap();
    let request = engine.recv_request().await;
    assert_eq!(request.request_id, "external-17");
    assert_eq!(request.arrival_time, 42.5);
    assert_eq!(request.data_parallel_rank, Some(3));
    assert_eq!(
        request
            .trace_headers
            .as_ref()
            .and_then(|headers| headers.get("traceparent")),
        Some(&"00-abc-def-01".to_string())
    );
    assert_eq!(request.generation.request_id, RequestId(17));
    assert_eq!(request.generation.prompt_token_ids(), vec![11, 22]);

    engine.send_outputs(EngineCoreOutputs {
        outputs: vec![EngineCoreOutput {
            request_id: request.request_id,
            new_token_ids: vec![31],
            finish_reason: Some(EngineCoreFinishReason::Length),
            generation: Some(GenerationOutput {
                finish: Some(GenerationFinish {
                    reason: "max_tokens".to_string(),
                    prompt_tokens: 2,
                    completion_tokens: 1,
                    images: 0,
                    message: None,
                }),
                ..GenerationOutput::default()
            }),
            ..EngineCoreOutput::default()
        }],
        ..EngineCoreOutputs::default()
    });

    assert!(matches!(
        events.next().await,
        Some(GenEvent::TextToken {
            id: 31,
            logprob: None,
            public_commit: None,
        })
    ));
    assert!(matches!(
        events.next().await,
        Some(GenEvent::Finished {
            reason: GenerationFinishReason::MaxTokens,
            prompt_tokens: 2,
            completion_tokens: 1,
            images: 0,
            ..
        })
    ));

    gateway.shutdown().await.unwrap();
}

#[tokio::test]
async fn cancellation_and_abort_use_distinct_controls() {
    let (client, mut engine) = EngineCoreClient::connect_mock("test-model");
    let gateway = EngineGateway::new(client);

    gateway.cancel("cancelled-request").await.unwrap();
    gateway.abort("aborted-request").await.unwrap();

    assert!(matches!(
        engine.recv().await,
        Some(MockClientMessage::Cancel(ids)) if ids == ["cancelled-request"]
    ));
    assert!(matches!(
        engine.recv().await,
        Some(MockClientMessage::Abort(ids)) if ids == ["aborted-request"]
    ));

    gateway.shutdown().await.unwrap();
}

#[tokio::test]
async fn decoder_prefix_decisions_preserve_exact_stream_controls() {
    let (client, mut engine) = EngineCoreClient::connect_mock("test-model");
    let gateway = EngineGateway::new(client);
    let mut request = text_request();
    request.stop_strings = vec!["boundary".to_string()];
    let mut events = gateway
        .submit_generation(GenerationSubmission::new("stop-request", request))
        .await
        .unwrap();
    let request = engine.recv_request().await;

    for token_id in [31, 32] {
        engine.send_outputs(EngineCoreOutputs {
            outputs: vec![EngineCoreOutput {
                request_id: request.request_id.clone(),
                new_token_ids: vec![token_id],
                ..EngineCoreOutput::default()
            }],
            ..EngineCoreOutputs::default()
        });
        assert!(matches!(
            events.next().await,
            Some(GenEvent::TextToken { id, .. }) if id == token_id
        ));
        if token_id == 31 {
            events.acknowledge_text_prefix();
            assert!(matches!(
                engine.recv().await,
                Some(MockClientMessage::Acknowledge {
                    request_id,
                    output_token_count: 1,
                }) if request_id == "stop-request"
            ));
        }
    }

    events.cancel_at_consumed_prefix(StreamCancelCause::StopStringMatched);
    assert!(matches!(
        engine.recv().await,
        Some(MockClientMessage::StopAt {
            request_id,
            output_token_count: 2,
        }) if request_id == "stop-request"
    ));
    gateway.shutdown().await.unwrap();
}
