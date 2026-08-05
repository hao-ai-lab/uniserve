use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use serde_json::Value;
use tower::ServiceExt as _;
use uniserve_engine_gateway::MockClientMessage;
use uniserve_engine_gateway::protocol::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs,
};
use uniserve_testkit::{canonical_engine_outputs, mock_engine_gateway, resolved_model_fixture};

use super::build_router;
use crate::AppState;

fn test_app() -> (axum::Router, uniserve_engine_gateway::MockEngine) {
    let (gateway, mock) = mock_engine_gateway("fixture-model");
    let engine_control = gateway.app_control();
    let runtime =
        uniserve_serving::ServingRuntime::new(resolved_model_fixture("fixture-model"), gateway);
    let state = Arc::new(AppState::new(
        "fixture-model".to_string(),
        runtime,
        engine_control,
    ));
    (build_router(state), mock)
}

async fn body_json(response: axum::response::Response) -> Value {
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read response body");
    serde_json::from_slice(&bytes).expect("JSON response")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn configured_routes_report_readiness_model_and_image_admission() {
    let (app, _mock) = test_app();

    let health = app
        .clone()
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);

    let version = app
        .clone()
        .oneshot(Request::get("/version").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(version.status(), StatusCode::OK);

    let models = app
        .clone()
        .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(models.status(), StatusCode::OK);
    let models = body_json(models).await;
    assert_eq!(models["data"].as_array().unwrap().len(), 1);
    assert_eq!(models["data"][0]["id"], "fixture-model");

    let image = app
        .oneshot(
            Request::post("/v1/images/generations")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "fixture-model",
                        "prompt": "paint a lighthouse"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(image.status(), StatusCode::BAD_REQUEST);
    let image = body_json(image).await;
    assert!(
        image["error"]["message"]
            .as_str()
            .unwrap()
            .contains("image_output")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_request_lowers_tokenizes_submits_once_and_assembles_ordered_output() {
    let (app, mut mock) = test_app();
    let engine = tokio::spawn(async move {
        let request = mock.recv_request().await;
        assert_eq!(request.request_id, "chatcmpl-route-funnel");
        let rendered = String::from_utf8_lossy(
            &request
                .generation
                .prompt_token_ids()
                .iter()
                .map(|token| *token as u8)
                .collect::<Vec<_>>(),
        )
        .into_owned();
        assert!(rendered.contains("hello from HTTP"));

        let request_id = request.request_id.clone();
        mock.send_outputs(canonical_engine_outputs(
            &request,
            EngineCoreOutputs {
                outputs: vec![
                    EngineCoreOutput {
                        request_id: request_id.clone(),
                        new_token_ids: vec![u32::from(b'H')],
                        ..EngineCoreOutput::default()
                    },
                    EngineCoreOutput {
                        request_id: request_id.clone(),
                        new_token_ids: vec![u32::from(b'i')],
                        ..EngineCoreOutput::default()
                    },
                    EngineCoreOutput {
                        request_id,
                        finish_reason: Some(EngineCoreFinishReason::Length),
                        ..EngineCoreOutput::default()
                    },
                ],
                ..EngineCoreOutputs::default()
            },
        ));

        loop {
            match tokio::time::timeout(std::time::Duration::from_millis(50), mock.recv()).await {
                Ok(Some(MockClientMessage::Add(_))) => {
                    panic!("received a second gateway submission")
                }
                Ok(Some(_)) => continue,
                Ok(None) | Err(_) => break,
            }
        }
    });

    let response = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("x-request-id", "route-funnel")
                .body(Body::from(
                    serde_json::json!({
                        "model": "fixture-model",
                        "messages": [{"role": "user", "content": "hello from HTTP"}],
                        "max_completion_tokens": 8
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response = body_json(response).await;
    assert_eq!(response["id"], "chatcmpl-route-funnel");
    assert_eq!(response["model"], "fixture-model");
    assert_eq!(response["choices"][0]["message"]["content"], "Hi");
    assert_eq!(response["choices"][0]["finish_reason"], "length");
    engine.await.unwrap();
}
