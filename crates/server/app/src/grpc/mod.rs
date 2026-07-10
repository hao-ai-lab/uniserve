//! gRPC Generate service backed by the shared [`uniserve_serving::text::TextRuntime`] facade.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use thiserror_ext::AsReport as _;
use tonic::{Request, Response, Status};
use tracing::info;

use crate::AppState;
use uniserve_protocol_adapters::grpc::{self as grpc_adapter, ResponseOpts};

pub use pb::generate_server::GenerateServer;
pub use uniserve_grpc_proto::pb;

#[cfg(test)]
mod tests;

/// gRPC Generate service implementation backed by the shared application state.
pub struct GenerateServiceImpl {
    state: Arc<AppState>,
}

impl GenerateServiceImpl {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl pb::generate_server::Generate for GenerateServiceImpl {
    type GenerateStreamStream =
        Pin<Box<dyn Stream<Item = Result<pb::GenerateResponse, Status>> + Send>>;

    /// Unary generate: collect all output and return a single response.
    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<pb::GenerateResponse>, Status> {
        let proto_req = request.into_inner();
        let response_opts = ResponseOpts::from_proto(proto_req.response.as_ref());

        // Resolve the requested model against one dynamic LoRA registry snapshot so
        // gRPC shares the same model+adapter selection contract as the HTTP routes:
        // validate against base names plus loaded adapter names, and attach the
        // resolved adapter on the semantic request. An empty proto3 model
        // field is treated as "unset" (no adapter), matching `to_text_request`.
        let model_name = (!proto_req.model.is_empty()).then(|| proto_req.model.clone());
        let lora_resolution = self
            .state
            .resolve_model_with_loras(model_name.as_deref())
            .await;
        let mut text_request = uniserve_protocol_adapters::grpc::to_text_request(
            proto_req,
            false,
            &lora_resolution.model_names,
        )?;
        text_request.adapter = lora_resolution.adapter;

        let request_id = text_request.request_id.clone();
        info!(%request_id, "grpc generate (unary)");

        let serve_request = grpc_adapter::to_serve_request(text_request)?;
        let stream = self
            .state
            .runtime()
            .serve(serve_request)
            .await
            .map_err(|e| Status::internal(e.to_report_string()))?;
        let (collected, finish_status) = grpc_adapter::collect_serve_events(stream).await?;
        Ok(Response::new(grpc_adapter::collected_response(
            collected,
            &finish_status,
            &response_opts,
        )))
    }

    /// Streaming generate: yield incremental responses as tokens are produced.
    async fn generate_stream(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStreamStream>, Status> {
        let proto_req = request.into_inner();
        let response_opts = ResponseOpts::from_proto(proto_req.response.as_ref());

        // Resolve the requested model against one dynamic LoRA registry snapshot so
        // gRPC shares the same model+adapter selection contract as the HTTP routes:
        // validate against base names plus loaded adapter names, and attach the
        // resolved adapter on the semantic request. An empty proto3 model
        // field is treated as "unset" (no adapter), matching `to_text_request`.
        let model_name = (!proto_req.model.is_empty()).then(|| proto_req.model.clone());
        let lora_resolution = self
            .state
            .resolve_model_with_loras(model_name.as_deref())
            .await;
        let mut text_request = uniserve_protocol_adapters::grpc::to_text_request(
            proto_req,
            true,
            &lora_resolution.model_names,
        )?;
        text_request.adapter = lora_resolution.adapter;

        let request_id = text_request.request_id.clone();
        info!(%request_id, "grpc generate (stream)");

        let serve_request = grpc_adapter::to_serve_request(text_request)?;
        let stream = self
            .state
            .runtime()
            .serve(serve_request)
            .await
            .map_err(|e| Status::internal(e.to_report_string()))?;

        let response_stream = grpc_adapter::response_stream(stream, response_opts);
        Ok(Response::new(Box::pin(response_stream)))
    }
}
