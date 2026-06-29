//! gRPC Generate service backed by the shared [`uniserve_text::TextLlm`] facade.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod convert;

use std::pin::Pin;
use std::sync::Arc;

use futures::{Stream, StreamExt as _};
use thiserror_ext::AsReport as _;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use tracing::info;
use uniserve_text::{DecodedTextEvent, TextOutputStreamExt as _};

use self::convert::ResponseOpts;
use uniserve_server_app::AppState;

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
 // resolved adapter as the request's `lora_request`. An empty proto3 model
 // field is treated as "unset" (no adapter), matching `to_text_request`.
        let model_name = (!proto_req.model.is_empty()).then(|| proto_req.model.clone());
        let lora_resolution = self
            .state
            .resolve_model_with_loras(model_name.as_deref())
            .await;
        let mut text_request =
            convert::to_text_request(proto_req, false, &lora_resolution.model_names)?;
        text_request.lora_request = lora_resolution.lora_request;

        let request_id = text_request.request_id.clone();
        info!(%request_id, "grpc generate (unary)");

        let stream = self.state.chat().text().generate(text_request).await;
        let stream = stream.map_err(|e| Status::internal(e.to_report_string()))?;

        let collected = stream
            .collect_output()
            .await
            .map_err(|e| Status::internal(e.to_report_string()))?;

 // Build the single aggregated response.
        let prompt_info = convert::to_prompt_info(
            &collected.prompt_token_ids,
            collected.prompt_logprobs.as_ref(),
            &response_opts,
        );

        let finish_info = uniserve_text::Finished {
            prompt_token_count: collected.prompt_token_ids.len(),
            output_token_count: collected.token_ids.len(),
            finish_reason: collected.finish_reason,
            kv_transfer_params: collected.kv_transfer_params,
        };

        let outputs = convert::to_sequence_output(
            &collected.text,
            &collected.token_ids,
            collected.logprobs.as_ref(),
            Some(&finish_info),
            &response_opts,
        );

        Ok(Response::new(pb::GenerateResponse {
            prompt_info: Some(prompt_info),
            outputs: Some(outputs),
        }))
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
 // resolved adapter as the request's `lora_request`. An empty proto3 model
 // field is treated as "unset" (no adapter), matching `to_text_request`.
        let model_name = (!proto_req.model.is_empty()).then(|| proto_req.model.clone());
        let lora_resolution = self
            .state
            .resolve_model_with_loras(model_name.as_deref())
            .await;
        let mut text_request =
            convert::to_text_request(proto_req, true, &lora_resolution.model_names)?;
        text_request.lora_request = lora_resolution.lora_request;

        let request_id = text_request.request_id.clone();
        info!(%request_id, "grpc generate (stream)");

        let stream = self.state.chat().text().generate(text_request).await;
        let stream = stream.map_err(|e| Status::internal(e.to_report_string()))?;

        let (tx, rx) = mpsc::channel(32);

        tokio::spawn(async move {
            futures::pin_mut!(stream);
            while let Some(event) = stream.next().await {
                let response = match event {
                    Err(e) => Err(Status::internal(e.to_report_string())),
                    Ok(DecodedTextEvent::Start {
                        prompt_token_ids,
                        prompt_logprobs,
                    }) => {
                        let prompt_info = convert::to_prompt_info(
                            &prompt_token_ids,
                            prompt_logprobs.as_ref(),
                            &response_opts,
                        );
                        Ok(pb::GenerateResponse {
                            prompt_info: Some(prompt_info),
                            outputs: None,
                        })
                    }
                    Ok(DecodedTextEvent::TextDelta {
                        delta,
                        token_ids,
                        logprobs,
                        finished,
                    }) => Ok(pb::GenerateResponse {
                        prompt_info: None,
                        outputs: Some(convert::to_sequence_output(
                            &delta,
                            &token_ids,
                            logprobs.as_ref(),
                            finished.as_ref(),
                            &response_opts,
                        )),
                    }),
                };

                if tx.send(response).await.is_err() {
 // Client disconnected.
                    break;
                }
            }
        });

        let response_stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(response_stream)))
    }
}
