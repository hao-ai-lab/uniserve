//! gRPC Generate service backed by the shared [`uniserve_text::TextLlm`] facade.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod convert;

use std::pin::Pin;
use std::sync::Arc;

use futures::{Stream, StreamExt as _, pin_mut};
use thiserror_ext::AsReport as _;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use tracing::info;
use uniserve_serving::{FinishStatus, RequestMetadata, ServeEvent, ServeRequest};
use uniserve_text::{CollectedTextOutput, DecodedLogprobs, Finished};

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

        let serve_request = serve_request_from_text(text_request)?;
        let stream = self
            .state
            .runtime()
            .serve(serve_request)
            .await
            .map_err(|e| Status::internal(e.to_report_string()))?;
        let (collected, finish_status) = collect_text_events(stream).await?;

        // Build the single aggregated response.
        let prompt_info = convert::to_prompt_info(
            &collected.prompt_token_ids,
            collected.prompt_logprobs.as_ref(),
            &response_opts,
        );

        let finish_info = uniserve_text::Finished {
            prompt_token_count: collected.prompt_token_ids.len(),
            output_token_count: collected.token_ids.len(),
            finish_reason: finish_status_to_text_reason(&finish_status),
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

        let serve_request = serve_request_from_text(text_request)?;
        let stream = self
            .state
            .runtime()
            .serve(serve_request)
            .await
            .map_err(|e| Status::internal(e.to_report_string()))?;

        let (tx, rx) = mpsc::channel(32);

        tokio::spawn(async move {
            futures::pin_mut!(stream);
            let mut prompt_tokens = 0_usize;
            let mut output_tokens = 0_usize;
            while let Some(event) = stream.next().await {
                let response = match event {
                    Err(e) => Err(Status::internal(e.to_report_string())),
                    Ok(ServeEvent::Accepted {
                        prompt_token_ids,
                        prompt_logprobs,
                        ..
                    }) => {
                        prompt_tokens = prompt_token_ids.len();
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
                    Ok(ServeEvent::TextDelta {
                        text,
                        token_ids,
                        logprobs,
                        ..
                    }) => Ok(pb::GenerateResponse {
                        prompt_info: None,
                        outputs: Some(convert::to_sequence_output(
                            &text,
                            &token_ids,
                            logprobs.as_ref(),
                            None,
                            &response_opts,
                        )),
                    }),
                    Ok(ServeEvent::Usage {
                        prompt_tokens: prompt,
                        visible_output_tokens,
                        ..
                    }) => {
                        prompt_tokens = prompt as usize;
                        output_tokens = visible_output_tokens as usize;
                        continue;
                    }
                    Ok(ServeEvent::Finished {
                        reason,
                        kv_transfer_params,
                        ..
                    }) => {
                        let finished = Finished {
                            prompt_token_count: prompt_tokens,
                            output_token_count: output_tokens,
                            finish_reason: finish_status_to_text_reason(&reason),
                            kv_transfer_params,
                        };
                        Ok(pb::GenerateResponse {
                            prompt_info: None,
                            outputs: Some(convert::to_sequence_output(
                                "",
                                &[],
                                None,
                                Some(&finished),
                                &response_opts,
                            )),
                        })
                    }
                    Ok(_) => continue,
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

fn serve_request_from_text(
    text_request: uniserve_text::TextRequest,
) -> Result<ServeRequest, Status> {
    ServeRequest::from_text_request(
        text_request,
        RequestMetadata {
            protocol_adapter: Some("grpc_generate".to_string()),
            route: Some("grpc.Generate".to_string()),
            ..RequestMetadata::default()
        },
    )
    .map_err(|error| match error {
        uniserve_serving::ServeError::UnsupportedRuntimeExtension { key, .. } => {
            Status::invalid_argument(format!("unsupported runtime extension `{key}`"))
        }
        uniserve_serving::ServeError::UnsupportedOutputCount { requested, .. } => {
            Status::invalid_argument(format!("only one sequence is supported, got {requested}"))
        }
        other => Status::internal(other.to_report_string()),
    })
}

async fn collect_text_events(
    stream: impl Stream<Item = uniserve_serving::Result<ServeEvent>> + Send,
) -> Result<(CollectedTextOutput, FinishStatus), Status> {
    pin_mut!(stream);
    let mut prompt_token_ids = std::sync::Arc::<[u32]>::from([]);
    let mut prompt_logprobs = None;
    let mut text = String::new();
    let mut token_ids = Vec::new();
    let mut logprobs: Option<DecodedLogprobs> = None;
    let mut finish_status = None;
    let mut kv_transfer_params = None;

    while let Some(event) = stream.next().await {
        match event.map_err(|e| Status::internal(e.to_report_string()))? {
            ServeEvent::Accepted {
                prompt_token_ids: ids,
                prompt_logprobs: start_prompt_logprobs,
                ..
            } => {
                prompt_token_ids = ids.into();
                prompt_logprobs = start_prompt_logprobs;
            }
            ServeEvent::TextDelta {
                text: delta,
                token_ids: delta_token_ids,
                logprobs: delta_logprobs,
                ..
            } => {
                text.push_str(&delta);
                token_ids.extend(delta_token_ids);
                if let Some(mut delta_logprobs) = delta_logprobs {
                    logprobs
                        .get_or_insert_with(|| DecodedLogprobs {
                            positions: Vec::new(),
                        })
                        .positions
                        .append(&mut delta_logprobs.positions);
                }
            }
            ServeEvent::Finished {
                reason,
                kv_transfer_params: params,
                ..
            } => {
                finish_status = Some(reason);
                kv_transfer_params = params;
                break;
            }
            _ => {}
        }
    }

    let Some(finish_status) = finish_status else {
        return Err(Status::internal(
            "stream closed before terminal finish event",
        ));
    };
    let finish_reason = finish_status_to_text_reason(&finish_status);

    Ok((
        CollectedTextOutput {
            text,
            prompt_token_ids,
            prompt_logprobs,
            logprobs,
            token_ids,
            finish_reason,
            kv_transfer_params,
        },
        finish_status,
    ))
}

fn finish_status_to_text_reason(status: &FinishStatus) -> uniserve_text::FinishReason {
    match status {
        FinishStatus::Stop { .. } => uniserve_text::FinishReason::Stop(None),
        FinishStatus::Length => uniserve_text::FinishReason::Length,
        FinishStatus::Abort => uniserve_text::FinishReason::Abort,
        FinishStatus::Error => uniserve_text::FinishReason::Error,
        FinishStatus::Repetition => uniserve_text::FinishReason::Repetition,
    }
}
