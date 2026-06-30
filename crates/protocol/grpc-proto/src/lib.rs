//! Generated gRPC protobuf types for UniServe.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
#[allow(unreachable_pub)]
#[allow(clippy::clone_on_ref_ptr)]
pub mod pb {
    tonic::include_proto!("uniserve");
}

#[cfg(test)]
mod tests {
    use prost::Message;

    use crate::pb;

 /// A request carrying a text prompt round-trips through protobuf encode and
 /// decode, preserving the `Prompt::Text` oneof and scalar fields.
    #[test]
    fn generate_request_text_prompt_round_trips() {
        let request = pb::GenerateRequest {
            request_id: "req-1".to_string(),
            model: "m".to_string(),
            temperature: Some(0.7),
            truncate_prompt_tokens: 4,
            priority: 2,
            prompt: Some(pb::generate_request::Prompt::Text("hello".to_string())),
            ..Default::default()
        };

        let bytes = request.encode_to_vec();
        let decoded = pb::GenerateRequest::decode(bytes.as_slice()).unwrap();

        assert_eq!(decoded.request_id, "req-1");
        assert_eq!(decoded.model, "m");
        assert_eq!(decoded.temperature, Some(0.7));
        assert_eq!(decoded.truncate_prompt_tokens, 4);
        assert_eq!(decoded.priority, 2);
        assert_eq!(
            decoded.prompt,
            Some(pb::generate_request::Prompt::Text("hello".to_string()))
        );
    }

 /// The token-ids prompt oneof preserves the `TokenIds` message through a
 /// protobuf round-trip.
    #[test]
    fn generate_request_token_ids_prompt_round_trips() {
        let request = pb::GenerateRequest {
            request_id: "req-2".to_string(),
            prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
                ids: vec![5, 6, 7],
            })),
            ..Default::default()
        };

        let decoded =
            pb::GenerateRequest::decode(request.encode_to_vec().as_slice()).unwrap();

        match decoded.prompt {
            Some(pb::generate_request::Prompt::TokenIds(token_ids)) => {
                assert_eq!(token_ids.ids, vec![5, 6, 7]);
            }
            other => panic!("expected TokenIds prompt, got {other:?}"),
        }
    }

 /// The `structured_output` oneof selects the `Choice` variant and preserves
 /// the nested choice strings across a protobuf round-trip.
    #[test]
    fn decoding_parameters_choice_oneof_round_trips() {
        let decoding = pb::DecodingParameters {
            repetition_penalty: 1.1,
            structured_output: Some(pb::decoding_parameters::StructuredOutput::Choice(
                pb::decoding_parameters::StringChoices {
                    choices: vec!["yes".to_string(), "no".to_string()],
                },
            )),
            ..Default::default()
        };

        let decoded =
            pb::DecodingParameters::decode(decoding.encode_to_vec().as_slice()).unwrap();

        assert_eq!(decoded.repetition_penalty, 1.1);
        match decoded.structured_output {
            Some(pb::decoding_parameters::StructuredOutput::Choice(choices)) => {
                assert_eq!(choices.choices, vec!["yes".to_string(), "no".to_string()]);
            }
            other => panic!("expected Choice structured output, got {other:?}"),
        }
    }

 /// A `FinishInfo` carrying a STOP reason round-trips: the enum tag decodes
 /// back to `FinishReason::Stop` and the `stop_reason` oneof preserves the
 /// stop string.
    #[test]
    fn finish_info_stop_reason_round_trips() {
        let finish = pb::FinishInfo {
            num_output_tokens: 12,
            finish_reason: pb::finish_info::FinishReason::Stop as i32,
            stop_reason: Some(pb::finish_info::StopReason::StopString("</s>".to_string())),
            ..Default::default()
        };

        let decoded = pb::FinishInfo::decode(finish.encode_to_vec().as_slice()).unwrap();

        assert_eq!(decoded.num_output_tokens, 12);
        assert_eq!(
            decoded.finish_reason,
            pb::finish_info::FinishReason::Stop as i32
        );
        assert_eq!(
            decoded.stop_reason,
            Some(pb::finish_info::StopReason::StopString("</s>".to_string()))
        );
    }

 /// The generated `FinishReason` enum maps each variant to its stable
 /// ProtoBuf field name and back.
    #[test]
    fn finish_reason_str_name_round_trips() {
        use pb::finish_info::FinishReason;

        assert_eq!(FinishReason::Stop.as_str_name(), "STOP");
        assert_eq!(FinishReason::Length.as_str_name(), "LENGTH");
        assert_eq!(FinishReason::from_str_name("ABORTED"), Some(FinishReason::Aborted));
        assert_eq!(FinishReason::from_str_name("NOT_FINISHED"), Some(FinishReason::NotFinished));
        assert_eq!(FinishReason::from_str_name("unknown"), None);
    }

 /// An empty buffer decodes into the message default (all proto3 fields at
 /// their zero values), confirming optional/absent fields stay unset.
    #[test]
    fn generate_request_decodes_empty_buffer_as_default() {
        let decoded = pb::GenerateRequest::decode(Vec::new().as_slice()).unwrap();

        assert_eq!(decoded, pb::GenerateRequest::default());
        assert_eq!(decoded.request_id, "");
        assert_eq!(decoded.temperature, None);
        assert!(decoded.prompt.is_none());
    }
}
