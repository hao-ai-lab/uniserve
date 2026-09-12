//! Public video schema preserves strict parsing and explicit grid-point controls.

use serde_json::json;
use uniserve_server::openai::VideoGenerationRequest;
use uniserve_server::openai::utils::ResolvedRequestContext;
use uniserve_server::openai::videos::lower_video_generation_request;

#[test]
fn video_steps_are_optional_and_preserved_by_lowering() {
    for steps in [None, Some(50)] {
        let mut body = json!({"model": "h3", "prompt": "A calm sea"});
        if let Some(count) = steps {
            body["steps"] = json!(count);
        }
        let request: VideoGenerationRequest = serde_json::from_value(body).unwrap();
        let input =
            lower_video_generation_request(request, "h3", ResolvedRequestContext::default())
                .unwrap();
        assert_eq!(input.steps, steps);
        assert_eq!(input.seconds, 5.0);
    }
}

#[test]
fn video_steps_reject_out_of_bounds_and_non_integer_values() {
    for steps in [json!(-1), json!(2.5), json!("50")] {
        assert!(
            serde_json::from_value::<VideoGenerationRequest>(json!({
                "model": "h3", "prompt": "A calm sea", "steps": steps,
            }))
            .is_err()
        );
    }
    for steps in [0, 1, 1001] {
        let request = serde_json::from_value(json!({
            "model": "h3", "prompt": "A calm sea", "steps": steps,
        }))
        .unwrap();
        assert!(
            lower_video_generation_request(request, "h3", ResolvedRequestContext::default(),)
                .is_err()
        );
    }
}

#[test]
fn video_unknown_fields_remain_rejected() {
    assert!(
        serde_json::from_value::<VideoGenerationRequest>(json!({
            "model": "h3", "prompt": "A calm sea", "stepz": 50,
        }))
        .is_err()
    );
}
