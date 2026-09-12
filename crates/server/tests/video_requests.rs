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
fn empty_references_preserve_the_lowered_request() {
    let body = json!({"model": "h3", "prompt": "A calm sea", "seed": 7});
    let mut explicit_empty = body.clone();
    explicit_empty["references"] = json!([]);
    let lower = |body| {
        lower_video_generation_request(
            serde_json::from_value(body).unwrap(),
            "h3",
            ResolvedRequestContext::default(),
        )
        .unwrap()
    };
    assert_eq!(lower(body), lower(explicit_empty));
}

#[test]
fn reference_order_and_soundtrack_selection_survive_lowering() {
    let references = json!([
        {"type": "video", "task": "continue_scene", "role": "preceding",
         "source": {"type": "url", "value": "https://example.org/clip.mp4"},
         "include_audio": false},
        {"type": "image", "task": "reference", "role": "reference",
         "source": {"type": "base64", "value": "YWJj"}}
    ]);
    let request: VideoGenerationRequest = serde_json::from_value(json!({
        "model": "h3", "prompt": "A calm sea", "references": references,
    }))
    .unwrap();
    let expected = request.references.clone();
    let lowered =
        lower_video_generation_request(request, "h3", ResolvedRequestContext::default()).unwrap();
    assert_eq!(lowered.references, expected);
}

#[test]
fn invalid_reference_bundles_fail_before_media_admission() {
    let image = json!({"type": "image", "task": "reference", "role": "reference",
        "source": {"type": "base64", "value": "YWJj"}});
    let mut audio = image.clone();
    audio["type"] = json!("audio");
    let mut video = image.clone();
    video["type"] = json!("video");
    let mut invalid_encoding = image.clone();
    invalid_encoding["source"]["value"] = json!("not base64!");
    let mut invalid_role = image.clone();
    invalid_role["task"] = json!("continue_shot");
    let mut local_url = image.clone();
    local_url["source"] = json!({"type": "url", "value": "file:///etc/passwd"});
    for references in [
        json!([audio]),
        json!([video]),
        json!([invalid_encoding]),
        json!([invalid_role]),
        json!([local_url]),
        json!(vec![image; 10]),
    ] {
        let request = serde_json::from_value(json!({
            "model": "h3", "prompt": "A calm sea", "references": references,
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
