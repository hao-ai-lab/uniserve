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
fn single_image_survives_lowering() {
    let references = json!([
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
    let mut missing_host = image.clone();
    missing_host["source"] = json!({"type": "url", "value": "https://"});
    for references in [
        json!([missing_host]),
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

fn image_reference(format: image::ImageFormat, width: u32, height: u32) -> uniserve_server::serving::references::VideoReference {
    use base64::Engine as _;
    let image = image::RgbImage::from_pixel(width, height, image::Rgb([7, 13, 29]));
    let mut bytes = std::io::Cursor::new(Vec::new());
    image.write_to(&mut bytes, format).unwrap();
    serde_json::from_value(json!({
        "type": "image", "task": "reference", "role": "reference",
        "source": {"type": "base64", "value": base64::engine::general_purpose::STANDARD.encode(bytes.into_inner())}
    })).unwrap()
}

#[test]
fn reference_admission_is_capability_gated_and_decode_bounded() {
    use uniserve_server::serving::references::admit_references;
    for format in [image::ImageFormat::Png, image::ImageFormat::Jpeg] {
        let reference = image_reference(format, 64, 32);
        assert!(admit_references(&[reference.clone()], false).unwrap_err().contains("model contract"));
        let image = admit_references(&[reference.clone()], true).unwrap().unwrap();
        assert_eq!((image.width, image.height, image.pixels.len()), (64, 32, 64 * 32 * 3));
        if format == image::ImageFormat::Png {
            assert_eq!(image.pixels, [7, 13, 29].repeat(64 * 32));
        }
        assert!(admit_references(&[reference.clone(), reference], true).unwrap_err().contains("at most 1"));
    }
    for (width, height) in [(31, 32), (32, 33), (4128, 32)] {
        assert!(admit_references(&[image_reference(image::ImageFormat::Png, width, height)], true)
            .unwrap_err().contains("dimensions"));
    }
    assert!(admit_references(&[image_reference(image::ImageFormat::Bmp, 32, 32)], true)
        .unwrap_err().contains("PNG or JPEG"));
    assert_eq!(admit_references(&[], false).unwrap(), None);
    assert_eq!(admit_references(&[], true).unwrap(), None);
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
