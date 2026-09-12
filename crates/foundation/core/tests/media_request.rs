//! Decoded reference ownership and geometry at the public engine boundary.

use uniserve_core::{
    DiffusionRequest, DiffusionRequestError, ImageReference, MediaGeometry, RequestId,
};

#[test]
fn image_reference_requires_a_complete_rgb_raster() {
    let mut request = DiffusionRequest {
        request_id: RequestId(1),
        prompt_token_ids: vec![7],
        seed: 0,
        priority: 0,
        geometry: MediaGeometry {
            frame_count: 124,
            video_units: 1,
            prompt_tokens: 1,
            denoise_steps: 50,
        },
        image_reference: Some(ImageReference {
            width: 2,
            height: 2,
            pixels: (0..12).collect(),
        }),
    };
    assert_eq!(request.validate(), Ok(()));
    request.image_reference.as_mut().unwrap().pixels.pop();
    assert_eq!(
        request.validate(),
        Err(DiffusionRequestError::InvalidImageReference)
    );
    request.image_reference.as_mut().unwrap().width = 4097;
    assert_eq!(
        request.validate(),
        Err(DiffusionRequestError::InvalidImageReference)
    );
    request.image_reference = None;
    assert_eq!(request.validate(), Ok(()));
}
