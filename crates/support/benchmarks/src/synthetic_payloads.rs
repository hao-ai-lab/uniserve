use serde_json::{Value, json};

pub fn text_prompt(prompt: &str, model: &str) -> Value {
    json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": true
    })
}

pub fn native_interleave_prompt(prompt: &str) -> Value {
    json!({
        "prompt": prompt,
        "mode": "interleave",
        "image": {
            "max_images": 1
        }
    })
}

pub fn native_image_prompt(prompt: &str, width: Option<u32>, height: Option<u32>) -> Value {
    let mut image = serde_json::Map::new();
    if let (Some(width), Some(height)) = (width, height) {
        image.insert("width".into(), json!(width));
        image.insert("height".into(), json!(height));
    }
    json!({
        "prompt": prompt,
        "mode": "image",
        "image": image
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_payload_keeps_interleave_explicit() {
        let payload = native_interleave_prompt("travel");
        assert_eq!(payload["mode"], "interleave");
    }
}
