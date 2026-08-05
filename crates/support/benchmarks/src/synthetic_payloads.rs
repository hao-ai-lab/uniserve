use serde_json::{Value, json};

pub fn text_prompt(prompt: &str, model: &str) -> Value {
    json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": true
    })
}
