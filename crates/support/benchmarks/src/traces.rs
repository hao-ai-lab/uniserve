use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceTask {
    Text,
    T2i,
    I2i,
    Default,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceRequest {
    pub id: String,
    pub task: TraceTask,
    pub prompt: String,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default)]
    pub steps: Option<u16>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub slo_ms: Option<u64>,
}

pub fn parse_jsonl(input: &str) -> anyhow::Result<Vec<TraceRequest>> {
    input
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(idx, line)| {
            serde_json::from_str::<TraceRequest>(line)
                .map_err(|error| anyhow::anyhow!("invalid trace row {}: {error}", idx + 1))
        })
        .collect()
}

pub fn validate_dimensions(request: &TraceRequest) -> anyhow::Result<()> {
    anyhow::ensure!(
        request.width.is_some() == request.height.is_some(),
        "trace request {} must provide width and height together",
        request.id
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_trace_jsonl_rows() {
        let rows = parse_jsonl(
            r#"{"id":"r1","task":"default","prompt":"travel","width":2048,"height":1152}"#,
        )
        .expect("trace rows");
        assert_eq!(rows[0].task, TraceTask::Default);
        validate_dimensions(&rows[0]).expect("dimensions");
    }
}
