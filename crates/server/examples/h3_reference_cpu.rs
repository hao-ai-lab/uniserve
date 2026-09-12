//! HTTP fixture for the CPU reference-input integration test.
//!
//! Uses the production router, serving admission, scheduler and worker transport.
//! The supplied worker executable owns synthetic numerical weights.

use uniserve_engine::WorkerConfig;
use uniserve_server::profile::ModelDescription;
use uniserve_server::{Config, build_router, build_state};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_ansi(false)
        .with_max_level(tracing::Level::INFO)
        .init();
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    anyhow::ensure!(
        args.len() == 3,
        "expected checkpoint root, worker executable, port"
    );
    let mut config = Config {
        model: args[0].clone(),
        model_description: ModelDescription::MiniMaxH3,
        served_model_name: Some("minimax-h3-ref".into()),
        // Numerical test recipe: a single forward, with the same admission capability.
        model_contract: Some(serde_json::json!({
            "denoise_steps": 1, "references": {"max": 1, "kinds": ["image"]}
        })),
        ..Config::default()
    };
    config.engine.max_batch = 1;
    config.engine.max_num_seqs = 1;
    config.engine.max_num_batched_tokens = 1024;
    config.engine.max_model_len = Some(64);
    config.engine.max_video_seconds = 1.5;
    config.engine.workers = vec![WorkerConfig::h3("cpu", 1, 6)];
    config.engine.worker_process.python = args[1].clone().into();
    config.engine.worker_process.pipeline_depth = 6;
    config.engine.worker_process.attention_backend =
        uniserve_worker_ipc::AttentionBackend::TorchSdpa;
    config.engine.worker_process.prefill_cuda_graph = false;
    let state = build_state(&config).await?;
    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", args[2])).await?;
    axum::serve(listener, build_router(state)).await?;
    Ok(())
}
