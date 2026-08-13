use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use clap::Parser;
use golbang_core::{LoadParams, Model};
use golbang_server::{router, AppState};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(
    name = "golbang-server",
    about = "golbang OpenAI-compatible server (P1, single request)"
)]
struct Args {
    /// GGUF path. Also reads GOLBANG_MODEL. Falls back to GOLBANG_TEST_MODEL.
    #[arg(long, env = "GOLBANG_MODEL")]
    model: Option<PathBuf>,

    #[arg(long, env = "GOLBANG_HOST", default_value = "127.0.0.1")]
    host: String,

    #[arg(long, env = "GOLBANG_PORT", default_value_t = 8088)]
    port: u16,

    /// Keep small when another process already holds most of VRAM.
    #[arg(long, env = "GOLBANG_N_CTX", default_value_t = 256)]
    n_ctx: u32,

    #[arg(long, env = "GOLBANG_N_GPU_LAYERS", default_value_t = 99)]
    n_gpu_layers: i32,
}

fn resolve_model(args: &Args) -> Result<PathBuf> {
    if let Some(p) = &args.model {
        return Ok(p.clone());
    }
    if let Ok(p) = std::env::var("GOLBANG_TEST_MODEL") {
        if !p.is_empty() {
            return Ok(PathBuf::from(p));
        }
    }
    anyhow::bail!("--model / GOLBANG_MODEL / GOLBANG_TEST_MODEL is required");
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let model_path = resolve_model(&args)?;
    let model_name = model_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("golbang")
        .to_string();

    let model = Model::load(
        &model_path,
        LoadParams {
            n_gpu_layers: args.n_gpu_layers,
            n_ctx: args.n_ctx,
        },
    )
    .with_context(|| format!("load {}", model_path.display()))?;

    let state = AppState {
        model: Arc::new(Mutex::new(model)),
        model_name,
    };

    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .context("host:port")?;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    tracing::info!(%addr, "listening");

    axum::serve(listener, router(state))
        .await
        .context("serve")?;
    Ok(())
}
