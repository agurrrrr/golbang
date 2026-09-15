use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::serve::ListenerExt;
use clap::Parser;
use golbang_core::{
    Engine, FifoPolicy, IterationBudget, LoadParams, Model, ReasoningFormat, SchedulerConfig,
    SpecParams, SpecType, parse_ggml_type, parse_rpc_servers, parse_tensor_overrides,
    parse_tensor_split, spawn_scheduler,
};
use golbang_server::{AppState, ChatRuntime, router};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(
    name = "golbang-server",
    about = "golbang OpenAI-compatible server (P2 concurrency core)"
)]
struct Args {
    /// GGUF path. Also reads GOLBANG_MODEL. Falls back to GOLBANG_TEST_MODEL.
    #[arg(long, env = "GOLBANG_MODEL")]
    model: Option<PathBuf>,

    /// Separate MTP/draft GGUF (llama-server `-md`).
    #[arg(long, env = "GOLBANG_MODEL_DRAFT")]
    model_draft: Option<PathBuf>,

    #[arg(long, env = "GOLBANG_HOST", default_value = "127.0.0.1")]
    host: String,

    #[arg(long, env = "GOLBANG_PORT", default_value_t = 8088)]
    port: u16,

    /// Per-sequence fair share. Total KV is n_ctx * n_parallel.
    #[arg(long, env = "GOLBANG_N_CTX", default_value_t = 256)]
    n_ctx: u32,

    /// Solo-slot KV cap. 0 = full pool (`n_ctx * n_parallel`). When a second
    /// slot joins, caps become min(this, max(used, pool / n_active)).
    /// llama only honours a cap above `n_ctx` when `--kv-unified` is set.
    #[arg(long, env = "GOLBANG_SINGLE_MAX_CTX", default_value_t = 0)]
    single_max_ctx: u32,

    /// Share one KV stream across sequences (llama-server `--kv-unified`).
    /// Lets a solo slot grow to the full pool when `--n-parallel > 1`.
    #[arg(long, env = "GOLBANG_KV_UNIFIED", default_value_t = false)]
    kv_unified: bool,

    /// KV cache element type for K (llama-server `--cache-type-k`).
    /// q8_0 halves VRAM; requires flash attention. Default f16.
    #[arg(long, env = "GOLBANG_KV_TYPE_K", default_value = "f16")]
    kv_type_k: String,

    /// KV cache element type for V (llama-server `--cache-type-v`).
    /// q8_0 halves VRAM; requires flash attention. Default f16.
    #[arg(long, env = "GOLBANG_KV_TYPE_V", default_value = "f16")]
    kv_type_v: String,

    #[arg(long, env = "GOLBANG_N_GPU_LAYERS", default_value_t = 99)]
    n_gpu_layers: i32,

    /// Keep MoE expert tensors of the first N layers on CPU (`-ncmoe`).
    #[arg(long, env = "GOLBANG_N_CPU_MOE", default_value_t = 0)]
    n_cpu_moe: u32,

    /// Flash attention: auto | on | off
    #[arg(long, env = "GOLBANG_FLASH_ATTN", default_value = "auto")]
    flash_attn: String,

    /// Logical llama n_batch. 0 = n_ctx.
    #[arg(long, env = "GOLBANG_N_BATCH", default_value_t = 0)]
    n_batch: u32,

    /// Physical llama n_ubatch. 0 = n_batch.
    #[arg(long, env = "GOLBANG_N_UBATCH", default_value_t = 0)]
    n_ubatch: u32,

    /// llama decode threads. 0 = library default.
    #[arg(long, env = "GOLBANG_N_THREADS", default_value_t = 0)]
    n_threads: i32,

    /// DSV4 suffix rollback snapshots. 1 lets bind drop the last `<think>` token
    /// after restoring a prefill checkpoint. 0 = llama.cpp default (no rollback).
    #[arg(long, env = "GOLBANG_N_RS_SEQ", default_value_t = 1)]
    n_rs_seq: u32,

    /// Disable mmap (`llama_model_params.load_mode = NONE`). llama-server `--no-mmap`.
    #[arg(long, env = "GOLBANG_NO_MMAP", default_value_t = false)]
    no_mmap: bool,

    /// Reported model id (llama-server `-a`). Default is the GGUF file name.
    #[arg(long, env = "GOLBANG_ALIAS")]
    alias: Option<String>,

    /// Override GGUF `tokenizer.chat_template` (llama-server `--chat-template-file`).
    #[arg(long, env = "GOLBANG_CHAT_TEMPLATE_FILE")]
    chat_template_file: Option<PathBuf>,

    /// Default jinja `reasoning_effort`. Qwen3.8: xhigh | medium | low.
    /// GLM-5.3: low | high | max (template default is max; `xhigh` is a
    /// Qwen-only level and is rejected here).
    #[arg(long, env = "GOLBANG_REASONING_EFFORT")]
    reasoning_effort: Option<String>,

    /// Force `</think>` after this many think tokens (0 = unlimited).
    /// Stops Qwen xhigh from spending the whole `max_tokens` on thinking.
    #[arg(long, env = "GOLBANG_REASONING_BUDGET", default_value_t = 0)]
    reasoning_budget: u32,

    /// Slot count / llama n_seq_max.
    #[arg(long, env = "GOLBANG_N_PARALLEL", default_value_t = 2)]
    n_parallel: u32,

    /// Bounded submit queue. Full → immediate 503 + Retry-After.
    #[arg(long, env = "GOLBANG_QUEUE_SIZE", default_value_t = 2)]
    queue_size: usize,

    /// Optional per-request generation timeout.
    #[arg(long, env = "GOLBANG_TIMEOUT_SECS")]
    timeout_secs: Option<u64>,

    /// Schedule policy. Only `fifo` is built in (P2).
    #[arg(long, env = "GOLBANG_POLICY", default_value = "fifo")]
    policy: String,

    /// Apply GGUF `tokenizer.chat_template` with minijinja (llama-server `--jinja`).
    #[arg(long, env = "GOLBANG_JINJA", default_value_t = false)]
    jinja: bool,

    /// none | deepseek | deepseek-legacy | auto. deepseek/auto extract `<think>`.
    #[arg(long, env = "GOLBANG_REASONING_FORMAT", default_value = "none")]
    reasoning_format: String,

    /// Require `Authorization: Bearer …` or `X-Api-Key`. Repeat or comma-separate.
    #[arg(long, env = "GOLBANG_API_KEY")]
    api_key: Vec<String>,

    /// CLIP / projector GGUF (llama-server `--mmproj`). Enables image_url parts.
    #[arg(long, env = "GOLBANG_MMPROJ")]
    mmproj: Option<PathBuf>,

    /// Speculative decoding, comma-separated: `draft-mtp`, `ngram-mod`.
    #[arg(long, env = "GOLBANG_SPEC_TYPE", default_value = "")]
    spec_type: String,

    /// Max MTP draft tokens (llama-server `--spec-draft-n-max`).
    #[arg(long, env = "GOLBANG_SPEC_DRAFT_N_MAX", default_value_t = 3)]
    spec_draft_n_max: i32,

    /// ngram-mod max draft tokens (llama-server ngram_mod.n_max).
    #[arg(long, env = "GOLBANG_SPEC_NGRAM_N_MAX", default_value_t = 64)]
    spec_ngram_n_max: i32,

    /// ngram-mod min draft tokens. llama-server default is 48; 1 fills
    /// MTP-fail steps on a cold table (P7).
    #[arg(long, env = "GOLBANG_SPEC_NGRAM_N_MIN", default_value_t = 1)]
    spec_ngram_n_min: i32,

    /// Min MTP draft probability (llama-server `--spec-draft-p-min`).
    #[arg(long, env = "GOLBANG_SPEC_DRAFT_P_MIN", default_value_t = 0.90)]
    spec_draft_p_min: f32,

    /// MTP KV cache type (`q8_0` matches the production Qwen unit).
    #[arg(long, env = "GOLBANG_SPEC_DRAFT_TYPE_K", default_value = "q8_0")]
    spec_draft_type_k: String,

    #[arg(long, env = "GOLBANG_SPEC_DRAFT_TYPE_V", default_value = "q8_0")]
    spec_draft_type_v: String,

    /// Load MTP tensors even without `--spec-type draft-mtp`.
    #[arg(long, env = "GOLBANG_LOAD_MTP", default_value_t = false)]
    load_mtp: bool,

    /// llama-server `--rpc`: comma-separated `host:port` of `ggml-rpc-server`.
    /// Requires the linked llama.cpp to have the RPC backend (`GGML_RPC=ON`)
    /// or a loadable `libggml-rpc.so` (`--rpc-backend`).
    #[arg(long, env = "GOLBANG_RPC")]
    rpc: Option<String>,

    /// Path to `libggml-rpc.so` when the pinned HIP/CUDA tree was built without RPC.
    #[arg(long, env = "GOLBANG_RPC_BACKEND")]
    rpc_backend: Option<PathBuf>,

    /// llama-server `--tensor-split`, e.g. `32,16` for MI50+V100.
    /// Empty = split by free VRAM.
    #[arg(long, env = "GOLBANG_TENSOR_SPLIT")]
    tensor_split: Option<String>,

    /// llama-server `-ot/--override-tensor`, e.g. `token_embd=ROCm0` or
    /// `\.ffn_(up|down)_exps=CPU`. Comma-separated `pattern=buft` pairs; `buft`
    /// is a backend buffer-type name (`CPU`, `ROCm0`, `CUDA0`, ...). Evaluated
    /// before `--n-cpu-moe`; the first regex match wins.
    #[arg(long, env = "GOLBANG_OVERRIDE_TENSOR")]
    override_tensor: Option<String>,

    /// llama-server `return_progress`. SSE `data:` chunks include
    /// `prompt_progress` (`total`/`cache`/`processed`/`time_ms`) so a long
    /// prefill keeps HTTP idle timers from firing. Default on. Per-request
    /// `return_progress` overrides. `--prompt-progress false` disables.
    #[arg(
        long,
        env = "GOLBANG_PROMPT_PROGRESS",
        default_value_t = true,
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        default_missing_value = "true"
    )]
    prompt_progress: bool,
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

fn parse_flash_attn(s: &str) -> Result<i32> {
    // llama.h: AUTO=-1, DISABLED=0, ENABLED=1
    match s.trim().to_ascii_lowercase().as_str() {
        "auto" | "-1" => Ok(-1),
        "on" | "1" | "true" | "enabled" => Ok(1),
        "off" | "0" | "false" | "disabled" => Ok(0),
        other => anyhow::bail!("--flash-attn must be auto|on|off, got {other}"),
    }
}

fn normalize_reasoning_effort(s: Option<&str>) -> Option<String> {
    match s.map(str::trim).filter(|s| !s.is_empty()) {
        None => None,
        Some(v) => match v.to_ascii_lowercase().as_str() {
            // Qwen3.8 levels. The template aliases high → xhigh itself, so
            // pass through unchanged (deploy/golbang-qwen38 uses `high`).
            "xhigh" | "high" | "medium" | "low" => Some(v.to_ascii_lowercase()),
            // GLM-5.3 only.
            "max" => Some("max".into()),
            other => {
                tracing::warn!(other, "unknown --reasoning-effort; using template default");
                None
            }
        },
    }
}

fn load_chat_template(args: &Args, model: &Model) -> Result<Option<String>> {
    if let Some(path) = &args.chat_template_file {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read --chat-template-file {}", path.display()))?;
        if text.trim().is_empty() {
            anyhow::bail!("--chat-template-file {} is empty", path.display());
        }
        return Ok(Some(text));
    }
    Ok(model.chat_template())
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
    let model_name = args
        .alias
        .clone()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| {
            model_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("golbang")
                .to_string()
        });

    let n_parallel = args.n_parallel.max(1);
    let flash_attn = parse_flash_attn(&args.flash_attn)?;
    let spec_types = SpecType::parse_list(&args.spec_type).map_err(anyhow::Error::msg)?;
    let spec = SpecParams {
        types: spec_types,
        n_max: args.spec_draft_n_max.max(0),
        p_min: args.spec_draft_p_min.clamp(0.0, 1.0),
        ngram_n_max: args.spec_ngram_n_max.max(0),
        ngram_n_min: args.spec_ngram_n_min.max(0),
        cache_type_k: parse_ggml_type(&args.spec_draft_type_k).map_err(anyhow::Error::msg)?,
        cache_type_v: parse_ggml_type(&args.spec_draft_type_v).map_err(anyhow::Error::msg)?,
    };
    let load_mtp = args.load_mtp || spec.wants_mtp();
    let rpc_servers = match args.rpc.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        Some(s) => parse_rpc_servers(s).map_err(anyhow::Error::msg)?,
        None => Vec::new(),
    };
    let tensor_split = match args
        .tensor_split
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        Some(s) => parse_tensor_split(s).map_err(anyhow::Error::msg)?,
        None => Vec::new(),
    };
    let tensor_overrides = match args
        .override_tensor
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        Some(s) => parse_tensor_overrides(s).map_err(anyhow::Error::msg)?,
        None => Vec::new(),
    };
    if !tensor_overrides.is_empty() {
        tracing::info!(?tensor_overrides, "tensor buffer-type overrides");
    }
    if !rpc_servers.is_empty() {
        tracing::info!(
            servers = ?rpc_servers,
            rpc_backend = args.rpc_backend.as_ref().map(|p| p.display().to_string()),
            tensor_split = ?tensor_split,
            "RPC offload enabled"
        );
    }
    // Captured before `spec` is moved into Model::load: the scheduler reserves
    // (verify_n_max + 1) * n_active draft cells from the KV pool (#8565).
    let spec_n_max = spec.verify_n_max().max(0) as u32;
    if spec.enabled() {
        tracing::info!(
            types = ?spec.types.iter().map(|t| t.as_str()).collect::<Vec<_>>(),
            n_max = spec.n_max,
            ngram_n_max = spec.ngram_n_max,
            ngram_n_min = spec.ngram_n_min,
            verify_n_max = spec.verify_n_max(),
            p_min = spec.p_min,
            "speculative decoding enabled"
        );
    }
    let model = Model::load(
        &model_path,
        LoadParams {
            n_gpu_layers: args.n_gpu_layers,
            n_ctx: args.n_ctx,
            n_seq_max: n_parallel,
            n_cpu_moe: args.n_cpu_moe,
            tensor_overrides,
            flash_attn,
            n_batch: args.n_batch,
            n_ubatch: args.n_ubatch,
            n_threads: args.n_threads,
            n_rs_seq: args.n_rs_seq,
            use_mmap: !args.no_mmap,
            load_mtp,
            spec,
            mmproj: args.mmproj.clone(),
            kv_unified: args.kv_unified,
            cache_type_k: parse_ggml_type(&args.kv_type_k).map_err(anyhow::Error::msg)?,
            cache_type_v: parse_ggml_type(&args.kv_type_v).map_err(anyhow::Error::msg)?,
            model_draft: args.model_draft.clone(),
            rpc_servers,
            rpc_backend: args.rpc_backend.clone(),
            tensor_split,
        },
    )
    .with_context(|| format!("load {}", model_path.display()))?;

    let policy: Box<dyn golbang_core::SchedulePolicy> = match args.policy.as_str() {
        "fifo" => Box::new(FifoPolicy::with_budget(IterationBudget::for_context(
            model.n_batch() as usize,
            model.n_ubatch() as usize,
            n_parallel as usize,
        ))),
        other => anyhow::bail!("unknown policy {other} (P2 ships fifo only)"),
    };

    let reasoning_format =
        ReasoningFormat::parse(&args.reasoning_format).map_err(anyhow::Error::msg)?;
    let enable_thinking = reasoning_format.extracts();
    let reasoning_effort = normalize_reasoning_effort(args.reasoning_effort.as_deref());
    let chat_template = if args.jinja {
        load_chat_template(&args, &model)?
    } else {
        None
    };
    let bos_token = if args.jinja {
        model.bos_token_str()
    } else {
        String::new()
    };
    if args.jinja {
        match chat_template.as_deref() {
            Some(t) => tracing::info!(
                bytes = t.len(),
                bos = %bos_token,
                thinking = enable_thinking,
                reasoning = reasoning_format.as_str(),
                reasoning_effort = reasoning_effort.as_deref().unwrap_or("template-default"),
                reasoning_budget = args.reasoning_budget,
                from_file = args.chat_template_file.is_some(),
                "jinja chat template loaded"
            ),
            None => {
                tracing::warn!("--jinja set but no chat template (GGUF or --chat-template-file)")
            }
        }
    }

    let api_keys: Vec<String> = args
        .api_key
        .iter()
        .flat_map(|s| s.split(','))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if !api_keys.is_empty() {
        tracing::info!(n = api_keys.len(), "api key auth enabled");
    }

    let vision = model.vision_enabled();
    let model_card = model.card().clone();
    let engine = Arc::new(Engine::new(model));
    let spawned = spawn_scheduler(
        engine,
        policy,
        SchedulerConfig {
            n_parallel,
            queue_capacity: args.queue_size.max(1),
            default_timeout: args.timeout_secs.map(Duration::from_secs),
            single_max_ctx: args.single_max_ctx,
            spec_n_max,
        },
    );

    let state = AppState {
        scheduler: spawned.handle,
        model_name,
        default_timeout: args.timeout_secs.map(Duration::from_secs),
        chat: ChatRuntime {
            use_jinja: args.jinja,
            template: chat_template,
            bos_token,
            reasoning_format,
            enable_thinking,
            reasoning_effort,
            reasoning_budget: args.reasoning_budget,
        },
        api_keys,
        vision,
        model_card,
        prompt_progress: args.prompt_progress,
        created: golbang_server::sse::unix_ts(),
    };

    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .context("host:port")?;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind {addr}"))?
        .tap_io(|stream| {
            if let Err(err) = stream.set_nodelay(true) {
                tracing::debug!(error = %err, "TCP_NODELAY failed");
            }
        });
    tracing::info!(
        %addr,
        n_parallel,
        queue = args.queue_size,
        n_ctx = args.n_ctx,
        single_max_ctx = args.single_max_ctx,
        kv_unified = args.kv_unified,
        "listening"
    );

    axum::serve(listener, router(state))
        .await
        .context("serve")?;
    Ok(())
}
