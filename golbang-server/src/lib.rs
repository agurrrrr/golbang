pub mod error;
pub mod metrics;
pub mod routes;
pub mod sse;
pub mod types;
pub mod webui;

use axum::extract::Request;
use axum::middleware::{self, Next};
use axum::response::Response;
use golbang_core::{ModelCard, ReasoningFormat, SchedulerHandle};

use crate::error::ApiError;

#[derive(Clone)]
pub struct ChatRuntime {
    pub use_jinja: bool,
    pub template: Option<String>,
    pub bos_token: String,
    pub reasoning_format: ReasoningFormat,
    pub enable_thinking: bool,
    /// Default jinja `reasoning_effort` (Qwen: xhigh|medium|low, GLM: low|high|max).
    /// Request field overrides.
    pub reasoning_effort: Option<String>,
    /// Default think-token cap. 0 = unlimited. Request field overrides.
    pub reasoning_budget: u32,
    /// Server-side sampling defaults (llama-server parity). A request field
    /// overrides each one. These exist because the raw llama.cpp defaults
    /// (temperature 1.0, top_k 0, top_p 1.0) sample the full vocabulary and
    /// degrade instruction models into multilingual word salad.
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
}

impl Default for ChatRuntime {
    fn default() -> Self {
        Self {
            use_jinja: false,
            template: None,
            bos_token: String::new(),
            reasoning_format: ReasoningFormat::default(),
            enable_thinking: false,
            reasoning_effort: None,
            reasoning_budget: 0,
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
        }
    }
}

#[derive(Clone)]
pub struct AppState {
    pub scheduler: SchedulerHandle,
    pub model_name: String,
    pub default_timeout: Option<std::time::Duration>,
    pub chat: ChatRuntime,
    pub api_keys: Vec<String>,
    pub vision: bool,
    pub model_card: ModelCard,
    /// llama-server `return_progress` default (`--prompt-progress`).
    pub prompt_progress: bool,
    pub created: u64,
    /// Serve the embedded minimal UI at `GET /` (`--webui`).
    pub webui: bool,
}

pub fn router(state: AppState) -> axum::Router {
    let mut app = axum::Router::new()
        .route(
            "/v1/chat/completions",
            axum::routing::post(routes::chat_completions),
        )
        .route("/models", axum::routing::get(routes::list_models))
        .route("/v1/models", axum::routing::get(routes::list_models))
        .route("/metrics", axum::routing::get(metrics::metrics));
    if state.webui {
        app = app
            .route("/", axum::routing::get(webui::index))
            .route("/index.html", axum::routing::get(webui::index));
    }
    app.layer(middleware::from_fn_with_state(
        state.clone(),
        api_key_middleware,
    ))
    .with_state(state)
}

async fn api_key_middleware(
    axum::extract::State(state): axum::extract::State<AppState>,
    req: Request,
    next: Next,
) -> Result<Response, ApiError> {
    if state.api_keys.is_empty() {
        return Ok(next.run(req).await);
    }
    let path = req.uri().path();
    if path == "/metrics"
        || path == "/health"
        || path == "/v1/health"
        || path == "/models"
        || path == "/v1/models"
        || path == "/"
        || path == "/index.html"
    {
        return Ok(next.run(req).await);
    }
    if req.method() == axum::http::Method::OPTIONS {
        return Ok(next.run(req).await);
    }

    let headers = req.headers();
    let mut provided = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    if provided.is_empty() {
        provided = headers
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
    }
    const BEARER: &str = "Bearer ";
    if let Some(rest) = provided.strip_prefix(BEARER) {
        provided = rest.to_string();
    }
    if state.api_keys.iter().any(|k| k == &provided) {
        return Ok(next.run(req).await);
    }
    Err(ApiError::unauthorized("Invalid API Key"))
}
