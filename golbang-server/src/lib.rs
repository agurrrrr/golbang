pub mod error;
pub mod metrics;
pub mod routes;
pub mod sse;
pub mod types;

use axum::extract::Request;
use axum::middleware::{self, Next};
use axum::response::Response;
use golbang_core::{ReasoningFormat, SchedulerHandle};

use crate::error::ApiError;

#[derive(Clone, Default)]
pub struct ChatRuntime {
    pub use_jinja: bool,
    pub template: Option<String>,
    pub bos_token: String,
    pub reasoning_format: ReasoningFormat,
    pub enable_thinking: bool,
    /// Default Qwen3.8 `reasoning_effort`. Request field overrides.
    pub reasoning_effort: Option<String>,
    /// Default think-token cap. 0 = unlimited. Request field overrides.
    pub reasoning_budget: u32,
}

#[derive(Clone)]
pub struct AppState {
    pub scheduler: SchedulerHandle,
    pub model_name: String,
    pub default_timeout: Option<std::time::Duration>,
    pub chat: ChatRuntime,
    pub api_keys: Vec<String>,
    pub vision: bool,
}

pub fn router(state: AppState) -> axum::Router {
    axum::Router::new()
        .route(
            "/v1/chat/completions",
            axum::routing::post(routes::chat_completions),
        )
        .route("/metrics", axum::routing::get(metrics::metrics))
        .layer(middleware::from_fn_with_state(
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
    if path == "/metrics" || path == "/health" || path == "/v1/health" {
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
