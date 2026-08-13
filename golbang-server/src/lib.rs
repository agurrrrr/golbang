pub mod error;
pub mod metrics;
pub mod routes;
pub mod sse;
pub mod types;

use golbang_core::SchedulerHandle;

#[derive(Clone)]
pub struct AppState {
    pub scheduler: SchedulerHandle,
    pub model_name: String,
    pub default_timeout: Option<std::time::Duration>,
}

pub fn router(state: AppState) -> axum::Router {
    axum::Router::new()
        .route(
            "/v1/chat/completions",
            axum::routing::post(routes::chat_completions),
        )
        .route("/metrics", axum::routing::get(metrics::metrics))
        .with_state(state)
}
