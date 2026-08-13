pub mod error;
pub mod routes;
pub mod sse;
pub mod types;

use std::sync::{Arc, Mutex};

use axum::Router;
use golbang_core::Model;

#[derive(Clone)]
pub struct AppState {
    pub model: Arc<Mutex<Model>>,
    pub model_name: String,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route(
            "/v1/chat/completions",
            axum::routing::post(routes::chat_completions),
        )
        .with_state(state)
}
