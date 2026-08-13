use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::error::ApiError;
use crate::sse;
use crate::types::{validate_request, ChatCompletionRequest};
use crate::AppState;

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    if let Err(msg) = validate_request(&req) {
        return Err(ApiError::invalid_request(msg, Some("messages")));
    }

    tracing::info!(
        stream = req.stream,
        n_messages = req.messages.len(),
        max_tokens = ?req.max_tokens,
        "chat.completions"
    );

    if req.stream {
        return Ok(sse::stream_completion(state, req).into_response());
    }

    let model_name = sse::model_name(&req, &state);
    let completion = tokio::task::spawn_blocking(move || {
        let mut model = state.model.lock().unwrap_or_else(|e| e.into_inner());
        sse::complete_blocking(&mut model, req, &model_name)
    })
    .await
    .map_err(|e| ApiError::internal(format!("worker join: {e}")))??;

    Ok(Json(completion).into_response())
}
