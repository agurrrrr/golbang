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
        return Ok(sse::stream_completion(state, req)?.into_response());
    }

    let completion = sse::complete(state, req).await?;
    Ok(Json(completion).into_response())
}
