use axum::Json;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use crate::AppState;
use crate::error::ApiError;
use crate::sse;
use crate::types::{ChatCompletionRequest, validate_request};
use golbang_core::ModelCard;

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
        n_tools = req.tools.len(),
        tools_enabled = req.tools_enabled(),
        max_tokens = ?req.max_tokens,
        "chat.completions"
    );

    if req.stream {
        return Ok(sse::stream_completion(state, req)?);
    }

    let completion = sse::complete(state, req).await?;
    Ok(Json(completion).into_response())
}

/// llama-server `GET /models` and `GET /v1/models`.
pub async fn list_models(State(state): State<AppState>) -> Json<Value> {
    Json(models_payload(
        &state.model_name,
        state.created,
        &state.model_card,
    ))
}

pub fn models_payload(name: &str, created: u64, card: &ModelCard) -> Value {
    let capabilities = if card.vision {
        json!(["completion", "multimodal"])
    } else {
        json!(["completion"])
    };
    json!({
        "models": [{
            "name": name,
            "model": name,
            "modified_at": "",
            "size": "",
            "digest": "",
            "type": "model",
            "description": "",
            "tags": [""],
            "capabilities": capabilities,
            "parameters": "",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "",
                "families": [""],
                "parameter_size": "",
                "quantization_level": ""
            }
        }],
        "object": "list",
        "data": [{
            "id": name,
            "object": "model",
            "created": created,
            "owned_by": "golbang",
            "meta": {
                "vocab_type": card.vocab_type,
                "n_vocab": card.n_vocab,
                "n_ctx": card.n_ctx,
                "n_ctx_train": card.n_ctx_train,
                "n_embd": card.n_embd,
                "n_params": card.n_params,
                "size": card.size,
                "ftype": card.ftype,
            }
        }]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_payload_openai_list_shape() {
        let card = ModelCard {
            n_vocab: 151936,
            n_embd: 4096,
            n_ctx: 140000,
            n_ctx_train: 262144,
            n_params: 27_000_000_000,
            size: 16_000_000_000,
            vocab_type: 2,
            ftype: 15,
            vision: true,
        };
        let v = models_payload("qwen3.8-27b-q6", 1_700_000_000, &card);
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"][0]["id"], "qwen3.8-27b-q6");
        assert_eq!(v["data"][0]["object"], "model");
        assert_eq!(v["data"][0]["owned_by"], "golbang");
        assert_eq!(v["data"][0]["meta"]["n_vocab"], 151936);
        assert_eq!(v["data"][0]["meta"]["n_ctx"], 140000);
        assert_eq!(v["models"][0]["capabilities"][1], "multimodal");
    }
}
