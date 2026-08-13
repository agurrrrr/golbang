use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::response::sse::{Event, KeepAlive, Sse};
use futures::Stream;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use golbang_core::{apply_chat_template, GenerateParams, Model};

use crate::error::ApiError;
use crate::types::{
    ChatCompletion, ChatCompletionChunk, ChatCompletionRequest, ChatMessage, Choice, ChunkChoice,
    Delta, Usage,
};
use crate::AppState;

pub fn completion_id() -> String {
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("chatcmpl-{n:x}")
}

pub fn unix_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn generate_params(req: &ChatCompletionRequest) -> GenerateParams {
    GenerateParams {
        max_tokens: req.max_tokens.unwrap_or(256).max(1),
        temperature: req.temperature.unwrap_or(1.0).max(0.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k.unwrap_or(0),
        seed: req.seed.unwrap_or(0),
        stop: req.stop.clone().map(|s| s.into_vec()).unwrap_or_default(),
    }
}

pub fn model_name(req: &ChatCompletionRequest, state: &AppState) -> String {
    req.model
        .clone()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| state.model_name.clone())
}

fn core_messages(messages: Vec<ChatMessage>) -> Vec<golbang_core::ChatMessage> {
    messages.into_iter().map(Into::into).collect()
}

pub fn stream_completion(
    state: AppState,
    req: ChatCompletionRequest,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel::<Event>(16);
    tokio::task::spawn_blocking(move || {
        if let Err(e) = run_stream(&state, req, &tx) {
            tracing::error!(error = %e.message, "stream generation failed");
            let payload = serde_json::json!({
                "error": { "message": e.message, "type": e.kind }
            });
            let _ = tx.blocking_send(Event::default().data(payload.to_string()));
            let _ = tx.blocking_send(Event::default().data("[DONE]"));
        }
    });
    Sse::new(ReceiverStream::new(rx).map(Ok)).keep_alive(KeepAlive::default())
}

fn run_stream(
    state: &AppState,
    req: ChatCompletionRequest,
    tx: &mpsc::Sender<Event>,
) -> Result<(), ApiError> {
    let id = completion_id();
    let created = unix_ts();
    let model_name = model_name(&req, state);
    let params = generate_params(&req);
    let messages = core_messages(req.messages);
    let applied = apply_chat_template(&messages);
    if !applied.used_chatml {
        tracing::warn!("serving request with raw prompt fallback");
    }

    let mut model = state.model.lock().unwrap_or_else(|e| e.into_inner());
    let mut generation = model.generate(&applied.prompt, params)?;

    send_chunk(
        tx,
        &id,
        created,
        &model_name,
        Delta {
            role: Some("assistant"),
            content: None,
        },
        None,
    )?;

    let mut n = 0u32;
    while let Some(item) = generation.next() {
        let t = item?;
        n += 1;
        if !t.piece.is_empty() {
            send_chunk(
                tx,
                &id,
                created,
                &model_name,
                Delta {
                    role: None,
                    content: Some(t.piece),
                },
                None,
            )?;
        }
    }

    let finish = generation
        .finish_reason()
        .map(|r| r.as_str().to_string());
    send_chunk(
        tx,
        &id,
        created,
        &model_name,
        Delta::default(),
        finish,
    )?;
    tx.blocking_send(Event::default().data("[DONE]"))
        .map_err(|_| ApiError::internal("client gone"))?;

    tracing::info!(
        prompt_tokens = generation.prompt_tokens(),
        completion_tokens = generation.completion_tokens(),
        streamed = n,
        finish = ?generation.finish_reason(),
        "sse finished"
    );
    Ok(())
}

fn send_chunk(
    tx: &mpsc::Sender<Event>,
    id: &str,
    created: u64,
    model: &str,
    delta: Delta,
    finish_reason: Option<String>,
) -> Result<(), ApiError> {
    let chunk = ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta,
            finish_reason,
        }],
    };
    let data = serde_json::to_string(&chunk).map_err(|e| ApiError::internal(e.to_string()))?;
    tx.blocking_send(Event::default().data(data))
        .map_err(|_| ApiError::internal("client gone"))?;
    Ok(())
}

pub fn complete_blocking(
    model: &mut Model,
    req: ChatCompletionRequest,
    model_name: &str,
) -> Result<ChatCompletion, ApiError> {
    let id = completion_id();
    let created = unix_ts();
    let params = generate_params(&req);
    let messages = core_messages(req.messages);
    let applied = apply_chat_template(&messages);
    if !applied.used_chatml {
        tracing::warn!("serving request with raw prompt fallback");
    }

    let mut generation = model.generate(&applied.prompt, params)?;
    let mut content = String::new();
    for item in &mut generation {
        content.push_str(&item?.piece);
    }

    let prompt_tokens = generation.prompt_tokens();
    let completion_tokens = generation.completion_tokens();
    let finish = generation.finish_reason().map(|r| r.as_str().to_string());

    tracing::info!(
        prompt_tokens,
        completion_tokens,
        finish = ?generation.finish_reason(),
        "json completion finished"
    );

    Ok(ChatCompletion {
        id,
        object: "chat.completion",
        created,
        model: model_name.to_string(),
        choices: vec![Choice {
            index: 0,
            message: crate::types::AssistantMessage {
                role: "assistant",
                content,
            },
            finish_reason: finish,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
}
