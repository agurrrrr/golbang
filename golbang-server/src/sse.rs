use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::response::sse::{Event, KeepAlive, Sse};
use futures::Stream;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use golbang_core::{
    apply_chat_template, CancellationToken, GenerateParams, Job, SlotEvent,
};

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

fn prompt_from(req: &ChatCompletionRequest) -> String {
    let messages = core_messages(req.messages.clone());
    let applied = apply_chat_template(&messages);
    if !applied.used_chatml {
        tracing::warn!("serving request with raw prompt fallback");
    }
    applied.prompt
}

pub fn stream_completion(
    state: AppState,
    req: ChatCompletionRequest,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let id = completion_id();
    let created = unix_ts();
    let model_name = model_name(&req, &state);
    let params = generate_params(&req);
    let prompt = prompt_from(&req);

    let (ev_tx, mut ev_rx) = mpsc::unbounded_channel();
    let (sse_tx, sse_rx) = mpsc::channel::<Event>(16);
    let cancel = CancellationToken::new();
    let cancel_on_drop = cancel.clone();

    let mut job = Job::new(prompt, params, cancel, ev_tx);
    job.timeout = state.default_timeout;
    state.scheduler.try_submit(job)?;

    tokio::spawn(async move {
        if send_chunk(
            &sse_tx,
            &id,
            created,
            &model_name,
            Delta {
                role: Some("assistant"),
                content: None,
            },
            None,
        )
        .await
        .is_err()
        {
            cancel_on_drop.cancel();
            return;
        }

        let mut n = 0u32;
        while let Some(ev) = ev_rx.recv().await {
            let send = match ev {
                SlotEvent::Token(t) => {
                    n += 1;
                    if t.piece.is_empty() {
                        continue;
                    }
                    send_chunk(
                        &sse_tx,
                        &id,
                        created,
                        &model_name,
                        Delta {
                            role: None,
                            content: Some(t.piece),
                        },
                        None,
                    )
                    .await
                }
                SlotEvent::Finished { reason, .. } => {
                    let r = send_chunk(
                        &sse_tx,
                        &id,
                        created,
                        &model_name,
                        Delta::default(),
                        Some(reason.as_str().to_string()),
                    )
                    .await;
                    let _ = sse_tx.send(Event::default().data("[DONE]")).await;
                    tracing::info!(streamed = n, finish = reason.as_str(), "sse finished");
                    r
                }
                SlotEvent::Failed(e) => {
                    tracing::error!(error = %e, "stream generation failed");
                    let payload = serde_json::json!({
                        "error": { "message": e.to_string(), "type": "server_error" }
                    });
                    let _ = sse_tx.send(Event::default().data(payload.to_string())).await;
                    let _ = sse_tx.send(Event::default().data("[DONE]")).await;
                    break;
                }
            };
            if send.is_err() {
                cancel_on_drop.cancel();
                tracing::info!("client gone; cancel accepted, slot reclaim at next decode");
                break;
            }
        }
    });

    Ok(Sse::new(ReceiverStream::new(sse_rx).map(Ok)).keep_alive(KeepAlive::default()))
}

async fn send_chunk(
    tx: &mpsc::Sender<Event>,
    id: &str,
    created: u64,
    model: &str,
    delta: Delta,
    finish_reason: Option<String>,
) -> Result<(), ()> {
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
    let data = serde_json::to_string(&chunk).map_err(|_| ())?;
    tx.send(Event::default().data(data)).await.map_err(|_| ())
}

pub async fn complete(
    state: AppState,
    req: ChatCompletionRequest,
) -> Result<ChatCompletion, ApiError> {
    let id = completion_id();
    let created = unix_ts();
    let model_name = model_name(&req, &state);
    let params = generate_params(&req);
    let prompt = prompt_from(&req);

    let (ev_tx, mut ev_rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    let _guard = cancel.clone().drop_guard();

    let mut job = Job::new(prompt, params, cancel, ev_tx);
    job.timeout = state.default_timeout;
    state.scheduler.try_submit(job)?;

    let mut content = String::new();
    let mut finish = None;
    let mut prompt_tokens = 0u32;
    let mut completion_tokens = 0u32;

    while let Some(ev) = ev_rx.recv().await {
        match ev {
            SlotEvent::Token(t) => content.push_str(&t.piece),
            SlotEvent::Finished {
                reason,
                prompt_tokens: p,
                completion_tokens: c,
            } => {
                finish = Some(reason.as_str().to_string());
                prompt_tokens = p;
                completion_tokens = c;
                break;
            }
            SlotEvent::Failed(e) => return Err(e.into()),
        }
    }

    tracing::info!(
        prompt_tokens,
        completion_tokens,
        finish = ?finish,
        "json completion finished"
    );

    Ok(ChatCompletion {
        id,
        object: "chat.completion",
        created,
        model: model_name,
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
