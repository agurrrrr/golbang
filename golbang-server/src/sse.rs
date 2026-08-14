use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::response::sse::{Event, KeepAlive, Sse};
use futures::Stream;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use golbang_core::{
    CancellationToken, ChatApplyOpts, GenerateParams, Job, ReasoningParser, SlotEvent,
    ToolCallParser, apply_chat_template_with,
};

use crate::AppState;
use crate::error::ApiError;
use crate::types::{
    ChatCompletion, ChatCompletionChunk, ChatCompletionRequest, ChatMessage, Choice, ChunkChoice,
    Delta, DeltaFunction, DeltaToolCall, OutgoingToolCall, Timings, Usage,
};

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

pub fn generate_params(req: &ChatCompletionRequest, state: &AppState) -> GenerateParams {
    let mut stop = req.stop.clone().map(|s| s.into_vec()).unwrap_or_default();
    if state.chat.use_jinja {
        for extra in ["<｜User｜>", "<｜end▁of▁sentence｜>"] {
            if !stop.iter().any(|s| s == extra) {
                stop.push(extra.to_string());
            }
        }
    }
    GenerateParams {
        max_tokens: req.max_tokens.unwrap_or(256).max(1),
        temperature: req.temperature.unwrap_or(1.0).max(0.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k.unwrap_or(0),
        seed: req.seed.unwrap_or(0),
        stop,
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

fn prompt_from(req: &ChatCompletionRequest, state: &AppState) -> String {
    let messages = core_messages(req.messages.clone());
    let tools = if req.tools_enabled() {
        req.tools.clone()
    } else {
        Vec::new()
    };
    let applied = apply_chat_template_with(
        &messages,
        &ChatApplyOpts {
            jinja: state.chat.use_jinja,
            template: state.chat.template.clone(),
            bos_token: state.chat.bos_token.clone(),
            enable_thinking: state.chat.enable_thinking,
            tools,
        },
    );
    if applied.used_jinja {
        tracing::debug!(
            thinking = state.chat.enable_thinking,
            n_tools = req.tools.len(),
            tools_enabled = req.tools_enabled(),
            prompt_chars = applied.prompt.len(),
            "applied GGUF jinja chat template"
        );
    } else if !applied.used_chatml {
        tracing::warn!("serving request with raw prompt fallback");
    }
    applied.prompt
}

fn outgoing_tool_calls(calls: Vec<golbang_core::ToolCall>) -> Vec<OutgoingToolCall> {
    calls.into_iter().map(OutgoingToolCall::from).collect()
}

fn tool_call_deltas(calls: &[OutgoingToolCall]) -> Vec<DeltaToolCall> {
    calls
        .iter()
        .enumerate()
        .map(|(i, tc)| DeltaToolCall {
            index: i as u32,
            id: Some(tc.id.clone()),
            type_: Some("function"),
            function: DeltaFunction {
                name: Some(tc.function.name.clone()),
                arguments: Some(tc.function.arguments.clone()),
            },
        })
        .collect()
}

fn finish_reason_for(reason: &str, has_tools: bool) -> String {
    if has_tools && reason != "cancelled" && reason != "timeout" {
        "tool_calls".into()
    } else {
        reason.to_string()
    }
}

pub fn stream_completion(
    state: AppState,
    req: ChatCompletionRequest,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let id = completion_id();
    let created = unix_ts();
    let model_name = model_name(&req, &state);
    let params = generate_params(&req, &state);
    let prompt = prompt_from(&req, &state);

    let (ev_tx, mut ev_rx) = mpsc::unbounded_channel();
    let (sse_tx, sse_rx) = mpsc::channel::<Event>(16);
    let cancel = CancellationToken::new();
    let cancel_on_drop = cancel.clone();
    let mut parser = ReasoningParser::from_prompt(state.chat.reasoning_format, &prompt);
    let mut tools = ToolCallParser::new();

    let mut job = Job::new(prompt, params, cancel, ev_tx);
    job.timeout = state.default_timeout;
    if let Err(e) = state.scheduler.try_submit(job) {
        if matches!(e, golbang_core::SubmitError::Full) {
            state
                .scheduler
                .metrics
                .service_unavailable_total
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        return Err(e.into());
    }

    tokio::spawn(async move {
        if send_chunk(
            &sse_tx,
            &id,
            created,
            &model_name,
            Delta {
                role: Some("assistant"),
                content: None,
                ..Default::default()
            },
            None,
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
                    let split = parser.push(&t.piece);
                    let content = match split.content {
                        Some(c) => tools.push(&c),
                        None => None,
                    };
                    if split.reasoning.is_none() && content.is_none() {
                        continue;
                    }
                    send_chunk(
                        &sse_tx,
                        &id,
                        created,
                        &model_name,
                        Delta {
                            role: None,
                            content,
                            reasoning_content: split.reasoning,
                            ..Default::default()
                        },
                        None,
                        None,
                    )
                    .await
                }
                SlotEvent::Finished {
                    reason, timings, ..
                } => {
                    let split = parser.finish();
                    if let Some(c) = split.content {
                        let _ = tools.push(&c);
                    }
                    let parsed = tools.finish();
                    if split.reasoning.is_some() || !parsed.content.is_empty() {
                        let _ = send_chunk(
                            &sse_tx,
                            &id,
                            created,
                            &model_name,
                            Delta {
                                role: None,
                                content: nonempty_owned(parsed.content),
                                reasoning_content: split.reasoning,
                                ..Default::default()
                            },
                            None,
                            None,
                        )
                        .await;
                    }
                    let outgoing = outgoing_tool_calls(parsed.calls);
                    if !outgoing.is_empty() {
                        let names: Vec<&str> =
                            outgoing.iter().map(|t| t.function.name.as_str()).collect();
                        tracing::info!(?names, n = outgoing.len(), "parsed tool_calls");
                        let _ = send_chunk(
                            &sse_tx,
                            &id,
                            created,
                            &model_name,
                            Delta {
                                tool_calls: Some(tool_call_deltas(&outgoing)),
                                ..Default::default()
                            },
                            None,
                            None,
                        )
                        .await;
                    }
                    let finish = finish_reason_for(reason.as_str(), !outgoing.is_empty());
                    let r = send_chunk(
                        &sse_tx,
                        &id,
                        created,
                        &model_name,
                        Delta::default(),
                        Some(finish.clone()),
                        Some(Timings::from(timings)),
                    )
                    .await;
                    let _ = sse_tx.send(Event::default().data("[DONE]")).await;
                    tracing::info!(
                        streamed = n,
                        finish = finish.as_str(),
                        prompt_n = timings.prompt_n,
                        prompt_tps = format!("{:.2}", timings.prompt_per_second()),
                        predicted_n = timings.predicted_n,
                        predicted_tps = format!("{:.2}", timings.predicted_per_second()),
                        "sse finished"
                    );
                    r
                }
                SlotEvent::Failed(e) => {
                    tracing::error!(error = %e, "stream generation failed");
                    let payload = serde_json::json!({
                        "error": { "message": e.to_string(), "type": "server_error" }
                    });
                    let _ = sse_tx
                        .send(Event::default().data(payload.to_string()))
                        .await;
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
    timings: Option<Timings>,
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
        timings,
    };
    let data = serde_json::to_string(&chunk).map_err(|_| ())?;
    tx.send(Event::default().data(data)).await.map_err(|_| ())
}

fn nonempty_owned(s: String) -> Option<String> {
    if s.is_empty() { None } else { Some(s) }
}

pub async fn complete(
    state: AppState,
    req: ChatCompletionRequest,
) -> Result<ChatCompletion, ApiError> {
    let id = completion_id();
    let created = unix_ts();
    let model_name = model_name(&req, &state);
    let params = generate_params(&req, &state);
    let prompt = prompt_from(&req, &state);

    let (ev_tx, mut ev_rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    let _guard = cancel.clone().drop_guard();
    let mut parser = ReasoningParser::from_prompt(state.chat.reasoning_format, &prompt);
    let mut tools = ToolCallParser::new();

    let mut job = Job::new(prompt, params, cancel, ev_tx);
    job.timeout = state.default_timeout;
    if let Err(e) = state.scheduler.try_submit(job) {
        if matches!(e, golbang_core::SubmitError::Full) {
            state
                .scheduler
                .metrics
                .service_unavailable_total
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        return Err(e.into());
    }

    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    let mut finish = None;
    let mut prompt_tokens = 0u32;
    let mut completion_tokens = 0u32;
    let mut timings = None;

    while let Some(ev) = ev_rx.recv().await {
        match ev {
            SlotEvent::Token(t) => {
                let split = parser.push(&t.piece);
                if let Some(r) = split.reasoning {
                    reasoning.push_str(&r);
                }
                if let Some(c) = split.content {
                    if let Some(safe) = tools.push(&c) {
                        content.push_str(&safe);
                    }
                }
            }
            SlotEvent::Finished {
                reason,
                prompt_tokens: p,
                completion_tokens: c,
                timings: t,
            } => {
                let split = parser.finish();
                if let Some(r) = split.reasoning {
                    reasoning.push_str(&r);
                }
                if let Some(c) = split.content {
                    let _ = tools.push(&c);
                }
                let parsed = tools.finish();
                content.push_str(&parsed.content);
                tool_calls = outgoing_tool_calls(parsed.calls);
                finish = Some(finish_reason_for(reason.as_str(), !tool_calls.is_empty()));
                prompt_tokens = p;
                completion_tokens = c;
                timings = Some(Timings::from(t));
                break;
            }
            SlotEvent::Failed(e) => return Err(e.into()),
        }
    }

    if !tool_calls.is_empty() {
        let names: Vec<&str> = tool_calls
            .iter()
            .map(|t| t.function.name.as_str())
            .collect();
        tracing::info!(?names, n = tool_calls.len(), "parsed tool_calls");
    }

    if let Some(t) = timings {
        tracing::info!(
            prompt_tokens,
            completion_tokens,
            finish = ?finish,
            prompt_tps = format!("{:.2}", t.prompt_per_second),
            predicted_tps = format!("{:.2}", t.predicted_per_second),
            "json completion finished"
        );
    } else {
        tracing::info!(
            prompt_tokens,
            completion_tokens,
            finish = ?finish,
            "json completion finished"
        );
    }

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
                reasoning_content: if reasoning.is_empty() {
                    None
                } else {
                    Some(reasoning)
                },
                tool_calls,
            },
            finish_reason: finish,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        timings,
    })
}
