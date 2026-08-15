use std::convert::Infallible;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::http::{HeaderValue, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use golbang_core::{
    CancellationToken, ChatApplyOpts, GenerateParams, Job, ReasoningParser, SlotEvent,
    ToolCallParser, apply_chat_template_with, load_media_bytes, prompt_opens_think,
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

pub fn generate_params(
    req: &ChatCompletionRequest,
    state: &AppState,
    prompt: &str,
) -> GenerateParams {
    let mut stop = req.stop.clone().map(|s| s.into_vec()).unwrap_or_default();
    if state.chat.use_jinja {
        for extra in [
            "<｜User｜>",
            "<｜end▁of▁sentence｜>",
            "<|im_end|>",
            "<|im_start|>",
        ] {
            if !stop.iter().any(|s| s == extra) {
                stop.push(extra.to_string());
            }
        }
    }
    let extracts = state.chat.reasoning_format.extracts();
    GenerateParams {
        max_tokens: req.max_tokens.unwrap_or(256).max(1),
        temperature: req.temperature.unwrap_or(1.0).max(0.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k.unwrap_or(0),
        seed: req.seed.unwrap_or(0),
        stop,
        reasoning_budget: if extracts {
            resolve_reasoning_budget(req.reasoning_budget, state.chat.reasoning_budget)
        } else {
            0
        },
        start_in_think: extracts && prompt_opens_think(prompt),
    }
}

fn resolve_reasoning_budget(req: Option<i32>, server: u32) -> u32 {
    match req {
        None => server,
        Some(n) if n <= 0 => 0,
        Some(n) => n as u32,
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

fn collect_images(req: &ChatCompletionRequest) -> Result<Vec<Vec<u8>>, ApiError> {
    let mut out = Vec::new();
    for m in &req.messages {
        for src in &m.content.media {
            let bytes = load_media_bytes(src)
                .map_err(|e| ApiError::invalid_request(e.to_string(), Some("messages")))?;
            out.push(bytes);
        }
    }
    Ok(out)
}

fn submit_job(
    state: &AppState,
    req: &ChatCompletionRequest,
    prompt: String,
    params: GenerateParams,
    cancel: CancellationToken,
    ev_tx: mpsc::UnboundedSender<SlotEvent>,
) -> Result<(), ApiError> {
    let images = collect_images(req)?;
    if !images.is_empty() && !state.vision {
        return Err(ApiError::invalid_request(
            "image input requires the server to be started with --mmproj",
            Some("messages"),
        ));
    }
    let mut job = Job::new(prompt, params, cancel, ev_tx);
    job.timeout = state.default_timeout;
    job.images = images;
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
    Ok(())
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
            reasoning_effort: req
                .reasoning_effort
                .clone()
                .or_else(|| state.chat.reasoning_effort.clone()),
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
) -> Result<Response, ApiError> {
    let id = completion_id();
    let created = unix_ts();
    let model_name = model_name(&req, &state);
    let prompt = prompt_from(&req, &state);
    let params = generate_params(&req, &state, &prompt);

    let (ev_tx, mut ev_rx) = mpsc::unbounded_channel();
    let (sse_tx, sse_rx) = mpsc::channel::<Event>(256);
    let cancel = CancellationToken::new();
    let cancel_on_drop = cancel.clone();
    let mut parser = ReasoningParser::from_prompt(state.chat.reasoning_format, &prompt);
    let mut tools = ToolCallParser::new();

    submit_job(&state, &req, prompt, params, cancel, ev_tx)?;

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
        let mut pending = PendingDelta::new();
        loop {
            tokio::select! {
                ev = ev_rx.recv() => {
                    let Some(ev) = ev else { break };
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
                            pending.push(split.reasoning, content);
                            if pending.due() {
                                flush_pending(&sse_tx, &id, created, &model_name, &mut pending).await
                            } else {
                                Ok(())
                            }
                        }
                        SlotEvent::Finished {
                            reason, timings, ..
                        } => {
                            if flush_pending(&sse_tx, &id, created, &model_name, &mut pending)
                                .await
                                .is_err()
                            {
                                Err(())
                            } else {
                                finish_stream(
                                    &sse_tx,
                                    &id,
                                    created,
                                    &model_name,
                                    &mut parser,
                                    &mut tools,
                                    reason,
                                    timings,
                                    n,
                                )
                                .await
                            }
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
                        SlotEvent::PromptProgress {
                            n_tokens,
                            progress,
                            tps,
                        } => {
                            // Comment + empty delta: axum KeepAlive is not enough for
                            // clients that only reset idle timers on `data:` events.
                            let comment = format!(
                                "prompt processing n_tokens={n_tokens} progress={progress:.2} tps={tps:.1}"
                            );
                            if sse_tx
                                .send(Event::default().comment(comment))
                                .await
                                .is_err()
                            {
                                cancel_on_drop.cancel();
                                tracing::info!(
                                    "client gone during prefill; cancel accepted, prefix kv retained"
                                );
                                break;
                            }
                            send_chunk(
                                &sse_tx,
                                &id,
                                created,
                                &model_name,
                                Delta::default(),
                                None,
                                None,
                            )
                            .await
                        }
                    };
                    if send.is_err() {
                        cancel_on_drop.cancel();
                        tracing::info!("client gone; cancel accepted, slot reclaim at next decode");
                        break;
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(40)), if !pending.is_empty() => {
                    if flush_pending(&sse_tx, &id, created, &model_name, &mut pending)
                        .await
                        .is_err()
                    {
                        cancel_on_drop.cancel();
                        tracing::info!("client gone; cancel accepted, slot reclaim at next decode");
                        break;
                    }
                }
            }
        }
    });

    let sse = Sse::new(ReceiverStream::new(sse_rx).map(Ok::<Event, Infallible>))
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(2)));
    let mut res = sse.into_response();
    res.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-cache, no-transform"),
    );
    res.headers_mut()
        .insert("X-Accel-Buffering", HeaderValue::from_static("no"));
    Ok(res)
}

struct PendingDelta {
    reasoning: String,
    content: String,
    n: u32,
    since: Instant,
}

impl PendingDelta {
    fn new() -> Self {
        Self {
            reasoning: String::new(),
            content: String::new(),
            n: 0,
            since: Instant::now(),
        }
    }

    fn push(&mut self, reasoning: Option<String>, content: Option<String>) {
        if let Some(s) = reasoning {
            self.reasoning.push_str(&s);
        }
        if let Some(s) = content {
            self.content.push_str(&s);
        }
        if self.n == 0 {
            self.since = Instant::now();
        }
        self.n += 1;
    }

    fn is_empty(&self) -> bool {
        self.n == 0
    }

    fn due(&self) -> bool {
        if self.n == 0 {
            return false;
        }
        self.n >= 8
            || self.reasoning.len() + self.content.len() >= 192
            || self.since.elapsed() >= Duration::from_millis(40)
    }

    fn take(&mut self) -> (Option<String>, Option<String>) {
        self.n = 0;
        self.since = Instant::now();
        (
            nonempty_owned(std::mem::take(&mut self.reasoning)),
            nonempty_owned(std::mem::take(&mut self.content)),
        )
    }
}

async fn flush_pending(
    sse_tx: &mpsc::Sender<Event>,
    id: &str,
    created: u64,
    model_name: &str,
    pending: &mut PendingDelta,
) -> Result<(), ()> {
    if pending.is_empty() {
        return Ok(());
    }
    let (reasoning, content) = pending.take();
    send_chunk(
        sse_tx,
        id,
        created,
        model_name,
        Delta {
            role: None,
            content,
            reasoning_content: reasoning,
            ..Default::default()
        },
        None,
        None,
    )
    .await
}

async fn finish_stream(
    sse_tx: &mpsc::Sender<Event>,
    id: &str,
    created: u64,
    model_name: &str,
    parser: &mut ReasoningParser,
    tools: &mut ToolCallParser,
    reason: golbang_core::FinishReason,
    timings: golbang_core::SlotTimings,
    n: u32,
) -> Result<(), ()> {
    let split = parser.finish();
    if let Some(c) = split.content {
        let _ = tools.push(&c);
    }
    let parsed = tools.finish();
    if split.reasoning.is_some() || !parsed.content.is_empty() {
        let _ = send_chunk(
            sse_tx,
            id,
            created,
            model_name,
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
        let names: Vec<&str> = outgoing.iter().map(|t| t.function.name.as_str()).collect();
        tracing::info!(?names, n = outgoing.len(), "parsed tool_calls");
        let _ = send_chunk(
            sse_tx,
            id,
            created,
            model_name,
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
        sse_tx,
        id,
        created,
        model_name,
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
    let prompt = prompt_from(&req, &state);
    let params = generate_params(&req, &state, &prompt);

    let (ev_tx, mut ev_rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    let _guard = cancel.clone().drop_guard();
    let mut parser = ReasoningParser::from_prompt(state.chat.reasoning_format, &prompt);
    let mut tools = ToolCallParser::new();

    submit_job(&state, &req, prompt, params, cancel, ev_tx)?;

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
            SlotEvent::PromptProgress { .. } => {}
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
