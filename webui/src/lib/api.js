import { settings } from './store.svelte.js';

function authHeaders() {
  const headers = { 'Content-Type': 'application/json' };
  const key = settings.apiKey.trim();
  if (key) headers['Authorization'] = `Bearer ${key}`;
  return headers;
}

export async function fetchModels() {
  const res = await fetch('/v1/models');
  if (!res.ok) throw new Error(`/v1/models HTTP ${res.status}`);
  const data = await res.json();
  const first = data?.data?.[0] ?? null;
  return first;
}

export async function getMetricsText() {
  const res = await fetch('/metrics');
  if (!res.ok) throw new Error(`/metrics HTTP ${res.status}`);
  return res.text();
}

async function errorMessage(res) {
  try {
    const body = await res.json();
    return body?.error?.message || `HTTP ${res.status}`;
  } catch {
    return `HTTP ${res.status}`;
  }
}

function dispatchSse(raw, handlers) {
  for (const line of raw.split('\n')) {
    if (!line.startsWith('data:')) continue;
    const data = line.slice(5).trim();
    if (data === '[DONE]') {
      handlers.done?.();
      continue;
    }
    let chunk;
    try {
      chunk = JSON.parse(data);
    } catch {
      continue;
    }
    if (chunk.prompt_progress) handlers.progress?.(chunk.prompt_progress);
    const choice = chunk.choices?.[0];
    if (choice?.delta?.reasoning_content) {
      handlers.reasoning?.(choice.delta.reasoning_content);
    }
    if (choice?.delta?.content) handlers.content?.(choice.delta.content);
    if (chunk.timings) handlers.timings?.(chunk.timings);
    if (choice?.finish_reason) handlers.finish?.(choice.finish_reason);
  }
}

/// Streamed chat completion (POST + manual SSE framing; EventSource cannot POST).
export async function chatStream(messages, handlers) {
  const res = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({
      model: 'golbang',
      messages,
      stream: true,
      temperature: settings.temperature,
      max_tokens: settings.maxTokens,
    }),
  });
  if (!res.ok) throw new Error(await errorMessage(res));

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf('\n\n')) >= 0) {
      dispatchSse(buffer.slice(0, idx), handlers);
      buffer = buffer.slice(idx + 2);
    }
  }
  if (buffer.trim()) dispatchSse(buffer, handlers);
}

/// Buffered chat completion (`stream: false`).
export async function chatOnce(messages) {
  const res = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({
      model: 'golbang',
      messages,
      stream: false,
      temperature: settings.temperature,
      max_tokens: settings.maxTokens,
    }),
  });
  if (!res.ok) throw new Error(await errorMessage(res));
  return res.json();
}
