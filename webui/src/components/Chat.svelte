<script>
  import { settings } from '../lib/store.svelte.js';
  import { chatStream, chatOnce } from '../lib/api.js';

  let messages = $state([]);
  let input = $state('');
  let busy = $state(false);
  let error = $state('');
  let progress = $state(null);
  let scroller = $state(null);

  const fmt = (v, d = 2) => (typeof v === 'number' ? v.toFixed(d) : '—');

  function scrollDown() {
    queueMicrotask(() => {
      if (scroller) scroller.scrollTop = scroller.scrollHeight;
    });
  }

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    error = '';
    input = '';
    messages.push({ role: 'user', content: text });
    messages.push({
      role: 'assistant',
      content: '',
      reasoning: '',
      timings: null,
      finish: null,
      streaming: true,
    });
    const idx = messages.length - 1;
    const target = messages[idx];
    const payload = messages.slice(0, idx).map((m) => ({ role: m.role, content: m.content }));
    busy = true;
    scrollDown();

    const handlers = {
      content: (t) => {
        target.content += t;
        scrollDown();
      },
      reasoning: (t) => {
        target.reasoning += t;
        scrollDown();
      },
      timings: (t) => (target.timings = t),
      finish: (f) => (target.finish = f),
      progress: (p) => (progress = p),
    };

    try {
      if (settings.stream) {
        await chatStream(payload, handlers);
      } else {
        const r = await chatOnce(payload);
        const m = r.choices?.[0]?.message ?? {};
        target.content = m.content ?? '';
        target.reasoning = m.reasoning_content ?? '';
        target.timings = r.timings ?? null;
        target.finish = r.choices?.[0]?.finish_reason ?? null;
      }
    } catch (e) {
      error = e.message;
    } finally {
      target.streaming = false;
      busy = false;
      progress = null;
      scrollDown();
    }
  }

  function reset() {
    messages = [];
    error = '';
  }

  function onKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      send();
    }
  }
</script>

<div class="chat">
  <div class="messages" bind:this={scroller}>
    {#if messages.length === 0}
      <p class="muted">
        서버 스모크 테스트용 최소 UI입니다. 프롬프트를 보내면 응답과 함께 이 요청의 prefill /
        decode 속도가 표시됩니다.
      </p>
    {/if}

    {#each messages as m, i (i)}
      <div class="msg {m.role}">
        <div class="role">{m.role === 'user' ? '나' : '모델'}</div>
        {#if m.role === 'assistant' && progress && busy && i === messages.length - 1}
          <div class="progress">
            <div
              style="width: {progress.total > 0
                ? Math.min(100, (progress.processed / progress.total) * 100)
                : 100}%"
            ></div>
          </div>
        {/if}
        <div class="body">
          {#if m.content}{m.content}{:else if m.streaming}<span class="muted">…</span>{/if}
        </div>
        {#if m.reasoning}
          <details class="reasoning">
            <summary>thinking</summary>
            <div>{m.reasoning}</div>
          </details>
        {/if}
        {#if m.timings}
          <div class="meta">
            <span>프롬프트 <b>{m.timings.prompt_n}</b> tok · <b>{fmt(m.timings.prompt_per_second)}</b> tok/s</span>
            <span>생성 <b>{m.timings.predicted_n}</b> tok · <b>{fmt(m.timings.predicted_per_second)}</b> tok/s</span>
            {#if m.timings.cache_n}<span>캐시 <b>{m.timings.cache_n}</b></span>{/if}
            {#if m.timings.draft_n}<span>드래프트 <b>{m.timings.draft_n_accepted}/{m.timings.draft_n}</b></span>{/if}
            {#if m.finish}<span>finish <b>{m.finish}</b></span>{/if}
          </div>
        {/if}
      </div>
    {/each}
  </div>

  {#if error}
    <p class="err" style="padding: 0 18px">{error}</p>
  {/if}

  <div class="composer">
    <textarea
      placeholder="메시지를 입력하고 Enter (Shift+Enter 줄바꿈)"
      rows="2"
      value={input}
      oninput={(e) => (input = e.currentTarget.value)}
      onkeydown={onKeydown}
      disabled={busy}
    ></textarea>
    <button class="primary" type="button" onclick={send} disabled={busy || !input.trim()}>
      {busy ? '생성 중' : '전송'}
    </button>
    <button type="button" onclick={reset} disabled={busy || messages.length === 0}>초기화</button>
  </div>
</div>
