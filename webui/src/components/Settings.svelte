<script>
  import { settings } from '../lib/store.svelte.js';

  let show = $state(false);

  const num = (e, key) => {
    settings[key] = Number(e.currentTarget.value);
  };
</script>

<div class="panel">
  <h2>API 키</h2>
  <label for="apikey">Authorization: Bearer …</label>
  <div class="row">
    <input
      id="apikey"
      type={show ? 'text' : 'password'}
      placeholder="서버가 요구할 때만 입력"
      value={settings.apiKey}
      oninput={(e) => (settings.apiKey = e.currentTarget.value)}
      autocomplete="off"
      spellcheck="false"
    />
    <button type="button" onclick={() => (show = !show)}>{show ? '숨김' : '표시'}</button>
  </div>
  <p class="muted" style="font-size: 12px; margin: 8px 0 0">
    브라우저 <code>localStorage</code>에만 저장됩니다. 서버가 <code>--api-key</code> 없이 떴다면
    비워 두십시오.
  </p>

  <details class="adv" style="margin-top: 10px">
    <summary>요청 설정</summary>
    <div class="row" style="margin-top: 8px">
      <div style="flex: 1">
        <label for="temperature">temperature</label>
        <input
          id="temperature"
          type="number"
          min="0"
          max="2"
          step="0.05"
          value={settings.temperature}
          oninput={(e) => num(e, 'temperature')}
        />
      </div>
      <div style="flex: 1">
        <label for="maxtokens">max_tokens</label>
        <input
          id="maxtokens"
          type="number"
          min="1"
          step="1"
          value={settings.maxTokens}
          oninput={(e) => num(e, 'maxTokens')}
        />
      </div>
    </div>
    <label for="stream" style="margin-top: 10px">응답 스트리밍</label>
    <input
      id="stream"
      type="checkbox"
      checked={settings.stream}
      onchange={(e) => (settings.stream = e.currentTarget.checked)}
    />
  </details>
</div>
