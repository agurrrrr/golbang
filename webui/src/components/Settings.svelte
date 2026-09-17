<script>
  import { settings } from '../lib/store.svelte.js';

  let show = $state(false);
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
    브라우저 <code>sessionStorage</code>에만 저장됩니다. 탭을 닫으면 지워지고 디스크에는 남지
    않습니다. 서버가 <code>--api-key</code> 없이 떴다면 비워 두십시오.
  </p>

  <label for="stream" style="margin-top: 14px">응답 스트리밍</label>
  <input
    id="stream"
    type="checkbox"
    checked={settings.stream}
    onchange={(e) => (settings.stream = e.currentTarget.checked)}
  />

  <p class="muted" style="font-size: 12px; margin: 12px 0 0">
    temperature·max_tokens·context length는 보내지 않고 서버 설정값을 그대로 씁니다.
  </p>
</div>
