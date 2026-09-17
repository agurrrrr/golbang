<script>
  import { getMetricsText } from '../lib/api.js';
  import { parseMetrics } from '../lib/metrics.js';

  let stats = $state(null);
  let error = $state('');
  let live = $state(true);

  async function refresh() {
    try {
      stats = parseMetrics(await getMetricsText());
      error = '';
    } catch (e) {
      error = e.message;
    }
  }

  $effect(() => {
    if (!live) return;
    refresh();
    const id = setInterval(refresh, 1500);
    return () => clearInterval(id);
  });

  const g = (k) => stats?.[k];
  const num = (v, d = 1) => (v === undefined ? '—' : Number(v).toFixed(d));
  const int = (v) => (v === undefined ? '—' : Math.round(v).toLocaleString());
  const ratio = (a, b) =>
    a !== undefined && b !== undefined && b > 0 ? ((a / b) * 100).toFixed(1) + '%' : '—';

  const prefillTps = $derived(g('golbang_prompt_tokens_per_second'));
  const decodeTps = $derived(g('golbang_predicted_tokens_per_second'));
  const cachedRatio = $derived(
    ratio(g('golbang_prompt_tokens_cached_total'), (g('golbang_prompt_tokens_total') ?? 0) + (g('golbang_prompt_tokens_cached_total') ?? 0)),
  );
  const draftRatio = $derived(ratio(g('golbang_draft_accepted_total'), g('golbang_draft_tokens_total')));
</script>

<div class="panel">
  <div class="row" style="justify-content: space-between; align-items: center">
    <h2 style="margin: 0">추론 속도</h2>
    <div class="row">
      <label for="live" style="margin: 0; font-size: 11px">live</label>
      <input
        id="live"
        type="checkbox"
        style="width: auto"
        checked={live}
        onchange={(e) => (live = e.currentTarget.checked)}
      />
      <button type="button" onclick={refresh}>↻</button>
    </div>
  </div>

  {#if error}
    <p class="err" style="font-size: 12px">{error}</p>
  {/if}

  <div class="stats" style="margin-top: 10px">
    <div class="stat">
      <div class="k">디코드</div>
      <div class="v">{num(decodeTps)}<span class="u">tok/s</span></div>
    </div>
    <div class="stat">
      <div class="k">프리필</div>
      <div class="v">{num(prefillTps)}<span class="u">tok/s</span></div>
    </div>
    <div class="stat">
      <div class="k">생성 토큰</div>
      <div class="v">{int(g('golbang_tokens_generated_total'))}</div>
    </div>
    <div class="stat">
      <div class="k">프리필 토큰</div>
      <div class="v">{int(g('golbang_prompt_tokens_total'))}</div>
    </div>
    <div class="stat">
      <div class="k">캐시 재사용</div>
      <div class="v">{cachedRatio}</div>
    </div>
    <div class="stat">
      <div class="k">드래프트 수락</div>
      <div class="v">{draftRatio}</div>
    </div>
    <div class="stat">
      <div class="k">처리 중</div>
      <div class="v">{int(g('llamacpp:requests_processing'))}</div>
    </div>
    <div class="stat">
      <div class="k">대기열</div>
      <div class="v">{int(g('llamacpp:requests_deferred'))}</div>
    </div>
    <div class="stat">
      <div class="k">슬롯 점유</div>
      <div class="v">{num(g('golbang_slot_occupancy'), 2)}</div>
    </div>
    <div class="stat">
      <div class="k">503</div>
      <div class="v">{int(g('golbang_http_503_total'))}</div>
    </div>
    <div class="stat">
      <div class="k">유효 n_ctx</div>
      <div class="v">{int(g('golbang_pool_ctx_effective'))}</div>
    </div>
    <div class="stat">
      <div class="k">큐 깊이</div>
      <div class="v">{num(g('golbang_queue_depth'), 2)}</div>
    </div>
  </div>
</div>
