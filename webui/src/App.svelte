<script>
  import Settings from './components/Settings.svelte';
  import Stats from './components/Stats.svelte';
  import Chat from './components/Chat.svelte';
  import { settings, saveSettings } from './lib/store.svelte.js';
  import { fetchModels } from './lib/api.js';

  let model = $state(null);
  let panelOpen = $state(false);
  let tab = $state('settings');

  // Persist the API key / stream preference whenever they change.
  $effect(() => {
    settings.apiKey;
    settings.stream;
    saveSettings();
  });

  $effect(() => {
    fetchModels()
      .then((m) => (model = m))
      .catch(() => {});
  });

  function onKeydown(e) {
    if (e.key === 'Escape') panelOpen = false;
  }
</script>

<svelte:window onkeydown={onKeydown} />

<div class="app">
  <header class="topbar">
    <span class="brand">golbang</span>
    {#if model}
      <span class="model">{model.id} · n_ctx {model.meta?.n_ctx ?? '?'}</span>
    {/if}
    <span class="spacer"></span>
    <button type="button" onclick={() => (panelOpen = true)}>⚙ 설정</button>
  </header>

  <Chat />

  {#if panelOpen}
    <div class="modal-backdrop">
      <div class="modal" role="dialog" aria-modal="true" aria-label="설정" tabindex="-1">
        <div class="modal-head">
          <div class="tabs">
            <button type="button" class:active={tab === 'settings'} onclick={() => (tab = 'settings')}>
              설정
            </button>
            <button type="button" class:active={tab === 'stats'} onclick={() => (tab = 'stats')}>
              통계
            </button>
          </div>
          <button type="button" onclick={() => (panelOpen = false)}>닫기</button>
        </div>
        <div class="modal-body">
          {#if tab === 'settings'}
            <Settings />
          {:else}
            <Stats />
          {/if}
        </div>
      </div>
    </div>
  {/if}
</div>
