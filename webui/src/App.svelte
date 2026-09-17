<script>
  import Settings from './components/Settings.svelte';
  import Stats from './components/Stats.svelte';
  import Chat from './components/Chat.svelte';
  import { settings, saveSettings } from './lib/store.svelte.js';
  import { fetchModels } from './lib/api.js';

  let model = $state(null);

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
</script>

<div class="app">
  <header class="topbar">
    <span class="brand">golbang</span>
    {#if model}
      <span class="model">{model.id} · n_ctx {model.meta?.n_ctx ?? '?'}</span>
    {/if}
    <span class="spacer"></span>
    <a href="/metrics" target="_blank" rel="noreferrer">/metrics</a>
    <a href="/v1/models" target="_blank" rel="noreferrer">/v1/models</a>
  </header>
  <div class="columns">
    <aside class="sidebar">
      <Settings />
      <Stats />
    </aside>
    <Chat />
  </div>
</div>
