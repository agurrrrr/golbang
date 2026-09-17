// Persisted settings shared across components (Svelte 5 runes module).
const KEY_STORE = 'golbang.apiKey';
const STREAM_STORE = 'golbang.stream';

function read(key, fallback) {
  try {
    const v = localStorage.getItem(key);
    return v === null ? fallback : v;
  } catch {
    return fallback;
  }
}

export const settings = $state({
  apiKey: read(KEY_STORE, ''),
  stream: read(STREAM_STORE, '1') !== '0',
  temperature: 0.7,
  maxTokens: 512,
});

export function saveSettings() {
  try {
    localStorage.setItem(KEY_STORE, settings.apiKey);
    localStorage.setItem(STREAM_STORE, settings.stream ? '1' : '0');
  } catch {
    // private mode / storage disabled — settings just won't persist
  }
}
