// Session-scoped settings shared across components (Svelte 5 runes module).
//
// The API key lives in `sessionStorage`: it is dropped when the tab closes and
// never written to disk (unlike `localStorage`). Nothing else is persisted.
const KEY_STORE = 'golbang.apiKey';
const STREAM_STORE = 'golbang.stream';

function read(key, fallback) {
  try {
    const v = sessionStorage.getItem(key);
    return v === null ? fallback : v;
  } catch {
    return fallback;
  }
}

export const settings = $state({
  apiKey: read(KEY_STORE, ''),
  stream: read(STREAM_STORE, '1') !== '0',
});

export function saveSettings() {
  try {
    sessionStorage.setItem(KEY_STORE, settings.apiKey);
    sessionStorage.setItem(STREAM_STORE, settings.stream ? '1' : '0');
  } catch {
    // private mode / storage disabled — settings just won't persist
  }
}
