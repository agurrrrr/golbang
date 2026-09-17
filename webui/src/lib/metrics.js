// Parse the Prometheus text exposition served at `/metrics`.
//
// Values may carry a trailing comment (`... 0.5  # total=... samples=...`),
// so only the first whitespace-delimited token after the name is used.
export function parseMetrics(text) {
  const out = {};
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const sp = trimmed.indexOf(' ');
    if (sp < 0) continue;
    const name = trimmed.slice(0, sp);
    const value = Number(trimmed.slice(sp + 1).trim().split(/\s+/)[0]);
    if (Number.isFinite(value)) out[name] = value;
  }
  return out;
}
