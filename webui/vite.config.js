import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { viteSingleFile } from 'vite-plugin-singlefile';

// The build emits a single self-contained `dist/index.html` so the Rust binary
// can `include_bytes!` it and `cargo build` never needs Node.
export default defineConfig({
  plugins: [svelte(), viteSingleFile()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    cssCodeSplit: false,
    assetsInlineLimit: 100000000,
    target: 'es2020',
    reportCompressedSize: false,
  },
});
