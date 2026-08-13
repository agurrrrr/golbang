use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::OnceLock;

use golbang_sys::*;

use crate::error::{Error, Result};
use crate::generate::{Generate, GenerateParams};
use crate::tokenizer::{Token, Tokenizer};

/// Load options. `n_ctx` is **per sequence**. Total KV is `n_ctx * n_seq_max`
/// so P1's 256-token slot stays 256 when `--n-parallel` grows.
/// Keep both small when another llama-server already holds most of VRAM.
#[derive(Clone, Debug)]
pub struct LoadParams {
    pub n_gpu_layers: i32,
    pub n_ctx: u32,
    pub n_seq_max: u32,
}

impl Default for LoadParams {
    fn default() -> Self {
        Self {
            n_gpu_layers: 99,
            n_ctx: 256,
            n_seq_max: 1,
        }
    }
}

/// RAII owner of `llama_model` + `llama_context`. GPU access is serialized
/// by the caller (P1: one request). `Send` so it can live on a worker thread.
pub struct Model {
    model: *mut llama_model,
    ctx: *mut llama_context,
    vocab: *const llama_vocab,
    n_vocab: i32,
    path: PathBuf,
}

unsafe impl Send for Model {}

impl Model {
    pub fn load(path: impl AsRef<Path>, params: LoadParams) -> Result<Self> {
        init_backend();

        let path = path.as_ref();
        if !path.is_file() {
            return Err(Error::Load {
                path: path.to_path_buf(),
                reason: "not a file".into(),
            });
        }
        let c_path = CString::new(path.to_string_lossy().as_bytes()).map_err(|_| Error::Load {
            path: path.to_path_buf(),
            reason: "path contains interior NUL".into(),
        })?;

        let n_ctx_seq = params.n_ctx.max(1);
        let n_seq_max = params.n_seq_max.max(1);
        let n_ctx_total = n_ctx_seq.saturating_mul(n_seq_max);
        tracing::info!(
            path = %path.display(),
            n_ctx_seq,
            n_ctx_total,
            n_seq_max,
            n_gpu_layers = params.n_gpu_layers,
            "loading GGUF"
        );

        let mut mparams = unsafe { llama_model_default_params() };
        mparams.n_gpu_layers = params.n_gpu_layers;

        let model = unsafe { llama_model_load_from_file(c_path.as_ptr(), mparams) };
        if model.is_null() {
            return Err(Error::Load {
                path: path.to_path_buf(),
                reason: "llama_model_load_from_file returned null".into(),
            });
        }

        let vocab = unsafe { llama_model_get_vocab(model) };
        if vocab.is_null() {
            unsafe { llama_model_free(model) };
            return Err(Error::Null("llama_model_get_vocab"));
        }
        let n_vocab = unsafe { llama_vocab_n_tokens(vocab) };
        if n_vocab <= 0 {
            unsafe { llama_model_free(model) };
            return Err(Error::Load {
                path: path.to_path_buf(),
                reason: format!("n_vocab={n_vocab}"),
            });
        }

        let mut cparams = unsafe { llama_context_default_params() };
        cparams.n_ctx = n_ctx_total;
        cparams.n_batch = n_ctx_seq;
        cparams.n_ubatch = n_ctx_seq;
        cparams.n_seq_max = n_seq_max;

        let ctx = unsafe { llama_init_from_model(model, cparams) };
        if ctx.is_null() {
            unsafe { llama_model_free(model) };
            return Err(Error::Load {
                path: path.to_path_buf(),
                reason: "llama_init_from_model returned null (VRAM?)".into(),
            });
        }

        tracing::info!(
            n_ctx = unsafe { llama_n_ctx(ctx) },
            n_ctx_seq = unsafe { llama_n_ctx_seq(ctx) },
            n_seq_max = unsafe { llama_n_seq_max(ctx) },
            n_vocab,
            n_layer = unsafe { llama_model_n_layer(model) },
            "model ready"
        );

        Ok(Self {
            model,
            ctx,
            vocab,
            n_vocab,
            path: path.to_path_buf(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn n_vocab(&self) -> i32 {
        self.n_vocab
    }

    pub fn n_ctx(&self) -> u32 {
        unsafe { llama_n_ctx(self.ctx) }
    }

    /// Per-sequence context (KV cells one slot may occupy).
    pub fn n_ctx_seq(&self) -> u32 {
        unsafe { llama_n_ctx_seq(self.ctx) }
    }

    pub fn n_seq_max(&self) -> u32 {
        unsafe { llama_n_seq_max(self.ctx) }
    }

    pub fn n_batch(&self) -> u32 {
        unsafe { llama_n_batch(self.ctx) }
    }

    pub fn tokenizer(&self) -> Tokenizer<'_> {
        Tokenizer::new(self.vocab, self.n_vocab)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<Token>> {
        self.tokenizer().encode(text, true, true)
    }

    pub fn decode_tokens(&self, tokens: &[Token]) -> Result<String> {
        self.tokenizer().decode(tokens, false, true)
    }

    pub fn token_to_piece(&self, token: Token) -> Result<String> {
        let bytes = self.tokenizer().token_to_piece(token, true)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn is_eog(&self, token: Token) -> bool {
        self.tokenizer().is_eog(token)
    }

    pub fn clear_kv(&mut self) {
        unsafe {
            let mem = llama_get_memory(self.ctx);
            if !mem.is_null() {
                llama_memory_clear(mem, true);
            }
        }
    }

    /// Drop one sequence's KV. Whole-sequence remove never fails (llama.h).
    pub fn clear_seq(&mut self, seq_id: i32) {
        unsafe {
            let mem = llama_get_memory(self.ctx);
            if !mem.is_null() {
                llama_memory_seq_rm(mem, seq_id, -1, -1);
            }
        }
    }

    /// Tokens already in `seq_id` (0 if empty).
    pub fn n_past_seq(&self, seq_id: i32) -> u32 {
        unsafe {
            let mem = llama_get_memory(self.ctx);
            if mem.is_null() {
                return 0;
            }
            let max = llama_memory_seq_pos_max(mem, seq_id);
            if max < 0 {
                0
            } else {
                (max + 1) as u32
            }
        }
    }

    /// Tokens already in seq 0 (0 if empty).
    pub fn n_past(&self) -> u32 {
        self.n_past_seq(0)
    }

    /// Prefill or decode a token chunk on seq 0. Positions tracked by llama.cpp
    /// when `n_seq_max == 1`; multi-seq contexts use explicit pos/seq ids.
    pub fn decode(&mut self, tokens: &[Token]) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        if self.n_seq_max() > 1 {
            let start = self.n_past_seq(0) as i32;
            let items: Vec<crate::batch::BatchToken> = tokens
                .iter()
                .enumerate()
                .map(|(i, &token)| crate::batch::BatchToken {
                    token,
                    pos: start + i as i32,
                    seq_id: 0,
                    logits: i + 1 == tokens.len(),
                })
                .collect();
            return self.decode_items(&items);
        }
        let n_batch = unsafe { llama_n_batch(self.ctx) }.max(1) as usize;
        let mut i = 0;
        while i < tokens.len() {
            let end = (i + n_batch).min(tokens.len());
            let mut chunk = tokens[i..end].to_vec();
            let batch = unsafe { llama_batch_get_one(chunk.as_mut_ptr(), chunk.len() as i32) };
            let rc = unsafe { llama_decode(self.ctx, batch) };
            if rc != 0 {
                return Err(Error::Decode(rc));
            }
            i = end;
        }
        Ok(())
    }

    /// Multi-sequence decode. `items.len()` must be ≤ `n_batch`.
    pub fn decode_items(&mut self, items: &[crate::batch::BatchToken]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let n_batch = self.n_batch().max(1) as usize;
        if items.len() > n_batch {
            return Err(Error::BatchTooLarge {
                got: items.len(),
                n_batch,
            });
        }

        let n = items.len() as i32;
        let mut batch = unsafe { llama_batch_init(n, 0, 1) };
        if batch.token.is_null() || batch.pos.is_null() || batch.seq_id.is_null() {
            unsafe { llama_batch_free(batch) };
            return Err(Error::Null("llama_batch_init"));
        }

        for (i, it) in items.iter().enumerate() {
            unsafe {
                *batch.token.add(i) = it.token;
                *batch.pos.add(i) = it.pos;
                *batch.n_seq_id.add(i) = 1;
                let seqs = *batch.seq_id.add(i);
                if seqs.is_null() {
                    llama_batch_free(batch);
                    return Err(Error::Null("llama_batch.seq_id"));
                }
                *seqs.add(0) = it.seq_id;
                *batch.logits.add(i) = i8::from(it.logits);
            }
        }
        batch.n_tokens = n;

        let rc = unsafe { llama_decode(self.ctx, batch) };
        unsafe { llama_batch_free(batch) };
        if rc != 0 {
            return Err(Error::Decode(rc));
        }
        Ok(())
    }

    /// Last-token logits from the previous `decode`. Valid until the next decode.
    pub fn logits(&self) -> Result<&[f32]> {
        self.logits_ith(-1)
    }

    /// Logits for batch index `i` (`-1` = last). Valid until the next decode.
    pub fn logits_ith(&self, i: i32) -> Result<&[f32]> {
        unsafe {
            let ptr = llama_get_logits_ith(self.ctx, i);
            if ptr.is_null() {
                return Err(Error::Null("llama_get_logits_ith"));
            }
            Ok(std::slice::from_raw_parts(ptr, self.n_vocab as usize))
        }
    }

    /// Token-at-a-time generation. Each `next`/`step` is one sample (P2 unit).
    pub fn generate(&mut self, prompt: &str, params: GenerateParams) -> Result<Generate<'_>> {
        Generate::start(self, prompt, params)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                llama_free(self.ctx);
                self.ctx = ptr::null_mut();
            }
            if !self.model.is_null() {
                llama_model_free(self.model);
                self.model = ptr::null_mut();
            }
        }
    }
}

fn init_backend() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        unsafe {
            llama_log_set(Some(forward_llama_log), ptr::null_mut());
            llama_backend_init();
            // cargo test / the server binary do not sit next to libggml-hip.so
            let bin = CString::new(golbang_sys::LLAMA_BIN_DIR).expect("LLAMA_BIN_DIR");
            ggml_backend_load_all_from_path(bin.as_ptr());
        }
        tracing::info!(
            bin = golbang_sys::LLAMA_BIN_DIR,
            sha = golbang_sys::LLAMA_CPP_SHA,
            "llama backend initialized"
        );
    });
}

unsafe extern "C" fn forward_llama_log(level: ggml_log_level, text: *const c_char, _ud: *mut c_void) {
    if text.is_null() {
        return;
    }
    let s = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    let trimmed = s.trim_end();
    if trimmed.is_empty() {
        return;
    }
    if level >= GGML_LOG_LEVEL_WARN {
        eprintln!("{trimmed}");
    }
    match level {
        GGML_LOG_LEVEL_ERROR => tracing::error!(target: "llama", "{trimmed}"),
        GGML_LOG_LEVEL_WARN => tracing::warn!(target: "llama", "{trimmed}"),
        _ => tracing::debug!(target: "llama", "{trimmed}"),
    }
}
