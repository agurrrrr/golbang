use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::OnceLock;

use golbang_sys::*;

use crate::error::{Error, Result};
use crate::generate::{Generate, GenerateParams};
use crate::tokenizer::{Token, Tokenizer};

/// Load options. `n_ctx` stays small by default so this can share a card
/// with another llama-server occupying most of VRAM (P0 leftover ~800 MiB).
#[derive(Clone, Debug)]
pub struct LoadParams {
    pub n_gpu_layers: i32,
    pub n_ctx: u32,
}

impl Default for LoadParams {
    fn default() -> Self {
        Self {
            n_gpu_layers: 99,
            n_ctx: 256,
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

        let n_ctx = params.n_ctx.max(1);
        tracing::info!(
            path = %path.display(),
            n_ctx,
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
        cparams.n_ctx = n_ctx;
        cparams.n_batch = n_ctx;
        cparams.n_ubatch = n_ctx;
        cparams.n_seq_max = 1;

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

    /// Tokens already in seq 0 (0 if empty).
    pub fn n_past(&self) -> u32 {
        unsafe {
            let mem = llama_get_memory(self.ctx);
            if mem.is_null() {
                return 0;
            }
            let max = llama_memory_seq_pos_max(mem, 0);
            if max < 0 {
                0
            } else {
                (max + 1) as u32
            }
        }
    }

    /// Prefill or decode a token chunk. Positions are tracked by llama.cpp.
    pub fn decode(&mut self, tokens: &[Token]) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
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

    /// Last-token logits from the previous `decode`. Valid until the next decode.
    pub fn logits(&self) -> Result<&[f32]> {
        unsafe {
            let ptr = llama_get_logits_ith(self.ctx, -1);
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
