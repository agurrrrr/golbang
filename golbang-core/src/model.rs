use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::OnceLock;

use golbang_sys::*;

use crate::error::{Error, Result};
use crate::generate::{Generate, GenerateParams};
use crate::speculative::{NgramMod, SpecParams, SpecType, topk_mode};
use crate::tokenizer::{Token, Tokenizer};
use crate::vision::{TokenizedVision, Vision};

/// Matches llama.cpp `LLM_FFN_EXPS_REGEX` — expert tensors left on CPU by `-ncmoe`.
const FFN_EXPS_REGEX: &str = r"\.ffn_(up|down|gate|gate_up)_(ch|)exps";

/// Load options. `n_ctx` is **per sequence**. Total KV is `n_ctx * n_seq_max`
/// so P1's 256-token slot stays 256 when `--n-parallel` grows.
/// Keep both small when another llama-server already holds most of VRAM.
///
/// `n_cpu_moe` pins expert weights of the first N layers to CPU (`-ncmoe`).
/// Required to fit large MoE GGUFs (e.g. DSV4 IQ2_M ~85 GiB) on 32 GiB VRAM.
/// `n_batch`/`n_ubatch` of 0 mean "same as `n_ctx`" — override for long context.
#[derive(Clone, Debug)]
pub struct LoadParams {
    pub n_gpu_layers: i32,
    pub n_ctx: u32,
    pub n_seq_max: u32,
    /// First N MoE layers' expert tensors stay on CPU. 0 = off.
    pub n_cpu_moe: u32,
    /// `LLAMA_FLASH_ATTN_TYPE_{AUTO,DISABLED,ENABLED}`.
    pub flash_attn: i32,
    /// Logical decode batch. 0 = `n_ctx`.
    pub n_batch: u32,
    /// Physical ubatch. 0 = `n_batch`.
    pub n_ubatch: u32,
    /// 0 = llama.cpp default thread count.
    pub n_threads: i32,
    /// Recurrent-state snapshots per seq (`llama_context_params.n_rs_seq`).
    /// DSV4 cannot `seq_rm` a suffix without at least 1. 0 = library default.
    /// MTP raises this to `--spec-draft-n-max` (llama `need_n_rs_seq`) so a
    /// rejected draft can be `seq_rm`'d instead of copying ~150 MiB of state.
    pub n_rs_seq: u32,
    /// `true` → `LLAMA_LOAD_MODE_MMAP` (llama default). `false` is `--no-mmap`.
    pub use_mmap: bool,
    /// Load MTP / nextn tensors. Implied by `--spec-type draft-mtp`.
    pub load_mtp: bool,
    pub spec: SpecParams,
    /// CLIP / projector GGUF (`--mmproj`). None = text only.
    pub mmproj: Option<PathBuf>,
}

impl Default for LoadParams {
    fn default() -> Self {
        Self {
            n_gpu_layers: 99,
            n_ctx: 256,
            n_seq_max: 1,
            n_cpu_moe: 0,
            flash_attn: LLAMA_FLASH_ATTN_TYPE_AUTO,
            n_batch: 0,
            n_ubatch: 0,
            n_threads: 0,
            // DSV4 suffix rm needs ≥1 snapshot (~12 MiB). Other archs clamp to 0.
            n_rs_seq: 1,
            use_mmap: true,
            load_mtp: false,
            spec: SpecParams::default(),
            mmproj: None,
        }
    }
}

/// RAII owner of `llama_model` + `llama_context`. GPU access is serialized
/// by the caller (P1: one request). `Send` so it can live on a worker thread.
pub struct Model {
    model: *mut llama_model,
    ctx: *mut llama_context,
    ctx_mtp: *mut llama_context,
    vocab: *const llama_vocab,
    n_vocab: i32,
    n_embd: i32,
    n_rs_seq: u32,
    path: PathBuf,
    vision: Option<Vision>,
    spec: SpecRuntime,
}

struct SpecRuntime {
    params: SpecParams,
    ngram: Option<NgramMod>,
    pending_h: Vec<Vec<f32>>,
    verify_h: Vec<Vec<f32>>,
    verify_h_rows: Vec<i32>,
    last_draft: Vec<Option<SpecType>>,
}

impl SpecRuntime {
    fn disabled(n_seq: u32) -> Self {
        let n = n_seq.max(1) as usize;
        Self {
            params: SpecParams::default(),
            ngram: None,
            pending_h: vec![Vec::new(); n],
            verify_h: vec![Vec::new(); n],
            verify_h_rows: vec![0; n],
            last_draft: vec![None; n],
        }
    }
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
        let n_batch = if params.n_batch == 0 {
            n_ctx_seq
        } else {
            params.n_batch.max(1)
        };
        let n_ubatch = if params.n_ubatch == 0 {
            n_batch
        } else {
            params.n_ubatch.max(1).min(n_batch)
        };
        let load_mtp = params.load_mtp || params.spec.wants_mtp();
        // llama-server: cparams.n_rs_seq = speculative.need_n_rs_seq() == n_max
        // when draft-mtp is on. n_rs_seq=1 cannot rewind a 3-token reject, so
        // the scheduler copied ~150 MiB PARTIAL_ONLY state every verify step.
        let n_rs_seq = if params.spec.wants_mtp() {
            params.n_rs_seq.max(params.spec.n_max.max(0) as u32)
        } else {
            params.n_rs_seq
        };
        if n_rs_seq != params.n_rs_seq {
            tracing::info!(
                from = params.n_rs_seq,
                to = n_rs_seq,
                n_max = params.spec.n_max,
                "n_rs_seq raised to spec n_max (llama need_n_rs_seq)"
            );
        }
        tracing::info!(
            path = %path.display(),
            n_ctx_seq,
            n_ctx_total,
            n_seq_max,
            n_batch,
            n_ubatch,
            n_gpu_layers = params.n_gpu_layers,
            n_cpu_moe = params.n_cpu_moe,
            flash_attn = params.flash_attn,
            n_rs_seq,
            use_mmap = params.use_mmap,
            load_mtp,
            spec = ?params.spec.types.iter().map(|t| t.as_str()).collect::<Vec<_>>(),
            mmproj = params.mmproj.as_ref().map(|p| p.display().to_string()),
            "loading GGUF"
        );

        let mut mparams = unsafe { llama_model_default_params() };
        mparams.n_gpu_layers = params.n_gpu_layers;
        mparams.load_mtp = load_mtp;
        mparams.load_mode = if params.use_mmap {
            LLAMA_LOAD_MODE_MMAP
        } else {
            LLAMA_LOAD_MODE_NONE
        };

        // Patterns must stay alive until `llama_model_load_from_file` returns.
        let cpu_moe = CpuMoeOverrides::new(params.n_cpu_moe);
        if let Some(ptr) = cpu_moe.as_ptr() {
            mparams.tensor_buft_overrides = ptr;
        }

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
        cparams.n_batch = n_batch;
        cparams.n_ubatch = n_ubatch;
        cparams.n_seq_max = n_seq_max;
        cparams.flash_attn_type = params.flash_attn;
        if n_rs_seq > 0 {
            cparams.n_rs_seq = n_rs_seq;
        }
        if params.n_threads > 0 {
            cparams.n_threads = params.n_threads;
            cparams.n_threads_batch = params.n_threads;
        }

        let ctx = unsafe { llama_init_from_model(model, cparams) };
        if ctx.is_null() {
            unsafe { llama_model_free(model) };
            return Err(Error::Load {
                path: path.to_path_buf(),
                reason: "llama_init_from_model returned null (VRAM?)".into(),
            });
        }

        let n_embd = unsafe { llama_model_n_embd_out(model) }.max(1);
        let n_nextn = unsafe { llama_model_n_layer_nextn(model) };

        let mut ctx_mtp = ptr::null_mut();
        let mut spec = SpecRuntime::disabled(n_seq_max);
        spec.params = params.spec.clone();
        if spec.params.wants_ngram() {
            spec.ngram = Some(NgramMod::new(24, 4 * 1024 * 1024));
        }
        spec.pending_h = vec![vec![0.0f32; n_embd as usize]; n_seq_max as usize];
        spec.verify_h = vec![Vec::new(); n_seq_max as usize];
        spec.verify_h_rows = vec![0; n_seq_max as usize];
        spec.last_draft = vec![None; n_seq_max as usize];

        if spec.params.wants_mtp() {
            if n_nextn <= 0 {
                tracing::warn!(
                    "--spec-type draft-mtp but GGUF has n_layer_nextn=0; MTP drafts off"
                );
                spec.params.types.retain(|t| *t != SpecType::DraftMtp);
            } else {
                let mut mparams_ctx = unsafe { llama_context_default_params() };
                mparams_ctx.n_ctx = n_ctx_total;
                mparams_ctx.n_batch = n_batch;
                mparams_ctx.n_ubatch = n_ubatch;
                mparams_ctx.n_seq_max = n_seq_max;
                // llama-server `common_base_params_to_speculative`:
                // n_outputs_max = n_parallel. Default 0 (= n_batch) reserves a
                // 2048×vocab logits tensor (~2 GiB) in the MTP compute buffer.
                // First 2048-token spec_process then pool-allocs on top and
                // GGML_ABORTs when VRAM is already 99% (n_parallel=2 on MI50).
                mparams_ctx.n_outputs_max = mtp_n_outputs_max(n_seq_max);
                mparams_ctx.n_outputs_max_per_seq = 1;
                mparams_ctx.flash_attn_type = params.flash_attn;
                mparams_ctx.n_rs_seq = 0;
                mparams_ctx.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
                mparams_ctx.ctx_other = ctx;
                mparams_ctx.type_k = spec.params.cache_type_k as ggml_type;
                mparams_ctx.type_v = spec.params.cache_type_v as ggml_type;
                if params.n_threads > 0 {
                    mparams_ctx.n_threads = params.n_threads;
                    mparams_ctx.n_threads_batch = params.n_threads;
                }
                ctx_mtp = unsafe { llama_init_from_model(model, mparams_ctx) };
                if ctx_mtp.is_null() {
                    tracing::warn!("failed to create MTP context; draft-mtp disabled");
                    spec.params.types.retain(|t| *t != SpecType::DraftMtp);
                } else {
                    unsafe {
                        golbang_llama_set_embeddings_nextn(ctx, true, false);
                        golbang_llama_set_embeddings_nextn(ctx_mtp, true, true);
                    }
                    tracing::info!(
                        n_nextn,
                        n_embd,
                        n_max = spec.params.n_max,
                        p_min = spec.params.p_min,
                        n_outputs_max = mtp_n_outputs_max(n_seq_max),
                        "MTP draft context ready"
                    );
                }
            }
        }

        let vision = match &params.mmproj {
            Some(mm) => match Vision::load(mm, model, params.n_threads, params.flash_attn) {
                Ok(v) => Some(v),
                Err(e) => {
                    unsafe {
                        if !ctx_mtp.is_null() {
                            llama_free(ctx_mtp);
                        }
                        llama_free(ctx);
                        llama_model_free(model);
                    }
                    return Err(e);
                }
            },
            None => None,
        };

        tracing::info!(
            n_ctx = unsafe { llama_n_ctx(ctx) },
            n_ctx_seq = unsafe { llama_n_ctx_seq(ctx) },
            n_seq_max = unsafe { llama_n_seq_max(ctx) },
            n_batch = unsafe { llama_n_batch(ctx) },
            n_ubatch = unsafe { llama_n_ubatch(ctx) },
            n_vocab,
            n_layer = unsafe { llama_model_n_layer(model) },
            n_nextn,
            mtp = !ctx_mtp.is_null(),
            vision = vision.is_some(),
            "model ready"
        );

        Ok(Self {
            model,
            ctx,
            ctx_mtp,
            vocab,
            n_vocab,
            n_embd,
            n_rs_seq,
            path: path.to_path_buf(),
            vision,
            spec,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn n_vocab(&self) -> i32 {
        self.n_vocab
    }

    pub fn spec_enabled(&self) -> bool {
        self.spec.params.enabled()
    }

    pub fn spec_n_max(&self) -> i32 {
        self.spec.params.verify_n_max()
    }

    pub fn n_rs_seq(&self) -> u32 {
        self.n_rs_seq
    }

    pub fn vision_enabled(&self) -> bool {
        self.vision.is_some()
    }

    /// GGUF `tokenizer.chat_template`, if present.
    pub fn chat_template(&self) -> Option<String> {
        unsafe {
            let p = llama_model_chat_template(self.model, ptr::null());
            if !p.is_null() {
                let s = CStr::from_ptr(p).to_string_lossy();
                if !s.is_empty() {
                    return Some(s.into_owned());
                }
            }
            let key = CString::new("tokenizer.chat_template").ok()?;
            let mut buf = vec![0u8; 32 * 1024];
            let mut n = llama_model_meta_val_str(
                self.model,
                key.as_ptr(),
                buf.as_mut_ptr() as *mut c_char,
                buf.len(),
            );
            if n < 0 {
                let need = (-n) as usize + 1;
                buf.resize(need, 0);
                n = llama_model_meta_val_str(
                    self.model,
                    key.as_ptr(),
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len(),
                );
            }
            if n > 0 {
                Some(String::from_utf8_lossy(&buf[..n as usize]).into_owned())
            } else {
                None
            }
        }
    }

    /// BOS piece with special tokens unparsed. Empty if the vocab has no BOS.
    pub fn bos_token_str(&self) -> String {
        let tok = unsafe { llama_vocab_bos(self.vocab) };
        if tok < 0 {
            return String::new();
        }
        match self.tokenizer().token_to_piece(tok, true) {
            Ok(b) => String::from_utf8_lossy(&b).into_owned(),
            Err(_) => String::new(),
        }
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

    pub fn n_ubatch(&self) -> u32 {
        unsafe { llama_n_ubatch(self.ctx) }
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

    /// Drop KV cells at positions `[p0, inf)` for `seq_id`. False if llama.cpp
    /// refused a partial remove — caller should restore a prefix checkpoint
    /// or `clear_seq` and full-prefill.
    pub fn rm_seq_from(&mut self, seq_id: i32, p0: i32) -> bool {
        unsafe {
            let mem = llama_get_memory(self.ctx);
            if mem.is_null() {
                return false;
            }
            llama_memory_seq_rm(mem, seq_id, p0, -1)
        }
    }

    /// Snapshot one sequence (PARTIAL_ONLY: raw KV + DSV4 recurrent state).
    pub fn seq_state_get(&mut self, seq_id: i32) -> Option<Vec<u8>> {
        let flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        let n = unsafe { llama_state_seq_get_size_ext(self.ctx, seq_id, flags) };
        if n == 0 {
            return None;
        }
        let mut buf = vec![0u8; n];
        let wrote =
            unsafe { llama_state_seq_get_data_ext(self.ctx, buf.as_mut_ptr(), n, seq_id, flags) };
        if wrote == 0 {
            return None;
        }
        buf.truncate(wrote);
        Some(buf)
    }

    /// Restore a snapshot from [`seq_state_get`]. PARTIAL_ONLY skips ISWA
    /// non-SWA base — caller must `rm_seq_from(seq, ckpt_n)` afterwards.
    pub fn seq_state_set(&mut self, seq_id: i32, data: &[u8]) -> bool {
        if data.is_empty() {
            return false;
        }
        let flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        let n = unsafe {
            llama_state_seq_set_data_ext(self.ctx, data.as_ptr(), data.len(), seq_id, flags)
        };
        n > 0
    }

    /// Tokens already in `seq_id` (0 if empty).
    pub fn n_past_seq(&self, seq_id: i32) -> u32 {
        unsafe {
            let mem = llama_get_memory(self.ctx);
            if mem.is_null() {
                return 0;
            }
            let max = llama_memory_seq_pos_max(mem, seq_id);
            if max < 0 { 0 } else { (max + 1) as u32 }
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

    pub fn spec_begin(&mut self, seq_id: i32, prompt: &[Token]) {
        // llama.cpp `common_speculative_begin` only seeds ngram. Do not wipe
        // `pending_h` — `spec_process` already filled it during prefill.
        if let Some(ng) = self.spec.ngram.as_mut() {
            ng.begin(seq_id, prompt);
        }
    }

    pub fn spec_reset_seq(&mut self, seq_id: i32) {
        self.spec_reset_hidden(seq_id);
        self.spec_clear_mtp_kv(seq_id);
        if let Some(slot) = self.spec.last_draft.get_mut(seq_id.max(0) as usize) {
            *slot = None;
        }
    }

    fn spec_reset_hidden(&mut self, seq_id: i32) {
        let i = seq_id as usize;
        if let Some(h) = self.spec.pending_h.get_mut(i) {
            h.fill(0.0);
        }
        if let Some(r) = self.spec.verify_h_rows.get_mut(i) {
            *r = 0;
        }
        if let Some(h) = self.spec.verify_h.get_mut(i) {
            h.clear();
        }
    }

    fn spec_clear_mtp_kv(&mut self, seq_id: i32) {
        if self.ctx_mtp.is_null() {
            return;
        }
        unsafe {
            let mem = llama_get_memory(self.ctx_mtp);
            if !mem.is_null() {
                llama_memory_seq_rm(mem, seq_id, -1, -1);
            }
        }
    }

    fn mtp_pos_max(&self, seq_id: i32) -> i32 {
        if self.ctx_mtp.is_null() {
            return -1;
        }
        unsafe {
            let mem = llama_get_memory(self.ctx_mtp);
            if mem.is_null() {
                return -1;
            }
            llama_memory_seq_pos_max(mem, seq_id)
        }
    }

    /// Drop MTP cells at `p0..` so the next process/draft can write those
    /// positions. Qwen35 is M-RoPE: a leftover draft cell makes `X >= Y`
    /// and `llama_decode` returns -1.
    ///
    /// llama-server does the same right after `common_speculative_draft`:
    /// `llama_memory_seq_rm(ctx_dft, seq, ckpt.pos_max + 1, -1)`.
    fn spec_rewind_mtp_kv(&mut self, seq_id: i32, p0: i32) {
        let max = self.mtp_pos_max(seq_id);
        if max < p0 {
            return;
        }
        if self.spec_rm_from(seq_id, p0) && self.mtp_pos_max(seq_id) < p0 {
            return;
        }
        tracing::warn!(
            seq_id,
            p0,
            max,
            "MTP draft KV rewind failed; clearing MTP seq"
        );
        self.spec_clear_mtp_kv(seq_id);
    }

    fn spec_prepare_process(&mut self, items: &[crate::batch::BatchToken]) {
        let n_seq = self.spec.pending_h.len();
        let mut first: Vec<Option<i32>> = vec![None; n_seq];
        for it in items {
            let sid = it.seq_id as usize;
            if sid >= n_seq {
                continue;
            }
            first[sid] = Some(first[sid].map_or(it.pos, |p| p.min(it.pos)));
        }
        for (sid, pos) in first.into_iter().enumerate() {
            if let Some(p0) = pos {
                self.spec_rewind_mtp_kv(sid as i32, p0);
            }
        }
    }

    /// After a target decode of `items`, catch MTP up (qwen35 single-head path).
    pub fn spec_process(&mut self, items: &[crate::batch::BatchToken]) -> Result<()> {
        if self.ctx_mtp.is_null() || items.is_empty() {
            return Ok(());
        }
        if items.iter().any(|it| it.token < 0) {
            return Ok(());
        }
        self.spec_prepare_process(items);
        let n_embd = self.n_embd as usize;
        let n = items.len();
        let mut h_tgt = vec![0.0f32; n * n_embd];
        for i in 0..n {
            let p = unsafe { golbang_llama_get_embeddings_nextn_ith(self.ctx, i as i32) };
            if p.is_null() {
                return Err(Error::Null("golbang_llama_get_embeddings_nextn_ith"));
            }
            let row = unsafe { std::slice::from_raw_parts(p, n_embd) };
            h_tgt[i * n_embd..(i + 1) * n_embd].copy_from_slice(row);
        }

        let mut embd = vec![0.0f32; n * n_embd];
        if n > 1 {
            embd[n_embd..].copy_from_slice(&h_tgt[..(n - 1) * n_embd]);
        }
        let mut seen = vec![false; self.spec.pending_h.len()];
        for (i, it) in items.iter().enumerate() {
            let sid = it.seq_id as usize;
            if sid < seen.len() && !seen[sid] {
                seen[sid] = true;
                if let Some(h) = self.spec.pending_h.get(sid) {
                    if h.len() == n_embd {
                        embd[i * n_embd..(i + 1) * n_embd].copy_from_slice(h);
                    }
                }
            }
        }

        // llama-server process() adds tokens with logits=0. Catch-up only
        // writes MTP KV; nextn/verify_h come from the target. Passing the
        // target's logits flags through would request 2048 outputs and
        // overflow n_outputs_max = n_seq_max.
        self.decode_mtp(&mtp_process_items(items), &embd)?;

        for sid in 0..self.spec.pending_h.len() {
            let idxs: Vec<usize> = items
                .iter()
                .enumerate()
                .filter(|(_, it)| it.seq_id == sid as i32)
                .map(|(i, _)| i)
                .collect();
            if idxs.is_empty() {
                continue;
            }
            let n_rows = idxs.len();
            let mut vh = vec![0.0f32; n_rows * n_embd];
            for (r, &bi) in idxs.iter().enumerate() {
                vh[r * n_embd..(r + 1) * n_embd]
                    .copy_from_slice(&h_tgt[bi * n_embd..(bi + 1) * n_embd]);
            }
            if let Some(last) = idxs.last() {
                self.spec.pending_h[sid] = h_tgt[last * n_embd..(last + 1) * n_embd].to_vec();
            }
            self.spec.verify_h[sid] = vh;
            self.spec.verify_h_rows[sid] = n_rows as i32;
        }
        Ok(())
    }

    pub fn spec_accept(&mut self, seq_id: i32, n_accepted: u16) {
        let i = seq_id as usize;
        let last = self.spec.last_draft.get(i).copied().flatten();
        if last == Some(SpecType::NgramMod) {
            if let Some(ng) = self.spec.ngram.as_mut() {
                ng.accept(seq_id, n_accepted);
            }
        }
        if let Some(slot) = self.spec.last_draft.get_mut(i) {
            *slot = None;
        }
        let n_rows = self.spec.verify_h_rows.get(i).copied().unwrap_or(0);
        if n_rows <= 0 {
            return;
        }
        let n_embd = self.n_embd as usize;
        let i_h = (n_accepted as i32).min(n_rows - 1) as usize;
        if let Some(vh) = self.spec.verify_h.get(i) {
            if vh.len() >= (i_h + 1) * n_embd {
                if let Some(dst) = self.spec.pending_h.get_mut(i) {
                    dst.clear();
                    dst.extend_from_slice(&vh[i_h * n_embd..(i_h + 1) * n_embd]);
                }
            }
        }
    }

    pub fn spec_draft(
        &mut self,
        seq_id: i32,
        prompt: &[Token],
        id_last: Token,
        n_past: i32,
        n_max: i32,
    ) -> Vec<Token> {
        let n_budget = n_max.max(0) as usize;
        if n_budget == 0 {
            return Vec::new();
        }
        // Keep the ngram table current even when MTP wins this step, so a later
        // MTP miss can still draft a long self-match (llama only ingests inside
        // ngram `draft_one`, which never runs if MTP filled the slot).
        if let Some(ng) = self.spec.ngram.as_mut() {
            ng.ingest_new(seq_id, prompt);
        }
        if let Some(slot) = self.spec.last_draft.get_mut(seq_id.max(0) as usize) {
            *slot = None;
        }
        // llama `common_speculative_draft`: first non-empty impl in --spec-type
        // order wins. Production is `draft-mtp,ngram-mod` (MTP then ngram).
        let types = self.spec.params.types.clone();
        for t in types {
            let drafted = match t {
                SpecType::NgramMod => self.draft_ngram(seq_id, prompt, id_last, n_budget),
                SpecType::DraftMtp => self.draft_mtp(seq_id, id_last, n_past, n_budget),
            };
            if !drafted.is_empty() {
                if let Some(slot) = self.spec.last_draft.get_mut(seq_id.max(0) as usize) {
                    *slot = Some(t);
                }
                return drafted;
            }
        }
        Vec::new()
    }

    fn draft_ngram(
        &mut self,
        seq_id: i32,
        prompt: &[Token],
        id_last: Token,
        n_budget: usize,
    ) -> Vec<Token> {
        let Some(ng) = self.spec.ngram.as_mut() else {
            return Vec::new();
        };
        let n_max = (self.spec.params.ngram_n_max.max(0) as usize).min(n_budget);
        let n_min = self.spec.params.ngram_n_min.max(0) as usize;
        let drafted = ng.draft(prompt, id_last, n_max, n_min);
        ng.note_draft(seq_id, drafted.len());
        drafted
    }

    fn draft_mtp(
        &mut self,
        seq_id: i32,
        id_last: Token,
        n_past: i32,
        n_budget: usize,
    ) -> Vec<Token> {
        if self.ctx_mtp.is_null() {
            return Vec::new();
        }
        let n_max = (self.spec.params.n_max.max(0) as usize).min(n_budget);
        if n_max == 0 {
            return Vec::new();
        }

        let n_embd = self.n_embd as usize;
        let sid = seq_id as usize;
        let Some(mut h) = self.spec.pending_h.get(sid).cloned() else {
            return Vec::new();
        };
        if h.len() != n_embd {
            h = vec![0.0; n_embd];
        }

        let p_min = self.spec.params.p_min;
        let mut drafted = Vec::new();
        let mut tok = id_last;
        let mut pos = n_past;
        let mut embd = h;

        for _step in 0..n_max {
            let item = crate::batch::BatchToken {
                token: tok,
                pos,
                seq_id,
                logits: true,
            };
            if let Err(e) = self.decode_mtp(&[item], &embd) {
                tracing::warn!(error = %e, "MTP draft decode failed");
                break;
            }
            let logits = match self.logits_ith_ctx(self.ctx_mtp, -1) {
                Ok(l) => l,
                Err(e) => {
                    tracing::warn!(error = %e, "MTP draft logits missing");
                    break;
                }
            };
            let (id, p) = topk_mode(logits, 10);
            if p < p_min {
                break;
            }
            drafted.push(id);

            let hp = unsafe { golbang_llama_get_embeddings_nextn_ith(self.ctx_mtp, -1) };
            if hp.is_null() {
                break;
            }
            embd = unsafe { std::slice::from_raw_parts(hp, n_embd) }.to_vec();
            tok = id;
            pos += 1;
        }
        // Draft cells are only used to pick token ids. The verify `process()`
        // must rewrite those positions; leftover cells fail M-RoPE (X < Y).
        self.spec_rewind_mtp_kv(seq_id, n_past);
        drafted
    }

    pub fn spec_rm_from(&mut self, seq_id: i32, p0: i32) -> bool {
        if self.ctx_mtp.is_null() {
            return true;
        }
        unsafe {
            let mem = llama_get_memory(self.ctx_mtp);
            if mem.is_null() {
                return false;
            }
            llama_memory_seq_rm(mem, seq_id, p0, -1)
        }
    }

    pub fn spec_state_get(&mut self, seq_id: i32) -> Option<Vec<u8>> {
        if self.ctx_mtp.is_null() {
            return None;
        }
        let flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        let n = unsafe { llama_state_seq_get_size_ext(self.ctx_mtp, seq_id, flags) };
        if n == 0 {
            return None;
        }
        let mut buf = vec![0u8; n];
        let wrote = unsafe {
            llama_state_seq_get_data_ext(self.ctx_mtp, buf.as_mut_ptr(), n, seq_id, flags)
        };
        if wrote == 0 {
            return None;
        }
        buf.truncate(wrote);
        Some(buf)
    }

    pub fn spec_state_set(&mut self, seq_id: i32, data: &[u8]) -> bool {
        if self.ctx_mtp.is_null() || data.is_empty() {
            return false;
        }
        let flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        let n = unsafe {
            llama_state_seq_set_data_ext(self.ctx_mtp, data.as_ptr(), data.len(), seq_id, flags)
        };
        n > 0
    }

    pub fn vision_eval(&mut self, seq_id: i32, prompt: &str, images: &[Vec<u8>]) -> Result<u32> {
        let n_batch = self.n_batch() as i32;
        let ctx = self.ctx;
        let vision = self.vision.as_mut().ok_or(Error::VisionDisabled)?;
        vision.eval_prompt(ctx, prompt, images, seq_id, n_batch)
    }

    pub fn vision_tokenize(&mut self, prompt: &str, images: &[Vec<u8>]) -> Result<TokenizedVision> {
        let vision = self.vision.as_mut().ok_or(Error::VisionDisabled)?;
        vision.tokenize(prompt, images)
    }

    pub fn vision_eval_from(
        &mut self,
        seq_id: i32,
        tok: &TokenizedVision,
        skip_tokens: usize,
        n_past: u32,
    ) -> Result<u32> {
        let n_batch = self.n_batch() as i32;
        let ctx = self.ctx;
        let vision = self.vision.as_mut().ok_or(Error::VisionDisabled)?;
        vision.eval_from(ctx, tok, seq_id, n_batch, skip_tokens, n_past)
    }

    fn decode_mtp(&mut self, items: &[crate::batch::BatchToken], embd: &[f32]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let n_embd = self.n_embd as usize;
        if embd.len() != items.len() * n_embd {
            return Err(Error::Decode(-2));
        }
        let n = items.len() as i32;
        let mut batch = unsafe { llama_batch_init(n, n_embd as i32, 1) };
        if batch.embd.is_null() || batch.pos.is_null() || batch.seq_id.is_null() {
            unsafe { llama_batch_free(batch) };
            return Err(Error::Null("llama_batch_init mtp"));
        }
        let mut token_buf: Vec<Token> = items.iter().map(|it| it.token).collect();
        batch.token = token_buf.as_mut_ptr();
        unsafe {
            std::ptr::copy_nonoverlapping(embd.as_ptr(), batch.embd, embd.len());
        }
        for (i, it) in items.iter().enumerate() {
            unsafe {
                *batch.pos.add(i) = it.pos;
                *batch.n_seq_id.add(i) = 1;
                let seqs = *batch.seq_id.add(i);
                if seqs.is_null() {
                    batch.token = ptr::null_mut();
                    llama_batch_free(batch);
                    return Err(Error::Null("llama_batch.seq_id mtp"));
                }
                *seqs.add(0) = it.seq_id;
                *batch.logits.add(i) = i8::from(it.logits);
            }
        }
        batch.n_tokens = n;
        let rc = unsafe { llama_decode(self.ctx_mtp, batch) };
        batch.token = ptr::null_mut();
        unsafe { llama_batch_free(batch) };
        if rc != 0 {
            return Err(Error::Decode(rc));
        }
        Ok(())
    }

    fn logits_ith_ctx(&self, ctx: *mut llama_context, i: i32) -> Result<&[f32]> {
        unsafe {
            let ptr = llama_get_logits_ith(ctx, i);
            if ptr.is_null() {
                return Err(Error::Null("llama_get_logits_ith"));
            }
            Ok(std::slice::from_raw_parts(ptr, self.n_vocab as usize))
        }
    }

    pub fn last_logits_vec(&self) -> Result<Vec<f32>> {
        Ok(self.logits()?.to_vec())
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        self.vision = None;
        unsafe {
            if !self.ctx_mtp.is_null() {
                llama_free(self.ctx_mtp);
                self.ctx_mtp = ptr::null_mut();
            }
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

/// llama-server `common_base_params_to_speculative`: `n_outputs_max = n_parallel`.
fn mtp_n_outputs_max(n_seq_max: u32) -> u32 {
    n_seq_max.max(1)
}

/// Strip logits so MTP process() stays within `n_outputs_max = n_seq_max`.
fn mtp_process_items(items: &[crate::batch::BatchToken]) -> Vec<crate::batch::BatchToken> {
    items
        .iter()
        .copied()
        .map(|it| crate::batch::BatchToken {
            logits: false,
            ..it
        })
        .collect()
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

unsafe extern "C" fn forward_llama_log(
    level: ggml_log_level,
    text: *const c_char,
    _ud: *mut c_void,
) {
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
        GGML_LOG_LEVEL_INFO => tracing::info!(target: "llama", "{trimmed}"),
        _ => tracing::debug!(target: "llama", "{trimmed}"),
    }
}

/// NULL-terminated `tensor_buft_overrides` that pin MoE expert tensors to CPU.
/// Owns the CStrings so the pointers stay valid through `llama_model_load_from_file`.
struct CpuMoeOverrides {
    _patterns: Vec<CString>,
    overrides: Vec<llama_model_tensor_buft_override>,
}

impl CpuMoeOverrides {
    fn new(n_cpu_moe: u32) -> Self {
        if n_cpu_moe == 0 {
            return Self {
                _patterns: Vec::new(),
                overrides: Vec::new(),
            };
        }
        let cpu_buft = unsafe { ggml_backend_cpu_buffer_type() };
        let mut patterns = Vec::with_capacity(n_cpu_moe as usize);
        let mut overrides = Vec::with_capacity(n_cpu_moe as usize + 1);
        for i in 0..n_cpu_moe {
            let pat = CString::new(format!("blk\\.{i}{FFN_EXPS_REGEX}"))
                .expect("MoE override pattern is ASCII");
            overrides.push(llama_model_tensor_buft_override {
                pattern: pat.as_ptr(),
                buft: cpu_buft,
            });
            patterns.push(pat);
        }
        overrides.push(llama_model_tensor_buft_override {
            pattern: ptr::null(),
            buft: ptr::null_mut(),
        });
        Self {
            _patterns: patterns,
            overrides,
        }
    }

    fn as_ptr(&self) -> Option<*const llama_model_tensor_buft_override> {
        if self.overrides.is_empty() {
            None
        } else {
            Some(self.overrides.as_ptr())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{mtp_n_outputs_max, mtp_process_items, FFN_EXPS_REGEX};
    use crate::batch::BatchToken;

    #[test]
    fn mtp_outputs_match_llama_server_n_parallel() {
        assert_eq!(mtp_n_outputs_max(0), 1);
        assert_eq!(mtp_n_outputs_max(1), 1);
        assert_eq!(mtp_n_outputs_max(2), 2);
    }

    #[test]
    fn mtp_process_drops_target_logits_flags() {
        let items = [
            BatchToken {
                token: 1,
                pos: 10,
                seq_id: 0,
                logits: true,
            },
            BatchToken {
                token: 2,
                pos: 11,
                seq_id: 0,
                logits: true,
            },
        ];
        let out = mtp_process_items(&items);
        assert_eq!(out.len(), 2);
        assert!(!out[0].logits && !out[1].logits);
        assert_eq!(out[0].token, 1);
        assert_eq!(out[0].pos, 10);
        assert_eq!(out[1].seq_id, 0);
    }

    #[test]
    fn cpu_moe_block_regex_matches_llama_cpp() {
        assert_eq!(
            format!("blk\\.{i}{FFN_EXPS_REGEX}", i = 0),
            r"blk\.0\.ffn_(up|down|gate|gate_up)_(ch|)exps"
        );
        assert_eq!(
            format!("blk\\.{i}{FFN_EXPS_REGEX}", i = 31),
            r"blk\.31\.ffn_(up|down|gate|gate_up)_(ch|)exps"
        );
    }
}
