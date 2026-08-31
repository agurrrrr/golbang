//! `--mmproj` / libmtmd wrapper. Images are raw file bytes (jpeg/png/…);
//! `mtmd_helper_bitmap_init_from_buf` decodes them.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;
use std::ptr;

use golbang_sys::*;

use crate::error::{Error, Result};
use crate::prefix_cache::{VisionChunk, VisionSeq};
use crate::tokenizer::Token;

/// Same default as `mtmd_default_marker()`. Insert this where an image was.
pub const MEDIA_MARKER: &str = "<__media__>";

pub struct Vision {
    ctx: *mut mtmd_context,
}

/// Owned `mtmd_tokenize` output. Chunks stay alive until eval (or drop).
pub struct TokenizedVision {
    pub seq: VisionSeq,
    chunks: *mut mtmd_input_chunks,
}

impl Drop for TokenizedVision {
    fn drop(&mut self) {
        if !self.chunks.is_null() {
            unsafe { mtmd_input_chunks_free(self.chunks) };
            self.chunks = ptr::null_mut();
        }
    }
}

unsafe impl Send for Vision {}

impl Vision {
    pub fn load(
        mmproj: &Path,
        text_model: *const llama_model,
        n_threads: i32,
        flash_attn: i32,
    ) -> Result<Self> {
        if !mmproj.is_file() {
            return Err(Error::Vision(format!(
                "mmproj is not a file: {}",
                mmproj.display()
            )));
        }
        let c_path = CString::new(mmproj.to_string_lossy().as_bytes())
            .map_err(|_| Error::Vision("mmproj path contains interior NUL".into()))?;
        let mut params = unsafe { mtmd_context_params_default() };
        params.use_gpu = true;
        params.print_timings = false;
        params.n_threads = if n_threads > 0 { n_threads } else { 4 };
        params.flash_attn_type = flash_attn;
        params.warmup = false;
        let marker = CString::new(MEDIA_MARKER).expect("marker is ASCII");
        params.media_marker = marker.as_ptr();

        let ctx = unsafe { mtmd_init_from_file(c_path.as_ptr(), text_model, params) };
        if ctx.is_null() {
            return Err(Error::Vision(format!(
                "mtmd_init_from_file failed: {}",
                mmproj.display()
            )));
        }
        let vision = unsafe { mtmd_support_vision(ctx) };
        if !vision {
            unsafe { mtmd_free(ctx) };
            return Err(Error::Vision(
                "mmproj does not advertise vision support".into(),
            ));
        }
        tracing::info!(path = %mmproj.display(), "mmproj ready");
        Ok(Self { ctx })
    }

    /// Encode images and decode text+image chunks into `lctx` on `seq_id`.
    /// Returns the new n_past (token positions, including image slots).
    pub fn eval_prompt(
        &mut self,
        lctx: *mut llama_context,
        prompt: &str,
        images: &[Vec<u8>],
        seq_id: i32,
        n_batch: i32,
    ) -> Result<u32> {
        let tok = self.tokenize(prompt, images)?;
        self.eval_from(lctx, &tok, seq_id, n_batch, 0, 0)
    }

    /// Tokenize only. CLIP encode happens later in [`Self::eval_from`].
    pub fn tokenize(&mut self, prompt: &str, images: &[Vec<u8>]) -> Result<TokenizedVision> {
        if images.is_empty() {
            return Err(Error::Vision("eval_prompt called with no images".into()));
        }
        if !prompt.contains(MEDIA_MARKER) {
            return Err(Error::Vision(format!(
                "prompt has {} image(s) but no {MEDIA_MARKER} marker",
                images.len()
            )));
        }

        let c_prompt = CString::new(prompt)
            .map_err(|_| Error::Vision("prompt contains interior NUL".into()))?;
        let text = mtmd_input_text {
            text: c_prompt.as_ptr(),
            text_len: prompt.len(),
            add_special: true,
            parse_special: true,
        };

        let mut bitmaps: Vec<*mut mtmd_bitmap> = Vec::with_capacity(images.len());
        for (i, bytes) in images.iter().enumerate() {
            if bytes.is_empty() {
                self.free_bitmaps(&bitmaps);
                return Err(Error::Vision(format!("image {i} is empty")));
            }
            // Signature varies across pins (`mtmd_helper_init_opt` arg);
            // the sys-crate wrapper mirrors the pinned header.
            let wrap = unsafe {
                mtmd_helper_bitmap_init_from_buf_compat(
                    self.ctx,
                    bytes.as_ptr(),
                    bytes.len(),
                    false,
                )
            };
            if wrap.bitmap.is_null() {
                self.free_bitmaps(&bitmaps);
                return Err(Error::Vision(format!(
                    "failed to decode image {i} ({} bytes)",
                    bytes.len()
                )));
            }
            bitmaps.push(wrap.bitmap);
        }

        let chunks = unsafe { mtmd_input_chunks_init() };
        if chunks.is_null() {
            self.free_bitmaps(&bitmaps);
            return Err(Error::Null("mtmd_input_chunks_init"));
        }

        let mut bmp_ptrs: Vec<*const mtmd_bitmap> =
            bitmaps.iter().map(|p| *p as *const mtmd_bitmap).collect();
        let rc = unsafe {
            mtmd_tokenize(
                self.ctx,
                chunks,
                &text,
                bmp_ptrs.as_mut_ptr(),
                bmp_ptrs.len(),
            )
        };
        self.free_bitmaps(&bitmaps);
        if rc != 0 {
            unsafe { mtmd_input_chunks_free(chunks) };
            return Err(Error::Vision(format!(
                "mtmd_tokenize returned {rc} (bitmap/marker count mismatch?)"
            )));
        }

        let seq = flatten_chunks(chunks)?;
        Ok(TokenizedVision { seq, chunks })
    }

    /// Eval chunks starting at `skip_tokens` / `n_past`. Cached prefix
    /// (including same-hash images) is not CLIP-encoded again.
    pub fn eval_from(
        &mut self,
        lctx: *mut llama_context,
        tok: &TokenizedVision,
        seq_id: i32,
        n_batch: i32,
        skip_tokens: usize,
        n_past: u32,
    ) -> Result<u32> {
        let n_batch = n_batch.max(1);
        let n_chunks = unsafe { mtmd_input_chunks_size(tok.chunks) };
        let mut tok_i = 0usize;
        let mut pos: llama_pos = n_past as llama_pos;
        let mut eval_chunks = 0u32;

        for ci in 0..n_chunks {
            let chunk = unsafe { mtmd_input_chunks_get(tok.chunks, ci) };
            if chunk.is_null() {
                return Err(Error::Null("mtmd_input_chunks_get"));
            }
            let n_tok = unsafe { mtmd_input_chunk_get_n_tokens(chunk) };
            let logits_last = ci + 1 == n_chunks;
            if tok_i.saturating_add(n_tok) <= skip_tokens {
                tok_i += n_tok;
                continue;
            }
            if tok_i < skip_tokens {
                let skip = skip_tokens - tok_i;
                eval_partial_chunk(lctx, chunk, seq_id, n_batch, skip, logits_last, &mut pos)?;
            } else {
                eval_full_chunk(
                    self.ctx,
                    lctx,
                    chunk,
                    seq_id,
                    n_batch,
                    logits_last,
                    &mut pos,
                )?;
            }
            eval_chunks += 1;
            tok_i += n_tok;
        }

        let n_past_out = pos.max(0) as u32;
        tracing::info!(
            seq = seq_id,
            n_tokens = tok.seq.n_tokens(),
            n_pos = tok.seq.n_pos(),
            skip_tokens,
            reuse_pos = n_past,
            n_past = n_past_out,
            n_images = tok.seq.images.len(),
            eval_chunks,
            "vision prompt evaluated"
        );
        Ok(n_past_out)
    }

    fn free_bitmaps(&self, bitmaps: &[*mut mtmd_bitmap]) {
        for &b in bitmaps {
            if !b.is_null() {
                unsafe { mtmd_bitmap_free(b) };
            }
        }
    }
}

fn flatten_chunks(chunks: *mut mtmd_input_chunks) -> Result<VisionSeq> {
    let n = unsafe { mtmd_input_chunks_size(chunks) };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let chunk = unsafe { mtmd_input_chunks_get(chunks, i) };
        if chunk.is_null() {
            return Err(Error::Null("mtmd_input_chunks_get"));
        }
        let typ = unsafe { mtmd_input_chunk_get_type(chunk) };
        if typ == MTMD_INPUT_CHUNK_TYPE_TEXT {
            let mut n_tok = 0usize;
            let ptr = unsafe { mtmd_input_chunk_get_tokens_text(chunk, &mut n_tok) };
            if n_tok > 0 && ptr.is_null() {
                return Err(Error::Null("mtmd_input_chunk_get_tokens_text"));
            }
            let toks = if n_tok == 0 {
                Vec::new()
            } else {
                unsafe { std::slice::from_raw_parts(ptr, n_tok) }.to_vec()
            };
            out.push(VisionChunk::Text(toks));
        } else {
            let n_tokens = unsafe { mtmd_input_chunk_get_n_tokens(chunk) } as u32;
            let n_pos = unsafe { mtmd_input_chunk_get_n_pos(chunk) }.max(0) as u32;
            out.push(VisionChunk::Media {
                id: cstr_id(unsafe { mtmd_input_chunk_get_id(chunk) }),
                n_tokens,
                n_pos,
            });
        }
    }
    Ok(VisionSeq::from_chunks(out))
}

fn cstr_id(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

fn eval_full_chunk(
    ctx: *mut mtmd_context,
    lctx: *mut llama_context,
    chunk: *const mtmd_input_chunk,
    seq_id: i32,
    n_batch: i32,
    logits_last: bool,
    n_past: &mut llama_pos,
) -> Result<()> {
    let typ = unsafe { mtmd_input_chunk_get_type(chunk) };
    if typ == MTMD_INPUT_CHUNK_TYPE_TEXT {
        let mut n_tok = 0usize;
        let ptr = unsafe { mtmd_input_chunk_get_tokens_text(chunk, &mut n_tok) };
        let toks = if n_tok == 0 {
            &[][..]
        } else {
            if ptr.is_null() {
                return Err(Error::Null("mtmd_input_chunk_get_tokens_text"));
            }
            unsafe { std::slice::from_raw_parts(ptr, n_tok) }
        };
        eval_text_tokens(lctx, toks, seq_id, n_batch, n_past, logits_last)
    } else {
        let rc = unsafe {
            mtmd_helper_eval_chunk_single(
                ctx,
                lctx,
                chunk,
                *n_past,
                seq_id,
                n_batch,
                logits_last,
                n_past,
            )
        };
        if rc != 0 {
            return Err(Error::Vision(format!(
                "mtmd_helper_eval_chunk_single returned {rc}"
            )));
        }
        Ok(())
    }
}

fn eval_partial_chunk(
    lctx: *mut llama_context,
    chunk: *const mtmd_input_chunk,
    seq_id: i32,
    n_batch: i32,
    skip: usize,
    logits_last: bool,
    n_past: &mut llama_pos,
) -> Result<()> {
    let typ = unsafe { mtmd_input_chunk_get_type(chunk) };
    if typ != MTMD_INPUT_CHUNK_TYPE_TEXT {
        return Err(Error::Vision(
            "vision prefix reuse landed mid-image; refusing to split media chunk".into(),
        ));
    }
    let mut n_tok = 0usize;
    let ptr = unsafe { mtmd_input_chunk_get_tokens_text(chunk, &mut n_tok) };
    if skip > n_tok {
        return Err(Error::Vision(format!(
            "vision text skip {skip} > chunk {n_tok}"
        )));
    }
    let toks = if n_tok == 0 {
        &[][..]
    } else {
        if ptr.is_null() {
            return Err(Error::Null("mtmd_input_chunk_get_tokens_text"));
        }
        unsafe { std::slice::from_raw_parts(ptr, n_tok) }
    };
    eval_text_tokens(lctx, &toks[skip..], seq_id, n_batch, n_past, logits_last)
}

fn eval_text_tokens(
    lctx: *mut llama_context,
    tokens: &[Token],
    seq_id: i32,
    n_batch: i32,
    n_past: &mut llama_pos,
    logits_last: bool,
) -> Result<()> {
    if tokens.is_empty() {
        return Ok(());
    }
    let n_batch = n_batch.max(1) as usize;
    let mut batch = unsafe { llama_batch_init(n_batch as i32, 0, 1) };
    if batch.token.is_null() || batch.pos.is_null() || batch.seq_id.is_null() {
        unsafe { llama_batch_free(batch) };
        return Err(Error::Null("llama_batch_init vision text"));
    }
    let mut i = 0;
    while i < tokens.len() {
        let take = (tokens.len() - i).min(n_batch);
        for j in 0..take {
            unsafe {
                *batch.token.add(j) = tokens[i + j];
                *batch.pos.add(j) = *n_past;
                *batch.n_seq_id.add(j) = 1;
                let seqs = *batch.seq_id.add(j);
                if seqs.is_null() {
                    llama_batch_free(batch);
                    return Err(Error::Null("llama_batch.seq_id vision text"));
                }
                *seqs.add(0) = seq_id;
                *batch.logits.add(j) = 0;
            }
            *n_past += 1;
        }
        batch.n_tokens = take as i32;
        i += take;
        if logits_last && i == tokens.len() {
            unsafe {
                *batch.logits.add(take - 1) = 1;
            }
        }
        let rc = unsafe { llama_decode(lctx, batch) };
        if rc != 0 {
            unsafe { llama_batch_free(batch) };
            return Err(Error::Vision(format!("vision text decode {rc}")));
        }
    }
    unsafe { llama_batch_free(batch) };
    Ok(())
}

impl Drop for Vision {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { mtmd_free(self.ctx) };
            self.ctx = ptr::null_mut();
        }
    }
}

/// Decode a data-URL or local path into file bytes. HTTP is left to the caller.
pub fn load_media_bytes(src: &str) -> Result<Vec<u8>> {
    let src = src.trim();
    if src.is_empty() {
        return Err(Error::Vision("empty image url".into()));
    }
    if let Some(rest) = src.strip_prefix("data:") {
        return decode_data_url(rest);
    }
    let path = src.strip_prefix("file://").unwrap_or(src);
    if path.starts_with("http://") || path.starts_with("https://") {
        return Err(Error::Vision(
            "http(s) image_url is not loaded here — send a data: URL or a local path".into(),
        ));
    }
    std::fs::read(path).map_err(|e| Error::Vision(format!("read image {path}: {e}")))
}

fn decode_data_url(rest: &str) -> Result<Vec<u8>> {
    let (_meta, payload) = rest
        .split_once(',')
        .ok_or_else(|| Error::Vision("data: URL missing comma".into()))?;
    decode_base64(payload)
}

fn decode_base64(s: &str) -> Result<Vec<u8>> {
    let s: String = s.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    if s.is_empty() {
        return Err(Error::Vision("empty base64 image".into()));
    }
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    let mut buf = 0u32;
    let mut n = 0u32;
    for &c in bytes {
        if c == b'=' {
            break;
        }
        let v = val(c).ok_or_else(|| Error::Vision("invalid base64 in data: URL".into()))?;
        buf = (buf << 6) | u32::from(v);
        n += 6;
        if n >= 8 {
            n -= 8;
            out.push((buf >> n) as u8);
            buf &= (1 << n) - 1;
        }
    }
    if out.is_empty() {
        return Err(Error::Vision("data: URL decoded to 0 bytes".into()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_url_decodes_png_header() {
        // "hi" as ascii, not a real png — just checks the decoder.
        let bytes = decode_data_url("text/plain;base64,aGk=").unwrap();
        assert_eq!(bytes, b"hi");
    }

    #[test]
    fn data_url_rejects_garbage() {
        assert!(decode_data_url("text/plain;base64,????").is_err());
    }

    #[test]
    fn media_marker_matches_mtmd_default() {
        assert_eq!(MEDIA_MARKER, "<__media__>");
    }
}
