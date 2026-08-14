//! `--mmproj` / libmtmd wrapper. Images are raw file bytes (jpeg/png/…);
//! `mtmd_helper_bitmap_init_from_buf` decodes them.

use std::ffi::CString;
use std::path::Path;
use std::ptr;

use golbang_sys::*;

use crate::error::{Error, Result};

/// Same default as `mtmd_default_marker()`. Insert this where an image was.
pub const MEDIA_MARKER: &str = "<__media__>";

pub struct Vision {
    ctx: *mut mtmd_context,
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
            let wrap = unsafe {
                mtmd_helper_bitmap_init_from_buf(self.ctx, bytes.as_ptr(), bytes.len(), false)
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

        let n_tok = unsafe { mtmd_helper_get_n_tokens(chunks) };
        let mut n_past: llama_pos = 0;
        let rc = unsafe {
            mtmd_helper_eval_chunks(
                self.ctx,
                lctx,
                chunks,
                0,
                seq_id,
                n_batch.max(1),
                true,
                &mut n_past,
            )
        };
        unsafe { mtmd_input_chunks_free(chunks) };
        if rc != 0 {
            return Err(Error::Vision(format!(
                "mtmd_helper_eval_chunks returned {rc}"
            )));
        }
        tracing::info!(
            seq = seq_id,
            n_tokens = n_tok,
            n_past,
            n_images = images.len(),
            "vision prompt evaluated"
        );
        Ok(n_past.max(0) as u32)
    }

    fn free_bitmaps(&self, bitmaps: &[*mut mtmd_bitmap]) {
        for &b in bitmaps {
            if !b.is_null() {
                unsafe { mtmd_bitmap_free(b) };
            }
        }
    }
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
