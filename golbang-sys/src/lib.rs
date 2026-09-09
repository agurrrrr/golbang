//! Low-level FFI for the SHA-pinned `llama.h` (P0).
//!
//! Safe wrappers live in `golbang-core` (P1). This crate only exposes the C ABI
//! generated from the SHA-pinned llama.cpp tree selected by `GOLBANG_GPU`
//! (`hip` → `llama.cpp-glm5next` / `cuda` → `llama.cpp-escha`).
//!
//! Current names: [`llama_model_load_from_file`], [`llama_init_from_model`],
//! [`llama_model_free`]. The older `llama_load_model_from_file` /
//! `llama_new_context_with_model` / `llama_free_model` symbols are DEPRECATED.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]
#![allow(clippy::all)]
#![allow(improper_ctypes)]
#![allow(unnecessary_transmutes)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

unsafe extern "C" {
    pub fn golbang_llama_set_embeddings_nextn(ctx: *mut llama_context, value: bool, masked: bool);
    pub fn golbang_llama_get_embeddings_nextn(ctx: *mut llama_context) -> *mut f32;
    pub fn golbang_llama_get_embeddings_nextn_ith(ctx: *mut llama_context, i: i32) -> *mut f32;
    pub fn golbang_llama_set_nextn_layer_offset(ctx: *mut llama_context, offset: i32);
    pub fn golbang_llama_get_ctx_other(ctx: *mut llama_context) -> *mut llama_context;
}

/// `mtmd_helper_bitmap_init_from_buf` grew an `mtmd_helper_init_opt` argument
/// (video support) in newer mtmd-helper.h. The cfg mirrors the pinned header
/// (set by this crate's build.rs) so dependents compile against either pin
/// through this wrapper instead of the raw binding.
pub unsafe fn mtmd_helper_bitmap_init_from_buf_compat(
    ctx: *mut mtmd_context,
    buf: *const u8,
    len: usize,
    placeholder: bool,
) -> mtmd_helper_bitmap_wrapper {
    #[cfg(mtmd_helper_init_opt)]
    return unsafe {
        mtmd_helper_bitmap_init_from_buf(ctx, buf, len, placeholder, mtmd_helper_init_opt_default())
    };
    #[cfg(not(mtmd_helper_init_opt))]
    return unsafe { mtmd_helper_bitmap_init_from_buf(ctx, buf, len, placeholder) };
}

/// llama.cpp commit bindgen and the linked `.so` were verified against.
pub const LLAMA_CPP_SHA: &str = env!("GOLBANG_LLAMA_SHA");

/// `build/bin` of the SHA-pinned tree — pass to [`ggml_backend_load_all_from_path`].
pub const LLAMA_BIN_DIR: &str = env!("GOLBANG_LLAMA_BIN");

/// Selected backend at build time: `"hip"` or `"cuda"`.
pub const GOLBANG_GPU: &str = env!("GOLBANG_GPU");

#[cfg(test)]
mod api_names {
    use super::*;

    #[test]
    fn current_load_api_symbols_are_bound() {
        let _load = llama_model_load_from_file;
        let _init = llama_init_from_model;
        let _free = llama_model_free;
        let _decode = llama_decode;
        let _tok = llama_tokenize;
        let _logits = llama_get_logits;
        let _n_layer = llama_model_n_layer;
        let _n_vocab = llama_vocab_n_tokens;
        let _n_nextn = llama_model_n_layer_nextn;
        let _set_h = golbang_llama_set_embeddings_nextn;
        let _get_h = golbang_llama_get_embeddings_nextn;
        let _mtmd = mtmd_init_from_file;
        let _bmp = mtmd_helper_bitmap_init_from_buf;
        let _rpc = llama_supports_rpc;
        let _rpc_proc = ggml_backend_reg_get_proc_address;
        let _rpc_reg = ggml_backend_register;
        let gpu = std::env::var("GOLBANG_GPU").unwrap_or_else(|_| "hip".to_string());
        let expected = if gpu.trim().eq_ignore_ascii_case("cuda") {
            "c5d759c8a9e02653e9acd2442599b4c8eccc5ba5"
        } else {
            "367ebbc20c2b20db411d5acf72b88d26a7c13d70"
        };
        assert_eq!(LLAMA_CPP_SHA, expected);
    }
}
