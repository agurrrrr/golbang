//! Low-level FFI for the SHA-pinned `llama.h` (P0).
//!
//! Safe wrappers live in `golbang-core` (P1). This crate only exposes the C ABI
//! generated from `/home/agurrrrr/code/local-llm/llama.cpp` @
//! `5b474eb69dac2d7c26ba8855310d3b60e02a5c4f`.
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

/// llama.cpp commit bindgen and the linked `.so` were verified against.
pub const LLAMA_CPP_SHA: &str = env!("GOLBANG_LLAMA_SHA");

/// `build/bin` of the SHA-pinned tree — pass to [`ggml_backend_load_all_from_path`].
pub const LLAMA_BIN_DIR: &str = env!("GOLBANG_LLAMA_BIN");

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
        assert_eq!(LLAMA_CPP_SHA, "5b474eb69dac2d7c26ba8855310d3b60e02a5c4f");
    }
}
