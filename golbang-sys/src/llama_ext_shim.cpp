// C ABI for llama-ext.h staging symbols (C++ linkage in libllama.so).
// Used by draft-mtp: nextn embeddings and the MTP context pointer.

#include "llama.h"
#include "llama-ext.h"

extern "C" {

void golbang_llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked) {
    llama_set_embeddings_nextn(ctx, value, masked);
}

float * golbang_llama_get_embeddings_nextn(struct llama_context * ctx) {
    return llama_get_embeddings_nextn(ctx);
}

float * golbang_llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i) {
    return llama_get_embeddings_nextn_ith(ctx, i);
}

void golbang_llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset) {
    llama_set_nextn_layer_offset(ctx, offset);
}

struct llama_context * golbang_llama_get_ctx_other(struct llama_context * ctx) {
    return llama_get_ctx_other(ctx);
}

} // extern "C"
