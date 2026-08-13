//! Repeat the SHA-pinned rpath. Server depends on golbang-core, not
//! golbang-sys, so DEP_LLAMA_BIN is not visible here.

fn main() {
    let root = std::env::var("GOLBANG_LLAMA_DIR")
        .unwrap_or_else(|_| "/home/agurrrrr/code/local-llm/llama.cpp".into());
    let bin = format!("{root}/build/bin");
    println!("cargo:rerun-if-env-changed=GOLBANG_LLAMA_DIR");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{bin}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/rocm/lib");
}
