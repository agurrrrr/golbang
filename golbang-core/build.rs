//! `cargo:rustc-link-arg` from golbang-sys does not apply to this crate's
//! binaries/tests. Repeat the SHA-pinned rpath so `libllama.so` resolves.

fn main() {
    let bin = std::env::var("DEP_LLAMA_BIN").unwrap_or_else(|_| {
        let root = std::env::var("GOLBANG_LLAMA_DIR")
            .unwrap_or_else(|_| "/home/agurrrrr/code/local-llm/llama.cpp".into());
        format!("{root}/build/bin")
    });
    println!("cargo:rerun-if-env-changed=GOLBANG_LLAMA_DIR");
    println!("cargo:rerun-if-env-changed=DEP_LLAMA_BIN");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{bin}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/rocm/lib");
}
