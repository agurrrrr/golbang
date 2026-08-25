//! `cargo:rustc-link-arg` from golbang-sys does not apply to this crate's
//! binaries/tests. Repeat the SHA-pinned rpath so `libllama.so` resolves.
//! GPU-specific search path is chosen by `GOLBANG_GPU` (hip → /opt/rocm/lib,
//! cuda → /opt/cuda/.../lib) so a CUDA build does not pull in ROCm libs.

fn main() {
    let bin = std::env::var("DEP_LLAMA_BIN").unwrap_or_else(|_| {
        let root = std::env::var("GOLBANG_LLAMA_DIR")
            .unwrap_or_else(|_| "/home/agurrrrr/code/local-llm/llama.cpp".into());
        format!("{root}/build/bin")
    });
    let gpu = std::env::var("GOLBANG_GPU").unwrap_or_else(|_| "hip".to_string());
    let extra = match gpu.as_str() {
        "cuda" => "/opt/cuda/targets/x86_64-linux/lib",
        _ => "/opt/rocm/lib",
    };
    println!("cargo:rerun-if-env-changed=GOLBANG_LLAMA_DIR");
    println!("cargo:rerun-if-env-changed=DEP_LLAMA_BIN");
    println!("cargo:rerun-if-env-changed=GOLBANG_GPU");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{bin}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{extra}");
    // Re-export the SHA-pinned bin dir so dependents that don't link
    // golbang-sys directly (e.g. golbang-server) can add the same rpath.
    println!("cargo:bin={bin}");
}
