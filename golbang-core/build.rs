//! `cargo:rustc-link-arg` from golbang-sys does not apply to this crate's
//! binaries/tests. Repeat the SHA-pinned rpath so `libllama.so` resolves.
//! GPU-specific search path is chosen by `GOLBANG_GPU` (hip/ds41 → /opt/rocm/lib,
//! cuda/ds41-cuda → /opt/cuda-12.8/.../lib, vulkan → none) so a CUDA/Vulkan
//! build does not pull in ROCm libs.

fn main() {
    let gpu = std::env::var("GOLBANG_GPU").unwrap_or_else(|_| "hip".to_string());
    let gpu = gpu.trim().to_lowercase();
    let bin = std::env::var("DEP_LLAMA_BIN").unwrap_or_else(|_| {
        let fallback = match gpu.as_str() {
            "ds41" | "ds41-cuda" => "/home/agurrrrr/code/local-llm/llama.cpp-ds41",
            _ => "/home/agurrrrr/code/local-llm/llama.cpp",
        };
        let root = std::env::var("GOLBANG_LLAMA_DIR").unwrap_or_else(|_| fallback.into());
        let sub = match gpu.as_str() {
            "vulkan" => "build-vulkan/bin",
            "ds41-cuda" => "build-cuda/bin",
            _ => "build/bin",
        };
        format!("{root}/{sub}")
    });
    let extra: Option<&str> = match gpu.as_str() {
        "cuda" | "ds41-cuda" => Some("/opt/cuda-12.8/targets/x86_64-linux/lib"),
        "vulkan" => None,
        _ => Some("/opt/rocm/lib"),
    };
    println!("cargo:rerun-if-env-changed=GOLBANG_LLAMA_DIR");
    println!("cargo:rerun-if-env-changed=DEP_LLAMA_BIN");
    println!("cargo:rerun-if-env-changed=GOLBANG_GPU");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{bin}");
    if let Some(extra) = extra {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{extra}");
    }
    // Re-export the SHA-pinned bin dir so dependents that don't link
    // golbang-sys directly (e.g. golbang-server) can add the same rpath.
    println!("cargo:bin={bin}");
}
