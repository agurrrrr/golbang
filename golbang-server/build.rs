//! Repeat the SHA-pinned rpath. Server depends on golbang-core (not
//! golbang-sys directly), so it reads the bin dir that golbang-core's
//! build.rs re-exports as `cargo:bin` (`DEP_GOLBANG_CORE_BIN`), falling back
//! to `GOLBANG_LLAMA_DIR` for standalone builds.

fn main() {
    let gpu = std::env::var("GOLBANG_GPU").unwrap_or_else(|_| "hip".to_string());
    let gpu = gpu.trim().to_lowercase();
    let bin = std::env::var("DEP_GOLBANG_CORE_BIN").unwrap_or_else(|_| {
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
    println!("cargo:rerun-if-env-changed=DEP_GOLBANG_CORE_BIN");
    println!("cargo:rerun-if-env-changed=GOLBANG_GPU");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{bin}");
    if let Some(extra) = extra {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{extra}");
    }
}
