//! Repeat the SHA-pinned rpath. Server depends on golbang-core (not
//! golbang-sys directly), so it reads the bin dir that golbang-core's
//! build.rs re-exports as `cargo:bin` (`DEP_GOLBANG_CORE_BIN`), falling back
//! to `GOLBANG_LLAMA_DIR` for standalone builds.

fn main() {
    let bin = std::env::var("DEP_GOLBANG_CORE_BIN").unwrap_or_else(|_| {
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
    println!("cargo:rerun-if-env-changed=DEP_GOLBANG_CORE_BIN");
    println!("cargo:rerun-if-env-changed=GOLBANG_GPU");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{bin}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{extra}");
}
