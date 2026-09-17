//! Repeat the rpath. Server depends on golbang-core (not golbang-sys directly),
//! so it reads the bin dir that golbang-core's build.rs re-exports as
//! `cargo:bin` (`DEP_GOLBANG_CORE_BIN`), falling back to `GOLBANG_LLAMA_DIR` /
//! `<repo>/vendor/<tree>` for standalone builds. Adds `$ORIGIN/lib` and
//! `$ORIGIN` alongside the absolute dev path for relocatable release tarballs
//! (scripts/package-release.sh).

fn main() {
    let gpu = std::env::var("GOLBANG_GPU").unwrap_or_else(|_| "hip".to_string());
    let gpu = gpu.trim().to_lowercase();
    let bin = std::env::var("DEP_GOLBANG_CORE_BIN").unwrap_or_else(|_| {
        let root = std::env::var("GOLBANG_LLAMA_DIR").unwrap_or_else(|_| default_root(&gpu));
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
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    if let Some(extra) = extra {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{extra}");
    }
}

/// Default `<repo>/vendor/<tree>` created by `scripts/build-llama.sh`.
fn default_root(gpu: &str) -> String {
    let name = match gpu {
        "cuda" => "llama.cpp-cuda-upstream",
        "ds41" | "ds41-cuda" => "llama.cpp-ds41",
        _ => "llama.cpp-glm5next",
    };
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    std::path::Path::new(&manifest)
        .parent()
        .expect("workspace root")
        .join("vendor")
        .join(name)
        .to_string_lossy()
        .into_owned()
}
