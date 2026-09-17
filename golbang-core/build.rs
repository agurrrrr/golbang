//! `cargo:rustc-link-arg` from golbang-sys does not apply to this crate's
//! binaries/tests. Repeat the rpath so `libllama.so` resolves: the absolute
//! build-tree path for dev, plus `$ORIGIN/lib` and `$ORIGIN` for relocatable
//! release tarballs (scripts/package-release.sh). GPU-specific search path
//! follows `GOLBANG_GPU` (hip/ds41 → /opt/rocm/lib, cuda/ds41-cuda →
//! /opt/cuda-12.8/.../lib, vulkan → none) so a CUDA/Vulkan build does not pull
//! in ROCm libs.

fn main() {
    let gpu = std::env::var("GOLBANG_GPU").unwrap_or_else(|_| "hip".to_string());
    let gpu = gpu.trim().to_lowercase();
    let bin = std::env::var("DEP_LLAMA_BIN").unwrap_or_else(|_| {
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
    println!("cargo:rerun-if-env-changed=DEP_LLAMA_BIN");
    println!("cargo:rerun-if-env-changed=GOLBANG_GPU");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{bin}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    if let Some(extra) = extra {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{extra}");
    }
    // Re-export the bin dir so dependents that don't link golbang-sys directly
    // (e.g. golbang-server) can add the same rpath.
    println!("cargo:bin={bin}");
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
