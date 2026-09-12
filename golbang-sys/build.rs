//! SHA-pinned `llama.h` bindgen + matching backend `.so`.
//!
//! Backend is selected via `GOLBANG_GPU` env (`hip` default, `cuda` /
//! `vulkan` optional).
//!
//! - `hip`  → `llama.cpp-glm5next` worktree (`origin/master` + glm5next PR
//!   + DPP / MMQ I=64 / GCN repack), gfx906 `.so` byte check, HIP/ROCm link.
//!   Rollback path: `llama.cpp-upgrade` @ `3ac5658c7` (kept, do not delete).
//! - `cuda` → `llama.cpp-escha` worktree (`escha-w2-dense`, GGML_OP_ESCHA_MUL_MAT),
//!   CUDA-symbol `.so` byte check, CUDA runtime (`cudart`/`cublas`) link.
//!   Rollback tree: `llama.cpp-cuda` @ `749f688fc` (kept, do not delete).
//! - `vulkan` → same tree/SHA as `hip` (`llama.cpp-glm5next`), but the
//!   `build-vulkan/` cmake dir (`GGML_VULKAN=ON`, Mesa RADV). No ROCm link;
//!   the system `libvulkan.so.1` is used. The gfx906 HIP kernel ports
//!   (DPP/mmq/GCN repack) are HIP-only — the Vulkan path tests the plain
//!   upstream kernels on the same model.
//! - `ds41` → DSV4.1 native runtime tree (`llama.cpp-ds41`, vcruz305
//!   `runtime/deepseek41` + gfx906 furnace ports), HIP build `build/bin`.
//!   Reuses the HIP link recipe; the tree/SHA differs from `hip`.
//! - `ds41-cuda` → same `llama.cpp-ds41` tree, CUDA-symbol `.so` check, CUDA
//!   runtime link, and the `build-cuda/` cmake dir (V100 sm_70, CUDA 12.8).
//!   This is a separate binary: the ds41 tree has `deepseek41` but not
//!   glm5next, so it must not replace the `hip`/`cuda` pins. See wiki
//!   `golbang-as-dsv41-runtime`.
//!
//! Do not point bindgen at a live header from a sibling tree. SHA or `.so`
//! drift is a hard error — rebuild that tree, then bump this pin.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// wiki `p0-ffi-notes` — 2026-09-01 G2 bump (glm5next, issue #98).
/// Rollback pin: `llama.cpp-upgrade` @ `3ac5658c710c0a6f3bf64d3232c4f2f386b6c2ee`.
const EXPECTED_SHA_HIP: &str = "367ebbc20c2b20db411d5acf72b88d26a7c13d70";
const EXPECTED_SHA_CUDA: &str = "c5d759c8a9e02653e9acd2442599b4c8eccc5ba5";
const DEFAULT_HIP_DIR: &str = "/home/agurrrrr/code/local-llm/llama.cpp-glm5next";
const DEFAULT_CUDA_DIR: &str = "/home/agurrrrr/code/local-llm/llama.cpp-escha";
/// DSV4.1 native runtime: vcruz305 `runtime/deepseek41` (`f37da5711`) + the
/// gfx906 furnace ports. Different tree/SHA from the `hip`/`cuda` pins because
/// this branch carries `deepseek41` and not glm5next.
const EXPECTED_SHA_DS41: &str = "24032ea2b12cc0cc38dfa58099bc1ecb6890d6fc";
const DEFAULT_DS41_DIR: &str = "/home/agurrrrr/code/local-llm/llama.cpp-ds41";
const EXPECTED_LLAMA_H_LINES: usize = 1638;

const HEADER_GIT_PATHS: &[(&str, &str)] = &[
    ("include/llama.h", "llama.h"),
    ("ggml/include/ggml.h", "ggml.h"),
    ("ggml/include/ggml-cpu.h", "ggml-cpu.h"),
    ("ggml/include/ggml-backend.h", "ggml-backend.h"),
    ("ggml/include/ggml-opt.h", "ggml-opt.h"),
    ("ggml/include/gguf.h", "gguf.h"),
    ("ggml/include/ggml-alloc.h", "ggml-alloc.h"),
    ("src/llama-ext.h", "llama-ext.h"),
    ("tools/mtmd/mtmd.h", "mtmd.h"),
    ("tools/mtmd/mtmd-helper.h", "mtmd-helper.h"),
];

fn main() {
    println!("cargo:rerun-if-env-changed=GOLBANG_LLAMA_DIR");
    println!("cargo:rerun-if-env-changed=GOLBANG_LLAMA_BIN_DIR");
    println!("cargo:rerun-if-env-changed=GOLBANG_GPU");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/llama_ext_shim.cpp");

    let gpu = env::var("GOLBANG_GPU").unwrap_or_else(|_| "hip".to_string());
    let gpu = gpu.trim().to_lowercase();
    match gpu.as_str() {
        "hip" | "cuda" | "vulkan" | "ds41" | "ds41-cuda" => {}
        other => panic!(
            "GOLBANG_GPU must be 'hip', 'cuda', 'vulkan', 'ds41' or 'ds41-cuda', got '{other}'"
        ),
    }

    // `ds41`/`ds41-cuda` reuse the HIP/CUDA link recipe but point at the DSV4.1
    // native runtime tree (different SHA + cmake dirs).
    let backend = match gpu.as_str() {
        "hip" | "ds41" => "hip",
        "cuda" | "ds41-cuda" => "cuda",
        "vulkan" => "vulkan",
        _ => unreachable!(),
    };

    let (expected_sha, default_dir) = match gpu.as_str() {
        "hip" | "vulkan" => (EXPECTED_SHA_HIP, DEFAULT_HIP_DIR),
        "cuda" => (EXPECTED_SHA_CUDA, DEFAULT_CUDA_DIR),
        "ds41" | "ds41-cuda" => (EXPECTED_SHA_DS41, DEFAULT_DS41_DIR),
        _ => unreachable!(),
    };

    let llama_dir =
        PathBuf::from(env::var("GOLBANG_LLAMA_DIR").unwrap_or_else(|_| default_dir.to_string()));
    if !llama_dir.is_dir() {
        panic!(
            "GOLBANG_LLAMA_DIR does not exist: {}. Set it to the llama.cpp tree at {expected_sha} (GOLBANG_GPU={gpu}).",
            llama_dir.display()
        );
    }

    let head = git_stdout(&llama_dir, &["rev-parse", "HEAD"]);
    if head != expected_sha {
        panic!(
            "llama.cpp HEAD is {head}, expected {expected_sha} (GOLBANG_GPU={gpu}). \
             Pin is that SHA + its {gpu} .so under the matching tree. \
             Do not mix a live header with a different .so. \
             Sibling trees (llama.cpp / -cuda / -escha / -dflash2 / -upgrade rollback) are different HEADs."
        );
    }

    // Default is `<tree>/build/bin` (production HIP/CUDA pin), or
    // `<tree>/build-vulkan/bin` for the Vulkan build. Override for a
    // same-SHA sibling cmake dir such as `build-rpc-hip` — do not point this
    // at a different git HEAD.
    let bin_dir = match env::var("GOLBANG_LLAMA_BIN_DIR") {
        Ok(p) if !p.trim().is_empty() => {
            let dir = PathBuf::from(p);
            if !dir.is_dir() {
                panic!(
                    "GOLBANG_LLAMA_BIN_DIR does not exist: {}. Same SHA as {expected_sha}, different cmake dir only.",
                    dir.display()
                );
            }
            dir
        }
        _ => llama_dir.join(match gpu.as_str() {
            "vulkan" => "build-vulkan/bin",
            "ds41-cuda" => "build-cuda/bin",
            _ => "build/bin",
        }),
    };

    // Backend-specific .so + byte check.
    let (backend_so, link_libs, link_search_extra): (PathBuf, &[&str], &[&str]);
    match backend {
        "hip" => {
            let hip_so = first_existing(&[
                bin_dir.join("libggml-hip.so"),
                bin_dir.join("libggml-hip.so.0"),
            ]);
            let bytes = fs::read(&hip_so).unwrap_or_else(|e| {
                panic!("failed to read {}: {e}", hip_so.display());
            });
            if !bytes.windows(b"gfx906".len()).any(|w| w == b"gfx906") {
                panic!(
                    "{} does not contain gfx906. SHA/.so drift at GOLBANG_GPU={gpu} — rebuild {}.",
                    hip_so.display(),
                    llama_dir.display()
                );
            }
            backend_so = hip_so;
            link_libs = &["llama", "ggml", "ggml-base", "ggml-cpu", "ggml-hip", "mtmd"];
            link_search_extra = &["/opt/rocm/lib"];
        }
        "vulkan" => {
            let vk_so = first_existing(&[
                bin_dir.join("libggml-vulkan.so"),
                bin_dir.join("libggml-vulkan.so.0"),
            ]);
            let bytes = fs::read(&vk_so).unwrap_or_else(|e| {
                panic!("failed to read {}: {e}", vk_so.display());
            });
            if !bytes.windows(b"ggml_vulkan".len()).any(|w| w == b"ggml_vulkan") {
                panic!(
                    "{} does not contain the ggml_vulkan marker. SHA/.so drift — rebuild build-vulkan.",
                    vk_so.display()
                );
            }
            backend_so = vk_so;
            link_libs = &["llama", "ggml", "ggml-base", "ggml-cpu", "ggml-vulkan", "mtmd"];
            // System loader finds libvulkan.so.1; no extra search path.
            link_search_extra = &[];
        }
        "cuda" => {
            let cuda_so = first_existing(&[
                bin_dir.join("libggml-cuda.so"),
                bin_dir.join("libggml-cuda.so.0"),
            ]);
            let bytes = fs::read(&cuda_so).unwrap_or_else(|e| {
                panic!("failed to read {}: {e}", cuda_so.display());
            });
            if !bytes
                .windows(b"__cudaRegisterFatBinary".len())
                .any(|w| w == b"__cudaRegisterFatBinary")
            {
                panic!(
                    "{} does not contain CUDA runtime symbols. SHA/.so drift at GOLBANG_GPU={gpu} — rebuild {}.",
                    cuda_so.display(),
                    llama_dir.display()
                );
            }
            backend_so = cuda_so;
            link_libs = &[
                "llama",
                "ggml",
                "ggml-base",
                "ggml-cpu",
                "ggml-cuda",
                "mtmd",
            ];
            // V100 (sm_70) needs CUDA 12.8; CUDA 13 at /opt/cuda dropped compute_70.
            link_search_extra = &["/opt/cuda-12.8/targets/x86_64-linux/lib"];
        }
        _ => unreachable!(),
    }

    let llama_so = first_existing(&[bin_dir.join("libllama.so"), bin_dir.join("libllama.so.0")]);

    println!(
        "cargo:warning=P0 link: llama.cpp {expected_sha} (GOLBANG_GPU={gpu}), backend={}, llama={}",
        backend_so.display(),
        llama_so.display()
    );

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let header_dir = out_dir.join("sha-headers");
    fs::create_dir_all(&header_dir).expect("create sha-headers");
    for (git_path, file_name) in HEADER_GIT_PATHS {
        extract_git_blob(
            &llama_dir,
            &expected_sha,
            git_path,
            &header_dir.join(file_name),
        );
    }

    let llama_h = header_dir.join("llama.h");
    let llama_h_text = fs::read_to_string(&llama_h).expect("read extracted llama.h");
    let line_count = llama_h_text.lines().count();
    if line_count != EXPECTED_LLAMA_H_LINES {
        // CUDA/HIP trees may drift slightly; log but do not hard-fail on line count.
        println!(
            "cargo:warning=extracted llama.h from {expected_sha} (GOLBANG_GPU={gpu}) has {line_count} lines, expected {EXPECTED_LLAMA_H_LINES}"
        );
    }
    if !llama_h_text.contains("llama_model_load_from_file")
        || !llama_h_text.contains("llama_init_from_model")
        || !llama_h_text.contains("llama_model_free")
    {
        panic!("extracted llama.h is missing current load API names");
    }

    cc::Build::new()
        .cpp(true)
        .file("src/llama_ext_shim.cpp")
        .include(&header_dir)
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-fPIC")
        .warnings(false)
        .compile("golbang_llama_ext");

    let mtmd_h = header_dir.join("mtmd.h");
    let mtmd_helper_h = header_dir.join("mtmd-helper.h");
    if !mtmd_h.is_file() || !mtmd_helper_h.is_file() {
        panic!(
            "extracted mtmd headers missing from {}",
            header_dir.display()
        );
    }

    // `mtmd_helper_bitmap_init_from_{buf,file}` grew an `mtmd_helper_init_opt`
    // argument (video support) in newer mtmd-helper.h. cfg follows the pinned
    // header, so the same crate source compiles against both pins.
    println!("cargo::rustc-check-cfg=cfg(mtmd_helper_init_opt)");
    let mtmd_helper_h_text =
        fs::read_to_string(&mtmd_helper_h).expect("read extracted mtmd-helper.h");
    if mtmd_helper_h_text.contains("mtmd_helper_init_opt") {
        println!("cargo:rustc-cfg=mtmd_helper_init_opt");
    }

    let bindings = bindgen::Builder::default()
        .header(llama_h.to_string_lossy())
        .header(mtmd_h.to_string_lossy())
        .header(mtmd_helper_h.to_string_lossy())
        .clang_arg(format!("-I{}", header_dir.display()))
        .clang_arg("-std=c11")
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .allowlist_var("LLAMA_.*")
        .allowlist_function("mtmd_.*")
        .allowlist_type("mtmd_.*")
        .allowlist_var("MTMD_.*")
        .allowlist_function("ggml_backend_load_all")
        .allowlist_function("ggml_backend_load_all_from_path")
        .allowlist_function("ggml_backend_load")
        .allowlist_function("ggml_backend_reg_count")
        .allowlist_function("ggml_backend_reg_get")
        .allowlist_function("ggml_backend_reg_name")
        .allowlist_function("ggml_backend_reg_by_name")
        .allowlist_function("ggml_backend_reg_get_proc_address")
        .allowlist_function("ggml_backend_register")
        .allowlist_function("ggml_backend_reg_dev_count")
        .allowlist_function("ggml_backend_reg_dev_get")
        .allowlist_function("ggml_backend_dev_count")
        .allowlist_function("ggml_backend_dev_get")
        .allowlist_function("ggml_backend_dev_name")
        .allowlist_function("ggml_backend_dev_description")
        .allowlist_function("ggml_backend_dev_type")
        .allowlist_function("ggml_backend_dev_by_type")
        .allowlist_function("ggml_backend_dev_memory")
        .allowlist_function("ggml_backend_cpu_buffer_type")
        .allowlist_function("ggml_backend_dev_buffer_type")
        .allowlist_type("ggml_log_level")
        .allowlist_type("ggml_backend_dev_type")
        .allowlist_var("GGML_LOG_.*")
        .prepend_enum_name(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen llama.h");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("write bindings.rs");

    println!("cargo:rustc-link-search=native={}", bin_dir.display());
    for extra in link_search_extra {
        println!("cargo:rustc-link-search=native={extra}");
    }
    // rpath so `cargo test` finds the SHA-pinned .so without LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", bin_dir.display());
    for extra in link_search_extra {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{extra}");
    }

    for lib in link_libs {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    match backend {
        "hip" => {
            for lib in ["amdhip64", "hipblas", "rocblas"] {
                println!("cargo:rustc-link-lib=dylib={lib}");
            }
        }
        "cuda" => {
            for lib in ["cudart", "cublas"] {
                println!("cargo:rustc-link-lib=dylib={lib}");
            }
        }
        "vulkan" => {
            // libvulkan.so.1 is a DT_NEEDED of libggml-vulkan.so itself;
            // no direct link needed.
        }
        _ => unreachable!(),
    }

    println!("cargo:rustc-env=GOLBANG_LLAMA_SHA={expected_sha}");
    println!("cargo:rustc-env=GOLBANG_LLAMA_BIN={}", bin_dir.display());
    println!("cargo:rustc-env=GOLBANG_LLAMA_DIR={}", llama_dir.display());
    println!("cargo:rustc-env=GOLBANG_GPU={gpu}");
    // Dependents read these as DEP_LLAMA_* (`links = "llama"`).
    println!("cargo:bin={}", bin_dir.display());
    println!("cargo:root={}", llama_dir.display());
}

fn git_stdout(repo: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()
        .unwrap_or_else(|e| panic!("git {} failed: {e}", args.join(" ")));
    if !out.status.success() {
        panic!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn extract_git_blob(repo: &Path, sha: &str, git_path: &str, dest: &Path) {
    let spec = format!("{sha}:{git_path}");
    let out = Command::new("git")
        .args(["show", &spec])
        .current_dir(repo)
        .output()
        .unwrap_or_else(|e| panic!("git show {spec} failed: {e}"));
    if !out.status.success() {
        panic!(
            "git show {spec} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    fs::write(dest, out.stdout).unwrap_or_else(|e| panic!("write {}: {e}", dest.display()));
}

fn first_existing(paths: &[PathBuf]) -> PathBuf {
    paths
        .iter()
        .find(|p| p.exists())
        .cloned()
        .unwrap_or_else(|| {
            panic!(
                "missing llama.cpp build artifact, looked for: {:?}",
                paths
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
            )
        })
}
