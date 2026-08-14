//! P0 link path (A): bindgen the SHA-pinned `llama.h`, link the existing gfx906 `.so`.
//!
//! Do not point bindgen at a live header from a different tree (llama.cpp.new / furnace / prefetch).
//! SHA or `.so` drift is a hard error — switch to (B) cmake rebuild in P3.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// docs/work-orders/P0-ffi-binding.md §4.0
const EXPECTED_SHA: &str = "5b474eb69dac2d7c26ba8855310d3b60e02a5c4f";
const DEFAULT_LLAMA_DIR: &str = "/home/agurrrrr/code/local-llm/llama.cpp";
const EXPECTED_LLAMA_H_LINES: usize = 1611;

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
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/llama_ext_shim.cpp");

    let llama_dir = PathBuf::from(
        env::var("GOLBANG_LLAMA_DIR").unwrap_or_else(|_| DEFAULT_LLAMA_DIR.to_string()),
    );
    if !llama_dir.is_dir() {
        panic!(
            "GOLBANG_LLAMA_DIR does not exist: {}. Set it to the llama.cpp tree at {EXPECTED_SHA}.",
            llama_dir.display()
        );
    }

    let head = git_stdout(&llama_dir, &["rev-parse", "HEAD"]);
    if head != EXPECTED_SHA {
        panic!(
            "llama.cpp HEAD is {head}, expected {EXPECTED_SHA}. \
             P0 is pinned to that SHA + its .so (link path A). \
             SHA/.so drift: abandon (A) and switch to (B) cmake rebuild in P3. \
             Do not mix a live header with the old .so. \
             Sibling trees (llama.cpp.new / -furnace / -prefetch) are different HEADs."
        );
    }

    let bin_dir = llama_dir.join("build/bin");
    let hip_so = first_existing(&[
        bin_dir.join("libggml-hip.so"),
        bin_dir.join("libggml-hip.so.0"),
    ]);
    let llama_so = first_existing(&[bin_dir.join("libllama.so"), bin_dir.join("libllama.so.0")]);

    let hip_bytes = fs::read(&hip_so).unwrap_or_else(|e| {
        panic!("failed to read {}: {e}", hip_so.display());
    });
    if !hip_bytes.windows(b"gfx906".len()).any(|w| w == b"gfx906") {
        panic!(
            "{} does not contain gfx906. SHA/.so drift — switch to (B) in P3.",
            hip_so.display()
        );
    }

    println!(
        "cargo:warning=P0 link (A): llama.cpp {EXPECTED_SHA}, hip={}, llama={}",
        hip_so.display(),
        llama_so.display()
    );

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let header_dir = out_dir.join("sha-headers");
    fs::create_dir_all(&header_dir).expect("create sha-headers");
    for (git_path, file_name) in HEADER_GIT_PATHS {
        extract_git_blob(
            &llama_dir,
            EXPECTED_SHA,
            git_path,
            &header_dir.join(file_name),
        );
    }

    let llama_h = header_dir.join("llama.h");
    let llama_h_text = fs::read_to_string(&llama_h).expect("read extracted llama.h");
    let line_count = llama_h_text.lines().count();
    if line_count != EXPECTED_LLAMA_H_LINES {
        panic!(
            "extracted llama.h from {EXPECTED_SHA} has {line_count} lines, expected {EXPECTED_LLAMA_H_LINES}"
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
        .allowlist_function("ggml_backend_reg_dev_count")
        .allowlist_function("ggml_backend_reg_dev_get")
        .allowlist_function("ggml_backend_dev_count")
        .allowlist_function("ggml_backend_dev_get")
        .allowlist_function("ggml_backend_dev_name")
        .allowlist_function("ggml_backend_dev_description")
        .allowlist_function("ggml_backend_dev_type")
        .allowlist_function("ggml_backend_dev_by_type")
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
    println!("cargo:rustc-link-search=native=/opt/rocm/lib");
    // rpath so `cargo test` finds the SHA-pinned .so without LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", bin_dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/rocm/lib");

    for lib in ["llama", "ggml", "ggml-base", "ggml-cpu", "ggml-hip", "mtmd"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    for lib in ["amdhip64", "hipblas", "rocblas"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }

    println!("cargo:rustc-env=GOLBANG_LLAMA_SHA={EXPECTED_SHA}");
    println!("cargo:rustc-env=GOLBANG_LLAMA_BIN={}", bin_dir.display());
    println!("cargo:rustc-env=GOLBANG_LLAMA_DIR={}", llama_dir.display());
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
