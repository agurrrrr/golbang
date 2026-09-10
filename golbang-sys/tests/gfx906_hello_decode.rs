//! P0 connection check: load a small GGUF, decode `"Hello"` once.
//!
//! Backend (`hip` | `cuda` | `vulkan`) is selected by `GOLBANG_GPU` at
//! build time; the test asserts the matching backend/device strings.
//!
//! ```text
//! GOLBANG_TEST_MODEL=/path/to/small.gguf cargo test -p golbang-sys -- --nocapture
//! ```
//!
//! No speed floor. Quality of the sampled token is not asserted.

use std::env;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::Path;
use std::ptr;
use std::sync::Mutex;

use golbang_sys::*;

static LOGS: Mutex<String> = Mutex::new(String::new());

unsafe extern "C" fn capture_log(_level: ggml_log_level, text: *const c_char, _ud: *mut c_void) {
    if text.is_null() {
        return;
    }
    let s = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    eprint!("{s}");
    if let Ok(mut buf) = LOGS.lock() {
        buf.push_str(&s);
    }
}

struct LlamaSession {
    model: *mut llama_model,
    ctx: *mut llama_context,
}

impl Drop for LlamaSession {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                llama_free(self.ctx);
                self.ctx = ptr::null_mut();
            }
            if !self.model.is_null() {
                llama_model_free(self.model);
                self.model = ptr::null_mut();
            }
            llama_backend_free();
        }
    }
}

fn logs_snapshot() -> String {
    LOGS.lock().map(|g| g.clone()).unwrap_or_default()
}

fn parse_offloaded_layers(logs: &str) -> Option<(u32, u32)> {
    // "offloaded 25/25 layers to GPU"
    for line in logs.lines() {
        let Some(rest) = line.split("offloaded ").nth(1) else {
            continue;
        };
        let Some(frac) = rest.split(" layers to GPU").next() else {
            continue;
        };
        let mut parts = frac.split('/');
        let n = parts.next()?.trim().parse().ok()?;
        let d = parts.next()?.trim().parse().ok()?;
        return Some((n, d));
    }
    None
}

#[test]
fn hello_decode_once_on_gpu() {
    let gpu = env::var("GOLBANG_GPU").unwrap_or_else(|_| "hip".to_string());
    let gpu = gpu.trim().to_lowercase();
    let (expect_backend, expect_device): (Vec<&str>, Vec<&str>) = match gpu.as_str() {
        "cuda" => (vec!["CUDA"], vec!["CUDA", "NVIDIA", "GeForce"]),
        "vulkan" => (vec!["Vulkan"], vec!["VEGA20", "RADV", "Vulkan"]),
        _ => (vec!["HIP", "ROCm"], vec!["gfx906", "0x66a1"]),
    };

    let model_path = match env::var("GOLBANG_TEST_MODEL") {
        Ok(p) if !p.is_empty() => p,
        _ => {
            eprintln!(
                "skip: set GOLBANG_TEST_MODEL to a small GGUF \
                 (P0 DoD: cargo test -p golbang-sys with that env)."
            );
            return;
        }
    };
    assert!(
        Path::new(&model_path).is_file(),
        "GOLBANG_TEST_MODEL is not a file: {model_path}"
    );

    {
        let mut buf = LOGS.lock().expect("log mutex");
        buf.clear();
    }

    let prompt = "Hello";
    const N_GPU_LAYERS: i32 = 99;

    unsafe {
        llama_log_set(Some(capture_log), ptr::null_mut());
        llama_backend_init();

        // Test binaries live in target/debug/deps, so the default
        // ggml_backend_load_all() search (exe dir / cwd) misses libggml-hip.so.
        let bin = CString::new(LLAMA_BIN_DIR).expect("bin dir");
        ggml_backend_load_all_from_path(bin.as_ptr());

        let sys = CStr::from_ptr(llama_print_system_info()).to_string_lossy();
        eprintln!("llama_print_system_info: {sys}");
        eprintln!("linked llama.cpp SHA {}", LLAMA_CPP_SHA);

        assert!(
            llama_supports_gpu_offload(),
            "llama_supports_gpu_offload() is false — HIP/ROCm backend not registered"
        );

        let mut backend_names = Vec::new();
        let n_reg = ggml_backend_reg_count();
        for i in 0..n_reg {
            let reg = ggml_backend_reg_get(i);
            if !reg.is_null() {
                let name = CStr::from_ptr(ggml_backend_reg_name(reg)).to_string_lossy();
                backend_names.push(name.into_owned());
            }
        }
        eprintln!("registered backends: {backend_names:?}");
        assert!(
            backend_names
                .iter()
                .any(|n| expect_backend.iter().any(|b| n.eq_ignore_ascii_case(b))),
            "no expected backend registered; got {backend_names:?} (expected {expect_backend:?})"
        );

        let mut device_blob = String::new();
        let n_dev = ggml_backend_dev_count();
        for i in 0..n_dev {
            let dev = ggml_backend_dev_get(i);
            if dev.is_null() {
                continue;
            }
            let name = CStr::from_ptr(ggml_backend_dev_name(dev)).to_string_lossy();
            let desc = CStr::from_ptr(ggml_backend_dev_description(dev)).to_string_lossy();
            eprintln!("device[{i}]: name={name} desc={desc}");
            device_blob.push_str(&name);
            device_blob.push(' ');
            device_blob.push_str(&desc);
            device_blob.push('\n');
        }

        let mut params = llama_model_default_params();
        params.n_gpu_layers = N_GPU_LAYERS;

        let c_path = CString::new(model_path.as_str()).expect("model path");
        let model = llama_model_load_from_file(c_path.as_ptr(), params);
        assert!(
            !model.is_null(),
            "llama_model_load_from_file failed for {model_path}\nlogs:\n{}",
            logs_snapshot()
        );

        let mut session = LlamaSession {
            model,
            ctx: ptr::null_mut(),
        };

        let n_layer = llama_model_n_layer(model);
        assert!(n_layer > 0, "llama_model_n_layer returned {n_layer}");

        let vocab = llama_model_get_vocab(model);
        assert!(!vocab.is_null(), "llama_model_get_vocab returned null");
        let n_vocab = llama_vocab_n_tokens(vocab);
        assert!(n_vocab > 0, "n_vocab={n_vocab}");

        let logs_after_load = logs_snapshot();
        let (offloaded, offload_denom) = parse_offloaded_layers(&logs_after_load).unwrap_or_else(|| {
            panic!(
                "no 'offloaded N/M layers to GPU' line after load — cannot prove n_gpu_layers applied\n{logs_after_load}"
            );
        });
        eprintln!(
            "offloaded {offloaded}/{offload_denom} layers; model n_layer={n_layer}; requested n_gpu_layers={N_GPU_LAYERS}"
        );
        assert!(
            offloaded > 0,
            "0 layers offloaded — CPU fallback, not gfx906"
        );
        assert!(
            offloaded as i32 >= n_layer,
            "requested n_gpu_layers={N_GPU_LAYERS} but only {offloaded}/{offload_denom} offloaded (n_layer={n_layer})"
        );

        let mut cparams = llama_context_default_params();
        cparams.n_ctx = 128;
        cparams.n_batch = 128;
        cparams.n_ubatch = 128;
        cparams.n_seq_max = 1;

        let ctx = llama_init_from_model(model, cparams);
        assert!(
            !ctx.is_null(),
            "llama_init_from_model failed\nlogs:\n{}",
            logs_snapshot()
        );
        session.ctx = ctx;

        let n_needed = llama_tokenize(
            vocab,
            prompt.as_ptr() as *const c_char,
            prompt.len() as i32,
            ptr::null_mut(),
            0,
            true,
            true,
        );
        assert!(
            n_needed < 0,
            "llama_tokenize size probe should be negative, got {n_needed}"
        );
        let n_tok = (-n_needed) as i32;
        assert!(n_tok > 0, "tokenized '{prompt}' to 0 tokens");
        let mut tokens = vec![0 as llama_token; n_tok as usize];
        let n_written = llama_tokenize(
            vocab,
            prompt.as_ptr() as *const c_char,
            prompt.len() as i32,
            tokens.as_mut_ptr(),
            n_tok,
            true,
            true,
        );
        assert_eq!(n_written, n_tok, "tokenize wrote {n_written}, expected {n_tok}");
        eprintln!("prompt '{prompt}' -> {n_written} tokens: {tokens:?}");

        let batch = llama_batch_get_one(tokens.as_mut_ptr(), n_written);
        let rc = llama_decode(ctx, batch);
        assert_eq!(rc, 0, "llama_decode returned {rc}\nlogs:\n{}", logs_snapshot());

        let logits = llama_get_logits(ctx);
        assert!(!logits.is_null(), "llama_get_logits returned null");

        let n_vocab_usz = n_vocab as usize;
        let row = std::slice::from_raw_parts(logits, n_vocab_usz);
        let (argmax, max_logit) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, v)| (i as llama_token, *v))
            .expect("empty logits");

        eprintln!("argmax={argmax} logit={max_logit} n_vocab={n_vocab}");
        assert!(
            argmax >= 0 && argmax < n_vocab,
            "argmax {argmax} not in [0, {n_vocab})"
        );

        let logs = logs_snapshot();
        let backendish = expect_backend
            .iter()
            .any(|b| logs.to_uppercase().contains(&b.to_uppercase()))
            || backend_names
                .iter()
                .any(|n| expect_backend.iter().any(|b| n.eq_ignore_ascii_case(b)));
        let deviceish = expect_device
            .iter()
            .any(|d| logs.to_uppercase().contains(&d.to_uppercase()))
            || device_blob
                .to_uppercase()
                .contains(&expect_device[0].to_uppercase());
        assert!(
            backendish,
            "logs/backends do not mention expected backend {expect_backend:?} — possible CPU fallback\n{logs}"
        );
        assert!(
            deviceish,
            "logs/devices do not mention expected device {expect_device:?}\nlogs:\n{logs}\ndevices:\n{device_blob}"
        );

        drop(session);
    }
}
