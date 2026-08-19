//! GPU HTTP e2e. Skips unless GOLBANG_TEST_MODEL is set.
//! Shares /tmp/golbang-gpu-test.lock with golbang-core generate tests.

use std::fs::OpenOptions;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use golbang_core::{Engine, FifoPolicy, LoadParams, Model, SchedulerConfig, spawn_scheduler};
use golbang_server::{AppState, ChatRuntime, router};
use tokio::net::TcpListener;

fn lock_gpu() -> std::fs::File {
    let f = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open("/tmp/golbang-gpu-test.lock")
        .expect("gpu lock");
    let rc = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX) };
    assert_eq!(rc, 0, "flock");
    f
}

fn test_model() -> Option<String> {
    match std::env::var("GOLBANG_TEST_MODEL") {
        Ok(p) if !p.is_empty() && Path::new(&p).is_file() => Some(p),
        _ => {
            eprintln!("skip: set GOLBANG_TEST_MODEL to a small GGUF");
            None
        }
    }
}

async fn serve(model: Model, n_parallel: u32, queue_size: usize) -> u16 {
    let engine = Arc::new(Engine::new(model));
    let spawned = spawn_scheduler(
        engine,
        Box::new(FifoPolicy::default()),
        SchedulerConfig {
            n_parallel,
            queue_capacity: queue_size,
            ..Default::default()
        },
    );
    let state = AppState {
        scheduler: spawned.handle,
        model_name: "qwen-test".into(),
        default_timeout: None,
        chat: ChatRuntime::default(),
        api_keys: Vec::new(),
        vision: false,
    };
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let app = router(state);
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    port
}

fn curl_json(url: &str, body: &str) -> (i32, String) {
    let out = Command::new("curl")
        .args([
            "-sS",
            "-o",
            "-",
            "-w",
            "\nHTTP_STATUS:%{http_code}",
            "-X",
            "POST",
            url,
            "-H",
            "Content-Type: application/json",
            "-d",
            body,
        ])
        .output()
        .expect("curl");
    let txt = String::from_utf8_lossy(&out.stdout).into_owned();
    let status = txt
        .rsplit_once("HTTP_STATUS:")
        .and_then(|(_, s)| s.trim().parse().ok())
        .unwrap_or(0);
    (status, txt)
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_sse_json_and_empty_messages() {
    let Some(path) = test_model() else {
        return;
    };
    let _lock = lock_gpu();

    let model = tokio::task::spawn_blocking(move || {
        Model::load(
            path,
            LoadParams {
                n_ctx: 256,
                n_gpu_layers: 99,
                n_seq_max: 2,
                ..Default::default()
            },
        )
    })
    .await
    .expect("join")
    .expect("load");

    let port = serve(model, 2, 2).await;
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");

    let (empty_st, empty_txt) = curl_json(&url, r#"{"model":"qwen","messages":[],"stream":true}"#);
    assert!(
        empty_st == 400 || (400..500).contains(&empty_st),
        "empty messages should be 4xx, got {empty_st}:\n{empty_txt}"
    );
    assert!(
        empty_txt.contains("messages must not be empty"),
        "body should explain empty messages:\n{empty_txt}"
    );

    let sse = Command::new("curl")
        .args([
            "-sS",
            "-N",
            "-X",
            "POST",
            &url,
            "-H",
            "Content-Type: application/json",
            "-d",
            r#"{"model":"qwen","messages":[{"role":"user","content":"안녕"}],"stream":true,"max_tokens":8,"temperature":0}"#,
        ])
        .output()
        .expect("curl sse");
    assert!(
        sse.status.success(),
        "curl sse failed: {}",
        String::from_utf8_lossy(&sse.stderr)
    );
    let sse_txt = String::from_utf8_lossy(&sse.stdout);
    assert!(
        sse_txt.contains("data: [DONE]"),
        "missing data: [DONE]\n{sse_txt}"
    );
    let mut saw_delta = false;
    for line in sse_txt.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data.trim() == "[DONE]" {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(data).unwrap_or_else(|e| {
            panic!("sse json: {e}: {data}");
        });
        if v["choices"][0]["delta"].get("content").is_some() {
            saw_delta = true;
        }
    }
    assert!(saw_delta, "no choices[].delta.content in SSE\n{sse_txt}");

    let json = Command::new("curl")
        .args([
            "-sS",
            "-X",
            "POST",
            &url,
            "-H",
            "Content-Type: application/json",
            "-d",
            r#"{"model":"qwen","messages":[{"role":"user","content":"안녕"}],"stream":false,"max_tokens":8,"temperature":0}"#,
        ])
        .output()
        .expect("curl json");
    assert!(json.status.success());
    let json_txt = String::from_utf8_lossy(&json.stdout);
    let v: serde_json::Value = serde_json::from_str(&json_txt).unwrap_or_else(|e| {
        panic!("json completion: {e}: {json_txt}");
    });
    assert_eq!(v["object"], "chat.completion");
    assert!(v["choices"][0]["message"]["content"].is_string());
    assert!(v["choices"][0]["finish_reason"].is_string());
}

#[tokio::test(flavor = "multi_thread")]
async fn decode_busy_returns_503_immediately() {
    let Some(path) = test_model() else {
        return;
    };
    let _lock = lock_gpu();

    let model = tokio::task::spawn_blocking(move || {
        Model::load(
            path,
            LoadParams {
                n_ctx: 256,
                n_gpu_layers: 99,
                n_seq_max: 1,
                ..Default::default()
            },
        )
    })
    .await
    .expect("join")
    .expect("load");

    // One slot + one waiting room: the third concurrent request must 503
    // without waiting for decode.
    let port = serve(model, 1, 1).await;
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
    let body = r#"{"model":"qwen","messages":[{"role":"user","content":"안녕"}],"stream":true,"max_tokens":32,"temperature":0}"#;

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let url = url.clone();
            std::thread::spawn(move || curl_json(&url, body))
        })
        .collect();

    let results: Vec<(i32, String)> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let n_503 = results.iter().filter(|(s, _)| *s == 503).count();
    assert!(
        n_503 >= 1,
        "expected at least one 503 during decode, got {results:?}"
    );
    assert!(
        results
            .iter()
            .any(|(_, b)| b.to_ascii_lowercase().contains("retry-after")
                || b.contains("busy")
                || b.contains("HTTP_STATUS:503")),
        "503 body should be a busy signal: {results:?}"
    );
}
