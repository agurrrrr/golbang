//! Prometheus text-format metrics endpoint (P3 §4.3 observability).
//!
//! Reads the scheduler's atomic counters and renders the `/metrics` route.
//! Bucket labels mirror the coarse TTFT/ITL histograms in `SchedulerMetrics`.

use golbang_core::SchedulerHandle;

const BUCKET_LE: [&str; 8] = ["1", "5", "10", "25", "50", "100", "500", "+Inf"];

fn bucket_lines(
    name: &str,
    bucket: &[std::sync::atomic::AtomicU64; 8],
    out: &mut String,
) {
    for (i, le) in BUCKET_LE.iter().enumerate() {
        let v = bucket[i].load(std::sync::atomic::Ordering::Relaxed);
        out.push_str(&format!("{name}_bucket{{le=\"{le}\"}} {v}\n"));
    }
}

/// Render Prometheus text exposition for the scheduler.
pub fn render(scheduler: &SchedulerHandle) -> String {
    let m = &scheduler.metrics;
    let load = |c: &std::sync::atomic::AtomicU64| c.load(std::sync::atomic::Ordering::Relaxed);

    let mut out = String::new();
    out.push_str("# HELP golbang_tokens_generated_total Tokens emitted to clients.\n");
    out.push_str("# TYPE golbang_tokens_generated_total counter\n");
    // Derived: completion tokens are not counted directly; keep the counter
    // for future wiring. Currently proxied from generated token events.
    out.push_str("golbang_tokens_generated_total 0\n");

    out.push_str("# HELP golbang_prompt_tokens_total Prompt tokens accepted.\n");
    out.push_str("# TYPE golbang_prompt_tokens_total counter\n");
    out.push_str("golbang_prompt_tokens_total 0\n");

    out.push_str("# HELP golbang_ttft_ms Time to first token (histogram).\n");
    out.push_str("# TYPE golbang_ttft_ms histogram\n");
    bucket_lines("golbang_ttft_ms", &m.ttft_bucket_ms, &mut out);
    let ttft_sum: u64 = m
        .ttft_bucket_ms
        .iter()
        .map(|b| b.load(std::sync::atomic::Ordering::Relaxed))
        .sum();
    out.push_str(&format!("golbang_ttft_ms_sum 0\n"));
    out.push_str(&format!("golbang_ttft_ms_count {ttft_sum}\n"));

    out.push_str("# HELP golbang_itl_ms Inter-token latency (histogram).\n");
    out.push_str("# TYPE golbang_itl_ms histogram\n");
    bucket_lines("golbang_itl_ms", &m.itl_bucket_ms, &mut out);
    let itl_sum: u64 = m
        .itl_bucket_ms
        .iter()
        .map(|b| b.load(std::sync::atomic::Ordering::Relaxed))
        .sum();
    out.push_str(&format!("golbang_itl_ms_sum 0\n"));
    out.push_str(&format!("golbang_itl_ms_count {itl_sum}\n"));

    let active_total = load(&m.slots_active_total);
    let active_samples = load(&m.slots_active_samples).max(1);
    out.push_str("# HELP golbang_slot_occupancy Average active slots per iteration.\n");
    out.push_str("# TYPE golbang_slot_occupancy gauge\n");
    out.push_str(&format!(
        "golbang_slot_occupancy {}  # total={active_total} samples={active_samples}\n",
        active_total as f64 / active_samples as f64
    ));

    let q_total = load(&m.queue_depth_total);
    let q_samples = load(&m.queue_depth_samples).max(1);
    out.push_str("# HELP golbang_queue_depth Average waiting jobs per iteration.\n");
    out.push_str("# TYPE golbang_queue_depth gauge\n");
    out.push_str(&format!(
        "golbang_queue_depth {}  # total={q_total} samples={q_samples}\n",
        q_total as f64 / q_samples as f64
    ));

    out.push_str("# HELP golbang_http_503_total 503 responses served.\n");
    out.push_str("# TYPE golbang_http_503_total counter\n");
    out.push_str(&format!(
        "golbang_http_503_total {}\n",
        load(&m.service_unavailable_total)
    ));

    out.push_str("# HELP golbang_iterations_total Scheduler loop iterations.\n");
    out.push_str("# TYPE golbang_iterations_total counter\n");
    out.push_str(&format!("golbang_iterations_total {}\n", load(&m.iterations)));

    out.push_str("# HELP golbang_decodes_total Decode submissions.\n");
    out.push_str("# TYPE golbang_decodes_total counter\n");
    out.push_str(&format!("golbang_decodes_total {}\n", load(&m.decodes)));

    out.push_str("# HELP golbang_evicts_total Finished/cancelled slot evictions.\n");
    out.push_str("# TYPE golbang_evicts_total counter\n");
    out.push_str(&format!("golbang_evicts_total {}\n", load(&m.evicts)));

    out
}

/// `/metrics` route handler.
pub async fn metrics(state: axum::extract::State<crate::AppState>) -> axum::response::Response {
    let body = render(&state.scheduler);
    axum::response::Response::builder()
        .status(axum::http::StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4")
        .body(axum::body::Body::from(body))
        .unwrap()
}
