//! Prometheus text-format metrics endpoint (P3 §4.3 observability).
//!
//! Reads the scheduler's atomic counters and renders the `/metrics` route.
//! Bucket labels mirror the coarse TTFT/ITL histograms in `SchedulerMetrics`.

use golbang_core::SchedulerHandle;

const BUCKET_LE: [&str; 8] = ["1", "5", "10", "25", "50", "100", "500", "+Inf"];

fn bucket_lines(name: &str, bucket: &[std::sync::atomic::AtomicU64; 8], out: &mut String) {
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
    let prompt_tokens = load(&m.prompt_tokens_total);
    let gen_tokens = load(&m.tokens_generated_total);
    let prompt_s = load(&m.prompt_us_total) as f64 / 1e6;
    let pred_s = load(&m.predicted_us_total) as f64 / 1e6;

    out.push_str("# HELP golbang_tokens_generated_total Tokens emitted to clients.\n");
    out.push_str("# TYPE golbang_tokens_generated_total counter\n");
    out.push_str(&format!("golbang_tokens_generated_total {gen_tokens}\n"));

    out.push_str(
        "# HELP golbang_prompt_tokens_total Prompt tokens prefilled (excludes cache reuse).\n",
    );
    out.push_str("# TYPE golbang_prompt_tokens_total counter\n");
    out.push_str(&format!("golbang_prompt_tokens_total {prompt_tokens}\n"));

    out.push_str("# HELP golbang_prompt_seconds_total Wall time spent prefilling.\n");
    out.push_str("# TYPE golbang_prompt_seconds_total counter\n");
    out.push_str(&format!("golbang_prompt_seconds_total {prompt_s:.6}\n"));

    out.push_str("# HELP golbang_predicted_seconds_total Wall time spent generating.\n");
    out.push_str("# TYPE golbang_predicted_seconds_total counter\n");
    out.push_str(&format!("golbang_predicted_seconds_total {pred_s:.6}\n"));

    let prompt_tps = if prompt_s > 0.0 {
        prompt_tokens as f64 / prompt_s
    } else {
        0.0
    };
    let pred_tps = if pred_s > 0.0 {
        gen_tokens as f64 / pred_s
    } else {
        0.0
    };
    out.push_str("# HELP golbang_prompt_tokens_per_second Lifetime average prefill throughput.\n");
    out.push_str("# TYPE golbang_prompt_tokens_per_second gauge\n");
    out.push_str(&format!(
        "golbang_prompt_tokens_per_second {prompt_tps:.4}\n"
    ));
    out.push_str(
        "# HELP golbang_predicted_tokens_per_second Lifetime average decode throughput.\n",
    );
    out.push_str("# TYPE golbang_predicted_tokens_per_second gauge\n");
    out.push_str(&format!(
        "golbang_predicted_tokens_per_second {pred_tps:.4}\n"
    ));

    let draft_n = load(&m.draft_tokens_total);
    let draft_acc = load(&m.draft_accepted_total);
    out.push_str("# HELP golbang_draft_tokens_total Speculative draft tokens proposed.\n");
    out.push_str("# TYPE golbang_draft_tokens_total counter\n");
    out.push_str(&format!("golbang_draft_tokens_total {draft_n}\n"));
    out.push_str("# HELP golbang_draft_accepted_total Speculative draft tokens accepted.\n");
    out.push_str("# TYPE golbang_draft_accepted_total counter\n");
    out.push_str(&format!("golbang_draft_accepted_total {draft_acc}\n"));

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
    out.push_str(&format!(
        "golbang_iterations_total {}\n",
        load(&m.iterations)
    ));

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
        .header(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )
        .body(axum::body::Body::from(body))
        .unwrap()
}
