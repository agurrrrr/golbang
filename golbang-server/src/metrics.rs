//! Prometheus text-format metrics endpoint (P3 §4.3 observability).
//!
//! Reads the scheduler's atomic counters and renders the `/metrics` route.
//! Bucket labels mirror the coarse TTFT/ITL histograms in `SchedulerMetrics`.

use golbang_core::{SchedulerHandle, SchedulerMetrics};

const BUCKET_LE: [&str; 8] = ["1", "5", "10", "25", "50", "100", "500", "+Inf"];

fn bucket_lines(name: &str, bucket: &[std::sync::atomic::AtomicU64; 8], out: &mut String) {
    for (i, le) in BUCKET_LE.iter().enumerate() {
        let v = bucket[i].load(std::sync::atomic::Ordering::Relaxed);
        out.push_str(&format!("{name}_bucket{{le=\"{le}\"}} {v}\n"));
    }
}

/// Render Prometheus text exposition for the scheduler.
pub fn render(scheduler: &SchedulerHandle) -> String {
    render_metrics(&scheduler.metrics)
}

/// Render the scheduler metrics. Split from `render` so it can be unit-tested
/// without a live scheduler handle (HAL-2 #246).
pub fn render_metrics(m: &SchedulerMetrics) -> String {
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

    let cached_tokens = load(&m.prompt_tokens_cached_total);
    out.push_str(
        "# HELP golbang_prompt_tokens_cached_total Prompt tokens reused from the prefix cache.\n",
    );
    out.push_str("# TYPE golbang_prompt_tokens_cached_total counter\n");
    out.push_str(&format!(
        "golbang_prompt_tokens_cached_total {cached_tokens}\n"
    ));

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

    // HAL-2 #246: llama.cpp-compatible aliases so a dashboard built for
    // llama.cpp / llama-swap reads golbang unchanged. Values mirror the
    // `golbang_` series above; `:` is a legal Prometheus name character
    // (recording-rule namespace convention). The `golbang_` names are kept
    // exactly as before.
    out.push_str(
        "# HELP llamacpp:prompt_tokens_total Prompt tokens prefilled (excludes cache reuse).\n",
    );
    out.push_str("# TYPE llamacpp:prompt_tokens_total counter\n");
    out.push_str(&format!("llamacpp:prompt_tokens_total {prompt_tokens}\n"));

    out.push_str("# HELP llamacpp:tokens_predicted_total Tokens emitted to clients.\n");
    out.push_str("# TYPE llamacpp:tokens_predicted_total counter\n");
    out.push_str(&format!("llamacpp:tokens_predicted_total {gen_tokens}\n"));

    out.push_str("# HELP llamacpp:prompt_tokens_seconds Wall time spent prefilling.\n");
    out.push_str("# TYPE llamacpp:prompt_tokens_seconds gauge\n");
    out.push_str(&format!("llamacpp:prompt_tokens_seconds {prompt_s:.6}\n"));

    out.push_str("# HELP llamacpp:predicted_tokens_seconds Wall time spent generating.\n");
    out.push_str("# TYPE llamacpp:predicted_tokens_seconds gauge\n");
    out.push_str(&format!("llamacpp:predicted_tokens_seconds {pred_s:.6}\n"));

    out.push_str(
        "# HELP llamacpp:prompt_tokens_cached_total Prompt tokens reused from the prefix cache.\n",
    );
    out.push_str("# TYPE llamacpp:prompt_tokens_cached_total counter\n");
    out.push_str(&format!(
        "llamacpp:prompt_tokens_cached_total {cached_tokens}\n"
    ));

    let req_active = load(&m.requests_active);
    let req_waiting = load(&m.requests_waiting);
    out.push_str("# HELP llamacpp:requests_processing Requests currently being processed.\n");
    out.push_str("# TYPE llamacpp:requests_processing gauge\n");
    out.push_str(&format!("llamacpp:requests_processing {req_active}\n"));
    out.push_str("# HELP llamacpp:requests_deferred Requests waiting in the queue.\n");
    out.push_str("# TYPE llamacpp:requests_deferred gauge\n");
    out.push_str(&format!("llamacpp:requests_deferred {req_waiting}\n"));

    // HAL-5 #249: KV admission + startup pool-fit.
    let ctx_req = load(&m.pool_ctx_requested);
    let ctx_eff = load(&m.pool_ctx_effective);
    let ub_eff = load(&m.pool_ubatch_effective);
    out.push_str("# HELP golbang_pool_ctx_requested Requested n_ctx before startup pool-fit.\n");
    out.push_str("# TYPE golbang_pool_ctx_requested gauge\n");
    out.push_str(&format!("golbang_pool_ctx_requested {ctx_req}\n"));
    out.push_str("# HELP golbang_pool_ctx_effective Effective n_ctx after startup pool-fit.\n");
    out.push_str("# TYPE golbang_pool_ctx_effective gauge\n");
    out.push_str(&format!("golbang_pool_ctx_effective {ctx_eff}\n"));
    out.push_str(
        "# HELP golbang_pool_ubatch_effective Effective n_ubatch after startup pool-fit.\n",
    );
    out.push_str("# TYPE golbang_pool_ubatch_effective gauge\n");
    out.push_str(&format!("golbang_pool_ubatch_effective {ub_eff}\n"));
    out.push_str("# HELP golbang_pool_fit_downgrades_total Pool-fit load retries.\n");
    out.push_str("# TYPE golbang_pool_fit_downgrades_total counter\n");
    out.push_str(&format!(
        "golbang_pool_fit_downgrades_total {}\n",
        load(&m.pool_fit_downgrades)
    ));

    out.push_str(
        "# HELP golbang_joins_deferred_total Joins kept waiting by the admission reservation.\n",
    );
    out.push_str("# TYPE golbang_joins_deferred_total counter\n");
    out.push_str(&format!(
        "golbang_joins_deferred_total {}\n",
        load(&m.joins_deferred)
    ));
    out.push_str(
        "# HELP golbang_joins_clamped_total Joins force-run because the request exceeded the pool.\n",
    );
    out.push_str("# TYPE golbang_joins_clamped_total counter\n");
    out.push_str(&format!(
        "golbang_joins_clamped_total {}\n",
        load(&m.joins_clamped)
    ));
    out.push_str(
        "# HELP golbang_queue_timeouts_total Deferred requests failed by timeout while queued.\n",
    );
    out.push_str("# TYPE golbang_queue_timeouts_total counter\n");
    out.push_str(&format!(
        "golbang_queue_timeouts_total {}\n",
        load(&m.queue_timeouts)
    ));

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    fn sample_metrics() -> SchedulerMetrics {
        let m = SchedulerMetrics::default();
        m.prompt_tokens_total.store(1000, Ordering::Relaxed);
        m.prompt_tokens_cached_total.store(640, Ordering::Relaxed);
        m.tokens_generated_total.store(250, Ordering::Relaxed);
        m.prompt_us_total.store(2_000_000, Ordering::Relaxed);
        m.predicted_us_total.store(500_000, Ordering::Relaxed);
        m.requests_active.store(2, Ordering::Relaxed);
        m.requests_waiting.store(3, Ordering::Relaxed);
        m
    }

    #[test]
    fn llama_cpp_aliases_exposed_with_matching_values() {
        let out = render_metrics(&sample_metrics());
        for n in [
            "llamacpp:prompt_tokens_total",
            "llamacpp:tokens_predicted_total",
            "llamacpp:prompt_tokens_seconds",
            "llamacpp:predicted_tokens_seconds",
            "llamacpp:prompt_tokens_cached_total",
            "llamacpp:requests_processing",
            "llamacpp:requests_deferred",
        ] {
            assert!(out.contains(&format!("# TYPE {n} ")), "missing TYPE {n}");
        }
        assert!(out.contains("llamacpp:prompt_tokens_total 1000\n"));
        assert!(out.contains("llamacpp:tokens_predicted_total 250\n"));
        assert!(out.contains("llamacpp:prompt_tokens_cached_total 640\n"));
        assert!(out.contains("llamacpp:prompt_tokens_seconds 2.000000\n"));
        assert!(out.contains("llamacpp:predicted_tokens_seconds 0.500000\n"));
        assert!(out.contains("llamacpp:requests_processing 2\n"));
        assert!(out.contains("llamacpp:requests_deferred 3\n"));
    }

    #[test]
    fn golbang_names_and_values_unchanged() {
        let out = render_metrics(&sample_metrics());
        assert!(out.contains("golbang_prompt_tokens_total 1000\n"));
        assert!(out.contains("golbang_prompt_tokens_cached_total 640\n"));
        assert!(out.contains("golbang_tokens_generated_total 250\n"));
        assert!(out.contains("golbang_prompt_seconds_total 2.000000\n"));
        assert!(out.contains("golbang_predicted_seconds_total 0.500000\n"));
    }

    #[test]
    fn hal5_pool_and_admission_metrics_exposed() {
        let m = sample_metrics();
        m.pool_ctx_requested.store(100_000, Ordering::Relaxed);
        m.pool_ctx_effective.store(100_000, Ordering::Relaxed);
        m.pool_ubatch_effective.store(2_048, Ordering::Relaxed);
        m.pool_fit_downgrades.store(1, Ordering::Relaxed);
        m.joins_deferred.store(7, Ordering::Relaxed);
        m.joins_clamped.store(2, Ordering::Relaxed);
        m.queue_timeouts.store(3, Ordering::Relaxed);
        let out = render_metrics(&m);
        assert!(out.contains("golbang_pool_ctx_requested 100000\n"));
        assert!(out.contains("golbang_pool_ctx_effective 100000\n"));
        assert!(out.contains("golbang_pool_ubatch_effective 2048\n"));
        assert!(out.contains("golbang_pool_fit_downgrades_total 1\n"));
        assert!(out.contains("golbang_joins_deferred_total 7\n"));
        assert!(out.contains("golbang_joins_clamped_total 2\n"));
        assert!(out.contains("golbang_queue_timeouts_total 3\n"));
    }
}
