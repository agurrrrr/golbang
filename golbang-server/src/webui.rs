//! Minimal built-in web UI (Svelte static build).
//!
//! Source lives in `webui/`; `npm run build` emits a single self-contained
//! `webui/dist/index.html` (vite-plugin-singlefile) which is committed and
//! embedded here. `cargo build` therefore never needs Node.
//!
//! The page is served unauthenticated (it embeds no secrets); it prompts for
//! the API key and passes it on `/v1/chat/completions` calls.

const INDEX_HTML: &[u8] = include_bytes!("../../webui/dist/index.html");

/// `GET /` and `GET /index.html`.
pub async fn index() -> axum::response::Response {
    axum::response::Response::builder()
        .status(axum::http::StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8")
        .header(axum::http::header::CACHE_CONTROL, "no-cache")
        .body(axum::body::Body::from(INDEX_HTML))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_ui_is_self_contained_html() {
        let html = std::str::from_utf8(INDEX_HTML).expect("utf-8");
        assert!(
            html.trim_start()
                .to_lowercase()
                .starts_with("<!doctype html")
        );
        assert!(html.contains("<title>golbang</title>"));
        // Single-file build: no external script/style asset references.
        assert!(!html.contains("src=\"/assets/"));
        assert!(!html.contains("href=\"/assets/"));
    }
}
