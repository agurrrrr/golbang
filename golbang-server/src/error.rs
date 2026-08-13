use axum::http::header::{HeaderValue, RETRY_AFTER};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
    pub kind: &'static str,
    pub param: Option<&'static str>,
}

impl ApiError {
    pub fn invalid_request(message: impl Into<String>, param: Option<&'static str>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            kind: "invalid_request_error",
            param,
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            kind: "server_error",
            param: None,
        }
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
            kind: "server_error",
            param: None,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({
            "error": {
                "message": self.message,
                "type": self.kind,
                "param": self.param,
                "code": null
            }
        });
        let mut res = (self.status, Json(body)).into_response();
        if self.status == StatusCode::SERVICE_UNAVAILABLE {
            res.headers_mut()
                .insert(RETRY_AFTER, HeaderValue::from_static("1"));
        }
        res
    }
}

impl From<golbang_core::Error> for ApiError {
    fn from(e: golbang_core::Error) -> Self {
        match e {
            golbang_core::Error::EmptyPrompt
            | golbang_core::Error::ContextFull { .. }
            | golbang_core::Error::Tokenize(_) => Self::invalid_request(e.to_string(), None),
            golbang_core::Error::Cancelled | golbang_core::Error::Timeout => {
                Self::internal(e.to_string())
            }
            other => Self::internal(other.to_string()),
        }
    }
}

impl From<golbang_core::SubmitError> for ApiError {
    fn from(e: golbang_core::SubmitError) -> Self {
        match e {
            golbang_core::SubmitError::Full => {
                Self::unavailable("server is busy; retry after the current decode")
            }
            golbang_core::SubmitError::Closed => Self::internal("scheduler closed"),
        }
    }
}
