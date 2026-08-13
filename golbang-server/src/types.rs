use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub max_tokens: Option<u32>,
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: bool,
    pub stop: Option<Stop>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum Stop {
    One(String),
    Many(Vec<String>),
}

impl Stop {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Stop::One(s) => vec![s],
            Stop::Many(v) => v,
        }
    }
}

impl From<ChatMessage> for golbang_core::ChatMessage {
    fn from(m: ChatMessage) -> Self {
        Self {
            role: m.role,
            content: m.content,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ChatCompletion {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<Timings>,
}

/// llama-server `timings` object (prompt / predicted tok/s).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Timings {
    pub cache_n: u32,
    pub prompt_n: u32,
    pub prompt_ms: f64,
    pub prompt_per_token_ms: f64,
    pub prompt_per_second: f64,
    pub predicted_n: u32,
    pub predicted_ms: f64,
    pub predicted_per_token_ms: f64,
    pub predicted_per_second: f64,
}

impl From<golbang_core::SlotTimings> for Timings {
    fn from(t: golbang_core::SlotTimings) -> Self {
        Self {
            cache_n: t.cache_n,
            prompt_n: t.prompt_n,
            prompt_ms: t.prompt_ms,
            prompt_per_token_ms: t.prompt_per_token_ms(),
            prompt_per_second: t.prompt_per_second(),
            predicted_n: t.predicted_n,
            predicted_ms: t.predicted_ms,
            predicted_per_token_ms: t.predicted_per_token_ms(),
            predicted_per_second: t.predicted_per_second(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: AssistantMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<Timings>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Default, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

pub fn validate_request(req: &ChatCompletionRequest) -> Result<(), String> {
    if req.messages.is_empty() {
        return Err("messages must not be empty".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_messages_is_invalid() {
        let req = ChatCompletionRequest {
            model: None,
            messages: vec![],
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            seed: None,
            stream: true,
            stop: None,
        };
        assert_eq!(
            validate_request(&req).unwrap_err(),
            "messages must not be empty"
        );
    }

    #[test]
    fn user_message_is_ok() {
        let req = ChatCompletionRequest {
            model: Some("qwen".into()),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "안녕".into(),
            }],
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            seed: None,
            stream: false,
            stop: None,
        };
        assert!(validate_request(&req).is_ok());
    }

    #[test]
    fn timings_json_matches_llama_server_keys() {
        let t = Timings {
            cache_n: 2,
            prompt_n: 8,
            prompt_ms: 100.0,
            prompt_per_token_ms: 12.5,
            prompt_per_second: 80.0,
            predicted_n: 16,
            predicted_ms: 2000.0,
            predicted_per_token_ms: 125.0,
            predicted_per_second: 8.0,
        };
        let v = serde_json::to_value(t).unwrap();
        for key in [
            "cache_n",
            "prompt_n",
            "prompt_ms",
            "prompt_per_token_ms",
            "prompt_per_second",
            "predicted_n",
            "predicted_ms",
            "predicted_per_token_ms",
            "predicted_per_second",
        ] {
            assert!(v.get(key).is_some(), "missing {key}");
        }
        assert!((v["predicted_per_second"].as_f64().unwrap() - 8.0).abs() < 1e-9);
    }
}
