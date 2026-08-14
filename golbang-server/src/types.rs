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
    #[serde(default)]
    pub tools: Vec<serde_json::Value>,
    pub tool_choice: Option<serde_json::Value>,
    /// Qwen3.8 jinja: `xhigh` | `high` | `medium` | `low`.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

impl ChatCompletionRequest {
    pub fn tools_enabled(&self) -> bool {
        if self.tools.is_empty() {
            return false;
        }
        match &self.tool_choice {
            Some(v) if v.as_str() == Some("none") => false,
            _ => true,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, deserialize_with = "deserialize_content")]
    pub content: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<IncomingToolCall>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct IncomingToolCall {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub function: IncomingFunction,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct IncomingFunction {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

fn deserialize_content<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(match v {
        None | Some(serde_json::Value::Null) => String::new(),
        Some(serde_json::Value::String(s)) => s,
        Some(serde_json::Value::Array(parts)) => {
            let mut out = String::new();
            for p in parts {
                if let Some(t) = p.get("text").and_then(|x| x.as_str()) {
                    out.push_str(t);
                } else if let Some(s) = p.as_str() {
                    out.push_str(s);
                }
            }
            out
        }
        Some(other) => other.to_string(),
    })
}

fn arguments_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => "{}".into(),
        serde_json::Value::String(s) => {
            let s = s.trim();
            if s.is_empty() {
                "{}".into()
            } else {
                s.to_string()
            }
        }
        other => other.to_string(),
    }
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
            name: m.name,
            reasoning_content: m.reasoning_content,
            tool_calls: m
                .tool_calls
                .into_iter()
                .filter(|tc| !tc.function.name.trim().is_empty())
                .map(|tc| golbang_core::ToolCall {
                    id: tc.id,
                    name: tc.function.name,
                    arguments: arguments_to_string(&tc.function.arguments),
                })
                .collect(),
            tool_call_id: m.tool_call_id,
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
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<OutgoingToolCall>,
}

#[derive(Clone, Debug, Serialize)]
pub struct OutgoingToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: &'static str,
    pub function: OutgoingFunction,
}

#[derive(Clone, Debug, Serialize)]
pub struct OutgoingFunction {
    pub name: String,
    pub arguments: String,
}

impl From<golbang_core::ToolCall> for OutgoingToolCall {
    fn from(tc: golbang_core::ToolCall) -> Self {
        Self {
            id: if tc.id.is_empty() {
                "call_1".into()
            } else {
                tc.id
            },
            type_: "function",
            function: OutgoingFunction {
                name: tc.name,
                arguments: if tc.arguments.is_empty() {
                    "{}".into()
                } else {
                    tc.arguments
                },
            },
        }
    }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<DeltaToolCall>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DeltaToolCall {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub type_: Option<&'static str>,
    pub function: DeltaFunction,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct DeltaFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
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
            tools: vec![],
            tool_choice: None,
            reasoning_effort: None,
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
                name: None,
                reasoning_content: None,
                tool_calls: vec![],
                tool_call_id: None,
            }],
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            seed: None,
            stream: false,
            stop: None,
            tools: vec![],
            tool_choice: None,
            reasoning_effort: None,
        };
        assert!(validate_request(&req).is_ok());
    }

    #[test]
    fn tools_request_deserializes() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "messages":[{"role":"user","content":"이전 작업 조회해봐"}],
                "tools":[{"type":"function","function":{"name":"get_history","parameters":{"type":"object"}}}],
                "tool_choice":"auto",
                "stream":true
            }"#,
        )
        .unwrap();
        assert!(req.tools_enabled());
        assert_eq!(req.tools[0]["function"]["name"], "get_history");
    }

    #[test]
    fn null_content_and_tool_role_ok() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "messages":[
                    {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_history","arguments":"{\"project_name\":\"test\"}"}}]},
                    {"role":"tool","tool_call_id":"call_1","content":"ok"}
                ]
            }"#,
        )
        .unwrap();
        assert_eq!(req.messages[0].content, "");
        assert_eq!(req.messages[0].tool_calls[0].function.name, "get_history");
        assert_eq!(req.messages[1].role, "tool");
        assert_eq!(req.messages[1].tool_call_id.as_deref(), Some("call_1"));
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
