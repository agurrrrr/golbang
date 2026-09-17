use golbang_core::MEDIA_MARKER;
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
    /// GLM-5.3: `low` | `high` | `max` (template maps the rest to `max`).
    /// Qwen3.8: `xhigh` | `high` | `medium` | `low`.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// GLM-5.3 jinja. `true` (chat default) drops previous-turn thinking;
    /// `false` keeps it in the prompt. Qwen/DSV4 ignore it.
    #[serde(default)]
    pub clear_thinking: Option<bool>,
    /// Max tokens inside `<think>` before `</think>` is forced.
    /// `None` = server default. `0` / negative = unlimited.
    /// Flat aliases match llama-server (`reasoning_budget_tokens`) and the
    /// harness/OpenRouter/Anthropic names halogen reads.
    #[serde(
        default,
        alias = "reasoning_budget_tokens",
        alias = "max_thinking_tokens",
        alias = "thinking_budget_tokens",
        alias = "thinking_budget",
        alias = "thinking_token_budget"
    )]
    pub reasoning_budget: Option<i32>,
    /// Turn thinking off for this request (`false` = no think, no forced close).
    /// Overrides the server/template default.
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    /// Nested thinking-control containers halogen also reads. Kept as raw JSON
    /// so one untyped object can carry several aliases without a bespoke
    /// Deserialize (`chat_template_kwargs.reasoning_effort`).
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Value>,
    /// OpenRouter-style `reasoning.effort` / `reasoning.max_tokens`.
    #[serde(default)]
    pub reasoning: Option<serde_json::Value>,
    /// Anthropic-style `thinking.budget_tokens`.
    #[serde(default)]
    pub thinking: Option<serde_json::Value>,
    /// llama-server `return_progress`. `None` uses `--prompt-progress`.
    #[serde(default)]
    pub return_progress: Option<bool>,
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

    /// Thinking budget with priority: explicit flat `reasoning_budget`
    /// (incl. flat aliases) > nested `thinking.budget_tokens` >
    /// `reasoning.max_tokens` > nested `chat_template_kwargs` budget names.
    /// `None` means the caller falls back to the server default and then to the
    /// answer-room policy.
    pub fn resolved_reasoning_budget(&self) -> Option<i32> {
        self.reasoning_budget
            .or_else(|| json_i32(self.thinking.as_ref(), &["budget_tokens"]))
            .or_else(|| json_i32(self.reasoning.as_ref(), &["max_tokens"]))
            .or_else(|| {
                json_i32(
                    self.chat_template_kwargs.as_ref(),
                    &[
                        "thinking_budget_tokens",
                        "max_thinking_tokens",
                        "thinking_budget",
                        "thinking_token_budget",
                    ],
                )
            })
    }

    /// `reasoning_effort` with priority: flat field >
    /// `chat_template_kwargs.reasoning_effort` > `reasoning.effort`.
    pub fn resolved_reasoning_effort(&self) -> Option<String> {
        self.reasoning_effort
            .clone()
            .or_else(|| json_string(self.chat_template_kwargs.as_ref(), &["reasoning_effort"]))
            .or_else(|| json_string(self.reasoning.as_ref(), &["effort"]))
    }

    /// Whether thinking is on for this request. `enable_thinking=false` wins
    /// over `default`; `chat_template_kwargs.enable_thinking` is also read.
    pub fn resolved_enable_thinking(&self, default: bool) -> bool {
        self.enable_thinking
            .or_else(|| json_bool(self.chat_template_kwargs.as_ref(), &["enable_thinking"]))
            .unwrap_or(default)
    }
}

fn json_i32(v: Option<&serde_json::Value>, keys: &[&str]) -> Option<i32> {
    let v = v?;
    for key in keys {
        if let Some(n) = v.get(*key).and_then(|x| x.as_i64()) {
            return Some(n as i32);
        }
        if let Some(s) = v.get(*key).and_then(|x| x.as_str())
            && let Ok(n) = s.trim().parse::<i32>()
        {
            return Some(n);
        }
    }
    None
}

fn json_string(v: Option<&serde_json::Value>, keys: &[&str]) -> Option<String> {
    let v = v?;
    for key in keys {
        if let Some(s) = v.get(*key).and_then(|x| x.as_str())
            && !s.trim().is_empty()
        {
            return Some(s.to_string());
        }
    }
    None
}

fn json_bool(v: Option<&serde_json::Value>, keys: &[&str]) -> Option<bool> {
    let v = v?;
    for key in keys {
        if let Some(b) = v.get(*key).and_then(|x| x.as_bool()) {
            return Some(b);
        }
    }
    None
}

/// Answer-room policy (halogen-flash-server `halogen-borrowable-techniques`
/// §5.1). When the request gives no thinking budget, reserve
/// `max(1024, 15% of max_tokens)` tokens for the answer and cap thinking at the
/// rest. An explicit request budget always wins (`0`/negative = unlimited).
/// If `max_tokens` is no larger than the answer room, allow a single think token
/// so the forced `</think>` still leaves room for a non-empty answer.
pub fn answer_room_budget(max_tokens: u32, requested: Option<i32>) -> u32 {
    match requested {
        Some(n) if n <= 0 => 0,
        Some(n) => n as u32,
        None => {
            let reserved = (u64::from(max_tokens) * 15 / 100) as u32;
            max_tokens.saturating_sub(reserved.max(1024)).max(1)
        }
    }
}

/// Compose the effective think cap: an explicit request budget wins verbatim;
/// otherwise the answer-room budget applies and the server default only lowers
/// it (so a `--reasoning-budget` unit still cannot starve the answer).
pub fn effective_reasoning_budget(max_tokens: u32, requested: Option<i32>, server: u32) -> u32 {
    match requested {
        Some(n) => answer_room_budget(max_tokens, Some(n)),
        None => {
            let automatic = answer_room_budget(max_tokens, None);
            if server == 0 {
                automatic
            } else {
                server.min(automatic)
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, deserialize_with = "deserialize_content")]
    pub content: MessageContent,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<IncomingToolCall>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

/// Text plus image/audio sources (`data:` / path). Markers are already in `text`.
#[derive(Clone, Debug, Default)]
pub struct MessageContent {
    pub text: String,
    pub media: Vec<String>,
}

impl MessageContent {
    pub fn text(s: impl Into<String>) -> Self {
        Self {
            text: s.into(),
            media: Vec::new(),
        }
    }
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

fn deserialize_content<'de, D>(deserializer: D) -> Result<MessageContent, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(match v {
        None | Some(serde_json::Value::Null) => MessageContent::default(),
        Some(serde_json::Value::String(s)) => MessageContent::text(s),
        Some(serde_json::Value::Array(parts)) => flatten_content_parts(&parts),
        Some(other) => MessageContent::text(other.to_string()),
    })
}

fn flatten_content_parts(parts: &[serde_json::Value]) -> MessageContent {
    let mut text = String::new();
    let mut media = Vec::new();
    for p in parts {
        let ty = p.get("type").and_then(|x| x.as_str()).unwrap_or("");
        if ty == "image_url" || ty == "input_image" {
            let url = p
                .get("image_url")
                .and_then(|u| u.get("url").or(Some(u)))
                .and_then(|x| x.as_str())
                .or_else(|| p.get("url").and_then(|x| x.as_str()))
                .unwrap_or("");
            if !url.is_empty() {
                text.push_str(MEDIA_MARKER);
                media.push(url.to_string());
            }
        } else if ty == "input_audio" {
            let url = p
                .pointer("/input_audio/data")
                .or_else(|| p.pointer("/input_audio/url"))
                .and_then(|x| x.as_str())
                .unwrap_or("");
            if !url.is_empty() {
                text.push_str(MEDIA_MARKER);
                media.push(url.to_string());
            }
        } else if let Some(t) = p.get("text").and_then(|x| x.as_str()) {
            text.push_str(t);
        } else if let Some(s) = p.as_str() {
            text.push_str(s);
        }
    }
    MessageContent { text, media }
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
            content: m.content.text,
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
    #[serde(skip_serializing_if = "is_zero_u32")]
    pub draft_n: u32,
    #[serde(skip_serializing_if = "is_zero_u32")]
    pub draft_n_accepted: u32,
}

fn is_zero_u32(v: &u32) -> bool {
    *v == 0
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
            draft_n: t.draft_n,
            draft_n_accepted: t.draft_n_accepted,
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
    /// llama-server `prompt_progress` (stream only, when return_progress).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_progress: Option<PromptProgress>,
}

/// llama-server `result_prompt_progress`.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct PromptProgress {
    pub total: u32,
    pub cache: u32,
    pub processed: u32,
    pub time_ms: u64,
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
            clear_thinking: None,
            reasoning_budget: None,
            enable_thinking: None,
            chat_template_kwargs: None,
            reasoning: None,
            thinking: None,
            return_progress: None,
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
                content: MessageContent::text("안녕"),
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
            clear_thinking: None,
            reasoning_budget: None,
            enable_thinking: None,
            chat_template_kwargs: None,
            reasoning: None,
            thinking: None,
            return_progress: None,
        };
        assert!(validate_request(&req).is_ok());
    }

    #[test]
    fn reasoning_budget_alias_deserializes() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"hi"}],"reasoning_budget_tokens":2048}"#,
        )
        .unwrap();
        assert_eq!(req.reasoning_budget, Some(2048));
        let req2: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"hi"}],"reasoning_budget":0}"#,
        )
        .unwrap();
        assert_eq!(req2.reasoning_budget, Some(0));
    }

    #[test]
    fn flat_thinking_budget_aliases_map_to_budget() {
        for body in [
            r#"{"messages":[{"role":"user","content":"hi"}],"reasoning_budget_tokens":2048}"#,
            r#"{"messages":[{"role":"user","content":"hi"}],"max_thinking_tokens":2048}"#,
            r#"{"messages":[{"role":"user","content":"hi"}],"thinking_budget_tokens":2048}"#,
            r#"{"messages":[{"role":"user","content":"hi"}],"thinking_budget":2048}"#,
            r#"{"messages":[{"role":"user","content":"hi"}],"thinking_token_budget":2048}"#,
        ] {
            let req: ChatCompletionRequest = serde_json::from_str(body).unwrap();
            assert_eq!(req.resolved_reasoning_budget(), Some(2048), "{body}");
        }
    }

    #[test]
    fn nested_thinking_budget_map_to_budget() {
        let anthropic: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled","budget_tokens":4096}}"#,
        )
        .unwrap();
        assert_eq!(anthropic.resolved_reasoning_budget(), Some(4096));

        let openrouter: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"hi"}],"reasoning":{"max_tokens":1234,"effort":"low"}}"#,
        )
        .unwrap();
        assert_eq!(openrouter.resolved_reasoning_budget(), Some(1234));
        assert_eq!(
            openrouter.resolved_reasoning_effort().as_deref(),
            Some("low")
        );

        let ctk: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"thinking_budget_tokens":777}}"#,
        )
        .unwrap();
        assert_eq!(ctk.resolved_reasoning_budget(), Some(777));
    }

    #[test]
    fn explicit_budget_and_effort_beat_nested() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"hi"}],"reasoning_budget":64,"thinking":{"budget_tokens":4096}}"#,
        )
        .unwrap();
        assert_eq!(req.resolved_reasoning_budget(), Some(64));

        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"low","chat_template_kwargs":{"reasoning_effort":"high"},"reasoning":{"effort":"medium"}}"#,
        )
        .unwrap();
        assert_eq!(req.resolved_reasoning_effort().as_deref(), Some("low"));
    }

    #[test]
    fn nested_reasoning_effort_fallback_order() {
        let ctk: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"reasoning_effort":"high"}}"#,
        )
        .unwrap();
        assert_eq!(ctk.resolved_reasoning_effort().as_deref(), Some("high"));

        let openrouter: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"hi"}],"reasoning":{"effort":"medium"}}"#,
        )
        .unwrap();
        assert_eq!(
            openrouter.resolved_reasoning_effort().as_deref(),
            Some("medium")
        );
    }

    #[test]
    fn enable_thinking_false_disables() {
        let flat: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"hi"}],"enable_thinking":false}"#,
        )
        .unwrap();
        assert!(!flat.resolved_enable_thinking(true));

        let nested: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":false}}"#,
        )
        .unwrap();
        assert!(!nested.resolved_enable_thinking(true));

        let default_on: ChatCompletionRequest =
            serde_json::from_str(r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
        assert!(default_on.resolved_enable_thinking(true));
    }

    #[test]
    fn answer_room_budget_reserves_answer_space() {
        // max_tokens 100: answer room (1024) exceeds max_tokens → one think token.
        assert_eq!(answer_room_budget(100, None), 1);
        // max_tokens 1024 == answer room → still one think token.
        assert_eq!(answer_room_budget(1024, None), 1);
        // 8192 * 15% = 1228 → 8192 - 1228 = 6964.
        assert_eq!(answer_room_budget(8192, None), 6964);
        // Above 6827 the 15% term dominates the 1024 floor.
        assert_eq!(answer_room_budget(10000, None), 8500);
    }

    #[test]
    fn answer_room_budget_explicit_request_wins() {
        for max_tokens in [100, 1024, 8192] {
            assert_eq!(answer_room_budget(max_tokens, Some(0)), 0, "unlimited");
            assert_eq!(
                answer_room_budget(max_tokens, Some(-1)),
                0,
                "negative unlimited"
            );
            assert_eq!(
                answer_room_budget(max_tokens, Some(777)),
                777,
                "explicit cap"
            );
        }
    }

    #[test]
    fn effective_budget_server_default_cannot_starve_answer() {
        // No request budget + server default: answer room lowers the cap.
        assert_eq!(effective_reasoning_budget(256, None, 4096), 1);
        assert_eq!(effective_reasoning_budget(8192, None, 4096), 4096);
        // Server default below the answer-room budget is kept as-is.
        assert_eq!(effective_reasoning_budget(8192, None, 100), 100);
        // No server default → pure answer room.
        assert_eq!(effective_reasoning_budget(8192, None, 0), 6964);
        // Explicit request budget bypasses the server cap.
        assert_eq!(effective_reasoning_budget(8192, Some(9999), 4096), 9999);
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
        assert_eq!(req.messages[0].content.text, "");
        assert_eq!(req.messages[0].tool_calls[0].function.name, "get_history");
        assert_eq!(req.messages[1].role, "tool");
        assert_eq!(req.messages[1].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn image_url_parts_become_media_markers() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "messages":[{
                    "role":"user",
                    "content":[
                        {"type":"text","text":"이게 뭐야?"},
                        {"type":"image_url","image_url":{"url":"data:image/png;base64,aGk="}}
                    ]
                }]
            }"#,
        )
        .unwrap();
        assert!(req.messages[0].content.text.contains(MEDIA_MARKER));
        assert!(req.messages[0].content.text.contains("이게 뭐야?"));
        assert_eq!(req.messages[0].content.media.len(), 1);
        assert!(req.messages[0].content.media[0].starts_with("data:"));
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
            draft_n: 0,
            draft_n_accepted: 0,
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

    #[test]
    fn return_progress_deserializes() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"hi"}],"stream":true,"return_progress":true}"#,
        )
        .unwrap();
        assert_eq!(req.return_progress, Some(true));
        let req2: ChatCompletionRequest =
            serde_json::from_str(r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
        assert_eq!(req2.return_progress, None);
    }

    #[test]
    fn prompt_progress_json_matches_llama_server() {
        let p = PromptProgress {
            total: 100,
            cache: 40,
            processed: 60,
            time_ms: 1234,
        };
        let v = serde_json::to_value(&p).unwrap();
        assert_eq!(v["total"], 100);
        assert_eq!(v["cache"], 40);
        assert_eq!(v["processed"], 60);
        assert_eq!(v["time_ms"], 1234);
    }
}
