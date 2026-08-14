//! Chat templates: Qwen ChatML (P1 default) and GGUF jinja (`--jinja`).
//!
//! `llama_chat_apply_template` is not a jinja parser. When `--jinja` is on we
//! apply `tokenizer.chat_template` with minijinja (same idea as llama-server).

use minijinja::{Environment, UndefinedBehavior, Value, context};
use tracing::warn;

use crate::tools::ToolCall;

/// One chat turn. Roles accepted by ChatML: `system`, `user`, `assistant`, `tool`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub name: Option<String>,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: Option<String>,
}

/// Result of applying a template. `used_chatml` / `used_jinja` record which path ran.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AppliedPrompt {
    pub prompt: String,
    pub used_chatml: bool,
    pub used_jinja: bool,
}

/// How to format messages. Default is hardcoded ChatML.
#[derive(Clone, Debug, Default)]
pub struct ChatApplyOpts {
    pub jinja: bool,
    pub template: Option<String>,
    pub bos_token: String,
    pub enable_thinking: bool,
    /// OpenAI `tools` array. Empty = do not inject the template tools header.
    pub tools: Vec<serde_json::Value>,
}

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

/// Apply Qwen ChatML. Empty messages or an unknown role → concatenate contents
/// and log a warning (P1 decision).
pub fn apply_chat_template(messages: &[ChatMessage]) -> AppliedPrompt {
    apply_chat_template_with(messages, &ChatApplyOpts::default())
}

pub fn apply_chat_template_with(messages: &[ChatMessage], opts: &ChatApplyOpts) -> AppliedPrompt {
    if opts.jinja {
        if let Some(tmpl) = opts.template.as_deref().filter(|s| !s.is_empty()) {
            match apply_jinja(tmpl, messages, opts) {
                Ok(prompt) => {
                    return AppliedPrompt {
                        prompt,
                        used_chatml: false,
                        used_jinja: true,
                    };
                }
                Err(reason) => {
                    warn!(reason, "jinja apply failed; falling back to ChatML");
                }
            }
        } else {
            warn!("--jinja set but GGUF has no chat_template; falling back to ChatML");
        }
    }

    match try_chatml(messages) {
        Ok(prompt) => AppliedPrompt {
            prompt,
            used_chatml: true,
            used_jinja: false,
        },
        Err(reason) => {
            warn!(reason, "ChatML apply failed; using raw prompt fallback");
            AppliedPrompt {
                prompt: raw_join(messages),
                used_chatml: false,
                used_jinja: false,
            }
        }
    }
}

fn apply_jinja(
    template: &str,
    messages: &[ChatMessage],
    opts: &ChatApplyOpts,
) -> std::result::Result<String, String> {
    let mut env = Environment::new();
    env.set_undefined_behavior(UndefinedBehavior::Lenient);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |msg: String| -> Result<Value, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    env.add_filter(
        "from_json",
        |s: String| -> Result<Value, minijinja::Error> {
            let v: serde_json::Value = serde_json::from_str(&s).map_err(|e| {
                minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string())
            })?;
            Ok(Value::from_serialize(v))
        },
    );

    env.add_template("chat", template)
        .map_err(|e| format!("compile chat template: {e}"))?;
    let tmpl = env
        .get_template("chat")
        .map_err(|e| format!("load chat template: {e}"))?;

    let msgs: Vec<serde_json::Value> = messages.iter().map(message_to_jinja).collect();
    let tools = opts.tools.clone();

    tmpl.render(context! {
        messages => msgs,
        tools => tools,
        bos_token => opts.bos_token.clone(),
        eos_token => "",
        add_generation_prompt => true,
        thinking => opts.enable_thinking,
        enable_thinking => opts.enable_thinking,
    })
    .map_err(|e| format!("render chat template: {e}"))
}

fn message_to_jinja(m: &ChatMessage) -> serde_json::Value {
    let mut obj = serde_json::json!({
        "role": m.role,
        "content": m.content,
    });
    if let Some(name) = &m.name {
        obj["name"] = serde_json::json!(name);
    }
    if let Some(rc) = &m.reasoning_content {
        obj["reasoning_content"] = serde_json::json!(rc);
    }
    if let Some(id) = &m.tool_call_id {
        obj["tool_call_id"] = serde_json::json!(id);
    }
    if !m.tool_calls.is_empty() {
        obj["tool_calls"] = serde_json::json!(
            m.tool_calls
                .iter()
                .map(|tc| {
                    serde_json::json!({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    })
                })
                .collect::<Vec<_>>()
        );
    }
    obj
}

fn try_chatml(messages: &[ChatMessage]) -> std::result::Result<String, &'static str> {
    if messages.is_empty() {
        return Err("empty messages");
    }
    let mut out = String::new();
    for m in messages {
        let role = m.role.trim();
        if !matches!(role, "system" | "user" | "assistant" | "tool") {
            return Err("unknown role");
        }
        out.push_str(IM_START);
        out.push_str(role);
        out.push('\n');
        out.push_str(&m.content);
        out.push_str(IM_END);
        out.push('\n');
    }
    if messages.last().map(|m| m.role.trim()) != Some("assistant") {
        out.push_str(IM_START);
        out.push_str("assistant\n");
    }
    Ok(out)
}

fn raw_join(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        if !out.is_empty() {
            out.push('\n');
        }
        if m.content.is_empty() {
            out.push_str(m.role.trim());
        } else {
            out.push_str(m.content.trim());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
            ..Default::default()
        }
    }

    #[test]
    fn user_only_gets_assistant_prefix() {
        let applied = apply_chat_template(&[msg("user", "안녕")]);
        assert!(applied.used_chatml);
        assert!(!applied.used_jinja);
        assert_eq!(
            applied.prompt,
            "<|im_start|>user\n안녕<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn system_and_user() {
        let applied = apply_chat_template(&[msg("system", "be brief"), msg("user", "hi")]);
        assert!(applied.used_chatml);
        assert_eq!(
            applied.prompt,
            "<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn trailing_assistant_skips_generation_prefix() {
        let applied = apply_chat_template(&[msg("user", "hi"), msg("assistant", "hey")]);
        assert!(applied.used_chatml);
        assert!(
            applied
                .prompt
                .ends_with("<|im_start|>assistant\nhey<|im_end|>\n")
        );
        assert!(
            !applied
                .prompt
                .ends_with("<|im_start|>assistant\n<|im_start|>assistant\n")
        );
    }

    #[test]
    fn tool_role_is_chatml() {
        let applied = apply_chat_template(&[msg("tool", "payload")]);
        assert!(applied.used_chatml);
        assert_eq!(
            applied.prompt,
            "<|im_start|>tool\npayload<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn unknown_role_falls_back() {
        let applied = apply_chat_template(&[msg("developer", "payload")]);
        assert!(!applied.used_chatml);
        assert_eq!(applied.prompt, "payload");
    }

    #[test]
    fn empty_falls_back() {
        let applied = apply_chat_template(&[]);
        assert!(!applied.used_chatml);
        assert!(applied.prompt.is_empty());
    }

    #[test]
    fn dsv4_jinja_thinking_on() {
        let tmpl = include_str!("../tests/fixtures/dsv4_chat_template.jinja");
        let applied = apply_chat_template_with(
            &[msg("user", "hi")],
            &ChatApplyOpts {
                jinja: true,
                template: Some(tmpl.to_string()),
                bos_token: "<｜begin▁of▁sentence｜>".into(),
                enable_thinking: true,
                ..Default::default()
            },
        );
        assert!(applied.used_jinja);
        assert_eq!(
            applied.prompt,
            "<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜><think>"
        );
    }

    #[test]
    fn dsv4_jinja_thinking_off() {
        let tmpl = include_str!("../tests/fixtures/dsv4_chat_template.jinja");
        let applied = apply_chat_template_with(
            &[msg("system", "be brief"), msg("user", "hi")],
            &ChatApplyOpts {
                jinja: true,
                template: Some(tmpl.to_string()),
                bos_token: "<｜begin▁of▁sentence｜>".into(),
                enable_thinking: false,
                ..Default::default()
            },
        );
        assert!(applied.used_jinja);
        assert_eq!(
            applied.prompt,
            "<｜begin▁of▁sentence｜>be brief<｜User｜>hi<｜Assistant｜></think>"
        );
    }

    #[test]
    fn dsv4_jinja_injects_tools_header() {
        let tmpl = include_str!("../tests/fixtures/dsv4_chat_template.jinja");
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_history",
                "description": "Query project task history",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_name": {"type": "string"}
                    },
                    "required": ["project_name"]
                }
            }
        })];
        let applied = apply_chat_template_with(
            &[msg("user", "이전 작업 조회해봐")],
            &ChatApplyOpts {
                jinja: true,
                template: Some(tmpl.to_string()),
                bos_token: "<｜begin▁of▁sentence｜>".into(),
                enable_thinking: true,
                tools,
            },
        );
        assert!(applied.used_jinja);
        assert!(
            applied.prompt.contains("## Tools"),
            "missing tools header:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.contains("get_history"),
            "missing tool schema:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.contains("<｜DSML｜tool_calls>"),
            "missing DSML instruction:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.ends_with("<｜Assistant｜><think>"),
            "bad gen prefix:\n{}",
            applied.prompt
        );
    }
}
