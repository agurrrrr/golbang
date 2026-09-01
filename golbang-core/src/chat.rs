//! Chat templates: Qwen ChatML (P1 default) and GGUF jinja (`--jinja`).
//!
//! `llama_chat_apply_template` is not a jinja parser. When `--jinja` is on we
//! apply `tokenizer.chat_template` with minijinja (same idea as llama-server).

use minijinja::value::Kwargs;
use minijinja::{context, Environment, UndefinedBehavior, Value};
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
    /// Qwen3.8 jinja: `xhigh` | `medium` | `low`. `None` → template default (`xhigh`).
    /// GLM-5.3 jinja: `low` | `high`; anything else renders as `max`.
    pub reasoning_effort: Option<String>,
    /// GLM-5.3 jinja `clear_thinking`. `true` drops previous-turn thinking
    /// and pins an empty `<think></think>`; template default is `false`
    /// (Z.ai card: set true for chat). `None` → template default.
    pub clear_thinking: Option<bool>,
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

    // GLM-5.3 chat template calls `v | tojson(ensure_ascii=False)` (chat:13/163)
    // when tools are attached. The builtin `tojson` rejects unknown kwargs, so
    // the render fails and the request silently falls back to ChatML, which the
    // model cannot follow. serde_json never escapes non-ASCII (equivalent to
    // `ensure_ascii=False`), so consume the flag and delegate to the builtin.
    env.add_filter(
        "tojson",
        |value: Value, indent: Option<Value>, kwargs: Kwargs| -> Result<Value, minijinja::Error> {
            let _ensure_ascii: Option<Value> = kwargs.get("ensure_ascii")?;
            minijinja::filters::tojson(&value, indent, kwargs)
        },
    );

    env.add_template("chat", template)
        .map_err(|e| format!("compile chat template: {e}"))?;
    let tmpl = env
        .get_template("chat")
        .map_err(|e| format!("load chat template: {e}"))?;

    let msgs: Vec<serde_json::Value> = messages.iter().map(message_to_jinja).collect();
    let tools = opts.tools.clone();

    // Qwen3.8 uses `reasoning_effort|default('xhigh')`. A present `none`
    // does not trigger default and the template raises.
    let reasoning_effort = match opts
        .reasoning_effort
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        Some(effort) => Value::from(effort),
        None => Value::UNDEFINED,
    };

    // GLM-5.3: `clear_thinking if clear_thinking is defined else false`.
    // A present `none` would not trigger the template default, so absent
    // means UNDEFINED here too.
    let clear_thinking = match opts.clear_thinking {
        Some(v) => Value::from(v),
        None => Value::UNDEFINED,
    };

    tmpl.render(context! {
        messages => msgs,
        tools => tools,
        bos_token => opts.bos_token.clone(),
        eos_token => "",
        add_generation_prompt => true,
        thinking => opts.enable_thinking,
        enable_thinking => opts.enable_thinking,
        reasoning_effort => reasoning_effort,
        clear_thinking => clear_thinking,
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
        obj["tool_calls"] = serde_json::json!(m
            .tool_calls
            .iter()
            .map(|tc| {
                serde_json::json!({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": arguments_for_jinja(&tc.arguments),
                    }
                })
            })
            .collect::<Vec<_>>());
    }
    obj
}

/// OpenAI and `ToolCall` keep `arguments` as a JSON string. Qwen3.8 jinja
/// (`raise_exception` at chat:150) requires a mapping; DSV4 accepts either
/// and runs `| from_json` on strings. Parse before render so a tool-call
/// follow-up does not fall back to ChatML.
fn arguments_for_jinja(raw: &str) -> serde_json::Value {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return serde_json::json!({});
    }
    match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(_) => serde_json::Value::String(raw.to_string()),
    }
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
        assert!(applied
            .prompt
            .ends_with("<|im_start|>assistant\nhey<|im_end|>\n"));
        assert!(!applied
            .prompt
            .ends_with("<|im_start|>assistant\n<|im_start|>assistant\n"));
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
                ..Default::default()
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

    #[test]
    fn qwen38_jinja_thinking_on_ends_with_think() {
        let tmpl = include_str!("../tests/fixtures/qwen38_chat_template.jinja");
        let applied = apply_chat_template_with(
            &[msg("user", "hi")],
            &ChatApplyOpts {
                jinja: true,
                template: Some(tmpl.to_string()),
                enable_thinking: true,
                ..Default::default()
            },
        );
        assert!(applied.used_jinja, "jinja failed:\n{}", applied.prompt);
        assert!(
            applied.prompt.ends_with("<|im_start|>assistant\n<think>\n"),
            "bad gen prefix:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.contains("Reasoning effort is set to xhigh"),
            "missing default xhigh instruction:\n{}",
            applied.prompt
        );
    }

    #[test]
    fn qwen38_jinja_thinking_off_closes_think() {
        let tmpl = include_str!("../tests/fixtures/qwen38_chat_template.jinja");
        let applied = apply_chat_template_with(
            &[msg("user", "hi")],
            &ChatApplyOpts {
                jinja: true,
                template: Some(tmpl.to_string()),
                enable_thinking: false,
                ..Default::default()
            },
        );
        assert!(applied.used_jinja, "jinja failed:\n{}", applied.prompt);
        assert!(
            applied
                .prompt
                .ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "thinking-off should emit empty think block:\n{}",
            applied.prompt
        );
        assert!(
            !applied.prompt.contains("Reasoning effort is set to xhigh"),
            "thinking-off should skip effort instruction:\n{}",
            applied.prompt
        );
    }

    #[test]
    fn qwen38_jinja_tools_and_developer() {
        let tmpl = include_str!("../tests/fixtures/qwen38_chat_template.jinja");
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
            &[
                msg("developer", "be brief"),
                msg("user", "이전 작업 조회해봐"),
            ],
            &ChatApplyOpts {
                jinja: true,
                template: Some(tmpl.to_string()),
                enable_thinking: true,
                reasoning_effort: Some("low".into()),
                tools,
                ..Default::default()
            },
        );
        assert!(applied.used_jinja, "jinja failed:\n{}", applied.prompt);
        assert!(
            applied.prompt.contains("<tools>"),
            "missing tools header:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.contains("get_history"),
            "missing tool schema:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.contains("<tool_call>"),
            "missing tool-call format:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.contains("be brief"),
            "developer system text missing:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.contains("Reasoning effort is set to low"),
            "low effort instruction missing:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.ends_with("<|im_start|>assistant\n<think>\n"),
            "bad gen prefix:\n{}",
            applied.prompt
        );
    }

    fn assistant_tool_call(name: &str, arguments: &str) -> ChatMessage {
        ChatMessage {
            role: "assistant".into(),
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: name.into(),
                arguments: arguments.into(),
            }],
            ..Default::default()
        }
    }

    #[test]
    fn qwen38_jinja_parses_openai_string_tool_arguments() {
        let tmpl = include_str!("../tests/fixtures/qwen38_chat_template.jinja");
        let applied = apply_chat_template_with(
            &[
                msg("user", "351 작업 상세 조회해"),
                assistant_tool_call("get_task_detail", r#"{"task_id":351}"#),
                ChatMessage {
                    role: "tool".into(),
                    content: "ok".into(),
                    tool_call_id: Some("call_1".into()),
                    ..Default::default()
                },
            ],
            &ChatApplyOpts {
                jinja: true,
                template: Some(tmpl.to_string()),
                enable_thinking: true,
                ..Default::default()
            },
        );
        assert!(
            applied.used_jinja,
            "string arguments must not fall back to ChatML:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.contains("<function=get_task_detail>"),
            "missing tool name:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.contains("<parameter=task_id>"),
            "arguments must be expanded as a mapping:\n{}",
            applied.prompt
        );
        assert!(
            applied
                .prompt
                .contains("<tool_response>\nok\n</tool_response>"),
            "missing tool response:\n{}",
            applied.prompt
        );
        assert!(
            !applied
                .prompt
                .contains("get_task_detail were passed as a JSON string"),
            "template still saw a string:\n{}",
            applied.prompt
        );
    }

    #[test]
    fn qwen38_jinja_empty_tool_arguments_are_object() {
        let tmpl = include_str!("../tests/fixtures/qwen38_chat_template.jinja");
        let applied = apply_chat_template_with(
            &[
                msg("user", "skill 로드해"),
                assistant_tool_call("skill_load", ""),
            ],
            &ChatApplyOpts {
                jinja: true,
                template: Some(tmpl.to_string()),
                enable_thinking: true,
                ..Default::default()
            },
        );
        assert!(
            applied.used_jinja,
            "empty arguments must not raise:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.contains("<function=skill_load>"),
            "missing empty-arg tool call:\n{}",
            applied.prompt
        );
    }

    #[test]
    fn dsv4_jinja_tool_call_history_parsed_object_arguments() {
        let tmpl = include_str!("../tests/fixtures/dsv4_chat_template.jinja");
        let applied = apply_chat_template_with(
            &[
                msg("user", "이전 작업 조회해봐"),
                assistant_tool_call("get_history", r#"{"project_name":"golbang"}"#),
                ChatMessage {
                    role: "tool".into(),
                    content: "[]".into(),
                    tool_call_id: Some("call_1".into()),
                    ..Default::default()
                },
            ],
            &ChatApplyOpts {
                jinja: true,
                template: Some(tmpl.to_string()),
                bos_token: "<｜begin▁of▁sentence｜>".into(),
                enable_thinking: true,
                ..Default::default()
            },
        );
        assert!(applied.used_jinja, "dsv4 jinja failed:\n{}", applied.prompt);
        assert!(
            applied
                .prompt
                .contains("<｜DSML｜invoke name=\"get_history\">"),
            "missing DSML invoke:\n{}",
            applied.prompt
        );
        assert!(
            applied
                .prompt
                .contains("<｜DSML｜parameter name=\"project_name\" string=\"true\">golbang"),
            "arguments must survive as DSML parameters:\n{}",
            applied.prompt
        );
    }

    /// `arguments_for_jinja` keeps unparseable JSON as a raw string; the Qwen
    /// template then raises (chat:150) and we must fall back to ChatML instead
    /// of panicking or rendering a half-baked jinja prompt. ChatML drops
    /// `tool_calls` (content only), so pin the response + generation prefix.
    #[test]
    fn qwen38_jinja_invalid_tool_arguments_fall_back_to_chatml() {
        let tmpl = include_str!("../tests/fixtures/qwen38_chat_template.jinja");
        let applied = apply_chat_template_with(
            &[
                msg("user", "351 작업 상세 조회해"),
                assistant_tool_call("get_task_detail", "not-json{"),
                ChatMessage {
                    role: "tool".into(),
                    content: "ok".into(),
                    tool_call_id: Some("call_1".into()),
                    ..Default::default()
                },
            ],
            &ChatApplyOpts {
                jinja: true,
                template: Some(tmpl.to_string()),
                enable_thinking: true,
                ..Default::default()
            },
        );
        assert!(
            !applied.used_jinja,
            "unparseable arguments must raise in the template:\n{}",
            applied.prompt
        );
        assert!(
            applied.used_chatml,
            "must fall back to ChatML:\n{}",
            applied.prompt
        );
        // ChatML has no jinja tool markup — only role + content.
        assert!(
            applied
                .prompt
                .contains(&format!("{IM_START}tool\nok{IM_END}\n")),
            "tool response must survive the fallback:\n{}",
            applied.prompt
        );
        assert!(
            !applied.prompt.contains("<function=get_task_detail>"),
            "ChatML must drop tool_calls:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.ends_with(&format!("{IM_START}assistant\n")),
            "bad generation prefix:\n{}",
            applied.prompt
        );
    }

    fn glm53_opts() -> ChatApplyOpts {
        ChatApplyOpts {
            jinja: true,
            template: Some(include_str!("../tests/fixtures/glm53_chat_template.jinja").to_string()),
            ..Default::default()
        }
    }

    /// GLM-5.3 jinja: effort defaults to `max`, generation prompt is the bare
    /// `<|assistant|><think>` (no trailing newline). `prompt_opens_think`
    /// must catch it so think tokens land in `reasoning_content`.
    #[test]
    fn glm53_jinja_default_effort_max_and_bare_think_gen_prompt() {
        let applied = apply_chat_template_with(&[msg("user", "1+1=")], &glm53_opts());
        assert!(
            applied.used_jinja,
            "glm53 jinja failed:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.contains("Reasoning Effort: Max"),
            "missing default max effort:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.starts_with("[gMASK]<sop>"),
            "missing GLM bos prefix:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.ends_with("<|assistant|><think>"),
            "bad gen prefix:\n{}",
            applied.prompt
        );
    }

    #[test]
    fn glm53_jinja_effort_low_high_pass_xhigh_falls_to_max() {
        for (effort, want) in [
            ("low", "Reasoning Effort: Low"),
            ("high", "Reasoning Effort: High"),
            ("max", "Reasoning Effort: Max"),
            // Template maps anything outside low/high to max — xhigh must not
            // reach this model as a distinct level.
            ("xhigh", "Reasoning Effort: Max"),
        ] {
            let opts = ChatApplyOpts {
                reasoning_effort: Some(effort.into()),
                ..glm53_opts()
            };
            let applied = apply_chat_template_with(&[msg("user", "1+1=")], &opts);
            assert!(
                applied.used_jinja,
                "glm53 jinja failed for {effort}:\n{}",
                applied.prompt
            );
            assert!(
                applied.prompt.contains(want),
                "effort {effort} must render {want}:\n{}",
                applied.prompt
            );
        }
    }

    fn assistant_with_think(content: &str, reasoning: &str) -> ChatMessage {
        ChatMessage {
            role: "assistant".into(),
            content: content.into(),
            reasoning_content: Some(reasoning.into()),
            ..Default::default()
        }
    }

    /// `clear_thinking=true` (Z.ai chat card): thinking from previous turns is
    /// dropped and replaced by an empty `<think></think>`. `false` keeps it.
    #[test]
    fn glm53_jinja_clear_thinking_true_drops_history_think() {
        let messages = &[
            msg("user", "1+1="),
            assistant_with_think("2", "old thoughts"),
            msg("user", "그럼 2+3은?"),
        ];

        let cleared = apply_chat_template_with(
            messages,
            &ChatApplyOpts {
                clear_thinking: Some(true),
                ..glm53_opts()
            },
        );
        assert!(
            cleared.used_jinja,
            "glm53 jinja failed:\n{}",
            cleared.prompt
        );
        assert!(
            !cleared.prompt.contains("old thoughts"),
            "cleared prompt must drop previous thinking:\n{}",
            cleared.prompt
        );
        assert!(
            cleared.prompt.contains("<think></think>"),
            "cleared history must pin an empty think block:\n{}",
            cleared.prompt
        );
        assert!(
            cleared.prompt.ends_with("<|assistant|><think>"),
            "bad gen prefix:\n{}",
            cleared.prompt
        );

        let kept = apply_chat_template_with(
            messages,
            &ChatApplyOpts {
                clear_thinking: Some(false),
                ..glm53_opts()
            },
        );
        assert!(
            kept.prompt.contains("<think>old thoughts</think>"),
            "clear_thinking=false must keep previous thinking:\n{}",
            kept.prompt
        );

        // Absent (template default false) behaves like false.
        let default = apply_chat_template_with(messages, &glm53_opts());
        assert!(
            default.prompt.contains("old thoughts"),
            "absent clear_thinking must keep previous thinking:\n{}",
            default.prompt
        );
    }

    /// GLM-5.3 serializes every tool with `tojson(ensure_ascii=False)`
    /// (chat:13). The builtin `tojson` rejects the kwarg, so any tools-bearing
    /// request failed the render, fell back to ChatML, and the model ended its
    /// turn with a literal `<|im_end|>` (shepherd sheep tasks: n_tools=56).
    /// The custom `tojson` filter must keep tools requests on the GLM template.
    #[test]
    fn glm53_jinja_tools_render_without_chatml_fallback() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_history",
                "description": "Query project task history",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_name": {"type": "string"},
                        "키워드": {"type": "string", "description": "한글 설명"}
                    },
                    "required": ["project_name"]
                }
            }
        })];
        let applied = apply_chat_template_with(
            &[msg("user", "이전 작업 조회해봐")],
            &ChatApplyOpts {
                tools,
                ..glm53_opts()
            },
        );
        assert!(
            applied.used_jinja,
            "tools request must not fall back to ChatML:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.contains("<tools>"),
            "missing tools header:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.contains("get_history"),
            "missing tool schema:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.contains("<tool_call>"),
            "missing tool-call instruction:\n{}",
            applied.prompt
        );
        // serde_json never escapes non-ASCII: the Korean schema text must
        // survive verbatim (ensure_ascii=False semantics).
        assert!(
            applied.prompt.contains("한글 설명"),
            "non-ASCII tool schema text must render unescaped:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.ends_with("<|assistant|><think>"),
            "bad gen prefix:\n{}",
            applied.prompt
        );
    }

    /// OpenAI `arguments` is a JSON string. GLM-5.3 jinja does
    /// `_args.items()` (chat:163) and will ChatML-fallback if that is not a
    /// mapping. `arguments_for_jinja` must expand it so a tool-call follow-up
    /// stays on the GLM template (arg_key/arg_value), not ChatML.
    #[test]
    fn glm53_jinja_parses_openai_string_tool_arguments() {
        let applied = apply_chat_template_with(
            &[
                msg("user", "이전 작업 조회해봐"),
                assistant_tool_call("get_history", r#"{"project_name":"golbang","limit":5}"#),
                ChatMessage {
                    role: "tool".into(),
                    content: "[]".into(),
                    tool_call_id: Some("call_1".into()),
                    ..Default::default()
                },
            ],
            &glm53_opts(),
        );
        assert!(
            applied.used_jinja,
            "string arguments must not fall back to ChatML:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.contains("<tool_call>get_history"),
            "missing GLM tool name:\n{}",
            applied.prompt
        );
        assert!(
            applied
                .prompt
                .contains("<arg_key>project_name</arg_key><arg_value>golbang</arg_value>"),
            "string arguments must expand as arg_key/arg_value:\n{}",
            applied.prompt
        );
        assert!(
            applied
                .prompt
                .contains("<arg_key>limit</arg_key><arg_value>5</arg_value>"),
            "numeric argument must use tojson:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.contains("<|observation|>"),
            "missing GLM observation prefix:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.contains("<tool_response>[]</tool_response>"),
            "missing tool response:\n{}",
            applied.prompt
        );
        assert!(
            applied.prompt.ends_with("<|assistant|><think>"),
            "bad gen prefix:\n{}",
            applied.prompt
        );
    }
}
