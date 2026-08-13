//! Qwen ChatML, hardcoded. `llama_chat_apply_template` is not a jinja parser
//! and cannot apply the GGUF `tokenizer.chat_template` for Qwen. minijinja is later.

use tracing::warn;

/// One chat turn. Roles accepted by ChatML: `system`, `user`, `assistant`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Result of applying a template. `used_chatml == false` means raw-join fallback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AppliedPrompt {
    pub prompt: String,
    pub used_chatml: bool,
}

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

/// Apply Qwen ChatML. Empty messages or an unknown role → concatenate contents
/// and log a warning (P1 decision).
pub fn apply_chat_template(messages: &[ChatMessage]) -> AppliedPrompt {
    match try_chatml(messages) {
        Ok(prompt) => AppliedPrompt {
            prompt,
            used_chatml: true,
        },
        Err(reason) => {
            warn!(reason, "ChatML apply failed; using raw prompt fallback");
            AppliedPrompt {
                prompt: raw_join(messages),
                used_chatml: false,
            }
        }
    }
}

fn try_chatml(messages: &[ChatMessage]) -> std::result::Result<String, &'static str> {
    if messages.is_empty() {
        return Err("empty messages");
    }
    let mut out = String::new();
    for m in messages {
        let role = m.role.trim();
        if !matches!(role, "system" | "user" | "assistant") {
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
        }
    }

    #[test]
    fn user_only_gets_assistant_prefix() {
        let applied = apply_chat_template(&[msg("user", "안녕")]);
        assert!(applied.used_chatml);
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
        assert!(applied.prompt.ends_with("<|im_start|>assistant\nhey<|im_end|>\n"));
        assert!(!applied.prompt.ends_with("<|im_start|>assistant\n<|im_start|>assistant\n"));
    }

    #[test]
    fn unknown_role_falls_back() {
        let applied = apply_chat_template(&[msg("tool", "payload")]);
        assert!(!applied.used_chatml);
        assert_eq!(applied.prompt, "payload");
    }

    #[test]
    fn empty_falls_back() {
        let applied = apply_chat_template(&[]);
        assert!(!applied.used_chatml);
        assert!(applied.prompt.is_empty());
    }
}
