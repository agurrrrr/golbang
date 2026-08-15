//! Split DeepSeek-style `<think>…</think>` from generated text.
//!
//! Matches llama.cpp `--reasoning-format`:
//! - `none`: leave thoughts in `content`
//! - `deepseek`: thoughts go to `reasoning_content` (including stream deltas)
//! - `deepseek-legacy`: same extraction, but stream also keeps tags in `content`

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

/// llama.cpp DSV4 also ends thinking on the tool-call open tag so a missing
/// `</think>` does not swallow the invoke into `reasoning_content`.
const EXTRA_THINK_ENDS: &[&str] = &[
    "<｜DSML｜tool_calls>",
    "<｜DSML｜function_calls>",
    "<tool_call>",
    "<tools_call>",
    "<function=",
    "<|tool▁calls▁begin|>",
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReasoningFormat {
    #[default]
    None,
    Deepseek,
    DeepseekLegacy,
}

impl ReasoningFormat {
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" | "off" | "false" | "" => Ok(Self::None),
            "deepseek" | "auto" => Ok(Self::Deepseek),
            "deepseek-legacy" | "deepseek_legacy" | "legacy" => Ok(Self::DeepseekLegacy),
            other => Err(format!(
                "--reasoning-format must be none|deepseek|deepseek-legacy|auto, got {other}"
            )),
        }
    }

    pub fn extracts(self) -> bool {
        !matches!(self, Self::None)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Deepseek => "deepseek",
            Self::DeepseekLegacy => "deepseek-legacy",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReasoningDelta {
    pub reasoning: Option<String>,
    pub content: Option<String>,
}

impl ReasoningDelta {
    pub fn is_empty(&self) -> bool {
        self.reasoning.as_ref().is_none_or(|s| s.is_empty())
            && self.content.as_ref().is_none_or(|s| s.is_empty())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Content,
    Think,
}

/// Incremental splitter. Tokens can split tags (`<` + `think>`).
pub struct ReasoningParser {
    format: ReasoningFormat,
    phase: Phase,
    hold: String,
}

impl ReasoningParser {
    pub fn new(format: ReasoningFormat, start_in_think: bool) -> Self {
        Self {
            format,
            phase: if start_in_think && format.extracts() {
                Phase::Think
            } else {
                Phase::Content
            },
            hold: String::new(),
        }
    }

    /// If the prompt already opened `<think>` (DSV4, or Qwen templates that
    /// close with `<think>\n`), start in think. Trailing whitespace is ignored.
    pub fn from_prompt(format: ReasoningFormat, prompt: &str) -> Self {
        let start_in_think = format.extracts() && prompt_opens_think(prompt);
        Self::new(format, start_in_think)
    }

    pub fn push(&mut self, piece: &str) -> ReasoningDelta {
        if piece.is_empty() {
            return ReasoningDelta::default();
        }
        if !self.format.extracts() {
            return ReasoningDelta {
                reasoning: None,
                content: Some(piece.to_string()),
            };
        }
        self.hold.push_str(piece);
        self.drain(false)
    }

    pub fn finish(&mut self) -> ReasoningDelta {
        if !self.format.extracts() {
            if self.hold.is_empty() {
                return ReasoningDelta::default();
            }
            let rest = std::mem::take(&mut self.hold);
            return ReasoningDelta {
                reasoning: None,
                content: Some(rest),
            };
        }
        self.drain(true)
    }

    fn drain(&mut self, flush: bool) -> ReasoningDelta {
        let mut reasoning = String::new();
        let mut content = String::new();
        let inline = matches!(self.format, ReasoningFormat::DeepseekLegacy);

        loop {
            if self.hold.is_empty() {
                break;
            }
            let tag = match self.phase {
                Phase::Content => OPEN,
                Phase::Think => CLOSE,
            };
            if let Some((idx, matched, consume)) = next_phase_marker(&self.hold, self.phase, tag) {
                let before = self.hold[..idx].to_string();
                let after = if consume {
                    self.hold[idx + matched.len()..].to_string()
                } else {
                    self.hold[idx..].to_string()
                };
                match self.phase {
                    Phase::Content => {
                        content.push_str(&before);
                        if inline {
                            content.push_str(matched);
                        }
                        self.phase = Phase::Think;
                    }
                    Phase::Think => {
                        reasoning.push_str(&before);
                        if inline {
                            content.push_str(&before);
                            if consume {
                                content.push_str(matched);
                            }
                        }
                        self.phase = Phase::Content;
                    }
                }
                self.hold = after;
                continue;
            }

            if flush {
                match self.phase {
                    Phase::Content => content.push_str(&self.hold),
                    Phase::Think => {
                        reasoning.push_str(&self.hold);
                        if inline {
                            content.push_str(&self.hold);
                        }
                    }
                }
                self.hold.clear();
                break;
            }

            let keep = suffix_that_is_any_tag_prefix(&self.hold, self.phase, tag);
            let emit_len = self.hold.len() - keep;
            if emit_len > 0 {
                let emit = self.hold[..emit_len].to_string();
                match self.phase {
                    Phase::Content => content.push_str(&emit),
                    Phase::Think => {
                        reasoning.push_str(&emit);
                        if inline {
                            content.push_str(&emit);
                        }
                    }
                }
                self.hold.drain(..emit_len);
            }
            break;
        }

        ReasoningDelta {
            reasoning: nonempty(reasoning),
            content: nonempty(content),
        }
    }
}

fn nonempty(s: String) -> Option<String> {
    if s.is_empty() { None } else { Some(s) }
}

/// Qwen3.6/3.8 jinja ends the prompt with `<think>\n`. DSV4 uses a bare `<think>`.
pub fn prompt_opens_think(prompt: &str) -> bool {
    prompt.trim_end().ends_with(OPEN)
}

/// Cap tokens spent inside `<think>` and force `</think>` so a long think
/// cannot eat `max_tokens` (shepherd 383–391: 12k think, no tool call).
#[derive(Clone, Debug, Default)]
pub struct ThinkBudget {
    limit: u32,
    in_think: bool,
    n: u32,
    close: Vec<crate::tokenizer::Token>,
    force_at: Option<usize>,
    hold: String,
}

impl ThinkBudget {
    pub fn disabled() -> Self {
        Self::default()
    }

    pub fn new(limit: u32, start_in_think: bool, close: Vec<crate::tokenizer::Token>) -> Self {
        if limit == 0 || close.is_empty() {
            return Self::disabled();
        }
        Self {
            limit,
            in_think: start_in_think,
            n: 0,
            close,
            force_at: None,
            hold: String::new(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.limit > 0 && !self.close.is_empty()
    }

    pub fn is_forcing(&self) -> bool {
        self.force_at.is_some()
    }

    pub fn close_len(&self) -> usize {
        self.close.len()
    }

    pub fn think_tokens(&self) -> u32 {
        self.n
    }

    pub fn forced_token(&self) -> Option<crate::tokenizer::Token> {
        self.force_at.and_then(|i| self.close.get(i).copied())
    }

    /// Count a just-emitted token. When the budget is spent, the next
    /// `forced_token()` is the first close-tag token.
    pub fn on_emit(&mut self, piece: &str) {
        if !self.enabled() {
            return;
        }
        if let Some(i) = self.force_at {
            let next = i + 1;
            if next >= self.close.len() {
                self.force_at = None;
                self.in_think = false;
                self.hold.clear();
            } else {
                self.force_at = Some(next);
            }
            return;
        }
        if !self.in_think {
            self.hold.push_str(piece);
            if let Some(idx) = self.hold.find(OPEN) {
                self.in_think = true;
                self.n = 0;
                self.hold.drain(..idx + OPEN.len());
                self.maybe_end_or_force();
            } else {
                let keep = suffix_that_is_tag_prefix(&self.hold, OPEN);
                let extra = self.hold.len() - keep;
                if extra > 0 {
                    self.hold.drain(..extra);
                }
            }
            return;
        }
        self.n = self.n.saturating_add(1);
        self.hold.push_str(piece);
        self.maybe_end_or_force();
    }

    fn maybe_end_or_force(&mut self) {
        if think_end_at(&self.hold).is_some() {
            self.in_think = false;
            self.force_at = None;
            self.hold.clear();
            return;
        }
        let keep = suffix_that_is_any_tag_prefix(&self.hold, Phase::Think, CLOSE);
        let extra = self.hold.len() - keep;
        if extra > 0 {
            self.hold.drain(..extra);
        }
        if self.in_think && self.n >= self.limit {
            self.force_at = Some(0);
        }
    }
}

fn think_end_at(hold: &str) -> Option<usize> {
    next_phase_marker(hold, Phase::Think, CLOSE).map(|(i, _, _)| i)
}

fn next_phase_marker<'a>(
    hold: &'a str,
    phase: Phase,
    primary: &'a str,
) -> Option<(usize, &'a str, bool)> {
    let mut best: Option<(usize, &'a str, bool)> = hold.find(primary).map(|i| (i, primary, true));
    if matches!(phase, Phase::Think) {
        for extra in EXTRA_THINK_ENDS {
            if let Some(i) = hold.find(extra) {
                match best {
                    Some((bi, _, _)) if bi <= i => {}
                    _ => best = Some((i, *extra, false)),
                }
            }
        }
    }
    best
}

fn suffix_that_is_any_tag_prefix(hay: &str, phase: Phase, primary: &str) -> usize {
    let mut keep = suffix_that_is_tag_prefix(hay, primary);
    if matches!(phase, Phase::Think) {
        for extra in EXTRA_THINK_ENDS {
            keep = keep.max(suffix_that_is_tag_prefix(hay, extra));
        }
    }
    keep
}

fn suffix_that_is_tag_prefix(hay: &str, tag: &str) -> usize {
    let max = hay.len().min(tag.len().saturating_sub(1));
    for n in (1..=max).rev() {
        if hay.as_bytes().ends_with(&tag.as_bytes()[..n]) {
            return n;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_passes_through() {
        let mut p = ReasoningParser::new(ReasoningFormat::None, true);
        let d = p.push("hello");
        assert_eq!(d.content.as_deref(), Some("hello"));
        assert!(d.reasoning.is_none());
    }

    #[test]
    fn start_in_think_then_content() {
        let mut p =
            ReasoningParser::from_prompt(ReasoningFormat::Deepseek, "<｜Assistant｜><think>");
        let a = p.push("hmm");
        assert_eq!(a.reasoning.as_deref(), Some("hmm"));
        assert!(a.content.is_none());
        let b = p.push("</think>2");
        assert!(b.reasoning.is_none());
        assert_eq!(b.content.as_deref(), Some("2"));
    }

    #[test]
    fn qwen_think_open_with_trailing_newline() {
        let mut p = ReasoningParser::from_prompt(
            ReasoningFormat::Deepseek,
            "<|im_start|>assistant\n<think>\n",
        );
        let a = p.push("hmm");
        assert_eq!(a.reasoning.as_deref(), Some("hmm"));
        assert!(a.content.is_none());
    }

    #[test]
    fn qwen_thinking_off_empty_block_stays_in_content() {
        let mut p = ReasoningParser::from_prompt(
            ReasoningFormat::Deepseek,
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        );
        let a = p.push("hi");
        assert!(a.reasoning.is_none());
        assert_eq!(a.content.as_deref(), Some("hi"));
    }

    #[test]
    fn split_close_tag() {
        let mut p = ReasoningParser::new(ReasoningFormat::Deepseek, true);
        let a = p.push("ab</th");
        assert_eq!(a.reasoning.as_deref(), Some("ab"));
        assert!(a.content.is_none());
        let b = p.push("ink>cd");
        assert!(b.reasoning.is_none());
        assert_eq!(b.content.as_deref(), Some("cd"));
    }

    #[test]
    fn tags_in_generation() {
        let mut p = ReasoningParser::new(ReasoningFormat::Deepseek, false);
        let d = p.push("x<think>y</think>z");
        assert_eq!(d.reasoning.as_deref(), Some("y"));
        assert_eq!(d.content.as_deref(), Some("xz"));
    }

    #[test]
    fn tool_open_ends_think_and_stays_in_content() {
        let mut p = ReasoningParser::new(ReasoningFormat::Deepseek, true);
        let d = p.push("hmm<tool_call>{\"name\":\"x\"}</tool_call>");
        assert_eq!(d.reasoning.as_deref(), Some("hmm"));
        assert_eq!(
            d.content.as_deref(),
            Some("<tool_call>{\"name\":\"x\"}</tool_call>")
        );
    }

    #[test]
    fn parse_names() {
        assert_eq!(
            ReasoningFormat::parse("auto").unwrap(),
            ReasoningFormat::Deepseek
        );
        assert!(ReasoningFormat::parse("nope").is_err());
    }

    #[test]
    fn budget_forces_close_after_limit() {
        let mut b = ThinkBudget::new(2, true, vec![7, 8]);
        assert!(b.forced_token().is_none());
        b.on_emit("a");
        assert!(b.forced_token().is_none());
        b.on_emit("b");
        assert_eq!(b.forced_token(), Some(7));
        b.on_emit("</think>");
        assert_eq!(b.forced_token(), Some(8));
        b.on_emit("\n\n");
        assert!(b.forced_token().is_none());
        assert!(!b.in_think);
    }

    #[test]
    fn budget_natural_close_does_not_force() {
        let mut b = ThinkBudget::new(10, true, vec![7]);
        b.on_emit("hmm");
        b.on_emit("</think>");
        assert!(b.forced_token().is_none());
        b.on_emit("hi");
        assert!(b.forced_token().is_none());
        assert!(!b.in_think);
    }

    #[test]
    fn budget_split_close_tag() {
        let mut b = ThinkBudget::new(10, true, vec![7]);
        b.on_emit("ab</th");
        assert!(b.in_think);
        b.on_emit("ink>cd");
        assert!(!b.in_think);
        assert!(b.forced_token().is_none());
    }

    #[test]
    fn budget_tool_open_ends_think() {
        let mut b = ThinkBudget::new(10, true, vec![7]);
        b.on_emit("hmm<tool_call>");
        assert!(!b.in_think);
        assert!(b.forced_token().is_none());
    }

    #[test]
    fn budget_disabled_when_limit_zero() {
        let mut b = ThinkBudget::new(0, true, vec![7]);
        b.on_emit("a");
        b.on_emit("b");
        assert!(b.forced_token().is_none());
    }

    #[test]
    fn budget_starts_on_generated_open_tag() {
        let mut b = ThinkBudget::new(1, false, vec![7]);
        b.on_emit("pre");
        assert!(b.forced_token().is_none());
        b.on_emit("<think>");
        assert!(b.forced_token().is_none());
        b.on_emit("x");
        assert_eq!(b.forced_token(), Some(7));
    }
}
