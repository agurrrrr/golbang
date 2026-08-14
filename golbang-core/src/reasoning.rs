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

    /// If the prompt already opened `<think>` (DSV4 thinking mode), start in think.
    pub fn from_prompt(format: ReasoningFormat, prompt: &str) -> Self {
        let start_in_think = format.extracts() && prompt.ends_with(OPEN);
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
}
