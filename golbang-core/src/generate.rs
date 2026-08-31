//! Autoregressive loop split into one sample per `step` / `Iterator::next`.
//! P2's scheduler joins/evicts at this grain — do not fold a whole request
//! into a single blocking call that the scheduler cannot interrupt.

use crate::error::{Error, Result};
use crate::model::Model;
use crate::sampler::{Sampler, SamplerParams};
use crate::tokenizer::Token;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    Timeout,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::Cancelled => "cancelled",
            Self::Timeout => "timeout",
        }
    }
}

#[derive(Clone, Debug)]
pub struct GenerateParams {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub seed: u64,
    pub stop: Vec<String>,
    /// 0 = do not force `</think>`. Counts generated tokens while thinking.
    pub reasoning_budget: u32,
    /// Prompt already closed with `<think>` (Qwen3.8 / DSV4 jinja).
    pub start_in_think: bool,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
            stop: Vec::new(),
            reasoning_budget: 0,
            start_in_think: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GeneratedToken {
    pub token: Token,
    pub piece: String,
}

pub struct Generate<'m> {
    model: &'m mut Model,
    sampler: Sampler,
    pending: Option<Token>,
    n_prompt: u32,
    n_generated: u32,
    max_tokens: u32,
    stop: Vec<String>,
    acc: String,
    utf8: Utf8Buf,
    finished: bool,
    finish: Option<FinishReason>,
}

impl<'m> Generate<'m> {
    pub(crate) fn start(
        model: &'m mut Model,
        prompt: &str,
        params: GenerateParams,
    ) -> Result<Self> {
        if prompt.is_empty() {
            return Err(Error::EmptyPrompt);
        }
        let tokens = model.encode(prompt)?;
        if tokens.is_empty() {
            return Err(Error::EmptyPrompt);
        }
        // Single-sequence path: the pool (`n_ctx`), not `n_ctx_seq`. The
        // scheduler applies `--single-max-ctx` on top of this for slots.
        let n_ctx = model.n_ctx();
        let n_prompt = tokens.len() as u32;
        if n_prompt >= n_ctx {
            return Err(Error::ContextFull {
                prompt: n_prompt,
                n_ctx,
            });
        }

        model.clear_kv();
        model.decode(&tokens)?;

        let remaining = n_ctx - n_prompt;
        let max_tokens = params.max_tokens.max(1).min(remaining);
        let sampler = Sampler::new(SamplerParams {
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            seed: params.seed,
        });

        Ok(Self {
            model,
            sampler,
            pending: None,
            n_prompt,
            n_generated: 0,
            max_tokens,
            stop: params.stop,
            acc: String::new(),
            utf8: Utf8Buf::default(),
            finished: false,
            finish: None,
        })
    }

    pub fn prompt_tokens(&self) -> u32 {
        self.n_prompt
    }

    pub fn completion_tokens(&self) -> u32 {
        self.n_generated
    }

    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.finish
    }

    /// One sample. Decode the previously yielded token first (if any).
    pub fn step(&mut self) -> Result<Option<GeneratedToken>> {
        if self.finished {
            return Ok(None);
        }

        if let Some(tok) = self.pending.take() {
            if self.model.n_past() + 1 > self.model.n_ctx() {
                self.finish_with(FinishReason::Length);
                return Ok(None);
            }
            self.model.decode(&[tok])?;
        }

        let token = {
            let logits = self.model.logits()?;
            self.sampler.sample(logits)
        };

        if self.model.is_eog(token) {
            let _ = self.utf8.flush();
            self.finish_with(FinishReason::Stop);
            return Ok(None);
        }

        let bytes = self.model.tokenizer().token_to_piece(token, true)?;
        let mut piece = self.utf8.push(&bytes);
        self.n_generated += 1;

        if self.n_generated >= self.max_tokens {
            piece.push_str(&self.utf8.flush());
            self.finish_with(FinishReason::Length);
            return Ok(Some(GeneratedToken { token, piece }));
        }

        if !self.stop.is_empty() {
            self.acc.push_str(&piece);
            if let Some(keep) = stop_cut_len(&self.stop, &self.acc, piece.len()) {
                piece.truncate(keep);
                self.finish_with(FinishReason::Stop);
                return Ok(Some(GeneratedToken { token, piece }));
            }
        }

        self.pending = Some(token);
        Ok(Some(GeneratedToken { token, piece }))
    }

    fn finish_with(&mut self, reason: FinishReason) {
        self.finished = true;
        self.finish = Some(reason);
    }
}

/// Stop-sequence check over the accumulated text. Returns how many bytes of
/// the piece emitted this step survive: the first-encountered stop sequence
/// (OpenAI rule: cut at the earliest occurrence) and anything after it are
/// cut so e.g. a literal `<|im_end|>` cannot leak into the visible response.
/// A stop that began inside an already-emitted piece keeps 0 (that piece was
/// streamed before the match completed).
pub(crate) fn stop_cut_len(stop: &[String], acc: &str, piece_len: usize) -> Option<usize> {
    let start = stop
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| acc.rfind(s.as_str()))
        .min()?;
    let piece_start = acc.len().saturating_sub(piece_len);
    Some(start.saturating_sub(piece_start))
}

impl Iterator for Generate<'_> {
    type Item = Result<GeneratedToken>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.step() {
            Ok(Some(t)) => Some(Ok(t)),
            Ok(None) => None,
            Err(e) => {
                self.finished = true;
                Some(Err(e))
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct Utf8Buf {
    buf: Vec<u8>,
}

impl Utf8Buf {
    pub(crate) fn push(&mut self, bytes: &[u8]) -> String {
        self.buf.extend_from_slice(bytes);
        self.take_valid()
    }

    pub(crate) fn flush(&mut self) -> String {
        if self.buf.is_empty() {
            return String::new();
        }
        let s = String::from_utf8_lossy(&self.buf).into_owned();
        self.buf.clear();
        s
    }

    fn take_valid(&mut self) -> String {
        match std::str::from_utf8(&self.buf) {
            Ok(s) => {
                let out = s.to_owned();
                self.buf.clear();
                out
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if valid == 0 {
                    if e.error_len().is_some() {
                        self.buf.remove(0);
                        let mut rest = self.take_valid();
                        rest.insert(0, '\u{FFFD}');
                        return rest;
                    }
                    return String::new();
                }
                let out = std::str::from_utf8(&self.buf[..valid]).unwrap().to_owned();
                self.buf.drain(..valid);
                out
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single-token stop (`<|im_end|>` piece) must not be streamed: keep 0
    /// bytes of the piece that completed the match.
    #[test]
    fn stop_cut_len_whole_piece_stop_keeps_nothing() {
        let stop = vec!["<|im_end|>".to_string()];
        let acc = "안녕하세요<|im_end|>";
        assert_eq!(stop_cut_len(&stop, acc, "<|im_end|>".len()), Some(0));
    }

    /// Stop begins mid-piece: keep the bytes before the stop start.
    #[test]
    fn stop_cut_len_mid_piece_stop_keeps_prefix() {
        let stop = vec!["<|user|>".to_string()];
        let piece = "텍<|user|>";
        let acc = format!("앞{piece}");
        assert_eq!(stop_cut_len(&stop, &acc, piece.len()), Some("텍".len()));
    }

    /// Stop began in a piece that was already streamed: keep 0.
    #[test]
    fn stop_cut_len_stop_started_earlier_keeps_zero() {
        let stop = vec!["END".to_string()];
        let acc = "ENEND";
        assert_eq!(stop_cut_len(&stop, acc, "ND".len()), Some(0));
    }

    /// No match: keep streaming (None).
    #[test]
    fn stop_cut_len_no_match_is_none() {
        let stop = vec!["<|im_end|>".to_string()];
        assert_eq!(stop_cut_len(&stop, "그냥 텍스트", 6), None);
    }

    /// Overlapping stops: the first-encountered (earliest) start wins,
    /// matching the OpenAI "cut at the first stop sequence" rule.
    #[test]
    fn stop_cut_len_picks_earliest_start() {
        let stop = vec!["AAA".to_string(), "AAB".to_string()];
        let acc = "xxAABAAA";
        assert_eq!(stop_cut_len(&stop, acc, acc.len()), Some(2));
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;
    use crate::model::{LoadParams, Model};
    use std::env;
    use std::fs::OpenOptions;
    use std::os::unix::io::AsRawFd;
    use std::path::Path;

    fn lock_gpu() -> std::fs::File {
        let f = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open("/tmp/golbang-gpu-test.lock")
            .expect("gpu lock");
        let rc = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX) };
        assert_eq!(rc, 0, "flock");
        f
    }

    #[test]
    fn generate_stops_at_max_tokens() {
        let path = match env::var("GOLBANG_TEST_MODEL") {
            Ok(p) if !p.is_empty() => p,
            _ => {
                eprintln!("skip: set GOLBANG_TEST_MODEL");
                return;
            }
        };
        assert!(Path::new(&path).is_file(), "{path}");
        let _lock = lock_gpu();

        let mut model = Model::load(
            &path,
            LoadParams {
                n_ctx: 256,
                n_gpu_layers: 99,
                n_seq_max: 1,
                ..Default::default()
            },
        )
        .expect("load");
        let params = GenerateParams {
            max_tokens: 4,
            temperature: 0.0,
            ..GenerateParams::default()
        };
        let n_vocab = model.n_vocab();
        let mut generation = model.generate("Hello", params).expect("prefill");
        let mut n = 0u32;
        for item in &mut generation {
            let t = item.expect("step");
            n += 1;
            assert!(t.token >= 0 && t.token < n_vocab);
        }
        assert!(n >= 1 && n <= 4, "got {n} tokens");
        assert!(generation.prompt_tokens() >= 1);
        assert_eq!(generation.completion_tokens(), n);
        let reason = generation.finish_reason().expect("finish");
        assert!(matches!(reason, FinishReason::Length | FinishReason::Stop));
    }
}
