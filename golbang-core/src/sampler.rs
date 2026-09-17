//! temperature / top-k / top-p over a logits row. Pure Rust so P2 can reuse
//! the same step without pulling llama.cpp sampler chains.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::tokenizer::Token;

#[derive(Clone, Debug)]
pub struct SamplerParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub seed: u64,
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
        }
    }
}

pub struct Sampler {
    temperature: f32,
    top_p: f32,
    top_k: i32,
    rng: StdRng,
}

impl Sampler {
    pub fn new(params: SamplerParams) -> Self {
        let seed = if params.seed == 0 {
            entropy_seed()
        } else {
            params.seed
        };
        Self {
            temperature: params.temperature,
            top_p: params.top_p.clamp(0.0, 1.0),
            top_k: params.top_k,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// temperature <= 0.0 is argmax (`sample` returns `argmax`). PLD is gated
    /// to greedy steps so its copy proposal matches the target's choice.
    pub fn greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// Sample one token id from a full-vocab logits row.
    pub fn sample(&mut self, logits: &[f32]) -> Token {
        assert!(!logits.is_empty(), "empty logits");

        if self.temperature <= 0.0 {
            return argmax(logits);
        }

        let mut pairs: Vec<(Token, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i as Token, l / self.temperature))
            .collect();

        if self.top_k > 0 && (self.top_k as usize) < pairs.len() {
            let k = self.top_k as usize;
            pairs.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
            pairs.truncate(k);
        }

        let max_l = pairs.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for p in &mut pairs {
            p.1 = (p.1 - max_l).exp();
            sum += p.1;
        }
        if !sum.is_finite() || sum <= 0.0 {
            return argmax(logits);
        }
        for p in &mut pairs {
            p.1 /= sum;
        }

        if self.top_p > 0.0 && self.top_p < 1.0 {
            pairs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            let mut cum = 0.0;
            let mut keep = 0;
            for (i, p) in pairs.iter().enumerate() {
                cum += p.1;
                keep = i + 1;
                if cum >= self.top_p {
                    break;
                }
            }
            pairs.truncate(keep.max(1));
            let s: f32 = pairs.iter().map(|p| p.1).sum();
            if s > 0.0 {
                for p in &mut pairs {
                    p.1 /= s;
                }
            }
        }

        let r: f32 = self.rng.gen_range(0.0..1.0);
        let mut cum = 0.0;
        for (id, p) in &pairs {
            cum += *p;
            if r <= cum {
                return *id;
            }
        }
        pairs.last().map(|p| p.0).unwrap_or(0)
    }
}

fn argmax(logits: &[f32]) -> Token {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as Token)
        .unwrap_or(0)
}

fn entropy_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xC0FFEE)
        ^ std::process::id() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_zero_is_argmax() {
        let mut s = Sampler::new(SamplerParams {
            temperature: 0.0,
            seed: 1,
            ..SamplerParams::default()
        });
        assert_eq!(s.sample(&[1.0, 5.0, 2.0]), 1);
    }

    #[test]
    fn top_k_one_is_argmax() {
        let mut s = Sampler::new(SamplerParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 1,
            seed: 1,
        });
        assert_eq!(s.sample(&[1.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn deterministic_with_seed() {
        let logits = [0.1, 0.2, 0.15, 0.05];
        let a = Sampler::new(SamplerParams {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 0,
            seed: 42,
        })
        .sample(&logits);
        let b = Sampler::new(SamplerParams {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 0,
            seed: 42,
        })
        .sample(&logits);
        assert_eq!(a, b);
    }
}
