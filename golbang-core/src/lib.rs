//! Safe wrappers over `golbang-sys`. Unsafe stays in this crate's FFI calls;
//! the public surface is RAII + iterators.

mod chat;
mod error;
mod generate;
mod model;
mod sampler;
mod tokenizer;

pub use chat::{apply_chat_template, AppliedPrompt, ChatMessage};
pub use error::Error;
pub use generate::{FinishReason, Generate, GenerateParams, GeneratedToken};
pub use model::{LoadParams, Model};
pub use sampler::{Sampler, SamplerParams};
pub use tokenizer::{Token, Tokenizer};
