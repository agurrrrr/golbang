use std::path::PathBuf;

#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    #[error("failed to load model from {}: {reason}", path.display())]
    Load { path: PathBuf, reason: String },
    #[error("tokenize failed: {0}")]
    Tokenize(String),
    #[error("detokenize failed: {0}")]
    Detokenize(String),
    #[error("llama_decode returned {0}")]
    Decode(i32),
    #[error("prompt length {prompt} exceeds context {n_ctx}")]
    ContextFull { prompt: u32, n_ctx: u32 },
    #[error("empty prompt")]
    EmptyPrompt,
    #[error("null pointer from llama.cpp ({0})")]
    Null(&'static str),
    #[error("batch of {got} tokens exceeds n_batch {n_batch}")]
    BatchTooLarge { got: usize, n_batch: usize },
    #[error("request cancelled")]
    Cancelled,
    #[error("request timed out")]
    Timeout,
    #[error("vision: {0}")]
    Vision(String),
    #[error("image input requires --mmproj")]
    VisionDisabled,
}

pub type Result<T> = std::result::Result<T, Error>;
