use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
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
}

pub type Result<T> = std::result::Result<T, Error>;
