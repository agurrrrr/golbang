//! Safe wrappers over `golbang-sys`. Unsafe stays in this crate's FFI calls;
//! the public surface is RAII + iterators. P2 adds a swappable schedule loop.

mod batch;
mod chat;
mod engine;
mod error;
mod generate;
mod model;
mod policy;
mod sampler;
mod scheduler;
mod slot;
mod tokenizer;

pub use batch::{BatchBuilder, BatchPlan, BatchToken};
pub use chat::{apply_chat_template, AppliedPrompt, ChatMessage};
pub use engine::Engine;
pub use error::Error;
pub use generate::{FinishReason, Generate, GenerateParams, GeneratedToken};
pub use model::{LoadParams, Model};
pub use policy::{FifoPolicy, SchedulePolicy, SlotView, WaitingJobView};
pub use sampler::{Sampler, SamplerParams};
pub use scheduler::{
    spawn_scheduler, Job, SchedulerConfig, SchedulerHandle, SchedulerMetrics, SpawnedScheduler,
    SubmitError,
};
pub use slot::{Slot, SlotEvent, SlotId, SlotPhase};
pub use tokenizer::{Token, Tokenizer};
pub use tokio_util::sync::CancellationToken;
