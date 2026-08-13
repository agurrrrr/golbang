//! Safe wrappers over `golbang-sys`. Unsafe stays in this crate's FFI calls;
//! the public surface is RAII + iterators. P2 adds a swappable schedule loop.

mod batch;
mod chat;
mod engine;
mod error;
mod generate;
mod model;
mod policy;
mod prefix_cache;
mod reasoning;
mod sampler;
mod scheduler;
mod slot;
mod tokenizer;

pub use batch::{BatchBuilder, BatchPlan, BatchToken};
pub use chat::{
    AppliedPrompt, ChatApplyOpts, ChatMessage, apply_chat_template, apply_chat_template_with,
};
pub use engine::Engine;
pub use error::Error;
pub use generate::{FinishReason, Generate, GenerateParams, GeneratedToken};
pub use model::{LoadParams, Model};
pub use policy::{FifoPolicy, SchedulePolicy, SlotView, WaitingJobView};
pub use prefix_cache::{PrefixStore, SlotPrefixCache, common_prefix_len};
pub use reasoning::{ReasoningDelta, ReasoningFormat, ReasoningParser};
pub use sampler::{Sampler, SamplerParams};
pub use scheduler::{
    Job, SchedulerConfig, SchedulerHandle, SchedulerMetrics, SpawnedScheduler, SubmitError,
    spawn_scheduler,
};
pub use slot::{Slot, SlotEvent, SlotId, SlotPhase, SlotTimings};
pub use tokenizer::{Token, Tokenizer};
pub use tokio_util::sync::CancellationToken;
