//! GPU worker surface. Callers run these methods on `spawn_blocking`;
//! the mutex serializes the single gfx906 context.

use std::sync::Mutex;

use crate::batch::BatchToken;
use crate::error::Result;
use crate::model::Model;
use crate::tokenizer::Token;

pub struct Engine {
    model: Mutex<Model>,
}

impl Engine {
    pub fn new(model: Model) -> Self {
        Self {
            model: Mutex::new(model),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Model> {
        self.model.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn n_ctx_seq(&self) -> u32 {
        self.lock().n_ctx_seq()
    }

    pub fn n_seq_max(&self) -> u32 {
        self.lock().n_seq_max()
    }

    pub fn n_batch(&self) -> u32 {
        self.lock().n_batch()
    }

    pub fn n_ubatch(&self) -> u32 {
        self.lock().n_ubatch()
    }

    pub fn n_vocab(&self) -> i32 {
        self.lock().n_vocab()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<Token>> {
        self.lock().encode(text)
    }

    pub fn token_to_piece(&self, token: Token) -> Result<Vec<u8>> {
        self.lock().tokenizer().token_to_piece(token, true)
    }

    pub fn is_eog(&self, token: Token) -> bool {
        self.lock().is_eog(token)
    }

    pub fn clear_seq(&self, seq_id: i32) {
        self.lock().clear_seq(seq_id);
    }

    /// Decode one planned batch and copy logits for every token that asked
    /// for them. Copies so the async side can sample after the lock drops.
    pub fn decode_and_logits(&self, items: &[BatchToken]) -> Result<Vec<Vec<f32>>> {
        let mut model = self.lock();
        model.decode_items(items)?;
        let mut rows = Vec::new();
        for (i, it) in items.iter().enumerate() {
            if it.logits {
                rows.push(model.logits_ith(i as i32)?.to_vec());
            }
        }
        Ok(rows)
    }
}
