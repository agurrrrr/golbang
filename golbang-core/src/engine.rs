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

    pub fn spec_enabled(&self) -> bool {
        self.lock().spec_enabled()
    }

    pub fn spec_n_max(&self) -> i32 {
        self.lock().spec_n_max()
    }

    pub fn n_rs_seq(&self) -> u32 {
        self.lock().n_rs_seq()
    }

    pub fn vision_enabled(&self) -> bool {
        self.lock().vision_enabled()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<Token>> {
        self.lock().encode(text)
    }

    /// Tokenize without BOS. Used for `</think>` forcing (`parse_special`).
    pub fn tokenize_special(&self, text: &str) -> Result<Vec<Token>> {
        self.lock().tokenizer().encode(text, false, true)
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

    /// Remove KV from position `p0` to the end of `seq_id`. See [`Model::rm_seq_from`].
    pub fn rm_seq_from(&self, seq_id: i32, p0: i32) -> bool {
        self.lock().rm_seq_from(seq_id, p0)
    }

    pub fn n_past_seq(&self, seq_id: i32) -> u32 {
        self.lock().n_past_seq(seq_id)
    }

    pub fn seq_state_get(&self, seq_id: i32) -> Option<Vec<u8>> {
        self.lock().seq_state_get(seq_id)
    }

    pub fn seq_state_set(&self, seq_id: i32, data: &[u8]) -> bool {
        self.lock().seq_state_set(seq_id, data)
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
        if let Err(e) = model.spec_process(items) {
            tracing::warn!(error = %e, "MTP process() failed; drafts may degrade");
        }
        Ok(rows)
    }

    /// Decode, sample each requested logits row in place, then run MTP
    /// `process()`. Qwen3.8 vocab is 248k; copying those rows to the async
    /// side was several megabytes per verify step.
    ///
    /// `sample_row(logit_i, row)` is called once per `items[i].logits`.
    pub fn decode_and_sample(
        &self,
        items: &[BatchToken],
        sample_row: &mut dyn FnMut(usize, &[f32]) -> Token,
    ) -> Result<Vec<Token>> {
        let mut model = self.lock();
        model.decode_items(items)?;
        let mut samples = Vec::new();
        let mut logit_i = 0usize;
        for (i, it) in items.iter().enumerate() {
            if it.logits {
                let row = model.logits_ith(i as i32)?;
                samples.push(sample_row(logit_i, row));
                logit_i += 1;
            }
        }
        if let Err(e) = model.spec_process(items) {
            tracing::warn!(error = %e, "MTP process() failed; drafts may degrade");
        }
        Ok(samples)
    }

    pub fn last_logits(&self) -> Result<Vec<f32>> {
        self.lock().last_logits_vec()
    }

    pub fn spec_begin(&self, seq_id: i32, prompt: &[Token]) {
        self.lock().spec_begin(seq_id, prompt);
    }

    pub fn spec_reset_seq(&self, seq_id: i32) {
        self.lock().spec_reset_seq(seq_id);
    }

    pub fn spec_draft(
        &self,
        seq_id: i32,
        prompt: &[Token],
        id_last: Token,
        n_past: i32,
        n_max: i32,
    ) -> Vec<Token> {
        self.lock()
            .spec_draft(seq_id, prompt, id_last, n_past, n_max)
    }

    pub fn spec_accept(&self, seq_id: i32, n_accepted: u16) {
        self.lock().spec_accept(seq_id, n_accepted);
    }

    pub fn spec_rm_from(&self, seq_id: i32, p0: i32) -> bool {
        self.lock().spec_rm_from(seq_id, p0)
    }

    pub fn spec_state_get(&self, seq_id: i32) -> Option<Vec<u8>> {
        self.lock().spec_state_get(seq_id)
    }

    pub fn spec_state_set(&self, seq_id: i32, data: &[u8]) -> bool {
        self.lock().spec_state_set(seq_id, data)
    }

    pub fn vision_eval(&self, seq_id: i32, prompt: &str, images: &[Vec<u8>]) -> Result<u32> {
        self.lock().vision_eval(seq_id, prompt, images)
    }
}
