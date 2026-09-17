//! Prefix cache: reuse KV of a common prompt prefix across slots.
//!
//! P3 §4.0 spike confirmed `llama_memory_seq_cp` / `llama_memory_seq_keep` /
//! `llama_memory_seq_rm` are bound (llama.cpp HIP pin `367ebbc20`, was
//! `3ac5658c7`). We use the
//! **slot-local reuse** path: each slot keeps the KV of its own prefix so a
//! re-bound job that shares the same prefix does not re-prefill it.
//!
//! P5: Stop/Length/Cancel/Timeout leave that KV resident. `remember` stores
//! prompt + generated token IDs so the next bind's LCP can include the
//! previous assistant turn, or a mid-prefill prefix after a client drop.
//! P8-B: `PrefixStore` holds host `seq_state` dumps of the shared tool/system
//! head. Empty-slot bind restores the longest matching dump (`seq_state_set`).
//! There is still no `llama_memory_seq_cp` between slots.
//!
//! Vision (Qwen3.8 `--mmproj`): `VisionSeq` is llama-server `server_tokens`
//! for one slot. Image cells compare FNV chunk ids, not vocab tokens. M-RoPE
//! uses `n_pos != n_tokens`; `pos_next` is what `seq_rm` / `n_past` need.

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use crate::slot::SeqCheckpoint;
use crate::tokenizer::Token;

/// Longest common prefix length between two token slices.
pub fn common_prefix_len(a: &[Token], b: &[Token]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}

/// Prefix tokens `[0..n_tokens)` used as a [`PrefixStore`] key.
///
/// `None` when `n_tokens` is 0 or longer than `prompt` — a store entry must
/// be an exact head of the prompt that produced the dump.
pub fn snapshot_key(prompt: &[Token], n_tokens: u32) -> Option<Vec<Token>> {
    let n = n_tokens as usize;
    if n == 0 || n > prompt.len() {
        None
    } else {
        Some(prompt[..n].to_vec())
    }
}

/// Search length for a bind-time store lookup.
///
/// `reuse_len == 0` is an empty slot (no local LCP). Search up to
/// `prompt.len() - 1` so a 12k tool-head dump can still hit, while leaving
/// one logits token.
pub fn host_search_len(prompt_len: usize, reuse_len: usize) -> usize {
    if reuse_len == 0 {
        prompt_len.saturating_sub(1)
    } else {
        reuse_len
    }
}

/// One mtmd chunk after `mtmd_tokenize`. Image/audio ids are FNV hashes
/// from `mtmd_helper_bitmap_init_from_buf`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VisionChunk {
    Text(Vec<Token>),
    Media {
        id: String,
        n_tokens: u32,
        n_pos: u32,
    },
}

/// One image/audio span in a flattened vision prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionImageSpan {
    pub start: usize,
    pub id: String,
    pub n_tokens: u32,
    pub n_pos: u32,
}

/// Flattened text tokens + media spans. Image cells use token id 0; LCP
/// compares span ids, not those placeholders. Qwen3.8 M-RoPE: `n_pos`
/// for an image is `max(nx, ny)`, smaller than `n_tokens`.
#[derive(Clone, Debug, Default)]
pub struct VisionSeq {
    pub tokens: Vec<Token>,
    pub images: Vec<VisionImageSpan>,
}

impl VisionSeq {
    pub fn from_chunks(chunks: impl IntoIterator<Item = VisionChunk>) -> Self {
        let mut seq = Self::default();
        for chunk in chunks {
            match chunk {
                VisionChunk::Text(toks) => seq.tokens.extend(toks),
                VisionChunk::Media {
                    id,
                    n_tokens,
                    n_pos,
                } => {
                    let start = seq.tokens.len();
                    seq.tokens
                        .extend(std::iter::repeat(0).take(n_tokens as usize));
                    seq.images.push(VisionImageSpan {
                        start,
                        id,
                        n_tokens,
                        n_pos,
                    });
                }
            }
        }
        seq
    }

    pub fn n_tokens(&self) -> usize {
        self.tokens.len()
    }

    pub fn n_pos(&self) -> u32 {
        self.pos_next(self.tokens.len())
    }

    pub fn image_at(&self, idx: usize) -> Option<&VisionImageSpan> {
        self.images.iter().find(|im| im.start == idx)
    }

    pub fn image_covering(&self, idx: usize) -> Option<&VisionImageSpan> {
        self.images
            .iter()
            .find(|im| idx >= im.start && idx < im.start.saturating_add(im.n_tokens as usize))
    }

    /// Position after the first `n_tokens` cells (M-RoPE-aware).
    pub fn pos_next(&self, n_tokens: usize) -> u32 {
        let n = n_tokens.min(self.tokens.len());
        let mut idx = 0;
        let mut pos = 0u32;
        while idx < n {
            if let Some(img) = self.image_at(idx) {
                pos = pos.saturating_add(img.n_pos);
                idx = idx.saturating_add(img.n_tokens as usize);
            } else {
                pos = pos.saturating_add(1);
                idx += 1;
            }
        }
        pos
    }

    /// Token index whose position is `max_pos` (llama-server `size_up_to_pos`).
    pub fn size_up_to_pos(&self, max_pos: u32) -> usize {
        let mut idx = 0;
        let mut pos = 0u32;
        while idx < self.tokens.len() && pos < max_pos {
            if let Some(img) = self.image_at(idx) {
                pos = pos.saturating_add(img.n_pos);
                idx = idx.saturating_add(img.n_tokens as usize);
            } else {
                pos = pos.saturating_add(1);
                idx += 1;
            }
        }
        idx
    }

    /// llama-server `get_common_prefix` with media: whole image or none.
    pub fn common_prefix(&self, other: &Self) -> usize {
        let max = self.tokens.len().min(other.tokens.len());
        let mut i = 0;
        while i < max {
            match (self.image_at(i), other.image_at(i)) {
                (Some(a), Some(b)) => {
                    if !a.id.is_empty() && a.id == b.id && a.n_tokens == b.n_tokens {
                        i = i.saturating_add(a.n_tokens as usize);
                        continue;
                    }
                    return i;
                }
                (None, None) => {
                    if self.tokens[i] != other.tokens[i] {
                        return i;
                    }
                    i += 1;
                }
                _ => return i,
            }
        }
        max
    }

    /// Bind-time reuse in **token cells**. Leaves one cell for logits and
    /// never splits an image. 0 → caller `clear_seq`.
    pub fn reuse_for_bind(&self, prompt: &Self, gpu_n_past: u32) -> usize {
        if prompt.tokens.is_empty() {
            return 0;
        }
        let mut n = self.common_prefix(prompt);
        n = n.min(prompt.tokens.len() - 1);
        if let Some(img) = prompt.image_covering(n) {
            n = img.start;
        }
        if n == 0 || gpu_n_past < prompt.pos_next(n) {
            return 0;
        }
        n
    }

    pub fn append_generated(&mut self, generated: &[Token]) {
        self.tokens.extend_from_slice(generated);
    }

    /// Drop cells whose position is at or past `n_pos` (cancel mid-decode).
    /// A partial image at the cut is dropped whole.
    pub fn truncate_to_pos(&mut self, n_pos: u32) {
        let mut idx = 0;
        let mut pos = 0u32;
        while idx < self.tokens.len() && pos < n_pos {
            if let Some(img) = self.image_at(idx) {
                if pos.saturating_add(img.n_pos) > n_pos {
                    break;
                }
                pos = pos.saturating_add(img.n_pos);
                idx = idx.saturating_add(img.n_tokens as usize);
            } else {
                pos = pos.saturating_add(1);
                idx += 1;
            }
        }
        self.tokens.truncate(idx);
        self.images.retain(|im| im.start < idx);
    }
}

/// Per-slot cache of the last bound prompt's prefix KV.
///
/// `tokens` is the prefix whose KV is already resident in the slot's seq.
/// When a new job shares this prefix, we skip prefilling those tokens and
/// start from `prefix_len`.
#[derive(Clone, Debug, Default)]
pub struct SlotPrefixCache {
    pub tokens: Vec<Token>,
    pub prefix_len: usize,
    /// Last vision prompt (+ generated text cells). `None` after a text bind
    /// or a full reset. Next `--mmproj` bind LCPs against this, not `tokens`.
    pub vision: Option<VisionSeq>,
}

impl SlotPrefixCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Given the new job's prompt tokens, decide how many leading tokens can
    /// be reused from the cached prefix. Returns (reused_len, cache).
    ///
    /// The caller then starts prefill at `reused_len` (the KV for tokens
    /// `[0..reused_len)` is already resident in the slot's sequence).
    pub fn reuse(&mut self, prompt: &[Token]) -> usize {
        let len = common_prefix_len(&self.tokens, prompt);
        // Keep the matched prefix as the new cache (so it grows/shrinks with
        // the observed workload).
        if len > 0 {
            self.tokens = prompt[..len].to_vec();
        }
        self.prefix_len = len;
        len
    }

    pub fn reset(&mut self) {
        self.tokens.clear();
        self.prefix_len = 0;
        self.vision = None;
    }

    pub fn remember_vision(&mut self, mut seq: VisionSeq, generated: &[Token], n_pos: u32) {
        seq.append_generated(generated);
        seq.truncate_to_pos(n_pos);
        self.prefix_len = seq.n_tokens();
        self.vision = Some(seq);
        self.tokens.clear();
    }

    /// After a successful Stop/Length, record the tokens whose KV is actually
    /// resident (`n_past` == `llama_memory_seq_pos_max + 1`). The last sampled
    /// token may not have been decoded yet, so `n_past` can be shorter than
    /// `prompt.len() + generated.len()`.
    pub fn remember(&mut self, prompt: &[Token], generated: &[Token], n_past: u32) {
        self.tokens.clear();
        self.tokens.extend_from_slice(prompt);
        self.tokens.extend_from_slice(generated);
        self.tokens.truncate(n_past as usize);
        self.prefix_len = self.tokens.len();
    }

    /// Bind-time reuse. Returns how many leading prompt tokens keep their KV.
    ///
    /// Always leaves at least one prompt token to prefill so the last cell has
    /// logits (llama-server `TAG_PROMPT_LOGITS`). `gpu_n_past` is the
    /// restorable cover: live GPU occupancy, or after a watermark
    /// `gpu_n.max(ckpt_n)`. LCP above that cover is clamped so session
    /// identity stays and a shorter host dump can still restore. Returns 0
    /// only when nothing is restorable (`clear_seq`).
    pub fn reuse_for_bind(&mut self, prompt: &[Token], gpu_n_past: u32) -> usize {
        if prompt.is_empty() {
            self.reset();
            return 0;
        }
        let mut n = common_prefix_len(&self.tokens, prompt);
        n = n.min(prompt.len() - 1);
        n = n.min(gpu_n_past as usize);
        if n == 0 {
            self.reset();
            return 0;
        }
        self.prefix_len = n;
        n
    }
}

/// Identity of the engine + weights + settings that produced a snapshot.
///
/// A dump restored under a different fingerprint can be silently wrong, so the
/// disk tier refuses any file whose fingerprint does not match. `engine_build`
/// is a compile-time tag; `model_path`/mtime/size pin the primary weights and
/// `n_ctx`/`config` cover settings that change the saved bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotFingerprint {
    pub engine_build: String,
    pub model_path: String,
    pub model_mtime: u64,
    pub model_size: u64,
    pub n_ctx: u32,
    pub config: String,
}

impl SnapshotFingerprint {
    pub fn new(
        engine_build: impl Into<String>,
        model_path: impl Into<String>,
        model_mtime: u64,
        model_size: u64,
        n_ctx: u32,
        config: impl Into<String>,
    ) -> Self {
        Self {
            engine_build: engine_build.into(),
            model_path: model_path.into(),
            model_mtime,
            model_size,
            n_ctx,
            config: config.into(),
        }
    }

    /// Fingerprint for a GGUF path (mtime/size read from the filesystem).
    /// Missing metadata records 0/0 — the path itself still discriminates.
    pub fn for_model(model: &Path, n_ctx: u32, config: impl Into<String>) -> Self {
        let (mtime, size) = file_stamp(model);
        Self::new(
            engine_build_tag(),
            model.display().to_string(),
            mtime,
            size,
            n_ctx,
            config,
        )
    }

    /// Stable FNV-1a 64 over all fields. Embedded in every disk file header and
    /// in the file name, so a new build/model never overwrites an old dump.
    pub fn hash(&self) -> u64 {
        let mut h = FNV_OFFSET;
        h = fnv1a(self.engine_build.as_bytes(), h);
        h = fnv1a(self.model_path.as_bytes(), h);
        h = fnv1a(&self.model_mtime.to_le_bytes(), h);
        h = fnv1a(&self.model_size.to_le_bytes(), h);
        h = fnv1a(&self.n_ctx.to_le_bytes(), h);
        fnv1a(self.config.as_bytes(), h)
    }
}

/// Compile-time build tag for the fingerprint. Code changes without a version
/// bump do not move this, but weights/settings do, which is the dangerous case.
fn engine_build_tag() -> String {
    format!(
        "{} {} {}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS
    )
}

fn file_stamp(path: &Path) -> (u64, u64) {
    match fs::metadata(path) {
        Ok(m) => {
            let mtime = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            (mtime, m.len())
        }
        Err(_) => (0, 0),
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a(bytes: &[u8], mut h: u64) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Disk file magic. `GBPK` = golbang prefix.
const DISK_MAGIC: &[u8; 4] = b"GBPK";
/// v2: disk entries are full-state sequence snapshots (v1 held PARTIAL_ONLY,
/// which is not self-sufficient on a fresh context). Bumping the version makes
/// an old v1 file fail the header check and get ignored.
const DISK_VERSION: u32 = 2;
/// Fixed header before the key: magic(4) + version(4) + fp(8) + n_tokens(4) +
/// key_len(4) = 24, then `key_len` i32 tokens, then the u64 data length.
const DISK_HEADER_FIXED: usize = 24;
/// Sanity caps so a corrupt header cannot trigger a huge allocation.
const DISK_MAX_KEY_TOKENS: u32 = 4_000_000;
const DISK_MAX_DATA_BYTES: u64 = 64 * 1024 * 1024 * 1024;

/// One disk-tier entry's metadata (kept in RAM; data lives in the file).
#[derive(Clone, Debug)]
struct DiskMeta {
    n_tokens: u32,
    file: String,
    bytes: u64,
    tick: u64,
}

/// Decoded disk file header (everything before `data`).
#[derive(Clone, Debug)]
struct DiskHeader {
    fp_hash: u64,
    n_tokens: u32,
    key: Vec<Token>,
    data_len: u64,
}

/// Disk tier for prefix snapshots (HAL-4 #248): a `PrefixStore`-shaped map
/// whose data lives in `<DIR>/<key-hash>.ckpt` and survives a restart.
///
/// Startup scans headers only (not the ~150 MiB payloads) into an in-memory
/// index. A bind lookup that misses the RAM map reads the matching file. Writes
/// are temp-file + rename so a crash cannot leave a half-written dump. LRU
/// eviction honours `max_bytes` (0 = unlimited).
#[derive(Clone, Debug)]
pub struct PrefixDisk {
    dir: PathBuf,
    fingerprint: SnapshotFingerprint,
    fp_hash: u64,
    max_bytes: u64,
    used_bytes: u64,
    index: HashMap<Vec<Token>, DiskMeta>,
    tick: u64,
}

impl PrefixDisk {
    /// Open (creating if needed) and scan `dir`. Mismatched/corrupt files are
    /// ignored, not deleted.
    pub fn open(
        dir: &Path,
        max_bytes: u64,
        fingerprint: SnapshotFingerprint,
    ) -> io::Result<Self> {
        fs::create_dir_all(dir)?;
        let fp_hash = fingerprint.hash();
        let mut disk = Self {
            dir: dir.to_path_buf(),
            fingerprint,
            fp_hash,
            max_bytes,
            used_bytes: 0,
            index: HashMap::new(),
            tick: 0,
        };
        disk.scan();
        disk.evict_over_cap();
        Ok(disk)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn fingerprint(&self) -> &SnapshotFingerprint {
        &self.fingerprint
    }

    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    fn scan(&mut self) {
        let Ok(rd) = fs::read_dir(&self.dir) else {
            return;
        };
        let mut skipped = 0usize;
        for ent in rd.flatten() {
            let path = ent.path();
            if path.extension().and_then(|e| e.to_str()) != Some("ckpt") {
                continue;
            }
            let header = match read_header(&path) {
                Ok(h) => h,
                Err(_) => {
                    skipped += 1;
                    continue;
                }
            };
            if header.fp_hash != self.fp_hash || header.key.is_empty() {
                skipped += 1;
                continue;
            }
            if header.n_tokens as usize != header.key.len() {
                skipped += 1;
                continue;
            }
            self.tick = self.tick.saturating_add(1);
            let tick = file_stamp(&path).0.max(self.tick);
            let file = file_name(&path);
            if let Some(old) = self.index.insert(
                header.key,
                DiskMeta {
                    n_tokens: header.n_tokens,
                    file,
                    bytes: header.data_len,
                    tick,
                },
            ) {
                self.used_bytes = self.used_bytes.saturating_sub(old.bytes);
            }
            self.used_bytes = self.used_bytes.saturating_add(header.data_len);
        }
        if skipped > 0 {
            tracing::warn!(
                skipped,
                dir = %self.dir.display(),
                "prefix disk: ignored mismatched/corrupt files"
            );
        }
    }

    /// Write a snapshot and update the index. Same key overwrites the file.
    pub fn put(&mut self, key: &[Token], ckpt: &SeqCheckpoint) -> io::Result<()> {
        if key.is_empty() || ckpt.data.is_empty() || key.len() != ckpt.n_tokens as usize {
            return Ok(());
        }
        let name = self.file_name(key);
        let path = self.dir.join(&name);
        let bytes = write_entry(&path, self.fp_hash, key, ckpt)?;
        let tick = self.next_tick();
        if let Some(old) = self.index.insert(
            key.to_vec(),
            DiskMeta {
                n_tokens: ckpt.n_tokens,
                file: name,
                bytes,
                tick,
            },
        ) {
            self.used_bytes = self.used_bytes.saturating_sub(old.bytes);
        }
        self.used_bytes = self.used_bytes.saturating_add(bytes);
        self.evict_over_cap();
        Ok(())
    }

    /// Longest usable entry for `prompt`. `search_len` is already
    /// `host_search_len(...)`-adjusted. Disk entries require **no rollback**
    /// (`n_tokens <= search_len`): restore the exact length and prefill the
    /// suffix. A rollback needs a token re-decode that the hybrid/indexer
    /// runtime cannot reconstruct, so such a hit is not offered here.
    pub fn find(&mut self, prompt: &[Token], search_len: usize) -> Option<SeqCheckpoint> {
        let mut best: Option<Vec<Token>> = None;
        for (key, meta) in &self.index {
            if meta.n_tokens == 0 || meta.n_tokens as usize > search_len {
                continue;
            }
            if common_prefix_len(key, prompt) != meta.n_tokens as usize {
                continue;
            }
            match &best {
                Some(b) if b.len() >= meta.n_tokens as usize => {}
                _ => best = Some(key.clone()),
            }
        }
        let key = best?;
        let meta = self.index.get(&key)?.clone();
        let data = read_entry_data(&self.dir.join(&meta.file)).ok()?;
        if data.len() as u64 != meta.bytes {
            return None;
        }
        self.touch(&key);
        Some(SeqCheckpoint {
            n_tokens: meta.n_tokens,
            data,
        })
    }

    fn file_name(&self, key: &[Token]) -> String {
        let mut h = fnv1a(&self.fp_hash.to_le_bytes(), FNV_OFFSET);
        for &t in key {
            h = fnv1a(&t.to_le_bytes(), h);
        }
        format!("{h:016x}.ckpt")
    }

    fn next_tick(&mut self) -> u64 {
        self.tick = self.tick.saturating_add(1);
        self.tick
    }

    fn touch(&mut self, key: &[Token]) {
        let t = self.next_tick();
        if let Some(m) = self.index.get_mut(key) {
            m.tick = t;
        }
    }

    /// Drop the oldest entries until `used_bytes <= max_bytes`.
    fn evict_over_cap(&mut self) {
        while self.max_bytes > 0 && self.used_bytes > self.max_bytes && !self.index.is_empty() {
            let Some(victim) = self
                .index
                .iter()
                .min_by_key(|(_, m)| m.tick)
                .map(|(k, _)| k.clone())
            else {
                break;
            };
            if let Some(meta) = self.index.remove(&victim) {
                self.used_bytes = self.used_bytes.saturating_sub(meta.bytes);
                let _ = fs::remove_file(self.dir.join(&meta.file));
            }
        }
    }
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_string()
}

fn write_entry(
    path: &Path,
    fp_hash: u64,
    key: &[Token],
    ckpt: &SeqCheckpoint,
) -> io::Result<u64> {
    let mut buf = Vec::with_capacity(DISK_HEADER_FIXED + key.len() * 4 + 8 + ckpt.data.len());
    buf.extend_from_slice(DISK_MAGIC);
    buf.extend_from_slice(&DISK_VERSION.to_le_bytes());
    buf.extend_from_slice(&fp_hash.to_le_bytes());
    buf.extend_from_slice(&ckpt.n_tokens.to_le_bytes());
    buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
    for &t in key {
        buf.extend_from_slice(&t.to_le_bytes());
    }
    buf.extend_from_slice(&(ckpt.data.len() as u64).to_le_bytes());
    buf.extend_from_slice(&ckpt.data);
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, &buf)?;
    fs::rename(&tmp, path)?;
    Ok(ckpt.data.len() as u64)
}

fn read_header(path: &Path) -> io::Result<DiskHeader> {
    let mut f = fs::File::open(path)?;
    read_header_from(&mut f)
}

fn read_header_from(f: &mut fs::File) -> io::Result<DiskHeader> {
    let mut fixed = [0u8; DISK_HEADER_FIXED];
    f.read_exact(&mut fixed)?;
    if &fixed[0..4] != DISK_MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
    }
    let version = u32::from_le_bytes(fixed[4..8].try_into().unwrap());
    if version != DISK_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported version",
        ));
    }
    let fp_hash = u64::from_le_bytes(fixed[8..16].try_into().unwrap());
    let n_tokens = u32::from_le_bytes(fixed[16..20].try_into().unwrap());
    let key_len = u32::from_le_bytes(fixed[20..24].try_into().unwrap());
    if key_len == 0 || key_len > DISK_MAX_KEY_TOKENS {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad key length"));
    }
    let mut key_buf = vec![0u8; key_len as usize * 4];
    f.read_exact(&mut key_buf)?;
    let key: Vec<Token> = key_buf
        .chunks_exact(4)
        .map(|c| Token::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mut dl = [0u8; 8];
    f.read_exact(&mut dl)?;
    let data_len = u64::from_le_bytes(dl);
    if data_len == 0 || data_len > DISK_MAX_DATA_BYTES {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad data length"));
    }
    Ok(DiskHeader {
        fp_hash,
        n_tokens,
        key,
        data_len,
    })
}

fn read_entry_data(path: &Path) -> io::Result<Vec<u8>> {
    let mut f = fs::File::open(path)?;
    let header = read_header_from(&mut f)?;
    let mut data = vec![0u8; header.data_len as usize];
    f.read_exact(&mut data)?;
    Ok(data)
}

/// Best-effort check for a filesystem that will not survive a reboot.
///
/// The disk tier only earns its keep if the bytes persist. `tmpfs` and
/// `ramfs` (and container `overlay` roots) are treated as volatile; a caller
/// should then stay on the RAM tier. Unknown mounts are assumed durable.
pub fn is_volatile_fs(path: &Path) -> bool {
    let Ok(mounts) = fs::read_to_string("/proc/mounts") else {
        return false;
    };
    let canon = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let mut best: Option<(usize, &str)> = None;
    for line in mounts.lines() {
        let mut it = line.split_whitespace();
        let _device = it.next();
        let Some(mount) = it.next() else { continue };
        let Some(fstype) = it.next() else { continue };
        if !canon.starts_with(mount) {
            continue;
        }
        let len = mount.len();
        if best.is_none_or(|(bl, _)| len >= bl) {
            best = Some((len, fstype));
        }
    }
    matches!(best.map(|(_, t)| t), Some("tmpfs" | "ramfs" | "overlay"))
}

/// Bind-miss numbers for the journal. Empty store → [`PrefixStore::miss_diag`]
/// returns `None` so a first-request miss stays silent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixStoreMiss {
    pub max_lcp: usize,
    /// Resident dump lengths, longest first. No key tokens.
    pub store_ns: Vec<u32>,
}

/// Per-entry eviction metadata (P9 borrow 2, SGLang session-reference aware
/// radix cache). `demand` marks a checkpoint captured at an observed reuse
/// boundary (P9 borrow 1); `refs` counts live sessions whose prompt has this
/// entry as an exact head. Both are **soft** protection: when only protected
/// entries remain they are still evicted, oldest-longest first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct EntryMeta {
    demand: bool,
    refs: u32,
}

/// Cross-slot host snapshot map (P8-B).
///
/// Each entry is a `SeqCheckpoint` captured at a known token length
/// (`n_tokens`), keyed by the prompt prefix tokens `[0..n_tokens)`. A bind
/// whose prompt matches an entry restores that dump instead of re-prefilling
/// the shared tool/system head. Lookup is a linear scan — no trie.
///
/// Caps: a single-digit entry limit plus a RAM ceiling. Eviction drops the
/// longest dump first so a 30k+ snapshot cannot push out the short tool head.
#[derive(Clone, Debug)]
pub struct PrefixStore {
    pub entries: HashMap<Vec<Token>, SeqCheckpoint>,
    /// Total bytes of all entry `data` (tracked for the RAM cap).
    pub total_bytes: usize,
    /// Soft RAM cap (bytes). 0 = unlimited.
    pub max_bytes: usize,
    /// Soft entry-count cap. 0 = unlimited. Default is a single digit.
    pub max_entries: usize,
    /// Monotonic "now" tick (advances on every insert/use).
    last_used_tick: u64,
    /// Per-entry tick of last use (keyed by the same prefix slice).
    last_used: HashMap<Vec<Token>, u64>,
    /// Per-entry eviction metadata (demand / session refs).
    meta: HashMap<Vec<Token>, EntryMeta>,
    /// P9 borrow 1: recently observed session prompts (heads) for LCP-based
    /// demand boundary discovery. Bounded ring.
    observed: VecDeque<Vec<Token>>,
    /// P9 borrow 1: boundary prefix -> times observed. The second observation
    /// arms a demand snapshot; one-off prefixes are never checkpointed.
    demand: HashMap<Vec<Token>, u8>,
    /// Optional disk tier (HAL-4 #248). `None` = RAM only, exactly P8.
    pub disk: Option<PrefixDisk>,
}

impl Default for PrefixStore {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            total_bytes: 0,
            max_bytes: 0,
            max_entries: Self::DEFAULT_MAX_ENTRIES,
            last_used_tick: 0,
            last_used: HashMap::new(),
            meta: HashMap::new(),
            observed: VecDeque::new(),
            demand: HashMap::new(),
            disk: None,
        }
    }
}

impl PrefixStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Recommended RAM cap for the shared tool/system head (~0.8 GiB for
    /// Qwen3.8 FA at 12545 tokens). 2 GiB leaves room for a few system prompts.
    pub const DEFAULT_MAX_BYTES: usize = 2 * 1024 * 1024 * 1024; // 2 GiB

    /// One tool head plus a handful of system prompts — never a ubatch trail.
    pub const DEFAULT_MAX_ENTRIES: usize = 8;

    /// Store-stub window. Scheduler promotion/eviction must use these same
    /// bounds. MIN sits below the current ~6191 tool-JSON head so a
    /// cross-project empty slot (LCP 6436, task #8910) can restore.
    pub const STORE_STUB_MIN: u32 = 6_144;
    pub const STORE_STUB_MAX: u32 = 16_384;
    pub const STORE_STUB_STRIDE: u32 = 2_048;

    /// P9 borrow 1: observed-prompt ring size and boundary-count cap.
    pub const OBSERVED_MAX: usize = 8;
    pub const DEMAND_MAX: usize = 32;
    /// Longest observed-prompt head kept for LCP (bounds host RAM).
    pub const OBSERVED_TOKEN_CAP: usize = 65_536;
    /// Ignore boundaries shorter than this: a tiny shared system head would
    /// arm a stop on every request for a one-iteration prefill split.
    pub const DEMAND_MIN_BOUNDARY: usize = 256;

    pub(crate) fn is_store_stub_len(n_tokens: u32) -> bool {
        (Self::STORE_STUB_MIN..=Self::STORE_STUB_MAX).contains(&n_tokens)
    }

    pub fn with_cap(max_bytes: usize) -> Self {
        let mut s = Self::new();
        s.max_bytes = max_bytes;
        s
    }

    pub fn with_limits(max_bytes: usize, max_entries: usize) -> Self {
        let mut s = Self::new();
        s.max_bytes = max_bytes;
        s.max_entries = max_entries;
        s
    }

    /// RAM tier plus a disk tier rooted at `dir` (HAL-4 #248). Scans existing
    /// `*.ckpt` headers into the index; a restart therefore reuses prior
    /// snapshots without any GPU work. `max_bytes` is the disk LRU cap.
    pub fn with_disk(
        dir: &Path,
        max_bytes: u64,
        fingerprint: SnapshotFingerprint,
    ) -> io::Result<Self> {
        let mut s = Self::with_cap(Self::DEFAULT_MAX_BYTES);
        s.disk = Some(PrefixDisk::open(dir, max_bytes, fingerprint)?);
        Ok(s)
    }

    /// Number of disk-tier entries (0 when the tier is off).
    pub fn disk_entries(&self) -> usize {
        self.disk.as_ref().map(PrefixDisk::len).unwrap_or(0)
    }

    /// Disk-tier bytes in use (0 when the tier is off).
    pub fn disk_bytes(&self) -> u64 {
        self.disk.as_ref().map(PrefixDisk::used_bytes).unwrap_or(0)
    }

    /// Persist a snapshot to the disk tier only, without touching the RAM map.
    ///
    /// The slot anchor chain (`Slot::prefix_ckpts`) holds dumps longer than the
    /// store's stub window (e.g. a 100k prefill-end anchor). Writing those to
    /// disk lets a restart restore the long session, while keeping them out of
    /// the 2 GiB RAM map avoids evicting the shared tool head.
    pub fn persist(&mut self, prefix: &[Token], ckpt: &SeqCheckpoint) {
        let Some(disk) = self.disk.as_mut() else {
            return;
        };
        if let Err(e) = disk.put(prefix, ckpt) {
            tracing::warn!(
                error = %e,
                dir = %disk.dir().display(),
                "prefix disk tier write failed"
            );
        }
    }

    /// Insert or refresh a snapshot for prefix `[0..n_tokens)`.
    ///
    /// Skips empty keys and key/`n_tokens` mismatches: those cannot hit
    /// `find_best` (`LCP == n_tokens`) and would only thrash the cap.
    pub fn put(&mut self, prefix: Vec<Token>, ckpt: SeqCheckpoint) {
        self.put_with(prefix, ckpt, false);
    }

    /// [`Self::put`] with an explicit demand flag (P9 borrow 1). A demand entry
    /// is a checkpoint captured at an observed reuse boundary, protected over
    /// plain fixed-stride stubs during eviction.
    pub fn put_with(&mut self, prefix: Vec<Token>, ckpt: SeqCheckpoint, demand: bool) {
        if prefix.is_empty() || ckpt.n_tokens == 0 {
            return;
        }
        if prefix.len() != ckpt.n_tokens as usize {
            return;
        }
        self.last_used_tick = self.last_used_tick.saturating_add(1);
        if let Some(old) = self.entries.insert(prefix.clone(), ckpt) {
            self.total_bytes = self.total_bytes.saturating_sub(old.data.len());
        }
        self.total_bytes = self.total_bytes.saturating_add(self.entry_bytes(&prefix));
        self.last_used.insert(prefix.clone(), self.last_used_tick);
        // A demand capture upgrades an existing stub; a stub never demotes a
        // demand entry.
        let meta = self.meta.entry(prefix).or_default();
        meta.demand |= demand;
        self.evict_if_over_cap();
    }

    /// Record a completed session prompt so a later prompt's LCP reveals the
    /// divergence boundary (P9 borrow 1). Bounded ring, deduped vs newest.
    pub fn record_prompt(&mut self, tokens: &[Token]) {
        if tokens.is_empty() {
            return;
        }
        let n = tokens.len().min(Self::OBSERVED_TOKEN_CAP);
        let head = &tokens[..n];
        if self.observed.back().is_some_and(|p| p.as_slice() == head) {
            return;
        }
        self.observed.push_back(head.to_vec());
        while self.observed.len() > Self::OBSERVED_MAX {
            self.observed.pop_front();
        }
    }

    /// LCP of `prompt` against observed prompts, when it is a real divergence
    /// (`reuse_len < b < prompt.len()`). Counts the boundary prefix; returns
    /// `Some(b)` on the second+ observation so the caller arms a prefill stop
    /// and snapshots exactly at `b` (Marconi selective retention).
    ///
    /// Returns `None` once the boundary is already resident: further binds hit
    /// it through `find_best`, so re-capturing would only churn the cap.
    pub fn observe_boundary(&mut self, prompt: &[Token], reuse_len: usize) -> Option<u32> {
        let mut best = 0usize;
        for p in &self.observed {
            let l = common_prefix_len(p, prompt);
            if l > best {
                best = l;
            }
        }
        if best < Self::DEMAND_MIN_BOUNDARY || best >= prompt.len() || best <= reuse_len {
            return None;
        }
        let key = prompt[..best].to_vec();
        let count = {
            let c = self.demand.entry(key.clone()).or_insert(0);
            *c = c.saturating_add(1);
            *c
        };
        while self.demand.len() > Self::DEMAND_MAX {
            if let Some(k) = self.demand.keys().next().cloned() {
                self.demand.remove(&k);
            } else {
                break;
            }
        }
        if count >= 2 && !self.entries.contains_key(&key) {
            Some(best as u32)
        } else {
            None
        }
    }

    /// Recompute per-entry session reference counts from the live session
    /// prompts (P9 borrow 2). An entry is referenced when it is an exact head
    /// of at least one live prompt. SGLang's `/close_session` has no golbang
    /// counterpart, so refs are recomputed on each cap enforcement instead of
    /// tracked incrementally.
    pub fn refresh_entry_refs(&mut self, live: &[&[Token]]) {
        for meta in self.meta.values_mut() {
            meta.refs = 0;
        }
        for (key, meta) in self.meta.iter_mut() {
            let mut count = 0u32;
            for p in live {
                if common_prefix_len(key, p) == key.len() {
                    count = count.saturating_add(1);
                }
            }
            meta.refs = count;
        }
    }

    /// Find the longest usable snapshot for `prompt`.
    ///
    /// An entry is usable when its LCP with `prompt` equals `n_tokens` (the
    /// whole snapshot prefix still matches) and `n_tokens <= reuse_len + 1`
    /// (the GPU can restore it and trim one logits token). Among usable
    /// entries, returns the longest. Linear scan.
    ///
    /// The RAM map is searched first. On a miss the disk tier (if enabled) is
    /// consulted with the same usability rule, so a restarted process can
    /// restore a session it never held in RAM.
    pub fn find_best(&mut self, prompt: &[Token], reuse_len: usize) -> Option<SeqCheckpoint> {
        if let Some(hit) = self.find_ram(prompt, reuse_len) {
            return Some(hit);
        }
        self.disk
            .as_mut()
            .and_then(|disk| disk.find(prompt, reuse_len))
    }

    /// Bind-time lookup. `reuse_len == 0` still searches the new prompt
    /// (`prompt.len() - 1`) so an empty slot can restore the shared head.
    pub fn find_best_for_bind(
        &mut self,
        prompt: &[Token],
        reuse_len: usize,
    ) -> Option<SeqCheckpoint> {
        self.find_best(prompt, host_search_len(prompt.len(), reuse_len))
    }

    /// RAM-map bind lookup only. Used by `restore_host_snapshot` so a disk hit
    /// (full-state snapshot) can take a different restore path.
    pub fn find_best_ram_for_bind(
        &mut self,
        prompt: &[Token],
        reuse_len: usize,
    ) -> Option<SeqCheckpoint> {
        self.find_ram(prompt, host_search_len(prompt.len(), reuse_len))
    }

    /// Disk-tier bind lookup only. Entries are full-state snapshots, so the
    /// caller must restore them with `seq_state_full_set`.
    pub fn find_disk_for_bind(
        &mut self,
        prompt: &[Token],
        reuse_len: usize,
    ) -> Option<SeqCheckpoint> {
        let search_len = host_search_len(prompt.len(), reuse_len);
        self.disk
            .as_mut()
            .and_then(|disk| disk.find(prompt, search_len))
    }

    fn find_ram(&mut self, prompt: &[Token], reuse_len: usize) -> Option<SeqCheckpoint> {
        let mut best: Option<(u32, Vec<Token>)> = None;
        for (prefix, ckpt) in &self.entries {
            if ckpt.n_tokens == 0 || ckpt.n_tokens > reuse_len as u32 + 1 {
                continue;
            }
            if common_prefix_len(prefix, prompt) != ckpt.n_tokens as usize {
                continue;
            }
            match best {
                Some((n, _)) if n >= ckpt.n_tokens => {}
                _ => best = Some((ckpt.n_tokens, prefix.clone())),
            }
        }
        if let Some((_, prefix)) = best {
            self.touch(&prefix);
            return Some(self.entries[&prefix].clone());
        }
        None
    }

    /// Numbers for a bind miss. `None` when the store is empty so a cold
    /// start does not warn. Does not include key tokens.
    pub fn miss_diag(&self, prompt: &[Token]) -> Option<PrefixStoreMiss> {
        if self.entries.is_empty() {
            return None;
        }
        let mut store_ns: Vec<u32> = self.entries.values().map(|c| c.n_tokens).collect();
        store_ns.sort_unstable_by(|a, b| b.cmp(a));
        let max_lcp = self
            .entries
            .keys()
            .map(|k| common_prefix_len(k, prompt))
            .max()
            .unwrap_or(0);
        Some(PrefixStoreMiss { max_lcp, store_ns })
    }

    /// Mark an entry recently used (advances the tick without changing data).
    fn touch(&mut self, prefix: &[Token]) {
        self.last_used_tick = self.last_used_tick.saturating_add(1);
        if let Some(t) = self.last_used.get_mut(prefix) {
            *t = self.last_used_tick;
        }
    }

    fn entry_bytes(&self, prefix: &[Token]) -> usize {
        self.entries.get(prefix).map(|c| c.data.len()).unwrap_or(0)
    }

    fn over_cap(&self) -> bool {
        (self.max_bytes > 0 && self.total_bytes > self.max_bytes)
            || (self.max_entries > 0 && self.entries.len() > self.max_entries)
    }

    /// Drop the longest dump until the store's own caps hold. Includes 6k–16k
    /// stubs: production `put` only inserts those, so skipping them would
    /// make `max_entries` / `max_bytes` a no-op.
    fn evict_if_over_cap(&mut self) {
        while self.over_cap() && !self.entries.is_empty() {
            let victim = self.pick_longest_victim(false);
            let Some(victim) = victim else {
                return;
            };
            self.remove_entry(&victim);
        }
    }

    /// Remove one entry and all its bookkeeping. Returns nothing; callers only
    /// need the side effects (bytes, ticks, meta).
    fn remove_entry(&mut self, prefix: &[Token]) {
        if let Some(ckpt) = self.entries.remove(prefix) {
            self.total_bytes = self.total_bytes.saturating_sub(ckpt.data.len());
        }
        self.last_used.remove(prefix);
        self.meta.remove(prefix);
    }

    /// Eviction score; the greatest tuple is evicted first.
    ///
    /// Order: unreferenced before referenced (SGLang session-reference aware
    /// radix cache, P9 borrow 2), non-demand before demand-boundary checkpoints
    /// (P9 borrow 1), then longest, then oldest. Both protections are soft:
    /// when only protected entries remain, the longest of those still goes.
    fn evict_score(
        &self,
        key: &[Token],
        ckpt: &SeqCheckpoint,
    ) -> (u8, u8, u32, std::cmp::Reverse<u64>) {
        let meta = self.meta.get(key).copied().unwrap_or_default();
        (
            u8::from(meta.refs == 0),
            u8::from(!meta.demand),
            ckpt.n_tokens,
            std::cmp::Reverse(self.last_used.get(key).copied().unwrap_or(0)),
        )
    }

    /// Longest dump, then oldest tick at the same length.
    /// `skip_stubs` keeps 6k–16k tool heads (global host-RAM cap only).
    fn pick_longest_victim(&self, skip_stubs: bool) -> Option<Vec<Token>> {
        self.entries
            .iter()
            .filter(|(_, ckpt)| !skip_stubs || !Self::is_store_stub_len(ckpt.n_tokens))
            .max_by(|a, b| self.evict_score(a.0, a.1).cmp(&self.evict_score(b.0, b.1)))
            .map(|(k, _)| k.clone())
    }

    /// Drop the longest non-stub dump until `total_bytes + chain_bytes <= cap`.
    ///
    /// Stubs stay: the global cap may remain over once only tool heads are
    /// left. The store's own 8-entry / 2 GiB caps still drop stubs via
    /// `evict_if_over_cap`. The caller decides whether slot chains still
    /// need trimming.
    pub fn evict_global_over_cap(&mut self, chain_bytes: usize, cap: usize) {
        while cap > 0 && self.total_bytes + chain_bytes > cap {
            let victim = self.pick_longest_victim(true);
            let Some(victim) = victim else {
                break;
            };
            self.remove_entry(&victim);
        }
    }

    /// Number of resident entries (for tests / metrics).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Test-only: whether an entry was captured as a demand boundary.
    #[cfg(test)]
    pub(crate) fn entry_is_demand(&self, key: &[Token]) -> bool {
        self.meta.get(key).map(|m| m.demand).unwrap_or(false)
    }

    /// Test-only: live-session reference count for an entry.
    #[cfg(test)]
    pub(crate) fn entry_refs(&self, key: &[Token]) -> u32 {
        self.meta.get(key).map(|m| m.refs).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(v: &[i32]) -> Vec<Token> {
        v.iter().map(|&x| x as Token).collect()
    }

    #[test]
    fn common_prefix_reuses_shared_head() {
        let a = t(&[1, 2, 3, 4]);
        let b = t(&[1, 2, 5, 6]);
        assert_eq!(common_prefix_len(&a, &b), 2);
    }

    #[test]
    fn common_prefix_empty_when_no_shared() {
        assert_eq!(common_prefix_len(&t(&[1]), &t(&[9])), 0);
    }

    #[test]
    fn reuse_updates_cache_to_matched_prefix() {
        let mut c = SlotPrefixCache::new();
        c.tokens = t(&[1, 2, 3, 4]);
        assert_eq!(c.reuse(&t(&[1, 2, 5])), 2);
        assert_eq!(c.tokens, t(&[1, 2]));
    }

    #[test]
    fn reuse_zero_keeps_prior_cache() {
        let mut c = SlotPrefixCache::new();
        c.tokens = t(&[1, 2]);
        assert_eq!(c.reuse(&t(&[9, 8])), 0);
        // reset clears; a no-match does not overwrite the cache
        assert_eq!(c.tokens, t(&[1, 2]));
    }

    #[test]
    fn remember_truncates_to_resident_n_past() {
        let mut c = SlotPrefixCache::new();
        c.remember(&t(&[1, 2, 3, 4]), &t(&[5, 6]), 5);
        assert_eq!(c.tokens, t(&[1, 2, 3, 4, 5]));
        assert_eq!(c.prefix_len, 5);
    }

    #[test]
    fn second_bind_reuses_common_prefix() {
        let mut c = SlotPrefixCache::new();
        assert_eq!(c.reuse_for_bind(&t(&[1, 2, 3, 4]), 0), 0);
        c.remember(&t(&[1, 2, 3, 4]), &t(&[5, 6]), 6);
        let n = c.reuse_for_bind(&t(&[1, 2, 3, 4, 5, 6, 7, 8]), 6);
        assert!(n > 0, "shared prefix must reuse, got {n}");
        assert_eq!(n, 6);
    }

    #[test]
    fn bind_without_common_prefix_clears() {
        let mut c = SlotPrefixCache::new();
        c.remember(&t(&[1, 2, 3]), &[], 3);
        assert_eq!(c.reuse_for_bind(&t(&[9, 8, 7]), 3), 0);
        assert!(c.tokens.is_empty());
    }

    #[test]
    fn generated_tokens_extend_next_lcp() {
        let mut c = SlotPrefixCache::new();
        c.remember(&t(&[1, 2, 3]), &t(&[4, 5]), 5);
        // Next prompt includes the previous assistant ids [4,5].
        let n = c.reuse_for_bind(&t(&[1, 2, 3, 4, 5, 6]), 5);
        assert_eq!(n, 5, "LCP must include generated assistant tokens");
    }

    #[test]
    fn reuse_for_bind_leaves_one_token_for_logits() {
        let mut c = SlotPrefixCache::new();
        c.remember(&t(&[1, 2, 3]), &[], 3);
        // Exact prompt match would be LCP==len; clamp so we still prefill.
        assert_eq!(c.reuse_for_bind(&t(&[1, 2, 3]), 3), 2);
    }

    #[test]
    fn reuse_for_bind_clamps_lcp_to_hint() {
        let mut c = SlotPrefixCache::new();
        c.remember(&t(&[1, 2, 3, 4]), &[], 4);
        // Cover (GPU occupancy or ckpt_n) is 2, LCP is 4: reuse the
        // restorable prefix and keep session identity.
        assert_eq!(c.reuse_for_bind(&t(&[1, 2, 3, 4, 5]), 2), 2);
        assert_eq!(c.tokens, t(&[1, 2, 3, 4]));
        assert_eq!(c.prefix_len, 2);
    }

    #[test]
    fn reuse_for_bind_clamps_when_lcp_exceeds_ckpt_hint() {
        let mut c = SlotPrefixCache::new();
        let prompt: Vec<Token> = (0..70).map(|i| i as Token).collect();
        c.remember(&prompt[..60], &prompt[60..], 70);
        let mut next = prompt.clone();
        next.push(99);
        // Watermark: GPU empty, hint_n = ckpt_n = 60, next-turn LCP = 70.
        assert_eq!(c.reuse_for_bind(&next, 60), 60);
        assert_eq!(c.tokens.len(), 70, "session identity kept");
        assert_eq!(c.prefix_len, 60);
    }

    #[test]
    fn cancel_mid_prefill_reuses_decoded_prefix() {
        let mut c = SlotPrefixCache::new();
        let prompt = t(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        // Same request cancelled after 6 prompt tokens landed in KV.
        c.remember(&prompt, &[], 6);
        assert_eq!(c.reuse_for_bind(&prompt, 6), 6);
    }

    #[test]
    fn cancel_after_full_prefill_leaves_logits_token() {
        let mut c = SlotPrefixCache::new();
        let prompt = t(&[1, 2, 3, 4, 5]);
        c.remember(&prompt, &t(&[6, 7]), 7);
        // Retry of the same prompt: keep all but one cell for logits.
        assert_eq!(c.reuse_for_bind(&prompt, 7), 4);
    }

    fn media(id: &str, n_tokens: u32, n_pos: u32) -> VisionChunk {
        VisionChunk::Media {
            id: id.into(),
            n_tokens,
            n_pos,
        }
    }

    #[test]
    fn vision_lcp_matches_same_image_hash() {
        let a = VisionSeq::from_chunks([
            VisionChunk::Text(t(&[1, 2, 3])),
            media("img-a", 4, 2),
            VisionChunk::Text(t(&[9])),
        ]);
        let b = VisionSeq::from_chunks([
            VisionChunk::Text(t(&[1, 2, 3])),
            media("img-a", 4, 2),
            VisionChunk::Text(t(&[9, 8])),
        ]);
        assert_eq!(a.common_prefix(&b), 8);
        assert_eq!(a.pos_next(3), 3);
        assert_eq!(a.pos_next(7), 5);
        assert_eq!(a.n_pos(), 6);
    }

    #[test]
    fn vision_lcp_stops_at_different_image() {
        let a = VisionSeq::from_chunks([VisionChunk::Text(t(&[1, 2])), media("img-a", 3, 2)]);
        let b = VisionSeq::from_chunks([VisionChunk::Text(t(&[1, 2])), media("img-b", 3, 2)]);
        assert_eq!(a.common_prefix(&b), 2);
    }

    #[test]
    fn vision_lcp_does_not_split_image() {
        let a = VisionSeq::from_chunks([
            VisionChunk::Text(t(&[1])),
            media("img-a", 4, 2),
            VisionChunk::Text(t(&[7])),
        ]);
        let b = VisionSeq::from_chunks([VisionChunk::Text(t(&[1])), media("img-a", 4, 2)]);
        assert_eq!(a.common_prefix(&b), 5);
    }

    #[test]
    fn vision_reuse_leaves_logits_and_skips_mid_image() {
        let prev = VisionSeq::from_chunks([
            VisionChunk::Text(t(&[1, 2, 3])),
            media("img-a", 4, 2),
            VisionChunk::Text(t(&[9, 8])),
        ]);
        // Exact same prompt: clamp off last text token (not mid-image).
        let n = prev.reuse_for_bind(&prev, prev.n_pos());
        assert_eq!(n, 8);
        assert_eq!(prev.pos_next(n), prev.n_pos() - 1);

        // Prompt that ends on an image: snap reuse to image start.
        let img_only =
            VisionSeq::from_chunks([VisionChunk::Text(t(&[1, 2, 3])), media("img-a", 4, 2)]);
        let n = prev.reuse_for_bind(&img_only, prev.n_pos());
        assert_eq!(n, 3, "must not reuse a partial image for logits");
        assert_eq!(img_only.pos_next(n), 3);
    }

    #[test]
    fn vision_reuse_rejects_when_gpu_shorter_than_pos() {
        let prev = VisionSeq::from_chunks([VisionChunk::Text(t(&[1, 2, 3])), media("img-a", 4, 2)]);
        let n = prev.reuse_for_bind(&prev, 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn vision_append_and_truncate_keeps_full_images() {
        let mut seq = VisionSeq::from_chunks([
            VisionChunk::Text(t(&[1, 2])),
            media("img-a", 4, 2),
            VisionChunk::Text(t(&[9])),
        ]);
        // n_pos = 2 + 2 + 1 = 5
        seq.append_generated(&t(&[10, 11, 12]));
        assert_eq!(seq.n_pos(), 8);
        seq.truncate_to_pos(6);
        assert_eq!(seq.n_pos(), 6);
        assert_eq!(seq.tokens, t(&[1, 2, 0, 0, 0, 0, 9, 10]));

        seq.truncate_to_pos(3);
        // pos 3 is inside the image (pos 2..4); drop the image.
        assert_eq!(seq.tokens, t(&[1, 2]));
        assert!(seq.images.is_empty());
    }

    #[test]
    fn remember_vision_clears_text_tokens() {
        let mut c = SlotPrefixCache::new();
        c.remember(&t(&[1, 2, 3]), &[], 3);
        let seq = VisionSeq::from_chunks([VisionChunk::Text(t(&[1, 2, 3])), media("img-a", 2, 1)]);
        c.remember_vision(seq, &t(&[9]), 5);
        assert!(c.tokens.is_empty());
        let vs = c.vision.as_ref().unwrap();
        assert_eq!(vs.n_tokens(), 6);
        assert_eq!(vs.n_pos(), 5);
    }

    #[test]
    fn reset_clears_vision() {
        let mut c = SlotPrefixCache::new();
        c.vision = Some(VisionSeq::from_chunks([VisionChunk::Text(t(&[1]))]));
        c.reset();
        assert!(c.vision.is_none());
    }

    fn ckpt(n_tokens: u32, bytes: u8) -> crate::slot::SeqCheckpoint {
        crate::slot::SeqCheckpoint {
            n_tokens,
            data: vec![bytes; n_tokens as usize],
        }
    }

    #[test]
    fn prefix_store_find_best_hit() {
        let mut store = PrefixStore::new();
        store.put(t(&[1, 2, 3, 4]), ckpt(4, 0xAA));
        // Prompt shares the whole 4-token snapshot; reuse_len=4 fits n_tokens<=5.
        let hit = store.find_best(&t(&[1, 2, 3, 4, 5, 6]), 4);
        assert!(hit.is_some());
        let hit = hit.unwrap();
        assert_eq!(hit.n_tokens, 4);
        assert_eq!(hit.data, vec![0xAA; 4]);
    }

    #[test]
    fn prefix_store_find_best_returns_longest() {
        let mut store = PrefixStore::new();
        store.put(t(&[1, 2]), ckpt(2, 0x11));
        store.put(t(&[1, 2, 3, 4]), ckpt(4, 0x22));
        let hit = store.find_best(&t(&[1, 2, 3, 4, 5, 6]), 5);
        assert!(hit.is_some());
        assert_eq!(
            hit.unwrap().n_tokens,
            4,
            "must return the longest usable snapshot"
        );
    }

    #[test]
    fn prefix_store_find_best_miss() {
        let mut store = PrefixStore::new();
        store.put(t(&[9, 8, 7]), ckpt(3, 0x33));
        // No shared prefix -> None.
        assert!(store.find_best(&t(&[1, 2, 3]), 3).is_none());
    }

    #[test]
    fn prefix_store_find_best_rejects_too_long() {
        let mut store = PrefixStore::new();
        store.put(t(&[1, 2, 3, 4, 5, 6, 7, 8]), ckpt(8, 0x44));
        // n_tokens=8 > reuse_len+1=4 -> unusable even on exact prefix match.
        assert!(store.find_best(&t(&[1, 2, 3, 4, 5, 6, 7, 8]), 3).is_none());
    }

    #[test]
    fn prefix_store_key_collision_overwrites() {
        let mut store = PrefixStore::new();
        store.put(t(&[1, 2, 3]), ckpt(3, 0x55));
        store.put(t(&[1, 2, 3]), ckpt(3, 0x66));
        assert_eq!(store.len(), 1);
        let hit = store.find_best(&t(&[1, 2, 3, 4]), 3).unwrap();
        assert_eq!(hit.data, vec![0x66; 3], "same key must be replaced");
        assert_eq!(store.total_bytes, 3);
    }

    #[test]
    fn prefix_store_evicts_longest_not_short_head() {
        let mut store = PrefixStore::with_cap(10); // tiny cap
        store.put(t(&[1, 2]), ckpt(2, 0x11)); // 2 bytes
        store.put(t(&[1, 2, 3, 4]), ckpt(4, 0x22)); // 4 bytes -> total 6
        store.put(t(&[1, 2, 3, 4, 5, 6]), ckpt(6, 0x33)); // 6 bytes -> total 12 > 10
        assert!(store.total_bytes <= 10, "must evict down to cap");
        // Long dump is the victim; the short prefix must still hit.
        assert!(
            store.find_best(&t(&[1, 2, 9]), 2).is_some(),
            "short prefix must not be pushed out by a longer dump"
        );
        assert!(
            store.entries.values().all(|c| c.n_tokens < 6),
            "the 6-token dump must be the victim"
        );
    }

    #[test]
    fn prefix_store_entry_cap_is_single_digit() {
        let mut store = PrefixStore::with_limits(0, 3);
        for n in 1..=5u32 {
            let prefix: Vec<Token> = (0..n).map(|i| i as Token).collect();
            store.put(prefix, ckpt(n, n as u8));
        }
        assert!(store.len() <= 3);
        // Longest dumps dropped; shortest remain.
        assert!(store.entries.values().all(|c| c.n_tokens <= 3));
        assert!(store.find_best(&t(&[0, 1, 9]), 2).is_some());
    }

    #[test]
    fn snapshot_key_len_eq_n_tokens() {
        let prompt: Vec<Token> = (0..20_000).map(|i| i as Token).collect();
        let key = snapshot_key(&prompt, 12_288).unwrap();
        assert_eq!(key.len(), 12_288);
        assert_eq!(key, prompt[..12_288]);
        assert!(snapshot_key(&prompt, 0).is_none());
        assert!(snapshot_key(&prompt, 20_001).is_none());
        assert_eq!(snapshot_key(&prompt, 20_000).unwrap().len(), 20_000);
    }

    #[test]
    fn empty_slot_store_hit_reuses_head() {
        let mut store = PrefixStore::new();
        let head: Vec<Token> = (0..12_288).map(|i| (i % 7) as Token).collect();
        store.put(head.clone(), ckpt(12_288, 0xAB));
        let mut prompt = head.clone();
        prompt.extend((0..1_000).map(|i| (100 + i % 3) as Token));
        // Empty slot: local reuse_len is 0; lookup still uses the new prompt.
        let hit = store
            .find_best_for_bind(&prompt, 0)
            .expect("store must hit");
        assert_eq!(hit.n_tokens, 12_288, "reused == tool-head ubatch boundary");
        assert_eq!(hit.data, vec![0xAB; 12_288]);
    }

    /// Shared tool/cwd head then per-project tokens (task #8910: LCP 6436).
    fn split_at_lcp(lcp: usize, n: usize) -> (Vec<Token>, Vec<Token>) {
        let shared: Vec<Token> = (0..lcp).map(|i| (i % 7) as Token).collect();
        let mut a = shared.clone();
        a.extend((lcp..n).map(|i| 100 + (i % 3) as Token));
        let mut b = shared;
        b.extend((lcp..n).map(|i| 200 + (i % 5) as Token));
        (a, b)
    }

    #[test]
    fn prefix_store_bind_hits_6144_when_8192_diverges_at_6436() {
        let lcp = 6_436;
        let (sess_a, sess_b) = split_at_lcp(lcp, 9_000);
        assert_eq!(common_prefix_len(&sess_a, &sess_b), lcp);

        let mut store = PrefixStore::new();
        store.put(sess_a[..8_192].to_vec(), ckpt(8_192, 0x81));
        assert!(
            store.find_best_for_bind(&sess_b, 0).is_none(),
            "8192 dump includes per-project tokens past LCP 6436"
        );

        store.put(sess_a[..6_144].to_vec(), ckpt(6_144, 0x61));
        let hit = store
            .find_best_for_bind(&sess_b, 0)
            .expect("6144 dump is below LCP 6436");
        assert_eq!(hit.n_tokens, 6_144);
        assert_eq!(hit.data, vec![0x61; 6_144]);
    }

    #[test]
    fn prefix_store_miss_diag_silent_when_empty() {
        let store = PrefixStore::new();
        assert!(
            store.miss_diag(&t(&[1, 2, 3])).is_none(),
            "cold-start miss must not journal"
        );
    }

    #[test]
    fn prefix_store_miss_diag_reports_lcp_and_store_ns() {
        let lcp = 6_436;
        let (sess_a, sess_b) = split_at_lcp(lcp, 13_000);
        let mut store = PrefixStore::new();
        store.put(sess_a[..8_192].to_vec(), ckpt(8_192, 0x81));
        store.put(sess_a[..12_288].to_vec(), ckpt(12_288, 0xC0));
        let diag = store.miss_diag(&sess_b).expect("store not empty");
        assert_eq!(diag.max_lcp, lcp, "longest LCP is the cwd split, not 8192");
        assert_eq!(
            diag.store_ns,
            vec![12_288, 8_192],
            "lengths only; longest first"
        );
    }

    #[test]
    fn prefix_store_entry_cap_two_sessions_keeps_6144_and_12288() {
        // MIN=6144 → 6 stubs/session. Shared 6144 is one key; later dumps
        // diverge. Cap 8 drops the longest (16384, then 14336). 12288 and
        // 6144 stay, so a same-project empty slot can still restore 12k.
        let mut store = PrefixStore::new();
        let window: [u32; 6] = [6_144, 8_192, 10_240, 12_288, 14_336, 16_384];
        for sess in 0..2u32 {
            for &n in &window {
                let prefix: Vec<Token> = (0..n)
                    .map(|i| {
                        if i < 6_144 {
                            i as Token
                        } else {
                            (sess * 1_000 + i) as Token
                        }
                    })
                    .collect();
                store.put(prefix, ckpt(n, sess as u8));
            }
        }
        assert_eq!(store.len(), PrefixStore::DEFAULT_MAX_ENTRIES);
        let ns: Vec<u32> = {
            let mut v: Vec<u32> = store.entries.values().map(|c| c.n_tokens).collect();
            v.sort_unstable();
            v
        };
        assert!(ns.contains(&6_144), "6144 shared head must survive");
        assert!(
            ns.contains(&12_288),
            "12288 must survive the MIN=6144 window"
        );
        assert!(!ns.contains(&16_384), "longest dumps evicted first");
    }

    #[test]
    fn prefix_store_skips_empty_or_mismatched_key() {
        let mut store = PrefixStore::new();
        store.put(Vec::new(), ckpt(4, 0x11));
        store.put(t(&[1, 2]), ckpt(4, 0x22));
        assert!(store.is_empty());
    }

    #[test]
    fn prefix_store_entry_cap_evicts_stubs() {
        // Production put() only inserts 6k–16k stubs. The store's own
        // max_entries must still shrink once only stubs remain.
        let mut store = PrefixStore::new();
        let mut keys = Vec::new();
        for i in 0..10u32 {
            let prefix: Vec<Token> = std::iter::repeat(i as Token).take(12_288).collect();
            keys.push(prefix.clone());
            store.put(prefix, ckpt(12_288, i as u8));
        }
        assert_eq!(
            store.len(),
            PrefixStore::DEFAULT_MAX_ENTRIES,
            "stub-only put must still honour max_entries"
        );
        assert!(
            !store.entries.contains_key(&keys[0]) && !store.entries.contains_key(&keys[1]),
            "oldest stubs must be the victims"
        );
        assert!(store.entries.contains_key(&keys[9]), "newest stub kept");
    }

    #[test]
    fn prefix_store_global_cap_keeps_stubs() {
        let mut store = PrefixStore::new();
        for i in 0..3u32 {
            let prefix: Vec<Token> = std::iter::repeat(i as Token).take(12_288).collect();
            store.put(prefix, ckpt(12_288, i as u8));
        }
        assert_eq!(store.len(), 3);
        store.evict_global_over_cap(0, 1);
        assert_eq!(
            store.len(),
            3,
            "global host-RAM cap must not drop store stubs"
        );
    }

    // ── HAL-4 #248: disk tier ──────────────────────────────────────────────

    fn temp_dir(tag: &str) -> PathBuf {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let d = std::env::temp_dir().join(format!(
            "golbang-prefix-disk-{}-{tag}-{n}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn fp(tag: &str) -> SnapshotFingerprint {
        SnapshotFingerprint::new("golbang-test", "/models/test.gguf", 1, 2, 4096, tag)
    }

    #[test]
    fn disk_tier_survives_reopen() {
        let dir = temp_dir("reopen");
        let head = t(&[1, 2, 3, 4, 5, 6]);
        {
            let mut store = PrefixStore::with_disk(&dir, 0, fp("a")).unwrap();
            store.persist(&head, &ckpt(6, 0xAB));
            assert_eq!(store.disk_entries(), 1);
        }
        // Simulate a process restart: a fresh store on the same directory.
        let mut store = PrefixStore::with_disk(&dir, 0, fp("a")).unwrap();
        assert_eq!(store.disk_entries(), 1, "index must reload on startup");
        assert!(store.entries.is_empty(), "data stays on disk, not in RAM");
        let hit = store
            .find_best(&t(&[1, 2, 3, 4, 5, 6, 7, 8]), 7)
            .expect("disk hit after restart");
        assert_eq!(hit.n_tokens, 6);
        assert_eq!(hit.data, vec![0xAB; 6]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn disk_tier_restore_matches_ram_restore() {
        let dir = temp_dir("parity");
        let head: Vec<Token> = (0..512).map(|i| (i % 9) as Token).collect();
        let prompt: Vec<Token> = head.iter().copied().chain([7, 8, 9]).collect();

        let mut ram = PrefixStore::new();
        ram.put(head.clone(), ckpt(512, 0x5A));
        let ram_hit = ram.find_best(&prompt, prompt.len() - 1).unwrap();

        let mut disk = PrefixStore::with_disk(&dir, 0, fp("a")).unwrap();
        disk.persist(&head, &ckpt(512, 0x5A));
        let disk_hit = disk.find_best(&prompt, prompt.len() - 1).unwrap();

        assert_eq!(disk_hit.n_tokens, ram_hit.n_tokens, "n_tokens must match");
        assert_eq!(disk_hit.data, ram_hit.data, "bytes must be byte-identical");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn disk_tier_fingerprint_mismatch_misses() {
        let dir = temp_dir("fp");
        let head = t(&[1, 2, 3, 4, 5, 6]);
        {
            let mut store = PrefixStore::with_disk(&dir, 0, fp("build-a")).unwrap();
            store.persist(&head, &ckpt(6, 0xAB));
        }
        // Different build/model/config: the header is ignored on scan.
        let mut other = PrefixStore::with_disk(&dir, 0, fp("build-b")).unwrap();
        assert_eq!(other.disk_entries(), 0, "foreign dump must not load");
        assert!(
            other.find_best(&t(&[1, 2, 3, 4, 5, 6, 7]), 6).is_none(),
            "different fingerprint must miss"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn disk_tier_n_ctx_changes_fingerprint() {
        let a = SnapshotFingerprint::new("g", "/m.gguf", 1, 2, 4096, "c");
        let b = SnapshotFingerprint::new("g", "/m.gguf", 1, 2, 8192, "c");
        assert_ne!(a.hash(), b.hash(), "n_ctx must change the key");
        let c = SnapshotFingerprint::new("g", "/other.gguf", 1, 2, 4096, "c");
        assert_ne!(a.hash(), c.hash(), "model path must change the key");
        let d = SnapshotFingerprint::new("g", "/m.gguf", 1, 2, 4096, "different");
        assert_ne!(a.hash(), d.hash(), "config must change the key");
    }

    #[test]
    fn disk_tier_lru_evicts_oldest() {
        let dir = temp_dir("lru");
        // Each entry is 100 data bytes; cap 250 forces one eviction of 3.
        let mut disk =
            PrefixDisk::open(&dir, 250, fp("a")).expect("open");
        for i in 0..3u32 {
            let key: Vec<Token> = std::iter::repeat(i as Token).take(100).collect();
            disk.put(&key, &ckpt(100, i as u8)).unwrap();
        }
        assert_eq!(disk.len(), 2, "cap 250 with 100-byte entries keeps two");
        assert!(disk.used_bytes() <= 250);
        // Oldest (key 0) was evicted; newest (key 2) still reads.
        assert!(
            disk.find(&vec![0 as Token; 110], 109).is_none(),
            "oldest entry evicted"
        );
        assert!(
            disk.find(&vec![2 as Token; 110], 109).is_some(),
            "newest entry kept"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn disk_tier_put_overwrites_same_key() {
        let dir = temp_dir("overwrite");
        let mut store = PrefixStore::with_disk(&dir, 0, fp("a")).unwrap();
        store.persist(&t(&[1, 2, 3]), &ckpt(3, 0x11));
        store.persist(&t(&[1, 2, 3]), &ckpt(3, 0x22));
        assert_eq!(store.disk_entries(), 1);
        let hit = store.find_best(&t(&[1, 2, 3, 4]), 3).unwrap();
        assert_eq!(hit.data, vec![0x22; 3], "newer dump wins");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn disk_tier_off_is_ram_only() {
        // Regression: no `--prefix-cache-dir` → no disk tier, RAM behavior.
        let mut store = PrefixStore::new();
        assert!(store.disk.is_none());
        assert_eq!(store.disk_entries(), 0);
        store.put(t(&[1, 2, 3]), ckpt(3, 0x33));
        let hit = store.find_best(&t(&[1, 2, 3, 4]), 3).unwrap();
        assert_eq!(hit.data, vec![0x33; 3]);
        assert_eq!(store.disk_bytes(), 0);
    }

    #[test]
    fn disk_tier_persist_skips_ram_map() {
        // `persist` writes a chain anchor to disk but not into the 2 GiB RAM
        // map, so a long 100k anchor cannot evict the shared tool head.
        let dir = temp_dir("persist");
        let long: Vec<Token> = (0..100_000).map(|i| (i % 11) as Token).collect();
        let mut store = PrefixStore::with_disk(&dir, 0, fp("a")).unwrap();
        store.persist(&long, &ckpt(100_000, 0x77));
        assert!(store.entries.is_empty(), "persist must not touch RAM");
        assert_eq!(store.disk_entries(), 1);
        let hit = store.find_best(&long, 100_000).unwrap();
        assert_eq!(hit.n_tokens, 100_000);
        assert_eq!(hit.data.len(), 100_000);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn disk_tier_scan_ignores_corrupt_files() {
        let dir = temp_dir("corrupt");
        let key = t(&[1, 2, 3, 4]);
        {
            let mut store = PrefixStore::with_disk(&dir, 0, fp("a")).unwrap();
            store.persist(&key, &ckpt(4, 0xAB));
        }
        fs::write(dir.join("garbage.ckpt"), b"not a snapshot").unwrap();
        let store = PrefixStore::with_disk(&dir, 0, fp("a")).unwrap();
        assert_eq!(store.disk_entries(), 1, "only the valid file loads");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn demand_boundary_arms_on_second_observation() {
        let mut store = PrefixStore::new();
        let (sess_a, sess_b) = split_at_lcp(3_000, 4_000);
        let mut sess_c = sess_a[..3_000].to_vec();
        sess_c.extend((0..500).map(|i| 300 + (i % 5) as Token));
        assert_eq!(common_prefix_len(&sess_a, &sess_c), 3_000);

        // First session seeds the ring; the second only records the boundary.
        store.record_prompt(&sess_a);
        assert_eq!(store.observe_boundary(&sess_b, 0), None);
        store.record_prompt(&sess_b);
        // Third session: second observation of the 3000 boundary → armed.
        assert_eq!(store.observe_boundary(&sess_c, 0), Some(3_000));
    }

    #[test]
    fn demand_boundary_ignores_short_and_reused() {
        let mut store = PrefixStore::new();
        let (sess_a, sess_b) = split_at_lcp(100, 200);
        store.record_prompt(&sess_a);
        assert_eq!(
            store.observe_boundary(&sess_b, 0),
            None,
            "100 < DEMAND_MIN_BOUNDARY"
        );

        let (long_a, long_b) = split_at_lcp(3_000, 4_000);
        store.record_prompt(&long_a);
        assert_eq!(
            store.observe_boundary(&long_b, 3_000),
            None,
            "boundary already covered by local reuse"
        );
    }

    #[test]
    fn demand_boundary_skips_when_already_resident() {
        let mut store = PrefixStore::new();
        let (sess_a, sess_b) = split_at_lcp(3_000, 4_000);
        store.record_prompt(&sess_a);
        let _ = store.observe_boundary(&sess_b, 0);
        store.record_prompt(&sess_b);
        let mut sess_c = sess_a[..3_000].to_vec();
        sess_c.extend((0..500).map(|i| 300 + (i % 5) as Token));

        // Capture the boundary; a later observation must not re-arm a stop.
        store.put(sess_c[..3_000].to_vec(), ckpt(3_000, 0xAB));
        assert_eq!(store.observe_boundary(&sess_c, 0), None);
    }

    #[test]
    fn eviction_prefers_unreferenced_nondemand() {
        let mut store = PrefixStore::with_limits(0, 2);
        let demand: Vec<Token> = (0..1_000).map(|i| (i % 7) as Token).collect();
        let plain: Vec<Token> = (0..500).map(|i| (500 + i % 7) as Token).collect();
        store.put_with(demand.clone(), ckpt(1_000, 0xAA), true);
        store.put(plain.clone(), ckpt(500, 0xBB));
        assert!(store.entry_is_demand(&demand));

        // New plain entry triggers eviction. The non-demand plain entry is the
        // victim; the demand boundary survives.
        let fresh: Vec<Token> = (0..400).map(|i| (900 + i % 7) as Token).collect();
        store.put(fresh.clone(), ckpt(400, 0xCC));
        assert!(store.entries.contains_key(&demand), "demand boundary kept");
        assert!(store.entries.contains_key(&fresh), "new entry kept");
        assert!(!store.entries.contains_key(&plain), "plain victim evicted");
    }

    #[test]
    fn refresh_refs_protects_live_session_head() {
        let mut store = PrefixStore::with_limits(0, 2);
        let head: Vec<Token> = (0..600).map(|i| (i % 7) as Token).collect();
        let other: Vec<Token> = (0..500).map(|i| (700 + i % 7) as Token).collect();
        store.put(head.clone(), ckpt(600, 0xAA));
        store.put(other.clone(), ckpt(500, 0xBB));

        let mut live = head.clone();
        live.extend((0..50).map(|i| 100 + i as Token));
        store.refresh_entry_refs(&[&live[..]]);
        assert_eq!(store.entry_refs(&head), 1);
        assert_eq!(store.entry_refs(&other), 0);

        // New plain entry: the unreferenced `other` goes, not the referenced head.
        let fresh: Vec<Token> = (0..400).map(|i| (900 + i % 7) as Token).collect();
        store.put(fresh, ckpt(400, 0xCC));
        assert!(store.entries.contains_key(&head), "referenced head kept");
        assert!(!store.entries.contains_key(&other), "unreferenced victim");
    }
}
