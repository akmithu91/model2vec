use anyhow::{anyhow, Result};
use std::path::Path;
use tokenizers::Tokenizer;

// ---------------------------------------------------------------------------
// SnapCfg
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct SnapCfg {
    pub max_forward_bytes:  usize,
    pub max_backward_bytes: usize,
    pub prefer_paragraph:   bool,
    pub trim_whitespace:    bool,
}

impl Default for SnapCfg {
    fn default() -> Self {
        Self {
            max_forward_bytes:  256,
            max_backward_bytes: 128,
            prefer_paragraph:   true,
            trim_whitespace:    true,
        }
    }
}

// ---------------------------------------------------------------------------
// Boundary snapping — operates on byte offsets throughout
// ---------------------------------------------------------------------------

fn clamp_char_boundary(text: &str, mut i: usize, dir: i32) -> usize {
    let len = text.len();
    i = i.min(len);
    while i > 0 && i < len && !text.is_char_boundary(i) {
        if dir >= 0 { i += 1; } else { i -= 1; }
        if i >= len { return len; }
    }
    i
}

fn snap_span(text: &str, start: usize, end: usize, cfg: SnapCfg) -> (usize, usize) {
    let len    = text.len();
    let mut s  = clamp_char_boundary(text, start.min(len), -1);
    let mut e  = clamp_char_boundary(text, end.min(len),   -1);
    let orig_e = e;

    let back_start = clamp_char_boundary(text, e.saturating_sub(cfg.max_backward_bytes), -1);
    let fwd_end    = clamp_char_boundary(text, (e + cfg.max_forward_bytes).min(len),     -1);

    // 1. Paragraph break forward
    if cfg.prefer_paragraph {
        if let Some(pos) = text[e..fwd_end].find("\n\n") {
            e = clamp_char_boundary(text, e + pos + 2, -1);
        }
    }

    // 2. Sentence boundary forward
    if e == orig_e {
        for (idx, ch) in text[e..fwd_end].char_indices() {
            if matches!(ch, '.' | '!' | '?' | '…' | '\n') {
                e = clamp_char_boundary(text, e + idx + ch.len_utf8(), -1);
                break;
            }
        }
    }

    // 3. Sentence boundary backward
    if e == orig_e {
        let mut best = None;
        for (idx, ch) in text[back_start..e].char_indices() {
            if matches!(ch, '.' | '!' | '?' | '…' | '\n') {
                best = Some(back_start + idx + ch.len_utf8());
            }
        }
        if let Some(new_e) = best {
            e = clamp_char_boundary(text, new_e, -1);
        }
    }

    // 4. Whitespace trimming
    if cfg.trim_whitespace {
        while e > s {
            let prev = clamp_char_boundary(text, e - 1, -1);
            if text[prev..e].chars().next().unwrap().is_whitespace() { e = prev; } else { break; }
        }
        while s < e {
            let c = text[s..].chars().next().unwrap();
            if c.is_whitespace() { s += c.len_utf8(); } else { break; }
        }
    }

    (s, e)
}

// ---------------------------------------------------------------------------
// Core chunking
//
// Returns a flat Vec<i32> of [start0, end0, start1, end1, ...]
// where start/end are byte offsets into the original text.
//
// Java side reconstructs each chunk as:
//   byte[] bytes = text.getBytes(UTF_8);
//   new String(bytes, start, end - start, UTF_8)
// ---------------------------------------------------------------------------

fn chunk_to_offsets(
    tokenizer:  &Tokenizer,
    text:       &str,
    chunk_size: usize,
    stride:     usize,
    snap_cfg:   SnapCfg,
) -> Result<Vec<i32>> {
    if chunk_size == 0 || stride == 0 {
        return Err(anyhow!("chunk_size and stride must be > 0"));
    }

    let enc     = tokenizer.encode(text, false)
        .map_err(|e| anyhow!("tokenizer.encode: {e}"))?;
    let offsets = enc.get_offsets(); // &[(usize, usize)] byte offsets
    let n       = offsets.len();

    if n == 0 { return Ok(vec![]); }

    // Upper bound: number of windows
    let n_windows = (n.saturating_sub(chunk_size)) / stride + 1;
    let mut out   = Vec::with_capacity(n_windows * 2);

    let mut start_tok = 0;
    while start_tok < n {
        let end_tok = (start_tok + chunk_size).min(n);
        let (byte_start, _) = offsets[start_tok];
        let (_, byte_end)   = offsets[end_tok - 1];

        let (s, e) = snap_span(text, byte_start, byte_end, snap_cfg);

        // Only emit non-empty spans
        if e > s {
            out.push(s as i32);
            out.push(e as i32);
        }

        if end_tok == n { break; }
        start_tok += stride;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Public struct — Box<Chunker> stored as jlong on the Java side
// ---------------------------------------------------------------------------

pub struct Chunker {
    tokenizer:  Tokenizer,
    chunk_size: usize,
    stride:     usize,
    snap_cfg:   SnapCfg,
}

impl Chunker {
    /// `repo_or_path` — HuggingFace repo ID  or  local directory containing tokenizer.json.
    pub fn new(repo_or_path: &str, chunk_size: usize, stride: usize, snap_cfg: SnapCfg) -> Result<Self> {
        let local = Path::new(repo_or_path);
        let tokenizer = if local.is_dir() {
            // Local directory: point directly at tokenizer.json
            Tokenizer::from_file(local.join("tokenizer.json"))
                .map_err(|e| anyhow!("from_file: {e}"))?
        } else if local.is_file() {
            // Caller passed the .json file itself
            Tokenizer::from_file(local)
                .map_err(|e| anyhow!("from_file: {e}"))?
        } else {
            // Treat as a HuggingFace repo ID — requires `http` feature
            Tokenizer::from_pretrained(repo_or_path, None)
                .map_err(|e| anyhow!("from_pretrained '{repo_or_path}': {e}"))?
        };

        Ok(Self { tokenizer, chunk_size, stride, snap_cfg })
    }

    /// Returns flat byte-offset pairs [start0, end0, start1, end1, ...].
    pub fn chunk(&self, text: &str) -> Result<Vec<i32>> {
        chunk_to_offsets(&self.tokenizer, text, self.chunk_size, self.stride, self.snap_cfg)
    }
}
