//! KenLM-backed CTC prefix beam search for MedASR.
//!
//! Google ships a 6-gram KenLM over MedASR's SentencePiece tokens
//! (`lm_6.kenlm`); the reference notebook decodes with pyctcdecode at
//! `beam_width=8`, which is what turns the model-card 6.6% WER (greedy) into
//! 4.6%. Rather than binding the KenLM C++ library, the ARPA is converted
//! offline (`export_lm.py`) into a flat binary the decoder can mmap and binary
//! search directly: per order, a sorted array of n-gram keys (token ids packed
//! 10 bits each) plus log10 probability / backoff arrays.
//!
//! Scoring reproduces pyctcdecode's `LanguageModel` with its defaults: each
//! token is one LM "word", `alpha=0.5`, `beta=1.5`, OOV offset `-10` (log10),
//! `<s>` begin-sentence context and `</s>` scored at the end of the utterance.
//! The one deliberate difference: pyctcdecode defers a word's LM score until
//! the next word starts; we score each token the moment it is emitted. The
//! total score of a finished hypothesis is identical — only mid-search beam
//! ranking can differ, by at most one word of lookahead.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

/// pyctcdecode defaults (DEFAULT_ALPHA / DEFAULT_BETA / DEFAULT_UNK_LOGP_OFFSET
/// / DEFAULT_MIN_TOKEN_LOGP). Alpha/beta weight the LM against the acoustic
/// model; the unk offset (log10) punishes tokens the LM has never seen; the
/// token floor (natural log) drops per-frame candidates below e^-5.
const ALPHA: f32 = 0.5;
const BETA: f32 = 1.5;
const UNK_LOGP_OFFSET: f32 = -10.0;
const MIN_TOKEN_LOGP: f32 = -5.0;
/// KenLM works in log10, the beam in natural log.
const LN_10: f32 = std::f32::consts::LN_10;

pub const BEAM_WIDTH: usize = 8;

/// Bits per token id in a packed n-gram key (ids go up to 514 < 1024).
const ID_BITS: u32 = 10;

struct Table {
    /// Byte offsets into the mmap.
    keys: usize,
    probs: usize,
    /// `None` for the highest order (no backoff weights).
    backs: Option<usize>,
    count: usize,
}

/// Memory-mapped n-gram LM produced by `export_lm.py`. All lookups are binary
/// searches over per-order sorted key arrays; nothing is deserialized up front,
/// so loading is instant and resident memory is whatever the OS pages in.
pub struct LmModel {
    map: Mmap,
    order: usize,
    bos: u32,
    eos: u32,
    unk: u32,
    tables: Vec<Table>,
}

impl LmModel {
    pub fn load(path: &Path) -> Result<LmModel, String> {
        let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let map = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?;
        if map.len() < 72 || &map[0..4] != b"VPL1" {
            return Err("not a VPL1 language model file".into());
        }
        let u32_at = |o: usize| u32::from_le_bytes(map[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(map[o..o + 8].try_into().unwrap());
        let order = u32_at(4) as usize;
        let (bos, eos, unk) = (u32_at(8), u32_at(12), u32_at(16));
        if order == 0 || order > 6 {
            return Err(format!("unsupported LM order {order}"));
        }

        let mut tables = Vec::with_capacity(order);
        let mut off = 24 + 8 * order;
        for o in 1..=order {
            let count = u64_at(24 + 8 * (o - 1)) as usize;
            let keys = off;
            let probs = keys + 8 * count;
            let has_backs = o < order;
            let backs = has_backs.then_some(probs + 4 * count);
            off = probs + 4 * count + if has_backs { 4 * count } else { 0 };
            off = (off + 7) & !7; // 8-byte alignment padding between orders
            tables.push(Table {
                keys,
                probs,
                backs,
                count,
            });
        }
        if off > map.len() {
            return Err("language model file is truncated".into());
        }
        Ok(LmModel {
            map,
            order,
            bos,
            eos,
            unk,
            tables,
        })
    }

    pub fn bos(&self) -> u32 {
        self.bos
    }

    pub fn context_len(&self) -> usize {
        self.order - 1
    }

    fn key_at(&self, t: &Table, i: usize) -> u64 {
        let o = t.keys + 8 * i;
        u64::from_le_bytes(self.map[o..o + 8].try_into().unwrap())
    }

    fn f32_at(&self, off: usize, i: usize) -> f32 {
        let o = off + 4 * i;
        f32::from_le_bytes(self.map[o..o + 4].try_into().unwrap())
    }

    /// (log10 prob, log10 backoff) of the `n`-gram with packed key `key`.
    fn lookup(&self, n: usize, key: u64) -> Option<(f32, f32)> {
        let t = &self.tables[n - 1];
        let (mut lo, mut hi) = (0usize, t.count);
        while lo < hi {
            let mid = (lo + hi) / 2;
            match self.key_at(t, mid).cmp(&key) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => {
                    let back = t.backs.map(|b| self.f32_at(b, mid)).unwrap_or(0.0);
                    return Some((self.f32_at(t.probs, mid), back));
                }
            }
        }
        None
    }

    fn is_oov(&self, w: u32) -> bool {
        self.lookup(1, w as u64).is_none()
    }

    /// log10 P(w | ctx) with Katz backoff, KenLM `BaseScore` semantics.
    /// `ctx` is most-recent-last, at most `order - 1` long.
    fn score(&self, ctx: &[u32], w: u32) -> f32 {
        let mut backoff = 0.0f32;
        for n in (2..=(ctx.len() + 1).min(self.order)).rev() {
            let c = &ctx[ctx.len() - (n - 1)..];
            if let Some((p, _)) = self.lookup(n, pack_with(c, w)) {
                return p + backoff;
            }
            if let Some((_, b)) = self.lookup(n - 1, pack(c)) {
                backoff += b;
            }
        }
        match self.lookup(1, w as u64) {
            Some((p, _)) => p + backoff,
            // OOV: score as <unk>, like KenLM does.
            None => {
                self.lookup(1, self.unk as u64)
                    .map(|(p, _)| p)
                    .unwrap_or(-10.0)
                    + backoff
            }
        }
    }

    /// The scaled LM contribution for emitting token `w` after `ctx`:
    /// pyctcdecode's `alpha * log10 * ln10 + beta`, plus the OOV offset.
    fn word_score(&self, ctx: &[u32], w: u32) -> f32 {
        let mut s = self.score(ctx, w);
        if self.is_oov(w) {
            s += UNK_LOGP_OFFSET;
        }
        ALPHA * s * LN_10 + BETA
    }

    /// End-of-utterance `</s>` score (no beta — it is not a word).
    fn end_score(&self, ctx: &[u32]) -> f32 {
        ALPHA * self.score(ctx, self.eos) * LN_10
    }
}

/// Pack token ids into an n-gram key, oldest word in the highest bits.
fn pack(gram: &[u32]) -> u64 {
    gram.iter().fold(0u64, |k, &w| (k << ID_BITS) | w as u64)
}

fn pack_with(ctx: &[u32], w: u32) -> u64 {
    (pack(ctx) << ID_BITS) | w as u64
}

#[derive(Clone)]
struct Beam {
    /// Collapsed CTC output so far (token ids, no blanks).
    prefix: Vec<u32>,
    /// ln probability of `prefix` over paths ending in blank / non-blank.
    p_b: f32,
    p_nb: f32,
    /// Cumulative scaled LM score of `prefix`.
    lm: f32,
    /// LM context: `<s>` followed by the last `order - 1` tokens.
    ctx: Vec<u32>,
}

impl Beam {
    fn score(&self) -> f32 {
        log_add(self.p_b, self.p_nb) + self.lm
    }
}

fn log_add(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

/// CTC prefix beam search over `[frames, vocab]` logits with n-gram LM fusion.
/// Returns the best hypothesis' token ids (collapsed, blank-free) — the same
/// shape of output as `greedy_ctc`, ready for the tokenizer.
pub fn beam_search(
    logits: &[f32],
    frames: usize,
    vocab: usize,
    blank: usize,
    lm: &LmModel,
) -> Vec<u32> {
    let ctx_len = lm.context_len();
    let mut beams = vec![Beam {
        prefix: Vec::new(),
        p_b: 0.0,
        p_nb: f32::NEG_INFINITY,
        lm: 0.0,
        ctx: vec![lm.bos()],
    }];

    let mut logp = vec![0.0f32; vocab];
    for f in 0..frames {
        let row = &logits[f * vocab..(f + 1) * vocab];
        log_softmax(row, &mut logp);

        // Candidate tokens for this frame: everything above the pyctcdecode
        // floor. The argmax always qualifies (a softmax max is ≥ 1/512 > e^-5).
        let mut next: Vec<Beam> = Vec::with_capacity(beams.len() * 4);
        // prefix -> index in `next`, so extensions from different parents merge.
        let mut index: std::collections::HashMap<Vec<u32>, usize> =
            std::collections::HashMap::with_capacity(beams.len() * 4);

        for beam in &beams {
            let p_tot = log_add(beam.p_b, beam.p_nb);

            // Stay on this prefix: blank, or a repeat of its last token.
            let p_b = p_tot + logp[blank];
            let p_nb = match beam.prefix.last() {
                Some(&last) => beam.p_nb + logp[last as usize],
                None => f32::NEG_INFINITY,
            };
            merge(
                &mut next,
                &mut index,
                &beam.prefix,
                p_b,
                p_nb,
                beam.lm,
                &beam.ctx,
            );

            // Extend with a new token.
            for c in 0..vocab {
                if c == blank || logp[c] < MIN_TOKEN_LOGP {
                    continue;
                }
                let c32 = c as u32;
                // Emitting the same token again needs a blank in between;
                // only the blank-ending mass can extend with it.
                let p = if beam.prefix.last() == Some(&c32) {
                    beam.p_b + logp[c]
                } else {
                    p_tot + logp[c]
                };
                if p == f32::NEG_INFINITY {
                    continue;
                }

                let mut prefix = beam.prefix.clone();
                prefix.push(c32);
                match index.get(&prefix) {
                    // Same extended prefix from another parent: the LM score
                    // is a function of the prefix alone, so it already matches.
                    Some(&i) => next[i].p_nb = log_add(next[i].p_nb, p),
                    None => {
                        let mut ctx = beam.ctx.clone();
                        ctx.push(c32);
                        if ctx.len() > ctx_len {
                            ctx.drain(..ctx.len() - ctx_len);
                        }
                        let lm_score = beam.lm + lm.word_score(&beam.ctx, c32);
                        index.insert(prefix.clone(), next.len());
                        next.push(Beam {
                            prefix,
                            p_b: f32::NEG_INFINITY,
                            p_nb: p,
                            lm: lm_score,
                            ctx,
                        });
                    }
                }
            }
        }

        next.sort_by(|a, b| b.score().total_cmp(&a.score()));
        next.truncate(BEAM_WIDTH);
        beams = next;
    }

    // Sentence-boundary </s> score, then pick the winner.
    beams
        .into_iter()
        .map(|mut b| {
            b.lm += lm.end_score(&b.ctx);
            b
        })
        .max_by(|a, b| a.score().total_cmp(&b.score()))
        .map(|b| b.prefix)
        .unwrap_or_default()
}

/// Fold a (prefix, p_b, p_nb) contribution into `next`, log-summing with any
/// beam that already has this prefix.
fn merge(
    next: &mut Vec<Beam>,
    index: &mut std::collections::HashMap<Vec<u32>, usize>,
    prefix: &[u32],
    p_b: f32,
    p_nb: f32,
    lm_score: f32,
    ctx: &[u32],
) {
    match index.get(prefix) {
        Some(&i) => {
            next[i].p_b = log_add(next[i].p_b, p_b);
            next[i].p_nb = log_add(next[i].p_nb, p_nb);
        }
        None => {
            index.insert(prefix.to_vec(), next.len());
            next.push(Beam {
                prefix: prefix.to_vec(),
                p_b,
                p_nb,
                lm: lm_score,
                ctx: ctx.to_vec(),
            });
        }
    }
}

fn log_softmax(row: &[f32], out: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (o, &v) in out.iter_mut().zip(row) {
        let e = v - max;
        *o = e;
        sum += e.exp();
    }
    let log_sum = sum.ln();
    for o in out.iter_mut() {
        *o -= log_sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Serialize a tiny LM in the VPL1 layout `export_lm.py` produces.
    /// `grams[o - 1]` holds (ids, log10 prob, log10 backoff) tuples for order o.
    fn write_lm(path: &Path, order: u32, grams: &[Vec<(Vec<u32>, f32, f32)>]) {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend(b"VPL1");
        for v in [order, 512, 513, 514, 0] {
            buf.extend(v.to_le_bytes());
        }
        let mut sorted: Vec<Vec<(u64, f32, f32)>> = grams
            .iter()
            .map(|g| {
                let mut v: Vec<_> = g.iter().map(|(ids, p, b)| (pack(ids), *p, *b)).collect();
                v.sort_by_key(|e| e.0);
                v
            })
            .collect();
        sorted.resize(order as usize, Vec::new());
        for g in &sorted {
            buf.extend((g.len() as u64).to_le_bytes());
        }
        for (i, g) in sorted.iter().enumerate() {
            for (k, ..) in g {
                buf.extend(k.to_le_bytes());
            }
            for (_, p, _) in g {
                buf.extend(p.to_le_bytes());
            }
            if (i as u32) < order - 1 {
                for (.., b) in g {
                    buf.extend(b.to_le_bytes());
                }
            }
            while buf.len() % 8 != 0 {
                buf.push(0);
            }
        }
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&buf).unwrap();
    }

    fn tiny_lm(name: &str) -> LmModel {
        let path = std::env::temp_dir().join(name);
        write_lm(
            &path,
            2,
            &[
                vec![
                    (vec![1], -1.0, -0.5),
                    (vec![2], -2.0, 0.0),
                    (vec![513], -3.0, 0.0),
                    (vec![514], -5.0, 0.0),
                ],
                vec![(vec![1, 2], -0.2, 0.0)],
            ],
        );
        LmModel::load(&path).unwrap()
    }

    #[test]
    fn lm_backoff_scoring() {
        let lm = tiny_lm("vp-lm-backoff.bin");
        let close = |a: f32, b: f32| (a - b).abs() < 1e-6;
        // Direct unigram.
        assert!(close(lm.score(&[], 1), -1.0));
        // Bigram hit.
        assert!(close(lm.score(&[1], 2), -0.2));
        // Bigram miss, context has no backoff weight: unigram only.
        assert!(close(lm.score(&[2], 1), -1.0));
        // Bigram miss with context backoff: bo(1) + P(2).
        assert!(close(lm.score(&[1], 513), -0.5 + -3.0));
        // OOV word: bo(1) + P(<unk>), and is_oov flags it.
        assert!(close(lm.score(&[1], 99), -0.5 + -5.0));
        assert!(lm.is_oov(99) && !lm.is_oov(1));
    }

    #[test]
    fn beam_search_matches_greedy_on_unambiguous_input() {
        let lm = tiny_lm("vp-lm-beam.bin");
        // vocab 3, blank 0; argmax path 0,1,1,0,2,2 -> collapsed [1, 2].
        let rows = [
            [9.0f32, 0.0, 0.0],
            [0.0, 9.0, 0.0],
            [0.0, 9.0, 0.0],
            [9.0, 0.0, 0.0],
            [0.0, 0.0, 9.0],
            [0.0, 0.0, 9.0],
        ];
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        assert_eq!(beam_search(&flat, rows.len(), 3, 0, &lm), vec![1, 2]);
    }

    /// Spot-check the exported medasr-lm.bin against values read straight out
    /// of Google's lm_6.arpa (token ids resolved via tokenizer.json):
    ///   "." = 9, "s" = 5, "c" = 30. Needs the 448 MB asset, so `#[ignore]`d:
    ///   MEDASR_MODELS_DIR=%APPDATA%/com.voicepill.app/models \
    ///   cargo test --lib lm_real_file -- --ignored
    #[test]
    #[ignore]
    fn lm_real_file_spot_checks() {
        let dir = std::env::var("MEDASR_MODELS_DIR").expect("set MEDASR_MODELS_DIR");
        let lm = LmModel::load(&Path::new(&dir).join("medasr-lm.bin")).expect("load lm");
        let close = |a: f32, b: f32| (a - b).abs() < 1e-4;

        let (p, _) = lm.lookup(1, 514).expect("<unk> unigram");
        assert!(close(p, -4.483_971));
        let (_, b) = lm.lookup(1, 512).expect("<s> unigram");
        assert!(close(b, -3.023_566_7));
        let (p, _) = lm.lookup(2, (512 << 10) | 513).expect("<s> </s> bigram");
        assert!(close(p, -4.124_89));
        let key = (((9u64 << 10) | 5) << 10) | 513;
        let (p, _) = lm.lookup(3, key).expect(". s </s> trigram");
        assert!(close(p, -1.854_261_8));
        // Full backoff query resolves to the exact 3-gram probability.
        assert!(close(lm.score(&[9, 5], 513), -1.854_261_8));
    }
}
