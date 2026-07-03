//! MedASR medical dictation via ONNX Runtime.
//!
//! MedASR is Google's LASR (Conformer-CTC) model — a different architecture from
//! Whisper, so it runs on its own inference backend rather than through
//! `whisper-rs`. The model is exported to a self-contained ONNX graph that takes a
//! raw 16 kHz mono f32 waveform and emits per-frame CTC logits `[1, frames, 512]`;
//! the log-mel frontend is baked into the graph, so this backend only has to feed
//! the same `&[f32]` buffer the audio coordinator already produces.
//!
//! Inference is CPU-only on purpose: the model is 105M params and transcribes ~44s
//! of audio in well under a second on CPU, so we avoid a second CUDA consumer in
//! the process (whisper.cpp already owns the GPU) and the packaging headache that
//! comes with it.
//!
//! Decoding mirrors the reference `LasrTokenizer` exactly:
//!   argmax each frame → collapse consecutive duplicate ids (CTC squash) →
//!   drop the blank/pad id (0) → SentencePiece detokenize via the `tokenizers`
//!   crate → map spoken punctuation (`{period}` → ".", …) to real characters.

use std::path::PathBuf;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use tokenizers::Tokenizer;

use crate::medasr_lm::{beam_search, LmModel};

/// CTC blank / pad id in the model's 512-way output. Everything collapses around it.
const BLANK_ID: i64 = 0;

/// Peak level to normalize the input waveform to. The LASR log-mel frontend has
/// no mean/variance normalization (`log(max(·, 1e-5))` on raw mel energies), so
/// a quiet mic pushes every bin toward the log floor and costs accuracy.
const TARGET_PEAK: f32 = 0.95;
/// Gain cap so a borderline-silent buffer isn't amplified into pure noise.
const MAX_GAIN: f32 = 30.0;

pub struct MedAsrBackend {
    models_dir: PathBuf,
    /// (model id, ONNX session, tokenizer, optional n-gram LM). `None` until
    /// first load. Without the LM file decoding falls back to greedy CTC.
    loaded: Option<(String, Session, Tokenizer, Option<LmModel>)>,
}

impl MedAsrBackend {
    pub fn new(models_dir: PathBuf) -> Self {
        MedAsrBackend {
            models_dir,
            loaded: None,
        }
    }

    fn onnx_path(&self, id: &str) -> PathBuf {
        self.models_dir.join(format!("{id}.onnx"))
    }

    fn tokenizer_path(&self, id: &str) -> PathBuf {
        self.models_dir.join(format!("{id}-tokenizer.json"))
    }

    fn lm_path(&self, id: &str) -> PathBuf {
        self.models_dir.join(format!("{id}-lm.bin"))
    }

    pub fn is_loaded(&self, id: &str) -> bool {
        self.loaded.as_ref().map(|(m, ..)| m == id).unwrap_or(false)
    }

    /// Load the ONNX session + tokenizer for `id` (reloading if the id changed).
    pub fn ensure_loaded(&mut self, id: &str) -> Result<(), String> {
        if self.is_loaded(id) {
            return Ok(());
        }

        let onnx = self.onnx_path(id);
        let tok = self.tokenizer_path(id);
        if !onnx.exists() {
            return Err(format!("model '{id}' is not downloaded"));
        }
        if !tok.exists() {
            return Err(format!("tokenizer for '{id}' is missing"));
        }

        let session = Session::builder()
            .map_err(|e| format!("ort session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("ort opt level: {e}"))?
            .commit_from_file(&onnx)
            .map_err(|e| format!("failed to load model '{id}': {e}"))?;

        let tokenizer =
            Tokenizer::from_file(&tok).map_err(|e| format!("failed to load tokenizer: {e}"))?;

        // The 6-gram LM turns greedy 6.6% WER into ~4.6% (model card). It is
        // optional on disk: without it we still transcribe, just greedily.
        let lm_file = self.lm_path(id);
        let lm = if lm_file.exists() {
            match LmModel::load(&lm_file) {
                Ok(lm) => Some(lm),
                Err(e) => {
                    eprintln!("[medasr] LM load failed ({e}); falling back to greedy decode");
                    None
                }
            }
        } else {
            None
        };

        self.loaded = Some((id.to_string(), session, tokenizer, lm));
        Ok(())
    }

    /// Transcribe a 16 kHz mono f32 buffer. `language` is ignored (English only).
    pub fn transcribe(&mut self, id: &str, audio: &[f32]) -> Result<String, String> {
        self.ensure_loaded(id)?;
        let (_, session, tokenizer, lm) = self.loaded.as_mut().unwrap();

        if audio.is_empty() {
            return Ok(String::new());
        }
        // The baked-in frame op needs at least one full STFT window (win_length=400).
        if audio.len() < 400 {
            return Ok(String::new());
        }

        // [1, N] waveform tensor, peak-normalized (see TARGET_PEAK).
        let peak = audio.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        let wave: Vec<f32> = if peak > 0.0 && peak < TARGET_PEAK {
            let gain = (TARGET_PEAK / peak).min(MAX_GAIN);
            audio.iter().map(|s| s * gain).collect()
        } else {
            audio.to_vec()
        };
        let input = Value::from_array(([1usize, wave.len()], wave))
            .map_err(|e| format!("input tensor: {e}"))?;

        let outputs = session
            .run(ort::inputs!["waveform" => input])
            .map_err(|e| format!("inference failed: {e}"))?;

        let (shape, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract logits: {e}"))?;

        // logits shape is [1, frames, vocab].
        let frames = shape[1] as usize;
        let vocab = shape[2] as usize;

        let ids = match lm {
            Some(lm) => beam_search(logits, frames, vocab, BLANK_ID as usize, lm),
            None => greedy_ctc(logits, frames, vocab),
        };
        let raw = tokenizer
            .decode(&ids, false)
            .map_err(|e| format!("decode failed: {e}"))?;
        Ok(post_process(&raw))
    }
}

/// Per-frame argmax → collapse consecutive duplicates → drop blank id.
/// Returns the SentencePiece ids to hand to the tokenizer.
fn greedy_ctc(logits: &[f32], frames: usize, vocab: usize) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::with_capacity(frames / 2);
    let mut prev: i64 = -1;
    for f in 0..frames {
        let row = &logits[f * vocab..(f + 1) * vocab];
        let mut best = 0usize;
        let mut best_v = row[0];
        for (i, &v) in row.iter().enumerate().skip(1) {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        let id = best as i64;
        if id != prev {
            // new run: emit unless it's the blank
            if id != BLANK_ID {
                out.push(id as u32);
            }
            prev = id;
        }
    }
    out
}

/// A dictation command's effect on the output stream.
enum Cmd {
    /// Replacement text, plus whether the space before / after the `{…}` token
    /// should be eaten. Punctuation attaches to the preceding word
    /// ("PE {period}" → "PE."); openers attach to the following one
    /// ("{open paren} stable" → "(stable"); joiners eat both ("5 {dash} 10" → "5-10").
    Text(&'static str, bool, bool),
    /// Uppercase the next word ("{all caps} copd" → "COPD").
    UpperNext,
    /// Capitalize the next word ("{cap} smith" → "Smith").
    CapNext,
    /// Uppercase everything until `{all caps off}`.
    UpperOn,
    UpperOff,
    /// Lowercase the next word ("{no caps} DNA" → "dna").
    LowerNext,
    /// Lowercase everything until `{no caps off}`.
    LowerOn,
    LowerOff,
}

fn command(phrase: &str) -> Option<Cmd> {
    use Cmd::*;
    Some(match phrase {
        "period" | "full stop" | "dot" => Text(".", true, false),
        "comma" => Text(",", true, false),
        "colon" => Text(":", true, false),
        "semicolon" | "semi colon" => Text(";", true, false),
        "question mark" => Text("?", true, false),
        "exclamation point" | "exclamation mark" => Text("!", true, false),
        "ellipsis" | "dot dot dot" => Text("…", true, false),
        "hyphen" | "dash" => Text("-", true, true),
        "em dash" => Text("—", true, true),
        "en dash" => Text("–", true, true),
        "slash" | "forward slash" => Text("/", true, true),
        "backslash" | "back slash" => Text("\\", true, true),
        "at sign" | "at symbol" => Text("@", true, true),
        "ampersand" | "and sign" => Text("&", true, true),
        "percent" | "percent sign" => Text("%", true, false),
        "degree" | "degrees" | "degree sign" => Text("°", true, false),
        "plus" | "plus sign" => Text("+", true, true),
        "minus sign" => Text("-", true, true),
        "equals" | "equals sign" | "equal sign" => Text("=", true, true),
        "underscore" => Text("_", true, true),
        "asterisk" | "star" => Text("*", true, true),
        "caret" => Text("^", true, true),
        "tilde" => Text("~", false, true),
        "backtick" | "back tick" | "grave accent" => Text("`", true, true),
        "pipe" | "vertical bar" => Text("|", true, true),
        "hash" | "hashtag" | "hash sign" | "pound sign" | "number sign" => Text("#", false, true),
        "dollar sign" => Text("$", false, true),
        "euro sign" => Text("€", false, true),
        "cent sign" => Text("¢", true, false),
        "section sign" => Text("§", false, true),
        "copyright sign" => Text("©", true, true),
        "trademark sign" => Text("™", true, false),
        "micro sign" => Text("µ", false, true),
        "bullet" | "bullet point" => Text("•", false, false),
        "times sign" | "multiplication sign" => Text("×", true, true),
        "division sign" | "divided by sign" => Text("÷", true, true),
        "plus or minus" | "plus minus" | "plus minus sign" => Text("±", false, true),
        "less than" | "less than sign" => Text("<", false, true),
        "greater than" | "greater than sign" => Text(">", false, true),
        "less than or equal to" | "less than or equals" => Text("≤", false, true),
        "greater than or equal to" | "greater than or equals" => Text("≥", false, true),
        "open paren" | "open parenthesis" | "left paren" | "left parenthesis" => {
            Text("(", false, true)
        }
        "close paren" | "close parenthesis" | "right paren" | "right parenthesis" => {
            Text(")", true, false)
        }
        "open bracket" | "left bracket" => Text("[", false, true),
        "close bracket" | "right bracket" => Text("]", true, false),
        "open brace" | "left brace" | "open curly brace" => Text("{", false, true),
        "close brace" | "right brace" | "close curly brace" => Text("}", true, false),
        "open angle bracket" | "left angle bracket" => Text("<", false, true),
        "close angle bracket" | "right angle bracket" => Text(">", true, false),
        "open quote" | "open quotes" | "begin quote" | "begin quotes" => Text("\"", false, true),
        "close quote" | "close quotes" | "end quote" | "end quotes" => Text("\"", true, false),
        "open single quote" => Text("'", false, true),
        "close single quote" | "end single quote" => Text("'", true, false),
        "apostrophe" | "single quote" => Text("'", true, true),
        "apostrophe s" => Text("'s", true, false),
        "no space" => Text("", true, true),
        "new line" | "next line" | "line break" => Text("\n", true, true),
        "new paragraph" | "next paragraph" | "paragraph" | "paragraph break" => {
            Text("\n\n", true, true)
        }
        "tab" | "tab key" => Text("\t", true, true),
        "all caps on" | "caps on" => UpperOn,
        "all caps off" | "caps off" | "end all caps" | "end caps" => UpperOff,
        "all caps" | "in all caps" => UpperNext,
        "cap" | "caps" | "capital" | "capitalize" => CapNext,
        "no caps on" | "lowercase on" => LowerOn,
        "no caps off" | "lowercase off" | "end no caps" | "end lowercase" => LowerOff,
        "no caps" | "lowercase" | "lower case" => LowerNext,
        _ => return None,
    })
}

/// Lowercase and collapse inner whitespace so "{Next  Line}" still matches.
fn normalize(phrase: &str) -> String {
    phrase
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Longest command phrase, in words, recognised by [`command`]
/// ("greater than or equal to"). Bounds the brace-repair scans below.
const MAX_CMD_WORDS: usize = 5;

/// Longest known command (≤[`MAX_CMD_WORDS`] words) that is a prefix of `s`;
/// returns its end byte offset.
fn prefix_command_end(s: &str) -> Option<usize> {
    let mut ends = Vec::new();
    let mut in_word = false;
    for (i, ch) in s.char_indices() {
        if ch.is_whitespace() {
            if in_word {
                ends.push(i);
                in_word = false;
                if ends.len() == MAX_CMD_WORDS {
                    break;
                }
            }
        } else {
            in_word = true;
        }
    }
    if in_word && ends.len() < MAX_CMD_WORDS {
        ends.push(s.len());
    }
    ends.iter()
        .rev()
        .copied()
        .find(|&end| command(&normalize(&s[..end])).is_some())
}

/// Longest known command (≤[`MAX_CMD_WORDS`] words) that is a suffix of `s`;
/// returns its start byte offset.
fn suffix_command_start(s: &str) -> Option<usize> {
    let trimmed = s.trim_end();
    let mut starts = Vec::new();
    let mut in_word = false;
    for (i, ch) in trimmed.char_indices() {
        if ch.is_whitespace() {
            in_word = false;
        } else if !in_word {
            in_word = true;
            starts.push(i);
        }
    }
    let take = starts.len().min(MAX_CMD_WORDS);
    starts[starts.len() - take..]
        .iter()
        .copied()
        .find(|&start| command(&normalize(&trimmed[start..])).is_some())
}

/// The model sometimes drops one brace of a command at utterance edges — a lone
/// "next line" comes out as "next line}", and a truncated tail as "{next line".
/// Re-insert the missing brace when the words next to the orphan brace spell a
/// known command; anything else is left untouched.
fn repair_braces(t: &str) -> String {
    let mut out = String::with_capacity(t.len() + 2);
    let mut rest = t;
    while let Some(i) = rest.find(['{', '}']) {
        if rest.as_bytes()[i] == b'{' {
            let after = &rest[i + 1..];
            match after.find(['{', '}']) {
                // Properly closed — copy the whole token through.
                Some(j) if after.as_bytes()[j] == b'}' => {
                    out.push_str(&rest[..i + 2 + j]);
                    rest = &rest[i + 2 + j..];
                }
                // Orphan opener: close it if a known command follows.
                _ => {
                    out.push_str(&rest[..i + 1]);
                    if let Some(end) = prefix_command_end(after) {
                        out.push_str(after[..end].trim());
                        out.push('}');
                        rest = &after[end..];
                    } else {
                        rest = after;
                    }
                }
            }
        } else {
            // Orphan closer: open it if a known command precedes.
            let before = &rest[..i];
            if let Some(start) = suffix_command_start(before) {
                out.push_str(&before[..start]);
                out.push('{');
                out.push_str(before[start..].trim_end());
            } else {
                out.push_str(before);
            }
            out.push('}');
            rest = &rest[i + 1..];
        }
    }
    out.push_str(rest);
    out
}

/// Casing state driven by the caps commands, applied word-by-word.
#[derive(Default)]
struct Caps {
    all: bool,
    upper_next: bool,
    cap_next: bool,
    lower: bool,
    lower_next: bool,
}

impl Caps {
    fn active(&self) -> bool {
        self.all || self.upper_next || self.cap_next || self.lower || self.lower_next
    }
}

fn push_word(out: &mut String, word: &mut String, caps: &mut Caps) {
    if word.is_empty() {
        return;
    }
    if caps.all || caps.upper_next {
        out.push_str(&word.to_uppercase());
    } else if caps.cap_next {
        let mut chars = word.chars();
        let first = chars.next().unwrap();
        out.extend(first.to_uppercase());
        out.push_str(chars.as_str());
    } else if caps.lower || caps.lower_next {
        out.push_str(&word.to_lowercase());
    } else {
        out.push_str(word);
    }
    caps.upper_next = false;
    caps.cap_next = false;
    caps.lower_next = false;
    word.clear();
}

/// Append literal text, applying the caps state per word.
fn push_cased(out: &mut String, s: &str, caps: &mut Caps) {
    if !caps.active() {
        out.push_str(s);
        return;
    }
    let mut word = String::new();
    for ch in s.chars() {
        if ch.is_whitespace() {
            push_word(out, &mut word, caps);
            out.push(ch);
        } else {
            word.push(ch);
        }
    }
    push_word(out, &mut word, caps);
}

/// Expand `{command}` tokens via [`command`], eating adjacent spaces per each
/// command's spacing spec and applying caps commands to the following words.
/// Unknown commands pass through verbatim so nothing dictated is silently
/// dropped.
fn expand_commands(t: &str) -> String {
    let mut out = String::with_capacity(t.len());
    let mut caps = Caps::default();
    let mut rest = t;
    while let Some(start) = rest.find('{') {
        push_cased(&mut out, &rest[..start], &mut caps);
        let after = &rest[start..];
        let Some(end) = after.find('}') else {
            push_cased(&mut out, after, &mut caps);
            return out;
        };
        match command(&normalize(&after[1..end])) {
            Some(Cmd::Text(text, eat_before, eat_after)) => {
                if eat_before {
                    while out.ends_with(' ') {
                        out.pop();
                    }
                }
                out.push_str(text);
                rest = &after[end + 1..];
                if eat_after {
                    rest = rest.trim_start_matches(' ');
                }
            }
            Some(cmd) => {
                match cmd {
                    Cmd::UpperNext => caps.upper_next = true,
                    Cmd::CapNext => caps.cap_next = true,
                    Cmd::UpperOn => {
                        caps.all = true;
                        caps.lower = false;
                    }
                    Cmd::UpperOff => caps.all = false,
                    Cmd::LowerNext => caps.lower_next = true,
                    Cmd::LowerOn => {
                        caps.lower = true;
                        caps.all = false;
                    }
                    Cmd::LowerOff => caps.lower = false,
                    Cmd::Text(..) => unreachable!(),
                }
                // The token itself vanishes; the doubled space it leaves behind
                // is collapsed by post_process.
                rest = &after[end + 1..];
            }
            None => {
                out.push_str(&after[..end + 1]);
                rest = &after[end + 1..];
            }
        }
    }
    push_cased(&mut out, rest, &mut caps);
    out
}

/// Turn MedASR's spoken-command tokens into real characters, and clean spacing.
/// The model emits e.g. `{period}` / `{next line}` / `{new paragraph}` and section
/// headers like `[FINDINGS]`; here we repair dropped braces, expand the commands
/// and normalise whitespace, leaving section headers intact.
fn post_process(s: &str) -> String {
    let stripped = s.replace("</s>", "");
    let repaired = {
        let r = repair_braces(&stripped);
        // A lone command sometimes arrives with no braces at all ("next line");
        // when the whole utterance is one known command, treat it as one.
        let whole = r.trim();
        if !whole.contains('{') && command(&normalize(whole)).is_some() {
            format!("{{{whole}}}")
        } else {
            r
        }
    };
    let t = expand_commands(&repaired);

    // Collapse runs of spaces (not newlines/tabs — `{tab}` must survive), drop
    // spaces hugging a newline (`{new paragraph} impression` leaves
    // "\n\n impression"), and trim spaces — but not newlines/tabs, so a lone
    // "{next line}" still pastes a line break.
    let mut cleaned = String::with_capacity(t.len());
    for ch in t.chars() {
        match ch {
            ' ' => {
                if !matches!(cleaned.chars().last(), Some(' ') | Some('\n') | None) {
                    cleaned.push(' ');
                }
            }
            '\n' => {
                while cleaned.ends_with(' ') {
                    cleaned.pop();
                }
                cleaned.push('\n');
            }
            _ => cleaned.push(ch),
        }
    }
    cleaned.trim_matches(' ').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn post_process_maps_spoken_punctuation() {
        let raw = "the lungs are clear {period} no effusion {comma} normal {new paragraph} impression {colon} normal</s>";
        assert_eq!(
            post_process(raw),
            "the lungs are clear. no effusion, normal\n\nimpression: normal"
        );
    }

    #[test]
    fn post_process_maps_line_commands_and_variants() {
        let raw = "line one {next line} line two {next paragraph} line three";
        assert_eq!(post_process(raw), "line one\nline two\n\nline three");
    }

    #[test]
    fn post_process_spacing_rules() {
        // Joiners eat both sides; openers attach forward; closers attach backward.
        assert_eq!(post_process("5 {dash} 10 mm"), "5-10 mm");
        assert_eq!(
            post_process("stable {open paren} unchanged {close paren} today"),
            "stable (unchanged) today"
        );
        assert_eq!(post_process("BUN {slash} creatinine"), "BUN/creatinine");
        assert_eq!(post_process("t {- } max"), "t {- } max"); // not a command
    }

    #[test]
    fn post_process_keeps_unknown_commands_and_headers() {
        let raw = "[FINDINGS] heart size normal {period} {frobnicate} widget";
        assert_eq!(
            post_process(raw),
            "[FINDINGS] heart size normal. {frobnicate} widget"
        );
    }

    #[test]
    fn post_process_orphan_braces_repaired_for_known_commands() {
        // Model dropped the opening brace at utterance start.
        assert_eq!(post_process("next line}"), "\n");
        assert_eq!(
            post_process("clear lungs next line} no effusion"),
            "clear lungs\nno effusion"
        );
        // Dropped closing brace.
        assert_eq!(post_process("{next line"), "\n");
        assert_eq!(post_process("odd {period text"), "odd. text");
        // Orphan braces around unknown words stay verbatim.
        assert_eq!(post_process("count 5}"), "count 5}");
        assert_eq!(post_process("odd {zzz text"), "odd {zzz text");
    }

    #[test]
    fn post_process_bare_command_transcript() {
        // Whole utterance is a single spoken command, braces missing entirely.
        assert_eq!(post_process("next line"), "\n");
        assert_eq!(post_process("new paragraph"), "\n\n");
        assert_eq!(post_process("period"), ".");
    }

    #[test]
    fn post_process_paragraph_alias() {
        assert_eq!(post_process("one {paragraph} two"), "one\n\ntwo");
        assert_eq!(post_process("{paragraph}"), "\n\n");
    }

    #[test]
    fn post_process_all_caps_next_word() {
        assert_eq!(
            post_process("diagnosis {all caps} copd confirmed"),
            "diagnosis COPD confirmed"
        );
        assert_eq!(post_process("{all caps} copd"), "COPD");
    }

    #[test]
    fn post_process_caps_range_and_cap_word() {
        assert_eq!(
            post_process("{all caps on} chest x-ray {all caps off} normal"),
            "CHEST X-RAY normal"
        );
        assert_eq!(post_process("doctor {cap} smith"), "doctor Smith");
    }

    #[test]
    fn post_process_symbol_commands() {
        assert_eq!(post_process("cost {dollar sign} 50"), "cost $50");
        assert_eq!(post_process("BP {greater than} 120"), "BP >120");
        assert_eq!(post_process("CD4 {less than or equal to} 200"), "CD4 ≤200");
        assert_eq!(post_process("a {pipe} b"), "a|b");
        assert_eq!(post_process("dose {plus or minus} 5 mg"), "dose ±5 mg");
        assert_eq!(post_process("item {hash} 3"), "item #3");
        assert_eq!(post_process("pause {em dash} resume"), "pause—resume");
        assert_eq!(post_process("{open brace} key {close brace}"), "{key}");
        assert_eq!(post_process("{bullet} first point"), "• first point");
    }

    #[test]
    fn post_process_no_space_joins_words() {
        assert_eq!(post_process("wave {no space} form"), "waveform");
        assert_eq!(post_process("{no space}"), "");
    }

    #[test]
    fn post_process_lowercase_commands() {
        assert_eq!(post_process("{no caps} URGENT note"), "urgent note");
        assert_eq!(
            post_process("{lowercase on} SOME WORDS {lowercase off} STAY"),
            "some words STAY"
        );
    }

    #[test]
    fn post_process_five_word_command_brace_repair() {
        // Longest command still repaired when a brace is dropped.
        assert_eq!(post_process("x {greater than or equal to y"), "x ≥y");
        assert_eq!(post_process("x greater than or equal to} y"), "x ≥y");
    }

    #[test]
    fn post_process_lone_line_break_survives_trim() {
        assert_eq!(post_process("{next line}"), "\n");
        assert_eq!(post_process("{tab}"), "\t");
        assert_eq!(post_process("text {new paragraph}"), "text\n\n");
    }

    #[test]
    fn greedy_ctc_collapses_and_drops_blank() {
        // vocab 3, blank=0. frames argmax: 0,1,1,0,2,2  -> collapse -> 0,1,0,2 -> drop 0 -> 1,2
        let vocab = 3;
        let rows = [
            [9.0, 0.0, 0.0], // 0 blank
            [0.0, 9.0, 0.0], // 1
            [0.0, 9.0, 0.0], // 1 (dup)
            [9.0, 0.0, 0.0], // 0 blank
            [0.0, 0.0, 9.0], // 2
            [0.0, 0.0, 9.0], // 2 (dup)
        ];
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        assert_eq!(greedy_ctc(&flat, rows.len(), vocab), vec![1u32, 2u32]);
    }

    /// Full ONNX + decode against Google's radiology test clip. Needs the 404 MB
    /// asset, so it's `#[ignore]`d; run with the staged model:
    ///   MEDASR_MODELS_DIR=%APPDATA%/com.voicepill.app/models \
    ///   MEDASR_TEST_WAV=<path to google/medasr test_audio.wav> \
    ///   cargo test --release medasr_end_to_end -- --ignored --nocapture
    #[test]
    #[ignore]
    fn medasr_end_to_end() {
        let dir = std::env::var("MEDASR_MODELS_DIR").expect("set MEDASR_MODELS_DIR");
        let wav = std::env::var("MEDASR_TEST_WAV").expect("set MEDASR_TEST_WAV");
        let audio = read_pcm16_mono(&wav);

        let mut be = MedAsrBackend::new(PathBuf::from(dir));
        let text = be.transcribe("medasr", &audio).expect("transcribe");
        println!("\nMedASR transcript:\n{text}\n");

        let low = text.to_lowercase();
        for needle in ["chest", "pulmonary", "lobe", "embolus"] {
            assert!(low.contains(needle), "expected '{needle}' in: {text}");
        }
    }

    /// Minimal PCM16 mono WAV reader (skips the 44-byte canonical header).
    #[cfg(test)]
    fn read_pcm16_mono(path: &str) -> Vec<f32> {
        let bytes = std::fs::read(path).expect("read wav");
        // find "data" chunk
        let mut i = 12;
        let (mut off, mut len) = (44usize, bytes.len() - 44);
        while i + 8 <= bytes.len() {
            let id = &bytes[i..i + 4];
            let sz = u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]])
                as usize;
            if id == b"data" {
                off = i + 8;
                len = sz.min(bytes.len() - (i + 8));
                break;
            }
            i += 8 + sz + (sz & 1);
        }
        bytes[off..off + len]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect()
    }
}
