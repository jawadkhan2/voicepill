//! Spoken formatting commands for the general-dictation engines (Parakeet,
//! Whisper): saying "next line" inserts a line break instead of typing the
//! words.
//!
//! These models transcribe a command like any other words, and punctuate it
//! from the pause around it ("Thanks. Next line. See you soon."), so commands
//! are matched on whole words, case-insensitively, ignoring punctuation glued
//! to them. The punctuation the model attached to the command itself is
//! dropped with it, as is a comma left dangling right before the break.
//!
//! The list is deliberately small: only multi-word phrases that rarely occur
//! in ordinary prose. Single words like "period" or "comma" would fire on
//! normal sentences ("a long period of time"), and these models punctuate on
//! their own anyway. MedASR has its own, larger command set (it emits
//! `{commands}` natively); see `medasr::post_process`.

/// `(spoken phrase, inserted text)`. Matched longest-first.
const COMMANDS: &[(&str, &str)] = &[
    ("new paragraph", "\n\n"),
    ("next paragraph", "\n\n"),
    ("new line", "\n"),
    ("next line", "\n"),
    ("line break", "\n"),
];

/// One whitespace-separated word of the transcript.
struct Word<'a> {
    /// Leading punctuation, e.g. `"` in `"Next`.
    pre: &'a str,
    /// The word itself, e.g. `Next`.
    core: &'a str,
    /// Trailing punctuation, e.g. `.` in `line.`.
    post: &'a str,
}

fn split_word(w: &str) -> Word<'_> {
    let is_core = |c: char| c.is_alphanumeric() || c == '\'' || c == '’';
    let start = w.find(is_core).unwrap_or(w.len());
    let end = w.rfind(is_core).map(|i| i + w[i..].chars().next().unwrap().len_utf8());
    let end = end.unwrap_or(start).max(start);
    Word {
        pre: &w[..start],
        core: &w[start..end],
        post: &w[end..],
    }
}

/// Number of words of `phrase` and its replacement, if `words[i..]` starts
/// with a command.
fn match_at(words: &[Word], i: usize) -> Option<(usize, &'static str)> {
    let mut best: Option<(usize, &'static str)> = None;
    for (phrase, text) in COMMANDS {
        let parts: Vec<&str> = phrase.split(' ').collect();
        if i + parts.len() > words.len() {
            continue;
        }
        let hit = parts.iter().enumerate().all(|(k, p)| {
            let w = &words[i + k];
            // Punctuation may sit before the first word or after the last one,
            // but not inside the phrase ("new. Line" is two sentences).
            let inner_ok = (k == 0 || w.pre.is_empty()) && (k + 1 == parts.len() || w.post.is_empty());
            inner_ok && w.core.eq_ignore_ascii_case(p)
        });
        if hit && best.map(|(n, _)| parts.len() > n).unwrap_or(true) {
            best = Some((parts.len(), text));
        }
    }
    best
}

/// Replace spoken commands in `text` with the characters they stand for.
pub fn apply(text: &str) -> String {
    let raw: Vec<&str> = text.split_whitespace().collect();
    let words: Vec<Word> = raw.iter().map(|w| split_word(w)).collect();
    if !(0..words.len()).any(|i| match_at(&words, i).is_some()) {
        return text.to_string();
    }

    let mut out = String::with_capacity(text.len());
    let mut capitalize_next = false;
    let mut i = 0;
    while i < words.len() {
        if let Some((n, insert)) = match_at(&words, i) {
            // Keep punctuation the model put *before* the command's first word
            // (e.g. an opening quote is unlikely, but keep it rather than lose
            // text) — then drop spaces and a dangling comma before the break.
            out.push_str(words[i].pre);
            while out.ends_with(' ') {
                out.pop();
            }
            if out.ends_with([',', ';']) {
                out.pop();
            }
            out.push_str(insert);
            capitalize_next = true;
            i += n;
            continue;
        }
        if !out.is_empty() && !out.ends_with('\n') {
            out.push(' ');
        }
        let w = raw[i];
        if capitalize_next {
            out.push_str(&capitalize_first(w));
            capitalize_next = false;
        } else {
            out.push_str(w);
        }
        i += 1;
    }
    out
}

fn capitalize_first(w: &str) -> String {
    match w.char_indices().find(|(_, c)| c.is_alphabetic()) {
        Some((idx, c)) => {
            let mut s = String::with_capacity(w.len());
            s.push_str(&w[..idx]);
            s.extend(c.to_uppercase());
            s.push_str(&w[idx + c.len_utf8()..]);
            s
        }
        None => w.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::apply;

    #[test]
    fn next_line_between_sentences() {
        assert_eq!(apply("Thanks. Next line. See you soon."), "Thanks.\nSee you soon.");
    }

    #[test]
    fn mid_sentence_commas_are_cleaned_and_next_word_capitalised() {
        assert_eq!(apply("Dear Sam, new paragraph, thanks for the notes."), "Dear Sam\n\nThanks for the notes.");
    }

    #[test]
    fn lone_command_is_just_the_break() {
        assert_eq!(apply("Next line."), "\n");
        assert_eq!(apply("New paragraph"), "\n\n");
    }

    #[test]
    fn trailing_command_ends_with_break() {
        assert_eq!(apply("Buy milk, next line"), "Buy milk\n");
    }

    #[test]
    fn longest_phrase_wins_and_case_is_ignored() {
        assert_eq!(apply("one NEW PARAGRAPH two"), "one\n\nTwo");
    }

    #[test]
    fn ordinary_text_is_untouched() {
        let s = "The new deadline is next Monday, line by line.";
        assert_eq!(apply(s), s);
        assert_eq!(apply("A new. Line here"), "A new. Line here");
    }
}
