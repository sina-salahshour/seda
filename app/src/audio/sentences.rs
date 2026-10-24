/// Splits streaming text into complete sentences for TTS.
///
/// Sentence ends: `.`, `!`, `?`, `؟`, `؛`, `…`, newline (and blank lines),
/// but not a `.` inside a number like `2.5`. Very long runs without a stop are
/// cut at a comma (`,` or `،`) or space once they pass ~200 chars.
///
/// Also: `.` after an abbreviation (`e.g.`, `Dr.`, initials) or a list number
/// at the start of a line does not end a sentence; a fenced code block is kept
/// whole as one piece (so [`speakable`] can drop it); a piece with fewer than
/// three letters is merged into the next one.
#[derive(Debug, Default)]
pub struct SentenceSplitter {
    /// Text not yet emitted.
    buf: String,
    /// A short piece waiting to be merged with the next one.
    held: Option<String>,
    /// Whether `buf` starts at the beginning of a line.
    mid_line: bool,
}

/// Pieces longer than this (in chars) are cut at a comma or space.
const LONG_RUN: usize = 200;
/// A long-run cut does not go earlier than this.
const MIN_CUT: usize = 100;
/// Without any comma or space, cut hard at this length.
const HARD_CUT: usize = 400;
/// Pieces with fewer letters are merged into the next one.
const MIN_LETTERS: usize = 3;

const TERMINATORS: &[char] = &['.', '!', '?', '؟', '؛', '…'];
const CLOSERS: &[char] = &['"', '\'', ')', ']', '»', '”', '’', '*', '_'];
const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "fig", "no", "approx", "inc", "ltd",
    "co", "cf", "al",
];

impl SentenceSplitter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a delta; returns the sentences it completed, in order.
    pub fn push(&mut self, delta: &str) -> Vec<String> {
        self.buf.push_str(delta);
        let mut out = Vec::new();
        while let Some(end) = self.next_boundary() {
            let piece: String = self.buf.drain(..end).collect();
            self.mid_line = !piece.ends_with('\n');
            // Drop the spaces after a sentence, noting a line break among them.
            let spaces = self.buf.len() - self.buf.trim_start().len();
            if spaces > 0 {
                if self.buf[..spaces].contains('\n') {
                    self.mid_line = false;
                }
                self.buf.drain(..spaces);
            }
            if let Some(sentence) = self.merge(&piece, false) {
                out.push(sentence);
            }
        }
        out
    }

    /// The trailing partial sentence, if any.
    pub fn flush(&mut self) -> Option<String> {
        let rest = std::mem::take(&mut self.buf);
        self.mid_line = false;
        self.merge(&rest, true)
    }

    /// Discard everything buffered.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Join `piece` to a held short piece; hold the result if it is still too
    /// short and more text may follow.
    fn merge(&mut self, piece: &str, last: bool) -> Option<String> {
        let piece = piece.trim();
        let text = match (self.held.take(), piece.is_empty()) {
            (None, true) => return None,
            (None, false) => piece.to_string(),
            (Some(held), true) => held,
            (Some(held), false) => {
                let sep = if is_fence(piece) { "\n" } else { " " };
                format!("{held}{sep}{piece}")
            }
        };
        if !last && letter_count(&text) < MIN_LETTERS {
            self.held = Some(text);
            return None;
        }
        Some(text)
    }

    /// Byte index just past the next complete piece in `buf`, if any.
    fn next_boundary(&self) -> Option<usize> {
        let buf = self.buf.as_str();
        if buf.is_empty() {
            return None;
        }

        // A fenced code block is one piece, up to and including its closing fence line.
        if !self.mid_line && is_fence(buf.trim_start_matches([' ', '\t'])) {
            let first_nl = buf.find('\n')?;
            let mut offset = first_nl + 1;
            loop {
                let line_end = buf[offset..].find('\n')? + offset;
                if is_fence(buf[offset..line_end].trim_start()) {
                    return Some(line_end + 1);
                }
                offset = line_end + 1;
            }
        }

        let chars: Vec<(usize, char)> = buf.char_indices().collect();
        let mut in_code = false;
        let mut i = 0;
        while i < chars.len() {
            let (idx, c) = chars[i];
            if c == '\n' {
                return Some(idx + 1);
            }
            if c == '`' {
                in_code = !in_code;
                i += 1;
                continue;
            }
            if in_code || !TERMINATORS.contains(&c) {
                i += 1;
                continue;
            }

            // A run of terminators and closing marks.
            let mut j = i;
            while j < chars.len() && TERMINATORS.contains(&chars[j].1) {
                j += 1;
            }
            let single_dot = j - i == 1 && c == '.';
            while j < chars.len() && CLOSERS.contains(&chars[j].1) {
                j += 1;
            }
            let Some(&(next_idx, next)) = chars.get(j) else {
                // Undecided until the next char arrives (or `flush`).
                return None;
            };
            if !single_dot || (next.is_whitespace() && !self.dot_is_abbreviation(&buf[..idx])) {
                return Some(next_idx);
            }
            i = j;
        }

        if chars.len() > LONG_RUN {
            return Some(long_cut(&chars).unwrap_or(buf.len()));
        }
        None
    }

    /// Whether a `.` right after `before` belongs to an abbreviation, an
    /// initial, or a list number rather than ending a sentence.
    fn dot_is_abbreviation(&self, before: &str) -> bool {
        let start = before
            .rfind(char::is_whitespace)
            .map(|i| i + before[i..].chars().next().map_or(1, char::len_utf8))
            .unwrap_or(0);
        let token = before[start..].trim_start_matches(['(', '[', '"', '\'', '«', '*', '_']);
        if token.is_empty() {
            return false;
        }
        // `1.` / `۱.` at the start of a line: a list number.
        if token.chars().all(|c| c.is_numeric()) {
            return !self.mid_line && before[..start].trim().is_empty();
        }
        // `e.g.`, `i.e.`, `U.S.`
        if token.contains('.') {
            return true;
        }
        let lower = token.to_lowercase();
        if lower.chars().count() == 1 && lower.chars().all(|c| c.is_ascii_alphabetic()) {
            return true;
        }
        ABBREVIATIONS.contains(&lower.as_str())
    }
}

/// Where to cut a long run: after the last comma, else the last space, not
/// earlier than [`MIN_CUT`] chars in; else a hard cut.
fn long_cut(chars: &[(usize, char)]) -> Option<usize> {
    let after = |k: usize| chars.get(k + 1).map(|&(idx, _)| idx);
    let window = &chars[MIN_CUT.min(chars.len())..];
    let base = MIN_CUT.min(chars.len());
    if let Some(k) = window
        .iter()
        .rposition(|&(_, c)| matches!(c, ',' | '،' | ';'))
    {
        return after(base + k);
    }
    if let Some(k) = window.iter().rposition(|&(_, c)| c.is_whitespace()) {
        return after(base + k);
    }
    if chars.len() > HARD_CUT {
        return Some(chars[HARD_CUT].0);
    }
    None
}

fn is_fence(line: &str) -> bool {
    line.starts_with("```") || line.starts_with("~~~")
}

fn letter_count(text: &str) -> usize {
    text.chars().filter(|c| c.is_alphabetic()).count()
}

/// Strip markdown and code for speech: drop fenced code blocks, inline code
/// markers, `*`/`_`/`#`/`>` markup, link targets, tables; collapse spaces.
/// Returns `None` when nothing speakable is left.
///
/// Inline code keeps its content when it is short, and is dropped otherwise;
/// bare URLs and emoji are dropped.
pub fn speakable(text: &str) -> Option<String> {
    let mut lines = Vec::new();
    let mut in_fence = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if is_fence(trimmed) {
            in_fence = !in_fence;
            continue;
        }
        if in_fence || is_table_row(trimmed) {
            continue;
        }
        let line = strip_line_markers(trimmed);
        let line = inline_code(line);
        let line = links(&line);
        let line = drop_urls(&line);
        let line = emphasis(&line);
        let line = drop_emoji(&line);
        lines.push(line);
    }
    let joined = lines.join(" ");
    let collapsed = joined.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().any(char::is_alphabetic) {
        Some(collapsed)
    } else {
        None
    }
}

fn is_table_row(line: &str) -> bool {
    if line.starts_with('|') {
        return true;
    }
    // A separator row like `---|:---:` or a row without outer pipes.
    let pipes = line.matches('|').count();
    if pipes >= 2 {
        return true;
    }
    pipes >= 1 && line.chars().all(|c| matches!(c, '-' | ':' | '|' | ' '))
}

/// Strip leading `>`, `#`, list bullets and list numbers.
fn strip_line_markers(mut line: &str) -> &str {
    loop {
        let before = line;
        line = line.trim_start();
        if let Some(rest) = line.strip_prefix('>') {
            line = rest;
        } else if line.starts_with('#') {
            let rest = line.trim_start_matches('#');
            if rest.is_empty() || rest.starts_with(char::is_whitespace) {
                line = rest;
            }
        } else if let Some(rest) = ["- ", "* ", "+ ", "• ", "– "]
            .iter()
            .find_map(|bullet| line.strip_prefix(bullet))
        {
            line = rest;
        } else if let Some(rest) = strip_list_number(line) {
            line = rest;
        } else if let Some(rest) = line
            .strip_prefix("[ ] ")
            .or_else(|| line.strip_prefix("[x] "))
        {
            line = rest;
        }
        if line == before {
            return line;
        }
    }
}

fn strip_list_number(line: &str) -> Option<&str> {
    let digits = line.find(|c: char| !c.is_numeric())?;
    if digits == 0 {
        return None;
    }
    let rest = &line[digits..];
    let rest = rest.strip_prefix('.').or_else(|| rest.strip_prefix(')'))?;
    if rest.is_empty() || rest.starts_with(char::is_whitespace) {
        Some(rest)
    } else {
        None
    }
}

/// Inline code longer than this is dropped instead of read.
const MAX_INLINE_CODE: usize = 24;

fn inline_code(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(open) = rest.find('`') {
        out.push_str(&rest[..open]);
        let ticks = rest[open..].chars().take_while(|&c| c == '`').count();
        let after = &rest[open + ticks..];
        let fence = "`".repeat(ticks);
        match after.find(&fence) {
            Some(close) => {
                let code = after[..close].trim();
                if code.chars().count() <= MAX_INLINE_CODE {
                    out.push_str(code);
                } else {
                    out.push(' ');
                }
                rest = &after[close + ticks..];
            }
            None => {
                // Unmatched: drop the ticks, keep the text.
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// `[text](url)` → `text`, `![alt](url)` → `alt`.
fn links(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(open) = rest.find('[') {
        let Some(close) = rest[open..].find("](").map(|i| i + open) else {
            break;
        };
        let Some(end) = rest[close + 2..].find(')').map(|i| i + close + 2) else {
            break;
        };
        let mut head = &rest[..open];
        if let Some(h) = head.strip_suffix('!') {
            head = h;
        }
        out.push_str(head);
        out.push_str(&rest[open + 1..close]);
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
    out
}

fn drop_urls(line: &str) -> String {
    line.split_whitespace()
        .filter(|word| {
            let w = word
                .trim_start_matches(['<', '(', '"', '\''])
                .to_ascii_lowercase();
            !(w.starts_with("http://")
                || w.starts_with("https://")
                || w.starts_with("www.")
                || w.contains("://"))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Drop `*`, `~~` and emphasis `_`; an `_` inside a word becomes a space.
fn emphasis(line: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut out = String::with_capacity(line.len());
    for (i, &c) in chars.iter().enumerate() {
        match c {
            '*' => {}
            '~' if chars.get(i + 1) == Some(&'~') || (i > 0 && chars[i - 1] == '~') => {}
            '_' => {
                let prev = i.checked_sub(1).map(|k| chars[k]);
                let next = chars.get(i + 1).copied();
                if prev.is_some_and(char::is_alphanumeric)
                    && next.is_some_and(char::is_alphanumeric)
                {
                    out.push(' ');
                }
            }
            _ => out.push(c),
        }
    }
    out
}

fn is_emoji(c: char) -> bool {
    matches!(c as u32,
        0x1F000..=0x1FAFF
        | 0x2600..=0x27BF
        | 0x2B00..=0x2BFF
        | 0x2190..=0x21FF
        | 0x2300..=0x23FF
        | 0x25A0..=0x25FF
        | 0xFE00..=0xFE0F
        | 0xE0020..=0xE007F
        | 0x20E3)
}

/// Drop emoji and the zero-width joiners that glue them; a ZWNJ (used in
/// Persian words) is kept.
fn drop_emoji(line: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut out = String::with_capacity(line.len());
    for (i, &c) in chars.iter().enumerate() {
        if is_emoji(c) {
            continue;
        }
        if c == '\u{200D}' {
            let near_emoji = i.checked_sub(1).is_some_and(|k| is_emoji(chars[k]))
                || chars.get(i + 1).is_some_and(|&n| is_emoji(n));
            if near_emoji {
                continue;
            }
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split_all(deltas: &[&str]) -> Vec<String> {
        let mut s = SentenceSplitter::new();
        let mut out = Vec::new();
        for d in deltas {
            out.extend(s.push(d));
        }
        out.extend(s.flush());
        out
    }

    fn split_chars(text: &str) -> Vec<String> {
        let mut s = SentenceSplitter::new();
        let mut out = Vec::new();
        for c in text.chars() {
            out.extend(s.push(&c.to_string()));
        }
        out.extend(s.flush());
        out
    }

    #[test]
    fn persian_sentences() {
        let text = "سلام دوست من! حالت چطوره؟ امروز هوا خوبه؛ بریم بیرون. باشه…";
        let expected = vec![
            "سلام دوست من!",
            "حالت چطوره؟",
            "امروز هوا خوبه؛",
            "بریم بیرون.",
            "باشه…",
        ];
        assert_eq!(split_all(&[text]), expected);
        assert_eq!(split_chars(text), expected);
    }

    #[test]
    fn streaming_emits_when_complete() {
        let mut s = SentenceSplitter::new();
        assert!(s.push("این یک جمله").is_empty());
        assert!(
            s.push(" است.").is_empty(),
            "the dot is undecided until the next char"
        );
        assert_eq!(s.push(" و این"), vec!["این یک جمله است."]);
        assert_eq!(s.flush().as_deref(), Some("و این"));
        assert_eq!(s.flush(), None);
    }

    #[test]
    fn decimals_are_not_split() {
        let text = "قیمت 2.5 دلار است. نسخه ۳.۱۴ آمد. Version 1.2.3 is out. Done.";
        assert_eq!(
            split_chars(text),
            vec![
                "قیمت 2.5 دلار است.",
                "نسخه ۳.۱۴ آمد.",
                "Version 1.2.3 is out.",
                "Done."
            ]
        );
    }

    #[test]
    fn abbreviations_are_not_split() {
        let text = "Use a tool, e.g. a hammer. Ask Dr. Smith or J. Doe. See i.e. this. فایل main.rs را ببین.";
        assert_eq!(
            split_chars(text),
            vec![
                "Use a tool, e.g. a hammer.",
                "Ask Dr. Smith or J. Doe.",
                "See i.e. this.",
                "فایل main.rs را ببین."
            ]
        );
    }

    #[test]
    fn mixed_and_question_marks() {
        let text = "مدل qwen3.8 خوبه؟! Really? Yes. «واقعاً؟» گفت.";
        assert_eq!(
            split_chars(text),
            vec!["مدل qwen3.8 خوبه؟!", "Really?", "Yes.", "«واقعاً؟»", "گفت."]
        );
    }

    #[test]
    fn short_sentences_merge() {
        // "نه." has two letters: merged forward.
        assert_eq!(split_chars("نه. من نمیام."), vec!["نه. من نمیام."]);
        assert_eq!(
            split_all(&["خب! "]),
            vec!["خب!"],
            "flush emits a held piece"
        );
        assert_eq!(split_chars("Ok. Go."), vec!["Ok. Go."]);
    }

    #[test]
    fn newlines_and_lists() {
        let text = "عنوان\n\n1. اولین مورد است\n2. دومین مورد\n- گزینه سوم\nپایان";
        assert_eq!(
            split_chars(text),
            vec![
                "عنوان",
                "1. اولین مورد است",
                "2. دومین مورد",
                "- گزینه سوم",
                "پایان"
            ]
        );
    }

    #[test]
    fn code_blocks_stay_whole() {
        let text =
            "این کد است:\n```rust\nfn main() {\n    println!(\"hi. there\");\n}\n```\nتمام شد.";
        let out = split_chars(text);
        assert_eq!(out.len(), 3, "{out:?}");
        assert_eq!(out[0], "این کد است:");
        assert!(
            out[1].starts_with("```rust") && out[1].ends_with("```"),
            "{out:?}"
        );
        assert_eq!(out[2], "تمام شد.");
        assert_eq!(speakable(&out[1]), None);
    }

    #[test]
    fn inline_code_does_not_split() {
        assert_eq!(
            split_chars("Run `a. b` now. بعد"),
            vec!["Run `a. b` now.", "بعد"]
        );
    }

    #[test]
    fn long_runs_are_cut() {
        let clause = "این یک بند نسبتاً طولانی بدون نقطه است، ";
        let text = clause.repeat(12);
        let out = split_chars(&text);
        assert!(out.len() >= 2, "{out:?}");
        for piece in &out {
            assert!(piece.chars().count() <= LONG_RUN + 5, "{piece}");
        }
        assert_eq!(out.join(" "), text.trim());
        for piece in &out[..out.len() - 1] {
            assert!(piece.ends_with('،'), "cut at a comma: {piece}");
        }
        // No commas: cut at a space.
        let words = "کلمه ".repeat(80);
        let out = split_chars(&words);
        assert!(out.len() >= 2);
        assert!(out.iter().all(|p| !p.contains("کلمهکلمه")));
        assert_eq!(out.join(" "), words.trim());
    }

    #[test]
    fn nothing_is_lost() {
        let text = "Hello world. سلام دنیا! چطوری؟ I'm fine… thanks; بله؛ نه\n\nOK";
        let out = split_chars(text);
        let joined: String = out.concat().split_whitespace().collect();
        let orig: String = text.split_whitespace().collect();
        assert_eq!(joined, orig);
    }

    #[test]
    fn speakable_strips_markdown() {
        assert_eq!(speakable("## **سلام** _دنیا_").as_deref(), Some("سلام دنیا"));
        assert_eq!(
            speakable("> - یک [لینک خوب](https://example.com/a.b) اینجا").as_deref(),
            Some("یک لینک خوب اینجا")
        );
        assert_eq!(
            speakable("ببین https://example.com/x و www.foo.org را").as_deref(),
            Some("ببین و را")
        );
        assert_eq!(
            speakable("دستور `cargo build` را اجرا کن").as_deref(),
            Some("دستور cargo build را اجرا کن")
        );
        assert_eq!(
            speakable("کد `let x = some_very_long_function_name(arg1, arg2);` را ببین").as_deref(),
            Some("کد را ببین")
        );
        assert_eq!(speakable("1. اولین مورد").as_deref(), Some("اولین مورد"));
        assert_eq!(speakable("۲) دومین").as_deref(), Some("دومین"));
        assert_eq!(
            speakable("متغیر snake_case خوبه").as_deref(),
            Some("متغیر snake case خوبه")
        );
        assert_eq!(speakable("~~غلط~~ درست").as_deref(), Some("غلط درست"));
        assert_eq!(speakable("![تصویر](a.png)").as_deref(), Some("تصویر"));
    }

    #[test]
    fn speakable_drops_tables_code_emoji() {
        assert_eq!(speakable("| a | b |"), None);
        assert_eq!(speakable("---|---"), None);
        assert_eq!(speakable("a | b | c"), None);
        assert_eq!(speakable("```\ncode here\n```"), None);
        assert_eq!(speakable("```py\nunclosed"), None);
        assert_eq!(
            speakable("قبل\n```\nx = 1\n```\nبعد").as_deref(),
            Some("قبل بعد")
        );
        assert_eq!(
            speakable("عالیه 😀👍🏽 ✅ مرسی ❤️").as_deref(),
            Some("عالیه مرسی")
        );
        assert_eq!(speakable("👨‍👩‍👧"), None);
        // ZWNJ in Persian words is kept.
        assert_eq!(speakable("می‌خواهم").as_deref(), Some("می‌خواهم"));
        assert_eq!(speakable("   \n\n  "), None);
        assert_eq!(speakable("---"), None);
        assert_eq!(speakable("**۱۲۳**"), None);
        assert_eq!(speakable("#"), None);
    }

    #[test]
    fn speakable_collapses_whitespace() {
        assert_eq!(
            speakable("  سلام   \n\n  دنیا\tخوب  ").as_deref(),
            Some("سلام دنیا خوب")
        );
        assert_eq!(speakable("C# is fine").as_deref(), Some("C# is fine"));
        assert_eq!(speakable("#hashtag").as_deref(), Some("#hashtag"));
    }
}
