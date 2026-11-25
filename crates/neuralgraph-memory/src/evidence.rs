//! Evidence passages: dated, verbatim grounding for the context block (opt-in, `context_evidence`).
//!
//! Episodic recall (`append_raw_messages`) adds whole human turns that match the question. On LongMemEval Oracle seed 1
//! (Qwen3.5-4B extractor) the answer was often in a sentence the extractor never turned into a memory, or in an
//! assistant reply (single-session-assistant questions), or updated later in the chat; whole-turn matching then shows the
//! wrong part, or none, and the reader answers from a stale memory. This mode instead:
//!
//! 1. splits every message in scope, user and assistant, into sentence windows of at most `context_passage_chars`;
//! 2. ranks the windows by reciprocal-rank fusion of passage-level BM25 on the question and provenance: the best window
//!    of each message that sourced one of the block's memories gets `1 / (rrf_k + memory rank)` on top;
//! 3. keeps the memory lines (and their quotes) up to `context_memory_chars`, then fills the rest of the
//!    `context_max_chars` budget with the best windows, rendered oldest first with their date and role under
//!    [`EVIDENCE_HEADER`], so a later value reads as the current one.
//!
//! No model calls and nothing at ingest (about 3 ms per question). On LongMemEval Oracle, 150 questions (seeds 1-3),
//! Gemma-4-e4b with the official judge prompts, NeuralGraph's routed reader on the same questions with and without this
//! mode: Qwen3.5-4B 62.0 -> 71.3 official (19:5, McNemar p = 0.007), gemini-3.5-flash-lite 65.3 -> 73.3 (20:8,
//! p = 0.036); positive on the held-out seeds 2-3 at both readers, no question type down by more than one question.
//! See `research/reports/E1_evidence_passages.md`.
//!
//! The text functions follow the Python prototype's semantics exactly (code-point lengths, `str.strip`, the split
//! regex `(?<=[.!?])\s+|\n+`); on the seed 1-3 databases this module reproduces the prototype's 150 blocks byte for byte.

use crate::pycompat::{is_py_space, py_len, py_prefix, py_rstrip, py_slice, py_strip};
use std::collections::HashMap;

pub const EVIDENCE_HEADER: &str =
    "Verbatim excerpts from past conversations (oldest first; when excerpts disagree, the latest one is current):";

const RRF_K: f64 = 60.0;
const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;

/// English function words plus the contractions left after apostrophes are dropped ("I've" -> "ive").
#[rustfmt::skip]
const STOPWORDS: &[&str] = &[
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be",
    "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "could", "did", "do", "does",
    "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself", "yourselves", "ive", "im", "id", "ill",
    "dont", "didnt", "doesnt",
];

/// Light suffix stripping, enough to match follower/followers and attend/attended/attending.
fn stem(t: &str) -> String {
    for (suf, n) in [("ies", 3), ("ing", 3), ("ed", 2), ("es", 2), ("s", 1)] {
        if t.len() > n + 3 && t.ends_with(suf) {
            return format!("{}{}", &t[..t.len() - n], if suf == "ies" { "y" } else { "" });
        }
    }
    t.to_string()
}

/// Lowercased ASCII alphanumeric runs (apostrophes dropped first), stopwords removed, stemmed.
pub fn tokens(text: &str) -> Vec<String> {
    let lower = text.to_lowercase().replace('\'', "");
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in lower.chars().chain(std::iter::once(' ')) {
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            cur.push(c);
        } else if !cur.is_empty() {
            if !STOPWORDS.contains(&cur.as_str()) {
                out.push(stem(&cur));
            }
            cur.clear();
        }
    }
    out
}

/// Python's `re.split(r"(?<=[.!?])\s+|\n+", text)`: a whitespace run right after sentence punctuation, or a newline run.
fn split_sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut prev: Option<char> = None;
    let mut it = text.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        let after_punct = matches!(prev, Some('.' | '!' | '?'));
        let run: Option<fn(char) -> bool> = if after_punct && is_py_space(c) {
            Some(is_py_space)
        } else if c == '\n' {
            Some(|x| x == '\n')
        } else {
            None
        };
        match run {
            Some(member) => {
                out.push(&text[start..i]);
                let mut last = c;
                while let Some(&(_, n)) = it.peek() {
                    if !member(n) {
                        break;
                    }
                    last = n;
                    it.next();
                }
                start = it.peek().map(|&(j, _)| j).unwrap_or(text.len());
                prev = Some(last);
            }
            None => prev = Some(c),
        }
    }
    out.push(&text[start..]);
    out
}

/// Greedy sentence windows of at most `limit` code points; an over-long sentence is hard-split.
pub fn passages(text: &str, limit: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for piece in split_sentences(text) {
        let mut s = py_strip(piece).to_string();
        if s.is_empty() {
            continue;
        }
        while py_len(&s) > limit {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            out.push(py_prefix(&s, limit).to_string());
            s = py_slice(&s, limit, None).to_string();
        }
        if !cur.is_empty() && py_len(&cur) + 1 + py_len(&s) > limit {
            out.push(std::mem::replace(&mut cur, s));
        } else {
            cur = py_strip(&format!("{cur} {s}")).to_string();
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Okapi BM25 of each document against the distinct query terms (natural-log idf with the +1 floor).
pub fn bm25(docs: &[Vec<String>], query: &[String]) -> Vec<f64> {
    let n = docs.len() as f64;
    let total: usize = docs.iter().map(Vec::len).sum();
    let avg = if docs.is_empty() || total == 0 { 1.0 } else { total as f64 / docs.len() as f64 };
    let mut df: HashMap<&str, usize> = HashMap::new();
    for d in docs {
        let mut seen: Vec<&str> = d.iter().map(String::as_str).collect();
        seen.sort_unstable();
        seen.dedup();
        for t in seen {
            *df.entry(t).or_default() += 1;
        }
    }
    let mut q: Vec<&str> = Vec::new();
    for t in query {
        if !q.contains(&t.as_str()) {
            q.push(t);
        }
    }
    docs.iter()
        .map(|d| {
            let mut tf: HashMap<&str, usize> = HashMap::new();
            for t in d {
                *tf.entry(t.as_str()).or_default() += 1;
            }
            let mut s = 0.0;
            for t in &q {
                let f = tf.get(t).copied().unwrap_or(0) as f64;
                if f > 0.0 {
                    let dft = df[t] as f64;
                    let idf = (1.0 + (n - dft + 0.5) / (dft + 0.5)).ln();
                    s += idf * f * (BM25_K1 + 1.0) / (f + BM25_K1 * (1.0 - BM25_B + BM25_B * d.len() as f64 / avg));
                }
            }
            s
        })
        .collect()
}

/// "2023-05-20 (Sat)" from an ISO timestamp; the raw value (or "undated") when it does not parse.
pub fn day_label(iso: &str) -> String {
    let head = py_prefix(iso, 19);
    let date = chrono::NaiveDateTime::parse_from_str(head, "%Y-%m-%dT%H:%M:%S")
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(head, "%Y-%m-%d %H:%M:%S"))
        .map(|t| t.date())
        .or_else(|_| chrono::NaiveDate::parse_from_str(head, "%Y-%m-%d"));
    match date {
        Ok(d) => d.format("%Y-%m-%d (%a)").to_string(),
        Err(_) if iso.is_empty() => "undated".into(),
        Err(_) => iso.to_string(),
    }
}

/// One message in scope, in the store's chronological order.
pub struct EvidenceMessage {
    pub message_id: String,
    pub role: String,
    pub sent_at: String,
    pub text: String,
}

struct Passage {
    key: (usize, usize),
    message: usize,
    text: String,
}

fn render(m: &EvidenceMessage, text: &str) -> String {
    format!("- [{}, {}] \"{text}\"", day_label(&m.sent_at), m.role)
}

/// The rendered passage lines that fit in `budget` characters (each line costs its length plus a newline), oldest
/// first. `provenance` maps a message id to the best rank of a block memory it sourced.
pub fn select(
    messages: &[EvidenceMessage],
    question: &str,
    provenance: &HashMap<String, usize>,
    passage_chars: usize,
    prov_weight: f64,
    budget: isize,
) -> Vec<String> {
    let mut ps = Vec::new();
    for (order, m) in messages.iter().enumerate() {
        for (i, text) in passages(&m.text, passage_chars).into_iter().enumerate() {
            ps.push(Passage { key: (order, i), message: order, text });
        }
    }
    if ps.is_empty() {
        return Vec::new();
    }
    let docs: Vec<Vec<String>> = ps.iter().map(|p| tokens(&p.text)).collect();
    let s = bm25(&docs, &tokens(question));
    let mut by_bm25: Vec<usize> = (0..ps.len()).collect();
    by_bm25.sort_by(|&a, &b| s[b].total_cmp(&s[a]).then(ps[a].key.cmp(&ps[b].key)));
    let mut fused: HashMap<usize, f64> = HashMap::new();
    for (r, &i) in by_bm25.iter().enumerate() {
        if s[i] > 0.0 {
            fused.insert(i, 1.0 / (RRF_K + r as f64));
        }
    }
    // only the best window of a sourcing message gets the provenance share, so a long reply cannot flood the block
    let mut best_in_msg: HashMap<usize, usize> = HashMap::new();
    for &i in &by_bm25 {
        best_in_msg.entry(ps[i].message).or_insert(i);
    }
    for (order, m) in messages.iter().enumerate() {
        if let (Some(&rank), Some(&i)) = (provenance.get(&m.message_id), best_in_msg.get(&order)) {
            *fused.entry(i).or_insert(0.0) += prov_weight / (RRF_K + rank as f64);
        }
    }
    let mut ranked: Vec<usize> = fused.keys().copied().collect();
    ranked.sort_by(|&a, &b| fused[&b].total_cmp(&fused[&a]).then(ps[a].key.cmp(&ps[b].key)));
    let mut chosen = Vec::new();
    let mut used: isize = 0;
    for i in ranked {
        let cost = py_len(&render(&messages[ps[i].message], &ps[i].text)) as isize + 1;
        if used + cost > budget {
            continue; // a shorter, lower-ranked window may still fit
        }
        used += cost;
        chosen.push(i);
    }
    chosen.sort_by_key(|&i| ps[i].key);
    chosen.into_iter().map(|i| render(&messages[ps[i].message], &ps[i].text)).collect()
}

/// The memory part of the block cut to `cap` characters at a line boundary (header always kept; nothing when no
/// memory line fits), as the prototype does on the crate's own rendering.
pub fn cap_memory_block(block: &str, cap: usize) -> String {
    let block = py_rstrip(block);
    if block.is_empty() {
        return String::new();
    }
    let mut lines = block.split('\n');
    let head = lines.next().unwrap_or_default();
    let mut kept = vec![head];
    let mut used = py_len(head);
    for l in lines {
        if used + 1 + py_len(l) > cap {
            break;
        }
        used += 1 + py_len(l);
        kept.push(l);
    }
    if kept.len() > 1 {
        kept.join("\n")
    } else {
        String::new()
    }
}

/// Memory block plus evidence block under one `limit`.
pub fn compose(memory_block: &str, evidence: &[String]) -> String {
    let mut parts = Vec::new();
    if !memory_block.is_empty() {
        parts.push(memory_block.to_string());
    }
    if !evidence.is_empty() {
        parts.push(format!("{EVIDENCE_HEADER}\n{}", evidence.join("\n")));
    }
    parts.join("\n")
}

/// Characters left for evidence lines once the memory block, the header and the two joining newlines are paid for.
pub fn evidence_budget(limit: usize, memory_block: &str) -> isize {
    limit as isize - py_len(memory_block) as isize - py_len(EVIDENCE_HEADER) as isize - 2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(id: &str, role: &str, at: &str, text: &str) -> EvidenceMessage {
        EvidenceMessage { message_id: id.into(), role: role.into(), sent_at: at.into(), text: text.into() }
    }

    #[test]
    fn tokens_drop_stopwords_apostrophes_and_stem() {
        assert_eq!(tokens("I've got 1,300 Followers now!"), vec!["got", "1", "300", "follower"]);
        assert_eq!(tokens("attending the attended studies"), vec!["attend", "attend", "study"]);
    }

    #[test]
    fn split_matches_python_regex() {
        // re.split(r"(?<=[.!?])\s+|\n+", ...) in Python 3.12
        assert_eq!(split_sentences("A b. C d!  E\n\nF? G"), vec!["A b.", "C d!", "E", "F?", "G"]);
        assert_eq!(split_sentences("x \ny"), vec!["x ", "y"]);
        assert_eq!(split_sentences("end. \n next"), vec!["end.", "next"]);
        assert_eq!(split_sentences("no split"), vec!["no split"]);
        assert_eq!(split_sentences("trail.\n"), vec!["trail.", ""]);
    }

    #[test]
    fn passages_pack_and_hard_split() {
        assert_eq!(passages("One. Two. Three.", 10), vec!["One. Two.", "Three."]);
        assert_eq!(passages("abcdefghij klm", 5), vec!["abcde", "fghij", "klm"]);
        assert_eq!(passages("", 5), Vec::<String>::new());
    }

    #[test]
    fn day_label_formats_or_falls_back() {
        assert_eq!(day_label("2023-05-20T10:00:00"), "2023-05-20 (Sat)");
        assert_eq!(day_label("2023-05-20"), "2023-05-20 (Sat)");
        assert_eq!(day_label(""), "undated");
        assert_eq!(day_label("someday"), "someday");
    }

    #[test]
    fn latest_value_is_kept_and_order_is_chronological() {
        let ms = vec![
            msg("m1", "user", "2023-05-01T09:00:00", "I have 1250 followers on Instagram."),
            msg("m2", "assistant", "2023-05-01T09:01:00", "Great, keep posting regularly to grow."),
            msg("m3", "user", "2023-05-20T09:00:00", "Update: I now have around 1300 followers on Instagram."),
        ];
        let lines = select(&ms, "How many followers do I have on Instagram now?", &HashMap::new(), 320, 1.0, 1000);
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("- [2023-05-01 (Mon), user]") && lines[1].contains("1300"));
    }

    #[test]
    fn provenance_rescues_a_message_with_no_word_overlap() {
        let ms =
            vec![msg("m1", "user", "2023-05-01T09:00:00", "Bought a red kettle."), msg("m2", "user", "", "Zebra.")];
        let none = select(&ms, "What colour?", &HashMap::new(), 320, 1.0, 1000);
        assert!(none.is_empty());
        let prov = HashMap::from([("m1".to_string(), 0)]);
        assert_eq!(
            select(&ms, "What colour?", &prov, 320, 1.0, 1000),
            vec!["- [2023-05-01 (Mon), user] \"Bought a red kettle.\""]
        );
    }

    #[test]
    fn budget_is_respected_and_memory_block_cut_at_lines() {
        let ms = vec![msg("m1", "user", "", "alpha beta gamma"), msg("m2", "user", "", "alpha beta")];
        let lines = select(&ms, "alpha beta", &HashMap::new(), 320, 1.0, 30);
        assert!(lines.iter().map(|l| py_len(l) + 1).sum::<usize>() <= 30);
        assert_eq!(cap_memory_block("Head\n- one\n- two", 10), "Head\n- one");
        assert_eq!(cap_memory_block("Head\n- one", 4), "");
        assert_eq!(compose("", &["- x".into()]), format!("{EVIDENCE_HEADER}\n- x"));
    }
}
