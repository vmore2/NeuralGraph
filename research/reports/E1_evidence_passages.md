# E1: evidence passages in the context block

**Verdict: WORKS at both readers tested.** Replacing episodic recall with dated, verbatim sentence-window passages
(`NEURALGRAPH_EXP=context_evidence,context_memory_chars=2400`) raises LongMemEval Oracle accuracy on the official judge by
+9.3 points with a local Qwen3.5-4B reader (19:5, McNemar p = 0.007) and by +8.0 with gemini-3.5-flash-lite (20:8,
p = 0.036) over 150 questions. It is positive on the held-out draws at both readers, no question type loses more than
one question, it adds no model call, and it costs about 3 ms per question. Off by default; with the switch off the
context block is byte-identical to `main`.

## Why
On seed 1 the retriever finds the answer-bearing session almost every time (session@k 97.8), yet the 4B reader missed 21
of 50 questions. Reading the misses: the answer sat in a sentence the extractor never turned into a memory, in an
**assistant** reply (single-session-assistant questions ask what the assistant said), or in a later message that
updated an earlier fact. Episodic recall (`append_raw_messages`) adds whole *user* turns that match the question, so it
shows the wrong part of a long turn, never the assistant's side, and nothing that says which value is current.

## What the mode does (`crates/neuralgraph-memory/src/evidence.rs`)
1. Split every message in scope, user and assistant, into sentence windows of at most `context_passage_chars` (320).
2. Score windows by reciprocal-rank fusion (k = 60) of passage-level BM25 on the question (k1 1.2, b 0.75, stopwords,
   light stemming) and provenance: the best window of each message that sourced one of the block's memories gets
   `1 / (60 + memory rank)`.
3. Keep the memory lines and quotes up to `context_memory_chars` (2400), then fill the rest of `context_max_chars` (6000)
   with the best windows, rendered **oldest first** with date and role under a header saying the latest excerpt is current.

Nothing changes at ingest and no model is called. Past `context_evidence_scan` (5000) messages in scope, only FTS
candidates plus sourcing messages are scored, so the cost stays bounded on a large store.

## Setup
- **Harness:** `chat-memory-longmemeval` ingest (deterministic, `--concurrency 1 --parallel 1`), `chat-memory-context lme`,
  `answer_eval.py` with the shipped routed flags (`--prompt concise --cascade auto_v1n --route-compute auto_v1n
  --route-compute-think --extra-judges --official-judge --threads 1`), `compare.py`. No harness file is modified.
- **Questions:** LongMemEval Oracle, the three stratified 50-question draws of the headline (seeds 1, 2, 3; 150 questions).
  Seed 1 was the development draw (the mode and its two settings were chosen on it); seeds 2 and 3 were held out and
  run once with the frozen settings.
- **Arms:** `routed` = the shipped crate-default block (quotes 1, 6000 chars, 3 raw user turns), i.e. NeuralGraph as on
  `main`; `evidence` = the same databases and question set with the switch on. Same reader, judge and flags in both arms.
- **Models (llama.cpp, one RTX 3050 6 GB laptop):** extractor and local reader `qwen3.5-4b` Q4_K_M, judge
  `gemma-4-e4b` Q4_K_M with the official LongMemEval prompts, embeddings `nomic-embed-text-v1.5` f16.
- **Second reader:** `gemini-3.5-flash-lite` reading the *same* context files (memories still extracted by the 4B model).
  Calls went through a local OpenAI-compatible adapter that drops the empty-think prefill, maps `reasoning_effort` to
  `minimal` (concise) / `low` (thinking calls), and raises `max_tokens` to 4096 because Gemini counts thinking tokens
  against it. The judge stayed the local Gemma.
- **Decision rule, fixed before any held-out result:** the claim stands only if (1) the held-out delta is positive,
  (2) the pooled 150-question McNemar p < 0.05 with the bootstrap 95% CI of the delta above 0, (3) no question type has a
  net loss of more than one question, (4) the sign is the same at both readers. All four hold at both readers.

## Results (official judge, % of 50 per draw)
| reader | draw | routed (main) | + evidence | wins:losses | strict_v2 | lenient_nonidk |
|---|---|---:|---:|---:|---|---|
| Qwen3.5-4B | seed 1 (dev) | 58 | 66 | 6:2 | 60 -> 74 (7:0) | 66 -> 78 |
| Qwen3.5-4B | seed 2 | 60 | 70 | 7:2 | 64 -> 74 (5:0) | 66 -> 74 |
| Qwen3.5-4B | seed 3 | 68 | 78 | 6:1 | 74 -> 84 (6:1) | 78 -> 86 |
| gemini-3.5-flash-lite | seed 1 (dev) | 66 | 76 | 7:2 | 64 -> 80 (8:0) | 74 -> 86 |
| gemini-3.5-flash-lite | seed 2 | 62 | 72 | 9:4 | 62 -> 74 (8:2) | 66 -> 76 |
| gemini-3.5-flash-lite | seed 3 | 68 | 72 | 4:2 | 76 -> 82 (4:1) | 80 -> 82 |

| reader | scope | routed | + evidence | delta | W:L | McNemar p | bootstrap 95% CI | strict_v2 delta |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Qwen3.5-4B | held-out 2-3 (100) | 64.0 | 74.0 | +10.0 | 13:3 | 0.021 | [+3.0, +18.0] | +10.0 |
| Qwen3.5-4B | seeds 1-3 (150) | 62.0 | 71.3 | +9.3 | 19:5 | 0.007 | [+3.3, +15.3] | +11.3 |
| gemini-3.5-flash-lite | held-out 2-3 (100) | 65.0 | 72.0 | +7.0 | 13:6 | 0.17 | [-1.0, +15.0] | +9.0 |
| gemini-3.5-flash-lite | seeds 1-3 (150) | 65.3 | 73.3 | +8.0 | 20:8 | 0.036 | [+1.3, +14.7] | +11.3 |

By question type, seeds 1-3 (official, % correct; net questions in brackets):
| type | n | Qwen3.5-4B | gemini-3.5-flash-lite |
|---|---:|---|---|
| single-session-assistant | 18 | 38.9 -> 83.3 (+8) | 38.9 -> 88.9 (+9) |
| knowledge-update | 22 | 72.7 -> 90.9 (+4) | 77.3 -> 90.9 (+3) |
| single-session-preference | 9 | 22.2 -> 55.6 (+3) | 11.1 -> 22.2 (+1) |
| temporal-reasoning | 35 | 77.1 -> 74.3 (-1) | 77.1 -> 80.0 (+1) |
| multi-session | 35 | 62.9 -> 62.9 (0) | 80.0 -> 77.1 (-1) |
| single-session-user | 19 | 100 -> 100 (0) | 94.7 -> 89.5 (-1) |
| abstention | 12 | 0 -> 0 (0) | 0 -> 0 (0) |

The gain is where the diagnosis said it would be: assistant-side facts and updated facts. "I don't know" answers fall
from 33 to 18 (4B) and 29 to 18 (Gemini) of 150.

## Cost
- **Latency:** context build median 82 -> 85 ms per question on the seed 1 databases (7 alternating repetitions, query
  embedding included); the evidence step is about 3 ms. No extra model call. Mean answer time per question, which reader
  generation dominates: 4B 66.8 -> 64.5 s, Gemini 21.4 -> 25.5 s (the longer context).
- **Context:** 3,834 -> 5,677 characters mean (about +550 tokens), still inside the existing 6000-character cap.

## Checks
- With the switch off, `chat-memory-context lme` output is byte-identical to `main` on all 150 questions.
- The Rust module reproduces the Python prototype that produced the measured contexts byte for byte on all 150 questions
  (seeds 1, 2, 3: 50/50 each), so every number above is the code in this PR.
- `cargo fmt --check`, `cargo clippy -p neuralgraph-memory --all-targets -- -D warnings` and `cargo test -p
  neuralgraph-memory` pass (on Windows `tests/update.rs` fails its two macOS/Linux release-target tests, as on `main`).

## What this does not show
- **Absolute level.** These runs use a 4B extractor and a 4B or flash-lite reader on one laptop GPU, so the routed
  baselines (62.0 and 65.3) sit below the posted 9B result (74 / 62 / 72, mean 69). The claim is the matched delta, not
  the absolute score. Both arms always shared databases, questions, reader, judge and flags.
- **A stronger stack.** The mode fixed about a quarter of the routed arm's misses at both readers (23% Gemini, 25% 4B,
  95% CI 5-39% and 9-38%). Carried over to the posted configurations, that would give about 76-77 for the 9B mean
  (69.3) and about 81 for the 20-question Sonnet run (75). These are projections, not runs.
- **LongMemEval S and LoCoMo** were not run.

## Tried and not included
- **A query-time ledger** (one extraction call per question, citation-checked rows, symbolic dates/counts/latest values)
  on top of the evidence block: larger mean gain at Gemini (65.3 -> 76.0, 28:12) but a net loss of 4 multi-session
  questions, which fails the no-regression rule above. Re-extracting the ledger with Gemini instead of the 4B model
  was worse on a 5-question pilot (2/5 vs 4/5).
- **Budget forcing** of capped thinking calls (closing the trace with a forced `Answer:` line instead of discarding it):
  routed 58 -> 60 on seed 1, within noise.

## Reproduce
Per draw, after the ingest and default dump of `research/overnight/README.md` sec. 3:
```bash
NEURALGRAPH_EXP=context_evidence,context_memory_chars=2400 \
  ./target/release/chat-memory-context lme --db-dir $D --n 50 --seed $SEED --out $R/lme_seed${SEED}_evidence_ctx.json
python3 $A --contexts $R/lme_seed${SEED}_evidence_ctx.json --out $R/lme_seed${SEED}_evidence_acc.json \
  --prompt concise --cascade auto_v1n --route-compute auto_v1n --route-compute-think \
  --extra-judges --official-judge --threads 1 --label "seed$SEED evidence"
python3 $C $R/lme_seed${SEED}_routed_acc.json $R/lme_seed${SEED}_evidence_acc.json --md \
  --metric strict,strict_v2,lenient_nonidk,official
```
Per-question rows for all twelve answer files (two readers x three draws x two arms) are in
[`research/results/evidence_passages/`](../results/evidence_passages/); `compare.py` on any routed/evidence pair
reproduces the per-draw rows above, and the pooled rows are the three draws concatenated.
