# FAQ Evaluation Report (OpenFOAM, merged from IndustrialFAQ-RAG)

Offline, deterministic, no LLM calls. Run against the shipped
`src/agentic_rag/faq_data/faq.tsv`: 50 entries, 100 aliases (2 per entry).

## Query sets

| Set | n | Construction |
|---|---:|---|
| Standard questions | 50 | TSV standard question, verbatim |
| Aliases | 100 | TSV similar questions, verbatim |
| Perturbations | 150 | Per entry: strip leading "如何", prefix "请问", or add "？" |
| Negative (unrelated) | 20 | Everyday non-OpenFOAM questions, e.g. weather, cooking, guitar |

Exact standard/alias queries resolve through the exact sentence-to-entry
map, so they are 100% at every threshold; the informative sets are
perturbations (robustness) and negatives (false positives).

## Threshold sweep

Hit rate / correct-entry rate. For negatives, "correct" means no match.

| Threshold | Standard | Alias | Perturbation | Negative (no-match) |
|---:|---:|---:|---:|---:|
| 0.5 | 1.000 | 1.000 | 1.000 | 0.850 (3 false positives) |
| 0.6 | 1.000 | 1.000 | 1.000 | 0.950 (1 false positive) |
| 0.7 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.8 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.9 | 1.000 | 1.000 | 0.480 | 1.000 |

The single false positive at 0.6 was "怎么学吉他" matching
"如何设置声学模拟" at score 0.60, via the alias "怎么设声学".
At 0.7 it disappears, and perturbations are still 100%.
At 0.9, perturbations collapse to 48%, so 0.9 is too strict.

## Decision

Default threshold changed from 0.6 to 0.7 in
`src/agentic_rag/faq_match.py` (`match_faq`, `text_match`) and in
`src/agentic_rag/main.py` (`--faq-threshold`). Callers who need the old
behaviour can still pass `--faq-threshold 0.6` explicitly.

## Limitations

- This is a synthetic probe, not a real-user query log. Perturbations
  are mechanical, and the 20 negatives are hand-picked.
- Coverage is only the 50 shipped entries; new OpenFOAM topics added
  later need their own standard questions and aliases in this eval.
- `tests/test_faq_eval.py` pins the default-threshold outcome:
  150 exact + 150 perturbation queries correct, 20 negatives unmatched.

## Reproduce

```bash
pytest tests/test_faq_eval.py -q
```
