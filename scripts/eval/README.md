# Evaluation Scripts

| Script | Purpose |
|--------|---------|
| `eval_comprehensive.py` | 39-test quality suite (math, code, reasoning, vision, parallel) |
| `warmup.py` | Server warmup utility |

## Quality Evaluation

```bash
python scripts/eval/eval_comprehensive.py --port 23334 --parallel 4 --thinking-budget 512
```

Designed to catch TP=2 precision errors:
- Off-by-one arithmetic (389 vs 391)
- Garbled code (`s[::-]` instead of `s[::-1]`)
- Wrong imports
- Vision/multimodal regressions

Run after kernel changes or patch updates to verify model quality.
