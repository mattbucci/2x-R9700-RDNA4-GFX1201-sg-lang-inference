# llmcompressor patches

Vendored copy of [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) pinned to commit `30845208` (v0.10.1.dev92), with our patches applied on top.

Same workflow as the top-level `patches/` dir for SGLang: each `NNN-*.patch` is a `git diff` against the pinned commit, applied in numeric order during setup.

Clone lives at `components/llmcompressor/` (gitignored, same as `components/sglang/`).

## Patches

| # | Title | Why |
|---|-------|-----|
| 001 | qwen3-moe-unfuse-fused-experts | transformers ≥5 restructured Qwen3MoE: the gate/router returns `(logits, scores, indices)` tuple, and experts are fused 3D `nn.Parameter` tensors instead of a `ModuleList`. Rewrite `CalibrationQwen3MoeSparseMoeBlock` to handle the new router, and add a `SequentialQwen3MoeExperts` (registered for `Qwen3MoeExperts`) that unfuses into per-expert `nn.Linear` modules so GPTQ's `targets="Linear"` can calibrate each expert — same pattern as `SequentialGemma4TextExperts`. Unblocks Qwen3-Coder-30B-A3B calibration. |

## Apply

```bash
# From repo root
cd components/llmcompressor
for p in ../../llmcompressor-patches/*.patch; do git apply "$p"; done
pip install -e .
```

## Add a new patch

1. Edit the file under `components/llmcompressor/src/llmcompressor/`.
2. `cd components/llmcompressor && git diff src/llmcompressor/<file> > ../../llmcompressor-patches/NNN-short-name.patch`
3. Document it in the table above.
4. Validate by re-running the failing calibration end-to-end.
