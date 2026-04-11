# RDNA4 Inference Project

Custom SGLang v0.5.10 + RDNA4 patches for 2x AMD Radeon AI PRO R9700.

## Documentation

| File | Purpose |
|------|---------|
| [README.md](README.md) | Setup, benchmarks, model support, architecture |
| [rules-for-agents.md](rules-for-agents.md) | RDNA4 constraints, launch flags, model status |
| [docs/known_issues.md](docs/known_issues.md) | Active issues only |
| [patches/README.md](patches/README.md) | Patch descriptions and apply order |

## Key Commands
```bash
scripts/setup.sh                     # Full setup
scripts/setup_sgl_kernel.sh --env X  # Native sgl_kernel (required)
scripts/build_awq_gemv.sh --env X    # HIP GEMV kernel (required)
scripts/run_devstral_awq.sh          # Devstral 24B
scripts/run_coder30b_awq.sh          # Coder-30B MoE
```
