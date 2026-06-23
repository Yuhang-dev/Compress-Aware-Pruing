# Compression-Aware Safety Entanglement

Project scaffold for the experiment plan in `experiment_plan_compression_aware_safety.md`.

The repository is phase-gated. Do not skip gates:

- Phase 0: environment, dataset manifest, pruning/compression evaluation pipeline
- Phase 1: refusal direction extraction and mechanism diagnosis
- Phase 2: Method A2 go/no-go
- Phase 3: Method B cross-compression robust training
- Phase 4: baselines and ablations
- Phase 5: cross-model generalization and final tables

## Remote Environment

Reuse the existing AutoDL environment and caches:

```bash
source scripts/remote_bootstrap.sh
```

This activates `pbp` and points Hugging Face, Torch, and pip caches to `/root/autodl-tmp`.

Expected remote checkout:

```bash
cd /root/autodl-tmp
git clone https://github.com/Yuhang-dev/Compress-Aware-Pruing.git cap
cd cap
source scripts/remote_bootstrap.sh
```

Do not use the deprecated FFAP checkout as this project's root.

## Local Layout

```text
configs/                  YAML experiment configs
data/                     dataset manifest and builders
src/casafety/             Python package
results/                  CSVs, figures, summaries
artifacts/                cached calibration and refusal vectors
scripts/                  phase and remote helper scripts
```

## First Commands

```bash
python -m casafety.env_report --output env_report.txt
python data/build_refusal_sft.py --config configs/base.yaml --write-manifest data/manifest.json
```

If `python -m casafety...` cannot find the package, either run
`source scripts/remote_bootstrap.sh` or export `PYTHONPATH=$PWD/src:$PYTHONPATH`.

To populate the remote Hugging Face cache:

```bash
export HF_TOKEN=...  # only in the private shell, never in git or logs
export HF_HUB_DISABLE_XET=1
unset HF_XET_HIGH_PERFORMANCE
python scripts/download_phase0_assets.py --config configs/base.yaml --models-only
```

After enabling GPU, run the full dependency/model download:

```bash
bash scripts/phase0_full_download.sh
```

The script intentionally skips `vllm`, `bitsandbytes`, `auto-gptq`, and `autoawq`
unless `INSTALL_HEAVY_EVAL_DEPS=1` is set, because those packages may pull large
PyTorch CUDA component wheels and disturb the known-good torch environment.

Run the Phase 0 smoke evaluation after Llama 2 is cached:

```bash
bash scripts/phase0_smoke_eval.sh
```

This writes `results/phase0_problem.csv` and
`results/phase0_problem_summary.csv`. It is a small keyword-judge smoke test,
not the final LlamaGuard3 evaluation.

Heavy model, dataset, pruning, and evaluation paths intentionally require explicit verification before use.
