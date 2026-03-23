# DPO Experiment Program — Qwen3.5-2B

You are running autonomous DPO experiments on Qwen3.5-2B using LLaMA-Factory.

## Objective

Minimize **eval_loss** through systematic hyperparameter exploration. The 2B model has more capacity than 0.8B — it may respond differently to the same hyperparameters.

## Rules

1. **One change at a time.** Each experiment tests a single hypothesis.
2. **Commit before running.** Every hypothesis is a git commit.
3. **Keep or discard.** If eval_loss improved, keep. If not, discard.
4. **Favor simplicity.** Simpler configs are preferred at equal performance.

## Key Differences from 0.8B

- Lower learning rate (5e-6 vs 1e-5) — larger models are more sensitive
- Smaller batch size (1 vs 2) — more VRAM per sample
- May benefit from lower LoRA rank (less capacity needed for adaptation)
- Watch for overfitting more closely (larger model, same dataset)

## Experiment Ideas

1. **Learning rate**: 2e-6, 5e-6, 1e-5, 2e-5
2. **LoRA rank**: 16, 32, 64 (with alpha = 2x rank)
3. **Beta**: 0.05, 0.1, 0.2
4. **Epochs**: 1, 2, 3
5. **Cutoff length**: 1024, 2048

## Constraints

- Model: Qwen3.5-2B (fixed)
- Dataset: wildbench_dpo_9b (997 DPO pairs)
- GPU: Single GPU, ~96 GB VRAM
- Do NOT modify dataset_dir, output_dir, or template
