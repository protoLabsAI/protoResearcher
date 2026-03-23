# DPO Experiment Program — Qwen3.5-0.8B

You are running autonomous DPO (Direct Preference Optimization) experiments on Qwen3.5-0.8B using LLaMA-Factory.

## Objective

Minimize **eval_loss** through systematic hyperparameter exploration. Each experiment modifies `config.yaml` and runs training on GPU.

## Rules

1. **One change at a time.** Each experiment should test a single hypothesis.
2. **Commit before running.** Every hypothesis is a git commit.
3. **Keep or discard.** If eval_loss improved, keep. If not, discard (git reset).
4. **Log everything.** Results go in results.tsv automatically.
5. **Favor simplicity.** If two configs give similar results, prefer the simpler one.

## Experiment Ideas (ordered by expected impact)

1. **Learning rate sweep**: Try 5e-6, 1e-5, 2e-5, 5e-5
2. **LoRA rank**: Try 16, 32, 64, 128 (with alpha = 2x rank)
3. **Batch size**: Try effective batch sizes 8, 16, 32 (batch_size * grad_accum)
4. **Beta (DPO weight)**: Try 0.05, 0.1, 0.2, 0.5
5. **Epochs**: Try 1, 2, 3 (watch for overfitting via train vs eval loss gap)
6. **Cutoff length**: Try 1024, 2048, 4096
7. **Warmup ratio**: Try 0.05, 0.1, 0.2
8. **Dropout**: Try 0, 0.05, 0.1

## Workflow

```
1. lab_bench action=config experiment=<name>     # Read current config
2. lab_bench action=edit key=<k> value=<v>       # Change one parameter
3. lab_bench action=run description="<what>"      # Run experiment
4. If improved: lab_bench action=keep             # Accept
   If not:      lab_bench action=discard          # Revert
5. Repeat from step 1
```

## Constraints

- Model: Qwen3.5-0.8B (fixed — do not change model_name_or_path)
- Dataset: wildbench_dpo_9b (fixed — 997 DPO pairs from GPT-5.4-mini vs Qwen3.5-9B)
- GPU: Single GPU (typically GPU 1), ~96 GB VRAM available
- Time budget: 5 minutes default (adjustable)
- Do NOT modify dataset_dir, output_dir, or template
