# Independent L2S Stage-1 Recipe

This directory reproduces L2S stage-1 without modifying VERL's original trainer, OPD code, or existing recipes.

For a response of length `L`, the rule reward is:

```text
correct:   2 * exp(-(-log(0.01) / 8192) * L)
incorrect: 0
```

The reward manager places this scalar only on the final valid response token. The recipe-local advantage estimator then sums the token rewards and broadcasts the sequence reward over every valid response token. It does not perform GRPO group normalization.

Run from the repository root:

```bash
bash verl/recipe/repro/deepseek_r1_distill_llama_8b_stage1.sh
```

The default model is `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`. A local checkpoint and dataset paths can be supplied without editing the script:

```bash
MODEL_PATH=/path/to/model \
TRAIN_FILE=/path/to/train.parquet \
TEST_FILE=/path/to/test.parquet \
bash verl/recipe/repro/deepseek_r1_distill_llama_8b_stage1.sh
```

Use `DRY_RUN=1` to print the full command without starting Ray or loading the model.
