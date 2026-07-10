# L2S Stage-1 Independent Recipe Implementation Plan

**Goal:** Reproduce the original L2S stage-1 behavior in an isolated VERL recipe without prefix, OPD, or GRPO-combination logic.

**Architecture:** Register a recipe-local reward manager that places the length-decayed correctness reward on the final valid response token. Register a recipe-local advantage estimator that sums this terminal reward and broadcasts it across all valid response tokens. Use the upstream PPO engine only as infrastructure; all L2S-specific behavior stays under `verl/recipe/repro/`.

**Runtime:** Python, PyTorch, VERL, Hydra, FSDP, and vLLM.

## Constraints

- Do not modify files under `verl/verl/` or existing OPD recipes.
- Correct response reward: `2 * exp(-(-log(0.01) / 8192) * response_length)`.
- Incorrect response reward: `0`.
- Place the scalar reward only on the final valid response token.
- Broadcast the sequence reward over the valid response mask for advantages and returns.
- Do not use prefix windows, OPD weights, prefix rewards, or direct-plus-GRPO.

## Tasks

1. Add tests for the decay formula, terminal reward placement, incorrect responses, and reward broadcasting; run them once to confirm the old implementation fails.
2. Replace the prefix reward manager with an L2S stage-1 reward manager and add the independent reward-as-advantage estimator; run the unit tests.
3. Replace the Qwen/prefix launch script with a DeepSeek-R1-Distill-Llama-8B script and validate its shell syntax and rendered Hydra arguments.
4. Verify no references to prefix/OPD/direct-plus-GRPO remain under `repro`, verify no tracked files outside `repro` changed, and run all recipe-local tests.
