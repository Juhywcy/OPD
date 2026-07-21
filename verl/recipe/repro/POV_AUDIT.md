# POV Gradient And Prefix Audits

This directory contains the implementations for experiments 3 and 4 in
`paper/OAL/LaTeX/experiment_todo.md`.

## Protocol

`pov_gradient_audit.py` freezes the supplied final POV checkpoint. For every
audit batch it samples response groups A and B with different deterministic
seeds. It computes Outcome-PG, sampled-token Raw OPD, and POV-OPD gradients on
A, and an independent outcome reference on B. Advantages and POV weights are
detached, all methods use the same response mask, loss aggregation, and
parameter scope, and no optimizer is created.

The sampled-token OPD signal is `teacher_logp - student_logp`. POV weights are
computed by the same `compute_outcome_aligned_logit_opd_advantage` function used
by training. The defaults mirror `scripts/train/run_prefix_trust_oal_opd.sh`:
rank weighting, anti-beta 0.1, aligned boost 0.5, window size 128, two baseline
blocks, drift allowance 0.1, CUSUM threshold 1.0, and decay lambda 1.0.

`prefix_horizon_audit.py` then matches each triggered A trajectory to an
untriggered A trajectory with the same prompt and outcome class. It uses a
fixed relative-length caliper, matches without replacement, puts the trigger
window in the post period, and transfers the trigger's normalized position to
the control. DID confidence intervals cluster-bootstrap by prompt.

## Outputs

- `config.json`: complete command configuration.
- `batches.jsonl`: batch metrics and per-trajectory window metrics.
- `gradient_summary.json`: shared-reference-set gradient results.
- `matched_pairs.jsonl`: accepted one-to-one matches.
- `prefix_horizon_summary.json`: trigger, matching, pre/post, DID, and CI results.

The default `GRADIENT_PARAMETER_REGEX=lm_head` is a gradient-subspace audit and
must be described that way in the paper. `GRADIENT_PARAMETER_REGEX='.*'` runs a
full-model audit, but per-window backward passes make it substantially slower.

## Smoke Test

Run two short audit batches before the full job:

```bash
MODEL_PATH=/path/to/final_pov_hf_checkpoint \
TEACHER_MODEL_PATH=/root/models/hbx/JustRL-DeepSeek-1.5B \
DATA_FILE=/path/to/held_out_audit.parquet \
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
NUM_BATCHES=2 \
N_RESPONSES=4 \
MAX_NEW_TOKENS=512 \
OUTPUT_DIR=/tmp/pov_audit_smoke \
bash scripts/audit/run_pov_audits.sh
```

After checking the summaries, remove the smoke-test overrides and launch the
fixed protocol on eight GPUs. Do not use DAPO-Math-17K training rows as the
held-out audit set.

