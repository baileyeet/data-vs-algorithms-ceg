# CLAUDE.md — Data vs. Algorithms CEG experiment

2x2 grid (old/new data x old/new algorithm) at GPT-2 scales measuring
compute-equivalent gain. Methodology invariants: README.md (9 requirements —
do not "improve"). Session procedures + cost ledger: RUNBOOK.md. Paid compute
only with explicit user confirmation, per tier. Commits: one line, never
mention Claude/Anthropic. Long jobs: persistent state-change monitors until
completion.

## Current state (2026-07-15)

- **Tier 1 (124M): closed, but A1 numbers under explicit correction** (see
  loader saga below). Headline pending re-derivation from v2 reruns.
  Threshold definition = mean neutral BPB over final-10% checkpoints of A0D0
  (dense-tail schedule; single-final variant reported as robustness).
- **Tier 2 (355M): all 4 arms trained** (A0D0 1.2300, threshold 1.228738;
  A0D1 1.1041; A1D1 1.0883; A1D0 1.2027). Medium A1 v2 reruns IN PROGRESS —
  **nothing is final until they finish.**
- **Cost ledger**: Tier 2 projected **~$235 vs $200–230 envelope — flagged to
  user, not resolved.** Rerun cascade actual ~$60 vs $35 estimate. Two-bucket
  tracking (prep/validation vs per-tier training) in RUNBOOK.md.

## Loader saga (resolved root cause — read before touching A1 evals)

Post-hoc loading of modded-nanoGPT checkpoints was unfaithful. Two independent
causes, both diagnosed and fixed:
1. **Unfaithful yarn reconstruction**: rotary/YaRN buffers are persistent=False
   (never in state_dict) and training mutates them on the window schedule.
   Replay reconstruction (Option B) FAILED exactness: deltas 0.021–0.035,
   step-dependent. Fix: wrapper saves `yarn_state` in every checkpoint
   (small track attrs: factor1/factor2; medium track: cos/sin — names differ).
2. **Eager-vs-compiled numerics gap**: even exact yarn left +0.0186 BPB (fp8
   paths differ eager vs compiled). Fix: loader compiles the model.
   **Validated formula = saved yarn_state + torch.compile → delta 0.0024 PASS**
   (tolerance 0.005 vs training-recorded BPB).
- **OPEN ITEM**: fidelity has passed on ONE checkpoint (small A1D1 v2 final).
  It must pass across early/mid/late checkpoints of both tracks before the
  fix is fully trusted.
- lambada accuracy is INVALID for A1 arms (loader has no logits path →
  is_greedy hardcoded false); excluded from A1 CORE, documented.

## Canonical-runs decision (user-ratified)

**v2 reruns (with yarn_state) are canonical for ALL A1 numbers at BOTH tiers —
primary BPB and CORE.** v1 metrics → results/superseded/ (audit trail, never
deleted). Tier-1 Shapley must be re-derived from v2 curves and issued as an
explicit correction, not silently. v1/v2 agreement is within the measured
±0.01 same-seed noise floor.

## Remaining queue (in order)

1. Medium A1 v2 reruns finish (chained, monitored).
2. Fidelity check incl. a medium checkpoint + early/mid/late coverage.
3. Consolidated CORE re-sweep of all v2 A1 runs (use --more-ckpts amortized
   -compile mode in eval/lm_eval_adapter_modded.py; HF_TOKEN + PID-unique
   ports required for parallel).
4. Tier-1 correction + CORE gate re-derivation; uploads of v2 arms.
5. Tier-2 Shapley + re-derived Tier-1 + 2-point cross-scale plot → user for
   Tier-3 go/no-go.
6. Tier 3 pre-flight gates in RUNBOOK.md (500GB volume ✅ done; hooks/rolling
   prune/bf16 saves/first-ckpt size check pending).

## Operational gotchas (hard-won)

- NEVER put a pkill pattern in the same ssh command as a relaunch (self-match
  kills the session). Kill and launch in separate ssh calls.
- NEVER run anything on GPUs during a measured training run (pollutes timed
  compute). Evals/sweeps only on idle GPUs.
- Measured runs: budgets must be chunk-aligned (train_old) and data-available
  (no-silent-wrap guards exit early otherwise).
- Volume filled at 280GB once (crashed a run mid-training); now 500GB. Prune
  swept checkpoints promptly; disk pre-flight before each tier.
- Pod SSH: direct TCP only for automation (ssh.runpod.io proxy is
  interactive-only). RunPod API key never worked (401) — pod lifecycle is
  manual via user's console.
- HF repo: baileymachihirota/data-vs-algorithms-ceg (private). Eval corpus
  frozen: /workspace/datasets/wiki_eval, sha256 cbdd72ac…, never regenerate.
