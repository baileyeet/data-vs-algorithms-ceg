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
  A0D1 1.1041; A1D1 v2 1.0815; A1D0 v2 1.2070). Medium A1 v2 reruns DONE
  2026-07-15, both within ±0.01 of v1. Not final until CORE re-sweep +
  Shapley + Tier-1 correction land.
- **Cost ledger**: Tier 2 projected **~$235 vs $200–230 envelope — flagged to
  user, not resolved.** Rerun cascade actual ~$60 vs $35 estimate. Two-bucket
  tracking (prep/validation vs per-tier training) in RUNBOOK.md.

## Loader saga (RESOLVED 2026-07-15 — read before touching A1 evals)

Post-hoc loading of modded-nanoGPT checkpoints was unfaithful. FOUR independent
causes, all diagnosed and fixed:
1. **Unfaithful yarn reconstruction**: rotary/YaRN buffers are persistent=False
   (never in state_dict) and training mutates them on the window schedule.
   Replay reconstruction (Option B) FAILED exactness: deltas 0.021–0.035,
   step-dependent. Fix: wrapper saves `yarn_state` in every checkpoint
   (small track attrs: factor1/factor2; medium track: cos/sin — names differ).
2. **Eager-vs-compiled numerics gap**: even exact yarn left +0.0186 BPB (fp8
   paths differ eager vs compiled). Fix: loader compiles the model.
3. **Medium split_embed flag** (caught by the early/mid/late batch, which
   FAILED 0.38–0.57 on all post-split medium ckpts): medium ties embed to
   lm_head.weight until split_step then unties; the flag is a plain attr,
   never in state_dict — a fresh model runs post-split ckpts with the
   diverged lm_head as embedding. Fix: loader derives it (weight equality +
   split_step from run_config) in BOTH build_model and load_into.
4. **Fidelity-instrument packing bias** (+0.002–0.006 on all small deltas):
   hand-rolled 1025-token segments without BOS splits ≠ training eval.
   Fix: loader_fidelity_check.py uses the wrapper's make_eval_chunks /
   evaluate_bpb_modded verbatim.
- **Validated: 8/8 PASS, all deltas 0.0000** (early/mid/late × both tracks ×
  both runs per track); load_into pre→post-split swap also exact. Formula =
  saved yarn_state + split_embed restore + torch.compile + training packing.
- lambada accuracy is INVALID for A1 arms (loader has no logits path →
  is_greedy hardcoded false); excluded from A1 CORE, documented.

## Canonical-runs decision (user-ratified)

**v2 reruns (with yarn_state) are canonical for ALL A1 numbers at BOTH tiers —
primary BPB and CORE.** v1 metrics → results/superseded/ (audit trail, never
deleted). Tier-1 Shapley must be re-derived from v2 curves and issued as an
explicit correction, not silently. v1/v2 agreement is within the measured
±0.01 same-seed noise floor.

## Remaining queue (in order)

1. ~~Medium A1 v2 reruns~~ DONE (a1d1 1.0815, a1d0 1.2070).
2. ~~Fidelity early/mid/late both tracks~~ DONE (8/8 exact).
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
