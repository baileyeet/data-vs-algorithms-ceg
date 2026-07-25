# CLAUDE.md — Data vs. Algorithms CEG experiment

2x2 grid (old/new data x old/new algorithm) at GPT-2 scales measuring
compute-equivalent gain. Methodology invariants: README.md (9 requirements —
do not "improve"). Session procedures + cost ledger: RUNBOOK.md. Paid compute
only with explicit user confirmation, per tier. Commits: one line, never
mention Claude/Anthropic. Long jobs: persistent state-change monitors until
completion.

## Current state (2026-07-16)

## STUDY COMPLETE (2026-07-25) — final result in report.md

TWO cross-scale curves, kept separate (report.md, analysis/make_report.py):
- **Curve 1 current-arch (124M, 355M):** algo advantage 13.7x -> 4.1x (steep
  decay). 1.5B GAP disclosed (no reproducible recipe; Option-B invented-arch
  rejected).
- **Curve 2 ScaleUp-arch (124M, 1.5B):** algo-on-DCLM 2.90x -> 2.34x (mild
  decay); data-dependent (censored <=1x on OWT both scales); data mult ~3.3x
  stable. 355M GAP disclosed (no documented recipe). ALL arms 5-GPU
  (A0-124M re-run on 5-GPU — 8-vs-5 mix had distorted 124M to 2.35x; gpu-hours
  NOT count-invariant, ~22% batch/accum cost).
Unified finding: both advantages shrink with scale; aggressively small-scale-
tuned current speedrun decays fast from a high base, older fundamental ScaleUp
decays gently. All on HF (1.5B/*, 124M-scaleup/*) + git. GPUs idle.
Hard-won lessons this phase: coupled-recipe (dims+batch+LR+schedule) doesn't
scale by dims alone; verify hardware consistency don't assume; never launch a
GPU job before the prior one's clean exit (port/contention).

## TIER 3 FULL RUN (launched 2026-07-21)

**4-arm 1.5B measured run LIVE on 5xH100** (pod 31.24.80.40:15474). Driver
`/root/xl_tier3.sh` (chained, aborts on any arm failure); per-arm logs
`/root/xl_<arm>.log`; main log `/root/xl_tier3.log`. ~2.35 days / ~$850-900.
Order: A0D0 -> A0D1 -> A1D0 -> A1D1. Each arm: rolling prune keep-3, dense-tail
schedule (n-ckpts 35), final uploaded to HF (1.5B/<ARM>).

**Config (user-ratified):** A0 arms (GPT-2-XL, train_old, size xl): budget
8.87e9 (full single-pass OWT; DCLM uses 8.87B of 18B), total-batch-tokens
512000, device-batch 10. A1 arms (2024 ScaleUp, train_gpt_xl_ceg): budget
9,999,114,240 (documented 20344 steps), device-batch 12; A1D0 --n-epochs 2
(10B on 8.87B OWT = ~1.13 epochs, minor repetition — user chose documented
budget over single-pass), A1D1 --n-epochs 1 (DCLM 18B).

**5 GPUs LOCKED** — validation confirmed stability at this batch/GPU count;
if the pod is relaunched, re-bootstrap and relaunch `/root/xl_tier3.sh`
(nproc 5), do NOT switch GPU count.

**Pre-launch gates ALL verified on GPU (2026-07-21):** (1) rolling prune
deletes live, disk bounded [smoked both trainers]; (2) A0 GPT-2-XL builds/fits,
~7.4 GPU-h/B throughput [= A1]; (3) modded loader REJECTS a real xl ckpt
[routing fixed]; (4) dense-tail schedule built into checkpoint_schedule; (5)
verify_row_hparams passes on xl A0 row (run_id added to ALLOWED_DIFF); (8) 6GB
HF upload works (42s, size-verified). De-risk validation earlier: clean
monotonic descent to 1.0715 BPB over full 2B schedule, no divergence.

**Monitoring:** per-arm completion reported; divergence watch (neutral BPB
rising >0.05 above the arm's running min = the Tier-1 signature: flag
immediately). Post each arm: run verify_row_hparams per row, confirm BPB
descent.

**Threshold: A0D0-1.5B = 1.187920 BPB** (final-10% mean; A0D0 final 1.1880,
65.6 GPU-h). Threshold trend 1.2744 (124M) -> 1.2287 (355M) -> 1.1879 (1.5B).
**CANONICAL threshold fn = analysis/threshold.py::final_tail_threshold** (mean
neutral_bpb over tokens>=0.9*total; token-fraction anchored, ONE deterministic
rule, no per-arm logic). VERIFIED 2026-07-24 bit-identical across all 3 tiers
vs reported (124M 1.274421 n=6, 355M 1.228738 n=6, 1.5B 1.187920 n=5) — no
shift. Tail count wobbles +-1 by boundary rounding but same rule. Final
analysis + report MUST source thresholds from this fn (not ad-hoc); arm
crossings from interpolated curve-vs-threshold (min-based), never tail means.

**1.5B ARM RESULTS (as they finish):** A0D0 1.1880 (=threshold). A0D1 1.0403
(crosses ~15.8 GPU-h; data helps a lot). A1D0 1.2086 — **DID NOT CROSS**
(ScaleUp-on-OWT < GPT-2-XL-on-OWT at 1.5B; algorithm CEG on old data <=1x;
crossing CENSORED -> handle in Shapley as no-crossing bound, do NOT fabricate
a crossing). A1D1 running (DCLM, likely crosses). Clean training throughout,
no divergence — the weak A1 result is the algorithm-version confound (2024
ScaleUp is an OLD modded gen), not a bug.

**SCALEUP CURVE = 2-POINT (124M + 1.5B), 355M UNFILLABLE (2026-07-24):** the
"scale dims only" approach FAILED — the ScaleUp recipe is a COUPLED bundle
(dims+batch+LR+schedule). Holding the 1.5B config fixed mis-tuned small sizes:
124M @ 8.87B stalled at 1.367 (used lr 0.0018 = HALF the right value; NOT a
batch fallback — confirmed 480-seq batch ran). FIX: per-size DOCUMENTED configs
in SCALEUP_CONFIG (train_gpt_xl_ceg.py): xl=2024-10-20 ScaleUp1B; xl124m=
2024-10-18 speedrun (lr 0.0036, batch 512->510 for 5-GPU, 5100 steps, ~2.66B
tok, warmdown 1450). 355M ABSENT — no documented era-appropriate recipe; refuse
to hand-derive LR (user-ratified: same unvalidated guess as the rejected skip
topology). Report: ScaleUp curve is 124M + 1.5B; 355M gap DISCLOSED. Two failed
small attempts (~$30-40) before getting the recipe right. Mis-tuned 124M ckpt
DELETED from HF. Running 124M now (2 A1 arms, driver /root/scaleup124.sh).

**(prior) SCALEUP CURVE RUNNING (launched 2026-07-24, 5xH100):** driver
/root/scaleup_curve.sh (chained, abort-on-fail); per-arm logs /root/su_<size>_<arm>.log;
main /root/scaleup_curve.log. 4 A1 arms: xl124m {a1d0 owt, a1d1 dclm} + xl355m {a1d0 owt, a1d1 dclm}, budget
8,870,000,000 (single-pass; = A0 baseline). n-epochs 1.
**BUDGET CORRECTED 2026-07-24 (user-approved):** constant-tokens/param
(0.80B/2.28B) UNDERTRAINED the small arms — xl124m/a1d0 finished its 0.8B
schedule at 1.407 (still descending), never reaching the 1.274 threshold (=
GPT-2 converged @8.87B; NO 124M model reaches it in 0.8B). Evidence it was
undertraining not weakness: ScaleUp @0.8B (1.407) BEAT GPT-2 @0.8B (1.458) —
real advantage HIDDEN by the short budget. Fix: match A0 baseline 8.87B so the
arm has runway to cross. 1.5B arms UNAFFECTED (10B -> plateaued). Rerun ~$120. Reuses existing A0/GPT-2 baselines +
thresholds (124M 1.274421, 355M 1.228738) — only A1 arms run. HF -> DISTINCT
124M-scaleup/ , 355M-scaleup/ labels. Smokes passed (124M 123.57M params, 355M
353.50M). NOTE: first launch aborted on EADDRINUSE (port 29500) from a
still-running smoke — contention, not a bug; relaunched clean on idle GPUs.
LESSON: wait for explicit clean-exit before launching next GPU job.

**FOLLOW-UP STUDY (user-ratified 2026-07-24, DO AFTER 1.5B matrix finishes):**
Option C — run the 2024-ScaleUp arch (train_gpt_xl_ceg's GPT, size-configured)
ALSO at 124M and 355M, giving a CLEAN 3-point ScaleUp curve (same algorithm
held fixed across all scales; the 1.5B point already in hand). Keeps the
existing current-arch 124M/355M results SEPARATE — report must show TWO curves,
never blended: (i) current-arch 2pt (124M/355M, SOTA speedrun, can't extend to
1.5B), (ii) ScaleUp-arch 3pt (124M/355M/1.5B, clean cross-scale). Rejected
Option B (scale current arch UP to 1.5B) — needs untestable invented skip
topology. Token budget for ScaleUp at small sizes: constant tokens/param
anchored on documented 1.5B ScaleUp (10B/1.556B = 6.43 tok/param) -> 124M
~0.80B, 355M ~2.28B (both single-pass on OWT 8.87B, no repetition). SMOKE each
before full run. Small ScaleUp runs are cheap (~$40-70 total).
CURRENT-ARCH 1.5B (Option B) DEFINITIVELY SKIPPED (user 2026-07-24) — leave the
current-arch curve's 1.5B gap as an honestly-disclosed limitation in the report.
INFRA CONFIRMED for ScaleUp small runs (2026-07-24): (a) TOKENIZER identical —
GPT-2 BPE (vocab 50304, eot 50256); reuse owt_nanogpt (D0) / dclm_nanogpt (D1)
shards as-is, NO re-tokenization (the 1.5B ScaleUp already ran on them). (b)
Dense-tail schedule, verify_row_hparams (arch-agnostic, run_id allowed), per-arm
HF upload, divergence-watch, rolling prune — ALL apply unchanged (same trainer/
wrapper path). HF paths use DISTINCT labels (e.g. 124M-scaleup/A1D0) so the two
curves stay separate on HF too. BUILD TASK before launch: train_gpt_xl_ceg.py
hardcodes 1.5B dims (n_layer=52,n_head=12,n_embd=1536) — parameterize by size:
124M=(12,6,768), 355M=(24,8,1024) [head_dim 128 throughout, plain GPT-2 dims];
verify param counts via probe, then smoke. A1D0-1.5B reporting: reached 1.2082,
1.71% short of threshold 1.1879 — report as "no crossing -> no algo CEG on OWT
at 1.5B", bounded; keep "gap to threshold %" framing ready for A1D1.

**MANDATORY report caveat (unchanged):** 1.5B A1 = OLDER modded generation
than 124M/355M (2024 ScaleUp plain transformer). Algorithm-version confound
front-and-center. Already in report.md + analysis/make_report.py.

## Tier-3 (1.5B) A1 recipe decision (user-ratified 2026-07-17) — background

**A1 @ 1.5B uses the DOCUMENTED 2024 ScaleUp1B recipe (Option A), NOT a
scaled current-arch (Option B).** Reproducible config lives in the vendored
log `train_new/modded-nanogpt/records/track_1_short/2024-10-20_ScaleUp1B/
ad8d7ae5-…txt` (full source embedded): GPTConfig(n_layer=52, n_head=12,
n_embd=1536), 20344 steps, batch 480 seq × 1024, ~10.0B tokens, lr 0.0036/2
= 0.0018 (Muon 0.1×), warmup 0, warmdown 5812. It is a PLAIN transformer:
Linear QKV, base-10000 Rotary, 4× MLP, weight tying, old Muon+AdamW — NO
YaRN/windows, NO split_embed, NO value embeds, NO U-net skips, NO fp8/MTP.
So the whole loader-fidelity problem class (yarn_state, split_embed) does
NOT apply — post-hoc load is a plain state_dict load.

**WHY not Option B (scale current medium arch):** scaling to 48 layers
requires re-deriving 3 hand-tuned architectural subsystems with no 1.5B
reference — window layout (derivable), value-embed placement (clean), and
the U-net skip topology (skip_in/out/backout, 3 skips baked into the scalar
layout). The skip topology is an UNTESTABLE guess: a wrong one yields a
plausible-but-non-representative model with NO divergence signature (unlike a
bad schedule). User: that confound is worse than A's.

**MANDATORY REPORT CAVEAT (user, front-and-center, not a footnote):** the
1.5B A1 arm is an OLDER modded-nanoGPT generation than the 124M/355M A1 arms
(current speedrun). Any change in the algorithm multiplier at 1.5B MUST be
interpreted with this algorithm-version confound stated prominently.

**Status (2026-07-17, pod TERMINATED for the night):** XL A1 trainer built =
train_new/train_gpt_xl_ceg.py (2024 arch, CEG-wrapped, reuses common.bpb.
evaluate_bpb = same instrument as A0). SMOKE PASSED: 1,549,467,648 params
(=GPT-2-XL), fits at 66GB/80GB, BPB descends, ckpt 5.8GB fp32 (~matches 6.2GB
projection), clean exit. De-risk VALIDATION (2B budget, a1d1/dclm, save 0) ran
HEALTHY to step 283/~4069 (~7%, BPB monotonic 3.16->1.68) then interrupted for
teardown — NOT run to completion. Partial curve in results/xl_validation_partial/.
Three wiring bugs fixed en route (xl max_stage guard; DataLoader skip-undersized-
shard for the 1876-tok remainder; + earlier Option-B grouping/window/ve-gate,
now moot). One smoke ckpt kept on /workspace/runs/xl_smoke/ckpt_000041.pt for a
later plain-load test.

**RESUME PLAN (next session), user-ratified 2026-07-18:** new pod -> re-bootstrap
(git archive sync + pip deps; container disk resets on terminate, /workspace
persists). Then: FINISH THE XL DE-RISK VALIDATION TO COMPLETION FIRST (~2h,
~$47) — do NOT treat the partial 7% as sufficient. Rationale (user): this
project's problems tend to surface mid-run, not at the start; confirm the full
BPB curve stays clean before committing to the paid four-arm tier. Only after a
clean full validation curve -> FULL Tier-3 go/no-go.
A0 1.5B arm ALREADY WIRED + VERIFIED (configs/model_sizes.json xl = GPT-2-XL
48L/25H/1600d, lr 2e-4; local param count 1557.7M = 1.558B). No A0 wiring needed. Full Tier-3 still needs: A0 1.5B arm wired (standard
GPT-2-XL in train_old), rolling checkpoint prune (ESSENTIAL at 5.8GB/ckpt),
bf16 A0 saves, and the loader/CORE path for xl (plain state_dict load + A0
lm-eval adapter, NOT the modded one). This is a USER go/no-go on paid full-tier
compute. Loader routing in eval/lm_eval_adapter_modded.py still wrongly treats
xl as medium-lineage (the _medium_lineage/_trainer_file xl bits) — REMOVE/redirect
xl to the A0 adapter before any xl CORE sweep.

**AT THE TIER-3 GO/NO-GO HARD STOP.** Queue items 1–5 done: v2 reruns,
fidelity 8/8 exact, CORE re-sweep, Tier-1 correction (Shapley re-derived from
v2), v2 A1 arms uploaded+verified to HF, CORE gate re-derived, Tier-2 Shapley
+ 2-point cross-scale plot done. Headline cross-scale trend below. Awaiting
user decision on Tier 3 (1.5B) — paid, needs explicit confirm + pre-flight
gates. Pod restarted on port 14523 (was 18124); container disk reset so repo
re-synced via `git archive` (clone auth still broken — use archive, not clone).

### Headline (v2-canonical, log-space Shapley)
| Scale | Data mult | Algo mult | Total | Threshold (neutral BPB) |
|-------|-----------|-----------|-------|--------------------------|
| 124M  | 2.23x     | 13.69x    | 30.5x | 1.274421 (new def)       |
| 355M  | 3.74x     | 4.06x     | 15.2x | 1.228738                 |
Algorithm advantage COLLAPSES with scale (13.7->4.1x); data GROWS (2.2->3.7x).
CORE (secondary, validity-gated): 124M 5 usable tasks (boolq drops — A0D0 at
chance 0.504), 355M 6; lambada excluded for A1 (no logits path). All A1 CORE
from v2 sweep. Arms cluster near each other on CORE at these scales (noisy,
limit=500) — BPB is the real signal.

## Prior state (2026-07-15)

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
3. ~~Consolidated CORE re-sweep of all v2 A1 runs~~ DONE (160 ckpts,
   results/core_sweep_v2/).
4. ~~Tier-1 correction + CORE gate re-derivation; uploads of v2 arms~~ DONE
   (Shapley re-derived; analysis/core_gate.py -> results/core_gate_v2.json;
   4 v2 A1 arms uploaded+verified to HF).
5. ~~Tier-2 Shapley + cross-scale plot~~ DONE (results/medium/ceg_newdef.json,
   results/cross_scale.png) -> **NOW AT USER GO/NO-GO for Tier 3.**
6. Tier 3 pre-flight gates in RUNBOOK.md (500GB volume ✅ done; hooks/rolling
   prune/bf16 saves/first-ckpt size check/XL recipe pending — only if user
   says GO).

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
