# CLAUDE.md — Data vs. Algorithms CEG experiment

2x2 grid (old/new data x old/new algorithm) at GPT-2 scales measuring
compute-equivalent gain. Methodology invariants: README.md (9 requirements —
do not "improve"). Session procedures + cost ledger: RUNBOOK.md. Paid compute
only with explicit user confirmation, per tier. Commits: one line, never
mention Claude/Anthropic. Long jobs: persistent state-change monitors until
completion.

## NEW EXPERIMENTS A + B (IN PROGRESS, launched 2026-08-08)

Two follow-on experiments on the completed 124M/355M/1.5B study. Ground rules
(user): budget NOT binding this round; ENGINEERING RISK + WALL-CLOCK are the
constraints; estimate + confirm before each paid launch; STOP-and-flag on
anything that doesn't transfer cleanly (tokenizer/ckpt/eval-harness/schedule);
flag the train-improving-while-neutral-BPB-worsening divergence signature.

**Pod (restarted 2026-08-08): 8xH100 at `31.24.80.43:12738 -i ~/.ssh/id_ed25519`.
KEEP RUNNING (user: 8xH100 SXM is "Low" availability on this account — don't
risk an availability gap; wall-clock > idle-GPU cost).** Container disk reset on
restart (only /workspace persists) — re-bootstrapped repo to /root/ceg via git
archive + `pip install datasets tiktoken numpy` (full requirements-lock still
needed for training). No HF_TOKEN on pod (downloads throttled/unauth).

**TOKENIZER FINDING (report note):** ALL arms in the completed study used GPT-2
BPE (A0, current-arch, AND ScaleUp all = vocab 50304). Tokenizer never actually
varied. So Exp A's "both tokenizers" collapses to one GPT-2-BPE tokenization +
nanogpt reshard; Exp B is the FIRST place tokenizer varies (each arch its own).

**EXP A — data-era ladder @ 124M (CONFIRMED, launched).** Era order (NO Wikipedia
as train — it's the neutral eval): OWT 2019 (reuse existing A0D0/A1D0), C4 2020
(NEW), RefinedWeb 2023 (NEW), DCLM 2024 (reuse A0D1/A1D1). C4+RefinedWeb each need
old-algo + current-arch = 4 new runs @124M, 8-GPU. Datasets verified accessible/
ungated/streamable: c4=`allenai/c4` cfg `en` field `text`; refinedweb=
`tiiuae/falcon-refinedweb` field `content` (wired into prepare.py). Tokenizing
9B unique GPT-2-BPE tokens each (seed 1234, shuffle-buffer 60000) -> /workspace/
datasets/{c4_gpt2,refinedweb_gpt2}; driver /root/tokenize_era.sh, logs /workspace/
session_logs/. ETA ~7h (unauth HF throttle; overlaps Exp B build). NEXT after
tokenize: (1) convert_to_nanogpt_bin for current-arch shards; (2) decontam
wiki_eval vs C4 and vs RefinedWeb SEPARATELY (eval/decontam.py), REPORT each
contamination % (expect ~20% CC-derived like DCLM); (3) report per-arm training
estimate, confirm, then train. Budgets: old-algo 8.87B single-pass; current-arch
native 124M budget. Output: CEG (data + algo contribution) vs dataset-release-year,
its OWN figure. NOT yet done: training launch (awaiting estimate+confirm).

**EXP B — architecture landscape (STAGED; B1 cleared to build).** Test whether
"algo advantage shrinks with scale" holds across lineages/classes. Feasibility
(4 web agents, 2026-08-08):
- Transformer lineage: Pythia (HF GPTNeoXForCausalLM; schedule REPRODUCIBLE;
  tok gpt-neox-20b; 160M/410M/1.4B) + SmolLM2 (HF LlamaForCausalLM; WSD =
  scale-down-friendly; own tok vocab 49152; 135M/360M/1.7B). current-arch +
  ScaleUp are the 2024-25 points (done).
- SSM lineage: Mamba (HF MambaForCausalLM; deps mamba-ssm+causal-conv1d) < Mamba-2
  (HF Mamba2ForCausalLM; +Triton SSD) < **H3 = HIGH RISK** (no HF class; dead repo;
  old flash-attn pin hostile to CUDA12/H100; training in separate safari repo).
  tok gpt-neox-20b. H3 125M/355M/1.3B, Mamba 130M/370M/1.4B, Mamba2 130M/370M/1.3B.
- **LFM2 DROPPED** (recipe NOT documented — arch/evals only; no LR/opt/batch;
  per no-invented-recipes rule). Arch is HF Lfm2ForCausalLM if ever revisited.
- KEY ENABLER: 4 of 5 arches (Pythia/SmolLM2/Mamba/Mamba2) are HF classes ->
  ONE shared from-scratch harness (HF model-from-config + BPB-over-HF-tokenizer +
  dense-tail ckpt + our metrics/threshold). Only H3 is bespoke.
DECISIONS (user-ratified): B1 first (Transformer lineage only) then B2 (Mamba/
Mamba2) then B3 (H3 timeboxed); **OWT-only** (data held fixed); **matched GPT-2
baseline at EVERY distinct candidate size** incl 1.3-1.7B, NO reuse-with-disclosure
(baseline defines that size's threshold denominator; trains parallel to harness
build so no wall-clock cost) — baseline question CLOSED, do not reopen.
TWO MANDATORY VALIDATION GATES before applying across candidates (user):
  (1) smoke-test the shared 8.87B re-anchored-schedule approach on Pythia-160M
      specifically, watching for the divergence signature (train loss down while
      neutral BPB up) — re-anchoring cosine/WSD to a shared budget is the same
      CLASS of op as the collapse-causing schedule-stretch; don't trust "agents
      said standard."
  (2) explicit correctness check of the new BPB-over-arbitrary-tokenizer
      instrument vs an independently-computed ground truth BEFORE it scores any
      real arm (precedent: the "conceptually clean" reload logic that was
      silently wrong).
8-GPU CONSISTENCY (user): everything Exp B on 8 GPUs; define schedules by GLOBAL
batch (tokens/step) + total tokens (GPU-count-invariant), realize as per-device=
global/(8*accum); VERIFY each candidate's global batch divides evenly by 8 at
build time, STOP-and-flag if not (the ScaleUp 5-vs-8 forced-batch-change trap).
GATE 1 FULL-LENGTH = PASS (2026-08-10): full Pythia-160M (4230 steps, re-anchored
cosine, owt_gpt2) BPB 3.51->1.293 monotonic, plateau at floor through fully-decayed
LR (0.1x peak); max rise above running min anywhere = 0.0009 (<<0.05). Late-decay
region clean. Re-anchored-schedule + shared harness validated end-to-end -> OK to
build out the Exp B lineage (matched baselines + Pythia/SmolLM2 full runs on their
own tokenizers, then B2 SSM) after the Exp A era-ladder + user review.
EXP A ERA-LADDER LAUNCHED (2026-08-10): union eval /workspace/datasets/wiki_eval_union
(2053 docs, 151 dropped/6.9% from 2204; C4 80 + RW 86 overlap 15). Driver
/root/era_ladder_driver.sh runs 8 arms @124M SEQUENTIAL on union eval (4 rescore
= explicit correction + 4 new C4/RefinedWeb), divergence-guarded, then auto-analysis
(ceg_shapley 2x2 correction -> era_correction_2x2.json; per-dataset crossings ->
era_ladder_results.json). NEXT after done: render results/era_ladder.png (CEG vs
release-year); report correction table (old 1.274421/2.23x/13.69x vs new
union-eval). 355M/1.5B union-rescore = disclosed-only (user async decision
pending; not blocking).
**VOLUME LOST + RECOVERED (2026-08-10):** the era-ladder chain aborted mid-run —
4 OLD-ALGO arms finished on torch 2.4.1, then the CURRENT-ARCH arms failed: the
modded trainer needs train_new/modded-nanogpt/triton_kernels.py which is GITIGNORED
so git-archive never synced it (restore via tar-pipe from local), + `pip install
kernels`, + **TORCH 2.10 REQUIRED** (study-lock 2.13.0 is cu130-only/undriveable;
2.4.1's triton lacks tensor_descriptor; use torch 2.10.0+cu128 = matches modded
requirements.txt). Then the whole pod's network volume was briefly lost, then
RECOVERED on a CPU pod **31.24.80.40:15961**. All /workspace data INTACT (tokenized
datasets, wiki_eval_union, era runs) — NO re-tokenization needed. **HF BACKUP in
progress** (durability): /root/hf_backup.py uploads datasets+eval+era-run-provenance
to MIRIBerkeley/data-vs-algorithms-ceg under `era-ladder/` (Xet DISABLED / setsid /
HF_TOKEN env-only). DCLM raw shards -> private-repo-internal ONLY (unreviewed
redistribution terms), never public. NEXT: user restarts pod w/ GPUs (volume
attached) -> re-bootstrap torch 2.10 + modded stack -> RE-RUN ALL 8 era arms on
2.10 (old-algo re-run too, for env consistency; the 2.4.1 runs are provenance only)
with per-arm HF upload -> analysis + era_ladder.png. Tokenized datasets are
torch-independent so reused as-is.
**PRE-LAUNCH CONFIRMATIONS (2026-08-10, user-required before the paid 8-arm launch):**
(1) PARAM-COUNT DISCLOSURE (MANDATORY in report): "@124M" = matched DIMENSIONS
(12L/12H/768d), NOT matched param count. Old-algo GPT-2 = 123,689,472 params;
current-arch modded = 498,773,000 (value-embeds/U-net additions ARE the algorithm
being measured — same config as the completed/accepted study's small A1). Real
counts for BOTH arms MUST appear in report.md Exp A section + era_ladder.png caption
+ a `param_counts` field in era_correction_2x2.json. (2) ALL FOUR original 124M arms
are re-scored on the union eval — the 8-arm re-run RE-TRAINS + union-eval-scores
A0D0(old-owt), A0D1(old-dclm), A1D0(new-owt), A1D1(new-dclm), NOT just A0D0; every
BPB in the corrected 2x2 (threshold + all 4 cells + crossings + Shapley) is on the
SAME union-eval basis; + 4 new C4/RefinedWeb arms = 8 total. (3) HF QUOTA: private
repo hit its storage cap after the backup added owt_gpt2+c4_gpt2 (~36GB) on top of
the 37.7GB completed-study finals — a storage-limit 403 blocks NEW writes only and
does NOT delete/alter existing objects, so prior-tier finals are intact (and also
exist on the volume + in git/results). Era datasets are NOT hosted on HF this round;
irreplaceable eval + era provenance backed up locally to ~/Desktop/era_ladder_backup/.
**EXP A ERA-LADDER COMPLETE (2026-08-11, commit 0f6fafe).** All 8 arms re-ran clean
on torch 2.10 / union eval (8-GPU pod 103.207.149.86:11685). Two data-prep bugs on the
NEW C4/RefinedWeb current-arch arms, both fixed (data-only, no numerics change): (1) own-val
shard ~96k tokens short of the fixed 2,097,152 eval batch -> regenerated 3.5M-tok held-out
val_000000.bin from the *_gpt2 train tail; (2) modded loader's OWT-tuned TRAIN_MAX_NUM_DOCS
overflowed on C4's shorter docs -> train_gpt_ceg.py sizes the doc buffer from a ~64 tok/doc
floor (commit cced40d, inert padding). RESULTS (results/era_correction_2x2.json,
era_ladder_results.json, era_ladder.png): corrected threshold 1.275975 (vs published
1.274421, unchanged); 2x2 data 2.39x / algo 16.10x / total 38.5x (vs 2.23/13.69/30.5) —
shift is union-eval + same-seed variance, NO torch component (both published+corrected are
2.10; torch-neutrality side-check: 2.4.1->2.10 at fixed eval = +0.0156 a0d0 / +0.0007 a0d1,
NOT negligible -> validated running all 8 on one torch). Per-dataset: OWT algo 23.8x; C4
CENSORED under BOTH algos (worse than OWT -> data-quality NON-monotonic in release year);
RefinedWeb data 3.27x/algo 11.6x; DCLM data 3.53x/algo 10.9x. Metrics for all 8 arms backed
up locally ~/Desktop/era_ladder_backup/run_metrics/. NEXT: Exp B B1 (config prep + matched
GPT-2 baselines on idle GPUs + OWT tokenization in neox/smollm2 tokenizers, all IN PROGRESS).
EXP B B1 PREP LAUNCHED (2026-08-12, pod 103.207.149.86:11685): (a) MATCHED GPT-2 BASELINES —
user greenlit ALL 6 candidate sizes. Driver /root/ceg/b1_baselines.sh (train_old, owt_gpt2,
8.87B, union eval, chain-abort keep-3), monitor bgl58yb2y. Matched dims (configs/model_sizes.json
b135/b160/b360/b410/b1400/b1700; param-matched +-1.3%, aspect in GPT-2 range [28-70]):
b135=21L/640/10H, b160=21L/704/11H, b360=19L/1152/18H, b410=28L/1024/16H, b1400=37L/1728/27H,
b1700=42L/1792/28H. --size reads model_sizes.json keys (commit 9753e1e). RECIPE: global batch
524288 fixed; lr 6e-4 small/3e-4 mid/2e-4 large; device-batch b135/b160=32, b360/b410=16,
b1400/b1700=8. LESSON: train_old eager attention mem ~ db*layers*heads (NOT param-bound) —
db64 OOMs the deep-narrow 21L small models even at 135M; scaled db to the b1700@db8 smoke-fit
envelope (all divide 64; grad-accum absorbs, same FLOPs). ~241 GPU-h/~30 wall-clock h/~$600.
(b) OWT TOKENIZATION (CPU) -> owt_neox + owt_smollm2, prepare.py HF path, train-tokens 12B/val 5M;
driver /root/ceg/owt_tokenize.sh, monitor beuzd79c3. train_hf format = single train.bin uint16 +
raw val_text.jsonl (evaluate_bpb_hf re-tokenizes) -> NO C4 val.bin trap; both vocabs fit uint16.
NEXT: re-anchored schedules for the 6 candidates, then Pythia/SmolLM2 vs matched baselines.
**EXP B EXECUTION = GATED PHASES (user-ratified 2026-08-12).** NOT a blind all-sizes/all-arch
launch (this session's bug pattern: every scale/data/arch transition surfaced a distinct bug).
Gate structure: (B1) candidates scale-validated small->mid->large (Pythia-160M + SmolLM2-135M
first = already Gate-1'd; then 360/410M; then 1.4/1.7B), STOP-and-flag on any scale-transition
bug; (Gate B1->B2) user reviews B1 result, then confirm; (B2) build+smoke Mamba/Mamba2 in shared
harness (deps mamba-ssm/causal-conv1d/Triton-SSD; reuses owt_neox) BEFORE full runs; (Gate
B2->B3) confirm or drop; (B3) H3 timeboxed/bespoke, high-risk. MONITORING: hourly heartbeat +
event alerts (combined monitor; user wants periodic status like the earlier tiers).
**EXP B B1 BASELINES COMPLETE + CANDIDATE PAIR LAUNCHING (2026-08-13).**
- ALL 6 matched GPT-2 baselines DONE (186 GPU-h actual, under ~241 est; big models better MFU).
  THRESHOLDS (CEG denominators) in results/b1_baseline_thresholds.json (commit c9021cf):
  b135=1.2616 b160=1.2668 b360=1.2448 b410=1.2437 b1400=1.1896 b1700=1.1792. FLAG: b160
  threshold marginally > b135 despite larger (within same-seed noise; each candidate vs its
  OWN-size baseline). Clean monotonic descent, no divergence/NaN, all arms 8.87B/union eval.
- DURABILITY (user-decided 2026-08-13): metrics+thresholds in GIT (durable, laptop-independent).
  6 baseline CHECKPOINTS (weights) pulled LOCAL to ~/Desktop/era_ladder_backup/b1_checkpoints/
  (b135 0.55 / b160 0.64 / b360 1.45 / b410 1.62 / b1400 5.66 GB done; b1700 ~6.9GB was mid-pull
  via task bo8taan09 -> VERIFY full size, re-pull /workspace/runs/b1_baseline_b1700/ckpt_016925.pt
  if truncated). NOT on HF (private quota still full) or git (too big). owt_neox + owt_smollm2
  datasets local at ~/Desktop/era_ladder_backup/b1_datasets/ + pod. USER: KEEP LOCAL until HF
  quota resolved — CORE eval of baselines planned later, so weights ARE needed. When HF quota
  fixed -> push 6 ckpts + era datasets to HF, then local copies redundant.
- CANDIDATE PAIR LAUNCHING (fork aeee317f, IN PROGRESS at compaction): gated scale-validation =
  Pythia-160M (owt_neox, config configs/hf/pythia_160m.json, crosses b160=1.2668) + SmolLM2-135M
  (owt_smollm2, fork WRITES configs/hf/smollm2_135m.json, crosses b135=1.2616), via
  train_hf/train_hf_ceg.py, global-batch-tokens 2097152 (Gate-1 value), token-budget 8870000000,
  block 1024, --neutral-eval-dir /workspace/datasets/wiki_eval_union, --hf-tokenizer per arch.
  Driver /root/ceg/b1_candidates.sh, out-dirs /workspace/runs/b1_cand_{pythia160m,smollm2_135m}.
  SmolLM2 (Llama) smoked first. NEXT after this validates clean (BPB descends + crosses baseline,
  no divergence): mid candidates (360/410M) -> large (1.4/1.7B), THEN Gate to B2 (Mamba/Mamba2).
- MONITORING GAP: the local heartbeat + backup loops REPEATEDLY DIE on laptop sleep (training on
  the pod is ALWAYS unaffected — detached). After the fork reports the candidate launch, RE-SET
  a completion monitor + metrics/checkpoint backup-to-local for b1_cand_* (same pattern as
  baselines: divcheck + disk + hourly heartbeat; pull metrics every ~10min + final ckpt on
  completion). User advised to run `caffeinate -dimsu` for continuous local monitoring/backup.
- POD 103.207.149.86:11685 torch 2.10, /workspace ephemeral (NOT the network volume). Baseline
  run dirs /workspace/runs/b1_baseline_* still present (ckpts prunable once all 6 confirmed local).
  All 6 baseline ckpts VERIFIED local incl b1700 (6.85GB = pod exact). owt_neox+owt_smollm2 local.
- CANDIDATE PAIR LAUNCHED + TRAINING (2026-08-13; fork aeee317f done; commits a705ba7 smollm2
  config + 4ce1469 driver). Chained driver /root/ceg/b1_candidates.sh, setsid. Common:
  global-batch-tokens 2097152, device-batch 32, grad_accum 8, block 1024, token-budget 8.87e9,
  neutral-eval-dir wiki_eval_union, n-checkpoints 25 keep-3, seed 1234 (divisibility PASS).
  Arm1 PYTHIA-160M (gptneox, 162,322,944 p): --config-json configs/hf/pythia_160m.json --data-dir
  owt_neox --hf-tokenizer EleutherAI/gpt-neox-20b --peak-lr 6e-4 --schedule cosine (Gate-1 solid),
  total_steps 4230 — TRAINING CLEAN (BPB 3.49->3.31 descending, no divergence). Arm2 SMOLLM2-135M
  (llama, 134,515,008 p exact, GQA 9/3, rope_theta 100000): --config-json configs/hf/smollm2_135m.json
  --data-dir owt_smollm2 --hf-tokenizer HuggingFaceTB/SmolLM2-135M --peak-lr 3e-3 --schedule wsd
  --wsd-decay-frac 0.10 — smoked OK, QUEUED after Pythia (~2.5h). Crosses: Pythia->b160=1.2668,
  SmolLM2->b135=1.2616. Logs /workspace/session_logs/{pythia160m,smollm2_135m}.log + status
  b1_candidates.status; out-dirs /workspace/runs/b1_cand_{pythia160m,smollm2_135m}. Candidate
  backup loop = task bedxnpp60 -> ~/Desktop/era_ladder_backup/b1_cand/ (metrics+final ckpts).
  **RECIPE FLAG RESOLVED (2026-08-13): SmolLM2-135M recipe VERIFIED against the OFFICIAL nanotron
  config (huggingface/smollm text/pretraining/smollm2/config_smollm2_135M.yaml) via gh api.
  CONFIRMED: peak LR = 0.003 = the driver's 3e-3 (fork recollection was RIGHT; no LR change).
  CORRECTED WSD shape to the documented recipe: decay fraction = lr_decay_steps 400000 /
  train_steps 2000000 = 0.20 (driver had 0.10); min_decay_lr = 0 -> --min-lr-ratio 0.0 (driver
  had 0.1); decay style linear (matches impl). KEPT --warmup-frac 0.01 = study re-anchoring
  convention (documented lr_warmup_steps 2000 / 2000000 = 0.1% -> ~4 steps on the 8.87B re-anchored
  run = too abrupt at 3e-3; Gate-1/Pythia use 0.01 too). rope_theta KEPT 100000 (matches released
  HuggingFaceTB/SmolLM2-135M config.json; pretraining-time value was 10000 but immaterial at
  block 1024 — flag, not a blocker). The 3e-3 that alarmed the fork is actually SmolLM v1's LR AND
  SmolLM2's — coincidentally the same. IMPLEMENTATION: chain's wrong SmolLM2 line NEUTRALIZED by
  renaming its config to configs/hf/smollm2_135m.json.disabled_for_chain (so it fast-fails ->
  CHAIN_ABORT -> chain exits, NO wasted training); corrected copy at configs/hf/smollm2_135m_v2.json.
  Corrected arm = /root/ceg/b1_smollm2_fixed.sh (--wsd-decay-frac 0.20 --min-lr-ratio 0.0), launched
  AUTONOMOUSLY by /root/ceg/b1_smollm2_launcher.sh (setsid; waits for Pythia DONE exit=0 + chain
  gone + GPUs free) with a FRESH pod-side monitor /root/smollm2_fixed_monitor.sh (the chain monitor
  /root/cand_monitor.sh exits at CHAIN_ABORT). Corrected-arm status /workspace/session_logs/
  b1_smollm2_fixed.status; monitor log smollm2_fixed_monitor.log. SmolLM2 crossing (b135=1.2616) now
  trustworthy once the corrected arm completes; Pythia crossing (b160=1.2668) already fine.**
**PYTHIA-160M CANDIDATE COMPLETE — DID NOT CROSS ITS BASELINE (2026-08-13, STOP-AND-FLAG).**
Full run clean: neutral_bpb 3.49->1.369106 monotonic, no divergence, 9.90 GPU-h, 4230 steps, owt_neox,
gpt-neox-20b tok, cosine peak-lr 6e-4 (documented Pythia-160M LR, faithful). Final ckpt_004230.pt +
metrics backed up LOCAL (~/Desktop/era_ladder_backup/b1_cand/). **RESULT: 1.369 > matched baseline
b160=1.2668 -> Pythia-160M does NOT reach the GPT-2-baseline threshold -> algo CEG <=1x at 160M
(CENSORED crossing, same handling class as ScaleUp-on-OWT@1.5B in the main study).** Consistent with
Gate-1 (owt_gpt2) which hit 1.293 — ALSO above b160; owt_neox run is 0.076 higher (tokenizer/data-
exposure diff at fixed token budget). Recipe verified faithful (Pythia-160M documented peak LR = 6e-4),
so this is a REAL finding not a recipe bug; but it is the FIRST candidate and a censored non-crossing,
so per gate rules DO NOT auto-proceed to mid (360/410M) candidates — needs user review of whether (a)
accept as a genuine small/no-advantage result, or (b) sanity-probe the baseline-vs-candidate fairness
(owt_gpt2 baseline vs owt_neox candidate at fixed 8.87B tokens; each-arch-own-tokenizer is the ratified
design). Corrected SmolLM2-135M arm STILL RUNNING (2nd data point; crosses b135=1.2616 or not TBD).
Fresh local backup loop for corrected SmolLM2 = scratchpad sm2_backup.sh (pulls final ckpt on
B1_SMOLLM2_FIXED_DONE). NO new paid launch pending user decision on the Pythia non-crossing.
B1 BUILD STATUS (2026-08-08, fork): harness=train_hf/train_hf_ceg.py; BPB-over-
HF-tokenizer instrument=eval/bpb_hf.py; Gate-2 check=scripts/gate2_bpb_check.py;
prepare.py HF-tokenizer path added; Pythia-160M cfg=configs/hf/pythia_160m.json.
Committed+pushed (77ad508 instrument, 546930e eval-mode fix, cbe977e harness).
**GATE 2 = PASS, EXACT** (bpb_hf reproduces common/bpb.py ground truth 1.18151583,
|Δ|=0.0 on both math-identity + raw-text-retokenize paths). Caught+fixed a
common/bpb.py-leaves-train()-mode gotcha (dropout skewed the identity check) —
does NOT affect the completed study (its evals set eval() per-eval).
**GATE 1 (Pythia-160M smoke)**: GPTNeoXForCausalLM from cfg, OWT-gpt2, cosine
re-anchored to 8.87B (global batch 2,097,152 tok/step = 1024 seq, /8 = 128/GPU
CLEAN divides), peak LR 6e-4, warmup 1%. Runs first 600 of ~4229 steps (max-steps
caps loop; schedule stays anchored to full budget). Through step 178: train_loss
AND neutral_bpb co-descend monotonically (7.01->5.33 / 2.495->2.146), NO
divergence signature. Completing to step 600 under monitor. CAVEAT: smoke covers
the EARLY portion (warmup + early descent) — not the late cosine decay; catches
early-divergence well, not late-convergence. NEXT (pending user): accept early-
clean+Gate2 as sufficient, OR full-length 160M run first, before building out the
6 sizes x arches + matched baselines.

## Current state (2026-07-16)

## CORE scoring for the new arms — DONE (2026-07-25)

CORE-subset now covers all four scales (was 124M/355M current-arch only). Added
1.5B (A0D0/A0D1 GPT-2-XL + A1D0/A1D1 ScaleUp) and ScaleUp-124M (A1D0/A1D1);
ScaleUp-124M A0 reuses the existing 124M GPT-2 A0 CORE (same small model — CORE
quality is scale/data/arch-dependent, not GPU-count). New plain-causal adapter
= **eval/lm_eval_adapter_scaleup.py** (exec-prefix loads GPT from
train_gpt_xl_ceg.py at cut marker `CONFIG = ceg.CONFIG`, plain state_dict load,
eager fp32, forwards full seq with dummy targets for full logits; single source
of truth, no code dup). GPT-2 arms use eval/lm_eval_adapter.py (verified
size-agnostic: loads XL dims 48/25/1600 from ckpt["config"]). Both smoke-passed.
Sweep: 15 tasks limit 500, 6 finals across 5 GPUs, all exit 0.
RESULTS (14 final CORE JSONs consolidated into results/core_finals/; gate
results/core_gate_v2.json — the sprawling per-checkpoint core_sweep*/ dirs were
pruned 2026-07-30 after finals were fixed, gate output byte-identical):
124M 5 usable / 355M 6 / 1.5B 6 / ScaleUp-124M 5 (bit-identical for 124M/355M —
edit was additive). Arms cluster near noise; slightly more spread at 1.5B but
still noise-limited — no quantitative CORE CEG claimed (BPB primary, CORE a
sanity gate). **lambada** = open-vocab (chance=None) so never gated at any scale;
but ScaleUp A1 has a REAL logits path so its lambada is a valid diagnostic
(acc 0.32@124M -> 0.52/0.55@1.5B, ppl 55->8; confirms adapter soundness) —
core_gate.py MODDED_A1_SCALES marks lambada invalid only for 124M/355M modded A1.
report.md CORE section + analysis/make_report.py extended to all 4 scales.
Committed 5d168a6 (results/gate/report) + a33b27f (adapter). GPUs idle.

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
- HF repo: **MIRIBerkeley/data-vs-algorithms-ceg** (private; MOVED 2026-07-28
  from baileymachihirota/… which now redirects). 54 files / 37.7 GB: full 2x2
  matrix finals (4 arms each) + eval_corpus. **Folders REORGANIZED 2026-07-28 to
  an explicit-flat scheme** naming curve + scale (server-side path move, no
  re-upload): `current-arch-124M/`, `current-arch-355M/` (current modded A1,
  8-GPU), `scaleup-124M/`, `scaleup-1.5B/` (2024-ScaleUp A1, 5-GPU) — replacing
  the old `124M/ 355M/ 124M-scaleup/ 1.5B/`. Each curve is internally
  hardware-consistent; the two are never mixed in one GPU-hour ratio. Repo README
  updated to the two-curve scheme. scripts/hf_upload.py derives the namespace
  from whoami() by default — set **HF_NAMESPACE=MIRIBerkeley** (and --size to a
  new folder name) to upload new arms to the canonical repo. Eval corpus frozen:
  /workspace/datasets/wiki_eval, sha256 cbdd72ac…, never regenerate.
