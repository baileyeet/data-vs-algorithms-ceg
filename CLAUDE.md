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
**BOTH CANDIDATES COMPLETE — BOTH CENSORED NON-CROSSINGS (2026-08-13, STRONGER STOP-AND-FLAG).**
Corrected SmolLM2-135M done clean: neutral_bpb 3.x->1.328436, 19.15 GPU-h, 4230 steps, WSD decayed
to ~0 (lr 3.58e-06 at end, confirms decay-frac 0.20 + min-lr 0.0), no divergence. Final ckpt_004230.pt
+ metrics LOCAL (~/Desktop/era_ladder_backup/b1_cand/, verified 538MB). PAIR RESULT vs matched
GPT-2 baselines:
  Pythia-160M   1.369  vs b160 1.2668  -> gap +0.102 (NO cross)
  SmolLM2-135M  1.328  vs b135 1.2616  -> gap +0.067 (NO cross)
BOTH above baseline -> algo CEG <=1x at 135-160M for BOTH lineages (censored). INTERNAL consistency:
SmolLM2 (2024 Llama/GQA/SwiGLU) 1.328 < Pythia (2023 GPTNeoX) 1.369 = better arch -> lower BPB, so the
instrument IS ordering arches sensibly. But a UNIFORM ~0.07-0.10 offset above baseline across two
DIFFERENT arches + tokenizers (neox vs smollm2) + schedules (cosine vs WSD) points to a SHARED
systematic factor, not two independent arch weaknesses. LEADING HYPOTHESIS: the baselines were trained
with train_old (owt_gpt2) but the candidates with train_hf (owt_neox/owt_smollm2) — DIFFERENT TRAINERS;
if train_hf trains a GPT-2-equivalent to a ~0.07-0.10 worse BPB than train_old (compile/attn-impl/
optimizer-detail differences), every candidate is penalized by the harness, not the architecture.
Removing a ~0.08 offset: SmolLM2 1.328-0.08=1.25 < 1.2616 (WOULD cross, small 2024 advantage); Pythia
1.369-0.08=1.29 > 1.2668 (still no cross) — a far more believable Exp B picture. PROPOSED CHEAP CHECK
(paid, ~$5-8/~1h, NEEDS USER CONFIRM): train a GPT-2 at b135 dims THROUGH train_hf, compare to the
train_old b135 threshold 1.2616. Match -> non-crossings are real architecture (accept/report). ~0.07-0.10
worse -> HARNESS CONFOUND -> re-derive baselines via train_hf (or run both arms through one trainer)
before any mid/large candidates. DO NOT proceed to mid (360/410M) candidates until this is resolved.
Gate-2 (BPB instrument) was validated exact earlier; what was NEVER head-to-head validated is
train_old-vs-train_hf training EQUIVALENCE on an identical GPT-2 config.
**HARNESS-EQUIVALENCE CHECK LAUNCHED (2026-08-13, user-approved).** GPT-2 @ b135 dims (21L/640/10H,
vocab 50304, dropouts ZEROED to match nanoGPT pretraining) run through the CANDIDATE pipeline
(train_hf, owt_gpt2 + gpt2 tokenizer, union eval, global-batch 2097152, cosine peak-lr 6e-4,
warmup-frac 0.01, min-lr 0.1, wd 0.01, 8.87B) to compare final neutral_bpb against the train_old
b135 baseline threshold 1.2616. This tests the AGGREGATE pipeline difference (trainer AND global batch
524288->2097152 AND warmup 700steps->0.01 AND wd 0.1->0.01 all differ between baseline and candidates).
Added gpt2 to train_hf build_model registry (GPT2Config/GPT2LMHeadModel); config configs/hf/gpt2_b135.json
(commit 1b471c2). Param count 136,245,120 (HF) vs 135,589,760 (train_old) = +655,360 = 1x wpe accounting,
immaterial. Smoke PASSED (10 steps, loss+bpb descend, bpb_hf(gpt2) sane 3.51). Driver /root/ceg/
gpt2_b135_harness.sh; monitor /root/gpt2_harness_monitor.sh -> gpt2_harness_monitor.log; divcheck
/root/divcheck_gpt2harness.py; status gpt2_b135_harness.status; out-dir /workspace/runs/gpt2_b135_harness;
local backup scratchpad gpt2_harness_backup.sh -> ~/Desktop/era_ladder_backup/b1_cand/. ~10-20 GPU-h/
~$5-8/~2h. INTERPRETATION: match 1.2616 -> candidate non-crossings are REAL architecture (accept/report
Pythia+SmolLM2 as small/no-advantage at 135-160M); ~0.08 worse -> PIPELINE CONFOUND.
**USER GUIDANCE (2026-08-13, if confound confirmed): DO NOT assume a flat correction. Before
re-deriving all baselines via train_hf, sanity-check whether the b135 offset MAGNITUDE holds at other
sizes (b160 next, then mid/large dims) or VARIES by architecture shape — the correction may not be
size/shape-invariant. So the follow-up would be at least a b160 harness check too, and comparing the
two offsets, before any blanket baseline re-derivation.**
**HARNESS CHECK RESULT = CONFOUND CONFIRMED (2026-08-13).** GPT-2 @ b135 via train_hf (candidate
pipeline) final neutral_bpb = 1.320257 (clean, 19.2 GPU-h, no divergence). train_old b135 = 1.2616.
OFFSET = +0.0587 — identical GPT-2 arch is 0.059 BPB WORSE purely from the train_hf pipeline. So the
train_old baselines are NOT the correct denominator for the train_hf candidates. CORRECTED (same-pipeline)
reading at ~135M: GPT-2(train_hf) 1.320 vs SmolLM2-135M(train_hf) 1.328 = SmolLM2 only 0.008 worse
(NOISE) -> SmolLM2-135M ~= GPT-2 = PARITY / no measurable algo advantage (NOT the 0.067 deficit the raw
vs-train_old numbers implied; the confound was inflating the apparent deficit). Pythia-160M 1.369 needs
the b160 GPT-2-via-train_hf denominator for a clean read (different size AND shape; per user guidance do
NOT assume the b135 0.0587 offset transfers) — rough subtraction leaves Pythia still ~0.04 worse even
corrected, but the b160 harness point is required to state it. Final metrics+ckpt local
(~/Desktop/era_ladder_backup/b1_cand/). NEXT (paid, needs confirm): GPT-2 @ b160 via train_hf ->
(1) fair denominator for Pythia; (2) 2nd offset to test size-invariance of the correction. THEN decide:
if offsets ~equal -> correction roughly size-stable at small scale; if they differ -> per-size train_hf
baseline re-derivation before any mid/large candidates. DO NOT proceed to mid (360/410M) candidates
until the denominator question is settled.
**ROOT-CAUSE DIAGNOSIS of the +0.0587 (2026-08-13, user-requested).** Compared train_hf vs train_old
settings on the identical GPT-2 config. IDENTICAL in both (ruled out): bf16 autocast, tf32 "high",
AdamW betas 0.9/0.95, grad-clip 1.0, wd param-grouping, AND torch.compile (BOTH ran compile=False =
eager). DIFFERENCES (run_config-confirmed): (1) PRIMARY = GLOBAL BATCH 524288 (baseline) vs 2097152
(candidate) = 4x, at the SAME peak LR 6e-4 -> 16925 vs 4230 optimizer steps over the same 8.87B tokens
= 4x fewer weight updates = large-batch-at-unscaled-LR UNDERTRAINING (classic effect; almost certainly
the bulk of 0.059). (2) SECONDARY = weight decay 0.1 (baseline) vs 0.01 (candidate) = less regularized,
same direction. (3) MINOR = warmup 4.1% vs 1%. CONCLUSION: the confound is a RECIPE/config mismatch,
NOT train_hf-code-is-worse -> FIXABLE at source. ROBUST FIX (better than a standing correction; de-risks
Mamba/Mamba2 which share train_hf): standardize ALL Exp B on the baseline's converged recipe (global
batch 524288 + wd 0.1 + warmup ~4%), re-run the 2 candidates via train_hf at 512k (~same GPU-h: same
tokens/FLOPs, 4x more steps) -> existing train_old baselines become valid denominators, NO 6-baseline
re-derivation. DEFINITIVE ISOLATION (optional, 1 run): GPT-2 @ b135 via train_hf at 512k/wd0.1 -> hits
~1.2616 => train_hf==train_old, confound = 100% batch/wd (dissolves); still ~1.32 => residual is
train_hf code. b160 harness (running) still adds the 2nd offset point + Pythia 2M-denominator, useful
under either decision. COST FLAG (user): if b160 offset != b135 offset, the vs-2M-pipeline correction is
size-dependent => all SIX baselines re-derived via train_hf (~$30-50), not two — but the batch-match fix
avoids that path entirely (reuses train_old baselines).
**BOTH HARNESS OFFSETS IN — SIZE-STABLE (2026-08-14).** b160 GPT-2 via train_hf (2M pipeline) final
neutral_bpb = 1.322368 (clean, 21.1 GPU-h). Offsets: b135 = 1.320257-1.2616 = +0.0587; b160 =
1.322368-1.2668 = +0.0556; Δ = 0.003 = SIZE-STABLE (~0.057). So the vs-2M-pipeline correction is NOT
size-dependent -> the all-6-baselines/$30-50 cost flag does NOT trigger. CORRECTED SAME-PIPELINE (2M)
CANDIDATE READS (candidate vs its matched train_hf GPT-2 denominator): SmolLM2-135M 1.328 vs 1.320 =
+0.008 PARITY (noise); Pythia-160M 1.369 vs 1.322 = +0.047 REAL DEFICIT. Internal SmolLM2(1.328)<
Pythia(1.369) = 2024>2023 consistent. NET B1 (2M regime): neither small modern transformer beats a
matched GPT-2 at 135-160M — SmolLM2 parity, Pythia deficit; NO algo advantage. b160 metrics+ckpt local
(644MB). CAVEAT: this result is in the UNDERTRAINED 2M-batch regime (4x fewer updates) -> differences
COMPRESSED; the relative parity/deficit may shift under proper 512k convergence. DECISION PENDING (user):
(A) accept the 2M-regime matched-denominator result as-is ($0, but undertrained + needs train_hf GPT-2
denominators at mid/large too if we stay 2M); (B) STANDARDIZE Exp B on the converged 512k recipe (global
batch 524288 + wd 0.1 + warmup ~4%), re-run the 2 small candidates via train_hf @ 512k (~29 GPU-h/
~$15-25), reuse the existing 6 train_old baselines as denominators for ALL sizes, run mid/large candidates
@ 512k -> better-resolved + no per-size harness re-derivation + de-risks Mamba. Recommend B (root-cause
fix, better science for the still-to-come mid/large curve). Optional 1 confirmatory GPT-2 @ b135 via
train_hf @ 512k/wd0.1 -> if ~1.2616 proves train_hf==train_old (confound = 100% recipe). Network blip
during b160 was LOCAL laptop only; pod unaffected, run clean.
**CONFIRMATORY RESULT (2026-08-14): RESIDUAL +0.0105 -> PLAN CHANGE.** GPT-2 @ b135 via train_hf at the
EXACT b135 baseline recipe (512k batch, wd 0.1, warmup 700 steps, cosine 6e-4, budget 8.87B/16919 steps)
tail-mean neutral_bpb = 1.272076 (n=5). vs train_old b135 = 1.2616 -> RESIDUAL +0.0105. So batch/wd/warmup
fixed 0.0482 (82%) of the 0.0587 2M-gap; a +0.0105 train_hf-vs-train_old residual REMAINS at matched recipe
(HF GPT2LMHeadModel class + EpochShuffledLoader vs nanoGPT). CRITICAL: +0.0105 ~= the candidate SIGNAL size
(SmolLM2 was +0.008) -> train_old baselines CANNOT serve as denominators for train_hf candidates (would bias
each result by ~a full signal, could flip SmolLM2 parity<->deficit). So the "reuse the 6 train_old baselines"
plan is DEAD. CORRECT METHODOLOGY = like-for-like: each candidate vs a GPT-2 trained through the IDENTICAL
train_hf pipeline @512k (zeroes BOTH batch confound AND residual; same trainer/batch, only arch differs).
Have it at b135 = 1.2721. The (initial) crash was the no-silent-wrap data guard (needed 346k tok > owt_gpt2's
8.8732B at 512k/16925 steps); fixed with budget 8.870B/16919 steps (matches other train_hf runs, -6 steps
negligible). Confirmatory ckpt+metrics local. **REVISED PLAN (needs user confirm — cost changed): adopt
GPT-2-via-train_hf@512k as THE Exp B denominator at each size. IMMEDIATE (3 runs ~$25-30): (1) GPT-2-train_hf
@512k @ b160 = Pythia's denominator; (2)+(3) re-run Pythia + SmolLM2 @512k (their documented recipes, batch
-> 512k). Then B1 = like-for-like at 512k. LATER (mid/large gates): GPT-2-train_hf@512k denominator at b360/
b410/b1400/b1700 = 4 more runs. TOTAL train_hf denominators = 6 (b135 done, 5 to go), REPLACING the role of
the 6 train_old baselines (which stay as a cross-check, differing by the ~0.0105 residual). This IS the
all-six-denominators cost the user asked be visible — now justified by the residual, not assumed.**
**RESIDUAL +0.0105 DIAGNOSED via code inspection (2026-08-14, user-requested — no fixable one-liner).**
Compared common/model_gpt2.py (train_old GPT) vs HF GPT2LMHeadModel: attention BOTH sdpa (F.scaled_dot_
product_attention is_causal); init BOTH std 0.02 + residual x1/sqrt(2L); optimizer BOTH AdamW betas
0.9/0.95 eps 1e-8 (only diff = train_old fused=True vs train_hf fused=False = negligible ~1e-6 numerics);
LR schedule IDENTICAL formula (linear warmup + 0.5(1+cos) to min-ratio); biases + LayerNorm eps 1e-5 +
gelu_new activation all MATCH. Param-count gap (136.2M HF vs 135.6M train_old) = REPORTING ARTIFACT
(model_gpt2 line 106 excludes wpe from its printed count; identical model). => NO identifiable
architectural/optimizer/schedule cause. Residual is most consistent with DATA-LOADER ORDERING (train_hf
EpochShuffledLoader vs train_old loader read the SAME owt_gpt2 tokens in different order -> different SGD
trajectory) and is 1.05x the study ±0.01 same-seed NOISE FLOOR. REPORT SENTENCE: "residual +0.0105
investigated — arch/init/schedule/optimizer verified matched; at same-seed noise floor, attributed to
loader ordering, not a systematic defect; cancels under like-for-like anyway." Not chased with paid runs
(a different-seed re-run would prove noise vs ~1sigma systematic, but like-for-like cancels it). 
**512k CHAIN LAUNCHED (2026-08-14).** Driver /root/ceg/b1_512k_chain.sh (setsid, chain-abort), monitor
/root/monitor_512k.sh -> monitor_512k.log, divcheck /root/divcheck_512k.py, status b1_512k_chain.status,
backup scratchpad chain512k_backup.sh -> ~/Desktop/era_ladder_backup/b1_cand/. 3 arms @ 512k/16919 steps,
budget 8.870B, union eval, only global batch held fixed (524288) each arch keeps documented recipe:
r512k_gpt2b160 (GPT-2 denom, gpt2_b160.json, owt_gpt2, wd0.1/warmup0.0414/6e-4 cosine) -> r512k_pythia160m
(pythia_160m.json, owt_neox, 6e-4 cosine/warmup0.01/wd0.01) -> r512k_smollm2 (smollm2_135m_v2.json,
owt_smollm2, 3e-3 WSD 0.20/min-lr0/warmup0.01/wd0.01). ~9h. On finish: b160 denom = Pythia's threshold;
compare SmolLM2 vs b135 denom (=1.2721 confirmatory) + Pythia vs r512k_gpt2b160; tail-mean each.
**512k RESULTS (2 of 3 in, 2026-08-14):** r512k_gpt2b160 (b160 GPT-2 denom, train_hf) tail-mean 1.251829;
r512k_pythia160m tail-mean 1.333728 -> Pythia deficit +0.0819 (CONVERGED, like-for-like). CLEAR no-advantage/
real deficit for Pythia-160M. LARGER than the 2M-regime +0.047 -> see compression note. NOTE b160 denom
train_hf 1.2518 came out BELOW train_old 1.2668 = the train_hf-vs-train_old gap FLIPPED SIGN (+0.0105 b135,
-0.0150 b160) = EMPIRICAL confirmation the +0.0105 residual is NOISE, not a systematic. SmolLM2 arm running.
**B1 SMALL-SCALE COMPLETE (2026-08-14, converged 512k like-for-like, pre-registered rule applied).**
Denoms (GPT-2 train_hf @512k): b135=1.272076 (confirmatory), b160=1.251829. Candidates:
  SmolLM2-135M 1.280267 vs b135 1.272076 -> Delta=+0.0082 (<1sigma) = PARITY within noise (one seed final;
    matches 2M estimate +0.008 -> robust).
  Pythia-160M 1.333728 vs b160 1.251829 -> Delta=+0.0819 (>2sigma) = REAL DEFICIT (significant).
Internal SmolLM2(1.280)<Pythia(1.334) = 2024 Llama beats 2023 GPTNeoX, consistent. HEADLINE: neither small
modern transformer shows an algo advantage over a matched well-trained GPT-2 at 135-160M — SmolLM2 parity,
Pythia significantly worse. (Contrast: completed-study current-arch/ScaleUp had 13.7x/2.9x algo advantage at
124M — DIFFERENT arches.) Compression corroborated: SmolLM2 (already parity) unchanged 2M->512k; Pythia
(real deficit) grew 0.047->0.082 on convergence. All 3 arms' ckpts+metrics local (~/Desktop/era_ladder_backup/
b1_cand/, 645/649/538MB). GPUs idle. NEXT = B1 MID candidates (360/410M) — GATED, needs user go (paid):
each needs its matched GPT-2 train_hf @512k denominator (b360, b410) + the candidate (Pythia-410M owt_neox,
SmolLM2-360M owt_smollm2) = ~4 runs. Same 512k pipeline + pre-registered rule apply. Then LARGE (1.4/1.7B),
then Gate B1->B2 (user review) before Mamba/Mamba2.
**B1 MID CHAIN + SmolLM2 REPLICATE LAUNCHED (2026-08-14).** Driver /root/ceg/b1_512k_mid.sh (setsid,
chain-abort), monitor /root/monitor_512k_mid.sh -> monitor_512k_mid.log, status b1_512k_mid.status, divcheck
/root/divcheck_512k.py (globs r512k_*), backup scratchpad mid512k_backup.sh. 5 arms @512k/16919 steps/8.870B/
union eval/db16 (repl db32), only global batch fixed 524288, each arch documented recipe. Fit-smoked
Pythia-410M+SmolLM2-360M @db16 OK (no OOM). Param-matched pairs: SmolLM2-360M 361.8M<->b360 362.0M;
Pythia-410M 405.3M<->b410 405.3M. Order: (1) r512k_gpt2b360 GPT-2 denom lr3e-4/wd0.1/warmup0.0414;
(2) r512k_gpt2b410 same; (3) r512k_smollm2_360m lr3e-3 wsd0.20/min0/warmup0.01/wd0.01 (rope_theta 100000
released); (4) r512k_pythia410m lr3e-4 cosine/warmup0.01/min0.1/wd0.01; (5) r512k_smollm2_seed2 = SmolLM2-135M
REPLICATE seed 2024 (orig was 1234) to confirm the +0.008 parity across seeds (user-requested; mid NOT blocked
on it -> appended last, 8-GPU consistency kept vs a GPU-split). Configs configs/hf/{gpt2_b360,gpt2_b410,
pythia_410m,smollm2_360m}.json committed 662fcf3. ~15h. ON FINISH apply pre-registered rule: SmolLM2-360M vs
r512k_gpt2b360, Pythia-410M vs r512k_gpt2b410; replicate: 2-seed SmolLM2-135M mean vs 1.2721 (2-seed sig at
|mean|>=0.018). Then LARGE (b1400/b1700 + Pythia-1.4B/SmolLM2-1.7B), then Gate B1->B2.
**B1 MID COMPLETE + SmolLM2 DIVERGENCE FLAG (2026-08-15).** All 5 arms exit 0, ckpts+metrics local. Denoms
(GPT-2 train_hf @512k tail-mean): b360=1.240024, b410=1.238212 (+ b135=1.272076). RESULTS (pre-registered
rule):
  Pythia-410M 1.261889 vs b410 -> Delta=+0.0237 = no advantage (slight deficit, 1-2sigma); CLEAN run
    (neutral rise-from-min +0.003). Pythia deficit SHRINKS w/ scale: 160M +0.082 -> 410M +0.024.
  SmolLM2-360M 1.293791 vs b360 -> Delta=+0.0538 = DEFICIT (>2sigma). **FLAG: DIVERGENCE SIGNATURE** —
    own-val keeps dropping (0.98->0.91) while neutral BPB plateaus then RISES (min 1.2795 -> last 1.2977,
    rise +0.018); model overfits OWT as neutral generalization stalls. divcheck didn't alarm (<0.05 thresh)
    but signature clear. Deficit ROBUST to handling (min-based Delta=+0.040 still >2sigma) so verdict holds,
    but the magnitude is confounded.
  SmolLM2-135M REPLICATE: seed1 1.280267, seed2(2024) 1.281817 (agree to 0.0015!), 2-seed mean 1.281042 vs
    b135 -> Delta=+0.0090 < 0.018 = PARITY CONFIRMED (not lucky-landing). seed2 fully clean (rise +0.000),
    seed1 mild (+0.009).
DIVERGENCE SCALES WITH SmolLM2 SIZE: 135M-seed2 +0.000, 135M-seed1 +0.009, 360M +0.018 -> SmolLM2's
documented LR 3e-3 does NOT transfer to the re-anchored 8.87B/512k setting; progressive OWT-overfitting as
capacity grows. Pythia (LR 3e-4) CLEAN throughout. HEADLINE ROBUST: NO architecture beats matched GPT-2
anywhere in 135-410M (SmolLM2 135M parity else deficit; Pythia deficit shrinking w/ scale) — holds even on
best/min neutral BPB. But SmolLM2's precise deficits are confounded by the recipe-transfer divergence, and
**SmolLM2-1.7B (large) would likely diverge WORSE** (no-invented-recipe rule forbids just lowering its LR).
DECISION PENDING (user) before LARGE tier: how to handle SmolLM2 recipe-transfer/divergence at 1.7B. All
metrics+5 ckpts local ~/Desktop/era_ladder_backup/b1_cand/. GPUs idle.
**LARGE TIER: SmolLM2-1.7B RECIPE CHECK + LAUNCHED (2026-08-16).** User pre-condition check RESOLVED: the
SmolLM2-1.7B documented recipe (nanotron config_smollm2_1B.yaml = 1.7B: hidden 2048/24L/32H MHA/interm 8192)
uses learning_rate 5e-4 (6x LOWER than the 3e-3 that diverged at 135M/360M!) + WSD 10% decay + min-lr 0. So
the LR-transfer/divergence issue likely does NOT apply at 1.7B on its OWN documented recipe -> the disclosed
gap does NOT stand; run SmolLM2-1.7B @5e-4 (NOT invented — its real recipe). User approved FULL large tier.
Driver /root/ceg/b1_512k_large.sh (setsid, chain-abort), monitor /root/monitor_512k_large.sh (TIGHTENED
SmolLM2-1.7B watch: alerts if neutral rise-from-min >= +0.02, below coarse 0.05, + logs own-val trend),
status b1_512k_large.status, backup scratchpad large512k_backup.sh. 4 arms @512k/16919 steps/8.870B/db8,
param-matched: Pythia-1.4B 1.415B<->b1400 1.415B; SmolLM2-1.7B 1.711B<->b1700 1.711B. Both fit-smoked @db8 OK.
Order (SmolLM2 LAST so a fallback does not touch others): (1) r512k_gpt2b1400 lr2e-4/wd0.1/warmup0.0414;
(2) r512k_gpt2b1700 same; (3) r512k_pythia1_4b lr2e-4 cosine/warmup0.01/min0.1/wd0.01; (4) r512k_smollm2_1_7b
lr5e-4 wsd0.10/min0/warmup0.01/wd0.01 (rope 130000). Configs committed 8a7fd30. FALLBACK: if SmolLM2-1.7B
@5e-4 STILL shows the own-val-down/neutral-up signature -> stop + disclose the 1.7B gap. ETA multi-day (1.4-1.7B
@ 8.87B ~15-20 GPU-h/arm x4 ~= 2.5-3.5 days). ON FINISH apply pre-registered rule: Pythia-1.4B vs r512k_gpt2b1400,
SmolLM2-1.7B vs r512k_gpt2b1700. Then B1 COMPLETE (small+mid+large) -> Gate B1->B2 (user review) before Mamba/Mamba2.
**LARGE ARM1 OOM + FIX (2026-08-16).** First large launch CHAIN_ABORTed: r512k_gpt2b1400 OOM'd at db8
(77.94/79.18GB used, needed +1.5GB). CAUSE: I fit-smoked Pythia-1.4B/SmolLM2-1.7B (both 24L) but NOT the
GPT-2 denoms b1400(37L)/b1700(42L) — DEEPER => higher activation mem => OOM at db8. FIX: per-arm device-batch
in b1_512k_large.sh — deep GPT-2 denoms db4 (accum16, re-smoked @db4 OK), Pythia/SmolLM2 db8 (accum8);
global batch 524288 preserved (numerically identical, just more accum). Relaunched clean (driver 103070,
GPUs 100%, arm1 db4/accum16/16919 steps training). Monitor 103144 + backup relaunched. LESSON: smoke EVERY
distinct config shape (depth matters for mem, not just param count).
**LARGE RESULTS (3/4 in, 2026-08-17).** Denoms (GPT-2 train_hf @512k tail-mean): b1400=1.178475,
b1700=1.183664 (1.4/1.7B within 0.005 = noise-flat). Pythia-1.4B 1.208745 vs b1400 -> Delta=+0.0303 =
DEFICIT (>2sigma). PYTHIA FULL CURVE (all deficits, none cross): 160M +0.082 -> 410M +0.024 -> 1.4B +0.030
= deficit shrinks then STABILIZES at ~0.02-0.03; NO algo advantage at any scale. r512k_smollm2_1_7b RUNNING
(final arm, @documented 5e-4, tightened divergence watch; threshold b1700=1.1837: adv<=1.158 parity 1.171-1.197
deficit>=1.210). Denominator trend monotonic-ish w/ scale: 1.272/1.252/1.240/1.238/1.178/1.184.
**B1 COMPLETE (2026-08-18). SmolLM2-1.7B DIVERGED EVEN AT 5e-4.** tail-mean 1.303774 min 1.256001 vs b1700
1.183664 -> Delta(tail)=+0.120 / Delta(min)=+0.072 = DEFICIT either way. DIVERGENCE +0.0423 rise-from-min
(own-val 2.80->0.84 while neutral 1.256->1.298); watch fired 36 alerts. KEY: lower documented LR did NOT
prevent divergence at 1.7B — it is WORSE than 360M (+0.042 vs +0.018). So SmolLM2's divergence is a
big-model + limited-budget (8.87B) OWT-overfitting problem, NOT just the 3e-3 LR; present even at each size's
documented LR, scaling with size (135M ~0 -> 360M +0.018 -> 1.7B +0.042). Run completed (monitor alerts, no
auto-kill); result USED with divergence caveat (deficit robust to min/tail).
=== FULL B1 RESULT (converged 512k, like-for-like train_hf GPT-2 denominators, pre-registered rule) ===
PYTHIA (GPTNeoX 2023): 160M +0.082 DEF | 410M +0.024 no-adv | 1.4B +0.030 DEF -> never crosses; deficit
  shrinks then stabilizes ~0.03; CLEAN runs throughout.
SmolLM2 (Llama 2024): 135M +0.009 PARITY (2-seed confirmed, clean) | 360M +0.054 DEF (div +0.018) | 1.7B
  +0.072min/+0.120tail DEF (div +0.042) -> best=135M parity; deficits grow w/ size, CONFOUNDED by divergence.
HEADLINE: NEITHER modern Transformer-lineage arch beats a matched well-trained GPT-2 anywhere 135M-1.7B
  (best = SmolLM2-135M parity; all else deficit). Contrast completed study's current-arch/ScaleUp = 13.7x/2.9x
  algo advantage @124M. All 6 denom + 6 candidate + 2 replicate metrics/ckpts local. GPUs idle.
=== GATE B1->B2 (user review REQUIRED before building Mamba/Mamba2) ===
NEXT: user reviews B1, then confirm -> B2 = build+smoke Mamba/Mamba2 in shared train_hf harness (deps
mamba-ssm/causal-conv1d/Triton-SSD; reuses owt_neox + gpt-neox-20b tok; matched GPT-2 denoms b1XX already
exist as train_hf @512k). SmolLM2 divergence lesson: watch SSM candidates for the same OWT-overfitting.
**B1 FINALIZATION (user-required before final, 2026-08-18). NARRATIVE (report, explicit not just table):
this is the direct empirical test of Peter's Exp-B question — does current-arch's small-scale speedrun
advantage GENERALIZE beyond small-scale-optimized tricks? Answer = NO (no published lineage beats matched
GPT-2 135M-1.7B). FIGURE DONE: results/b1_cross_scale.png (analysis/plot_b1.py, results/b1_results.json,
commit cf62239) — Delta-vs-scale per lineage; FILLED=clean, OPEN+best->tail bar=divergence-confounded
(SmolLM2 360M/1.7B); parity line + ±1sigma band + 'advantage region (nothing observed)'; Peter narrative in
caption. V1 CLOSED (pipeline equivalence at LARGE scale): train_hf denoms vs the existing train_old baselines
are both GPT-2 @512k at every size -> residual profile b135..b1700 = +0.011/-0.015/-0.005/-0.006/-0.011/
+0.005 = all within ±0.015, SIGN-VARYING, incl 1.4B(-0.011)/1.7B(+0.005) -> residual is noise at ALL scales,
Pythia's clean large numbers trustworthy (not extrapolated). NO new compute (existing data). V2 RUNNING:
Pythia-410M replicate seed 2024 (r512k_pythia410m_seed2, driver+monitor launched; ~5h) to confirm the
near-noise +0.024. AFTER V2: (1) update figure 410M to 2-seed mean; (2) UPLOAD everything to git (configs +
metrics.csv + plot_b1.py + figure) AND HF private repo (final ckpts: 6 candidates + 6 denoms + replicates),
CONFIRM each upload (do not assume). **HF CAVEAT: private repo MIRIBerkeley/data-vs-algorithms-ceg hit its
storage CAP earlier (403 on new writes; decision was keep-local-until-quota-resolved). 12 large ckpts ~40GB
will likely 403 -> attempt + report honestly, may be blocked pending quota fix.** B2 (Mamba/Mamba2) HELD
until V2 closes.
**METHODOLOGY NOTE #1 — UNDERTRAINING BIASES TOWARD PARITY (not just adds noise).** The 2M-batch regime
(4x fewer updates) systematically COMPRESSED candidates toward their GPT-2 denominators: Pythia deficit
0.047 (2M) -> 0.082 (converged 512k). Undertraining pulls BOTH arch and baseline toward each other, biasing
the measured algo gap TOWARD THE NULL (parity). So undertraining is NOT conservative for detecting an
advantage — it risks FALSE NEGATIVES ("no difference" that is really undertraining). => all CEG arms MUST be
compared at convergence; report this explicitly. **METHODOLOGY NOTE #2 — PRE-REGISTERED SIGNIFICANCE RULE
(fixed 2026-08-14 BEFORE seeing SmolLM2, applies to all B1/B2/B3 candidates).** Noise sigma=0.013 BPB on the
candidate-minus-denominator gap Delta (empirical: RMS of the two sign-flipped same-arch GPT-2 cross-pipeline
diffs +0.0105/-0.0150; = study's +-0.01 same-seed floor x sqrt2 for a difference). BANDS: |Delta|>=0.026
(2sigma) = REAL effect (advantage if Delta<0, deficit if Delta>0), one seed; |Delta|<0.013 (1sigma) = PARITY
within noise, one seed final; 0.013<=|Delta|<0.026 (1-2sigma) = gray zone -> SECOND SEED **only on the
advantage side** (Delta in -0.026..-0.013), because only there does the science conclusion turn on sub-noise
precision (a gray-zone DEFICIT still = "no advantage", no second seed). 2nd seed halves mean-noise to ~0.009
(2-seed significance at |mean|>=0.018). For SmolLM2 vs 1.2721: advantage if <=1.246, parity if 1.259-1.285,
second-seed only if 1.246-1.259, real deficit if >=1.298.
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
- **HF ACTUAL STATE (CORRECTED 2026-08-18 via huggingface_hub repo_info — the raw tree API
  gave a BUGGY PARTIAL read that wrongly showed only 10.8GB/no-scaleup; ignore that):
  86 files / ~88.2 GB, NEAR CAP. Holds the FULL completed study: current-arch-124M/ +
  current-arch-355M/ + scaleup-124M/ + scaleup-1.5B/ (four 6.2GB ckpts). B1 UPLOAD (2026-08-18):
  10/14 exp-b-b1/ ckpts uploaded+VERIFIED; 4 largest (b1700-denom, pythia-1.4B, smollm2-1.7B,
  pythia-410M-seed2 = 19.6GB) 403'd on the cap. ALL 14 B1 ckpts safe LOCAL (no durability risk).
  DURABILITY GAPS: current-arch finals pulled local + verified (hf_current_arch_finals/, 25/25);
  scaleup finals were HF-ONLY -> pulling local now (hf_scaleup_finals/, 24 files/26.8GB, size-verified).
  PLAN (superseded — see PUBLIC-REPO resolution below).**
- **HF PRIVATE OVER-LIMIT (2026-08-18): "Private repository storage limit reached for MIRIBerkeley" —
  now blocks READS+writes to ALL private repos. My B1 upload (10 ckpts ~15GB) tipped it over ~88GB.
  scaleup finals (HF-ONLY, ~27GB) are currently UN-downloadable (limit blocks reads) -> scaleup
  durability gap REMAINS OPEN until private quota resolved OR private space freed. Nothing auto-deleted.**
- **RESOLUTION = PUBLIC REPO (user-authorized 2026-08-18 "public is fine"): B1 ckpts -> NEW PUBLIC repo
  MIRIBerkeley/data-vs-algorithms-ceg-expB (public storage 8.7TB, SEPARATE from the exhausted private
  quota -> no deletion needed). B1 ckpts are OWT-trained (owt_gpt2/neox/smollm2), NO DCLM/gated data ->
  clean to be public. Uploading all 14 (6 denom + 6 candidate + 2 replicate) to exp-b-b1/<label>/.
  Private repo data-vs-algorithms-ceg UNTOUCHED (study finals incl DCLM-trained arms stay private).
  The 10 B1 ckpts I put in the PRIVATE repo earlier are now redundant (also going public) -> deleting
  them from private would free ~15GB + likely restore private read access to back up scaleup (OPTIONAL
  follow-up). scripts/hf_upload_b1 (scratchpad) parameterized by REPO.**
- HF repo (HISTORICAL record, superseded by the verified state above): **MIRIBerkeley/data-vs-algorithms-ceg** (private; MOVED 2026-07-28
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
