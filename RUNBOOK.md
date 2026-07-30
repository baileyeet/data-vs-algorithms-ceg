# Paid-session runbook

Every session that spends money follows a written checklist here. Rules:
user confirms before each launch; track actual vs. estimated cost per tier;
terminate pods before walking away (volume persists, pods bill while idle).

Costs actually incurred — two buckets, per user: **prep/validation** vs
**per-tier training**, so the $10k budget stays auditable:

| Date | Bucket | Session | Est. | Actual | Notes |
|------|--------|---------|------|--------|-------|
| 2026-07-12/13 | prep/validation | Session 0+1 merged (8xH100 @ $23.92/hr) | $50-90 | ~$115 (final at terminate) | 4/4 smokes + CORE validity + 27B-token prep; overage: 3 kernel-dep smoke retries + initial single-stream tokenization |

## STATUS (2026-07-28): STUDY COMPLETE — everything below is historical

The full study is finished (deliverable: `report.md`; both cross-scale curves
done, all arms on HF at **MIRIBerkeley/data-vs-algorithms-ceg**, GitHub in sync).
Everything below this banner is kept as the **reproduction runbook + audit
trail** and reflects the state *at the time it was written* — in particular the
"awaiting user / go-no-go" language and the unchecked Tier-3 pre-flight boxes are
FROZEN pre-Tier-3 snapshots; those tiers all subsequently ran and completed. Do
not read them as open action items.

**Cost ledger is INCOMPLETE:** the two-bucket table above holds only the
prep/validation row. Per-tier training costs (Tier 1/2/3 + the ScaleUp curve)
were tracked in-session (scattered figures in `CLAUDE.md`, e.g. Tier 3
~$850–900) but were never consolidated back into this table. If a full audited
ledger is needed, reconstruct it from the CLAUDE.md session notes.

## COMPACTION-PROOF STATE (user-mandated, 2026-07-15 — do not drop)

Four facts that must survive any context compaction. If a future session is
unsure of any of these, re-read this section before acting:

1. **Loader fidelity RESOLVED (2026-07-15): 8/8 PASS, all deltas 0.0000**,
   early/mid/late × both tracks × both runs per track. The first multi-point
   batch had FAILED (medium 0.38–0.57, small marginal 0.005–0.006), exposing
   two more defects beyond the yarn/compile pair: (a) medium's `split_embed`
   tying flag is a plain attribute, never in state_dict — post-split
   checkpoints ran with the diverged lm_head as embedding; loader now derives
   it (weight equality + split_step from run_config) in build_model AND
   load_into; (b) the fidelity script's hand-rolled packing (1025-token
   segments, no BOS splits) carried a systematic +0.002–0.006 bias — it now
   uses the wrapper's make_eval_chunks/evaluate_bpb_modded verbatim.
   Validated instrument = saved yarn_state + split_embed restore +
   torch.compile + training-eval packing.
2. **Medium A1 v2 reruns DONE (2026-07-15)**: a1d1_v2 final neutral BPB
   1.0815 (v1 1.0883, Δ0.0068), a1d0_v2 final 1.2070 at 4.672 GPU-h (v1
   1.2027, Δ0.0043) — both within the ±0.01 same-seed floor. Metrics/configs
   in results/medium/ (v2 files) and results/superseded/ (v1 a1d0).
3. **Cost ledger flag, unresolved:** Tier 2 projected ~$235 vs the $200–230
   envelope — flagged to user, NOT resolved. Rerun cascade actual ~$60 vs
   ~$35 estimate. Both go in the two-bucket table below at closeout.
4. DONE 2026-07-16: v2 CORE re-sweep (160 ckpts), Tier-2 Shapley, Tier-1
   correction, CORE gate re-derivation, v2 HF uploads all complete. Tier 2 is
   now final. **At the Tier-3 go/no-go hard stop — awaiting user.** Do NOT
   launch Tier 3 (paid) without explicit per-tier confirmation AND the
   pre-flight gates below.

Open cleanup (RESOLVED): the stale ckpt_001390.pt under the 124M/A1D1 path was
subsequently removed — that arm now holds exactly its v2 final (ckpt_002780.pt)
+ metrics + config, verified 2026-07-28.

**Cost-efficiency rule (user, 2026-07-13): CPU-bound work (tokenization,
decontam scans, merging, re-chunking) is NOT run on GPU pods by default at
Tier 2+. Flag it to the user first — cheap CPU instance vs. accept-the-cost is
their explicit call each time.** Using idle GPU/CPU time on an already-running
pod for validation smokes is fine.

## Session 0 — account validation + smoke test (~$1-3, single H100)

1. `export RUNPOD_API_KEY=...` (from .env.local, never committed)
2. `python scripts/runpod_ctl.py list-gpus` — pick a datacenter with H100 SXM
   on-demand availability; confirm the account can see 8x nodes at all.
3. `python scripts/runpod_ctl.py create-volume --size-gb 300 --datacenter <DC> --confirm`
4. `python scripts/runpod_ctl.py create-pod --preset h100x1 --volume-id <V> --confirm`
5. On the pod (ssh):
   - clone repo, `pip install -r requirements-lock.txt`
   - `python scripts/check_sizes.py` (env sanity)
   - toy data: tiny OWT (~3M tok) + wiki slices via `data/prepare.py`, then a
     short `train_old --size small` smoke on them (few minutes; tiny batch/block)
   - **modded-nanogpt smoke test** (the first CUDA run of the patched trainer):
     `torchrun --nproc_per_node=1 train_new/train_wrapper.py --size small --arm a1d1
     --data-glob 'datasets/toy_owt_nanogpt/train_*.bin' --neutral-eval-dir
     datasets/toy_wiki_gpt2 --token-budget 3000000 --n-checkpoints 3
     --val-batch-size 131072 --eval-windows-per-chunk 16 --out-dir runs/smoke_new`
   - same for medium track (`--size medium`), tiny budget
   - `train_old` smoke: single-GPU, 124M, toy budget (validates CUDA path + DDP-off)
   - verify both produce metrics.csv with sane BPB + run_config.json, and that
     `eval/neutral_bpb.py --run-dir` loads the old-arm checkpoints
6. `python scripts/runpod_ctl.py terminate --pod-id <ID> --confirm`
7. Record actual cost above. Report to user before Session 1.

## Session 1 — data prep (~$2-5, CPU pod, hours)

1. `create-pod --preset cpu --volume-id <V> --confirm`
2. On the pod, tmux, then per corpus (all output onto /workspace):
   - OWT 9B + val:  `python data/prepare.py --dataset openwebtext --tokenizer gpt2
     --out /workspace/datasets/owt_gpt2 --train-tokens 9000000000 --val-tokens 2000000`
   - DCLM 9B/18B + val: same with `--dataset dclm --shuffle-buffer 100000`
     (18B superset; 9B arms read a prefix — document in meta)
   - Wikipedia neutral eval: `--dataset wikipedia --val-tokens 2000000 --shuffle-buffer 20000`
   - decontam Wikipedia against BOTH corpora (parallel scanner), re-tokenize
     cleaned jsonl, freeze the eval set (hash it; identical across all runs)
   - convert both training sets: `scripts/convert_to_nanogpt_bin.py` (shards for A1 arms)
3. Spot-check token counts vs meta.json; `df -h /workspace` (~60-70GB expected)
4. Terminate pod; record cost. **Stop: user confirms Tier 1 before any GPU run.**

## Tier 1 hard exit criteria (Tier 1 is NOT done until all of these hold)

- [ ] modded-nanoGPT checkpoint loader for eval/lm_eval_adapter.py written AND
      verified (loads an A1 checkpoint, hellaswag score sanity-checked) — without
      it the A1 arms never get CORE scores, only BPB. Not allowed to slip.
- [ ] CORE validity table recorded (per-task score vs chance, toy + A0D0-final
      checkpoints) and the quantitative-use decision for 124M/355M documented.
- [ ] verify_row_hparams.py passes on both rows.
- [ ] All 4 arms' metrics.csv + run_config.json + kept checkpoints copied to
      results/small/ and pushed.
- [ ] Actual vs. estimated cost recorded above; user informed before Tier 2.

## Tier 3 pre-flight gates (ALL hard requirements before launch)

- [ ] User grows network volume to 500GB in console (belt) AND the architectural
      fixes below are in place (braces) — user directive: both, not either/or.
- [ ] In-training CORE-subset hooks in both trainers (removes checkpoint-retention
      need for post-hoc sweeps; permanent fix for the reload-fidelity bug class —
      rotary/YaRN buffers are persistent=False and mutate on schedule).
- [ ] Rolling checkpoint prune during runs (keep last 3 + finals; metrics.csv holds
      the full curve) — peak footprint ~20GB/run.
- [ ] A0 checkpoints saved bf16 (6.2GB -> 3.1GB), fp32 exception for the final
      (reload-audit fidelity).
- [x] FIRST real 1.5B checkpoint size MEASURED (2026-07-17, XL-A1 2024-arch smoke):
      5.8GB fp32 for 1,549,467,648 params (~1.55B, = GPT-2-XL). Matches the 6.2GB
      fp32 projection (within 20%). No OOM on 8xH100. XL recipe (Option A, 2024
      ScaleUp1B) is WIRED + smoke-passes (BPB descends 3.16->2.82; timing excludes
      warmup). A1-XL checkpoints have NO yarn_state/split_embed (plain arch) ->
      post-hoc load is a plain state_dict load; CORE via the A0 lm-eval adapter.
      NOTE: at ~5.8GB/ckpt, the full-tier run needs rolling prune (gate below).
- [ ] Report disclosure: CORE provenance may differ across tiers (post-hoc
      yarn-replay loader for T1/T2 vs in-training hooks for T3) — state both
      instruments + validation criteria in methodology notes.
- [ ] XL recipe wired into wrapper from upstream's documented 1.5B scaling result.

## Sessions 2+ — training tiers (8xH100; confirm each with user first)

Tier 1 = 124M all 4 arms; Tier 2 = 355M; Tier 3 = 1.5B (wire XL recipe first);
Tier 4 = 770M optional. Per tier:
1. Estimate cost, get user confirmation, launch `h100x8`.
2. Run the 4 arms (A0 arms: torchrun train_old; A1 arms: torchrun train_wrapper).
   `--arm` flag set correctly; A1D0 adds `--n-epochs 2`.
3. `scripts/verify_row_hparams.py` on both rows before analysis.
4. Immediately eval all checkpoints (neutral BPB is logged during training; run
   lm-eval CORE subset + validity check on finals) BEFORE teardown.
5. Prune checkpoints (keep finals + threshold-straddling), pull metrics/run_configs
   into the repo (runs/ is gitignored — copy CSVs/JSONs into results/<size>/ which
   IS tracked), terminate pod.
6. `analysis/ceg_shapley.py` + `threshold_sensitivity.py` + `plots.py` on the tier.
7. Record actual vs. estimated cost; report to user before next tier.
