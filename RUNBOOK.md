# Paid-session runbook

Every session that spends money follows a written checklist here. Rules:
user confirms before each launch; track actual vs. estimated cost per tier;
terminate pods before walking away (volume persists, pods bill while idle).

Costs actually incurred (update as we go):

| Date | Session | Est. | Actual | Notes |
|------|---------|------|--------|-------|
| —    | —       | —    | —      | —     |

## Session 0 — account validation + smoke test (~$1-3, single H100)

1. `export RUNPOD_API_KEY=...` (from .env.local, never committed)
2. `python scripts/runpod_ctl.py list-gpus` — pick a datacenter with H100 SXM
   on-demand availability; confirm the account can see 8x nodes at all.
3. `python scripts/runpod_ctl.py create-volume --size-gb 300 --datacenter <DC> --confirm`
4. `python scripts/runpod_ctl.py create-pod --preset h100x1 --volume-id <V> --confirm`
5. On the pod (ssh):
   - clone repo, `pip install -r requirements-lock.txt`
   - `python scripts/check_sizes.py` (env sanity)
   - toy data: rerun the Phase 1 toy prep commands from README (few minutes)
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
