# New-algorithm arm: modded-nanoGPT

`modded-nanogpt/` is a clone of https://github.com/KellerJordan/modded-nanogpt
(the "new algorithm" A1 in the 2x2 grid). Recipes:

- **124M track**: the main speedrun `train_gpt.py` (Muon, untied embeddings,
  rotary, ReLU^2, FlexAttention, value embeddings, etc.)
- **355M track**: `records/medium_track/` (~5,960 steps, batch 512, bf16) —
  adjust step count to our 9B/18B token budgets, not its native budget
- **1.5B track**: the README's documented 1.5B scaling result — starting recipe
  for Tier 3
- **770M**: no first-party recipe (optional Tier 4; expect LR/warmup tuning)

## Required adaptations (Phase 2 work, tracked here)

1. **Size parameterization** from `configs/model_sizes.json` — note the
   modded-nanogpt architecture is NOT GPT-2 (that's the point); "size" means
   matching param count class, using its own head/embed conventions.
2. **BPB logging** at log-spaced checkpoints (`common/checkpoint_schedule.py`),
   evaluating the same neutral corpus tokens as the old arm.
3. **Timed compute** already excludes warmup by speedrun convention — keep
   that, and also exclude our added eval/checkpoint time.
4. **Data**: reads its own .bin format (`.bin` with header, see its
   `data/fineweb.py`); our shards need a converter, or point its loader at our
   flat format. A1D0 must do exactly 2 epochs with reshuffle (requirement #9).
5. **Token budget override**: 18B tokens at every size (fixed, not scaled).
6. **Eval wrapper**: `eval/lm_eval_adapter.py` needs a second `make_lm` path
   that loads modded-nanogpt checkpoints.
7. **run_config.json parity with train_old**: must record `arm`, `gpu_name`,
   `torch_version`, `algorithm: "new_modded_nanogpt"`, n_params, and the
   checkpoint schedule, so analysis code can treat both arms' runs uniformly.

## Local-validation caveat

Upstream `train_gpt.py` hard-requires CUDA (FlexAttention + Muon distributed
setup + torch.compile), so full CPU/MPS validation on a Mac is not realistic.
Phase 2 plan: validate data-format/BPB/checkpoint-schedule plumbing locally
with unit-level tests, and do the first real end-to-end validation in the
first minutes on the Tier 1 pod (cost ~ a few dollars) before launching real
runs. If a genuinely free GPU (Colab) is available, the smoke test can move
earlier.
