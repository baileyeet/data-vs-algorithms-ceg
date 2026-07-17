"""New-algorithm arm: CEG wrapper around the modded-nanogpt speedrun trainer.

This is the entry point for A1 runs. It parses our standard CLI, computes the
step count from --token-budget using the upstream batch-size schedule, then
executes train_new/train_gpt_ceg.py (--size small, 124M speedrun) or
train_new/train_gpt_medium_ceg.py (--size medium, 355M track) — minimally
patched copies of the upstream train_gpt.py / train_gpt_medium.py; every
patched region there is marked "# CEG:". The patched trainer calls back into
this module (registered as the module alias "ceg") for everything
measurement-related:

  #1 BPB (neutral + own-val) at every checkpoint, via a shim over the
     modded-nanogpt forward/loss API (same formula/windowing as common/bpb.py)
  #4 timed compute keeps the upstream convention (clock starts after kernel
     warmup/compile) and additionally excludes our eval/checkpoint time
  #6 log-spaced checkpoints from common/checkpoint_schedule.py
  #9 explicit epoch handling: the patched data generator hard-fails instead of
     silently wrapping past --n-epochs, and reshuffles the shard/document
     traversal with a per-epoch seed (helpers here)

The training algorithm itself (model, optimizer, LR/batch/window schedules) is
untouched — hyperparameters remain a function of (algorithm, size) only.

Run (must be launched with torchrun; upstream hard-requires distributed+CUDA):
  torchrun --nproc_per_node=8 train_new/train_wrapper.py --size small \
    --arm a1d1 --data-glob datasets/dclm_nanogpt/train_*.bin \
    --neutral-eval-dir datasets/wiki_gpt2 --token-budget 18000000000 \
    --out-dir runs/a1d1_small

Toy (single GPU; shrink --val-batch-size so the upstream warmup val pass fits;
eval chunk must satisfy eval_seq_len * eval_windows_per_chunk <= val_batch_size/8):
  torchrun --nproc_per_node=1 train_new/train_wrapper.py --size small \
    --data-glob datasets/toy_owt_nanogpt/train_*.bin \
    --neutral-eval-dir datasets/toy_wiki_gpt2 --token-budget 2000000 \
    --val-batch-size 131072 --eval-windows-per-chunk 16 \
    --out-dir runs/toy_a1d0_smoke
"""

import argparse
import csv
import glob
import json
import math
import os
import runpy
import sys
import time
from bisect import bisect_left
from itertools import accumulate, pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODDED_DIR = Path(__file__).resolve().parent / "modded-nanogpt"
sys.path.insert(0, str(MODDED_DIR))  # triton_kernels (+ evals if enabled)
sys.path.insert(0, str(REPO_ROOT))   # common.*

from common.bpb import load_eval_corpus  # noqa: E402
from common.checkpoint_schedule import checkpoint_tokens  # noqa: E402

LN2 = math.log(2)
BOS_ID = 50256
NANOGPT_MAGIC, NANOGPT_VERSION = 20240520, 1
CSV_COLUMNS = ["step", "tokens", "timed_hours", "gpu_hours", "train_loss_ema",
               "neutral_bpb", "ownval_bpb", "lr", "wallclock_s"]

# Filled in by main(); read by train_gpt_ceg.py through the "ceg" module alias.
CONFIG = None
# Mutable run state (loss EMA tensor, cached eval corpora, wallclock origin).
STATE = SimpleNamespace(loss_ema=None, wall0=None, mtp_ones=None,
                        neutral_chunks=None, neutral_bytes=None,
                        ownval_chunks=None, ownval_bytes=None)


class DataExhausted(Exception):
    """Raised by the patched data generator when --n-epochs of data are consumed
    (methodology requirement #9: never silently wrap)."""


# -----------------------------------------------------------------------------
# Epoch handling helpers (used by the CEG-patched distributed_data_generator)

def epoch_file_order(n_files: int, epoch: int, seed: int) -> list[int]:
    """Shard traversal order for a given epoch. Epoch 0 keeps the upstream
    (sorted) order so single-epoch runs are byte-identical to upstream;
    epochs >= 1 are permuted with a per-epoch seed."""
    if epoch == 0:
        return list(range(n_files))
    rng = np.random.default_rng([seed, epoch])
    return [int(i) for i in rng.permutation(n_files)]


def epoch_doc_rng(seed: int, epoch: int, file_index: int):
    """RNG for permuting the document traversal inside one shard, or None for
    epoch 0 (upstream sequential order)."""
    if epoch == 0:
        return None
    return np.random.default_rng([seed, epoch, file_index])


# -----------------------------------------------------------------------------
# Token budget -> step count, replicating the upstream stage/boundary math

def simulate_step_tokens(durations, batch_sizes, scheduled: int, extension: int) -> list[int]:
    """Tokens consumed per training step, replicating TrainingSchedule's
    boundary rounding exactly (ends = [0, round(c*S)..., S+E]). `durations`
    covers the scheduled stages; `batch_sizes` covers scheduled + extension."""
    assert len(batch_sizes) == len(durations) + 1
    ends = [0]
    for c in accumulate(durations):
        ends.append(round(c * scheduled))
    assert ends[-1] == scheduled, "stage durations must sum to 1.0"
    ends.append(scheduled + extension)
    step_tokens = []
    for (start, end), b in zip(pairwise(ends), batch_sizes):
        step_tokens.extend([b] * (end - start))
    assert len(step_tokens) == scheduled + extension
    return step_tokens


def simulate_step_tokens_medium(bs_schedule, bs_extension: int,
                                scheduled: int, extension: int) -> list[int]:
    """Tokens consumed per training step for the medium track, replicating its
    get_bs() exactly: scheduled step s uses
    bs_schedule[int(len(bs_schedule) * (s / scheduled))] (equal
    1/len(bs_schedule) fractions of the scheduled steps, including the float
    division); extension steps use bs_extension."""
    n = len(bs_schedule)
    step_tokens = [bs_schedule[int(n * (s / scheduled))] for s in range(scheduled)]
    step_tokens.extend([bs_extension] * extension)
    return step_tokens


def _smallest_scheduled(total_fn, token_budget: int) -> int:
    """Smallest S with total_fn(S) >= token_budget (total_fn nondecreasing)."""
    lo, hi = 1, 2
    while total_fn(hi) < token_budget:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if total_fn(mid) >= token_budget:
            hi = mid
        else:
            lo = mid + 1
    return lo


def solve_scheduled_iterations(durations, batch_sizes, token_budget: int,
                               upstream_scheduled: int, upstream_extension: int):
    """Smallest scheduled-iteration count S such that total trained tokens
    >= token_budget (ceil semantics, same as train_old). The extension phase
    is kept at the upstream extension/scheduled ratio (>= 1 step). The batch
    schedule's fractions-of-total-steps are preserved by construction."""
    def ext(s):
        return max(1, round(s * upstream_extension / upstream_scheduled))

    def total(s):
        return sum(simulate_step_tokens(durations, batch_sizes, s, ext(s)))

    lo = _smallest_scheduled(total, token_budget)
    return lo, ext(lo)


def solve_scheduled_iterations_medium(bs_schedule, bs_extension: int, token_budget: int,
                                      upstream_scheduled: int, upstream_extension: int):
    """Medium-track analogue of solve_scheduled_iterations, for the indexed
    get_bs() batch schedule. Same semantics: smallest scheduled count whose
    total trained tokens >= token_budget; extension kept at the upstream
    extension/scheduled ratio (>= 1 step); the batch schedule's equal
    fractions-of-scheduled-steps are preserved by get_bs()'s indexing."""
    def ext(s):
        return max(1, round(s * upstream_extension / upstream_scheduled))

    def total(s):
        return sum(simulate_step_tokens_medium(bs_schedule, bs_extension, s, ext(s)))

    lo = _smallest_scheduled(total, token_budget)
    return lo, ext(lo)


def ckpt_steps_from_step_tokens(step_tokens, n_checkpoints: int, first_frac: float) -> list[int]:
    """Map the log-spaced token schedule (common/checkpoint_schedule.py) onto
    steps of a *variable* tokens-per-step schedule: each token target becomes
    the first step whose cumulative token count reaches it."""
    cum = list(accumulate(step_tokens))
    total_steps = len(cum)
    targets = checkpoint_tokens(cum[-1], n_checkpoints, first_frac)
    steps = sorted({min(total_steps, max(1, bisect_left(cum, t) + 1)) for t in targets})
    if steps[-1] != total_steps:
        steps.append(total_steps)
    return steps


def _finalize_step_schedule(scheduled: int, extension: int, step_tokens):
    """Cache the derived per-step token schedule + checkpoint schedule on
    CONFIG (shared tail of derive_step_counts / derive_step_counts_medium)."""
    CONFIG.scheduled_iterations = scheduled
    CONFIG.extension_iterations = extension
    CONFIG.step_tokens = step_tokens
    CONFIG.cum_tokens = [0, *accumulate(step_tokens)]
    CONFIG.total_steps = len(step_tokens)
    CONFIG.total_tokens_trained = CONFIG.cum_tokens[-1]
    CONFIG.ckpt_steps = ckpt_steps_from_step_tokens(
        step_tokens, CONFIG.n_checkpoints, CONFIG.first_ckpt_frac)
    return scheduled, extension


def derive_step_counts(stages, upstream_scheduled: int, upstream_extension: int):
    """Called from train_gpt_ceg.py with its TRAINING_STAGES. Derives the step
    counts for CONFIG.token_budget and caches the per-step token schedule plus
    the checkpoint schedule on CONFIG. Returns (scheduled, extension)."""
    assert CONFIG is not None, "derive_step_counts requires CONFIG (run via train_wrapper.py)"
    durations = [s.duration for s in stages[:-1]]
    batch_sizes = [s.batch_size for s in stages]
    scheduled, extension = solve_scheduled_iterations(
        durations, batch_sizes, CONFIG.token_budget, upstream_scheduled, upstream_extension)
    step_tokens = simulate_step_tokens(durations, batch_sizes, scheduled, extension)
    return _finalize_step_schedule(scheduled, extension, step_tokens)


def derive_step_counts_medium(bs_schedule, bs_extension: int,
                              upstream_scheduled: int, upstream_extension: int):
    """Called from train_gpt_medium_ceg.py with its Hyperparameters batch
    schedule (train_bs_schedule / train_bs_extension). Same semantics as
    derive_step_counts, for the medium track's indexed get_bs() schedule."""
    assert CONFIG is not None, "derive_step_counts_medium requires CONFIG (run via train_wrapper.py)"
    scheduled, extension = solve_scheduled_iterations_medium(
        bs_schedule, bs_extension, CONFIG.token_budget, upstream_scheduled, upstream_extension)
    step_tokens = simulate_step_tokens_medium(bs_schedule, bs_extension, scheduled, extension)
    return _finalize_step_schedule(scheduled, extension, step_tokens)


def stage_summary(step_tokens):
    """Compress the per-step token schedule into [(n_steps, tokens_per_step)]."""
    out = []
    for t in step_tokens:
        if out and out[-1][1] == t:
            out[-1][0] += 1
        else:
            out.append([1, t])
    return [tuple(x) for x in out]


# -----------------------------------------------------------------------------
# .bin shard helpers (modded-nanogpt format: 256-int32 header)

def shard_token_count(pattern: str) -> int:
    total = 0
    for f in sorted(glob.glob(pattern)):
        header = np.fromfile(f, dtype=np.int32, count=256)
        assert header[0] == NANOGPT_MAGIC and header[1] == NANOGPT_VERSION, \
            f"{f}: not a modded-nanogpt .bin (run scripts/convert_to_nanogpt_bin.py)"
        total += int(header[2])
    return total


def load_nanogpt_shards(pattern: str):
    """Concatenate all shards matching pattern into one uint16 token stream."""
    parts = []
    for f in sorted(glob.glob(pattern)):
        header = np.fromfile(f, dtype=np.int32, count=256)
        assert header[0] == NANOGPT_MAGIC and header[1] == NANOGPT_VERSION
        parts.append(np.fromfile(f, dtype=np.uint16, count=int(header[2]), offset=256 * 4))
    return np.concatenate(parts) if parts else None


# -----------------------------------------------------------------------------
# BPB shim for the modded-nanogpt forward/loss API
#
# Same convention as common/bpb.py: the model predicts every token of the
# stream except the very first EOT, in non-overlapping seq_len windows, and the
# final partial window's padding is masked; bpb = nats / (ln2 * bytes).
# Difference forced by the model API: windows are packed into larger chunks and
# separated via flash-attn varlen cu_seqlens (exactly equivalent to one window
# per forward, since attention never crosses a cu_seqlens boundary), and
# document BOS positions also reset attention — that is how the modded model
# defines its own conditional distribution (mirrors its upstream val loader).

def _next_multiple_of_n(v, *, n):
    return int(math.ceil(v / n) * n)


def make_eval_chunks(tokens: np.ndarray, seq_len: int, windows_per_chunk: int,
                     bos_id: int = BOS_ID):
    """Precompute eval chunks: list of (x int32, y int64, mask bool, cu int32).
    All chunks share one fixed length C and one fixed cu_seqlens length so the
    compiled model sees a single static shape."""
    assert seq_len % 16 == 0, "modded-nanogpt attention requires seq_len % 16 == 0"
    n = len(tokens)
    assert n >= 2, "eval stream too short"
    n_targets = n - 1
    C = seq_len * windows_per_chunk
    window_starts = np.arange(0, C, seq_len, dtype=np.int64)
    raw, max_segs = [], 0
    for s in range(0, n_targets, C):
        r = min(C, n_targets - s)
        x = np.zeros(C, dtype=np.int32)
        y = np.zeros(C, dtype=np.int64)
        mask = np.zeros(C, dtype=bool)
        x[:r] = tokens[s:s + r]
        y[:r] = tokens[s + 1:s + 1 + r]
        mask[:r] = True
        bos = np.flatnonzero(x[:r] == bos_id).astype(np.int64)
        segs = np.unique(np.concatenate([window_starts, bos]))
        raw.append((x, y, mask, segs))
        max_segs = max(max_segs, len(segs))
    # fixed-length cu_seqlens (padded with C, like the upstream loader pads
    # with num_tokens_local); +2 head-room, rounded up for a stable shape
    M = _next_multiple_of_n(max_segs + 2, n=128)
    chunks = []
    for x, y, mask, segs in raw:
        cu = np.full(M, C, dtype=np.int32)
        cu[:len(segs)] = segs
        chunks.append((x, y, mask, cu))
    return chunks, C


def evaluate_bpb_modded(forward_fn, chunks, total_bytes: int) -> dict:
    """forward_fn(x_np, y_np, cu_np) -> per-position nats (torch tensor or
    array-like, length C). Framework-light so the windowing/masking math can be
    unit-tested on CPU with a fake forward."""
    total_nats = 0.0
    n_targets = 0
    for x, y, mask, cu in chunks:
        loss = forward_fn(x, y, cu)
        if hasattr(loss, "detach"):
            loss = loss.detach().float().cpu().numpy()
        loss = np.asarray(loss, dtype=np.float64).reshape(-1)
        total_nats += float(loss[mask].sum())
        n_targets += int(mask.sum())
    return {
        "bpb": total_nats / (LN2 * total_bytes),
        "nats_per_token": total_nats / n_targets,
        "n_tokens": n_targets,
        "n_bytes": total_bytes,
    }


def _make_eval_forward(model, get_bigram_hash, schedule_cfg):
    import torch

    @torch.no_grad()
    def forward(x_np, y_np, cu_np):
        x = torch.from_numpy(np.ascontiguousarray(x_np))  # int32, CPU
        bigram = get_bigram_hash(x)
        return model(
            x.to(device="cuda", non_blocking=True),
            torch.from_numpy(y_np).to(device="cuda", non_blocking=True),
            torch.from_numpy(cu_np).to(device="cuda", non_blocking=True),
            bigram.to(device="cuda", non_blocking=True),
            schedule_cfg,
        )

    return forward


def _make_eval_forward_medium(model, schedule_cfg):
    """Medium-track forward shim: model(inputs, targets, cu_seqlens,
    schedule_cfg) — no bigram-hash inputs. Returns per-position nats (the
    medium CEG copy's eval branch uses reduction="none")."""
    import torch

    @torch.no_grad()
    def forward(x_np, y_np, cu_np):
        return model(
            torch.from_numpy(np.ascontiguousarray(x_np)).to(device="cuda", non_blocking=True),
            torch.from_numpy(y_np).to(device="cuda", non_blocking=True),
            torch.from_numpy(cu_np).to(device="cuda", non_blocking=True),
            schedule_cfg,
        )

    return forward


def _ensure_eval_corpora():
    if STATE.neutral_chunks is not None:
        return
    neutral_tokens, neutral_bytes, _ = load_eval_corpus(CONFIG.neutral_eval_dir)
    STATE.neutral_chunks, _ = make_eval_chunks(
        neutral_tokens, CONFIG.eval_seq_len, CONFIG.eval_windows_per_chunk)
    STATE.neutral_bytes = neutral_bytes
    # own-val from the converted dataset dir: val_*.bin shards + meta.json bytes
    val_pattern = str(Path(CONFIG.dataset_dir) / "val_*.bin")
    meta_path = Path(CONFIG.dataset_dir) / "meta.json"
    if glob.glob(val_pattern) and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if "val_bytes" in meta:
            val_tokens = load_nanogpt_shards(val_pattern)
            STATE.ownval_chunks, _ = make_eval_chunks(
                val_tokens, CONFIG.eval_seq_len, CONFIG.eval_windows_per_chunk)
            STATE.ownval_bytes = meta["val_bytes"]


# -----------------------------------------------------------------------------
# Callbacks used by train_gpt_ceg.py

def update_loss_ema(step_loss, batch_size: int, grad_accum: int, world_size: int):
    """Track a per-token train-loss EMA on GPU (no host sync in the timed
    region; read via .item() only while the clock is paused). Note: upstream
    training loss includes the multi-token-prediction weighting, so early
    values are inflated relative to plain next-token CE — diagnostic only."""
    if step_loss is None:
        return
    per_token = step_loss.float() / (batch_size // (grad_accum * world_size))
    if STATE.loss_ema is None:
        STATE.loss_ema = per_token
    else:
        STATE.loss_ema = 0.95 * STATE.loss_ema + 0.05 * per_token


def write_run_config(model, training_manager, world_size: int, master: bool):
    """run_config.json with the same fields as train_old/train.py, plus the
    metrics.csv header. Called once after model+optimizer construction."""
    import torch
    STATE.wall0 = time.time()
    if not master:
        return
    out_dir = Path(CONFIG.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    muon_lr = adam_lr = None
    if hasattr(training_manager, "optimizer"):  # small: combined NorMuonAndAdam
        for p_cfg in training_manager.optimizer.param_cfgs.values():
            if p_cfg.optim == "normuon" and muon_lr is None:
                muon_lr = p_cfg.initial_lr
            elif p_cfg.optim == "adam" and adam_lr is None:
                adam_lr = p_cfg.initial_lr
    else:  # medium: separate NorMuon + DistAdam optimizers
        muon_lr = training_manager.muon_opt.param_groups[0]["initial_lr"]
        adam_lr = training_manager.adam_opt.param_groups[0]["initial_lr"]
    cfg = {
        **CONFIG.cli,
        "lr": muon_lr,                      # NorMuon base LR (matrix params)
        "adam_lr": adam_lr,                 # Adam base LR (embeddings/scalars)
        "world_size": world_size,
        "total_steps": CONFIG.total_steps,
        "tokens_per_step": round(CONFIG.total_tokens_trained / CONFIG.total_steps),
        "tokens_per_step_stages": stage_summary(CONFIG.step_tokens),
        "grad_accum": 8 // world_size,
        "n_params": sum(p.numel() for p in model.parameters()),
        "device": f"cuda:{os.environ.get('LOCAL_RANK', '0')}",
        "gpu_name": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "algorithm": CONFIG.algorithm,
        "ckpt_steps": CONFIG.ckpt_steps,
        "num_scheduled_iterations": CONFIG.scheduled_iterations,
        "num_extension_iterations": CONFIG.extension_iterations,
        "total_tokens_trained": CONFIG.total_tokens_trained,
    }
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        csv.writer(f).writerow(CSV_COLUMNS)


def run_checkpoint(step: int, model, training_manager, timed_seconds: float,
                   world_size: int, master: bool, print0=None, get_bigram_hash=None):
    """BPB evals + metrics row + checkpoint save. Master only; the caller
    pauses the timed clock around this and barriers afterwards.
    get_bigram_hash is the small track's bigram-hash input builder; the medium
    track's forward takes no bigram inputs, so it omits the argument."""
    if not master:
        return
    import torch
    if print0 is None:
        def print0(s, console=False):
            print(s)
    _ensure_eval_corpora()
    fwd_cfg = training_manager.get_forward_args()
    # static-shape stand-in for the (eval-unused) MTP weights, so the eval
    # graph is not recompiled when the MTP schedule changes length
    if STATE.mtp_ones is None:
        STATE.mtp_ones = torch.ones(1, device="cuda")
    fwd_cfg.mtp_weights = STATE.mtp_ones
    if get_bigram_hash is not None:  # small track
        forward = _make_eval_forward(model, get_bigram_hash, fwd_cfg)
    else:  # medium track
        forward = _make_eval_forward_medium(model, fwd_cfg)
    nb = evaluate_bpb_modded(forward, STATE.neutral_chunks, STATE.neutral_bytes)
    ob = {"bpb": float("nan")}
    if STATE.ownval_chunks is not None:
        ob = evaluate_bpb_modded(forward, STATE.ownval_chunks, STATE.ownval_bytes)
    loss_ema = float(STATE.loss_ema.item()) if STATE.loss_ema is not None else float("nan")
    if hasattr(training_manager, "optimizer"):  # small: combined NorMuonAndAdam
        cur_lr = next(p.lr for p in training_manager.optimizer.param_cfgs.values()
                      if p.optim == "normuon")
    else:  # medium: separate NorMuon optimizer (lr set each step by step_optimizers)
        cur_lr = training_manager.muon_opt.param_groups[0]["lr"]
    tokens_done = CONFIG.cum_tokens[min(step, len(CONFIG.cum_tokens) - 1)]
    timed_hours = timed_seconds / 3600
    out_dir = Path(CONFIG.out_dir)
    with open(out_dir / "metrics.csv", "a", newline="") as f:
        csv.writer(f).writerow(
            [step, tokens_done, f"{timed_hours:.6f}", f"{timed_hours * world_size:.6f}",
             f"{loss_ema:.4f}", f"{nb['bpb']:.6f}", f"{ob['bpb']:.6f}",
             f"{cur_lr:.2e}", f"{time.time() - STATE.wall0:.1f}"])
    print0(f"CEG ckpt step {step:6d} | tokens {tokens_done:12,} "
           f"| gpu_h {timed_hours * world_size:8.4f} | neutral_bpb {nb['bpb']:.4f} "
           f"| ownval_bpb {ob['bpb']:.4f} | loss {loss_ema:.4f}", console=True)
    if CONFIG.save_checkpoints:
        if CONFIG.size == "small":
            model_args = {"vocab_size": model.vocab_size, "num_layers": model.num_layers,
                          "num_heads": model.num_heads, "head_dim": model.head_dim,
                          "model_dim": model.embed.embedding_dim}
        else:  # medium GPT keeps head dims on its attention blocks, not the module
            attn = model.blocks[0].attn
            model_args = {"vocab_size": model.embed.num_embeddings, "num_layers": model.num_layers,
                          "num_heads": attn.num_heads, "head_dim": attn.head_dim,
                          "model_dim": model.embed.embedding_dim,
                          "max_seq_len": model.yarn.max_seq_len}
        # yarn/rotary buffers are persistent=False (not in state_dict) and are
        # mutated on the window schedule during training; save them explicitly
        # so post-hoc loading is exact (the reload-fidelity fix)
        raw = getattr(model, "_orig_mod", model)
        yarn_state = {}
        for name in ("yarn", "yarn_paired_head"):
            y = getattr(raw, name, None)
            if y is None:
                continue
            st = {"angular_freq": y.angular_freq.detach().cpu()}
            # small track buffers: factor1/factor2; medium track: cos/sin
            for attr in ("factor1", "factor2", "cos", "sin"):
                t = getattr(y, attr, None)
                if t is not None:
                    st[attr] = t.detach().cpu()
            if hasattr(y, "attn_scale"):
                st["attn_scale"] = y.attn_scale
            yarn_state[name] = st
        torch.save(
            {"model": model.state_dict(),  # note: compiled module ("_orig_mod." prefixes)
             "model_args": model_args, "yarn_state": yarn_state,
             "ws_state": {"ws_short": getattr(training_manager, "ws_short", None),
                          "ws_long": getattr(training_manager, "ws_long", None)},
             "size": CONFIG.size, "arm": CONFIG.arm, "step": step, "tokens": tokens_done,
             "timed_seconds": timed_seconds, "world_size": world_size},
            out_dir / f"ckpt_{step:06d}.pt")


# -----------------------------------------------------------------------------
# CLI / entry point

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=["small", "medium", "xl"])
    ap.add_argument("--arm", default="", choices=["", "a1d0", "a1d1"],
                    help="grid cell (recorded in run_config; required for real runs)")
    ap.add_argument("--data-glob", required=True,
                    help="train shard glob in modded-nanogpt .bin format, e.g. "
                         "datasets/owt_nanogpt/train_*.bin (from scripts/convert_to_nanogpt_bin.py)")
    ap.add_argument("--neutral-eval-dir", required=True,
                    help="fixed neutral corpus dir in our flat format (val.bin + meta.json)")
    ap.add_argument("--token-budget", type=int, required=True)
    ap.add_argument("--n-epochs", type=int, default=1, help="A1D0 uses 2")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-checkpoints", type=int, default=25)
    ap.add_argument("--first-ckpt-frac", type=float, default=0.001)
    ap.add_argument("--eval-seq-len", type=int, default=1024,
                    help="non-overlapping BPB window (matches train_old block_size)")
    ap.add_argument("--eval-windows-per-chunk", type=int, default=32,
                    help="BPB windows packed per forward via varlen cu_seqlens")
    ap.add_argument("--val-batch-size", type=int, default=4 * 64 * 1024 * 8,
                    help="upstream val/warmup batch tokens; shrink for toy datasets "
                         "(val shards must hold >= this many tokens)")
    ap.add_argument("--model-max-seq-len", type=int, default=0,
                    help="YaRN/rotary position-table size per rank. 0 = upstream "
                         "derivation (val_batch_size // world_size), which silently "
                         "couples the model's position range to the val-pass size; "
                         "set explicitly (49152 covers the largest training stage) "
                         "when shrinking --val-batch-size for toy shards")
    ap.add_argument("--seed", type=int, default=1234,
                    help="per-epoch reshuffle seed (epoch 0 keeps upstream order)")
    ap.add_argument("--save-checkpoints", type=int, default=1)
    return ap.parse_args()


def main():
    global CONFIG
    args = get_args()
    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        sys.exit("train_wrapper.py must be launched via torchrun (the upstream trainer "
                 "initializes CUDA+distributed at import), e.g.\n"
                 "  torchrun --nproc_per_node=8 train_new/train_wrapper.py ...")
    assert args.eval_seq_len % 16 == 0, "--eval-seq-len must be a multiple of 16"
    assert args.val_batch_size % 128 == 0, "--val-batch-size must be a multiple of 128"
    assert args.eval_seq_len * args.eval_windows_per_chunk <= args.val_batch_size // 8, \
        "eval chunk exceeds the model's max_seq_len (val_batch_size // 8)"
    # fail fast (pre-compile) if the model's position table cannot cover the
    # largest per-rank flattened training sequence: stage batches reach
    # 24*2048*8 tokens across 8 ranks = 49152 per rank. Without this check the
    # YaRN rotary assert fires ~7 minutes in, after torch.compile.
    world = int(os.environ.get("WORLD_SIZE", "1"))
    max_stage_global = {"small": 24 * 2048 * 8, "medium": 524288}[args.size]
    max_stage_per_rank = max_stage_global // world
    model_msl = args.model_max_seq_len or args.val_batch_size // world
    assert model_msl >= max_stage_per_rank, (
        f"model max_seq_len {model_msl} < largest per-rank training sequence "
        f"{max_stage_per_rank}; raise --model-max-seq-len (or --val-batch-size)")

    data_glob = os.path.abspath(args.data_glob)
    train_files = sorted(glob.glob(data_glob))
    if not train_files:
        sys.exit(f"no train shards match {data_glob}")
    dataset_dir = str(Path(train_files[0]).parent)
    if not glob.glob(str(Path(dataset_dir) / "val_*.bin")):
        sys.exit(f"no val_*.bin shards in {dataset_dir} (needed for the upstream "
                 f"kernel-warmup pass; scripts/convert_to_nanogpt_bin.py produces them)")
    avail = shard_token_count(data_glob)
    if args.token_budget > avail * args.n_epochs:
        raise SystemExit(
            f"token budget {args.token_budget:,} > available {avail * args.n_epochs:,} "
            f"({avail:,} unique x {args.n_epochs} epoch(s)); pass --n-epochs explicitly "
            f"(requirement #9: no silent wrapping)")

    CONFIG = SimpleNamespace(
        cli=vars(args),
        size=args.size, arm=args.arm,
        algorithm={"small": "new_modded_nanogpt",
                   "medium": "new_modded_nanogpt_medium",
                   "xl": "new_modded_nanogpt_xl"}[args.size],
        token_budget=args.token_budget, n_epochs=args.n_epochs,
        n_checkpoints=args.n_checkpoints, first_ckpt_frac=args.first_ckpt_frac,
        eval_seq_len=args.eval_seq_len, eval_windows_per_chunk=args.eval_windows_per_chunk,
        val_batch_size=args.val_batch_size, model_max_seq_len=args.model_max_seq_len,
        seed=args.seed, save_checkpoints=args.save_checkpoints,
        out_dir=os.path.abspath(args.out_dir),
        neutral_eval_dir=os.path.abspath(args.neutral_eval_dir),
        train_files=data_glob,
        val_files=str(Path(dataset_dir) / "val_*.bin"),
        dataset_dir=dataset_dir,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        Path(CONFIG.out_dir).mkdir(parents=True, exist_ok=True)

    # Expose this module to the patched trainer as "ceg", then run it.
    trainer = {"small": "train_gpt_ceg.py", "medium": "train_gpt_medium_ceg.py",
               "xl": "train_gpt_xl_ceg.py"}[args.size]
    sys.modules["ceg"] = sys.modules[__name__]
    runpy.run_path(str(Path(__file__).resolve().parent / trainer),
                   run_name=trainer.removesuffix(".py"))


if __name__ == "__main__":
    main()
