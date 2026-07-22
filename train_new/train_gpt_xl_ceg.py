"""Tier-3 (1.5B) A1 arm — the DOCUMENTED 2024 ScaleUp1B recipe (Option A).

This is a minimally-CEG-wrapped copy of modded-nanoGPT's first-party 1.5B
scale-up, reproduced from the vendored log
  records/track_1_short/2024-10-20_ScaleUp1B/ad8d7ae5-*.txt
which embeds the exact training source. Config (kept verbatim): GPTConfig(
n_layer=52, n_head=12, n_embd=1536), AdamW(lm_head) + Muon(transformer.h) with
lr 0.0036/2 = 0.0018 (Muon 0.1x), warmup 0, trapezoidal warmdown, batch 480
seq x 1024 tok = 491,520 tok/step, ~10B tokens over 20,344 steps.

*** ALGORITHM-VERSION CONFOUND (mandatory, front-and-center in any report) ***
This is an OLDER modded-nanoGPT generation than the 124M/355M A1 arms, which
use the CURRENT speedrun tracks. This is a PLAIN transformer: standard Linear
QKV, base-10000 rotary, 4x ReLU^2 MLP, weight tying, old Muon+AdamW. It has
NONE of the current-arch machinery (no YaRN/long-short windows, no split_embed,
no value embeds, no U-net skips, no fp8, no MTP). Consequently the whole
loader-fidelity problem class (yarn_state, split_embed) does NOT apply here —
post-hoc loading is a plain state_dict load. Any change in the algorithm
multiplier at 1.5B MUST be interpreted with this version confound stated
prominently, not buried.

WHY this and not a scaled current-arch (Option B): scaling the current medium
arch to ~48 layers requires re-deriving several hand-tuned architectural
subsystems with no 1.5B reference — most dangerously the U-net skip topology,
whose miscalibration yields a plausible-but-non-representative model with NO
divergence signature (untestable). User-ratified 2026-07-17: Option A.

CEG glue only (everything else is upstream): reads ceg.CONFIG for data/budget/
out-dir; derives step count from --token-budget (preserving the documented
warmdown ratio); logs neutral BPB via the SHARED common.bpb.evaluate_bpb
instrument (byte-identical to the A0 arm) at log-spaced checkpoints; writes the
standard CEG metrics.csv; saves plain state_dict checkpoints.
"""

import glob
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch._inductor.config as inductor_config
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import ceg  # registered by train_wrapper.py (sys.modules["ceg"] = wrapper)

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from common.bpb import evaluate_bpb, load_eval_corpus  # noqa: E402
from common.checkpoint_schedule import checkpoint_steps, prune_checkpoints  # noqa: E402

TOKENS_PER_STEP_SEQS = 8 * 60      # global batch in sequences (480), documented
DEVICE_BATCH_SIZE = 12             # sequences per device, documented
SEQUENCE_LENGTH = 1024             # documented
TOKENS_PER_STEP = TOKENS_PER_STEP_SEQS * SEQUENCE_LENGTH   # 491,520
LEARNING_RATE = 0.0036 / 2         # documented (0.0018)
WARMDOWN_RATIO = 5812 / 20344      # documented trapezoidal warmdown fraction
NUM_VOCAB = 50304                  # 50257 padded to next multiple of 128

# =============================================================================
# Muon optimizer  (lifted verbatim from the 2024 ScaleUp1B log)
# =============================================================================

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz (2024 version, verbatim)."""

    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 backend_steps=5, rank=0, world_size=1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        backend_steps=backend_steps)
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g is None:
                        curr_idx += p.numel()
                        continue
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=group['backend_steps'])
                    g *= max(g.size(0), g.size(1)) ** 0.5
                    updates_flat[curr_idx:curr_idx + p.numel()] = g.flatten()
                curr_idx += p.numel()
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx + p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()


# =============================================================================
# Model  (lifted verbatim; GPT.forward extended only with loss_reduction kwarg)
# =============================================================================

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx, targets=None, return_logits=True, loss_reduction="mean"):
        # CEG: only addition vs the upstream forward is the loss_reduction kwarg,
        # so common.bpb.evaluate_bpb (which calls model(x, y, loss_reduction="sum"))
        # is the SAME instrument used by the A0 arm. Architecture math unchanged.
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))
        if targets is not None:
            logits = self.lm_head(x)
            logits = logits.float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=loss_reduction)
        else:
            logits = self.lm_head(x[:, [-1], :])
            logits = logits.float()
            loss = None
        if not return_logits:
            logits = None
        return logits, loss


# =============================================================================
# Data loader  (lifted verbatim; our shards are magic-20240520/uint16 compatible)
# =============================================================================

def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    return int(header[2])


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        all_files = sorted(glob.glob(filename_pattern))
        assert len(all_files) > 0, f"no files match {filename_pattern}"
        # CEG: our tokenized corpora can end in a tiny remainder shard (e.g.
        # dclm train_000072.bin = 1876 tokens); the 2024 loader's per-shard
        # assert can't consume a shard smaller than one global batch. Skip such
        # undersized shards (methodologically negligible vs an ~18B corpus)
        # instead of crashing, matching how the current trackers tolerate them.
        min_shard = num_processes * B * T + 1
        self.files, ntok_total, skipped = [], 0, []
        for fname in all_files:
            shard_ntok = _peek_data_shard(fname)
            if shard_ntok < min_shard:
                skipped.append((fname, shard_ntok))
                continue
            self.files.append(fname)
            ntok_total += int(shard_ntok)
        assert self.files, f"no shard >= {min_shard} tokens in {filename_pattern}"
        if skipped and process_rank == 0:
            print(f"[DataLoader] skipped {len(skipped)} undersized shard(s) "
                  f"< {min_shard} tokens: {[(Path(f).name, n) for f, n in skipped]}",
                  flush=True)
        self.ntok_total = ntok_total
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()


# =============================================================================
# CEG training entry (module-level, executed by runpy under train_wrapper.py)
# =============================================================================

CONFIG = ceg.CONFIG

assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
master_process = (ddp_rank == 0)

torch.manual_seed(CONFIG.seed)
np.random.seed(CONFIG.seed)

# ---- derive schedule from --token-budget (preserve documented warmdown ratio)
num_iterations = max(1, round(CONFIG.token_budget / TOKENS_PER_STEP))
warmdown_iters = max(1, round(WARMDOWN_RATIO * num_iterations))
warmup_iters = 0

B, T = DEVICE_BATCH_SIZE, SEQUENCE_LENGTH
assert TOKENS_PER_STEP_SEQS % (B * ddp_world_size) == 0, \
    f"global batch {TOKENS_PER_STEP_SEQS} seqs not divisible by B*world {B * ddp_world_size}"
train_accumulation_steps = TOKENS_PER_STEP_SEQS // (B * ddp_world_size)

train_loader = DistributedDataLoader(CONFIG.train_files, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"XL(2024 ScaleUp1B) train tokens: {train_loader.ntok_total:,} "
          f"across {len(train_loader.files)} shards | steps={num_iterations} "
          f"warmdown={warmdown_iters} accum={train_accumulation_steps}", flush=True)

# ---- model: documented 1.5B dims. Keep an uncompiled handle for eval so
#      common.bpb.evaluate_bpb (@torch.no_grad) never hits the compile path.
model_raw = GPT(GPTConfig(vocab_size=NUM_VOCAB, n_layer=52, n_head=12, n_embd=1536)).cuda()
n_params = sum(p.numel() for p in model_raw.parameters())
if master_process:
    print(f"n_params: {n_params:,}", flush=True)
if hasattr(inductor_config, "coordinate_descent_tuning"):
    inductor_config.coordinate_descent_tuning = True
model_c = torch.compile(model_raw)
model = DDP(model_c, device_ids=[ddp_local_rank])
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

optimizer1 = torch.optim.AdamW(model_raw.lm_head.parameters(), lr=LEARNING_RATE,
                               betas=(0.9, 0.95), weight_decay=0, fused=True)
optimizer2 = Muon(model_raw.transformer.h.parameters(), lr=0.1 * LEARNING_RATE,
                  momentum=0.95, rank=ddp_rank, world_size=ddp_world_size)
optimizers = [optimizer1, optimizer2]


def get_lr(it):
    assert it <= num_iterations
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it < num_iterations - warmdown_iters:
        return 1.0
    else:
        return (num_iterations - it) / warmdown_iters


schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# ---- eval corpora (neutral = shared frozen instrument; own-val optional)
neutral_tokens, neutral_bytes, _ = load_eval_corpus(CONFIG.neutral_eval_dir)
ownval_tokens = ownval_bytes = None
_ownval_dir = Path(CONFIG.dataset_dir)
if (_ownval_dir / "val.bin").exists() and (_ownval_dir / "meta.json").exists():
    try:
        ownval_tokens, ownval_bytes, _ = load_eval_corpus(_ownval_dir)
    except Exception:
        ownval_tokens = ownval_bytes = None

out_dir = Path(CONFIG.out_dir)
ckpt_steps = set(checkpoint_steps(num_iterations * TOKENS_PER_STEP, TOKENS_PER_STEP,
                                  CONFIG.n_checkpoints, CONFIG.first_ckpt_frac))
CSV_COLUMNS = ["step", "tokens", "timed_hours", "gpu_hours", "train_loss_ema",
               "neutral_bpb", "ownval_bpb", "lr", "wallclock_s"]

if master_process:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps({
        "size": "xl", "arch": "scaleup1b_2024", "arm": CONFIG.arm,
        "algorithm": "new_modded_nanogpt_xl_2024scaleup",
        "n_params": n_params, "n_layer": 52, "n_head": 12, "n_embd": 1536,
        "vocab_size": NUM_VOCAB, "num_iterations": num_iterations,
        "warmdown_iters": warmdown_iters, "warmup_iters": warmup_iters,
        "learning_rate": LEARNING_RATE, "token_budget": CONFIG.token_budget,
        "tokens_per_step": TOKENS_PER_STEP, "sequence_length": SEQUENCE_LENGTH,
        "global_batch_seqs": TOKENS_PER_STEP_SEQS,
        "device_batch_size": DEVICE_BATCH_SIZE,
        "train_accumulation_steps": train_accumulation_steps,
        "world_size": ddp_world_size, "seed": CONFIG.seed,
        "run_id": str(uuid.uuid4()), "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(device),
        "ckpt_steps": sorted(ckpt_steps),
    }, indent=2))
    csv_f = open(out_dir / "metrics.csv", "w", newline="")
    import csv as _csv
    csv_w = _csv.writer(csv_f)
    csv_w.writerow(CSV_COLUMNS)
    print(f"checkpoint steps: {sorted(ckpt_steps)}", flush=True)


def run_evals(step, tokens_done, timed_seconds, loss_ema, cur_lr, wall0):
    # master-only; eval on the UNCOMPILED module (shared params) so the
    # @torch.no_grad instrument does not trip the compiled/DDP graph.
    nb = evaluate_bpb(model_raw, neutral_tokens, neutral_bytes, SEQUENCE_LENGTH,
                      device, device_batch_size=8, autocast_dtype=torch.bfloat16)
    ob = {"bpb": float("nan")}
    if ownval_tokens is not None:
        ob = evaluate_bpb(model_raw, ownval_tokens, ownval_bytes, SEQUENCE_LENGTH,
                          device, device_batch_size=8, autocast_dtype=torch.bfloat16)
    timed_hours = timed_seconds / 3600
    csv_w.writerow([step, tokens_done, f"{timed_hours:.6f}",
                    f"{timed_hours * ddp_world_size:.6f}", f"{loss_ema:.4f}",
                    f"{nb['bpb']:.6f}", f"{ob['bpb']:.6f}", f"{cur_lr:.2e}",
                    f"{time.time() - wall0:.1f}"])
    csv_f.flush()
    print(f"CEG ckpt step {step:6d} | tokens {tokens_done:12,} "
          f"| gpu_h {timed_hours * ddp_world_size:8.4f} | neutral_bpb {nb['bpb']:.4f} "
          f"| ownval_bpb {ob['bpb']:.4f} | loss {loss_ema:.4f}", flush=True)
    if CONFIG.save_checkpoints:
        torch.save({"model": model_raw.state_dict(),
                    "model_args": {"vocab_size": NUM_VOCAB, "n_layer": 52,
                                   "n_head": 12, "n_embd": 1536},
                    "size": "xl", "arch": "scaleup1b_2024", "arm": CONFIG.arm,
                    "step": step, "tokens": tokens_done,
                    "timed_seconds": timed_seconds, "world_size": ddp_world_size},
                   out_dir / f"ckpt_{step:06d}.pt")
        # rolling prune: bound disk at ~keep*5.8GB (metrics.csv holds the full
        # curve). keep_checkpoints from ceg.CONFIG; final is always retained.
        deleted = prune_checkpoints(out_dir, getattr(CONFIG, "keep_checkpoints", 3))
        if deleted:
            print(f"[prune] deleted {len(deleted)} old ckpt(s): {deleted}", flush=True)


# ---- train loop
loss_ema = float("nan")
timed_seconds = 0.0            # accumulated TRAINING time (excludes warmup + eval/ckpt)
wall0 = time.time()
torch.cuda.synchronize()
t0 = time.time()
train_loader.reset()
x, y = train_loader.next_batch()

for step in range(num_iterations + 1):
    last_step = (step == num_iterations)
    # exclude the first 10 steps (torch.compile warmup / slow initial steps)
    if step == 10:
        timed_seconds = 0.0
        torch.cuda.synchronize()
        t0 = time.time()

    if last_step or (step in ckpt_steps):
        # stop the clock around eval/checkpoint
        torch.cuda.synchronize()
        timed_seconds += time.time() - t0
        cur_lr = LEARNING_RATE * get_lr(min(step, num_iterations))
        tokens_done = step * TOKENS_PER_STEP
        if master_process:
            run_evals(step, tokens_done, timed_seconds, loss_ema, cur_lr, wall0)
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.time()

    if last_step:
        break

    # -------- training step --------
    model.train()
    for i in range(1, train_accumulation_steps + 1):
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        x, y = train_loader.next_batch()
        if i < train_accumulation_steps:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad /= train_accumulation_steps
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    model.zero_grad(set_to_none=True)

    lv = train_loss.item()
    loss_ema = lv if math.isnan(loss_ema) else 0.9 * loss_ema + 0.1 * lv

if master_process:
    csv_f.close()
    print(f"peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB", flush=True)
dist.destroy_process_group()
