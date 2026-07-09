"""Old-algorithm arm: GPT-2 reproduction trainer (llm.c/nanoGPT-style, PyTorch).

Implements the methodology requirements that matter at training time:
  #1 logs BPB (neutral corpus + own-val) at every checkpoint, never raw CE only
  #4 timed compute excludes warmup/compile steps and all eval/checkpoint time;
     reported as GPU-hours (timed seconds * world_size)
  #5 hyperparameters are a function of (algorithm, size) only — the data dir is
     the ONLY thing that differs between the two data arms of a row
  #6 checkpoints on a log-spaced (denser-early) schedule
  #9 explicit epoch handling; the loader hard-fails rather than silently
     wrapping past --n-epochs

Run (toy example):
  python train_old/train.py --size small --data-dir datasets/toy_owt_gpt2 \
    --neutral-eval-dir datasets/toy_wiki_gpt2 --token-budget 200000 \
    --total-batch-tokens 8192 --device-batch-size 4 --block-size 256 \
    --n-checkpoints 5 --out-dir runs/toy_a0d0

Multi-GPU: torchrun --nproc_per_node=8 train_old/train.py ...
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.bpb import evaluate_bpb, load_eval_corpus
from common.checkpoint_schedule import checkpoint_steps
from common.data_loader import EpochShuffledLoader
from common.model_gpt2 import GPT, GPTConfig

# llm.c/GPT-3-style per-size defaults. PIN before Tier 1 and never re-tune per
# data arm (requirement #5).
SIZE_HPARAMS = {
    "small":  {"lr": 6e-4},
    "medium": {"lr": 3e-4},
    "large":  {"lr": 2.5e-4},
    "xl":     {"lr": 2e-4},
}


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=["small", "medium", "large", "xl"])
    ap.add_argument("--arm", default="", choices=["", "a0d0", "a0d1", "a1d0", "a1d1"],
                    help="grid cell this run belongs to (recorded in run_config; required for real runs)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--neutral-eval-dir", required=True,
                    help="fixed neutral corpus (Wikipedia at full scale), tokenized with THIS arm's tokenizer")
    ap.add_argument("--token-budget", type=int, required=True)
    ap.add_argument("--n-epochs", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--total-batch-tokens", type=int, default=524288)
    ap.add_argument("--device-batch-size", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=None, help="override per-size default")
    ap.add_argument("--warmup-steps", type=int, default=700)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--n-checkpoints", type=int, default=25)
    ap.add_argument("--first-ckpt-frac", type=float, default=0.001)
    ap.add_argument("--warmup-timed-steps", type=int, default=10,
                    help="initial steps excluded from timed compute (compile/kernel warmup)")
    ap.add_argument("--max-steps", type=int, default=0, help="cut short (toy runs); 0 = full budget")
    ap.add_argument("--eval-device-batch-size", type=int, default=8)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--wandb", default="", help="wandb project name; empty = CSV only")
    ap.add_argument("--save-checkpoints", type=int, default=1)
    return ap.parse_args()


def main():
    args = get_args()

    # --- distributed setup ---
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        torch.distributed.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    master = rank == 0
    device_type = device.split(":")[0]
    autocast_dtype = torch.bfloat16 if device_type == "cuda" else None
    torch.manual_seed(args.seed)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    # --- model ---
    sizes = json.loads((Path(__file__).resolve().parent.parent / "configs" / "model_sizes.json").read_text())
    sc = sizes[args.size]
    config = GPTConfig(block_size=args.block_size, n_layer=sc["n_layer"],
                       n_head=sc["n_head"], n_embd=sc["n_embd"])
    model = GPT(config).to(device)
    raw_model = model
    if args.compile:
        model = torch.compile(model)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        raw_model = model.module

    lr = args.lr or SIZE_HPARAMS[args.size]["lr"]
    optimizer = raw_model.configure_optimizers(args.weight_decay, lr, (0.9, 0.95), device_type)

    # --- data ---
    loader = EpochShuffledLoader(args.data_dir, args.block_size, args.device_batch_size,
                                 n_epochs=args.n_epochs, seed=args.seed,
                                 rank=rank, world_size=world)
    avail = loader.unique_tokens * args.n_epochs
    if args.token_budget > avail:
        raise SystemExit(f"token budget {args.token_budget:,} > available "
                         f"{avail:,} ({args.n_epochs} epoch(s)); pass --n-epochs explicitly "
                         f"(requirement #9: no silent wrapping)")

    tokens_per_step = args.total_batch_tokens
    per_micro = args.device_batch_size * args.block_size
    assert tokens_per_step % (per_micro * world) == 0, \
        "total-batch-tokens must be divisible by device_batch_size*block_size*world"
    grad_accum = tokens_per_step // (per_micro * world)
    total_steps = math.ceil(args.token_budget / tokens_per_step)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    ckpt_steps = set(checkpoint_steps(total_steps * tokens_per_step, tokens_per_step,
                                      args.n_checkpoints, args.first_ckpt_frac))

    neutral_tokens, neutral_bytes, _ = load_eval_corpus(args.neutral_eval_dir)
    ownval_dir = Path(args.data_dir)
    ownval_tokens = ownval_bytes = None
    if (ownval_dir / "val.bin").exists():
        ownval_tokens, ownval_bytes, _ = load_eval_corpus(ownval_dir)

    out_dir = Path(args.out_dir)
    if master:
        out_dir.mkdir(parents=True, exist_ok=True)
        gpu_name = (torch.cuda.get_device_name(device) if device_type == "cuda"
                    else "Apple MPS" if device_type == "mps" else "cpu")
        (out_dir / "run_config.json").write_text(json.dumps(
            {**vars(args), "lr": lr, "world_size": world, "total_steps": total_steps,
             "tokens_per_step": tokens_per_step, "grad_accum": grad_accum,
             "n_params": raw_model.num_params(), "device": device,
             "gpu_name": gpu_name, "torch_version": torch.__version__,
             "algorithm": "old_gpt2", "ckpt_steps": sorted(ckpt_steps)}, indent=2))
        csv_path = out_dir / "metrics.csv"
        csv_f = open(csv_path, "w", newline="")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["step", "tokens", "timed_hours", "gpu_hours", "train_loss_ema",
                        "neutral_bpb", "ownval_bpb", "lr", "wallclock_s"])
        wb = None
        if args.wandb:
            import wandb as wb_mod
            wb = wb_mod.init(project=args.wandb, name=out_dir.name, config=vars(args))
        print(f"size={args.size} params={raw_model.num_params():,} device={device} "
              f"steps={total_steps} tokens/step={tokens_per_step} grad_accum={grad_accum}")
        print(f"checkpoint steps: {sorted(ckpt_steps)}")

    def lr_at(step):
        if step < args.warmup_steps:
            return lr * (step + 1) / args.warmup_steps
        t = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        t = min(1.0, t)
        coeff = 0.5 * (1 + math.cos(math.pi * t))
        return lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * coeff)

    def run_evals(step, tokens_done, timed_seconds, loss_ema, cur_lr, wall0):
        if not master:
            return
        nb = evaluate_bpb(raw_model, neutral_tokens, neutral_bytes, args.block_size,
                          device, args.eval_device_batch_size, autocast_dtype)
        ob = {"bpb": float("nan")}
        if ownval_tokens is not None:
            ob = evaluate_bpb(raw_model, ownval_tokens, ownval_bytes, args.block_size,
                              device, args.eval_device_batch_size, autocast_dtype)
        timed_hours = timed_seconds / 3600
        row = [step, tokens_done, f"{timed_hours:.6f}", f"{timed_hours * world:.6f}",
               f"{loss_ema:.4f}", f"{nb['bpb']:.6f}", f"{ob['bpb']:.6f}",
               f"{cur_lr:.2e}", f"{time.time() - wall0:.1f}"]
        csv_w.writerow(row)
        csv_f.flush()
        if wb:
            wb.log({"step": step, "tokens": tokens_done, "gpu_hours": timed_hours * world,
                    "neutral_bpb": nb["bpb"], "ownval_bpb": ob["bpb"],
                    "train_loss_ema": loss_ema, "lr": cur_lr})
        print(f"step {step:6d} | tokens {tokens_done:12,} | gpu_h {timed_hours * world:8.4f} "
              f"| neutral_bpb {nb['bpb']:.4f} | ownval_bpb {ob['bpb']:.4f} | loss {loss_ema:.4f}")
        if args.save_checkpoints:
            torch.save({"model": raw_model.state_dict(), "config": vars(config),
                        "size": args.size, "step": step, "tokens": tokens_done,
                        "timed_seconds": timed_seconds, "world_size": world},
                       out_dir / f"ckpt_{step:06d}.pt")

    # --- train loop ---
    model.train()
    timed_seconds = 0.0
    loss_ema = None
    wall0 = time.time()
    stop = False
    for step in range(1, total_steps + 1):
        if device_type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        cur_lr = lr_at(step - 1)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr
        optimizer.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for micro in range(grad_accum):
            try:
                x, y = loader.next_batch()
            except StopIteration:
                stop = True
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if ddp:
                model.require_backward_grad_sync = micro == grad_accum - 1
            if autocast_dtype is not None:
                with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            loss = loss / grad_accum
            loss_acc += loss.item()
            loss.backward()
        if stop:
            if master:
                print("data exhausted; stopping")
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if step > args.warmup_timed_steps:  # requirement #4: exclude warmup/compile
            timed_seconds += dt
        loss_ema = loss_acc if loss_ema is None else 0.95 * loss_ema + 0.05 * loss_acc

        if step in ckpt_steps or step == total_steps:
            run_evals(step, step * tokens_per_step, timed_seconds, loss_ema, cur_lr, wall0)
            if ddp:
                torch.distributed.barrier()

    if master:
        csv_f.close()
        print(f"done. timed GPU-hours: {timed_seconds * world / 3600:.4f} "
              f"(wall: {(time.time() - wall0) / 3600:.4f} h)")
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
