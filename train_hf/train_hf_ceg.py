"""Shared from-scratch trainer for HF-class architectures (Experiment B).

One harness for every architecture whose model is an HF `transformers` class
(Pythia=GPTNeoX, SmolLM2=Llama, Mamba, Mamba-2). Same CEG methodology as
train_old/train.py:
  #1 logs neutral-corpus BPB (via eval/bpb_hf, the arch's own tokenizer) +
     own-val BPB at every checkpoint
  #4 timed compute excludes warmup/compile + all eval/checkpoint time; GPU-hours
     = timed_seconds * world_size
  #6 dense-tail log-spaced checkpoint schedule (common/checkpoint_schedule)

The model is built RANDOM-INIT from a JSON spec {"arch","config",...}. The
architecture's native schedule SHAPE (cosine / WSD, warmup fraction, peak LR) is
re-anchored to our token budget; the global batch is specified in TOKENS/step
(GPU-count-invariant) and realized as per-device = global/(block*device_bs*world)
— the run ABORTS if that is not an integer (no silent batch nudging; the
ScaleUp 5-vs-8 trap).

metrics.csv columns are byte-identical to train_old/train.py so all downstream
analysis (threshold.py, plots) works unchanged.

Run: torchrun --nproc_per_node=8 train_hf/train_hf_ceg.py --config-json <spec> \
       --data-dir <tokdir> --neutral-eval-dir <wiki> --hf-tokenizer <id> \
       --token-budget 8870000000 --global-batch-tokens 2097152 \
       --device-batch-size 32 --block-size 2048 --peak-lr 6e-4 \
       --schedule cosine --warmup-frac 0.01 --min-lr-ratio 0.1 \
       --weight-decay 0.01 --betas 0.9 0.95 --out-dir <out>
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.bpb import load_eval_corpus  # noqa: E402
from common.checkpoint_schedule import checkpoint_steps, prune_checkpoints  # noqa: E402
from common.data_loader import EpochShuffledLoader  # noqa: E402
from eval.bpb_hf import evaluate_bpb_hf  # noqa: E402


def build_model(spec):
    """spec = {"arch": "gptneox"|"llama"|"mamba"|"mamba2", "config": {...}}."""
    from transformers import (GPTNeoXConfig, GPTNeoXForCausalLM,
                              LlamaConfig, LlamaForCausalLM)
    reg = {"gptneox": (GPTNeoXConfig, GPTNeoXForCausalLM),
           "llama": (LlamaConfig, LlamaForCausalLM)}
    try:  # SSM classes only exist in newer transformers; import lazily
        from transformers import MambaConfig, MambaForCausalLM
        reg["mamba"] = (MambaConfig, MambaForCausalLM)
    except ImportError:
        pass
    try:
        from transformers import Mamba2Config, Mamba2ForCausalLM
        reg["mamba2"] = (Mamba2Config, Mamba2ForCausalLM)
    except ImportError:
        pass
    cfg_cls, model_cls = reg[spec["arch"]]
    config = cfg_cls(**spec["config"])
    return model_cls(config), config


def make_optimizer(model, lr, betas, weight_decay):
    """AdamW; weight decay on >=2D params (matmuls/embeddings), none on 1D
    (biases, norms) — the standard GPT/NeoX/Llama grouping."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-json", required=True, help="model spec {arch, config}")
    ap.add_argument("--arm", default="", help="grid label recorded in run_config")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--neutral-eval-dir", required=True,
                    help="frozen neutral corpus dir with val_text.jsonl (re-tokenized by THIS arch)")
    ap.add_argument("--hf-tokenizer", required=True, help="HF tokenizer id for BPB eval")
    ap.add_argument("--token-budget", type=int, required=True)
    ap.add_argument("--n-epochs", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--global-batch-tokens", type=int, required=True)
    ap.add_argument("--device-batch-size", type=int, required=True)
    ap.add_argument("--block-size", type=int, default=2048)
    ap.add_argument("--peak-lr", type=float, required=True)
    ap.add_argument("--schedule", choices=["cosine", "wsd"], default="cosine")
    ap.add_argument("--warmup-frac", type=float, default=0.01)
    ap.add_argument("--wsd-decay-frac", type=float, default=0.10, help="WSD final decay fraction")
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.95])
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--n-checkpoints", type=int, default=25)
    ap.add_argument("--first-ckpt-frac", type=float, default=0.001)
    ap.add_argument("--warmup-timed-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=0, help="cut short (smoke/gate); 0 = full")
    ap.add_argument("--eval-device-batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--save-checkpoints", type=int, default=1)
    ap.add_argument("--keep-checkpoints", type=int, default=3)
    return ap.parse_args()


def main():
    args = get_args()
    spec = json.loads(Path(args.config_json).read_text())

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        torch.distributed.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local_rank}"; torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    master = rank == 0
    device_type = device.split(":")[0]
    autocast_dtype = torch.bfloat16 if device_type == "cuda" else None
    torch.manual_seed(args.seed)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    # --- global-batch -> grad-accum, with the hard 8-GPU divisibility gate ---
    per_micro = args.device_batch_size * args.block_size
    if args.global_batch_tokens % (per_micro * world) != 0:
        raise SystemExit(
            f"STOP (ScaleUp-trap guard): global_batch_tokens={args.global_batch_tokens} "
            f"not divisible by device_batch_size*block_size*world "
            f"={args.device_batch_size}*{args.block_size}*{world}={per_micro*world}. "
            f"Adjust device-batch-size so the GLOBAL batch is realized exactly on "
            f"{world} GPUs — do NOT nudge the global batch.")
    grad_accum = args.global_batch_tokens // (per_micro * world)
    tokens_per_step = args.global_batch_tokens
    # total_steps anchors the LR schedule + checkpoint schedule to the FULL
    # token budget (the re-anchored native schedule). --max-steps only cuts the
    # LOOP short (smoke/gate) so we observe the EARLY part of the real schedule,
    # never a schedule squeezed into the short run.
    total_steps = math.ceil(args.token_budget / tokens_per_step)
    run_steps = min(total_steps, args.max_steps) if args.max_steps else total_steps

    model, config = build_model(spec)
    model = model.to(device)
    raw_model = model
    n_params = sum(p.numel() for p in raw_model.parameters())
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        raw_model = model.module

    optimizer = make_optimizer(raw_model, args.peak_lr, tuple(args.betas), args.weight_decay)

    loader = EpochShuffledLoader(args.data_dir, args.block_size, args.device_batch_size,
                                 n_epochs=args.n_epochs, seed=args.seed,
                                 rank=rank, world_size=world)
    avail = loader.unique_tokens * args.n_epochs
    need = run_steps * tokens_per_step
    if need > avail:
        raise SystemExit(f"STOP: need {need:,} tokens but only {avail:,} available "
                         f"({args.n_epochs} epoch(s)) — tokenize more or pass --n-epochs")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    ckpt_steps = set(checkpoint_steps(total_steps * tokens_per_step, tokens_per_step,
                                      args.n_checkpoints, args.first_ckpt_frac))

    out_dir = Path(args.out_dir)
    if master:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_config.json").write_text(json.dumps(
            {**vars(args), "arch": spec["arch"], "hf_config": config.to_dict(),
             "world_size": world, "total_steps": total_steps,
             "tokens_per_step": tokens_per_step, "grad_accum": grad_accum,
             "n_params": n_params, "device": device,
             "gpu_name": torch.cuda.get_device_name(device) if device_type == "cuda" else "cpu",
             "torch_version": torch.__version__, "algorithm": f"hf_{spec['arch']}",
             "ckpt_steps": sorted(ckpt_steps)}, indent=2, default=str))
        csv_f = open(out_dir / "metrics.csv", "w", newline="")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["step", "tokens", "timed_hours", "gpu_hours", "train_loss_ema",
                        "neutral_bpb", "ownval_bpb", "lr", "wallclock_s"])
        print(f"arch={spec['arch']} params={n_params:,} total_steps={total_steps} "
              f"run_steps={run_steps} tokens/step={tokens_per_step} grad_accum={grad_accum} "
              f"world={world} per_micro_tokens/gpu={per_micro}")
        print(f"ckpt steps: {sorted(ckpt_steps)}")

    def lr_at(step):  # step is 0-indexed
        warmup = max(1, round(args.warmup_frac * total_steps))
        if step < warmup:
            return args.peak_lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        prog = min(1.0, prog)
        if args.schedule == "cosine":
            coeff = 0.5 * (1 + math.cos(math.pi * prog))
            return args.peak_lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * coeff)
        # WSD: stable at peak until the final decay window, then linear decay
        decay_start = 1.0 - args.wsd_decay_frac
        if prog < decay_start:
            return args.peak_lr
        d = (prog - decay_start) / max(1e-9, args.wsd_decay_frac)
        return args.peak_lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * (1 - d))

    neutral_bytes = None  # only used for a sanity print
    def run_evals(step, tokens_done, timed_seconds, loss_ema, cur_lr, wall0):
        if not master:
            return
        nb = evaluate_bpb_hf(raw_model, tokenizer, args.neutral_eval_dir,
                             args.block_size, device, args.eval_device_batch_size,
                             autocast_dtype)
        ob = {"bpb": float("nan")}
        if (Path(args.data_dir) / "val_text.jsonl").exists():
            ob = evaluate_bpb_hf(raw_model, tokenizer, args.data_dir, args.block_size,
                                 device, args.eval_device_batch_size, autocast_dtype)
        raw_model.train()
        th = timed_seconds / 3600
        csv_w.writerow([step, tokens_done, f"{th:.6f}", f"{th * world:.6f}",
                        f"{loss_ema:.4f}", f"{nb['bpb']:.6f}", f"{ob['bpb']:.6f}",
                        f"{cur_lr:.2e}", f"{time.time() - wall0:.1f}"])
        csv_f.flush()
        print(f"step {step:6d} | tokens {tokens_done:13,} | gpu_h {th * world:8.4f} "
              f"| neutral_bpb {nb['bpb']:.4f} | ownval_bpb {ob['bpb']:.4f} | loss {loss_ema:.4f}")
        if args.save_checkpoints:
            torch.save({"model": raw_model.state_dict(), "arch": spec["arch"],
                        "hf_config": config.to_dict(), "step": step,
                        "tokens": tokens_done, "timed_seconds": timed_seconds,
                        "world_size": world}, out_dir / f"ckpt_{step:06d}.pt")
            prune_checkpoints(out_dir, args.keep_checkpoints)

    model.train()
    timed_seconds = 0.0
    loss_ema = None
    wall0 = time.time()
    V = config.vocab_size
    for step in range(1, run_steps + 1):
        if device_type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        cur_lr = lr_at(step - 1)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr
        optimizer.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for micro in range(grad_accum):
            x, y = loader.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if ddp:
                model.require_backward_grad_sync = micro == grad_accum - 1
            with torch.autocast(device_type=device_type, dtype=autocast_dtype) if autocast_dtype else _null():
                logits = model(input_ids=x).logits
                loss = F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1),
                                       ignore_index=-1)
            loss = loss / grad_accum
            loss_acc += loss.item()
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if step > args.warmup_timed_steps:
            timed_seconds += dt
        loss_ema = loss_acc if loss_ema is None else 0.95 * loss_ema + 0.05 * loss_acc
        if step in ckpt_steps or step == run_steps:
            run_evals(step, step * tokens_per_step, timed_seconds, loss_ema, cur_lr, wall0)
            if ddp:
                torch.distributed.barrier()

    if master:
        csv_f.close()
        print(f"done. timed GPU-hours: {timed_seconds * world / 3600:.4f}")
    if ddp:
        torch.distributed.destroy_process_group()


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
