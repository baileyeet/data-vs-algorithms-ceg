"""lm-eval adapter for modded-nanoGPT (A1) checkpoints. CUDA-only.

The modded model classes are defined inside train_new/train_gpt_ceg.py, which
is a script that begins distributed training at import. We therefore exec only
its prefix — everything up to (excluding) the Hyperparameters instantiation —
in an isolated namespace to obtain the class definitions, then rebuild the
model from the checkpoint's saved `model_args` and load its state dict
(stripping torch.compile's `_orig_mod.` prefixes).

Usage (on a GPU pod):
  python eval/lm_eval_adapter_modded.py --ckpt /workspace/runs/small_a1d1/ckpt_XXXX.pt \
    --tasks hellaswag --limit 500 --out results.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CUT_MARKER = "args = Hyperparameters()"


def load_modded_classes(trainer_file):
    """Exec the class-definition prefix of the CEG trainer script."""
    import os

    # the script prefix reads torchrun env at import; single-process defaults
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # PID-unique port: parallel single-process loaders must not share the
    # rendezvous port (7 of 8 crash on bind otherwise)
    os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 2000))
    # triton_kernels lives in the upstream clone next to the trainer
    upstream = str(Path(trainer_file).parent / "modded-nanogpt")
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    src = Path(trainer_file).read_text()
    idx = src.find(CUT_MARKER)
    assert idx > 0, f"cut marker {CUT_MARKER!r} not found in {trainer_file}"
    prefix = src[:idx]
    ns = {"__name__": "modded_classes", "__file__": str(trainer_file)}
    # the prefix imports `ceg` (the wrapper registers itself under that name
    # when launched normally); provide a minimal stand-in
    import types

    ceg_stub = types.ModuleType("ceg")
    ceg_stub.CONFIG = types.SimpleNamespace(model_max_seq_len=0, out_dir="/tmp")
    ceg_stub.DataExhausted = type("DataExhausted", (Exception,), {})
    sys.modules.setdefault("ceg", ceg_stub)
    exec(compile(prefix, str(trainer_file), "exec"), ns)
    # the model classes read the module-global `args` (Hyperparameters);
    # instantiate it with defaults — eval only touches structural fields
    ns["args"] = ns["Hyperparameters"]()
    return ns


def build_model(ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    margs = ckpt["model_args"]
    algo = ckpt.get("algorithm", "new_modded_nanogpt")
    trainer = (ROOT / "train_new" /
               ("train_gpt_medium_ceg.py" if "medium" in algo else "train_gpt_ceg.py"))
    ns = load_modded_classes(trainer)
    # older checkpoints omit max_seq_len from model_args; infer from the saved
    # rotary table (yarn.factor1 has 2*max_seq_len rows) or default to 262144
    if "max_seq_len" not in margs:
        key = next((k for k in ckpt["model"]
                    if k.removeprefix("_orig_mod.").endswith("yarn.factor1")), None)
        margs["max_seq_len"] = (ckpt["model"][key].shape[0] // 2 if key else 262144)
    model = ns["GPT"](**margs).cuda()
    sd = {k.removeprefix("_orig_mod."): v.cuda() for k, v in ckpt["model"].items()}
    # distributed training pads the weight banks to multiples of world_size
    # (e.g. qk_bank 60->64 rows at world 8); single-process reload expects the
    # unpadded size — padding rows are appended, so truncate from the end.
    # Sanity of this assumption is checked by the caller (score vs chance).
    msd = model.state_dict()
    for k, v in list(sd.items()):
        if k in msd and v.shape != msd[k].shape:
            assert v.shape[1:] == msd[k].shape[1:] and v.shape[0] > msd[k].shape[0], \
                f"unexpected mismatch for {k}: {v.shape} vs {msd[k].shape}"
            sd[k] = v[: msd[k].shape[0]]
    # assign=True: parameters adopt the checkpoint's dtypes (the trainer casts
    # banks/linears to bf16 post-construction; plain load would upcast to the
    # fresh model's fp32 and break the bf16/fp8-only kernels)
    model.load_state_dict(sd, assign=True)
    model.eval()
    return model, ckpt, ns


class ModdedLM:
    """Minimal loglikelihood interface over the modded model's forward API.

    The modded forward is loss-oriented (returns per-position losses in eval
    mode via the CEG patch), so continuation logprob = sum of target-position
    losses (negated). Requests are packed one sequence per forward with varlen
    cu_seqlens, mirroring the wrapper's BPB shim.
    """

    def __init__(self, model, ns, batch_tokens=16384):
        import tiktoken

        self.model = model
        self.ns = ns
        self.enc = tiktoken.get_encoding("gpt2")
        self.batch_tokens = batch_tokens
        self._cfg = None

    def _cfg_lazy(self):
        if self._cfg is None:
            # end-of-training forward config: pure next-token MTP weights
            # ([1.0], final stage), extension-stage windows with the final
            # YaRN extension applied (short 6 blocks, long ws_post_yarn_ext=20
            # blocks, block=128) — matches the final in-training eval state
            self._cfg = self.ns["ForwardScheduleConfig"](
                mtp_weights=torch.tensor([1.0], device="cuda"),
                ws_short=6 * 128, ws_long=20 * 128, train_max_seq_len=2048)
        return self._cfg

    def _packed_losses(self, seqs):
        """One forward over many sequences packed with varlen cu_seqlens
        (mirrors the training-eval shim's shape — the fused CE kernel needs
        large flat batches, not tiny per-request calls). Returns per-position
        losses for each sequence."""
        flat, cu = [], [0]
        for ids in seqs:
            flat.extend(ids)
            cu.append(len(flat))
        pad = (-len(flat)) % 128
        if pad:
            flat.extend([50256] * pad)
            cu.append(len(flat))  # padding tail is its own ignored segment
        toks_cpu = torch.tensor(flat, dtype=torch.int32)
        bigram = self.ns["get_bigram_hash"](toks_cpu).cuda(non_blocking=True)
        toks = toks_cpu.cuda()
        targets = torch.roll(toks, -1).to(torch.int64)  # per-segment shift handled below
        cu_t = torch.tensor(cu, dtype=torch.int32, device="cuda")
        with torch.no_grad():
            losses = self.model(toks, targets, cu_t, bigram, self._cfg_lazy())
        # losses[i] = loss predicting flat[i+1] from prefix within its segment;
        # positions at segment ends predict across boundaries — never read them
        out = []
        for s, e in zip(cu, cu[1:]):
            out.append(losses[s : e - 1])
            if len(out) == len(seqs):
                break
        return out

    def loglikelihood(self, requests):
        encoded = []
        for inst in requests:
            context, continuation = inst.args
            ctx = self.enc.encode_ordinary(context) if context else [50256]
            cont = self.enc.encode_ordinary(continuation)
            encoded.append((ctx, cont))
        out = []
        batch, meta = [], []
        def flush():
            if not batch:
                return
            for losses, (ctx, cont) in zip(self._packed_losses(batch), meta):
                cl = losses[len(ctx) - 1 : len(ctx) + len(cont) - 1]
                out.append((-cl.float().sum().item(), False))
            batch.clear(); meta.clear()
        for ctx, cont in encoded:
            ids = ctx + cont
            if sum(len(b) for b in batch) + len(ids) > self.batch_tokens:
                flush()
            batch.append(ids)
            meta.append((ctx, cont))
        flush()
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, ckpt, ns = build_model(args.ckpt)
    print(f"loaded {args.ckpt}: step={ckpt.get('step')} algo={ckpt.get('algorithm')}")

    import lm_eval
    from lm_eval.api.model import LM

    inner = ModdedLM(model, ns)

    class Wrapper(LM):
        def loglikelihood(self, requests):
            return inner.loglikelihood(requests)

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError

        def generate_until(self, requests):
            raise NotImplementedError

    results = lm_eval.simple_evaluate(model=Wrapper(), tasks=args.tasks.split(","),
                                      limit=args.limit)
    print(json.dumps(results["results"], indent=2, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps(results["results"], indent=2, default=str))


if __name__ == "__main__":
    main()
