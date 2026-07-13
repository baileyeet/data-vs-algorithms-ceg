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
    return ns


def build_model(ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    margs = ckpt["model_args"]
    algo = ckpt.get("algorithm", "new_modded_nanogpt")
    trainer = (ROOT / "train_new" /
               ("train_gpt_medium_ceg.py" if "medium" in algo else "train_gpt_ceg.py"))
    ns = load_modded_classes(trainer)
    model = ns["GPT"](**margs).cuda()
    sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd)
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

    def _seq_losses(self, ids):
        """per-position losses for one padded-to-128 sequence"""
        pad = (-len(ids)) % 128
        toks = torch.tensor(ids + [50256] * pad, dtype=torch.int32, device="cuda")
        inputs, targets = toks[:-1], toks[1:].to(torch.int64)
        cu = torch.tensor([0, len(inputs)], dtype=torch.int32, device="cuda")
        with torch.no_grad():
            losses = self.model(inputs, targets, cu,
                                self.ns["get_bigram_hash"](inputs),
                                self.ns["eval_forward_args"]()
                                if "eval_forward_args" in self.ns else None)
        return losses[: len(ids) - 1]

    def loglikelihood(self, requests):
        out = []
        for inst in requests:
            context, continuation = inst.args
            ctx = self.enc.encode_ordinary(context) if context else [50256]
            cont = self.enc.encode_ordinary(continuation)
            ids = ctx + cont
            losses = self._seq_losses(ids)
            cont_losses = losses[len(ctx) - 1 : len(ids) - 1]
            out.append((-cont_losses.float().sum().item(), False))
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
