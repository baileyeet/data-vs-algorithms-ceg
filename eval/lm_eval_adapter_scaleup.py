"""lm-eval adapter for the 2024-ScaleUp1B (A1 xl-lineage) checkpoints.

These are the Tier-3 1.5B A1 arms and the ScaleUp-curve 124M A1 arms — a PLAIN
causal transformer (standard Linear QKV, base-10000 rotary, 4x ReLU^2 MLP,
weight tying; NO YaRN/split_embed/value-embeds/fp8/MTP). The modded adapter
(eval/lm_eval_adapter_modded.py) rejects them via _reject_if_xl because none of
its loader-fidelity machinery (yarn_state, split_embed, compiled fp8 numerics)
applies. This adapter is the dedicated plain-causal path.

Loading is a plain state_dict load: the GPT class lives in
train_new/train_gpt_xl_ceg.py, whose module body starts distributed training on
import, so — like the modded adapter — we exec only its class-definition prefix
(everything up to `CONFIG = ceg.CONFIG`) in an isolated namespace, then rebuild
the model from the checkpoint's saved `model_args` and load its state dict
(stripping torch.compile's `_orig_mod.` prefixes).

Unlike the modded A1 arms, this arch HAS a real logits path, so lambada_openai
is VALID here (is_greedy is computed, not hardcoded false).

Eval is eager/uncompiled fp32 — matching the trainer, which evaluates BPB on an
uncompiled model handle (train_gpt_xl_ceg.py keeps `model_raw` for that reason).

Usage (on a GPU pod, after `pip install lm-eval`):
  python eval/lm_eval_adapter_scaleup.py --ckpt /workspace/runs/xl_a1d0/ckpt_020343.pt \
    --tasks hellaswag,arc_easy --limit 500 --out results.json
"""

import argparse
import json
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRAINER = ROOT / "train_new" / "train_gpt_xl_ceg.py"
CUT_MARKER = "CONFIG = ceg.CONFIG"


def load_scaleup_classes():
    """Exec the class-definition prefix of the ScaleUp trainer (everything
    before the module-level training entry `CONFIG = ceg.CONFIG`). The prefix
    is pure definitions — imports, constants, the Muon optimizer, and the
    self-contained GPT model classes — with no distributed init or training
    side effects. `import ceg` in the prefix is satisfied with a stub."""
    # the trainer imports `ceg` (registered by train_wrapper.py at launch); a
    # stub suffices since the prefix only binds the name, never reads CONFIG.
    ceg_stub = types.ModuleType("ceg")
    ceg_stub.CONFIG = types.SimpleNamespace()
    ceg_stub.DataExhausted = type("DataExhausted", (Exception,), {})
    sys.modules.setdefault("ceg", ceg_stub)

    src = TRAINER.read_text()
    idx = src.find(CUT_MARKER)
    assert idx > 0, f"cut marker {CUT_MARKER!r} not found in {TRAINER}"
    prefix = src[:idx]
    ns = {"__name__": "scaleup_classes", "__file__": str(TRAINER)}
    exec(compile(prefix, str(TRAINER), "exec"), ns)
    return ns


def make_lm(ckpt_path, device, tokenizer_name="gpt2"):
    import tiktoken
    from lm_eval.api.model import LM
    from lm_eval.api.instance import Instance

    ns = load_scaleup_classes()
    GPT, GPTConfig = ns["GPT"], ns["GPTConfig"]
    block = ns.get("SEQUENCE_LENGTH", 1024)

    enc = tiktoken.get_encoding(tokenizer_name)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt.get("arch") == "scaleup1b_2024", \
        f"not a ScaleUp checkpoint (arch={ckpt.get('arch')!r}); use the right adapter"
    margs = ckpt["model_args"]
    model = GPT(GPTConfig(**margs)).to(device)
    sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    # weight tying (wte.weight is lm_head.weight) means the checkpoint may omit
    # the duplicate; load_state_dict on a tied model handles either form.
    model.load_state_dict(sd)
    model.eval()

    class CkptLM(LM):
        def loglikelihood(self, requests: list[Instance]):
            out = []
            with torch.no_grad():
                for inst in requests:
                    context, continuation = inst.args
                    ctx = enc.encode_ordinary(context) if context else [enc.eot_token]
                    cont = enc.encode_ordinary(continuation)
                    ids = (ctx + cont)[-(block + 1):]
                    n_cont = min(len(cont), len(ids) - 1)
                    x = torch.tensor([ids[:-1]], device=device)
                    # targets forces the full-logits branch (targets=None returns
                    # only the last position); the returned loss is unused.
                    logits, _ = model(x, torch.zeros_like(x))
                    logp = F.log_softmax(logits[0].float(), dim=-1)
                    tgt = torch.tensor(ids[1:], device=device)
                    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                    cont_lp = tok_lp[-n_cont:].sum().item()
                    greedy = bool((logp[-n_cont:].argmax(-1) == tgt[-n_cont:]).all().item())
                    out.append((cont_lp, greedy))
            return out

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError("CORE tasks are loglikelihood (context+cont)")

        def generate_until(self, requests):
            raise NotImplementedError("CORE tasks are loglikelihood-based")

    return CkptLM()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", required=True, help="comma-separated lm-eval task names")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import lm_eval

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm = make_lm(args.ckpt, device)
    results = lm_eval.simple_evaluate(model=lm, tasks=args.tasks.split(","), limit=args.limit)
    print(json.dumps(results["results"], indent=2, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps(results["results"], indent=2, default=str))


if __name__ == "__main__":
    main()
