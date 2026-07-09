"""lm-evaluation-harness adapter for our checkpoints (both training arms).

CORE's tasks are loglikelihood/multiple-choice, so only loglikelihood is
implemented; generate_until raises. Requirement #3: CORE is secondary and must
pass a validity check (per-task scores meaningfully above chance) before being
used quantitatively at 124M/355M.

Usage (after `pip install lm-eval`):
  python eval/lm_eval_adapter.py --ckpt runs/toy_a0d0/ckpt_000025.pt \
    --tasks hellaswag,arc_easy,piqa --limit 200 --out results.json

The DCLM CORE task list is in eval/core_tasks.txt.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.model_gpt2 import GPT, GPTConfig


def make_lm(ckpt_path, device, batch_size=8, tokenizer_name="gpt2"):
    import tiktoken
    from lm_eval.api.model import LM
    from lm_eval.api.instance import Instance

    enc = tiktoken.get_encoding(tokenizer_name)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    block = ckpt["config"]["block_size"]

    class CkptLM(LM):
        def loglikelihood(self, requests: list[Instance]):
            out = []
            with torch.no_grad():
                for i in range(0, len(requests), batch_size):
                    chunk = requests[i : i + batch_size]
                    for inst in chunk:
                        context, continuation = inst.args
                        ctx = enc.encode_ordinary(context) if context else [enc.eot_token]
                        cont = enc.encode_ordinary(continuation)
                        ids = (ctx + cont)[-(block + 1):]
                        n_cont = min(len(cont), len(ids) - 1)
                        x = torch.tensor([ids[:-1]], device=device)
                        logits, _ = model(x)
                        logp = F.log_softmax(logits[0].float(), dim=-1)
                        tgt = torch.tensor(ids[1:], device=device)
                        tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                        cont_lp = tok_lp[-n_cont:].sum().item()
                        greedy = bool((logp[-n_cont:].argmax(-1) == tgt[-n_cont:]).all().item())
                        out.append((cont_lp, greedy))
            return out

        def loglikelihood_rolling(self, requests):
            return [(self.loglikelihood([type(r)(
                request_type="loglikelihood", doc=r.doc,
                arguments=("", r.args[0]), idx=r.idx)])[0][0],) for r in requests]

        def generate_until(self, requests):
            raise NotImplementedError("CORE tasks are loglikelihood-based")

    return CkptLM()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", required=True, help="comma-separated lm-eval task names")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import lm_eval

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    lm = make_lm(args.ckpt, device, args.batch_size)
    results = lm_eval.simple_evaluate(model=lm, tasks=args.tasks.split(","), limit=args.limit)
    print(json.dumps(results["results"], indent=2, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps(results["results"], indent=2, default=str))


if __name__ == "__main__":
    main()
