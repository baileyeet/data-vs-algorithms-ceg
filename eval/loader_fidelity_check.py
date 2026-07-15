"""Loader fidelity check: post-hoc BPB must reproduce the training-recorded
final BPB for a checkpoint. Run on the pod:
  python eval/loader_fidelity_check.py <ckpt> <recorded_bpb> [--medium]
Exits nonzero if |loader BPB - recorded| > 0.005 (half the same-seed floor).

Uses the wrapper's own eval machinery (make_eval_chunks + evaluate_bpb_modded,
i.e. the exact training instrument, including BOS-boundary segmentation and
128-aligned windows) so the check isolates model-reload fidelity from window
-packing differences. The hand-rolled packing this replaced carried a
systematic +0.002..0.006 BPB bias vs the training eval.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.bpb import load_eval_corpus  # noqa: E402
from eval.lm_eval_adapter_modded import build_model  # noqa: E402
from train_new.train_wrapper import (  # noqa: E402
    _make_eval_forward, _make_eval_forward_medium, evaluate_bpb_modded,
    make_eval_chunks)

TOL = 0.005
EVAL_DIR = "/workspace/datasets/wiki_eval"


def main():
    ckpt_path = sys.argv[1]
    recorded = float(sys.argv[2])
    is_medium = "--medium" in sys.argv

    model, ckpt, ns, ws = build_model(ckpt_path)
    if ws is None:
        sys.exit("legacy checkpoint without yarn_state — not a valid fidelity target")
    ws_short, ws_long = ws
    print("yarn: exact (saved state), model compiled")
    mtp = torch.ones(1, device="cuda")
    if is_medium:
        cfg = ns["ForwardScheduleConfig"](
            mtp_weights=mtp, ws_short=ws_short, ws_long=ws_long)
        forward = _make_eval_forward_medium(model, cfg)
    else:
        cfg = ns["ForwardScheduleConfig"](
            mtp_weights=mtp, ws_short=ws_short, ws_long=ws_long,
            train_max_seq_len=2048)
        forward = _make_eval_forward(model, ns["get_bigram_hash"], cfg)

    rc = json.loads((Path(ckpt_path).parent / "run_config.json").read_text())
    tokens, nbytes, _ = load_eval_corpus(EVAL_DIR)
    chunks, _ = make_eval_chunks(
        tokens, rc["eval_seq_len"], rc["eval_windows_per_chunk"])
    bpb = evaluate_bpb_modded(forward, chunks, nbytes)["bpb"]
    delta = abs(bpb - recorded)
    verdict = "PASS" if delta <= TOL else "FAIL"
    print(f"{verdict}: loader BPB {bpb:.4f} vs recorded {recorded:.4f} (delta {delta:.4f})")
    sys.exit(0 if delta <= TOL else 1)


if __name__ == "__main__":
    main()
