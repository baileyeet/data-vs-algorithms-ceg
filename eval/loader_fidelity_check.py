"""Loader fidelity check: post-hoc BPB must reproduce the training-recorded
final BPB for a checkpoint. Run on the pod:
  python eval/loader_fidelity_check.py <ckpt> <recorded_bpb> [--medium]
Exits nonzero if |loader BPB - recorded| > 0.005 (half the same-seed floor).
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.lm_eval_adapter_modded import ModdedLM, _replay_yarn, build_model

TOL = 0.005


def main():
    ckpt_path = sys.argv[1]
    recorded = float(sys.argv[2])
    is_medium = "--medium" in sys.argv

    model, ckpt, ns = build_model(ckpt_path)
    from eval.lm_eval_adapter_modded import restore_yarn

    ws = restore_yarn(model, ckpt, is_medium)
    if ws is not None:
        ws_short, ws_long = ws
        print("yarn: exact (saved state)")
    else:
        ws_short, ws_long = _replay_yarn(model, ckpt_path, ckpt, is_medium)
        print("yarn: replay (approximate — legacy checkpoint)")
    lm = ModdedLM(model, ns, is_medium=is_medium)
    if is_medium:
        lm._cfg = ns["ForwardScheduleConfig"](
            mtp_weights=torch.tensor([1.0], device="cuda"),
            ws_short=ws_short, ws_long=ws_long)
    else:
        lm._cfg = ns["ForwardScheduleConfig"](
            mtp_weights=torch.tensor([1.0], device="cuda"),
            ws_short=ws_short, ws_long=ws_long, train_max_seq_len=2048)

    meta = json.loads(Path("/workspace/datasets/wiki_eval/meta.json").read_text())
    toks = np.fromfile("/workspace/datasets/wiki_eval/val.bin", dtype=np.uint16)
    total_nats = 0.0
    win, n_targets = 1024, len(toks) - 1
    seqs = []
    for start in range(0, n_targets, win):
        end = min(start + win, n_targets)
        seqs.append([int(t) for t in toks[start : end + 1]])
    for i in range(0, len(seqs), 16):
        for losses in lm._packed_losses(seqs[i : i + 16]):
            total_nats += float(losses.float().sum())
    bpb = total_nats / (math.log(2) * meta["val_bytes"])
    delta = abs(bpb - recorded)
    verdict = "PASS" if delta <= TOL else "FAIL"
    print(f"{verdict}: loader BPB {bpb:.4f} vs recorded {recorded:.4f} (delta {delta:.4f})")
    sys.exit(0 if delta <= TOL else 1)


if __name__ == "__main__":
    main()
