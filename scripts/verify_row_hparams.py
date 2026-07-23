"""Requirement #5 pre-flight check: within an algorithm row, the two data arms
must have IDENTICAL hyperparameters — only the data may differ.

Usage:
  python scripts/verify_row_hparams.py runs/small_a0d0 runs/small_a0d1
Exit 0 if the run_configs match everywhere except the allowed fields.
"""

import json
import sys
from pathlib import Path

# fields that legitimately differ between the two data arms of one row
# (n_epochs: A1D0's 2-epoch OWT repetition per requirement #9; everything else
# — including token_budget, schedule, and the neutral eval corpus, which is
# shared within a row since the row fixes the tokenizer — must be identical)
ALLOWED_DIFF = {"arm", "data_dir", "data_glob", "out_dir", "wandb", "n_epochs",
                "run_id"}  # run_id: per-run uuid (xl-2024 config), always differs
WARN_DIFF = {"device", "gpu_name", "torch_version", "world_size"}
# measurement-instrumentation fields: checkpoint cadence / eval batching /
# timing bookkeeping ONLY. These control WHERE the training curve is sampled
# and how evals are batched — they cannot alter the optimization trajectory
# (evals run between steps, touch no model/optimizer/loader state, and sit
# outside the timed clock). Reported as NOTE, never a pass-wave for anything
# training-relevant: val_batch_size and model_max_seq_len are deliberately
# EXCLUDED (they set the A1 rotary-table size), as are block_size,
# total_batch_tokens, lr/warmup/wd fields (trajectory-relevant by definition).
MEASUREMENT_DIFF = {
    "n_checkpoints",          # how many curve samples
    "first_ckpt_frac",        # where the first sample lands
    "ckpt_steps",             # the derived sample schedule
    "save_checkpoints",       # whether weights are persisted at samples
    "eval_device_batch_size", # BPB eval batching (train_old)
    "eval_seq_len",           # BPB eval windowing (train_new wrapper)
    "eval_windows_per_chunk", # BPB eval packing (train_new wrapper)
    "warmup_timed_steps",     # timing bookkeeping: steps excluded from clock
}


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    a = json.loads((Path(a_dir) / "run_config.json").read_text())
    b = json.loads((Path(b_dir) / "run_config.json").read_text())
    bad, warns = [], []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k, "<missing>"), b.get(k, "<missing>")
        if va == vb:
            continue
        if k in WARN_DIFF:
            warns.append(f"  WARN {k}: {va!r} vs {vb!r} (hardware/env — should match for CEG timing)")
        elif k in MEASUREMENT_DIFF:
            warns.append(f"  NOTE {k}: differs (measurement-only field; does not affect trajectory)")
        elif k not in ALLOWED_DIFF:
            bad.append(f"  FAIL {k}: {va!r} vs {vb!r}")
    if a.get("token_budget") != b.get("token_budget"):
        bad.append(f"  FAIL token_budget must match within a row: "
                   f"{a.get('token_budget')} vs {b.get('token_budget')}")
    if a.get("algorithm") != b.get("algorithm"):
        bad.append(f"  FAIL different algorithms: {a.get('algorithm')} vs {b.get('algorithm')} "
                   f"— this tool compares arms within one row")
    for w in warns:
        print(w)
    if bad:
        print(f"{a_dir} vs {b_dir}: HYPERPARAMETER MISMATCH (requirement #5 violated)")
        print("\n".join(bad))
        sys.exit(1)
    print(f"{a_dir} vs {b_dir}: hyperparameters identical (requirement #5 OK)"
          + (f", {len(warns)} warning(s)" if warns else ""))


if __name__ == "__main__":
    main()
