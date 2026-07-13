"""Assemble the final markdown report from per-size analysis outputs.

Expects a results directory laid out as (produced per RUNBOOK step 5-6):
  results/<size>/ceg.json            ceg_shapley.py --out-json (+ "n_params")
  results/<size>/sensitivity.csv     threshold_sensitivity.py output
  results/<size>/curves.png          plots.py curves
  results/<size>/sensitivity.png     plots.py sensitivity
  results/cross_scale.png            plots.py cross_scale (all sizes)

Usage: python analysis/make_report.py --results-dir results --out report.md
"""

import argparse
import json
import math
from pathlib import Path

SIZE_ORDER = ["small", "medium", "large", "xl"]
ARMS = ["a0d0", "a0d1", "a1d0", "a1d1"]
ARM_DESC = {"a0d0": "old algo, old data", "a0d1": "old algo, new data",
            "a1d0": "new algo, old data (2-epoch OWT)", "a1d1": "new algo, new data"}


def size_section(size, d):
    r = json.loads((d / "ceg.json").read_text())
    m, lg, C = r["multipliers"], r["shapley_log"], r["gpu_hours_to_threshold"]
    label = f"{r.get('n_params', 0)/1e6:.0f}M" if r.get("n_params") else size
    out = [f"## {label} ({size})",
           "",
           f"Reference threshold: **{r['threshold_neutral_bpb']:.4f} BPB** "
           f"(neutral corpus; = fully-trained A0D0 at this size).",
           "",
           "| Arm | | GPU-hours to threshold | crossing |",
           "|-----|--|------------------------|----------|"]
    for a in ARMS:
        out.append(f"| {a.upper()} | {ARM_DESC[a]} | {C[a]:.2f} | {r['crossing_type'][a]} |")
    prod = m["data"] * m["algorithm"]
    out += ["",
            f"**Shapley split (log-compute):** data {lg['data']:.3f}, "
            f"algorithm {lg['algorithm']:.3f} (sum {lg['total']:.3f}).",
            f"**As multipliers:** data contributes a **{m['data']:.2f}x** compute "
            f"reduction, algorithm a **{m['algorithm']:.2f}x**; product "
            f"{prod:.2f}x vs observed total **{m['total']:.2f}x** "
            f"({'consistent' if r['sanity_product_equals_total'] else 'MISMATCH — investigate'}).",
            ""]
    if (d / "curves.png").exists():
        out += [f"![training curves]({d.name}/curves.png)", ""]
    if (d / "sensitivity.png").exists():
        out += [f"![threshold sensitivity]({d.name}/sensitivity.png)", ""]
    return out, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="report.md")
    args = ap.parse_args()
    root = Path(args.results_dir)

    lines = ["# Data vs. Algorithms: Compute-Equivalent Gain Across Scale",
             "",
             "2x2 grid (old/new data x old/new algorithm) trained at multiple "
             "GPT-2 sizes; compute measured in timed GPU-hours to a fixed "
             "neutral-corpus BPB threshold; savings Shapley-decomposed in "
             "log-compute space.",
             ""]
    results = []
    for size in SIZE_ORDER:
        d = root / size
        if (d / "ceg.json").exists():
            sec, r = size_section(size, d)
            lines += sec
            results.append((size, r))
    if (root / "cross_scale.png").exists():
        lines += ["## Cross-scale result", "",
                  "Shapley multipliers as a function of model size — the direct "
                  "test of whether the data/algorithm split is scale-invariant:",
                  "", "![cross-scale](cross_scale.png)", ""]
    if len(results) >= 2:
        lines += ["| Size | data x | algorithm x | total x |", "|--|--|--|--|"]
        for size, r in results:
            m = r["multipliers"]
            lines.append(f"| {size} | {m['data']:.2f} | {m['algorithm']:.2f} "
                         f"| {m['total']:.2f} |")
        lines.append("")
    lines += ["## Methodology notes & confounds",
              "",
              "- Loss metric is bits-per-byte on a fixed, decontaminated Wikipedia "
              "slice (tokenizer-invariant; identical raw text across all runs). "
              "Per-dataset val BPB was logged as same-distribution diagnostics only.",
              "- Decontamination of the eval corpus (13-gram overlap against the "
              "actual tokenized training samples) removed 562/2766 candidate docs "
              "(20.3%): 409 (14.8%) matched OpenWebText — Reddit-curated pages "
              "quote Wikipedia heavily — and a further 153 (5.5%) matched only "
              "DCLM-baseline (Common Crawl carries Wikipedia mirrors). Frozen "
              "corpus: 2,204 docs / 4,425,879 bytes / 1,086,611 GPT-2 tokens, "
              "sha256 cbdd72ac..., identical across every arm, size, and tier.",
              "- Compute is timed GPU-hours on the runs' actual hardware, excluding "
              "kernel-warmup/compile and eval time; raw FLOPs are never compared "
              "across arms (precision differs).",
              "- Hyperparameters are fixed per (algorithm, size) row; verified "
              "identical between data arms with scripts/verify_row_hparams.py.",
              "- A1D0 trains on exactly 2 reshuffled epochs of OpenWebText (the "
              "corpus has ~9B unique tokens vs the arm's 18B budget). All other "
              "arms are single-pass. This is a known, deliberate confound in that "
              "one cell; literature places safe repetition at <=4 epochs.",
              "- Threshold crossings are interpolated in log-compute between "
              "checkpoints (log-spaced, denser early), never snapped.",
              "- DCLM CORE scores are secondary and validity-gated (tasks near "
              "chance at small scale are excluded).",
              ""]
    Path(args.out).write_text("\n".join(lines))
    print(f"wrote {args.out} ({len(results)} size(s))")


if __name__ == "__main__":
    main()
