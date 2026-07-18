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
            "a1d0": "new algo, old data (OWT)", "a1d1": "new algo, new data"}


def _load_ceg(d):
    # canonical = new-definition (final-10%-mean threshold), v2 for A1 arms;
    # fall back to the legacy single-final ceg.json only if newdef is absent.
    for name in ("ceg_newdef.json", "ceg.json"):
        p = d / name
        if p.exists():
            return json.loads(p.read_text())
    return None


def size_section(size, d):
    r = _load_ceg(d)
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
             "",
             "**Scope / status.** Tiers 1 (124M) and 2 (355M) are final and "
             "v2-canonical (all A1 numbers re-derived from yarn_state reruns "
             "after the loader-fidelity fixes). Tier 3 (1.5B) is in progress; "
             "its A1 arm carries a specific confound flagged below.",
             ""]
    results = []
    for size in SIZE_ORDER:
        d = root / size
        if (d / "ceg_newdef.json").exists() or (d / "ceg.json").exists():
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
    # CORE-subset validity gate (secondary metric), if present
    gate_p = root / "core_gate_v2.json"
    if gate_p.exists():
        g = json.loads(gate_p.read_text())
        lines += ["## CORE-subset (secondary, validity-gated)", "",
                  "DCLM CORE is a secondary check; BPB is primary. A task is used "
                  "only where the A0D0 reference clears chance by >=2 sigma at its "
                  "final checkpoint. lambada is excluded for A1 arms (the modded "
                  "loader has no logits path). All A1 CORE is from the v2 sweep.",
                  ""]
        for scale in ("124M", "355M"):
            if scale in g:
                usable = g[scale]["usable_tasks"]
                lines.append(f"- **{scale}**: {len(usable)} usable tasks "
                             f"({', '.join(usable)}).")
        lines += ["",
                  "Note: `boolq` drops at 124M (A0D0 sits at chance, 0.504) but "
                  "clears at 355M — a scale effect, not an error. Arms cluster "
                  "closely on CORE at these scales (limit=500, near-noisy), so no "
                  "quantitative CORE-based CEG is claimed; it is a sanity gate.",
                  ""]

    lines += ["## Tier 3 (1.5B): the algorithm-version confound (READ FIRST)",
              "",
              "**The 1.5B A1 arm uses a DIFFERENT, older algorithm generation "
              "than the 124M/355M A1 arms — this must front the interpretation of "
              "any 1.5B result, not sit in a footnote.** The current modded-nanoGPT "
              "speedrun (used at 124M/355M) has no first-party, reproducible 1.5B "
              "recipe: scaling its architecture to ~48 layers would require "
              "inventing several hand-tuned subsystems (notably a U-net skip "
              "topology) with no reference and no divergence signature if wrong. "
              "The only reproducible first-party 1.5B result is the 2024 "
              "'ScaleUp1B' — a PLAIN transformer (standard attention, base rotary, "
              "ReLU^2 MLP, weight tying, old Muon+AdamW; none of the current "
              "YaRN/split-embed/value-embed/skip machinery). We use that (Option A: "
              "fully reproducible) for the 1.5B A1 arm.",
              "",
              "Consequence: the algorithm axis is NOT held fixed across scale at "
              "1.5B. Any change in the algorithm multiplier from 355M to 1.5B is "
              "confounded with this architecture-generation change and must be read "
              "as a lower-bound-flavoured estimate of the current algorithm's "
              "scaling, not a clean same-algorithm measurement. (The A0 arm is the "
              "standard GPT-2-XL scale-up, so the data axis is unaffected.)",
              ""]

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
              "- Token budget is part of the algorithm bundle: the A1 (modded) "
              "arms train at 2x the upstream-native budget for each size (small "
              "~731M, medium ~4.35B tokens), NOT a fixed cross-arm budget. The "
              "original fixed-18B design (which forced A1D0 to repeat OWT for 2 "
              "epochs) was abandoned after the recipe collapsed under ~50x "
              "schedule stretch. At the 2x-native budgets both A1 arms are "
              "single-pass — well under one epoch of their ~9B+ corpora — so there "
              "is no epoch-repetition confound in any cell.",
              "- Threshold crossings are interpolated in log-compute between "
              "checkpoints (log-spaced, denser early), never snapped.",
              "- Reference-threshold definition: mean neutral BPB over all "
              "checkpoints in the final 10% of A0D0's training (per size). "
              "The single-final-checkpoint variant is reported alongside as "
              "robustness. Why a range exists at 124M: the threshold sits on "
              "the threshold arm's end-of-training plateau, where its curve "
              "is flattest; with the original purely log-spaced checkpoint "
              "schedule the plateau was sampled sparsely, so vertical noise "
              "of order the same-seed rerun floor (~0.01 BPB) translated "
              "into ~20% swings in that arm's compute-to-threshold and hence "
              "~±8% (data) / ~±16% (total) in the multipliers. Fixed going "
              "forward by adding linearly-spaced checkpoints over the final "
              "10% of every arm's schedule; the reported range brackets the "
              "definitional freedom on the pre-fix 124M data.",
              "- DCLM CORE scores are secondary and validity-gated (tasks near "
              "chance at small scale are excluded).",
              ""]
    Path(args.out).write_text("\n".join(lines))
    print(f"wrote {args.out} ({len(results)} size(s))")


if __name__ == "__main__":
    main()
