"""Exp A data-era ladder: corrected 2x2 (union eval) + per-dataset CEG vs
dataset release-year. All 8 arms re-run on torch 2.10, scored on wiki_eval_union.

Methodology is the canonical study methodology, reused not re-derived:
  - threshold = analysis/threshold.py::final_tail_threshold on era_a0d0_owt
  - crossings = analysis/ceg_shapley.py::hours_to_threshold (first crossing,
    interpolated in (log gpu-hours, bpb)); never-crossing arms are CENSORED,
    reported as no-crossing bounds, never fabricated.
  - 2x2 log-space Shapley (same convention as ceg_shapley).

"@124M" = matched DIMENSIONS (12L/12H/768d), NOT matched param count: old-algo
GPT-2 = 123,689,472 params; current-arch modded = 498,773,000 (value-embed /
U-net additions ARE the algorithm being measured). Disclosed everywhere.
"""
import csv, json, math, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from threshold import final_tail_threshold
from ceg_shapley import load_curve, hours_to_threshold

RUN = os.path.expanduser("~/Desktop/era_ladder_backup/run_metrics")
PROV = os.path.expanduser("~/Desktop/era_ladder_backup/prov")  # recovered 2.4.1 (torch check only)

PARAM_COUNTS = {"old_algo": 123689472, "current_arch": 498773000,
                "note": "@124M = matched DIMENSIONS (12L/12H/768d), NOT matched "
                        "param count; current-arch's value-embed/U-net additions "
                        "ARE the algorithm"}
PUBLISHED = {"threshold": 1.274421, "data": 2.23, "algorithm": 13.69, "total": 30.5}

# dataset -> (old-algo run, current-arch run, release_year)
DATASETS = {
    "OWT":        ("era_a0d0_owt",      "era_a1d0_owt",       2019),
    "C4":         ("era_a0_c4",         "era_a1_c4",          2020),
    "RefinedWeb": ("era_a0_refinedweb", "era_a1_refinedweb",  2023),
    "DCLM":       ("era_a0d1_dclm",     "era_a1d1_dclm",      2024),
}


def cross(run):
    h, flag = hours_to_threshold(load_curve(f"{RUN}/{run}"), THR)
    return h, flag


def main():
    global THR
    THR, ntail, _ = final_tail_threshold(f"{RUN}/era_a0d0_owt/metrics.csv")

    # ---- per-arm crossings ----
    hrs, flag = {}, {}
    for ds, (a0, a1, _yr) in DATASETS.items():
        for run in (a0, a1):
            hrs[run], flag[run] = cross(run)

    # ---- 2x2 correction (OWT, DCLM) x (old, current); all four cross ----
    C = {"a0d0": hrs["era_a0d0_owt"], "a0d1": hrs["era_a0d1_dclm"],
         "a1d0": hrs["era_a1d0_owt"], "a1d1": hrs["era_a1d1_dclm"]}
    logC = {k: math.log(v) for k, v in C.items()}
    data_log = 0.5 * ((logC["a0d0"] - logC["a0d1"]) + (logC["a1d0"] - logC["a1d1"]))
    algo_log = 0.5 * ((logC["a0d0"] - logC["a1d0"]) + (logC["a0d1"] - logC["a1d1"]))
    total_log = logC["a0d0"] - logC["a1d1"]
    mult = {"data": math.exp(data_log), "algorithm": math.exp(algo_log),
            "total": math.exp(total_log)}

    # ---- torch-neutrality side-check (2.4.1 recovered vs 2.10) ----
    torch_chk = {}
    for a in ("era_a0d0_owt", "era_a0d1_dclm"):
        t24, _, _ = final_tail_threshold(f"{PROV}/{a}/metrics.csv")
        t210, _, _ = final_tail_threshold(f"{RUN}/{a}/metrics.csv")
        torch_chk[a] = {"torch_2.4.1_tail": round(t24, 6), "torch_2.10_tail": round(t210, 6),
                        "delta_2.10_minus_2.4.1": round(t210 - t24, 6)}

    corr = {
        "experiment": "Exp A data-era ladder @124M (union eval, all arms torch 2.10)",
        "corrected_threshold_neutral_bpb": round(THR, 6),
        "threshold_n_tail_checkpoints": ntail,
        "eval_set": "wiki_eval_union (2053 docs; union-decontam vs C4 and RefinedWeb)",
        "cells_2x2": {
            "gpu_hours_to_threshold": {k: round(v, 4) for k, v in C.items()},
            "labels": {"a0d0": "old-algo · OWT", "a0d1": "old-algo · DCLM",
                       "a1d0": "current-arch · OWT", "a1d1": "current-arch · DCLM"},
            "multipliers": {k: round(v, 3) for k, v in mult.items()},
            "shapley_log": {"data": data_log, "algorithm": algo_log, "total": total_log},
            "sanity_product_equals_total": math.isclose(mult["data"] * mult["algorithm"],
                                                        mult["total"], rel_tol=1e-9),
        },
        "param_counts": PARAM_COUNTS,
        "comparison_to_published": {
            "published_124M": PUBLISHED,
            "corrected_124M": {"threshold": round(THR, 6), "data": round(mult["data"], 3),
                               "algorithm": round(mult["algorithm"], 3), "total": round(mult["total"], 3)},
            "deltas": {"threshold": round(THR - PUBLISHED["threshold"], 6),
                       "data": round(mult["data"] - PUBLISHED["data"], 3),
                       "algorithm": round(mult["algorithm"] - PUBLISHED["algorithm"], 3),
                       "total": round(mult["total"] - PUBLISHED["total"], 3)},
            "framing": "EXPLICIT CORRECTION: all 4 original arms (A0D0/A0D1/A1D0/A1D1) "
                       "re-run on torch 2.10 and re-scored on the stricter union eval. "
                       "Threshold essentially unchanged (+0.0016). Both published and "
                       "corrected arms are torch 2.10, so the published->corrected shift "
                       "carries NO torch component; it is the union-eval change plus "
                       "fresh same-seed re-run variance.",
        },
        "torch_neutrality_check": {
            "detail": torch_chk,
            "interpretation": "At fixed union eval, torch 2.4.1->2.10 shifts the tail "
                              "threshold by +0.0156 (a0d0_owt) and +0.0007 (a0d1_dclm) — "
                              "NOT uniformly negligible. Does not affect the published<->"
                              "corrected comparison (both 2.10); reinforces running every "
                              "era arm on one torch version.",
        },
    }
    Path("results/era_correction_2x2.json").write_text(json.dumps(corr, indent=2))

    # ---- per-dataset era ladder ----
    a0d0_h = hrs["era_a0d0_owt"]  # global baseline (old-algo OWT)
    per = {}
    for ds, (a0, a1, yr) in DATASETS.items():
        ho, hc = hrs[a0], hrs[a1]
        entry = {"release_year": yr,
                 "old_algo": {"run": a0, "gpu_hours_to_threshold": (round(ho, 4) if ho else None),
                              "crossing": flag[a0]},
                 "current_arch": {"run": a1, "gpu_hours_to_threshold": (round(hc, 4) if hc else None),
                                  "crossing": flag[a1]}}
        # CEG vs the single global baseline a0d0 (OWT old-algo)
        entry["old_algo"]["ceg_vs_a0d0"] = (round(a0d0_h / ho, 3) if ho else None)
        entry["current_arch"]["ceg_vs_a0d0"] = (round(a0d0_h / hc, 3) if hc else None)
        # data CEG within each algorithm (vs that algorithm's OWT)
        entry["data_ceg_old_algo"] = (round(hrs["era_a0d0_owt"] / ho, 3) if ho else None)
        entry["data_ceg_current_arch"] = (round(hrs["era_a1d0_owt"] / hc, 3) if hc else None)
        entry["algo_ceg_at_dataset"] = (round(ho / hc, 3) if (ho and hc) else None)
        per[ds] = entry

    ladder = {
        "experiment": "Exp A per-dataset era ladder @124M (union eval)",
        "corrected_threshold_neutral_bpb": round(THR, 6),
        "baseline": "old-algo OWT (a0d0), gpu_hours=%.4f — CEG measured vs this" % a0d0_h,
        "param_counts": PARAM_COUNTS,
        "datasets": per,
        "censored_arms": [r for r in flag if flag[r] == "never_crossed"],
        "key_finding": "C4 (2020) is CENSORED under BOTH algorithms — its neutral BPB "
                       "never reaches the OWT-baseline threshold, so the data-quality "
                       "ladder is NON-MONOTONIC in release year (2019 OWT > 2020 C4). "
                       "RefinedWeb (2023) and DCLM (2024) both beat OWT.",
    }
    Path("results/era_ladder_results.json").write_text(json.dumps(ladder, indent=2))

    print(json.dumps(corr, indent=2))
    print("\n--- per-dataset ladder ---")
    print(json.dumps(ladder, indent=2))


if __name__ == "__main__":
    main()
