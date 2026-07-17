"""CORE-subset validity gate + final-checkpoint score table (v2-canonical).

The DCLM CORE score is a SECONDARY metric here (BPB is primary). A task is
only used quantitatively at a given scale if the reference model (A0D0, the
canonical baseline) clears chance by >=2 sigma at its final checkpoint:

    usable(task, scale)  <=>  acc_final(A0D0) - 2*stderr > chance

That gated subset is then reported for every arm at that scale. Two hard
caveats, both from the loader saga:
  * lambada_openai accuracy is INVALID for A1 arms — the modded loader has no
    logits/generation path (is_greedy hardcoded False -> acc 0.0), so it is
    excluded from A1 CORE and never contributes to the gate for A1 rows.
  * A1 scores are read from the v2-canonical sweep (results/core_sweep_v2);
    A0 scores from the standard-loader sweeps (results/core_sweep[_t2]).

Chance baselines are 1/(n_choices). The three bigbench_*_multiple_choice
tasks have per-instance-variable option counts, so a fixed chance is not
well defined; they are reported but marked 'ambiguous-chance' and never
admitted to the gate (consistent with their near-chance behaviour at these
scales). Usage:
    python analysis/core_gate.py
"""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

# chance = 1/n_choices; None => not well defined (excluded from gate)
CHANCE = {
    "arc_easy": 0.25, "arc_challenge": 0.25, "openbookqa": 0.25,
    "hellaswag": 0.25, "commonsense_qa": 0.20, "agieval_lsat_ar": 0.20,
    "boolq": 0.50, "copa": 0.50, "piqa": 0.50, "winogrande": 0.50,
    "xwinograd_en": 0.50,
    "lambada_openai": None,  # A0: open-vocab acc; A1: invalid by construction
    "bigbench_cs_algorithms_multiple_choice": None,
    "bigbench_dyck_languages_multiple_choice": None,
    "bigbench_language_identification_multiple_choice": None,
}
A1_INVALID = {"lambada_openai"}  # no logits path in the modded loader

SCALES = {
    "124M": {
        "a0d0": RESULTS / "core_sweep/small_a0d0_dense_ckpt_016925.json",
        "a0d1": RESULTS / "core_sweep/small_a0d1_ckpt_016925.json",
        "a1d0": RESULTS / "core_sweep_v2/small_a1d0_2x_v2_ckpt_002780.json",
        "a1d1": RESULTS / "core_sweep_v2/small_a1d1_2x_v2_ckpt_002780.json",
    },
    "355M": {
        "a0d0": RESULTS / "core_sweep_t2/medium_a0d0_ckpt_016925.json",
        "a0d1": RESULTS / "core_sweep_t2/medium_a0d1_ckpt_016925.json",
        "a1d0": RESULTS / "core_sweep_v2/medium_a1d0_v2_ckpt_009480.json",
        "a1d1": RESULTS / "core_sweep_v2/medium_a1d1_v2_ckpt_009480.json",
    },
}


def acc_se(task_json, task):
    m = task_json.get(task, {})
    return m.get("acc,none"), m.get("acc_stderr,none")


def main():
    report = {}
    for scale, arms in SCALES.items():
        data = {a: json.loads(Path(p).read_text()) for a, p in arms.items()}
        tasks = sorted(data["a0d0"].keys())
        # gate from A0D0 reference final
        gate = {}
        for t in tasks:
            chance = CHANCE.get(t)
            acc, se = acc_se(data["a0d0"], t)
            usable = (chance is not None and acc is not None and se is not None
                      and acc - 2 * se > chance)
            gate[t] = {"chance": chance, "a0d0_acc": acc, "a0d0_stderr": se,
                       "usable": bool(usable)}
        usable_tasks = [t for t in tasks if gate[t]["usable"]]

        rows = {}
        for a in ("a0d0", "a0d1", "a1d0", "a1d1"):
            rows[a] = {}
            for t in usable_tasks:
                acc, se = acc_se(data[a], t)
                if a.startswith("a1") and t in A1_INVALID:
                    rows[a][t] = None  # invalid for A1
                else:
                    rows[a][t] = acc
        report[scale] = {"gate": gate, "usable_tasks": usable_tasks, "rows": rows}

    out = RESULTS / "core_gate_v2.json"
    out.write_text(json.dumps(report, indent=2))

    # human-readable
    for scale, r in report.items():
        print(f"\n=== {scale}: CORE-subset gate (from A0D0 final, 2sigma>chance) ===")
        print(f"usable tasks ({len(r['usable_tasks'])}): {', '.join(r['usable_tasks'])}")
        excluded = [t for t in r["gate"] if not r["gate"][t]["usable"]]
        print(f"excluded ({len(excluded)}): {', '.join(excluded)}")
        print(f"{'task':<34}{'chance':>7}{'A0D0':>8}{'A0D1':>8}{'A1D0':>8}{'A1D1':>8}")
        for t in r["usable_tasks"]:
            c = r["gate"][t]["chance"]
            def fmt(a):
                v = r["rows"][a][t]
                return "  INV" if v is None else f"{v:.3f}"
            print(f"{t:<34}{c:>7.2f}"
                  f"{fmt('a0d0'):>8}{fmt('a0d1'):>8}{fmt('a1d0'):>8}{fmt('a1d1'):>8}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
