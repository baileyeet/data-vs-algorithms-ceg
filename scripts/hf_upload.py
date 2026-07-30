"""Upload trained artifacts to the project's private HF Hub repo.

Env: HF_TOKEN (write scope). Repo: <namespace>/data-vs-algorithms-ceg (private,
created on first use). The canonical repo now lives at MIRIBerkeley/ — set
HF_NAMESPACE=MIRIBerkeley to target it (default is the token's own namespace).

Folder scheme (explicit, per the two kept-separate cross-scale curves):
  current-arch-124M/ , current-arch-355M/   (A1 = current modded speedrun)
  scaleup-124M/ , scaleup-1.5B/             (A1 = 2024 ScaleUp plain arch)

Per finished run:   HF_NAMESPACE=MIRIBerkeley python scripts/hf_upload.py run \
                        --run-dir runs/su124_a0d0_5gpu --size scaleup-124M --arm A0D0
Eval corpus (once): python scripts/hf_upload.py eval-corpus \
                        --corpus-dir /workspace/datasets/wiki_eval \
                        --scan-log /root/decontam.log
Verify anything:    python scripts/hf_upload.py verify --path scaleup-1.5B/A0D0

Uploads: final checkpoint (largest step), run_config.json, metrics.csv.
NOT uploaded: tokenized training shards (regenerable; would double storage).
Designed to run under nohup right after a run finishes so it overlaps the next
run's training rather than adding idle pod time. Every upload is verified by
listing the repo path and comparing file names + sizes; exit code is nonzero
on any mismatch, so callers can retry.
"""

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO_BASENAME = "data-vs-algorithms-ceg"

README = """# Data vs. Algorithms: Compute-Equivalent Gain — trained artifacts

Private artifact store for the 2x2 (data x algorithm) CEG experiment across
GPT-2 scales. Code + write-up: github.com/baileyeet/data-vs-algorithms-ceg
(final results in report.md).

Two cross-scale curves are kept STRICTLY SEPARATE — their A1 arms are different
modded-nanoGPT generations, so the folders name the curve explicitly:
- `current-arch-124M/`, `current-arch-355M/` — A1 = the CURRENT modded-nanoGPT
  speedrun (SOTA, small-scale-tuned). 1.5B is a disclosed gap (no reproducible
  recipe).
- `scaleup-124M/`, `scaleup-1.5B/` — A1 = the 2024 ScaleUp lineage (older, plain
  transformer). 355M is a disclosed gap (no documented era-appropriate recipe).

Each `<curve-scale>/<ARM>/` holds the final checkpoint + run_config.json +
metrics.csv (full BPB-vs-GPU-hours curve). Arms: A0=GPT-2 reproduction,
A1=modded-nanoGPT; D0=OpenWebText, D1=DCLM-baseline. NOTE: the current-arch and
scaleup A0 baselines at 124M are DIFFERENT runs (8-GPU vs 5-GPU, for
hardware-consistent GPU-hour accounting within each curve).

`eval_corpus/` — the frozen decontaminated Wikipedia BPB corpus shared by every
run (val.bin + raw text + meta + contamination-scan report).

Tokenized training shards are deliberately NOT stored here (deterministically
regenerable from public sources; meta.json in eval_corpus records seeds).
Private: redistribution terms of DCLM-derived checkpoints not yet reviewed.
"""


def get_api_and_repo():
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set")
    api = HfApi(token=token)
    # canonical repo lives under MIRIBerkeley/; override via HF_NAMESPACE.
    # defaults to the token's own namespace (back-compat with earlier uploads).
    ns = os.environ.get("HF_NAMESPACE") or api.whoami()["name"]
    repo_id = f"{ns}/{REPO_BASENAME}"
    api.create_repo(repo_id, private=True, repo_type="model", exist_ok=True)
    return api, repo_id


def ensure_readme(api, repo_id):
    files = api.list_repo_files(repo_id)
    if "README.md" not in files:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write(README)
        api.upload_file(path_or_fileobj=f.name, path_in_repo="README.md",
                        repo_id=repo_id)


def verify(api, repo_id, dest, expected: dict):
    """expected: {filename: local_size_bytes}"""
    infos = {i.path.split("/")[-1]: i.size
             for i in api.get_paths_info(repo_id, [f"{dest}/{n}" for n in expected])}
    ok = True
    for name, size in expected.items():
        got = infos.get(name)
        if got != size:
            print(f"VERIFY FAIL {dest}/{name}: local {size} vs hub {got}")
            ok = False
        else:
            print(f"verified {dest}/{name} ({size:,} bytes)")
    return ok


def upload_files(api, repo_id, dest, paths):
    expected = {}
    for p in paths:
        p = Path(p)
        api.upload_file(path_or_fileobj=str(p), path_in_repo=f"{dest}/{p.name}",
                        repo_id=repo_id)
        expected[p.name] = p.stat().st_size
    return verify(api, repo_id, dest, expected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "eval-corpus", "verify"])
    ap.add_argument("--run-dir")
    ap.add_argument("--size")
    ap.add_argument("--arm")
    ap.add_argument("--corpus-dir")
    ap.add_argument("--scan-log")
    ap.add_argument("--path")
    args = ap.parse_args()

    api, repo_id = get_api_and_repo()
    ensure_readme(api, repo_id)

    if args.cmd == "run":
        assert args.run_dir and args.size and args.arm
        rd = Path(args.run_dir)
        ckpts = sorted(rd.glob("*.pt"),
                       key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
        if not ckpts:
            sys.exit(f"no checkpoints in {rd}")
        final = ckpts[-1]
        dest = f"{args.size}/{args.arm.upper()}"
        files = [final, rd / "run_config.json", rd / "metrics.csv"]
        ok = upload_files(api, repo_id, dest, [f for f in files if Path(f).exists()])
        sys.exit(0 if ok else 1)
    elif args.cmd == "eval-corpus":
        assert args.corpus_dir
        cd = Path(args.corpus_dir)
        files = [cd / "val.bin", cd / "meta.json", cd / "val_text.jsonl"]
        if args.scan_log and Path(args.scan_log).exists():
            files.append(Path(args.scan_log))
        ok = upload_files(api, repo_id, "eval_corpus",
                          [f for f in files if f.exists()])
        sys.exit(0 if ok else 1)
    else:
        for f in api.list_repo_files(repo_id):
            if not args.path or f.startswith(args.path):
                print(f)


if __name__ == "__main__":
    main()
