"""Phase 1 check: instantiate every size config (no training) and verify param
counts against the GPT-2 family table (124M/355M/770M/1.5B)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.model_gpt2 import GPT, GPTConfig

EXPECTED_M = {"small": 124, "medium": 355, "large": 770, "xl": 1500}

sizes = json.loads((Path(__file__).resolve().parent.parent / "configs" / "model_sizes.json").read_text())
ok = True
for name, sc in sizes.items():
    # count with true GPT-2 vocab (50257) for fidelity to published counts
    model = GPT(GPTConfig(vocab_size=50257, n_layer=sc["n_layer"],
                          n_head=sc["n_head"], n_embd=sc["n_embd"]))
    total = model.num_params(non_embedding=False)
    print(f"{name:7s} ({sc['label']:>4s}): n_layer={sc['n_layer']:2d} n_head={sc['n_head']:2d} "
          f"n_embd={sc['n_embd']:4d} -> {total/1e6:8.1f}M params "
          f"(non-pos-emb {model.num_params()/1e6:.1f}M)")
    if abs(total / 1e6 - EXPECTED_M[name]) / EXPECTED_M[name] > 0.10:
        print(f"  MISMATCH vs expected ~{EXPECTED_M[name]}M")
        ok = False
    del model
print("all sizes OK" if ok else "SIZE MISMATCH", flush=True)
sys.exit(0 if ok else 1)
