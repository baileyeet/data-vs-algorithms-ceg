#!/bin/bash
# One-shot pod setup for training sessions. Run on a fresh RunPod pytorch pod:
#   bash pod_bootstrap.sh   (repo already synced to /root/ceg by the operator)
# Idempotent; safe to re-run.
set -euo pipefail

echo "== deps"
pip install --quiet 'torch==2.10' kernels tqdm tiktoken datasets transformers accelerate lm-eval huggingface_hub
# stale torch-2.4-era companions break transformers imports under torch 2.10
pip uninstall -y -q torchvision torchaudio 2>/dev/null || true

echo "== sanity"
python3 - <<'EOF'
import torch, tiktoken, datasets, transformers, lm_eval
assert torch.cuda.is_available(), "no CUDA"
print("torch", torch.__version__, "| GPUs:", torch.cuda.device_count(),
      torch.cuda.get_device_name(0))
EOF

echo "== volume"
test -d /workspace/datasets || { echo "FATAL: /workspace/datasets missing — volume not attached?"; exit 1; }
python3 - <<'EOF'
import hashlib, json
h = hashlib.sha256(open('/workspace/datasets/wiki_eval/val.bin','rb').read()).hexdigest()
assert h.startswith('cbdd72ac'), f"frozen eval corpus hash mismatch: {h[:16]}"
for d in ('owt_gpt2', 'dclm_gpt2', 'owt_nanogpt', 'dclm_nanogpt'):
    json.load(open(f'/workspace/datasets/{d}/meta.json'))
print("volume artifacts OK (eval corpus hash verified)")
EOF

echo "== repo check"
cd /root/ceg && python3 scripts/check_sizes.py | tail -1

echo "BOOTSTRAP COMPLETE — ready for training launches"
