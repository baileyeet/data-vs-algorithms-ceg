#!/bin/bash
# Exp B B1 scale-validation candidate pair (train_hf, from-scratch, own tokenizer, union eval,
# re-anchored to 8.87B, global batch 2,097,152 / device-batch 32 / block 1024 -> grad_accum 8).
# Pythia-160M (gptneox, cosine, Gate-1-validated) then SmolLM2-135M (llama, wsd). Chain-abort.
set -u
cd /root/ceg
LOG=/workspace/session_logs; mkdir -p $LOG
EVAL=/workspace/datasets/wiki_eval_union
MARK=$LOG/b1_candidates.status
echo "=== B1 CANDIDATE PAIR START $(date -u) ===" > $MARK

run() {  # name cfg datadir tok schedule lr
  local name=$1 cfg=$2 dd=$3 tok=$4 sched=$5 lr=$6 out=/workspace/runs/b1_cand_$1
  echo ">>> START $name ($sched lr=$lr) $(date -u)" | tee -a $MARK
  rm -rf $out; mkdir -p $out
  torchrun --nproc_per_node=8 train_hf/train_hf_ceg.py \
    --config-json $cfg --arm $name \
    --data-dir /workspace/datasets/$dd --neutral-eval-dir $EVAL --hf-tokenizer "$tok" \
    --token-budget 8870000000 --global-batch-tokens 2097152 --device-batch-size 32 --block-size 1024 \
    --peak-lr $lr --schedule $sched --warmup-frac 0.01 --wsd-decay-frac 0.10 --min-lr-ratio 0.1 \
    --n-checkpoints 25 --first-ckpt-frac 0.001 --seed 1234 --save-checkpoints 1 --keep-checkpoints 3 \
    --out-dir $out > $LOG/$name.log 2>&1
  local rc=$?; echo "DONE $name exit=$rc $(date -u)" | tee -a $MARK; return $rc
}

run pythia160m   configs/hf/pythia_160m.json  owt_neox    "EleutherAI/gpt-neox-20b"    cosine 6e-4 || { echo "CHAIN_ABORT pythia160m"   | tee -a $MARK; exit 1; }
run smollm2_135m configs/hf/smollm2_135m.json owt_smollm2 "HuggingFaceTB/SmolLM2-135M" wsd    3e-3 || { echo "CHAIN_ABORT smollm2_135m" | tee -a $MARK; exit 1; }
echo "B1_CANDIDATES_ALL_DONE $(date -u)" | tee -a $MARK
