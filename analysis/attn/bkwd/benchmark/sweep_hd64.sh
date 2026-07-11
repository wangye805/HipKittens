#!/bin/bash
cd /workspace/meta_gen_ai/sub2_bwd_hd64/HipKittens/analysis/attn/bkwd/benchmark
export THUNDERKITTENS_ROOT=/workspace/meta_gen_ai/sub2_bwd_hd64/HipKittens
# configs: B H HKV N
CFGS=(
 "1 8 1 512"
 "1 8 1 2048"
 "1 8 1 4096"
 "4 8 1 1024"
 "1 32 1 1024"
 "1 8 8 1024"
 "2 16 4 1024"
 "4 16 2 2048"
)
for c in "${CFGS[@]}"; do
  read B H HKV N <<< "$c"
  echo "======== B=$B H=$H HKV=$HKV N=$N ========"
  make -s SRC=attn_bkwd_non_causal_hd64.cpp TARGET=tk_kernel_bkwd GPU_TARGET=CDNA4 ATTN_B=$B ATTN_H=$H ATTN_H_KV=$HKV ATTN_N=$N 2>&1 | grep -E "error:" | head -3
  make -s SRC=attn_bkwd_prep_hd64.cpp TARGET=tk_kernel_bkwd_prep GPU_TARGET=CDNA4 ATTN_B=$B ATTN_H=$H ATTN_H_KV=$HKV ATTN_N=$N 2>&1 | grep -E "error:" | head -3
  HIP_VISIBLE_DEVICES=7 BB=$B HH=$H HKV=$HKV NN=$N DD=64 python3 test_nc.py 2>&1 | grep -E "^N=" 
done
echo "======== SWEEP DONE ========"
