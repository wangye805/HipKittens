#!/bin/bash
cd /workspace/dsa_triton/dsa_dev/hk_fwd
HK=/workspace/mi355_flash_attn_v3/HipKittens
F="-mllvm -amdgpu-mfma-vgpr-form -DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -O3 -I$HK/include -DTILE_K=64 -DDC=64 -DMINWAVES=1 -DREG_RESHUFFLE -DNO_PV_BARRIER -DHK_PERMLANE_REDUCE -DNTILES=4 -DHIP_ENABLE_WARP_SYNC_BUILTINS"
hipcc $F hk_s2_occ1_stream_asm_mode_test.cpp -o hk_asm_pr 2>&1 | grep -iE error | head
echo "== sweep (+permlane_reduce) =="
for s in 0 1 2; do for k in 0 1; do HIP_VISIBLE_DEVICES=2 ./hk_asm_pr $s $k 0 2>&1 | grep -oE "seed=[0-9].*"; done; done
HIP_VISIBLE_DEVICES=2 ./hk_asm_pr 0 1 40 2>&1 | grep -oE "ninv=[0-9]+.*"
