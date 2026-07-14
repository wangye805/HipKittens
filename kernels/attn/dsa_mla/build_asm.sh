#!/bin/bash
cd /workspace/dsa_triton/dsa_dev/hk_fwd
HK=/workspace/mi355_flash_attn_v3/HipKittens
F="-mllvm -amdgpu-mfma-vgpr-form -DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -O3 -I$HK/include -DTILE_K=64 -DDC=64 -DMINWAVES=1 -DREG_RESHUFFLE -DNO_PV_BARRIER -DNTILES=4 -DHIP_ENABLE_WARP_SYNC_BUILTINS"
echo "== resource =="
hipcc $F -Rpass-analysis=kernel-resource-usage -c hk_s2_occ1_stream_asm_mode.cpp -o /tmp/asm.o 2>&1 | grep -iE "VGPRs:|AGPRs:|Occup|Spill" | head
echo "== build test =="
hipcc $F hk_s2_occ1_stream_asm_mode_test.cpp -o hk_asm_test 2>&1 | grep -iE error | head
echo "== run =="
for s in 0 1; do for k in 0 1; do HIP_VISIBLE_DEVICES=2 ./hk_asm_test $s $k 0; done; done
HIP_VISIBLE_DEVICES=2 ./hk_asm_test 0 1 40
