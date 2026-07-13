#!/bin/bash
# RUN_occ2.sh — build/verify-occ/validate/bench for the [4x2] occ-2 kernel (hk_s2_occ2).
# Run INSIDE the container (yewang_hk_mi355). Reproduces every occ-2 number.
#
#   bash RUN_occ2.sh                 # default best flags, NTILES=36
#   DEV=3 NTILES=8 bash RUN_occ2.sh
#   FLAGS="-DDBUF" bash RUN_occ2.sh  # override the flag set
#
# GATE (the whole point of occ-2): resource usage MUST report Occupancy: 2, AGPRs: 0, 0 spill.
set -e
HK=${HK:-/workspace/mi355_flash_attn_v3/HipKittens}
cd "$(dirname "$0")"
DEV=${DEV:-2}
NTILES=${NTILES:-36}
NCTA=${NCTA:-128}
SCLK=${SCLK:-2400}
# current-best occ-1 wins carried over; MINWAVES=2 asks the compiler for occ-2.
FLAGS=${FLAGS:--mllvm -amdgpu-mfma-vgpr-form -DTILE_K=64 -DDC=64 -DMINWAVES=2 -DDBUF -DHK_PERMLANE_REDUCE -DREG_RESHUFFLE -DNO_PV_BARRIER}
CFLAGS="-DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -O3 -I$HK/include -DHIP_ENABLE_WARP_SYNC_BUILTINS"
echo "## config: FLAGS='$FLAGS'  NTILES=$NTILES  NCTA=$NCTA  DEV=$DEV  SCLK=$SCLK"

echo "## [1] OCC-2 GATE: resource usage (want VGPRs<=256, AGPRs:0, Occupancy:2, 0 spill)"
hipcc $CFLAGS -DNTILES=$NTILES $FLAGS -Rpass-analysis=kernel-resource-usage -c hk_s2_occ2.cpp -o /tmp/occ2.o 2>/tmp/occ2_ru.txt || true
grep -iE "hk_s2_occ2|VGPRs:|AGPRs:|Occup|Spill|ScratchSize|LDS|group segment" /tmp/occ2_ru.txt | head
grep -iE "error:|fatal" /tmp/occ2_ru.txt | head && true

echo "## [2] correctness (bf16-faithful CPU ref: O + LSE) — seeds x sink x ninv"
hipcc $CFLAGS -DNTILES=$NTILES $FLAGS hk_s2_occ2_test.cpp -o /tmp/occ2_test \
  2>/tmp/occ2_build.txt || { grep -iE "error:|fatal" /tmp/occ2_build.txt | head -30; exit 1; }
for seed in 0 1 2; do for sink in 0 1; do HIP_VISIBLE_DEVICES=$DEV /tmp/occ2_test $seed $sink 0; done; done
for ninv in 13 40 100; do HIP_VISIBLE_DEVICES=$DEV /tmp/occ2_test 0 1 $ninv; done

echo "## [3] perf: cyc/tile"
hipcc $CFLAGS -DNTILES=$NTILES -DNCTA=$NCTA $FLAGS hk_s2_occ2_bench.cpp -o /tmp/occ2_bench 2>/dev/null
echo -n "  full: "; HIP_VISIBLE_DEVICES=$DEV /tmp/occ2_bench $SCLK
echo "## done (occ-1 steady 6104; Leon 5396; occ-2 model ~3139)"
