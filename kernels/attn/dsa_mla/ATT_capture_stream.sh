#!/bin/bash
# Capture ATT trace of the stream occ-1 kernel WITH source-line info, PRODUCTION config, decode ISA->src stack.
# Production: seqlen q/kv = 4096 (T_KV), topk = 1152 (NTILES=18 @ TILE_K=64), H=128 x 64 q-tiles = 8192 CTAs.
set -e
HK=/workspace/mi355_flash_attn_v3/HipKittens
cd /workspace/dsa_triton/dsa_dev/hk_fwd
NTILES=${NTILES:-18}
GRID_N=${GRID_N:-8192}
DEV=${DEV:-2}
TAG=${TAG:-stream_prod_vcndmask}
F="-mllvm -amdgpu-mfma-vgpr-form"

echo "## build $TAG (NTILES=$NTILES GRID_N=$GRID_N T_KV=4096, -g line tables)"
hipcc -DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -O3 -g -gline-tables-only \
  -I$HK/include -DNTILES=$NTILES -DTILE_K=64 -DDC=64 -DMINWAVES=1 -DDBUF -DGRID_N=$GRID_N \
  -DHIP_ENABLE_WARP_SYNC_BUILTINS $F hk_s2_occ1_stream_att.cpp -o ${TAG}_att
echo "## sanity"; HIP_VISIBLE_DEVICES=$DEV ./${TAG}_att

echo "## ATT capture (1 CU, all 4 SIMDs)"
rm -rf ${TAG}_out
HIP_VISIBLE_DEVICES=$DEV rocprofv3 --att --att-library-path /opt/rocm/lib --att-target-cu 1 \
  --att-simd-select 0xF --kernel-include-regex hk_s2_occ1_stream -d ${TAG}_out \
  -- ./${TAG}_att 2>&1 | tail -4

echo "## extract code object (gfx950 line)"
SZ=$(roc-obj-ls ${TAG}_att 2>/dev/null | grep gfx950 | grep -oE 'size=[0-9]+' | grep -oE '[0-9]+')
OFF=$(roc-obj-ls ${TAG}_att 2>/dev/null | grep gfx950 | grep -oE 'offset=[0-9]+' | grep -oE '[0-9]+')
echo "  offset=$OFF size=$SZ"
dd if=${TAG}_att of=${TAG}.hsaco bs=1 skip=$OFF count=$SZ 2>/dev/null

CODEJSON=$(find ${TAG}_out -name code.json | head -1)
echo "## decode ISA->src: $CODEJSON"
python3 resolve_isa_src.py "$CODEJSON" ${TAG}.hsaco > ${TAG}_isa_stack.txt 2>/dev/null
wc -l ${TAG}_isa_stack.txt

# name inner dir == tarball name
tar czf ${TAG}_trace.tgz ${TAG}_out
ls -la ${TAG}_trace.tgz ${TAG}_isa_stack.txt
