# DSA-V4 sparse-MLA fwd — streamed occ-1 kernel (`hk_s2_occ1_stream`)

Streamed occ-1 fwd (TILE_K=64, DC=64, QB=16, NW=4, D=512 dense K=V, base-2 softmax, sink, LSE).
Build flags: `-mllvm -amdgpu-mfma-vgpr-form -DTILE_K=64 -DDC=64 -DMINWAVES=1 -DDBUF -DHK_PERMLANE_REDUCE -DREG_RESHUFFLE -DNO_PV_BARRIER` (current best).

## Steady-state tile wall (ATT trace, median loop period) — the real metric

Measured per-tile from the ATT wave timeline (NOT the bench cyc/tile, which folds in prologue/epilogue):
**ours 6104 cyc/tile vs Leon PR3833 5396 cyc/tile (~13% behind).** Bench cyc/tile (full-kernel/NTILES) went 9669 → 8629 over this session.

## Wins this session (all correctness-identical, O worst ~1e-4, LSE ~6e-4)

| step | bench cyc/tile | mechanism |
|------|---:|-----------|
| start (msv mask + bpermute reduce) | 9669 | |
| mask: v_cndmask → additive → **single `& 0xFF800000`** → **hoisted into QK** | 9552 | mask = pure `v_and`+`v_fmac`, no vcc serialization; topk read hoisted (latency hidden under QK mma) |
| **col_max/col_sum: `ds_bpermute` → `permlane16/32_swap`** | 9209 | `-DHK_PERMLANE_REDUCE` (patch in `include/.../tile/reductions.cuh`): no LDS datapath, no broadcast; relieves lgkmcnt contention w/ mma-feed. −340 |
| **P-roundtrip `s_barrier` removed** | 9086 | per-warp `ps[warpid]`, intra-warp reshuffle needs only `lgkmcnt` |
| **P-roundtrip → in-register permlane reshuffle** (`-DREG_RESHUFFLE`) | 8629 | `rt_16x16 → rt_32x16` via `permlane32_swap`+`permlane16_swap` (derived+verified), replaces flat_store+load LDS roundtrip. −407 |

Key reusable findings:
- **permlane > ds_bpermute for cross-lane reductions** on gfx950 (same-head stride-16 pattern): `permlane16/32_swap` use the lane-swap net, not LDS; butterfly all-reduce leaves result in all lanes (no broadcast).
- **P-roundtrip relayout in-register**: `t=permlane32_swap(A,B); {d[k],d[k+2]}=permlane16_swap(t.x,t.y)` for `rt_16x16→rt_32x16` (per (tile i, packed k): A=tile[2i].d[k], B=tile[2i+1].d[k]). Verified 0/1024.
- **Mask**: `topk & 0xFF800000` → -inf bias directly (valid idx < 2^23 → 0; -1 → -inf), no cmp/cndmask.

## Register map (per lane, current config) — the occ-1 wall

| component | VGPR | live range |
|---|--:|---|
| acc `rt<f32,512,16>` | 128 | always (must stay VGPR — AGPR costs +1276 rescale ferries, measured) |
| Q `q_arr[8]` | 64 | prologue→QK |
| K `kb0+kb1` | 64 | QK |
| V `vb0+vb1` | 64 | PV |
| s / tkv / pb / pop / rowvecs | 16/16/8/8/~8 | |

**Peak = QK phase: acc128 + Q64 + K64 + s16 + tkv16 ≈ 296 → 256 VGPR + ~62 AGPR overflow.** That overflow (goff/spills read via `v_accvgpr_read` next to the mma) triggers the CDNA `mma→accvgpr` AGPR-port hazard.

## Negative results (documented, not dead ends)

- **Gather interleave into QK/PV** (`-DGATHER_INTERLEAVE`, guarded off): regresses (8629→~8900–8950). At occ-1 the mma issues back-to-back and the mma-feed stalls are hard `s_waitcnt` halts — **no idle issue slots** to hide the gather in; it's additive regardless of placement. Also the held `goff[16]` spills to AGPR → `mma→accvgpr` +36-cyc `s_nop` hazard. Gather is **completion(L2-traffic)-bound**, not issue-bound — real lever is traffic reduction. See branch `dsa-mla-fwd-gather-interleave`.
- **Finer K-chunk (DC=32)**: −AGPR (62→30, K double-buffer shrank) but VGPR stays 256 (wall is acc+Q, not K) → marginal (8724).
- **acc→AGPR (drop vgpr-form)**: +1276 (rescale ferries). acc must stay VGPR.

## Next levers
1. **occ-1 assembly-mode**: park Q in AGPR via HK `art` tiles (`kernels/attn/bkwd` does exactly this — Q/K/V operands in AGPR, compute in VGPR). Frees QK-peak VGPR (296→232), kills the overflow hazard. Requires assembly-mode QK rewrite (art `load<N,M>` + `mma_ABt<N,M,R>`).
2. **occ-2** (the structural unlock): LDS-bound (128KB ks double-buffer → need ≤80KB) + acc-split `[4×2]`. See `OCC2_PLAN.md`.
