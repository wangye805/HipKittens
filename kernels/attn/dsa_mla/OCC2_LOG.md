# DSA-V4 sparse-MLA fwd — OCC-2 [4x2] kernel (`hk_s2_occ2`)

Isolated sub-workspace: `dsa_dev/hk_fwd_occ2/` (does NOT touch the occ-1 `hk_fwd/` tree).
Target branch for commit (from laptop): `dsa-mla-fwd-occ2` (separate from occ-1 `dsa-mla-fwd`).

Build (inside container `yewang_hk_mi355`, HK=`/workspace/mi355_flash_attn_v3/HipKittens`):
`bash RUN_occ2.sh`  → occ-gate + correctness + bench in one shot.
Best flags: `-mllvm -amdgpu-mfma-vgpr-form -DTILE_K=64 -DDC=64 -DMINWAVES=2 -DDBUF -DHK_PERMLANE_REDUCE -DREG_RESHUFFLE -DNO_PV_BARRIER`.

## CURRENT VERDICT
**v1 CORRECT + occ-2 gate PASSED, but currently SLOWER than occ-1 (perf tuning needed).**
- Occ-2 gate PASS (measured on MI355, gfx950): VGPRs **220**, AGPRs **0**, ScratchSize **0**, SGPR 50. Register
  limit → floor(512/220)=2 waves/SIMD. LDS = ks[2] 131072 + Scomb 16384 + topk(NTILES=36) 9216 = 156.7 KB
  → 1 CTA/CU = 8 warps = **2 waves/SIMD**. Occ-2 confirmed structurally.
- Correctness PASS all cases (seeds 0-2 × sink 0/1 × ninv 13/40/100): O worst **5e-5** (bar 1e-4),
  LSE worst **2.8e-4** (bar 6e-4).
- **Perf REGRESSION (honest):** back-to-back same-clock bench NCTA=128 NTILES=36 @2400MHz:
  occ-1 **7658** cyc/tile vs occ-2 **8474** cyc/tile (+11% SLOWER). Wall/launch 0.115 vs 0.127 ms.
  → The 2nd wave is NOT hiding the mma-feed stalls yet. Diagnosis: the CTA barriers (4/tile: gather-vis +
  2× S-combine + tile-bottom) re-sync the two waves each tile → they drift into LOCKSTEP and hit the
  `s_waitcnt lgkmcnt` mma-feed stall SIMULTANEOUSLY (both stall together = no fill), and the 2-barrier fp32
  combine adds serialized half-idle sections. This is exactly the "slot staggering" problem from
  KERNEL_DESIGN_AXES worked-example A (MXFP8 GEMM): 2 waves on one matrix core must be OFFSET so they never
  hit the mma cluster at the same time. Steady-tile ATT pending to confirm the lockstep.

### Next levers (to convert the correct occ-2 into a win)
1. **Wave staggering** (the MXFP8 lever): a conditional prologue offset so the 2 waves on a SIMD run half a
   tile apart → wave B issues mma while wave A is in its feed-stall, and vice-versa. Hard part: the CTA
   barriers re-sync; may need per-mg (not CTA-wide) sync, or a skewed schedule the barriers don't undo.
2. **Cut the S-combine cost**: 2 fp32 barriers → try (a) symmetric 1-barrier (2 slots, 32 KB; needs LDS
   trim, e.g. shrink ks or bf16 combine if the O bar holds), or (b) fold the combine barrier into the
   existing gather/tile barriers.
3. Only after 1-2: re-measure steady tile vs 6104 / 5396 / 3139.

## Design (why each choice)

occ-1 pole = the `v_mfma` operand-feed stall (~1696 cyc/tile): at occ-1 the mma issues back-to-back and the
feed-stalls are hard `s_waitcnt` halts with no idle issue slots. occ-2 (a 2nd resident wave) is the only
lever. Target ~3139 cyc/tile (from 6104 occ-1; Leon PR3833 = 5396).

**Topology:** ONE CTA per token, 8 warps = 4 M-groups × 2 sub-warps → 2 waves/SIMD = occ-2 (via a bigger
CTA, NOT 2 independent CTAs). `mg = warpid>>1` (query-head, QB=16), `sub = warpid&1` (D-half owner).

**Split-K QK + split-N PV (NOT redundant full-QK).** Each sub reduces its D-half → partial `s[64,16]`;
combine → full; both run identical softmax; each does PV over its own D_V-half → `acc[256,16]` = 64 VGPR.
- *Why split-K, first-principles:* total matrix work across the 2 waves = `2·(QK/2) + 2·(PV/2) = QK+PV`
  — SAME as occ-1. Redundant full-QK in both subs = `2·QK + 2·(PV/2) = 1.5·(QK+PV)` → 1.5× the mma, which
  is bad *exactly* at occ-2 where the 2 waves share ONE matrix core (would become mma-throughput-bound).
- *Why acc-split `[256,16]`:* full `acc[512,16]` = 128 VGPR; occ-2 needs per-wave (arch VGPR + AGPR) ≤ 256
  (gfx950 combined file = 512/lane, occ = floor(512/per-wave)). Half-acc = 64 VGPR keeps acc in VGPR (no
  AGPR rescale ferry, which costs +1276 cyc — measured). Est. QK peak ≈ 192 VGPR/wave → occ-2 comfortable.

**S-combine = fp32, hand-rolled LDS, 2-barrier.**
- *Why fp32 (not bf16):* the partial raw score is ~2.5; bf16 ulp ~0.016 → after ×scale(0.044) ≈ 7e-4 score
  error → ~5e-4 relative P → risks the O ~1e-4 bar. Keep the combine in fp32.
- *Why hand-rolled (not HK `store`/`load`):* HK's tile shared↔register path has NO fp32 case (bf16/half/fp8
  only; fp32 hits `static_assert(false, "Unsupported type")`). But sub0's lane L and sub1's lane L hold
  IDENTICAL (key,query) elements (the rt layout depends only on the tile type, not the warp), so a plain
  `__shared__ float Scomb[mg][lane*16 + local]` with the same index in both subs is exact. 16 KB.
- *Protocol (2 barriers, single 16KB slot):* sub1 writes partial → `__syncthreads` A → sub0 reads+adds+
  writes full back → `__syncthreads` B → sub1 reads full. Both subs end with full `s`.

**LDS budget:** `ks[2]` (st_bf<64,512>×2) = 128 KB + `Scomb[4][1024]f` = 16 KB + `topk_all` (NTILES·64·4)
≈ 149 KB (NTILES=18) < 160 KB/CU → 1 CTA/CU → the 8-warp CTA gives occ-2. Double-buffer KV KEPT (single-
buffering would expose the L2-bound gather; the 2nd *wave* is the overlap lever, the 2nd *buffer* stays).

## Ports carried from occ-1 (all correctness-identical there)
- `-DDBUF` VGPR double-buffer of K/V chunks (`ds_read(next) ‖ mma(current)`, Bench B validated).
- `-DHK_PERMLANE_REDUCE` col_max/col_sum via permlane16/32_swap (patch in HK reductions.cuh).
- `-DREG_RESHUFFLE` in-register rt_16x16→rt_32x16 P relayout (no LDS P-roundtrip). `ps` buffer dropped.
- mask = `topk & 0xFF800000` additive -inf, tkv hoisted into QK. Applied to full `s` AFTER combine.

## Open risks to resolve on first remote build
1. Occupancy gate: confirm `Occupancy: 2`, `AGPRs: 0`, `0 spill` from `-Rpass-analysis` (the raison d'être).
2. HK API shapes: `subtile_inplace<64>(acc,c)` on `[256,16]`; `transpose([16,256],[256,16])`; mma tile-shape
   acceptance at NC=4. Build will confirm.
3. Correctness: O worst ~1e-4, LSE ~6e-4 vs the bf16-faithful CPU ref (`hk_s2_occ2_test.cpp`).
4. Steady tile: capture via `ATT_capture_occ2.sh`, median loop period; compare to 6104/5396/3139 at matched sclk.

## Next levers (after v1 lands + measures)
- If the 2-barrier combine serializes too much: try symmetric 1-barrier (2 slots, 32 KB) by shrinking ks
  (single-buffer + re-overlap via the wave) OR bf16 combine (check the O bar first).
- If mma-throughput-bound: the 16x16x32 mma shape is already assumed; re-check.
- Merge with the occ-1 assembly-mode (Q-in-AGPR) track only if it further relieves the QK VGPR peak.

## LOG (newest first)
### 2026-07-13 — v1 [4x2] CORRECT + occ-2 gate PASS, perf regresses [DIAGNOSTIC]
- IDEA: attack the occ-1 mma-feed pole with a 2nd resident wave via the [4x2] cooperative split (axis 5→2).
- EDIT: new `hk_s2_occ2.cpp` (+ _test/_bench/_att, RUN_occ2.sh, ATT_capture_occ2.sh); based on
  `hk_s2_occ1_stream.cpp` with warpid→(mg,dh), split-K QK, fp32 2-barrier S-combine, acc[256,16], split-N PV.
- FIX 1 (compile): renamed the D-half var `sub`→`dh` (it shadowed HK's `sub()` softmax op).
- FIX 2 (correctness): the half-tile store needs a TILE-typed coord — `store(...,coord<>{0,mg,0,dh})` left
  `unit_coord` in the non-tile branch so the col index wasn't scaled by tile width (dh=1 landed at col 1, not
  256 → D[256:512] untouched, half of O = -999). `coord<OtT>{0,mg,0,dh}` scales c*256. Confirmed by a
  D-region untouched-count diagnostic (255 untouched + 1 at D[256] = exactly the c=1 signature). → PASS.
- WHAT I SAW: gate VGPR220/AGPR0/occ2/0-spill; correctness O 5e-5 / LSE 2.8e-4 all cases; but bench
  8474 vs occ-1 7658 (+11% slower, same clock, back-to-back).
- WHY/LEARN: correct occ-2 structure is buildable & fits, but the 2 waves stall in lockstep (barrier re-sync)
  so no mma-feed hiding + combine overhead. The structural unlock is done; the WIN needs wave staggering
  (see Next levers). Matched-config A/B held (both DBUF/DC64/TILE_K64/NTILES36/NCTA128).
