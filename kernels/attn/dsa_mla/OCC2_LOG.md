# DSA-V4 sparse-MLA fwd — OCC-2 [4x2] kernel (`hk_s2_occ2`)

Isolated sub-workspace: `dsa_dev/hk_fwd_occ2/` (does NOT touch the occ-1 `hk_fwd/` tree).
Target branch for commit (from laptop): `dsa-mla-fwd-occ2` (separate from occ-1 `dsa-mla-fwd`).

Build (inside container `yewang_hk_mi355`, HK=`/workspace/mi355_flash_attn_v3/HipKittens`):
`bash RUN_occ2.sh`  → occ-gate + correctness + bench in one shot.
Best flags: `-mllvm -amdgpu-mfma-vgpr-form -DTILE_K=64 -DDC=64 -DMINWAVES=2 -DDBUF -DHK_PERMLANE_REDUCE -DREG_RESHUFFLE -DNO_PV_BARRIER`.

## CURRENT VERDICT (2026-07-13, measured on MI355)
**occ-2 [4x2] is CORRECT and the occ-2 gate PASSES, but it lands at ~PARITY with occ-1 — not the 3139
target.** Root cause (measured + reconciled with prior knowledge): the 2nd wave can only hide the QK-side
*latency* stalls (~571 cyc gain, from the NO_COMBINE ablation), NOT the PV V-read, which is `ds_read_b64_tr`
**crossbar-bandwidth bound (~122 B/cyc) and occupancy-IMMUNE** (memory `reference_gfx950_transpose_read_ceiling`).
That modest ~571 overlap gain is then cancelled by the cross-wave S-combine (~600–1350 cyc, intrinsic to
split-K). Net: parity.

### Ablation (bench cyc/tile, NTILES=36 NCTA=128 @2400MHz, back-to-back same session — RELATIVE valid)
| variant | bench | correct | note |
|---|---:|---|---|
| occ-1 (baseline) | 7658 | ✓ | |
| occ-2 NO_COMBINE (wrong math, overlap CEILING) | **7087** | ✗ | +7.5% — the pure 2nd-wave gain |
| occ-2 SYM bf16 1-barrier | 7685 | ~ (O 2e-4) | **parity**; combine ≈600 eats the gain |
| occ-2 2-barrier fp32 (DEFAULT, fully correct) | 8452 | ✓ (O 5e-5) | 2 barriers + serial half-idle |
| occ-2 SYM fp32 1-barrier | — | — | LDS OOM (32KB Scomb + ks[2] 128KB + topk > 160KB) |
| occ-2 REDUNDANT_QK (no combine, full QK x2) | 8617 | ✓ | WORST — doubles QK ds_read → worsens LDS-feed pole |

### The real lever to reach ~3139 (not yet done)
Per `reference_gfx950_transpose_read_ceiling`: occ-2 is V-read-bound and the only lever below ~3139 is a
**transpose-free row-major V read** (a 2nd LDS copy of V in row-major, since K=V, so PV reads b128 row loads
~240 B/cyc instead of the crossbar-capped ~122 B/cyc transpose). That frees the V-read from the crossbar so
the 2nd wave can actually fill the PV stalls. BUT a 2nd V LDS copy (+64KB) collides with the combine LDS
budget → requires LDS rework (single-buffer ks + re-overlap via the wave, or a combine-free key-split... which
moves the combine to acc). This is the substantive next phase; the correct occ-2 skeleton here is its base.

---
### (superseded framing) v1 CORRECT + occ-2 gate PASSED, initially slower
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
### 2026-07-13 (pm2) — v2 INDEPENDENT 8-wave + two-level tiling (`hk_s2_occ2b.cpp`) [DIAGNOSTIC]
- IDEA: independent 8-wave occ-2 (no combine) with two-level tiling (gather TILE_K=64 → LDS, compute TK_SUB
  sub-tiles) so VGPR is set by TK_SUB, not the gather TILE_K (the "chunked GEMM" correction).
- EDIT: new `hk_s2_occ2b.cpp` (+ _test/_bench/_att, ATT_capture_occ2b.sh). Each warp = a full head (occ-1
  algorithm); inner sub-loop over NSUB=TILE_K/TK_SUB with per-sub-tile online softmax. NW=8, MINWAVES=2.
  Fix: inner loop var `sub`→`sb` (shadowed HK sub()); do NOT `#pragma unroll` the NSUB loop (unrolling → 300B
  spill; rolled → 12B).
- WHAT I SAW (MI355, TILE_K=64 TK_SUB=32 DC=64 NW=8):
  - gate: VGPR 256, AGPR 0, **ScratchSize 12 B/lane** (near-clean).
  - correctness PASS all cases: O worst 5e-5 (one seed 2.7e-4), LSE 3.4e-4.
  - occ-2 CONFIRMED in HW: ATT shows **2 concurrent slots/SIMD** (sl0+sl1); the 2 waves/slot are successive
    CTAs over time.
  - PERF: bench 11914 cyc/**gather**-tile (=64 keys) vs occ-1 6104 steady/tile (64 keys) → **~1.57× WORSE
    per key.** ATT steady: sub-tiles at ~2900 (sub0→sub1) + ~6700 (→next tile) = ~9600/gather-tile; the two
    occ-2 slots run offset (sl0 ~2900/6700, sl1 ~4100/5600) so occ-2 overlap IS happening.
- WHY/LEARN: TK_SUB=32 halves the flash tile → **the per-sub-tile online softmax + `mul_col(acc)` rescale
  (128 VGPR) run 2× per 64 keys**, and that VALU/rescale doubling outweighs the occ-2 stall-hiding. Two-level
  tiling fixes the VGPR fit and keeps a large gather, but the softmax-doubling is a NEW cost occ-1 didn't pay.
  Net: correct occ-2, but slower. Artifacts: `occ2b_trace.tgz` (→ occ2b_out/ui_output_agent_8667_dispatch_2/),
  `occ2b_isa_stack.txt`.
- NEXT: the rescale is the tax. Options: (a) don't rescale acc every sub-tile — keep the running max/denom and
  rescale acc ONCE per gather-tile (or defer to the epilogue, flash-decoding style) so `mul_col(acc)` is 1×/tile
  not NSUB×; (b) larger TK_SUB with Q streamed to hold VGPR (TK_SUB=64 → NSUB=1 = no doubling, but needs
  −64 VGPR from Q-stream to fit); (c) accept the softmax 2× only for s/P (cheap) but not acc.


### 2026-07-13 (pm) — combine ablation + steady-tile ATT: occ-2 is V-read-bound → PARITY [DIAGNOSTIC]
- IDEA: find why occ-2 regressed; isolate combine cost + measure real steady tile + wave overlap.
- MEASURE: ATT steady tile (occ2 2-barrier, GRID_N=512 NTILES=18) = ~7124 cyc/wave (median of 18 V-load-
  cluster periods); 2 waves/SIMD confirmed in HW (se*_wv0 & wv1 present); wave offset ~1924 cyc (partial
  stagger, not lockstep). Combine ablations (bench table above): NO_COMBINE 7087 (overlap ceiling, +7.5%),
  SYM bf16 7685 (parity), 2-barrier 8452, REDUNDANT_QK 8617.
- WHY/LEARN: the 2nd wave only hides QK latency stalls (~571). The PV V-read is `ds_read_b64_tr`
  crossbar-BANDWIDTH bound (~122 B/cyc), occupancy-IMMUNE (prior: reference_gfx950_transpose_read_ceiling),
  so occ-2 can't hide it. The split-K S-combine (~600–1350) then cancels the ~571 gain → parity. REDUNDANT_QK
  loses because it doubles QK ds_reads (worsens the LDS-feed pole) — confirms the kernel is LDS-feed bound,
  not matrix-throughput bound. Real lever = transpose-free row-major V (needs LDS rework). Flags added:
  NO_COMBINE, SYM_COMBINE[+SYM_FP32], REDUNDANT_QK. DEFAULT kept = 2-barrier fp32 (fully correct, O 5e-5).

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
