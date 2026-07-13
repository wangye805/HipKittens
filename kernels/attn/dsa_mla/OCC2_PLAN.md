# occ-2 plan — DSA-V4 sparse-MLA fwd (from the occ-1 `hk_s2_occ1_stream` baseline)

## Why occ-2 (what occ-1 cannot do)

The occ-1 steady tile is **6104 cyc** (vs Leon PR3833 **5396**). The dominant cost is the **`v_mfma` operand-feed stall (~1696 cyc/tile)** — the wave frozen on `s_waitcnt lgkmcnt` waiting for K/V LDS reads. At occ-1 (single wave) these are **hard halts with no idle issue slots**, so:
- gather issue can't be hidden (proven: interleave regresses — additive),
- softmax/mask/reshuffle only hide to the extent they fit *existing* slack (already exploited),
- the mma-feed stall itself is irreducible without a second instruction stream.

**occ-2 provides that second wave**: while wave A stalls on its LDS reads, wave B issues its mma/VALU, and vice-versa. This is the only lever that attacks the 1696 pole. Target from prior analysis: **~3139 cyc/tile**.

## The two binding constraints for occ-2 (2 waves/SIMD = 2 CTAs/CU)

1. **LDS (primary wall).** Current per-CTA LDS ≈ **128 KB** — dominated by `ks[2]` double-buffer (`st_bf<64,512>` × 2 = 2 × 64 KB). CU LDS ~160 KB → only **1 CTA fits** → occ-1. Need **≤ ~80 KB/CTA** for 2 CTAs.
   - Options: (a) single-buffer KV (drop the 2nd ks buffer) — loses gather/compute overlap, but occ-2's 2nd wave may compensate; (b) shrink the KV tile resident in LDS (stream a smaller KV window); (c) share/compress the topk + ps buffers.
2. **VGPR.** occ-1 is at 256 VGPR (2 waves × 256 = 512 = whole file → borderline). `acc` = 128 VGPR must stay VGPR (rescale ferries cost +1276 if AGPR). For occ-2 each wave needs to fit → **acc-split `[4×2]`**: split the [512,16] acc so each wave holds a slice, keeping per-wave VGPR in budget while acc stays VGPR.

## Plan sketch

1. **Cut LDS ≤ 80 KB**: start with single-buffered KV (`ks[1]`), measure occ (expect 2). Re-introduce overlap via the 2nd *wave* rather than the 2nd *buffer*.
2. **acc-split `[4×2]`**: partition acc so per-wave VGPR ≤ ~256 with 2 waves resident; keep acc in VGPR (cheap rescale preserved).
3. Validate occ=2 via `-Rpass-analysis=kernel-resource-usage` (Occupancy: 2).
4. Measure steady tile from ATT (median loop period), compare to occ-1 6104 and Leon 5396.

## De-risk references (already validated in prior sessions)
- occ-2 `[4×2]` HK skeleton (VGPR 229 / AGPR 0 / occ 2 / 0 spill) PASSED — see `project_occ2_gate_hk_gfx950` memory.
- `ds_read ‖ mfma` streams at occ-2 cadence PASSED (Bench B).
- The occ-2 target (~3139) reconciled: occ-2 is the right lever for the *mma-feed* pole (the earlier "occ-2 wrong lever" note was about the *gluon* kernel, not HK).

## Interaction with the occ-1 assembly-mode track
The occ-1 `art` (AGPR-park Q) work is orthogonal and complementary: it relieves the QK-peak VGPR, which *also* helps fit 2 waves. If occ-2 lands, re-evaluate whether Q-in-AGPR is still needed. Keep the two tracks on separate branches; merge the winner.
