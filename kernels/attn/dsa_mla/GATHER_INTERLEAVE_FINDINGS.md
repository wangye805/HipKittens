# Gather-interleave experiment — negative result (occ-1)

Reproduce: build the current-best flags **plus `-DGATHER_INTERLEAVE`** (the guarded code in
`hk_s2_occ1_stream.cpp` + `gather_phase1`/`gather_phase2_slice` in `sub_gather.cuh`).

## Idea (sound in principle)
The gather issue block sits serially at the loop top (~948 cyc/tile: phase-1 addr-calc+topk-read
~344, phase-2 the 16 `buffer_load_lds` DMAs ~800). Split it: phase-1 batched at top, **phase-2
DMAs interleaved 2-per-QK-chunk** to hide the DMA-issue stalls under the QK mma.

## Result: regresses 8629 → ~8900 (phase2 in QK) / ~9443 (phase2 in PV)

Correctness stays PASS; purely perf. Two independent costs, both traced:

### 1. The real constraint: occ-1 has NO idle issue slots
The QK `v_mfma`s issue back-to-back at +16 cyc (SIMD fully occupied), and the mma-feed stalls are
**hard `s_waitcnt lgkmcnt` halts** (wave frozen — can't issue *anything*, including a
`buffer_load`). So the interleaved DMA burst sits *between* mma groups as its own block —
**additive to the critical path**, exactly like the top-of-loop block. Hiding one instruction
stream inside another's stalls is fundamentally an **occ-2** thing (needs a 2nd wave). Confirmed
on the timeline: DMAs issue in a ~24-cyc burst, then the wave does mma+wait, then the next burst.

### 2. The held `goff[16]` spills to AGPR → `mma→accvgpr` hazard
Phase-1 computes `goff[16]` held across QK. VGPR is already 256 (QK peak = acc128+Q64+K64+...),
so `goff` spills to **AGPR**. Phase-2 reads it via `v_accvgpr_read` right after the QK mma burst →
the CDNA **`mma→accvgpr` AGPR-port structural hazard**: the matrix unit owns the AGPR read/write
port, so the read stalls until the in-flight mmas drain. Measured: a `s_nop 4` barrier absorbing
**~36 cyc** (varies 20–36 with mma backlog). This hazard is *absent* in the block form (gather's
AGPR reads don't sit next to an mma) — proven by comparison (block form has zero such `s_nop 4`).

### DC=32 (finer K-chunk) side-test
Reduces AGPR overflow 62→30 (K double-buffer shrinks) but VGPR stays 256 (wall is acc+Q, not K).
Gather-interleave still regresses (8899). Confirms the wall isn't the K buffer.

## Conclusion
The gather is **completion (L2-traffic) bound**, not issue-bound in a way interleaving helps. The
original early-block issue already hides completion over the full iteration (~6000 cyc). The occ-1
levers for the gather are (a) **reduce gather traffic** (A→B token-contiguous / fewer cache lines),
and (b) **occ-2** to actually hide the issue. Interleaving only relocates the cost and adds the
AGPR hazard.

The instrument was pointing at the right cost (`buffer_load` stall = L2 gather traffic), but that
stall is the DMA *doing work* — only reducing the work shrinks it.
