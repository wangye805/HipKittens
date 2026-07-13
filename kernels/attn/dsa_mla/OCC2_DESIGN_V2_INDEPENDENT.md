# occ-2 design v2 — INDEPENDENT 8-wave (BLOCK_H=128, TILE_K=32), no cross-wave combine

Status: DESIGN (analysis only; not yet built). Supersedes the cooperative [4x2] approach (`hk_s2_occ2.cpp`,
see OCC2_LOG.md) as the primary occ-2 path. Motivated by the Primus-Turbo/flydsl fwd structure
(BLOCK_H=128, TILE_K=32, 8 waves, one head per wave) and by the measured failure of v1.

---

## 0. Why pivot away from the cooperative [4x2] (measured, this session)

v1 split D across 2 sub-warps → each wave holds half the acc (64 VGPR) → fits occ-2. But splitting D forces
a **cross-wave S-combine** before softmax (partial scores must be summed). Measured on MI355:

| variant | bench cyc/tile @2400 | note |
|---|---:|---|
| occ-1 (baseline) | 7658 | |
| occ-2 [4x2] **NO_COMBINE** (wrong math) | **7087** | the pure 2nd-wave overlap gain = ~571 (+7.5%) |
| occ-2 [4x2] SYM bf16 combine | 7685 | parity — the combine (~600) eats the gain |
| occ-2 [4x2] 2-barrier fp32 combine | 8452 | worse |

The overlap gain is real (~571) but the combine cancels it exactly → parity. **The combine is intrinsic to
splitting D.** v2 removes it by NOT splitting D: each wave does a whole head, exactly like occ-1.

---

## 1. The v2 structure

- **BLOCK_H = 128** = 8 heads × QB=16 queries. **8 warps, one head per warp**, fully independent (occ-1's
  per-warp algorithm, verbatim). No D-split, no partial-S, no combine, **zero cross-wave sync added.**
- 8 warps/CTA = **2 waves/SIMD = occ-2** (the CTA itself provides occupancy 2, same as v1).
- **TILE_K = 32** — the VGPR enabler (see §4). KV tile = 32 keys.
- **1 CTA per (token, head-block-of-8)**; grid = tokens × ceil(H/8). Same total work.
- **Shared `ks`**: in this kernel topk is **per-token** (Tkg indexed by `tok=blockIdx.x`, shared across all
  heads — as in occ-1). So all 8 warps attend the SAME gathered KV tile → **one `ks[2]` buffer**, filled
  cooperatively by all 512 threads. No per-wave gather blow-up.

The only shared state is `ks` (read by all) + the cooperative gather (write) → the SAME single
gather-visibility barrier occ-1 already has. No new barriers. This is the whole point: v2 = occ-1 kernel with
`NW=8, TILE_K=32, MINWAVES=2` + the resource work in §4 + the gather-overlap work in §3.

---

## 2. What changes, by axis (vs occ-1)

**Axis 1 — decomposition.** BLOCK_H 64→128 (4→8 heads/CTA); TILE_K 64→32 (NTILES doubles). More, smaller
tiles; more heads per shared gather (see §3). Grid unchanged in total work.

**Axis 2 — pipelining / latency hiding (the reason occ-2 wins here).** Two hides, both impossible at occ-1:
  1. *mma-feed (QK) latency stalls.* At occ-1 the mma issues back-to-back and the `s_waitcnt lgkmcnt` feed
     stalls are hard halts with **no idle issue slots** (memory: gather-interleave regressed for this reason).
     The 2nd wave fills them. This is the measured ~571 gain — now kept, because there's no combine to pay it back.
  2. *the gather* (see §3) — the second, larger opportunity the user flagged.

**Axis 3 — data movement / the gather (see §3).** ks LDS halves (TILE_K=32: st_bf<32,512>×2 = 64 KB vs 128).
V-read crossbar cap unchanged (see §5).

**Axis 4 — layouts.** Unchanged tile layouts; `s`=[32,16], K/V chunks=[32,DC], acc=[512,16] (full head).

**Axis 5 — occupancy / resource.** VGPR is now THE constraint (§4); LDS is comfortable (64 KB); occ-2 from
the 8-warp CTA.

---

## 3. The gather — the piece occ-1 never overlapped (and why v2 can)

Current occ-1 loop: `gather(ks[nxt])` (async) → `s_waitcnt vmcnt(NDMA)` (drain tile j) → barrier → compute(j).
The double-buffer overlaps the gather **completion** of tile j+1 with the compute of tile j. But two problems
at occ-1:
- The gather **issue** (address calc + topk LDS reads + the `raw_buffer_load_lds` batch) needs issue slots;
  at occ-1 there are none free (back-to-back mma) → interleaving it regressed (measured, memory
  `project_dsa_v4_fwd_bottleneck`). So the issue cost is exposed.
- The gather is **L2-completion bound** (~the dominant traffic). If a tile's compute is shorter than the
  gather completion, the gather is exposed even with the double-buffer. At BLOCK_H=64 the compute-per-gather
  is only 4 heads → often < gather time → **gather not fully hidden ("not overlapped yet").**

v2 fixes both:
- **occ-2 gives idle issue slots**: while wave A issues/waits on the gather, wave B issues mma. So gather
  **issue** is finally free to interleave (the occ-1 regression was an occupancy problem, not a placement one).
- **BLOCK_H=128 doubles compute-per-shared-gather** (8 heads share one gather → 2× the compute to hide the
  same L2 completion). Combined with occ-2's 2 in-flight waves, the gather **completion** hides under compute.

Net: v2 should convert the gather from partly-exposed to (largely) hidden — a win occ-1 structurally can't get.
Caveat: total KV **L2 traffic is unchanged** (topk_len × 512 × 2B); if the kernel is *purely* L2-bandwidth
bound, hiding only helps up to the bandwidth floor. TILE_K=32 makes gathers smaller and 2× more numerous
(more issue overhead, but each amortized over 8 heads). Design knob: revisit whether to gather TILE_K=64 into
LDS but stream compute at 32 (decouple gather granularity from compute granularity) if issue overhead bites.

---

## 4-PRE. Correction: the compute sub-tile (TK_SUB), not the gather TILE_K, drives VGPR

The v1/occ-1 kernel streams the GEMM over D (`DC` loop) but loads **all TILE_K keys** into the staged
register chunk `k_c=[TILE_K,DC]` — so VGPR scales with TILE_K *only because the key dimension is not chunked*.
The correct structure is **two-level tiling** (CUTLASS mainloop-tile vs mma-instruction-tile):
  - **outer = gather tile**: gather `TILE_K` keys HBM→LDS (`ks`), kept LARGE for gather amortization.
  - **inner = compute sub-tile**: loop `TK_SUB` keys at a time LDS→reg→mma→online-softmax→acc.

Under this, the staged operands are `[TK_SUB, DC]` and (with per-sub-tile online softmax) `s,P=[TK_SUB,QB]`
— **all set by TK_SUB, independent of the gather TILE_K**. `acc[512,16]=128` and `Q=64` are unchanged.
So the footprint is `acc(128) + Q(64) + small(TK_SUB)` ≈ **~200–216** at TK_SUB=16–32 → clean occ-2 **at any
gather TILE_K**. This removes the false tradeoff v2 was built on: keep **gather TILE_K=64** (128KB ks, the max
that double-buffers <160KB; fewer/bigger gathers) AND fit occ-2 via a small TK_SUB. The only added cost is one
`mul_col(acc)` rescale per sub-tile — identical to what a small TILE_K would incur, so no net loss, and the
gather wins. This SUPERSEDES the "shrink TILE_K to 32" framing below.

Measured proof that (un-chunked) TILE_K drives register pressure — NW=8 DC=64, gate-only compile:
| config | VGPRs (clamped 256) | ScratchSize spill |
|---|---|---|
| TILE_K=64 | 256 | **228 B/lane** (~57 VGPR) |
| TILE_K=32 | 256 | **20 B/lane** (~5 VGPR) |
Same clamp, 11× the spill → TILE_K staging is real VGPR (the KV *tile* is in LDS/free; the MFMA operand
*staging* out of LDS is not). Chunking the key dim makes this spill TK_SUB-controlled instead.

## 4. (superseded framing) shrinking TILE_K to fit VGPR

Each independent wave holds the FULL head → `acc[512,16] = 128 VGPR` (immovable; must stay VGPR — AGPR costs
+1276 rescale ferries, measured). occ-2 needs per-wave (arch VGPR + AGPR) ≤ 256 (gfx950 combined file 512/lane,
occ = floor(512/per-wave)), with **0 spill** (spill on a knife-edge kernel is a nondeterminism/perf risk).

**Measured data point (this session):** occ-1 kernel at `NW=8, TILE_K=32, DC=64, MINWAVES=2` →
**VGPRs 256, AGPRs 0, ScratchSize 20 bytes/lane (SPILL).** So TILE_K=32 alone gets close but does NOT cleanly
fit — ~5 VGPR over. The design must shave the QK/PV working set below 128 (since acc eats the other 128):

**K-dbuf and V-dbuf are NEVER co-resident** — K lives only in the QK loop, V only in the PV loop (separate
phases, no prefetch of V during QK in the current code). So each phase has its own peak and only the MAX binds:

Budget (per lane, TILE_K=32):
| component | DC=64 | DC=32 | live in |
|---|--:|--:|---|
| acc [512,16] f32 | 128 | 128 | whole loop (immovable) |
| Q q_arr (full D, invariant) | 64 | 64 | whole loop (held resident across tiles) |
| K dbuf (2×[32,DC]) | 32 | 16 | **QK only** |
| V dbuf (2×[32,DC]) | 32 | 16 | **PV only** (not with K) |
| s [32,16] + tkv[16] + rowvecs | ~24 | ~24 | QK→softmax |
| pop [32,16] bf16 | 4 | 4 | PV |
| **QK-phase peak = acc+Q+K+s+tkv** | **128+64+32+24 = 248** | **128+64+16+24 = 232** | binding |
| PV-phase peak = acc+Q+V+pop | 128+64+32+4 = 228 | 128+64+16+4 = 212 | not binding |

So the **binding peak is QK = 248** at DC=64 (V does NOT appear here). Yet the measured gate was **256 + 20B
spill (~276 needed)** → there is **~28 VGPR of real overhead above the clean accounting** (mask `float bias[16]`
= 16 VGPR when it's live, gather/loop/address SGPR→VGPR temporaries, DBUF bookkeeping, wider mma liveness).

**Implication of the corrected accounting (this is the key consequence):** since only **K** (not V) sits in the
binding QK peak, **DC only shaves the K side → DC=64→32 saves ~16 VGPR of the binding peak (248→232), not 32.**
So DC=32 alone brings ~276 → ~260 — likely STILL a few over 256. **The real fat is Q (64 VGPR, held resident
for the whole loop although it's only used in QK).** Levers, in order:
  1. **Stream Q** (don't hold all D-chunks resident; load the DC-chunk from LDS/regs inside the QK loop like K)
     → −up to 64 VGPR of the binding peak. Q is invariant across tiles, so this is a reuse trade (re-read Q
     each tile). This is almost certainly REQUIRED for a clean 0-spill occ-2, more than DC is.
  2. **DC=32** → −16 more (K side). Compose with (1).
  3. Shrink mask-`bias`/`tkv`/rowvec liveness (the ~28 overhead) — e.g. apply the mask per-chunk so `bias[16]`
     isn't all live at once.
Target: QK-phase peak ≤ ~248−overhead so total ≤ 256, AGPR 0, **0 spill**. DC=32 → NCH=16 finer chunks; the
Bench-B streaming result (ds_read‖mma at the floor) held at ILP4/8 so finer chunks should still stream.

---

## 5. The ceiling — still V-read bound (be honest about the target)

The PV V-read uses `ds_read_b64_tr`, which is **crossbar-bandwidth bound ~122 B/cyc and occupancy-IMMUNE**
(memory `reference_gfx950_transpose_read_ceiling`). Two waves both reading the shared ks V region = 2× crossbar
requests to the same banks → PV does **not** speed up under occ-2. So v2's wins are QK-stall hiding + gather
hiding, NOT PV. Expected: beat occ-1 meaningfully (target the ~7% NO_COMBINE ceiling **plus** the gather hide,
plausibly landing between occ-1's 6104 and the ~5000s), but **not** the 3139 model — that model is V-read-bound
and needs the separate **transpose-free row-major V** lever (a 2nd LDS copy of V in row-major, since K=V, so PV
reads b128 rows ~240 B/cyc). That lever is orthogonal and composes on top of v2; it's the path to ~3139/2500.

---

## 6. Risks / open questions
- **VGPR knife-edge.** 0 spill at DC=32 must be confirmed; spill reintroduces nondeterminism/perf loss. If
  DC=32 still spills, stream Q.
- **NTILES doubles (TILE_K=32).** 2× loop iterations/barriers/gather-calls — the per-tile overhead must not
  exceed the occ-2 gain. Mitigation: gather at 64 / compute at 32 (decoupled granularity).
- **L2-bandwidth floor.** If gather is purely BW-bound, hiding caps at the BW floor; measure the compute:gather
  ratio at BLOCK_H=128 to confirm we've moved to compute-bound.
- **PV crossbar cap** persists → ceiling above 3139 without row-major V.
- Does topk-per-token truly hold for the target DSV4 config (all 8 heads share ks)? Confirmed for THIS kernel
  (Tkg indexed by tok). Re-verify against the official spec before shipping.

## 7. Implementation plan (minimal, staged)
1. `hk_s2_occ2b.cpp` = copy of `hk_s2_occ1_stream.cpp`; NW=8, MINWAVES=2, and **two-level tiling** (§4-PRE):
   gather TILE_K=64 HBM→LDS (outer), compute in TK_SUB=16/32 inner sub-tiles with per-sub-tile online softmax.
   This is the real occ-2 fit: VGPR set by TK_SUB (~200–216) → clean 0-spill occ-2 with a large gather tile.
   If Q's 64 VGPR still pushes it over, stream Q too (secondary). Confirm gate (VGPR ≤256, AGPR 0, 0 spill, occ 2).
2. Correctness (bf16 ref, BLOCK_H=128): O ~1e-4, LSE ~6e-4.
3. Bench + ATT steady tile vs occ-1 (matched clock). Expect a real win from QK-stall hiding alone.
4. Add gather interleave (now that occ-2 provides idle slots): re-enable the `GATHER_INTERLEAVE` path (it
   regressed at occ-1 — re-test at occ-2). Measure the gather-hide gain.
5. Only then: transpose-free row-major V (2nd LDS copy) toward the 3139/2500 floor.
