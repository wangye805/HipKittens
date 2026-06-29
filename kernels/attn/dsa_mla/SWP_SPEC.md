# HK DSA-MLA fwd SWP rewrite — target the latest gluon early-gather pipeline (PR2922 c4b07fe5a)

## Reference (gluon, gfx950, 3.03 ms @ T=4096 H=128 TOPK=1152, -9.3% over the non-pipelined 3.34)
`aiter/.../_gluon_kernels/gfx950/attention/dsa_fwd_v4_gluon.py` (commit c4b07fe5a). Structure:
- **2-deep KV shared buffer** (smem_klora[2], smem_krope[2]); TILE_K=32; NO 3rd buffer.
- **One-tile lag**: carry `S_prev` = masked/scaled QK of tile t (small [BLOCK_H,TILE_K] reg tile).
- Per loop iter (the cross-tile overlap):
  1. `wait_group(0)` — drain K[t+1] (cur_buf) before QK reads it.
  2. **evacuate V[t]** from pv_buf(1-cur_buf) → registers (`permute.load(dot_v_b)`); frees that buffer.
  3. **gather tile t+2** into the freed buffer (async buffer_load_to_shared); overlaps the MFMAs below.
  4. deep-prefetch topk[t+3].
  5. **QK(t+1)** from cur_buf (2 mma) → S_cur; overlaps gather + softmax.
  6. **softmax(S_prev=tile t)** (max/exp/sum, acc rescale) [VALU]; overlaps the QK MFMA.  ← the win
  7. **PV(t)** = P@V_lora from registers [MFMA]; overlaps the in-flight gather.
  8. promote: S_prev=S_cur, cur_buf flip.
- WARM-UP (gather K1, QK0→S_prev, no softmax/PV) + PRE-DRAIN + DRAIN tail (last 2 tiles).
- Bit-exact vs non-pipelined; 404 VGPR, 0 spill.

## HK mapping (keys-as-rows discipline from hk_fwd3, now stable-correct)
Build `hk_fwd4.cpp` from hk_fwd3's validated compute, adding the pipeline. Keep hk_fwd3 as the stable
single-buffer reference + correctness oracle (hk_fwd3_test).
- **2-deep shared**: `KlS ks[2]`, `KrS krs[2]`; per-warp `ps[NW]` roundtrip unchanged.
- **Carry** `ST S_prev` (= [TILE_K,QB] col_l rt_16x16, small) across the loop.
- **Async gather** = `raw_buffer_load_lds` (global→LDS DMA, vmcnt-tracked) with the FORWARD-swizzle
  addressing validated in gcheck (16B/thread int4-equivalent). This is the foundation step — must be
  correct + overlap-capable (vmcnt waits, not full s_waitcnt(0)).  ⚠ the synchronous int4 gather tripped
  the knife-edge; the async + explicit scheduling is what both speeds up AND stabilizes codegen.
- **V evacuation**: `load(v_l, ks[1-cur])` (col_l rt_32x16) into registers BEFORE gathering t+2 into ks[1-cur].
- **softmax∥QK overlap**: emit the QK mma_ABt, then the softmax VALU, then PV mma_AtB, fenced with
  `__builtin_amdgcn_sched_group_barrier(MFMA_MASK,...)/(VALU_MASK,...)` (gqa's sched_barrier_pairs) so the
  compiler keeps the softmax VALU under the QK MFMA shadow and can't re-cluster. This PINS the schedule →
  off the knife-edge.
- **Register relief** (if needed for 2-buffer + V-in-regs): subtile the PV mma (consume V in 16-col chunks,
  gqa subtile_inplace) so we never hold the full [32,512]=128-VGPR V tile. gluon stays at 404 VGPR via the
  dedup headroom; HK may need subtiling to fit 2 buffers + S_prev + V-regs under 256 VGPR.

## Build order (incremental, validate each vs hk_fwd3_test / bench)
1. **Async gather foundation** [the hard part — mechanism resolved 2026-06-28]:
   ⚠ `raw_buffer_load_lds` writes LDS **lane-consecutively**: `G::load` does `s_mov_b32 m0, <lds_byte>`
   then `raw_buffer_load_lds(SRD, (as3_uint32_ptr)0, 16, swizzled_offsets[i], SOFF, 0, cache_all)` — i.e.
   M0 = LDS base, lds_ptr arg = 0, and each lane writes LDS[M0 + lane*16] reading global[per-lane voffset].
   So the per-lane FORWARD-swizzle LDS addressing (my sync int4 gather) does NOT work for the DMA — it
   silently writes lane-consecutive (gcheck async = 96% wrong, confirmed). The async gather must use the
   **REVERSE-swizzle**: for each lane-consecutive LDS slot, compute which logical (k,d) it holds (inverse
   swizzle), then set the per-lane global voffset = topk[k]*stride + col_off + d. This is exactly the
   ORIGINAL hk_fwd.cpp `gather_to_shared` (raw_buffer_load_lds + LDS-offset->g_row/g_col reverse map) — that
   gather was correct in isolation; reuse its addressing, set M0 per G::load, lds_ptr=0, size=16. Validate
   in gcheck (ASYNC_GATHER path, currently the WRONG forward-swizzle version — replace it).
   Alternative: study G::load's `prefill_swizzled_offsets` and write a topk-indirected variant.
   ✅ DONE 2026-06-28: reverse-swizzle async gather VALIDATED in gcheck (ASYNC_GATHER path): worst 0.0,
   NW=1/4, K row_l + V col_l. size=bpt=16, lds_ptr=lds_base+i*nw*bpw (per-warp), reverse map lbo->(g_row,
   g_col)->topk. This is the pipeline's gather primitive. NEXT = step 2 (2-buffer + prefetch).
2. **2-buffer + prefetch-t+1** (naive pipeline, no overlap scheduling): gather t+1 during compute t.
   Validate + measure (expect modest gain; the compiler may not overlap — like the old hk_fwd_swp).
3. **Early-gather (evacuate V, gather t+2 into freed buf) + S_prev carry**: the gluon structure.
4. **Explicit sched_group_barrier** for softmax∥QK overlap (the actual win + codegen stabilization).
5. Subtile PV if VGPR>256/spills.
Benchmark each vs gluon 3.03 ms (idle GPU, HIP_VISIBLE_DEVICES=2).

## Honest ceiling
gluon is ALREADY pipelined (3.03 ms). The 684-cyc K-feed is BW-bound (project memory). HK's edge is finer
instruction-level scheduling control; realistic target = match-to-modestly-beat gluon, not a multiple.
