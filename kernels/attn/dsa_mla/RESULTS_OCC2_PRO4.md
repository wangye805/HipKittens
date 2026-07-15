# DSA-MLA occ-2 fwd — `hk_s2_occ2_pro4` (BEST PERF, has 3-VGPR spill) — reference only

Same as `pro4b` (S2 + sink hoist) but the sink load is hoisted to **kernel top** (flydsl `hoist_sink`)
so its HBM latency flies during the whole loop. Faster than the spill-free `pro4b`, but keeps `sink_h`
live across the loop → **3-VGPR spill** (VGPR 256, occ-2 preserved).

Same-session interleaved (MI355X dev2, full grid NCTA=4096, FLOP=4·S·H·topk·D):
- pro4  (this, spill):     ~1.4546 ms  (**−0.38%** vs stag2ep)
- pro4b (spill-free):      ~1.4579 ms  (−0.15%)
- stag2ep (baseline):      ~1.4600 ms

Correct: nt=8/16/36/64/72 × seeds × sink × invalid-topk PASS. Steady cyc/tile ~3628 (loop unchanged).

**The shipped/clean kernel is `pro4b` (0-spill).** This `pro4` is kept for reference — pick it if the
extra ~0.2% is worth a 3-VGPR spill. Only diff vs pro4b: sink load position (kernel-top vs pre-barrier).
