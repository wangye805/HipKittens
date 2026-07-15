# DSA-MLA occ-2 fwd — `hk_s2_occ2_pro4b` (stag2ep + prologue/epilogue overlap, SPILL-FREE)

MI355X / gfx950. `pro4b` = `stag2ep` (682/849/921 TF) + two setup/epilogue overlap tweaks, kept
**0-spill** so it's a clean baseline to resume from. Marginal but consistent (~0.15% full-grid) win.

## Changes vs stag2ep (both pure reorders — no algorithm change)
- **S2** — issue the tile-0/1 KV gather *before* `zero(acc)`/init, so the cold-gather HBM latency has
  the 128-VGPR `zero(acc)` + vec init to hide behind (previously init hid only the shorter topk DMA).
- **E1b** — issue the per-head `sink` HBM load *just before* the epilogue skew-balance `s_barrier`, so
  the barrier wait hides the sink load's HBM latency. **Spill-free** (no cross-loop liveness).
  - NOTE: a more aggressive variant (`pro4`, not committed) hoists `sink` to kernel top → −0.38%
    full-grid but **+3 VGPR spill**. We chose the spill-free E1b (−0.15%, VGPR 256 / 0 spill / occ-2)
    as the clean baseline. To chase the extra 0.2%, move the sink load to kernel top and accept the spill.

## Verify / build
```
CFB="-DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -O3 -I$HK/include \
     -DHIP_ENABLE_WARP_SYNC_BUILTINS -mllvm -amdgpu-mfma-vgpr-form -DHK_PERMLANE_REDUCE"
hipcc $CFB -DDC=64 -DNTILES=36 hk_s2_occ2_pro4b_test.cpp  -o t && ./t 0 1 5     # correctness (topk1152)
hipcc $CFB -DDC=64 -DNTILES=36 -DNCTA=4096 hk_s2_occ2_pro4b_bench.cpp -o b && ./b 2400
```
Correct: nt=8/16/36/64/72 × seeds × sink/no-sink × invalid-topk all PASS (O worst ~1e-3, LSE < 1e-3).
Resource: VGPR 256, AGPR 0, **0 spill**, occupancy 2.

## Perf (same-session interleaved, MI355X dev2, FLOP=4·S·H·topk·D)
| | full-grid (NCTA=4096) |
|---|---:|
| stag2ep | ~1.4600 ms |
| **pro4b** | **~1.4579 ms (−0.15%)** |
| pro4 (spill, not committed) | ~1.4546 ms (−0.38%) |

Steady cyc/tile ≈ 3,628 (unchanged vs stag2ep 3,648 — S2/E1b don't touch the loop).

## Why the win is small (context for whoever resumes)
Per-CTA decomposition (traces): our steady loop is ~10% faster/tile than flydsl, but our setup is
~4.4k heavier and epilogue ~2k heavier. S2/E1b attack those — **but at the full grid the occ-2 sibling
CTA already hides most prologue/epilogue latency**, so the recoverable throughput is ~0.1–0.4%, near the
noise floor. This is a confirmed dead-end for meaningful gains.

Skipped (documented, not worth it): **S1** (zero-acc elimination — needs +VGPR/peel we don't have) and
**S3** (register-topk prime — sibling-hidden at scale).

## The real remaining lever
The steady read pole: **3,628 → ~3,139 floor** (CU-shared LDS crossbar, V-read-bound). Needs
**read reduction, not setup tuning** — e.g. transpose-free row-major V (V-read 122→240 B/cyc since
K=V), which costs LDS capacity, not VGPR. That's where the next real work is.

## Also in this branch
`noxpose/` — investigation showing the epilogue transpose is a free relabel and operand-swap is a
dead-end (see noxpose/README.md).
