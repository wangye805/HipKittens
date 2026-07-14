# DSA-MLA occ-2 fwd — best kernel: `hk_s2_occ2_stag2ep` (prologue + epilogue optimized)

MI355X / gfx950. **Apples-to-apples with flydsl/Primus-Turbo**: FLOP = `4·S·H·topk·D_V`
(2 GEMMs QK+PV × 2 FLOP/MAC), S=4096, H=128, D_V=512, sink ON, LSE in the timed loop.
TF = FLOP / latency. Config swept: topk ∈ {512, 1152 (pro-cr4), 2048}, NCTA=S=4096.

## Result (same node/session; clock-limited, so absolute TF only compares within a session)

| topk | latency | **stag2ep TF** | flydsl same-node | our gluon (pr3833) |
|---|---|---:|---:|---:|
| 512  | 0.806 ms | **682** | – | 528 |
| 1152 | 1.458 ms | **849** | 792 → **+7%** | – |
| 2048 | 2.388 ms | **921** | 870 → **+6%** | 707 |

Correctness: all configs PASS (nt=8/16/36/64/72 × 8 seeds × sink/no-sink × invalid-topk),
O worst ~1e-3, LSE < 1e-3. VGPR 256, 0 spill, occ-2.

## Build / run

```
CFB="-DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -O3 -I$HK/include \
     -DHIP_ENABLE_WARP_SYNC_BUILTINS -mllvm -amdgpu-mfma-vgpr-form -DHK_PERMLANE_REDUCE"
hipcc $CFB -DDC=64 -DNTILES=36 hk_s2_occ2_stag2ep_test.cpp  -o t && ./t 0 1 5           # correctness (topk=1152)
hipcc $CFB -DDC=64 -DNTILES=36 -DNCTA=4096 hk_s2_occ2_stag2ep_bench.cpp -o b && ./b     # bench
```

## Optimization chain (all same-node, topk=1152)

| kernel | TF | change |
|---|---:|---|
| hk_s2_occ2_stag2ab | 802 | committed baseline (fixed-max + double-buffer + stagger + asymmetric-barrier) |
| stag2q  | 806 | drop premature post-Q-load `s_waitcnt(0)` (compiler drains Q at QK) |
| stag2tk | 808 | per-tile topk **prefetch** (deep pow2 ring TK_RING=8) via `raw_buffer_load_lds` DMA (no reg-hop, no forced `vmcnt(0)`); topk needn't be fully loaded upfront |
| stag2pro | 817 | **prologue**: topk-DMA-first, Q-load-last, `__syncthreads`→bare `s_barrier` + targeted `s_waitcnt vmcnt` (30k→21k prologue) |
| **stag2ep** | **849** | **epilogue fold**: replace per-element `div_col(acc,l_tot)` (~128 slow divides/lane) with `div(scal,afix,l_tot)` on the 16-elem row_vec + one `mul_col(acc,scal)`. Removed the big epilogue VALU block. |

Also present: `hk_s2_occ2_stag2pro` (the prologue-optimized milestone, before the epilogue fold).

## Key mechanism notes (gfx950)

- `__syncthreads()` = `s_barrier` + `s_waitcnt(0)` on **both** vmcnt & lgkmcnt (full flush). Use bare
  `__builtin_amdgcn_s_barrier()` + a **targeted** `s_waitcnt vmcnt(N)` where only one counter must drain.
- `raw_buffer_load_lds` (topk/KV gather DMA) is **vmcnt**-tracked (not lgkmcnt) — the LDS write is part of
  the VMEM op, so vmcnt dropping means data landed in LDS.
- HK global `load()` = `buffer_load` + bf16 convert; the convert forces a `vmcnt` drain (Q not purely async).
- Staggering (asymmetric Bc/Bd) is doing real work — it hides gather-DMA latency (WG0 on gather while WG1
  does PV) — the exact bottleneck flydsl pays as exposed `vmcnt` (41% of its stall). Keep it.

## Dead-ends (measured, don't re-try)

- `stag2pro2`: issue Q before the KV-prime drain to overlap them → −1..3% WORSE. HK `load()`'s bundled bf16
  convert forces `vmcnt(0)`, which drains the KV prime too (over-drain). Real overlap needs a **split Q load**
  (issue raw buffer_loads, defer the convert) — not done.

## Remaining lever (next: separate sub-workspace)

The epilogue still has `transpose(acc_t, acc)` [512,16]→[16,512], which lowers to **128 `v_readfirstlane_b32`
+ a 32-iter exec-mask loop** (cross-lane shuffle — slow). flydsl avoids it by matching the PV-mma D-layout to
`g.Og` (swap the mma operands so `acc` comes out `[heads,D]` = store-ready) + `cvt_pk_bf16`, no transpose.
Moderate change (PV mma operand swap → acc layout flip → `mul_col`→`mul_row` + subtile axis + drop transpose).
Being explored in a separate sub-workspace.
