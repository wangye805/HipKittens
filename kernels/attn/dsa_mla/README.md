# DSA-V4 sparse-MLA forward (occ-2, gfx950 / MI350–MI355)

HipKittens forward kernel for DeepSeek DSA-V4 sparse multi-latent attention: dense `K == V`
(`kv_lora_rank` D = 512), per-token top-k gathered rows, base-2 online softmax with a per-head
attention sink, and LSE output. Occupancy-2 (2 waves/SIMD), staggered ping-pong.

## Files
- `kernel.cpp` — the kernel (`dsa_mla_fwd`) + `dispatch` + `PYBIND11_MODULE(tk_kernel)`. Batched:
  one CTA per query token (`grid = Qg.batch()`).
- `sub_gather.cuh`, `sub_topk.cuh` — async KV-gather / top-k DMA submodules.
- `test_batched.cpp` — standalone C++ multi-token correctness vs a bf16-faithful CPU reference.
- `test_python.py` — torch harness (timing + robustness vs a torch reference).

## Usage
```
make                 # build the tk_kernel pybind module (TOPK=1152 DC=64 by default)
python test_python.py
make test_cpp        # C++ correctness only (no python/torch needed)
make TOPK=2048       # rebuild for a different top-k (NTILES = TOPK/32)
```

## Tensor layout (all 4-D, bf16 unless noted)
| arg | shape | notes |
|---|---|---|
| Q   | `[S, 8, 16, D]` | per-token query; head = warp(0..7)*16 + lane-group(0..15) |
| KV  | `[1, 1, T_KV, D]` | dense KV cache (K==V), shared across tokens |
| topk| `[S, 1, NTILES, 32]` int32 | per-token gathered row indices (-1 = masked) |
| sink| `[1, 1, 8, 16]` f32 | per-head sink, shared across tokens |
| O   | `[S, 8, 16, D]` | output |
| LSE | `[S, 1, 8, 16]` f32 | log-sum-exp |

`dispatch(Q, KV, topk, sink, O, LSE)`.

## Measured (MI355X, dev2, batched, python harness)
| topk | median | TFLOPS |
|---|---|---|
| 1152 | 1.484 ms | 833 |
Correct vs torch reference (O within-tol 99.9%, cos 0.9996, LSE max_abs ~2e-3). FLOP = 4·S·H·topk·D.
(A few % under the internal bench because this batched kernel writes S *distinct* outputs — real
per-token O-store traffic — whereas the dev bench reused one output buffer.)

## Config
`TILE_K=32`, `D=512`, `NW=8` (128 heads/token), `DC=64`, occ-2. `TOPK` (→ `NTILES=TOPK/32`),
`DC` (64 best; 128 single-buffer ties, 256 spills), `HAS_SINK` are Makefile knobs.

## Notes
This is the spill-free variant (`pro4b` lineage): stag2ep + prologue KV-gather/init overlap (S2)
+ pre-barrier sink load (E1b). VGPR 256 / AGPR 0 / 0 spill / occ-2. The higher-perf `pro4`
variant (kernel-top sink hoist, −0.2% faster, +3 VGPR spill) lives on the dev branch
`dsa-mla-fwd-occ2` if you want it.
