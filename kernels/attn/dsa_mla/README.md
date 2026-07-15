# DSA-V4 sparse-MLA forward (occ-2, gfx950 / MI350–MI355)

HipKittens forward kernel for DeepSeek DSA-V4 sparse multi-latent attention: dense `K == V`
(`kv_lora_rank` D = 512), per-token top-k gathered rows, base-2 online softmax with a per-head
attention sink, and LSE output. Occupancy-2 (2 waves/SIMD), staggered ping-pong, double-buffered
K/V feed. One CTA per query token (`grid = Qg.batch()`).

## Files
- `kernel.cpp` — the kernel (`dsa_mla_fwd`) + `dispatch` + `PYBIND11_MODULE(tk_kernel)`.
- `sub_gather.cuh`, `sub_topk.cuh` — async KV-gather / top-k DMA submodules.
- `test_python.py` — torch harness: timing (TFLOPS) + correctness vs a torch reference.
- `test_batched.cpp` — standalone C++ multi-token correctness vs a bf16-faithful CPU reference.

## Reproduce

Build the pybind module and run the python harness (timing + correctness):
```
make                          # TOPK=1152 DC=64 HAS_SINK=1 by default
python test_python.py         # S=4096 topk=1152 sink=1
python test_python.py 4096 2048 1     # args: S  TOPK  HAS_SINK
make TOPK=2048 && python test_python.py 4096 2048 1
```

C++ correctness only (no python/torch), multi-token vs CPU reference:
```
make test_cpp                 # builds test_batched, runs S=8
# or manually, matching the perf flags:
hipcc test_batched.cpp -DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -O3 -w \
  -I$THUNDERKITTENS_ROOT/include -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
  -mllvm -amdgpu-mfma-vgpr-form -DHK_PERMLANE_REDUCE -DDC=64 -DNTILES=36 -o test_batched
./test_batched  <S>  <seed>  <n_invalid_topk>      # e.g. ./test_batched 8 0 40
```

`NTILES = TOPK / 32`. Build flags that matter: `-mllvm -amdgpu-mfma-vgpr-form` (matched-VGPR MFMA
form) and `-DHK_PERMLANE_REDUCE` (permlane softmax reduce) — both required for the measured perf.

## Benchmarks (MI355X gfx950, dev 2, batched, python harness)
FLOP = `4·S·H·topk·D` (QK+PV, apples-to-apples with flydsl/Primus-Turbo). S=4096, H=128, D=512, sink on, LSE in loop.

| topk | median | TFLOPS |
|---:|---:|---:|
| 512  | 0.842 ms | 653 |
| 1152 | 1.486 ms | 833 |
| 2048 | 2.407 ms | 914 |

Correctness vs torch reference: O within-tol ≈ 99.9%, cos ≈ 0.9996, LSE max_abs ≈ 2e-3.
(These batched numbers run a few % under the internal dev bench (682/848/922) because this kernel
writes S *distinct* outputs — real per-token O-store traffic — plus python launch overhead; the dev
bench reused a single output buffer.)

Resource (NTILES=36, DC=64): **VGPRs 256, AGPRs 0, occupancy 2, 1 VGPR spill** (perf-neutral — the
1-reg spill is the cost of per-token batched indexing; the non-batched dev kernel is 0-spill).

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

## Config knobs (Makefile / -D)
`TOPK` (→ `NTILES = TOPK/32`), `DC` (64 best; 128 single-buffer ties, 256 spills), `HAS_SINK`,
`KV_ROWS` (dense KV rows, default 4096). Fixed: `TILE_K=32`, `D=512`, `NW=8` (128 heads/token), occ-2.

## Notes
Spill-free-lineage variant (`pro4b`): stag2ep (fixed-max + double-buffer + stagger + asymmetric
barrier + prologue/epilogue overlap). The steady loop is read-pole-bound (~3.6k cyc/tile vs a ~3.1k
LDS-crossbar floor). The higher-perf `pro4` variant (kernel-top sink hoist, ≈0.2% faster, +3 VGPR
spill) and the full version history live on the dev branch `dsa-mla-fwd-occ2`.
