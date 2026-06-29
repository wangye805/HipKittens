# DSA V4 sparse-MLA forward — HipKittens (gfx950 / CDNA4)

HipKittens port of the DeepSeek V4 **sparse multi-head latent attention (MLA)** forward with attention
sink. Mirrors the proven gluon kernel's structure (`sparse_mla_fwd_v4`, backend="gluon") using HK's
register-tile primitives, with the goal of hand-scheduling the cross-tile software pipeline that gluon's
compiler refuses to emit.

## What this is
Per query token, attention is computed over a **top-k selected** set of KV positions (sparse + SWA).
Config: `BLOCK_H=64` heads/program = `NW=4` warps × `QB=16` heads (M-split), `TILE_K=32` keys/tile,
`D_V=512` (lora) + `D_ROPE=64` (rope) = `D_QK=576`.

## Design (gqa keys-as-rows DISCIPLINE + gluon STRUCTURE)
- **keys-as-rows**: `S = [TILE_K, QB]` col_l. `col_max`/`sub_col`/`col_sum`/`div_col`/`mul_col` all consume
  one `row_vec` family (per head) → no vec-layout conversions.
- **QK = two `mma_ABt(S, K, Q)`** with *natural* K`[TILE_K,D_V]` + Q`[QB,D_V]` (both row_l). No transpose,
  no transposed gather, no Q-transpose. rope adds a second `mma_ABt`.
- **V = the same gathered K shared tile re-loaded as col_l** (the gluon "V = K.permute" win: one gather,
  two register loads). PV = `mma_AtB(acc, V, P)` → `acc=[D_V, QB]`.
- **P operand** rt_16x16 → rt_32x16 via a per-warp LDS roundtrip (`st_32x16`, `__s_barrier` between
  store and load).
- **gather**: topk-indirected, forward-swizzle (iterate logical `(k,d)`, write `subtile_offset +
  st.swizzle(...)`) so the layout matches what `load(rt, st)` consumes. `-1` topk safe-clamped to row 0.
- **online softmax** with running `m_i/l_i`, `acc` rescale by `alpha`; **sink fold** (V4) in the epilogue.
- **M-split**: each warp owns `QB=16` heads; per-warp Q load + O store use the **depth** coord
  `{0,warpid,0,0}` (the col-dim coord silently fails for col-tile≥2 on HK).
- **`-1` invalid-key mask** (`#ifdef ENABLE_MASK`): in-kernel broadcast mask tile, add `-1e30` to invalid
  keys' scores pre-softmax. Correct at NW=1; NW=4 needs register-pressure relief (revisit with pipeline).

### The key correctness lesson (NW>1)
The months-long NW>1 fragility was **register-lifetime corruption at the VGPR=256 cap**, NOT a sync race or
compiler bug. Holding `k_l[128 VGPR]` and `v_l[128 VGPR]` simultaneously blew the knife-edge →
non-deterministic NaN. **Loading V late** (after QK, when `k_l` is dead) relieves the pressure and makes it
deterministic. Barriers/sched_barriers do not fix it; pressure relief does.

## Files
- `hk_fwd3.cpp` — the kernel (single-buffer; double-buffer pipeline + hand-scheduled SWP are the next step).
- `hk_fwd3_test.cpp` — self-contained HIP harness + bf16-faithful CPU sparse-MLA+sink reference.
- `mini3.cpp` / `mini3b.cpp` / `mini3_loop.cpp` / `gcheck.cpp` (+ `_test`) — single-concern validators
  (compute/layout, natural-operand QK, online softmax, gather isolation).

## Build & test (gfx950)
```
hipcc hk_fwd3_test.cpp -DNW=4 -DNTILES=4 -DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -O3 \
  -I<HipKittens>/include -DHIP_ENABLE_WARP_SYNC_BUILTINS -o /tmp/h3
HIP_VISIBLE_DEVICES=<idle gpu> /tmp/h3 <seed> <has_sink> <n_invalid>
```
Validated: NW∈{1,4} × nt∈{4,36} × sink∈{0,1} × multiple seeds all PASS (mean ~1e-4, 0 nan); `-DENABLE_MASK`
adds `-1` masking (NW=1). Do **not** use `-ffast-math` in correctness tests (it breaks NaN detection).

## Status / next
Correct single-buffer kernel. TODO: full-grid launch + benchmark vs gluon (3.1 ms MI350) → double-buffer
async pipeline → hand-scheduled cross-tile SWP (overlap prev-tile softmax/rescale under current-tile MFMA).
