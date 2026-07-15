# hk_fwd_noxpose — investigation: can we eliminate the epilogue transpose?

**Verdict: NO WIN. The epilogue `transpose` is already a near-free register relabel; the
operand-swap that would "naturally" produce `[heads,D]` yields an *unstorable* accumulator.**
Base kernel = `hk_s2_occ2_stag2ep` (682/849/921 TF). MI355X / gfx950 / occ-2.

## The idea (from a sibling session)
stag2ep's epilogue does `mul_col(acc[D,heads]) ; transpose(acc_t[heads,D]) ; store`. The hint was:
swap the PV-mma operands so `acc` comes out `[heads,D]` store-ready → drop the transpose.

## What the PV mma does
`mma_AtB(D,A,B,C)` computes `D = Aᵀ·B` and **always** emits `D` in **col_l** (all HK mma variants do).
- stag2ep:  `mma_AtB(acc, V, P)` → `acc = Vᵀ·P = [D, heads]` col_l.
- swapped:  `mma_AtB(acc, P, V)` → `acc = Pᵀ·V = [heads, D]` col_l.  ← values verified correct (LSE + O math).

## Why the swap can't be stored (the core finding)
HK's `transpose(dst,src)` is **not** a cross-lane shuffle — it's a subtile *relabel*:
`dst.tiles[j][i].data[k] = src.tiles[i][j].data[k]` (swap grid indices, flip col_l↔row_l type, **no data
movement**). It works because a 16×16 MFMA **accumulator (col_l)** and an **operand (row_l)** are exact
lane-level transposes, so reinterpreting is a valid zero-cost transpose.

Consequence: an mma accumulator's `data[]` is in **row_l-compatible** arrangement. The `store` overloads
expect either a proper row_l tile (via the relabel) or a *memory-layout* col_l tile — **not** a raw
col_l accumulator. Storing the swapped `[heads,D]` col_l accumulator directly (col-store) writes garbage
(verified: O stays at the −999 sentinel, 86% unwritten/mis-addressed).

The relabel always **swaps dims**, so:
- `[D,heads]col_l --relabel--> [heads,D]row_l`  → storable as O[heads,D].  ✓  (this is stag2ep)
- `[heads,D]col_l --relabel--> [D,heads]row_l`  → storable only as Oᵀ.    ✗  (the swap)

Getting `[heads,D]col_l → [heads,D]row_l` (same dims, flip layout) needs a **real** cross-lane transpose,
which would *cost more* than the relabel we were trying to remove. So the swap is a strict dead-end.

## Measurements (MI355X, FLOP=4·S·H·topk·D, ms/launch)
| variant | NT=16 | NT=36 | NT=64 | notes |
|---|---|---|---|---|
| stag2ep (V,P mma; full transpose+store) | 0.805 | 1.460 | 2.392 | committed best |
| noxpose (V,P mma; **per-channel** transpose+store) | 0.805 | 1.461 | 2.386 | **this dir** — correct, ==stag2ep |
| dbg1 (P,V swap; broken store) | — | 1.383 | 2.262 | WRONG output; store does less work |
| dbg2 (V,P mma; broken store) | — | 1.395 | 2.322 | WRONG output; isolates the artifact |

The apparent "5% no-transpose win" (dbg1 vs stag2ep) is an **artifact**: the broken stores skip real
memory work / get DCE'd. Two *correct* epilogues (full vs per-channel) are identical → the transpose is
free and there is nothing to recover. ISA confirms: **0 `v_readfirstlane`/`ds_bpermute` in the epilogue**
(the 224 `v_readfirstlane` are gather-address scalarization in the loop).

## What `hk_s2_occ2_noxpose.cpp` actually is
stag2ep with a **per-channel** transpose+store epilogue (small 16-VGPR temp per D-channel instead of a
full `acc_t[16,512]`). Correct (nt=8/16/36/64/72 × seeds × sink × invalid-topk all PASS, VGPR256/occ2/0
spill) but **not faster**. Kept only as a documented reference.

### Gotcha discovered here (reusable)
`store(g.Og, tile, coord<>{...,c})` with a **default** `coord<>` does **NOT** scale the channel index `c`
by the tile width (the `unit_coord` else-branch passes `c` through unscaled) → mis-addresses for c≠0.
Use a **tile-typed** coord: `store(g.Og, tt, coord<OcT>{0,warpid,0,c})` so `c` is scaled by `OcT::cols`.
(Full-width stores at c=0 hide this; only partial-width per-channel stores expose it.)

## Bottom line
Keep `hk_s2_occ2_stag2ep` as the best fwd kernel. The transpose is not a lever. Real remaining levers are
in the loop (gather traffic / mma feeding), not the epilogue.
