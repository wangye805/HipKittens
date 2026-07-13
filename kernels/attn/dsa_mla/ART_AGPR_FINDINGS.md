# occ-1 assembly-mode: park Q in AGPR via HK `art` — mechanism VALIDATED

Goal: relieve the QK-peak VGPR (acc128 + Q64 + K64 + ... ≈ 296 → 256 + AGPR overflow) by putting
the Q mma-operand in AGPR (Q is resident, loaded once → cheap). Uses HK's `art` (assembly register
tile) type — see `reference_hk_art_agpr` and `analysis/attn/bkwd` (which parks Q/K/V in AGPR).

## Validated by `probe_art.cpp` (ISA)

s[64,16] = K[64,32]·Q[16,32]ᵀ with `Q_ranges=range<256,259>` (AGPR a[0:3]), K/s in VGPR:
```
ds_read_b128 a[0:3], v0                                    <- Q loaded STRAIGHT to AGPR (no ferry)
v_mfma_f32_16x16x32_bf16 v[64:67], v[80:83], a[0:3], v[64:67]
                          s(VGPR)   K(VGPR)   Q(AGPR!)  acc(VGPR)
```
Confirms: Q in AGPR, loaded directly (no VGPR→AGPR ferry), mma reads it natively, no mma→accvgpr
hazard (mma reads its own operand). acc/K stay VGPR.

## API notes (learned)
- `art<T,rows,cols,layout,shape,RANGES>`, `RANGES = ducks::art::split_many_t<type_list<range<lo,hi>>, regs_per_base_tile>`, `lo>=256`→AGPR. #sub-ranges must == height*width.
- `ducks::art::clobber<RANGES>()` to reserve.
- Load from **shared** (no global→art simple load): `addr=get_address(art, st_subtile); load<0,N>(art, st_subtile, addr)` for N=0..#sub-ranges-1. Loads a bf16 row-layout tile straight into its pinned regs (AGPR if ranged there).
- mma: `mma_ABt(D,A,B,C)` (looping) or `mma_ABt<N,M,K>(...)` — ALL operands must be `art`; D col_l, A/B row_l. Normal `rt` `mma_ABt` rejects art.

## The handoff blocker (definitive)
`art` is a **phantom/assembly-only type — NO `.data`/`.tiles` members**; data lives only in the pinned
physical registers, touched solely by assembly-mode ops (`load<N,M>`, `mma_ABt<N,M,K>`, `store<N,M>`,
`maps.cuh` element/reduce ops). There is **no cheap `art→rt` conversion** (the `copy` in
`assembly/conversions.cuh` is art→art fp32→bf16, not art→rt). So the QK mma output `s` (col_l fp32,
art) cannot be handed to the existing **rt-based** mask/softmax/reshuffle path by a field copy.

Two ways to actually integrate, both costly:
- **(a) Full-art fwd**: rewrite mask/softmax/reshuffle/PV in `art` ops (like `analysis/attn/bkwd`, ~300
  lines assembly-mode). Cleanest, but a big rewrite.
- **(b) Round-trip `s` art→shared→rt**: adds a per-tile LDS round-trip for `s` (and the col_l art store
  needs row-major + `st_16x16_swizzled_s` + b64 split — a relayout, like the P-roundtrip). This new cost
  likely negates the Q-in-AGPR VGPR benefit.

## Cost/benefit
Benefit = QK-peak VGPR 296→232 + AGPR-overflow ferries gone (~100-200 cyc/tile) + unblocks gather-interleave.
It does **NOT** reach occ-2 (that's LDS-bound, orthogonal). So it's a **modest occ-1 gain requiring a large
full-art rewrite** (option a). Recommend weighing against occ-2 (higher-value structural lever) before
committing to the full-art fwd. Mechanism is proven (probe_art: Q in AGPR, mma reads it natively, no ferry).

Remaining pieces if pursued (option a): K/V/Q as art (Q→AGPR ranges, K/V→VGPR, double-buffered),
mask/softmax/reshuffle in `maps.cuh` art ops, manual non-overlapping register map + `clobber`, thread
`mma_ABt<N,M,K>` split indices through the D-chunk loop. Branch `dsa-mla-fwd-art-agpr`.
