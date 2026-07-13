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

## Remaining for full integration (into hk_s2_occ1_stream QK)
1. **art store of col_l fp32 `s`**: the art store to shared requires row-major + `st_16x16_swizzled_s` + b64 (2-reg) split — the same relayout as the P-roundtrip. In the real kernel `s` is *not* stored — it feeds mask/softmax. Need: either the softmax ops accept `art s`, or convert `art s → rt s` after the mma (cheap, both VGPR).
2. **K as art** (VGPR ranges) + **double-buffer** in art (kb0/kb1 as art with distinct VGPR ranges), streamed via `load<0,N>` per chunk.
3. **Q as art in AGPR**, resident (q_arr[NCH] → art with AGPR ranges), loaded once.
4. Thread `mma_ABt<N,M,K>` split indices through the D-chunk loop.
5. Manual register map: acc VGPR (128), K-dbuf VGPR, Q AGPR (64), s VGPR — non-overlapping ranges + clobber.

Expected: QK peak VGPR 296→232, AGPR-overflow ferries gone. Branch `dsa-mla-fwd-art-agpr`.
