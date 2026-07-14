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

## UPDATE — Piece 1 (multi-chunk QK in art) VALIDATED (probe_art_qk.cpp)

Full-contraction QK in art (2 D-chunks; scales to 8): s[64,16] = Σ_c K_c[64,64]·Q_c[16,64]ᵀ, with
**Q resident in AGPR** (a[0:15]), K double-buffered in VGPR, s(acc) in VGPR. ISA confirms:
- every Q chunk read from AGPR (`v_mfma v[s], v[K], a[Q], v[s]`), Q loaded straight to AGPR (`ds_read a[…]`),
- **only 2 `accvgpr` in the whole kernel** (vs ~64 in the rt version) — the QK-feed is FERRY-FREE.

So Q→AGPR eliminates the AGPR-overflow ferries that caused the +48-cyc mma bubbles at occ-1. Register
map used: s=v[0:15], K0/K1=v[16:79], Q0..=a[0:15] (clobbered). Scaling to 8 chunks = 8 Q art tiles
a[0:63] + K dbuf, mechanical.

Reductions finding: the bwd works in art because it RECOMPUTES P=exp2(S−LSE) from the saved LSE →
needs no reductions (just exp2+sub_row, which art has). The FORWARD needs col_max/col_sum, which are
NOT in the art op set (bwd never needed them) — so the full-art fwd requires implementing art
col_max/col_sum (permlane-butterfly, art-op style; same as the rt patch). All other softmax ops exist
for art: exp2, mul, add/sub, add_row/sub_row, sub_col/mul_col, zero, copy.

Remaining build order: (2) art col_max/col_sum, (3) softmax in art, (4) PV in art + full register map.

## UPDATE — Piece 2 WIP (probe_art_reduce.cpp): test-first caught 2 real bugs

Implemented art_col_max/art_col_sum via `macros::v_mov_b32_p2up<reg>()` (reads a pinned art register
into a compiler value) → local reduce in C++ → permlane32+16 butterfly. Executable test: QK-art →
reduce → store row_vec → compare CPU col_max/col_sum of K·Qᵀ.

Bugs the test caught (both invisible to the earlier ISA-only checks):
1. **Register placement (FIXED):** art tiles at low VGPRs (v[0:79]) corrupt the compiler's low regs
   (v0=threadIdx, arg/address regs) → GPU memory-access fault. Fix: place art tiles HIGH (v[64:143]),
   leave low VGPRs for the compiler — matches the bwd kernel (art at v[40:103], v0-39 free). RULE:
   art register ranges must avoid the low VGPRs the compiler needs for addressing/threadIdx.
2. **Numeric (OPEN):** after the fault fix it runs but col_max/col_sum are wrong (worst ~7 / ~62) —
   bug is in one of {art load addressing (get_address/subtile), the looping mma_ABt, the reduction
   axis/layout}, all previously validated only by ISA. Next: isolate via a known-pattern art-load
   readback (verify get_address/load), then the mma output layout, then the reduce.

Lesson: ISA validation proves *placement*, NOT *correctness* — numeric block tests are mandatory for
assembly-mode art. The test-first setup is working as intended.

## UPDATE — Piece 2 PASS (art col_max/col_sum numerically correct)

probe_art_reduce: QK-art (Q->AGPR) -> art_col_max/col_sum -> compare CPU = worst 0.0000 PASS.
This also numerically validates QK-art (s correct). art_col_max/col_sum work: v_mov_b32_p2up<reg>
reads each S_ranges reg -> local C++ reduce -> permlane32+permlane16 butterfly; lane->query = L%16,
all 16 queries exact.

KEY API BUG found (fixed in both probes): **art `load<N,M>` indexes (height, width), NOT flat.**
K [64,64] rt_16x32 = height 4 x width 2 -> load<0,0>,load<0,1>,load<1,0>,...,load<3,1> (N=0..3,M=0..1),
NOT flat load<0,0..7>. The flat version compiled + looked right in ISA but loaded wrong data (piece 1
probe_art_qk had this bug, invisible to ISA-only). Reinforces: assembly-mode needs numeric tests.

Pieces done: 1 (QK-art Q->AGPR, ferry-free) + 2 (art col_max/col_sum). Next: 3 softmax in art
(exp2/sub_col/mul_col/add_row exist), 4 PV in art + full register map, then integrate into hk_s2_occ1_stream.

## UPDATE — Piece 3 PASS (softmax core in art)

probe_art_softmax: QK-art -> m=art_col_max(s) -> denom = sum exp2(scale*(s-m)) -> vs CPU = worst 0.0000.
Validates scale + sub(m) + exp2 + col_sum fused over the art s regs (per-lane m from col_max; the
sub is a per-lane scalar since a lane holds one query's keys). All done via v_mov_b32_p2up reads +
permlane butterfly. Pieces 1,2,3 all numerically validated. Next: piece 4 PV-in-art (mma_AtB with
P bf16 art from copy(s->bf16), acc art) + full non-overlapping register map -> integrate into kernel.

## Piece 4 design RESOLVED (PV mma_AtB in art)

- art mma_AtB requires ALL operands (D,A,B,C) col_l.
- PV: acc[DC=64,16q] = V[64key,64DC]^T . P[64key,16q]. A=V rt_16x32 col_l, B=P must be rt_32x16
  (the mma is 16x16x32 -> K=32 contraction; P rt_16x16 has K=16, rejected). Same P shape the rt
  kernel uses (pop=rt_32x16).
- art `load<>` CANNOT produce rt_32x16 from shared ("Unsupported shape" shared_to_register.cuh
  108/191). So P32 is NOT loaded -> it is built by the in-register permlane reshuffle from s
  (rt_16x16 bf16) = the REG_RESHUFFLE p16->p32 recipe already in hk_s2_occ1_stream:
    t = permlane32_swap(tile[2i].d[k], tile[2i+1].d[k]); {d[k],d[k+2]} = permlane16_swap(t.x,t.y)
- => piece 4 = (a) art in-register reshuffle s(rt_16x16 bf16) -> P32(rt_32x16), (b) mma_AtB(acc,V,P32)
  over 8 DC-chunks accumulating acc[512,16], (c) online rescale acc *= alpha (mul_col, exists).
  Then full non-overlapping register map + integrate into hk_s2_occ1_stream + verify O/LSE + steady tile.
- probe_art_pv.cpp = design record (isolates the mma shape check; load-blocked by the above).

## Piece 4 IMPLEMENTED — key realization: art only for QK; PV stays normal-rt

The PV mma does NOT need art. Art (assembly-mode) is only needed to park Q in AGPR during QK.
After softmax, BRIDGE art s(fp32 rt_16x16) -> normal rt via p2up element-wise (art_to_rt), then reuse
the EXISTING proven normal-rt PV path unchanged: copy(pb,s_n) -> p16_to_p32(pop,pb) -> mma_AtB(acc,V,pop).
This avoids art mma_AtB entirely (which needs P as rt_32x16, unbuildable via art load).

probe_art_pv2.cpp validates:
- BRIDGE_ONLY build: QK-art (Q->AGPR) + art_to_rt bridge -> runs clean (bridge sound).
- art_to_rt: read art regs via v_mov_b32_p2up, bit-copy (reinterpret_cast<uint32_t*>) into normal
  rt .data[] (element type not float-assignable -> must bit-copy, like p16_to_p32 does).
- V-load caveat: loading V from a STANDALONE st_32x32 [64,64] faults in the probe harness (isolated
  setup artifact); in the kernel V loads from the wide KS [64,512] subtile = proven working code, so
  integration is unaffected. NOMMA+NOLOAD build confirms fault is only that standalone load.

INTEGRATION PLAN (piece 4 -> kernel): in hk_s2_occ1_stream, replace the QK mma with art-QK (Q in AGPR,
K/S art) + art_to_rt(s -> s_n) right after; everything downstream (col_max/softmax/copy/p16_to_p32/PV)
stays as the current normal-rt code. Register win = Q parked in AGPR during QK relieves the ~296 VGPR
peak (the accvgpr-ferry source). Then verify O/LSE end-to-end + measure steady tile vs 6104/Leon 5396.
