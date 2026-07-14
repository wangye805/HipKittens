# DSA-MLA occ-2 forward — best kernel: `hk_s2_occ2_stag2ab`

MI355X / gfx950. Bench = full-kernel wall/NTILES @ 2400 MHz, NCTA=128, NTILES=72, TILE_K=32, DC=64.
occ-2 = 8 warps / CTA = 2 waves / SIMD, BLOCK_H=128 (128 query heads/CTA, 16/warp).

## Result

| kernel | bench cyc/tile | steady tile (ATT) | vs Leon |
|---|---:|---:|---|
| **`hk_s2_occ2_stag2ab` (best)** | **4797** | **3564** | −10% bench / −33% steady |
| `hk_s2_occ2_tk32fm` (lockstep + fixed-max, reference) | 5610 (DC64) / 5410 (DC128) | 4232 (DC128) | ~parity |
| Leon PR3833 (ref) | 5336 | 5396 | — |

Also note: this is occ-2 with **128 heads/CTA** vs Leon's 64, so at equal bench cyc/tile it clears **~2× the heads per SIMD** → ~2× Leon per head.

Correctness: **84/84 PASS** (5 tile counts × 8 seeds × 2 sink modes + 4 invalid-topk configs), O worst ~1e-3, LSE < 1e-3, bf16-faithful CPU ref.

## Build / run (inside the gfx950 docker)

```
HK=/path/to/HipKittens
CFB="-DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -O3 -I$HK/include \
     -DHIP_ENABLE_WARP_SYNC_BUILTINS -mllvm -amdgpu-mfma-vgpr-form -DHK_PERMLANE_REDUCE"
# correctness
hipcc $CFB -DDC=64 -DNTILES=72 hk_s2_occ2_stag2ab_test.cpp  -o t && ./t 0 1 5
# bench
hipcc $CFB -DDC=64 -DNTILES=72 -DNCTA=128 hk_s2_occ2_stag2ab_bench.cpp -o b && ./b 2400
```
Resource: VGPR 256, AGPR 0, ScratchSize 0 → occ-2 (2 waves/SIMD).

## The 4 stacked levers (baseline 6485 → 4797, −26%)

1. **fixed-max softmax** (flydsl fast-path): bound = tile-0 colmax set once; `p=exp2(s−bound)`, `alpha≡1`,
   so the per-tile full-D `mul_col(acc)` rescale is *never emitted*. Exact by softmax shift-invariance,
   LSE uses `m_i=bound`. −13.5%.
2. **double-buffer K/V staging @ DC=64**: prefetch next chunk's `ds_read` under the current mma → hides the
   `ds_read→mma` bubble. DC=64 is the knee (DC=128+DBUF spills; DC=32 doubles loop VALU).
3. **staggered ping-pong**: 2 warp groups (WG0=heads0-63, WG1=heads64-127), WG1 skewed one cluster via a
   prologue `s_barrier`. Puts one group's softmax-VALU under the other's PV-MFMA. Only pays *after* DBUF
   clears the intra-wave bubbles (complementary: DBUF hides intra-wave, stagger hides cross-phase).
4. **asymmetric barrier** (the big one): WG0 stops ONLY at the before-PV barrier, WG1 ONLY at the after-PV
   barrier; they pair by arrival-count (WG0-Bc(j) ↔ WG1-Bd(j-1)), primed/closed by the prologue-skew +
   epilogue barriers. → **1 barrier/tile per warp** (vs 2) while keeping the half-tile stagger. 4-ring gives
   the gather-visibility slack. −6.5%.

## Steady-state decomposition (`stag2ab`, tile ≈ 3564)

```
VALU 990(27%)  MFMA 846(23%)  BAR 658(18%)  LDS 552(15%)  WAIT 168(4%)
```
Gather WAIT is fully hidden (570→168). Remaining floor is the *work*: softmax VALU + matrix + crossbar.

## Dead-ends (measured negative — don't re-try)

- Stagger **without** double-buffer: slower (arbiter already hides intra-wave bubbles).
- `DC=128 + DBUF`: spills (256 VGPR + 128 scratch) → 8748.
- `DC=32`: no spill but doubles loop/index VALU (36%) → slower.
- `stagsp` ({gather+QK} | {softmax+PV}): softmax-VALU clashes with gather-VALU → slower.
- **split-softmax** (move tail into C2 to balance C1/C2): +5% slower.
- **set-priority** (`s_setprio(1)` for lagger): neutral.
  - Insight: the slot idle asymmetry (leader 67% busy vs lagger 87%) is *benign* — during the leader's
    hard barrier-wait the lagger runs on the same SIMD, so total work/tile is unchanged. Scheduling levers
    are exhausted; further gains need work-reduction (softmax VALU, or transpose-free row-major V).

`hk_s2_occ2_tk32fm` is committed alongside as the clean lockstep+fixed-max reference (no stagger machinery).
