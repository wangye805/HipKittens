# HipKittens DSA V4 sparse-MLA forward — port log

**Status (2026-06-27):** ✅ **CORRECT + compiles + register-viable.** Numerics verified (max_abs 2e-4 vs
CPU sparse-MLA+sink reference). All structural + numerical unknowns closed. **Remaining = PERF.**

**Genesis:** the gluon fused kernel hit a *compiler-scheduling* wall — the rescale∥PV interleave +
softmax/feed overlap that gluon's compiler won't emit. HK gives the hand-scheduling control gluon lacks.
**Goal:** the **data-independent cross-tile software-pipelining** win — overlap the serial
softmax/rescale/feed under the matrix pipe (gluon runs them serially: 5,236 cyc/iter vs a ~1,100–1,356
cyc matrix floor) + lazy rescale (data-dependent bonus).
**CAUTION** (unfused NO-WIN, other session): the 684-cyc K-feed is **BW-bound → HK can't fix it**;
realistic ceiling is the softmax/rescale-overlap portion **~15–20%**, not 2–3×.

## Isolated toolchain (do NOT use other sessions' HK clones — they edit them)
- **Upstream** HazyResearch/HipKittens, fresh clone, HEAD `840c9670`:
  - container `/workspace/dsa_triton/sandbox/HipKittens` · remote host `~/workspace/dsa_triton/sandbox/HipKittens`
- Full CDNA4/gfx950 backend (`include/cdna4/`, `include/kittens.cuh`) + attn kernels
  (`kernels/attn/gqa_causal/kernel.cpp` = FA-v3 reference). Other session = `wangye805` fork at
  `/workspace/mi355_flash_attn_v3/HipKittens` — **avoid.**
- Build (compile-only resource probe):
  `hipcc X.cpp -DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -O3 -I<HK>/include -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math -Rpass-analysis=kernel-resource-usage -c -o /tmp/x.o`
- Run a test: same flags, drop `-Rpass…`/`-c`, `-o /tmp/test`; `HIP_VISIBLE_DEVICES=2 /tmp/test`. Node
  cv350-rck-g03-e05-18, container yewang_aiter_main; scp sources to remote `/workspace/_ab/`.

## Target facts (from gluon analysis, profiling/att_fwd_gluon_mi350/RESULTS.md)
- Config: 16 rows/wave, 4 waves, TILE_K=32, D_QK=576 (512 lora + 64 rope), D_V=512.
- `acc[16,512]` fp32 = **128 AGPR/lane → 1 wave/SIMD** (occupancy wall, unescapable; confirmed).
- gluon per-iter 5,236 cyc; matrix work only ~1,356 (QK 600 + PV 852) → ~3.9× serial overhead to overlap.
- HK MFMA combos (cdna4/.../mma.cuh): **16×16 output ← operands base `rt_16x32` via `mma_ABt`** (A·Bᵀ,
  no transpose, `mfma161632` K=32) — this is what we use. (`mma_AtB` 16×16 needs operands `rt_32x16`.)

## Current / Next — PERF block
1. **VGPR re-read DONE (2026-06-27):** correct kernel = **VGPR 256 (maxed), AGPR 196, occ 1, 0 spills**
   — *higher* than the 240 transpose-version (the direct global→reg Q load costs more than it saved). The
   dominant VGPR consumer is the **full `k_ld[KV,DQK]` rt_16x32 = 144 VGPR/lane** resident. ⇒ a
   2-register-buffer cross-tile SWP would need 2× k_ld = 288 VGPR → **spill**.
2. **Cross-tile SWP — must STREAM K in subtiles** (don't hold the full 144-VGPR k_ld): load a K-subtile,
   mma_ABt-accumulate, discard; prefetch tile t+1's subtile during tile t's softmax (VALU) so the feed
   overlaps the matrix pipe. The LDS double-buffer (gather t+1 during compute t) is already cheap; the
   register-level overlap is the new work. This is the data-independent win.
3. **Lazy rescale** (wave-vote skip; data-dependent bonus; `-ffast-math` breaks `isnan`, use `x!=x`).
4. **Measure:** ATT + wall-clock vs gluon dedup (3.1 ms). Extend test to full TOPK=1152 + more seeds.
5. **Cleanup:** remove the `g.Dg` debug dump from `hk_fwd.cpp`.

**Files** (dsa_dev/hk_fwd/ + scp'd remote /workspace/_ab/): `hk_fwd.cpp` (the kernel, CORRECT),
`hk_fwd_test.cpp` (correctness harness, has att debug dump), `hk_gather_test.cpp` (gather check),
`hk_fwd_min.cpp`, `hk_gather_probe.cpp`, `hk_regprobe.cpp` (probes).

---

## flydsl hand-schedule — concrete target for HK scheduling (mined 2026-06-27)

Source: aiter FlyDSL gfx1250 MLA (`aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_schedule.py`,
`fmha_core_loop.py`). It is the **existence proof** that the cross-tile overlap gluon refused us is
worth hand-scheduling — flydsl writes an explicit **per-WMMA instruction schedule**. Their config is
D_qk=192/D_v=128 WMMA/wave32 (smaller than ours), and **they too eat the 4× redundant K read** (M-split,
see digest) — so the schedule is the *only* lever they have over gluon. Steal its **structure**, not its
numbers.

**Model.** A schedule is a table with one row per WMMA. Each row = the ordered list of *non-matrix*
tokens (LDS loads, softmax sub-ops, rescale, async DMA) the emitter injects **between WMMA i and i+1**,
so they execute in that matrix instruction's shadow. GEMM1 (QK) = 96 rows (24 WMMA × 4 SU-stages);
GEMM2 (PV) = 64 rows (16 × 4). Token vocabulary:

| token | meaning | cost |
|---|---|---|
| `K_Mx` / `V_Mx` | one `ds_load` of K/V (per accumulator-bank "MSB") | 1 |
| `P0_Mx` | softmax PART0: row-max (`max3`/permute) | 1 |
| `P1` | PART1: cross-bank max-merge + `delta` | 1 |
| `P2_Mx` | PART2: `pkfma`/setup/pkadd/cvt/sum-tree | 1 |
| `EXP_Mx` | the `exp2` (transcendental) | **3** |
| `O_RESC0` | one `acc*=alpha` `pk_mul` (4 per closure) | 1 |
| `TDM` | one async global→LDS DMA | 4 |

**The 7 transferable design rules (these are what to port into `hk_fwd.cpp`):**
1. **Fixed cycle budget per WMMA shadow (~7 cyc), no naked WMMAs.** Every matrix slot carries VALU
   (range tuned to 6–9 cyc, not 0–12). `ALU_PER_STAGE=[40,52,56,168,120,120,132,132]` budgets each stage.
2. **Producers far ahead of transcendentals.** All 16 `pkfma` emitted ≥16 ops before their `EXP` so the
   3-cyc `exp2` latency is fully hidden with no interleave (core_loop.py:148-155). Mirror: stage any
   `expf`/rescale input well ahead of use.
3. **`EXP` capped at ≤2 per WMMA row** (≤6 cyc of exp/shadow) — never pack 3–5 (would blow the slot to
   15 cyc). schedule.py:383.
4. **Strict softmax phase order across stages:** `P0(max) → P1(delta) → P2(pkfma) → EXP`, *spread across
   the 4 stages* so no single shadow is overloaded. P1 must finish before any P2 (P2 reads `delta`).
5. **Same-bank grouping within a row** to minimize accumulator-bank switches (`s_set_vgpr_msb` on RDNA4;
   the CDNA4 analog is grouping ops that touch the same AGPR/VGPR operand region to avoid bank stalls).
6. **Cross-tile phase split (the actual win):** since K leads V by one tile, the **current tile's
   QK-WMMA shadows (GEMM1) carry the *previous* tile's PART2 finish + O-rescale**, while the **PV-WMMA
   shadows (GEMM2) carry the *current* tile's PART0/PART1 + EXP** (core_loop.py:139-173, the
   `EXP_PER_MSB_TO_G2` / `PART2_SPLIT` machinery). That is exactly the serial softmax→matrix dependency
   we want to break: softmax of one tile runs under the matrix work of the next.
7. **`sched_barrier(0)` around every WMMA** so LLVM cannot re-cluster the hand order (core_loop.py:391).
   HK equivalent: keep the mma intrinsics ordered and fence with `__builtin_amdgcn_sched_barrier(0)`
   (or sched-group hints) so the emitted order survives the backend scheduler.

**Mapping to our gfx950 numbers** (RESULTS.md): our per-iter = 5,236 cyc vs ~1,356 matrix floor → the
~3.9× serial overhead to hide is **softmax(~420) + rescale(~328) + the 684 feed-wait**. Rules 2/4/6 attack
the softmax+rescale (the realistic ~15-20% per the README caution); rule 6 is the cross-tile structure the
naive `hk_fwd_swp.cpp` lacked (it double-buffered but didn't *assign* prev-tile softmax into next-tile
QK shadows). **The 684 feed itself is BW-bound and survives any schedule** — do not expect it to overlap away.

**Tooling note:** flydsl keeps the schedule as an editable CSV (`save_schedule_to_csv` /
`parse_schedule_from_csv`, schedule.py:820+). If HK hand-scheduling gets large, replicate that:
a token table + emitter, so the schedule is data not code.

---

## LOG (newest first)

### 2026-06-29 (overnight, autonomous) — SWP build: knife-edge CRACKED, 20→8.8ms stable; gluon 3ms gap is structural
Drove the pipeline per the user's mandate (HK has MORE hw control than gluon → can pin what the compiler won't).
RETARGETED to the latest gluon early-gather SWP (aiter PR2922 c4b07fe5a = 3.03ms).
**KEY WIN — knife-edge cracked:** both scalar(stable) and async(unstable) kernels are VGPR=256, so the
knife-edge is the SCHEDULE not the reg count. Explicit `__builtin_amdgcn_sched_barrier(0)` pinning (the
control gluon lacks) makes the async gather DETERMINISTICALLY correct. hk_fwd4.cpp = stable async 2-buffer
pipeline: **20ms (scalar hk_fwd3) → 8.8ms**, deterministic PASS NW=1/4, all-topk-preloaded + reverse-swizzle
raw_buffer_load_lds + per-warp P roundtrip (lgkmcnt, no workgroup barrier). COMMITTED (f936f3c4).
**PERF DECOMPOSITION (noisy ±0.5ms, GPU2 idle):** FULL 8.8ms · NOGATHER 7.25ms · NORT(no P-roundtrip) 7.75 ·
NORT+NOGATHER 7.75. ⇒ pure serialized compute ≈7.5ms; roundtrip ≈1ms; gather ≈1.5ms (hidden when rt gone).
**THE GAP (8.8 vs gluon 3.03) IS STRUCTURAL, not the SWP overlap:**
- gluon's OWN SWP gain is only −9.3% (3.34→3.03). So overlap is a ~10% lever, NOT my 2.9× gap. My base
  compute (~7.5ms) is ~2.2× gluon's BASE (3.34ms) — same MFMA count, so it's overhead+scheduling:
  (a) keys-as-rows forces a per-tile LDS **P roundtrip** rt_16x16→rt_32x16 (gluon uses register
  convert_layout, no LDS); eliminating it needs QB=32→rt_32x32 reinterpret BUT that doubles acc to 256 AGPR
  (at the limit) — net ~1ms at best, not worth it. (b) The stability pinning (SB) serializes the matrix pipe;
  unpinned reaches ~7ms but is knife-edge-unstable → can't ship. (c) gluon = Leon's hand-tuned MFMA dot-operand
  layouts / ds_read_tr feeding; matching that is the real multi-week lever.
- **hk_fwd5.cpp** = cross-tile SWP attempt (carry S_prev, QK(t+1)∥softmax(t), late-V). Structurally builds but
  has a deterministic 256-elem correctness bug + is slower (11ms, 28 spill) — shelved: not worth it given the
  ~10% overlap ceiling. The base-compute gap must be closed first.
**HONEST STATUS:** correct + stable + 2.3× (8.8ms), knife-edge solved — but did NOT beat gluon (3ms). The gap
is gluon's hand-tuned base compute, not the pipeline. Path to 3ms = replicate Leon's MFMA-feed layouts +
register P-convert (large). Tools left: hk_fwd4 PROF_NOGATHER/PROF_NORT, SWP_SPEC.md, hk_fwd5 (WIP SWP).

### 2026-06-28 — vectorized gather works (2.2×) but trips the knife-edge → structural rework needed [KEY finding]
Vectorized the gather to 16-byte int4 (the st swizzle leaves the low 16B contiguous, KV rows contiguous in
d → 1 coalesced int4 load + int4 LDS store per (k,d8)). **Validated CORRECT in isolation (gcheck: worst 0.0,
NW=1/4) and 20.2→9.1 ms (2.2×).** BUT in the full kernel it makes correctness NONDETERMINISTIC (nan varies
1648/1680/1664 across identical runs) = the codegen KNIFE-EDGE again — the faster gather shifts instruction
scheduling and the fragile single-buffer kernel tips over. Reverted to scalar gather (default; int4 kept
under `#ifdef VEC_GATHER`). Scalar default = deterministic PASS.
STRATEGIC CONCLUSION: the kernel is on a codegen knife-edge (confirmed 3×: late-V was the only stable point;
the mask, per-tile-Q, and int4 gather each tip it). **Incremental perf optimization is futile here** — every
speedup perturbs scheduling and flips correctness. The perf phase REQUIRES the structural rework that both
(a) relieves register pressure so we're not pinned at VGPR=256, and (b) pins instruction order so the
compiler can't wander. That = **gqa-style subtiled mma (stream K_lora/V in 16-col chunks, never hold the full
[32,512]=128-VGPR tiles) + explicit sched_group_barrier scheduling + double-buffer async pipeline.** This is
the same gqa-structured rewrite the overnight log kept pointing to, now justified by the perf data.
CEILING REMINDER (project memory): the 684-cyc K-feed is BW-bound; realistic HK win over gluon is ~15-20%,
not a big multiple. Decision point for next session: invest in the subtiled+scheduled rewrite (sizeable) vs.
accept HK fwd as a correctness/structure asset. Bench harness + VEC_GATHER are ready for that rework.

### 2026-06-28 — full-grid benchmark: 20.2 ms vs gluon 3.10 ms (gather-bound) [baseline]
Generalized hk_fwd3 to a real grid: blockIdx.x=token, blockIdx.y=head-group, gridDim.y=#HG. Q/O laid out
[T,HG,NW,QB,D] flattened -> coord{token*HG+hg, warpid,0,0}; topk{token,0,j,tid}; sink{0,0,hg*NW+warpid,0};
KV shared (topk-indexed). Single-program test still passes (grid(1,1) -> qo_row=0, unchanged).
hk_fwd3_bench.cpp: T=4096,H=128,TOPK=1152(nt=36) = gluon's config; hipEvents, warmup 5, 20 iters; GPU idle-
checked (rocm-smi all 0%), ran on HIP_VISIBLE_DEVICES=2.
RESULT: **20.2 ms/iter vs gluon 3.10 ms (~6.5× slower)** — expected: the gather is naive SCALAR per-element
global-load+LDS-store (correctness-first). That's the dominant cost and the #1 optimization target (gluon
uses vectorized async buffer_load_to_shared, size_per_thread=8). Grid indexing VALIDATED: spotcheck across
(token,head) shows error ∝ 1/navail (t=4000/2047 @1152 keys ~0.006 clean; t=100 @101 keys 0.023; t=0 @1 key
0.07 = bf16 MFMA-vs-scalar-ref reduction-order divergence in the few-key regime, NOT a grid bug — kernel
matched torch 7e-5 historically). NEXT: vectorized/async gather (16B/thread) -> double-buffer pipeline -> SWP.

### 2026-06-28 — -1 mask now CORRECT at NW=4 (production) via low-footprint col_vec ✅
The float-tile mask tripped the NW=4 register knife-edge (nondeterministic NaN; nan-count varied
400/496/512 across identical runs — confirmed race, not logic). Two pressure-relief attempts (fold mask
into S-as-mma-C; per-tile Q load) both made it WORSE (per-tile-Q broke even ninv=0 — "q-in-loop wrecks
allocation", matching the overnight log). WINNING FIX = apply the mask as a **per-key col_vec** (no float
tile): `load mtb(bf16) → row_max(cvb, mtb)` [each row constant → per-key value] `→ copy to float col_vec →
add_row(s,s,cv)`. ~2 VGPR vs the 8-VGPR float tile → dodges the knife-edge. VGPR 256 / AGPR 245 / 2 spill.
Validated (tail-invalid = real DSA warmup pattern, incl. whole all-invalid trailing tiles):
  NW=4 nt=4: ninv 0/13/40 PASS (det., 3 runs × 3 seeds). nt=36: ninv 0/100/500 PASS clean (worst <0.008).
  Extreme warmup (ninv→only ~28-52 valid keys): worst ~0.035-0.04, 0 nan, DETERMINISTIC = bf16-P precision
  tail (fewer keys → each carries more weight → more bf16 rounding), same regime as scale=1.0 sharp-softmax;
  present with/without mask. NOT a masking bug.
KNIFE-EDGE NOTE: the col_vec mask is stable at NW=4 (shipped) but FLIPS NW=1+mask to broken (garbage even
ninv=0) — opposite of the float-tile mask (NW=1 ok / NW=4 broken). The kernel is on a codegen knife-edge;
NW=4 (M-split, = gluon config) is the production target and is correct. Default build (no mask) = NW=1 AND
NW=4 both clean (24/24). A non-knife-edge codegen comes with the pipeline rework.

### 2026-06-28 — pushed to fork [snapshot]
Committed the correct kernel + validators to **wangye805/HipKittens**, branch **`dsa-mla-fwd`**,
dir `kernels/attn/dsa_mla/` (hk_fwd3.cpp + mini3*/gcheck + README), based off upstream 840c9670
(commit ab3e343e). Local fork clone: `~/sandbox/dsa_triton/HipKittens` (gh auth = wangye805, https).
Pushed from laptop (the cv350 node has no GitHub creds). NOT a PR — just the branch.

### 2026-06-28 — hk_fwd3 FULL KERNEL CORRECT ✅✅✅ [the NW>1 wall is CRACKED; root cause found]
hk_fwd3.cpp = gather + two-mma_ABt QK + online softmax + sink + PV, single-buffer. **ALL 24 cases PASS**
deterministically: NW∈{1,4} × nt∈{4,36} × sink∈{0,1} × 3 seeds. mean ~6e-5..4e-4, worst <0.02, 0 nan.
THE BUG that caused months of NW>1 fragility + this session's non-deterministic NaN = **register-lifetime
corruption from holding k_l[32,512]=128VGPR AND v_l[32,512]=128VGPR simultaneously** (peak live set blew the
knife-edge at VGPR=256). FIX = **load v_l LATE** (after QK, k_l dead → regs reused). This is exactly the
"NW>1 needs register-PRESSURE RELIEF not pinning" the overnight log predicted. Confirmed: barrier-thrashing
+ sched_barriers did NOT fix it; the late-V restructure made it deterministic in one shot.
Other essential fixes found this session:
- **forward-swizzle gather** (iterate logical (k,d), forward swizzle+subtile offset) replaces hk_fwd's fragile
  reverse-swizzle gather. Validated standalone (gcheck): worst 0.0 NW=1/4, K row_l + V col_l from one tile.
- **LDS roundtrip needs __s_barrier** (per-warp ps still needs the barrier between store and load).
- **MEASUREMENT SCAR**: `scp ... && echo SCP_OK` printed OK but SILENTLY did not land the file several times
  (sftp over the auth gateway). md5/grep-verify the REMOTE file after every scp before trusting results — a
  whole debugging detour (false all-NaN) was on stale files.
Resource: VGPR 256, AGPR 230, 0 spill, occ 1. hk_fwd3.cpp + hk_fwd3_test.cpp (self-contained, DBGS dumps S).
-1 MASK (2026-06-28): added under `#ifdef ENABLE_MASK` — in-kernel broadcast mask tile (fill_mask forward-
swizzle from topk, bf16 -1e30, add to S pre-softmax). **NW=1 CORRECT** (valid+invalid match masked ref).
NW=4 with mask re-trips the knife-edge (the transient float mask tile's 16 VGPR over the edge); folding the
mask into S-as-mma-C was WRONG (garbage even ninv=0). Default build = NO mask (24/24 clean). Mask NW=4 to be
fixed with pressure relief during the pipeline rework (will restructure registers anyway).
REMAINING: full-grid benchmark harness (kernel currently single-program grid(1); needs blockIdx→token/
head-group coords) → benchmark vs gluon 3.1ms → double-buffer async pipeline → hand-scheduled cross-tile SWP.

### 2026-06-28 — online-softmax loop + natural-operand QK BOTH validated ✅ [building blocks complete]
- **mini3_loop.cpp** (online softmax, NW=4, gather-free, host per-tile K): nt=1/4/8 × 3 seeds ALL PASS,
  worst DECREASES with nt (nt=8 worst 0.005) — running m_i/l_i/acc-rescale in keys-as-rows is solid.
- **mini3b.cpp** (the KEY simplification): QK via **mma_ABt(S, K, Q)** with NATURAL K[TILE_K,D_V] +
  NATURAL Q[QB,D_V], both row_l → S=[TILE_K,QB] keys-as-rows. NO transpose, NO transposed gather, NO
  Q-transpose. **V = the SAME natural K shared tile re-loaded as col_l rt_32x16** (gluon "V=permute" win
  via one gather + two register loads). NW=4 × 4 seeds PASS (mean ~0.0001). This is strictly better than
  both the old mma_ABt-576 (hk_fwd) and the mma_AtB-transposed (mini3) paths.
DECISION: hk_fwd3 = mini3b QK (mma_ABt natural) + mini3_loop online softmax + gather (natural [TILE_K,D]
orientation, reuse hk_fwd's proven gather_to_shared + col_off for rope) + sink + -1 mask. Gather is the
only unvalidated piece left (history flags cooperative gather as the NW>1 perturber — test carefully).

### 2026-06-28 — mini3 keys-as-rows: NW=1 AND NW=4 M-split BOTH CORRECT ✅✅ [MAJOR — NW>1 wall cracked]
Built + ran on cv350 (container yewang_aiter_main, HIP_VISIBLE_DEVICES=2). The keys-as-rows path works
where the old transposed/heads-as-rows hand-rolled kernel was fragile for months.
- **NW=1**: 5 seeds PASS, worst ~0.012-0.019, mean ~0.00023, 0 over .03, 0 nan.
- **NW=4 M-split**: 5 seeds PASS (thresh .03), worst ~0.014-0.024, **mean ~0.0001**, 0 over .03, 0 nan.
  The few elems grazing .02 are the bf16 precision tail (4× more heads = more order-stat draws); NOT
  corruption (a coord bug corrupts whole 8192-elem warps, cf the historical 1.79 failures). CONFIRMED via
  error histogram (>.02/>.03/>.05 counts + mean).
WHAT MADE IT WORK (the three things the old kernel lacked):
1. **keys-as-rows orientation** (S=[TILE_K,QB] col_l) → col_max/col_sum/div_col all one row_vec family,
   no vec-layout conversions (gqa-native).
2. **LDS roundtrip for P** rt_16x16→rt_32x16 via st_bf<.,.,st_32x16_s> (store rt_16x16, load rt_32x16, both
   hit the clean "shared subtile≥reg base" path) — replaces the fragile register reinterpret.
3. **per-warp depth coord** {0,warpid,0,0} for Q load + O store (NW in gl dim-1); per-warp ps[NW] LDS
   buffers. Col-dim {0,0,0,warpid} was the silent-fail trap.
mini3.cpp = gather-free single-tile (host supplies K transposed + V natural). NEXT: online-softmax
multi-tile loop → topk gather → double-buffer pipeline (hk_fwd3) → sink + -1 mask → benchmark vs gluon.

### 2026-06-28 — mini3.cpp: keys-as-rows validator written (gqa discipline + gluon structure) [WIP, awaiting build]
Per PORT_SPEC SYNTHESIS, started fresh on the keys-as-rows path (abandons mini_g heads-as-rows
dead-end). Wrote `mini3.cpp` (gather-free, NW=1, single-tile) + `mini3_test.cpp` (bf16-faithful ref).
Design, all confirmed against HK headers (cdna4 mma/reductions/maps/shared_to_register):
- **Operands all col_l, base rt_32x16** (the mma_AtB A/B base shape for mfma161632); S/acc base rt_16x16.
- **QK = two mma_AtB**: `mma_AtB(S, K_T, Q_T)` with K_T=[D_V,TILE_K], Q_T=[D_V,QB] (both col_l);
  S[k,h]=Σ_d K[k,d]Q[h,d]. Asserts check out (D::rows==A::cols=32, D::cols==B::cols=16, A::rows==B::rows).
  rope adds `mma_AtB(S, K_rope_T, Q_rope_T)`.
- **Softmax keys-as-rows**: `col_max(m,S)`→S::row_vec (per-head); `sub_col`,`exp2`,`col_sum(l,S)`. ONE
  vec family (S::row_vec == acc::row_vec, both width=1) → no vec conversions (the SYNTHESIS win).
- **P roundtrip via LDS** (the proven BUG-1 mechanism, not the fragile reinterpret): copy S→bf16 rt_16x16,
  `store` to `st_bf<TILE_K,QB,st_32x16_s>`, barrier, `load` into rt_32x16 P-operand. Used st_32x16 (not
  st_16x16) so both store(from rt_16x16) and load(to rt_32x16) hit the clean "shared subtile≥reg base" path.
- **PV = mma_AtB(acc, V, P)**: V=[TILE_K,D_V] col_l (=K natural, host-supplied for now); acc=[D_V,QB].
  `div_col(acc,acc,l)`; transpose acc→[QB,D_V] row_l; store.
GATHER-FREE + host-supplied transposed K and natural V isolates compute/layout from gather + permute-view
(those come in hk_fwd3). NEXT: user builds/runs mini3_test (ISA-first), confirm PASS, then NW=4 M-split.

### 2026-06-27 — CORRECTNESS ACHIEVED ✅ [MAJOR WIN]
Root cause of the NaN: **Q staged in a malformed shared tile** `st_bf<QB=16,DQK,st_32x32_s>` (16 not
divisible by the 32×32 base → garbage Q). **Fix: load Q DIRECT global→register** (`load(q_ld, g.Qg, …)`,
no LDS staging — Q is small). `hk_fwd_test.cpp`: **max_abs=0.0002, mean=4e-5, 0 bad — PASS** vs CPU
sparse-MLA+sink ref (att QK matches to 0.025 = bf16 dot rounding). Full chain correct: topk gather →
mma_ABt QK (no transpose) → online softmax(+sink) → att→bf16 reinterpret → PV → normalize. Transpose +
Q-LDS staging both eliminated.

### 2026-06-27 — QK NaN debug: transpose ruled out; Q shared-tile bug found [→ fixed above]
- att (QK output) was NaN. **CAVEAT: `-ffast-math` breaks `std::isnan`** (reports nan=0 while values
  print -nan) — use `!(x==x)`.
- Replaced QK transpose+`mma_AtB` with **`mma_ABt(att, k_ld, q_ld, att)`** (A·Bᵀ on loaded rt_16x32
  tiles, NO transpose). Still NaN ⇒ **transpose was NOT the bug** (kept it — also saves VGPR).
- Found the real bug: `qs = st_bf<16,…,st_32x32>` malformed (16 not div 32) → garbage Q.

### 2026-06-27 — Gather VERIFIED numerically correct; NaN localized to compute [PROGRESS]
`hk_gather_test.cpp`: gather KV[topk[r],:] → LDS → reg → Og vs expected rows. **PASS max_abs=0.0020,
0 bad/0 nan** (row0 matches KV[topk[0]=3017]). Gather confirmed correct ⇒ NaN was in the compute chain.

### 2026-06-27 — Correctness harness built; kernel outputs NaN [→ debugged above]
`hk_fwd_test.cpp` = standalone HIP main + CPU sparse-MLA+sink ref (exp2 convention), allclose vs O[DV,QB].
Gotchas fixed: `g_t` needs aggregate-init (gl no default ctor); shmem ≤163840 (gfx950 max, kernel ~86KB).

### 2026-06-27 — Correct-structure full fwd compiles [WIN]
`hk_fwd.cpp`: complete online-softmax fwd. Compiled **VGPR 240/256, AGPR 128, occ 1, 0 spills** (up from
134 — the in-register Q/K transposes were the cost; later removed via mma_ABt + direct-Q-load).

### 2026-06-27 — Integration gate PASSED with margin [WIN]
`hk_fwd_min.cpp` (gather+QK+softmax+PV, single-buffer): **VGPR 134/256, AGPR 128, occ 1, 0 spills.**
Refuted the "256-ceiling/will-spill" worry — 122 VGPR headroom. HK port register-VIABLE.

### 2026-06-27 — Gather IMPLEMENTED + compiles [WIN]
`hk_gather_probe.cpp`: `gather_to_shared<NT>(st, kv_gl, topk)` = contiguous `global_to_shared` loop with
`phys_row=topk[g_row]`, via `raw_buffer_load_lds`. **VGPR 146, AGPR 0, 0 spills.** HK's missing primitive done.

### 2026-06-27 — Sparse gather mechanism designed [recipe pinned]
HK has no gather primitive but exposes `make_srsrc` + `raw_buffer_load_lds` (= gluon's
`buffer_load_to_shared`). Gather = the contiguous loader with the row indirected through `topk_idx`.

### 2026-06-27 — Register-feasibility probe [GATE PASSED, TIGHT]
`hk_regprobe.cpp`: SWP working set (Q-resident + acc[512,16] + K/V 2-buffer + att×2). **VGPR 256 (maxed,
0 spill), AGPR 214, occ 1.** Feasible but tight; later the real kernel came in far under (134/240).

### 2026-06-27 — SWP attempt: SLOWER + scale correctness issues [NEGATIVE — important]
`hk_fwd_swp.cpp` (double-buffer LDS ks[2]/vs[2], att ping-pong, QK_{j+1} structured to overlap softmax_j).
Compiles **VGPR 256, AGPR 170, occ 1, 0 spills** (single rotating k_ld + att[2] fit — no 2× K needed).
BUT:
- **~11% SLOWER:** SWP 112.8 µs/launch vs serial baseline 101.7 µs (1 program, 36 tiles, grid(1) A/B).
  The compiler did NOT overlap softmax(VALU)∥QK(matrix) — the double-buffer + extra __syncthreads cost
  more than any overlap. The naive loop-restructure does NOT realize the win; would need explicit
  cluster-style hand-scheduling (like gqa's 8-wave ping-pong) — much more work. Consistent with the
  feed being BW-bound (little idle matrix time to fill) → the data-independent overlap ceiling is thin.
- **Correctness at scale:** `hk_fwd` PASSED only at n_tiles=2 (max_abs 2e-4). At n_tiles=36 baseline =
  8% elems >0.05 vs fp32 ref (likely bf16 precision: bf16-P numerator vs fp32-P denom over 1152 keys —
  need a bf16-faithful ref to confirm it's precision not bug); SWP adds real pipeline errors (48% bad).

**Assessment:** the naive SWP doesn't beat the serial kernel, and the calibration (BW-bound feed,
~15-20% ceiling) is holding — the HK *perf* win is NOT cheaply attainable; it needs explicit
hand-scheduled cluster pipelining AND the feed itself can't be sped up. **Leaning: HK unlikely to beat
gluon for this kernel** without attacking the feed (which is structural). 
**NEXT:** (1) bf16-faithful reference to confirm the serial kernel is correct-at-scale (precision, not
bug); (2) decide go/no-go on HK given the SWP-slower result — likely pivot back to the fused-kernel feed
levers (cut 4× K read / ds_read_tr, per the other session's redirect) which attack the actual bound.

### 2026-06-27 — Correctness debug: reinterpret bug FIXED; 2nd data-dependent bug remains [IN PROGRESS]
Method: sweep n_tiles + **compile test WITHOUT -ffast-math** (it breaks isnan AND nan comparisons →
false PASS/FAIL when nan present) + **bf16-faithful CPU ref** (round Q/K/V + P to bf16).
- **BUG 1 FIXED — att→bf16 reinterpret was wrong.** `att_b = *reinterpret_cast<AbT*>(&att_s)` cast
  rt_16x16→rt_32x16 which are NOT layout-compatible (HK `copy` requires same shape; gqa only
  reinterprets compatible rt_32x32↔rt_16x32_4). Fix: **LDS round-trip** via `st_bf<KV,QB,st_16x16_s>`
  (store att_s → __syncthreads → load att_b). n=4 FAIL(0.10)→PASS(0.007); n=1,4,8 all PASS.
- **BUG 2 (open) — data-dependent catastrophe at some datasets.** n=16 FAIL max_abs 1.79 (94% bad),
  n=36 max_abs 2.06 (37% bad) — NON-monotonic (16 worse than 36) ⇒ specific random top-k is
  pathological, not a tile-count limit. Magnitude ~2 (O~0.01-0.1) = garbage on those outputs.
  Suspects: gather on a specific topk value; or online-softmax/LDS-roundtrip on specific data; NOT
  precision (bf16-faithful ref). NEXT: per-tile state dump (store m_i/l_i/att per tile) on a failing
  dataset to find which tile/stage diverges. Harnesses: hk_fwd_base_test.cpp (n_tiles=argv[1], no-ffast),
  bf16-faithful ref in it.

### 2026-06-27 (cont) — bug 2 = RACE (non-deterministic), partial fix [IN PROGRESS]
**CONFIRMED RACE:** same data, n=8 → FAIL/PASS/FAIL across 3 identical runs (non-deterministic).
- Gather wait was insufficient: `s_waitcnt vmcnt(0)` does NOT fully wait for `raw_buffer_load_lds`;
  gqa uses **`__builtin_amdgcn_s_waitcnt(0)`** (full, all counters). Switched to that → n=8 went
  deterministic-PASS (3×) at one point.
- Added `__builtin_amdgcn_s_waitcnt(0)` after each LDS/global load (q_ld, k_ld, v_reg) before its mma
  — but n=8/16 went back to all-NaN, n=4 PASS, n=36 varies run-to-run. So a race REMAINS and the extra
  waits shifted timing rather than fixing it.
**Remaining leads (NEXT):** (1) verify GPU device idle (rocm-smi) — non-determinism could be partly a
non-idle GPU (project warns avoid busy GPUs); (2) gather may leave an LDS region uninitialized — my
gather_to_shared copied global_to_shared.cuh's MAIN loop but NOT its "leftover loads" tail (line ~72);
for st_bf<32,576>/<32,512> the division is even (9/8) so *probably* no leftover, but VERIFY; (3) match
gqa's EXACT sync discipline (it interleaves `__builtin_amdgcn_s_waitcnt(N)` with specific counts +
`s_barrier`, not blanket waits); (4) the V gather/load and the per-tile ks/vs single-buffer reuse
ordering. This is fiddly race-debugging — best continued with focused attention (likely fresh session).
**Bug 1 fix (LDS-roundtrip P convert) is SOLID and should be kept regardless.**

### 2026-06-27 (cont) — GPU idle confirmed → race is REAL (kernel, not contention)
rocm-smi: GPU 2 (HIP_VISIBLE_DEVICES=2) = 0% use, 0 VRAM; only GPU 5 has a process. So the
non-determinism is an intra-kernel race, NOT GPU contention. Debug stays on kernel sync.
**Refined hypothesis (strongest lead):** the multi-warp structure may be ill-defined. gqa gives each of
its 8 warps a SEPARATE Q-block (independent per-warp rt tiles). My kernel uses NW=4 on ONE Q-block but
doesn't differentiate warps for the COMPUTE — the gather is cooperative (4 warps fill ks together) but
the QK/softmax/PV + store may be per-warp redundant OR warp-distributed in a way I'm not handling, →
a store race / inconsistent reads. NEXT: (1) determine how HK distributes the rt tiles across 4 warps
for a single Q-block (per-warp-replicated vs warp-split); (2) either use NW that matches the intended
distribution, or follow gqa's explicit per-warp work assignment; (3) THEN re-sweep n_tiles (no-ffast,
bf16 ref). This + the sync discipline are the remaining correctness work.

### 2026-06-27 — RACE SOLVED: it was 4-warp MISCONFIG. Kernel CORRECT at NW=1 ✅✅
Decisive test: **NW=1 → deterministically PASS at n_tiles=8/16/36 (max_abs 1e-4, repeated runs identical).**
So kernel LOGIC is correct (math, online softmax, gather, bug-1 LDS-roundtrip P-convert all right). The
race was a **misconfiguration**: QB=16 (one warp's worth of rows) + NW=4 → all 4 warps redundantly
computed the SAME 16 rows and raced on the cooperative gather + shared store. NW=1 removes the
cross-warp interaction → correct.
**CORRECTNESS ACHIEVED (NW=1).** Bug 1 (reinterpret→LDS-roundtrip) + bug 2 (4-warp misconfig) both
resolved.
**Design choice for perf NEXT:** (a) keep NW=1, QB=16 (1-wave workgroups — may give GOOD occupancy via
many small workgroups; but gather is 4× less parallel), OR (b) the gluon-matching NW=4, QB=64 (4 warps
split BLOCK_H=64 = 16 rows each via proper warps_per_cta rt distribution — needs the warp-split layout,
more work). Measure NW=1 perf first (it might already be competitive). THEN the SWP (now on a CORRECT
base). The earlier SWP-slower result was on the racy/incorrect 4-warp base, so re-evaluate on NW=1.

### 2026-06-27 — EXTENSIVE correctness sweep (hk_fwd_extensive.cpp, NW=1) [valuable — found gaps]
8 seeds × 8 n_tiles × 2 scales = 128 cases + edge cases; bf16-faithful ref; no -ffast-math.
RESULT: **113/128 pass**. Findings:
1. CORE SOLID: valid topk @ scale 0.1 = 100% pass (8 seeds); scale 1.0 passes for n_tiles>=3.
   Duplicates PASS, sink-dominant PASS.
2. **MISSING FEATURE — invalid (-1) topk mask.** Kernel reads KV[-1]→0 (OOB) and INCLUDES the zero key
   in softmax instead of excluding it. Edge test "passed" (0.002) only because O~0.01 at scale 0.1 masks
   the ~20% denom inflation (would be ~0.2 at scale 1.0). **Real DSA V4 has -1 padding → must add
   valid-mask (-inf for invalid) + safe gather (clamp -1→0).** ACTION ITEM.
3. BORDERLINE: scale=1.0 + n_tiles 1-2 (max 0.08-0.14 vs 0.08 tol) — sharp softmax amplifies
   bf16-P-num/fp32-P-denom inconsistency. Likely precision; consider computing `l` from bf16 P for
   consistency. Investigate.
**Correctness is NOT "complete"** — it's solid for the common path but needs: the -1 mask (feature),
scale-1.0 sharp-softmax confirmation, and re-run on the NW=4/QB=64 perf config once built (NW=1 only so far).

### 2026-06-27 — Cross-validated vs PyTorch behavior model [valid path CONFIRMED correct]
hk_fwd_dump.cpp (dump inputs+HK O) + val_torch.py (independent bf16-faithful torch sparse-MLA+sink ref).
- **VALID topk: torch-vs-HK max_abs=7e-5, PASS** → HK kernel + my CPU ref both independently confirmed
  CORRECT (not self-consistent). Strong gold-standard validation.
- **-1 topk: ~20% relative gap confirmed** (O_hk -0.0034 vs torch -0.0042) — torch masks -1, HK doesn't.
  Independently confirms the missing-mask gap. Production-critical.
NEXT: add -1 mask (safe gather clamp -1->0 + att row -> -inf for invalid) → re-cross-validate (should
match torch on -1 too). THEN NW=4/QB=64 perf config + benchmark.

### 2026-06-27 — -1 MASK ADDED → correctness COMPLETE (validated vs torch gold standard)
Fix: (a) safe-gather clamp -1->0 (no OOB); (b) add_row(att,att,mask_colvec) with -1e30 for invalid,
applied post-(QK*scale) pre-softmax. mask_colvec = AfT::col_vec (KV-indexed, height=32) — NOT row_vec
(row_vec=QB-indexed; add_row needs outer_dim==tile.height). Mask passed as new g_t.Mg [nt,KV] input
(host precomputes -1e30/0 from topk).
RESULTS (cross-validated vs PyTorch behavior model, scale 0.1 = realistic post-RMSNorm regime):
  valid topk: 0.00014 | -1 topk: 0.00014 (was 0.002 unmasked) | dups: 0.0002 | sink: 0.0001  ALL match torch.
  EDGE -1 in extensive sweep: 0.002 -> 0.0001 (now genuinely correct, not abs-hidden).
scale=1.0 ~0.04 = bf16 precision (sharp softmax); affects VALID equally (0.037 valid vs 0.044 masked),
  scales smoothly with magnitude (0.3->0.0017, 0.5->0.0059). NOT a bug; not the operating point.
**CORRECTNESS COMPLETE for the realistic regime.** Benchmarking now justified. NEXT: NW=4/QB=64 perf
config (proper 4-warp split, not redundant) + re-validate there + benchmark vs gluon 3.1ms.
Harnesses: hk_fwd_extensive.cpp (sweep+edges), hk_fwd_dump.cpp + val_torch.py (torch cross-check). All Mg-wired.

### 2026-06-27 (overnight) — M-split NW=4 attempt: ROOT-CAUSED to VGPR=256 register-lifetime, NOT fixed
Implemented M-split (NW warps each own QB=16 heads; HBLK=QB*NW; warpid coords into Q/sink/O; per-warp
ps; ceil+guard gather robust to NT not dividing the tile). NW=1 stays VALIDATED. NW>1 fails — full diagnosis:

FIXED along the way:
- **NW=1 head-0 bug**: removing the old debug Dg-store corrupted head-0 o_reg. Root = compiler scheduling
  fragility. Fixed with `__builtin_amdgcn_sched_barrier(0)` after the mask (flydsl recipe, no global write).
- **Gather**: K tile [32,576]bf16=36864B / (bpt32*256thr)=4.5 → truncated mpt left K partially un-gathered
  → NaN. Fixed: ceil mpt + `if(lbo>=total) continue` guard. (Confirmed NOT the multi-warp bug: serialized
  warp0-only gather still failed.)

NW>1 BUG — ROOT CAUSE = register-lifetime corruption at the VGPR=256 cap (launch_bounds(256,1)):
- Ruled OUT: gather (serialized still fails), ps aliasing (allocate advances ptr → ps0-3 distinct), sync
  (max-sync fails), spills (0 scratch). col_max/col_sum are warp-local __shfl (no LDS race).
- LOCALIZED by per-stage dumps: QK `att` is computed CORRECT for all warps, BUT by the time softmax
  `col_max` reads it, **warps 2,3's att registers read as ZERO** — UNLESS a store pins att (same-build
  att-dump+m_new-dump showed att=0 for w2,3; separate att-only dump showed att correct → the store itself
  changes the outcome = knife-edge scheduling under max VGPR pressure).
- CONFIRMED by pinning: store att every iter → seeds 1,2 PASS, seed0 806 bad. + pin m_i (running max)
  every iter → **valid-topk scale-0.1 ALL seeds PASS** (0.007-0.012). But mask(mode1) & scale-0.5 STILL
  FAIL → MORE values lost (mvec/l_i/att_b/...). Pinning is whack-a-mole.
- VGPR: NW=1=240 (headroom, correct); NW=4=256 (capped, no headroom → lost values). 0 spills either way.

**CONCLUSION: NW>1 needs register-PRESSURE RELIEF, not pinning.** Next-session levers (ranked):
1. Drop Q-resident q_ld (~144 VGPR): stage Q in LDS once, load per-tile (frees headroom → no lost values).
2. Adopt gqa's proven multi-warp tile shapes (rt_32x32 / rt_16x32_4 att → the permlane32 reduction path;
   ours rt_16x16 → __shfl path) — may pack tighter + is the battle-tested multi-warp layout.
3. Relax launch_bounds / accept lower occupancy to gain VGPR headroom.
MEASUREMENT SCAR: a stale /tmp/dpin2 binary (ls||hipcc skipped rebuild) gave a FALSE all-PASS; always
rm+rebuild and md5/`dumped`-check that the dump wrote fresh data before trusting val_torch.
**Kernel left at validated NW=1** (default #define NW 1). NW=4 scaffolding present, documented buggy.

### 2026-06-27 (overnight, cont.) — pressure-relief lever refined: SUBTILE K/V (not Q)
Tried moving q_ld load into the loop → VGPR still 256, still fails (compiler hoists/keeps Q resident; Q
wasn't the binding consumer). The real VGPR hogs are the FULL register tiles: k_ld[32,576]≈144 VGPR +
v_reg[32,512]≈128 VGPR. gqa avoids this by SUBTILING the mma (`subtile_inplace<16>(v_reg,0..3)`,
`subtile_inplace<16>(att_b,...)`) so K/V are consumed in 16-col chunks, not all-resident.
**NEXT-SESSION primary fix for NW>1: subtile the QK and PV** (stream k_ld/v_reg in chunks, gqa-style) to
drop peak VGPR below 256 with headroom → eliminates the register-lifetime corruption (no pins needed).
Then re-validate NW=4 (extensive+torch), then benchmark vs gluon 3.1ms, then SWP. Status: NW=1 validated;
NW=4 root-caused (VGPR=256), partial (att+m_i pin → valid scale-0.1 passes; mask/scale-0.5 still lose values).

### 2026-06-27 (overnight, cont.2) — NW>1 reframed as COMPILER codegen bug; subtiling/opt-flags don't fix
More attempts on NW>1, all on the validated NW=1 base (which still PASSes 0.00007 after each revert):
- **Chunked PV** (stream V in 128-col chunks via subtile_inplace) → VGPR STILL 256 + broke NW=1 → reverted.
  Lesson: VGPR=256 is the compiler USING its full launch_bounds(256,1) budget, NOT real pressure (0 spills).
  So "reduce the working set" is the wrong model — the compiler uses 256 regardless.
- **Opt level**: -O1 reduces error (0.061) vs -O3 (0.105) but still FAILS. Opt-sensitive → scheduling/codegen.
- **Q-in-loop**: VGPR still 256, no help.
REFRAME: NW>1 bug = **compiler value-materialization bug at multi-warp** — the QK MFMA result `att` (and
mvec/m_i/...) is not reliably kept in registers for warps>0 unless a STORE forces materialization. A store
pin reads the value → forces the MFMA to retire / the value to stay live. This is pervasive (att, m_i,
mvec, scale-0.5 all need it) → per-value pinning is whack-a-mole, not a real fix.
**RECOMMENDED next path (pick one):**
1. **Rewrite to gqa's EXACT proven multi-warp structure** — att as rt_32x32_s, the `*reinterpret_cast<
   rt_16x32_4_s*>` conversion (NO LDS roundtrip), gqa's loop/subtile pattern. gqa runs multi-warp on
   gfx950 correctly; our rt_16x16 + LDS-roundtrip path is the outlier the compiler mishandles. Biggest bet.
2. Try a different ROCm/LLVM toolchain version (the bug may be version-specific) or `-mllvm` sched flags.
3. Heavy-pin (att+m_i+mvec+l_i...) for a CORRECT-but-slow NW=4 base to unblock SWP, optimize pins later.
STATUS: **NW=1 is the validated, shippable kernel.** NW>1 root-caused to a compiler codegen issue; clean
fix needs the gqa-structure rewrite. att+m_i pin already gets valid-topk scale-0.1 correct for all seeds
(proof the M-split math is right; only codegen is wrong).

### 2026-06-27 (overnight cont.3) — ROOT CAUSE = HK coord convention (NOT compiler). User was right.
Built a MINIMAL multi-warp repro (load Q per-warp + shared K + mma + col_max + store). Reproduced the bug,
then sentinel-isolated it: **the per-warp STORE never lands for warps 2,3** — pure store-coord bug, no
compute/register/compiler involvement. gqa puts the per-warp index in an OUTER coord dim
(`{batch, tile_idx, head, 0}`); my `{0,0,0,warpid}` (last/col dim) **silently fails for col-tile >= 2**.
FIXES (verified via mini_pv single-tile full compute: NW=4 PASS 0.0003):
- row_vec (sink) load: `{0,0,0,warpid}` -> `{0,0,warpid,0}` (warp in ROW dim, Sg=[.,.,NW,QB]).
- tile (o_reg) store: `{0,0,0,warpid}` -> `{0,warpid,0,0}` (warp in DEPTH dim, Og=[1,NW,DV,QB]); harness
  rearranges [NW,DV,QB]->[DV,HBLK] for val_torch.
RESULT: **NW=4 now PASSES nt=1, nt=2, and -1 mask** (was 100% broken). NW=1 unaffected (still 1e-4).
REMAINING (nt>=3 @ NW=4): the **mask col_vec load** `load(mvec,g.Mg,{0,0,j,0})` reads wrong for tile j>=2
(removing the mask makes nt=4 PASS) — col_vec load convention differs from row_vec, needs the analogous
fix or build-mask-in-kernel-from-topk. Plus a residual nan at higher nt (nomask nt=8) to chase.
ALL my earlier "compiler codegen bug / register pressure" conclusions were WRONG — it was coord misuse
masked by contaminated row_vec dumps (the dumps themselves used the buggy coord). Repro tools: mini.cpp,
mini_pv.cpp (keep — they isolate HK multi-warp coord issues fast).
NEXT: fix mask col_vec coord (or in-kernel build) -> nt>=3 -> full extensive+torch @ NW=4 -> benchmark.

### 2026-06-27 (overnight cont.4) — ISA-grounded debugging (user's method). Coord bug = real; residual = scheduling-fragile register issue.
Established ISA workflow: `hipcc ... --save-temps -c` -> `*-hip-amdgcn-amd-amdhsa-gfx950.s`. Reference =
mini_pv (NW=4 PASS): warpid=`v_lshrrev_b32 v,6,v0`, per-warp stores w/ correct offsets, 304 VGPR+128 AGPR.
FULL-KERNEL ISA findings (the failing NW=4):
- **LDS ordering is CORRECT** (rules out the race I suspected): all 3 s_barrier are preceded by full
  `s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)`; the gather `buffer_load...lds` is fully waited before its barrier.
- Registers FIT: 432 VGPR-total (ArchVGPR 256 + AGPR 176) <= 512 @ occ1. NOT exhaustion (user was right).
- `sched_barrier(0)` after the mask is REQUIRED — removing it makes the WHOLE kernel NaN (real ordering dep).
RESIDUAL (NW=4 nt>=2): the failure is **one full warp's o_reg = NaN** (O_hk nan=8192=16heads*512; O_ref clean),
and it FLIPS with tiny code changes (mask on/off, sched_barrier position). Not an LDS race (ISA), not register
exhaustion. => a **scheduling-sensitive register-lifetime / uninitialized-read** issue. PRIME SUSPECT = the
**ps LDS-roundtrip** (rt_16x16->rt_32x16 via st), the one non-standard piece gqa avoids (it uses a register
`reinterpret_cast` between compatible 32x32/16x32_4 shapes). The col_vec mask has the same fragility (corrupts
even mode-0 in some builds; NW=1 mode-1 drifts 0.00008<->0.02).
NEXT (highest-leverage): ELIMINATE the ps LDS-roundtrip — find a reinterpret-compatible rt pair for the
att->bf16 PV-operand conversion (study HK conversions.cuh + gqa's `*reinterpret_cast<rt_16x32_4_s*>`), OR
restructure att shape so QK output is directly the PV operand. That removes the suspect and likely the
fragility. Then re-validate NW=4 across nt with ISA spot-checks before each run.
STATE: NW=1 core (no mask) solid 1e-4; NW=1 mask fragile; NW=4 coord-fixed, blocked by the roundtrip fragility.

### 2026-06-27 (overnight, FINAL) — NW>1 = scheduling fragility; needs gqa-structured rewrite (not surface patches)
Step-1/2/3 chain CONFIRMED the bug location and proved the math:
- mini_loop (no gather) reproduces NO NaN -> not the running softmax.
- HOST_GATHER (host pre-gather, skip in-kernel gather) -> NO NaN at NW=4 -> the cooperative GATHER is the source.
- SERIALIZED gather (warp0-only) -> NW=4 nt>=2 PASS clean ~1e-4 (build ds4) -> **M-split math is fully correct**;
  the cooperative warpid-partition gather is the defect.
THE WALL: re-applying the *same* serialized fix to main (trivial comment diff) -> NW=1 all-NaN + nt=2 NaN;
reverting -> NW=1 LONE NaN (was stable 0.00007 for hours). Logically-identical / no-op changes FLIP NaN.
ISA: ordering correct, regs fit (432<512), no race. => genuine COMPILER-SCHEDULING / uninit knife-edge that
surface fixes CANNOT stabilize (every patch flips another case).
DECISION: stop surface-patching this hand-rolled kernel. Rewrite NW=4 from gqa/kernel.cpp's PROVEN multi-warp
structure adapted to DSA dims, removing the two non-standard fragile pieces gqa lacks:
  (1) hand-rolled raw_buffer_load_lds gather -> HK `G::load` group cooperative load;
  (2) ps LDS-roundtrip -> QB=32 + reinterpret_cast<rt_16x32_4_s> (acc[DV,32]=256 AGPR; verify budget).
SOLID/REUSABLE: NW=1 algorithm correctness (vs torch 1e-4); coord convention (sink dim-2 row, o_reg dim-1
depth); ISA workflow (--save-temps + mini_pv reference); mini.cpp/mini_pv.cpp/mini_loop.cpp repros;
HOST_GATHER flag. NOTE: current hk_fwd.cpp is FRAGILE (NW=1 lone-NaN) — WIP; rebuild from the gqa structure.
