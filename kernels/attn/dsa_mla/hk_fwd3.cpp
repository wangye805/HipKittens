// hk_fwd3.cpp — DSA V4 sparse-MLA forward, HipKittens gfx950.
// gqa keys-as-rows DISCIPLINE + gluon STRUCTURE, assembled from validated pieces:
//   QK  = two mma_ABt (natural K,Q row_l)  -> S=[TILE_K,QB] keys-as-rows   (mini3b)
//   V   = the same gathered K shared tile re-loaded col_l                  (mini3b)
//   softmax = col_max/sub_col/exp2/col_sum, running m_i/l_i/acc-rescale    (mini3_loop)
//   P->PV operand via per-warp LDS roundtrip rt_16x16->rt_32x16            (mini3)
//   gather = topk-indirected raw_buffer_load_lds into shared [TILE_K,D]    (hk_fwd, +col_off)
//   M-split NW warps x QB heads; per-warp Q/O in depth coord              (mini3)
//   sink fold epilogue (V4)                                                (hk_fwd)
// Single-buffer (correctness first); double-buffer pipeline is the next step.
#include "kittens.cuh"
using namespace kittens;

constexpr int QB = 16, TILE_K = 32, D_V = 512, D_ROPE = 64, D_QK = D_V + D_ROPE;
constexpr float LOG2E = 1.4426950408889634f;
#ifndef NW
#define NW 4
#endif
#define NT (kittens::WARP_THREADS * NW)

// Gather KV[topk[k], col_off + d] -> dst[k, d], iterating LOGICAL (k,d) and using the SAME
// forward swizzle (+subtile offset) that `load(rt, st)` consumes. Plain global-load + LDS-store
// (correctness-first; vectorize/async later). Caller fences with s_waitcnt + __syncthreads.
template<int N_THREADS, ducks::st::all ST, ducks::gl::all GL>
__device__ inline void gather_to_shared(ST& dst, const GL& kv, const int* topk,
                                        int col_off, int t_kv) {
    using Tp = typename ST::dtype;
    constexpr int SUBR = ST::underlying_subtile_rows;
    constexpr int SUBC = ST::underlying_subtile_cols;
    constexpr int SPR  = ST::underlying_subtiles_per_row;
    constexpr int SUBB = ST::underlying_subtile_bytes;
    constexpr int rows = ST::rows, cols = ST::cols, total = rows * cols;
    const int row_stride = kv.template stride<2>();          // D_QK
    Tp* gbase = (Tp*)&kv[coord<>{0, 0, 0, 0}];
    char* lds = (char*)&dst.data[0];
    const int t = threadIdx.x;
#ifdef VEC_GATHER
    // Vectorized 16-byte (8 bf16) gather: the st swizzle leaves the low 16B contiguous, and KV rows
    // are contiguous in d, so each (k, d8) is one coalesced int4 load + one int4 LDS store.
    // VALIDATED CORRECT in isolation (gcheck) and ~2.2x faster (20->9ms), BUT it perturbs the
    // single-buffer kernel's codegen knife-edge -> nondeterministic NaN. Enable only once the kernel
    // is stabilized (subtiled mma + explicit scheduling / double-buffer pipeline).
    constexpr int VW = 16 / sizeof(Tp);                     // 8 for bf16
    static_assert(cols % VW == 0, "cols must be divisible by 16B run");
    constexpr int nvec = total / VW;
    for (int e = t; e < nvec; e += N_THREADS) {
        const int idx = e * VW;
        const int k = idx / cols, d = idx % cols;           // d is VW-aligned
        int pr = topk[k]; if (pr < 0) pr = 0;               // safe gather (-1 padding)
        const int sub_id = (k / SUBR) * SPR + (d / SUBC);
        const uint32_t off = sub_id * SUBB + dst.swizzle({k % SUBR, d % SUBC});
        *(int4*)(lds + off) = *(const int4*)(gbase + (size_t)pr * row_stride + col_off + d);
    }
#else
    for (int e = t; e < total; e += N_THREADS) {            // scalar gather (stable, correctness-first)
        const int k = e / cols, d = e % cols;
        int pr = topk[k]; if (pr < 0) pr = 0;
        Tp val = gbase[(size_t)pr * row_stride + col_off + d];
        const int sub_id = (k / SUBR) * SPR + (d / SUBC);
        const uint32_t off = sub_id * SUBB + dst.swizzle({k % SUBR, d % SUBC});
        *(Tp*)(lds + off) = val;
    }
#endif
}

using QlT = rt<bf16,  QB,     D_V,    row_l, rt_16x32_s>;
using QrT = rt<bf16,  QB,     D_ROPE, row_l, rt_16x32_s>;
using KlT = rt<bf16,  TILE_K, D_V,    row_l, rt_16x32_s>;   // K_lora natural (QK A)
using KrT = rt<bf16,  TILE_K, D_ROPE, row_l, rt_16x32_s>;   // K_rope natural (QK A)
using VT  = rt<bf16,  TILE_K, D_V,    col_l, rt_32x16_s>;   // V = K_lora shared, col_l (PV A)
using ST  = rt<float, TILE_K, QB,     col_l, rt_16x16_s>;
using PbT = rt<bf16,  TILE_K, QB,     col_l, rt_16x16_s>;
using PopT= rt<bf16,  TILE_K, QB,     col_l, rt_32x16_s>;
using OT  = rt<float, D_V,    QB,     col_l, rt_16x16_s>;
using OtT = rt<float, QB,     D_V,    row_l, rt_16x16_s>;
using KlS = st_bf<TILE_K, D_V,    st_32x32_s>;
using KrS = st_bf<TILE_K, D_ROPE, st_32x32_s>;
using MtS = st_bf<TILE_K, QB, st_16x16_s>;        // per-key invalid mask, broadcast over QB (bf16)
using MbT = rt<bf16, TILE_K, QB, col_l, rt_16x16_s>;  // mask tile in registers (for row_max -> col_vec)

// Fill mask tile mt[k,h] = (topk[k] < 0 ? -1e30 : 0)  (forward swizzle, like the gather).
template<int N_THREADS>
__device__ inline void fill_mask(MtS& dst, const int* topk) {
    constexpr int SUBR = MtS::underlying_subtile_rows, SUBC = MtS::underlying_subtile_cols;
    constexpr int SPR = MtS::underlying_subtiles_per_row, SUBB = MtS::underlying_subtile_bytes;
    constexpr int total = TILE_K * QB;
    char* lds = (char*)&dst.data[0];
    for (int e = threadIdx.x; e < total; e += N_THREADS) {
        const int k = e / QB, h = e % QB;
        bf16 val = (bf16)((topk[k] < 0) ? -1e30f : 0.0f);
        const int sub_id = (k / SUBR) * SPR + (h / SUBC);
        const uint32_t off = sub_id * SUBB + dst.swizzle({k % SUBR, h % SUBC});
        *(bf16*)(lds + off) = val;
    }
}

struct g_t {
    gl<bf16,-1,-1,-1,-1> Qlg, Qrg, KVg, Og;
    gl<int, -1,-1,-1,-1> Tkg;     // [.,.,NTILES,TILE_K] topk
    gl<float,-1,-1,-1,-1> Sg;     // [.,.,NW,QB] sink
    gl<float,-1,-1,-1,-1> Dg;     // [.,NW,QB,TILE_K] DEBUG: tile-0 S (post-QK)
    int n_tiles, t_kv;
    float scale;
    int has_sink;
};

__launch_bounds__(NT,1)
__global__ void hk_fwd3(const g_t g){
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    KlS &ks = al.allocate<KlS>();
    KrS &krs = al.allocate<KrS>();
#ifdef ENABLE_MASK
    MtS &mts = al.allocate<MtS>();
#endif
    auto (&ps)[NW] = al.allocate<st_bf<TILE_K, QB, st_32x16_s>, NW>();
    __shared__ int topk[TILE_K];

    const int tid = threadIdx.x;
    const int warpid = kittens::warpid();
    // Full grid: blockIdx.x = token, blockIdx.y = head-group; gridDim.y = #head-groups.
    // Q/O laid out [T, Hgroups, NW, QB, D] flattened -> dim0 = token*HG + hg, dim1 = warpid.
    const int HG = gridDim.y;
    const int qo_row = blockIdx.x * HG + blockIdx.y;     // token*HG + hg
    const int sink_row = blockIdx.y * NW + warpid;        // head-block within [H]
    const int tok = blockIdx.x;

    QlT q_l; load(q_l, g.Qlg, coord<>{qo_row,warpid,0,0});
    QrT q_r; load(q_r, g.Qrg, coord<>{qo_row,warpid,0,0});
    __builtin_amdgcn_s_waitcnt(0);

    OT acc; zero(acc);
    typename ST::row_vec m_i, l_i, m_new, alpha;
    neg_infty(m_i); zero(l_i);

    for(int j=0; j<g.n_tiles; j++){
        if (tid < TILE_K) topk[tid] = g.Tkg[coord<>{tok,0,j,tid}];
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);
        gather_to_shared<NT>(ks,  g.KVg, topk, 0,   g.t_kv);
        gather_to_shared<NT>(krs, g.KVg, topk, D_V, g.t_kv);
#ifdef ENABLE_MASK
        fill_mask<NT>(mts, topk);
#endif
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_sched_barrier(0);
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);

        ST s; zero(s);
        { KlT k_l; load(k_l, ks);
          KrT k_r; load(k_r, krs);
          __builtin_amdgcn_s_waitcnt(0);
          __builtin_amdgcn_sched_barrier(0);
          mma_ABt(s, k_l, q_l, s);
          mma_ABt(s, k_r, q_r, s);
        }   // k_l/k_r dead -> regs freed before v_l[32,512] is loaded for PV
        mul(s, s, g.scale);
#ifdef ENABLE_MASK
        // -1 invalid-key mask as a per-key col_vec (LOW footprint: no float tile, dodges the NW=4
        // register knife-edge that the float-tile mask tripped). Correct + deterministic at NW=4.
        // row_max reduces the broadcast mask tile (each row constant) -> per-key col_vec.
        { MbT mtb; load(mtb, mts); __builtin_amdgcn_s_waitcnt(0);
          typename MbT::col_vec cvb; row_max(cvb, mtb);
          typename ST::col_vec cv; copy(cv, cvb);
          add_row(s, s, cv); }   // invalid keys -> -1e30 -> exp2 0 -> excluded from softmax/PV
#endif
        __builtin_amdgcn_sched_barrier(0);
#ifdef DBGS
        if (j == 0) { rt<float,QB,TILE_K,row_l,rt_16x16_s> st_; transpose(st_, s);
                      store(g.Dg, st_, coord<>{qo_row,warpid,0,0}); }
#endif

        col_max(m_new, s, m_i);
        sub(alpha, m_i, m_new); exp2(alpha, alpha);
        sub_col(s, s, m_new); exp2(s, s);
        mul(l_i, l_i, alpha); col_sum(l_i, s, l_i);
        mul_col(acc, acc, alpha);

        PbT pb; copy(pb, s);
        store(ps[warpid], pb);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        PopT pop; load(pop, ps[warpid]);
        __builtin_amdgcn_s_waitcnt(0);

        VT v_l; load(v_l, ks);   // load V late (k_l dead) to cut peak VGPR
        __builtin_amdgcn_s_waitcnt(0);
        mma_AtB(acc, v_l, pop, acc);
        copy(m_i, m_new);
        __syncthreads();
    }

    // ---- epilogue: sink fold + normalize ----
    if (g.has_sink) {
        typename ST::row_vec sink, m_fin, afix, l_tot;
        load(sink, g.Sg, coord<>{0,0,sink_row,0});
        max(m_fin, m_i, sink);
        sub(afix, m_i, m_fin); exp2(afix, afix);
        sub(sink, sink, m_fin); exp2(sink, sink);
        mul(l_tot, l_i, afix); add(l_tot, l_tot, sink);
        mul_col(acc, acc, afix);
        div_col(acc, acc, l_tot);
    } else {
        div_col(acc, acc, l_i);
    }

    OtT acc_t; transpose(acc_t, acc);
    store(g.Og, acc_t, coord<>{qo_row,warpid,0,0});
}
