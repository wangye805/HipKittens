// dsa_mla_fwd.cpp — ASYMMETRIC barriers: WG0 stops ONLY before PV, WG1 ONLY after PV.
// 1 barrier/tile per warp (vs 2); pairs by arrival-count, keeps the half-tile stagger.
// prev: — 2-CLUSTER {gather+QK+softmax, PV}: softmax overlaps the other group PV.
// 2 barriers/tile == baseline. Derived from 3-cluster.
// prev-3-CLUSTER staggered ping-pong {gather+QK, softmax, PV}.
// Folds the cheap gather into QK (de-bursts DMA behind QK compute) -> 3 barriers/tile (was 4).
// Original 4-phase note:
// STEP 4: STAGGERED ping-pong. Two warp groups run the same 4-phase body
// {A:gather t+2 half, B:QK, C:softmax, D:PV} offset by ONE phase via a prologue skew (WG1 does an
// extra s_barrier; WG0 does a matching epilogue barrier so total counts match → no deadlock).
// Staggered s_barrier pairs "WG0 phase p" with "WG1 phase p-1" (MXFP8-GEMM mechanism), so WG0's
// gather-DMA overlaps WG1's compute and the single-VALU softmax of one group overlaps the other's
// QK/PV. 4-buffer ring (one-phase skew needs the extra buffer to avoid t+2 vs t-1 collision).
// Correctness needs only: matched barrier count (no deadlock) + ring-depth race safety. Each warp
// computes its own heads independently, so the time-shift does not change the result.
// Derived from hk_s2_occ2_ph.cpp.
#include "kittens.cuh"
#ifndef DSA_NO_PYBIND
#include "pyutils/pyutils.cuh"
#endif
#include "sub_topk.cuh"
#include "sub_gather.cuh"
using namespace kittens;
#define NO_TKV_HOIST 1

#ifndef TILE_K
#define TILE_K 32
#endif
#ifndef TK_SUB
#define TK_SUB 32
#endif
#ifndef DC
#define DC 64
#endif
constexpr int QB = 16, D = 512;
constexpr int NSUB = TILE_K / TK_SUB;
constexpr int NCH  = D / DC;
constexpr int NB   = TILE_K / 16;
constexpr float LOG2E = 1.4426950408889634f, LN2 = 0.6931471805599453f;
#ifndef NW
#define NW 8
#endif
#define NT (kittens::WARP_THREADS * NW)
#ifndef NTILES
#define NTILES 36          // topk / TILE_K  (e.g. topk=1152 -> 36).  Set via -DNTILES=<topk/32>.
#endif
#ifndef HAS_SINK
#define HAS_SINK 1
#endif
// base-2 softmax scale = (1/sqrt(D)) * log2(e)   (D=512)
constexpr float SCALE = 1.4426950408889634f / 22.627416997969522f;
#ifndef MINWAVES
#define MINWAVES 2
#endif
#define RING 4
#define TK_RING 8   // topk ring: deeper than KV ring (covers prefetch-3 + stagger lag), power-of-2 -> &7

using QcT = rt<bf16,  QB,     DC, row_l, rt_16x32_s>;
using KcT = rt<bf16,  TK_SUB, DC, row_l, rt_16x32_s>;
using VcT = rt<bf16,  TK_SUB, DC, col_l, rt_32x16_s>;
using ST_ = rt<float, TILE_K, QB, col_l, rt_16x16_s>;
using PbT = rt<bf16,  TILE_K, QB, col_l, rt_16x16_s>;
using PopT= rt<bf16,  TILE_K, QB, col_l, rt_32x16_s>;

__device__ static inline void p16_to_p32(PopT &dst, const PbT &src){
    typedef uint32_t u2 __attribute__((ext_vector_type(2)));
    #pragma unroll
    for(int i=0;i<PopT::height;i++)
        #pragma unroll
        for(int k=0;k<2;k++){
            uint32_t A=*reinterpret_cast<const uint32_t*>(&src.tiles[2*i][0].data[k]);
            uint32_t B=*reinterpret_cast<const uint32_t*>(&src.tiles[2*i+1][0].data[k]);
            u2 t=__builtin_amdgcn_permlane32_swap(A,B,false,true);
            u2 r=__builtin_amdgcn_permlane16_swap(t.x,t.y,false,true);
            *reinterpret_cast<uint32_t*>(&dst.tiles[i][0].data[k])   = r.x;
            *reinterpret_cast<uint32_t*>(&dst.tiles[i][0].data[k+2]) = r.y;
        }
}
using OT  = rt<float, D,      QB, col_l, rt_16x16_s>;
using OtT = rt<float, QB,     D, row_l, rt_16x16_s>;
using KS  = st_bf<TILE_K, D, st_32x32_s>;

struct globals {
    gl<bf16,-1,-1,-1,-1> Qg;    // [S, NW, QB, D]   query, per-token
    gl<bf16,-1,-1,-1,-1> KVg;   // [1, 1, T_KV, D]  dense KV cache (K==V), shared across tokens
    gl<int, -1,-1,-1,-1> Tkg;   // [S, 1, NTILES, TILE_K]  per-token topk row indices
    gl<float,-1,-1,-1,-1> Sg;   // [1, 1, NW, QB]   per-head attention sink, shared across tokens
    gl<bf16,-1,-1,-1,-1> Og;    // [S, NW, QB, D]   output, per-token
    gl<float,-1,-1,-1,-1> Lg;   // [S, 1, NW, QB]   LSE, per-token per-head
    dim3 grid()  { return dim3(Qg.batch()); }          // one CTA per query token
    dim3 block() { return dim3(NT); }
    size_t dynamic_shared_memory() { return RING * sizeof(KS); }
};

#define GATHER_HALF(buf, ti) gather_kv_async<NT>((buf), g.KVg, topk_all + (((ti)%TK_RING))*TILE_K, 0, (int)g.KVg.rows())
#define SB() __builtin_amdgcn_sched_barrier(0)
#define BAR() do { __builtin_amdgcn_s_barrier(); SB(); } while(0)

__launch_bounds__(NT,MINWAVES)
__global__ void dsa_mla_fwd(const globals g){
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    KS (&ks)[RING] = al.allocate<KS, RING>();
    __shared__ int topk_all[TK_RING * TILE_K];   // topk ring (TK_RING slots, decoupled from KV RING)

    const int warpid = kittens::warpid();
    const int wg = warpid / 4;                 // 0 = leads, 1 = lags one phase
    const int lane = threadIdx.x % kittens::WARP_THREADS;
    const int tok = blockIdx.x;
    constexpr int nt = NTILES;

    // (1) topk DMA FIRST — needed by the KV gather (before Q is even used)
    load_topk_tile_dma<NT,TILE_K,TK_RING>(topk_all, g.Tkg, tok, 0);
    if (nt > 1) {
        load_topk_tile_dma<NT,TILE_K,TK_RING>(topk_all, g.Tkg, tok, 1);
    }
    if (nt > 2) {
        load_topk_tile_dma<NT,TILE_K,TK_RING>(topk_all, g.Tkg, tok, 2);
    }

    OT acc;
    typename ST_::row_vec m_i, l_i, m_new, alpha;
    constexpr int NDMA = gather_ndma<NT, KS>();

    // (2/3) drain topk (needed before the gather reads topk from LDS), then barrier for cross-warp vis.
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    // (4) KV prime tiles 0,1 -- ISSUE the gather FIRST so its cold HBM latency has work to hide behind.
    if (wg == 0) GATHER_HALF(ks[0], 0);
    if (wg == 1) GATHER_HALF(ks[0], 0);
    if (nt > 1) {
        if (wg == 0) GATHER_HALF(ks[1], 1);
        if (wg == 1) GATHER_HALF(ks[1], 1);
    }
    // (S2) init (VALU) now overlaps the KV gather HBM latency (longer than the topk DMA it used to hide).
    zero(acc);
    neg_infty(m_i);
    zero(l_i);
    asm volatile("s_waitcnt vmcnt(%0)" :: "i"(NDMA));   // drain ONLY tile0's half; leave tile1 in flight
    __builtin_amdgcn_s_barrier();                       // publish tile0 for WG0's barrier-free first QK

    // (2/5) Q load LAST — issued after both drains so it can't be force-drained early; overlaps the
    //       skew + iter-0 gather, and the compiler drains it PER-GROUP right before each group's QK.
    QcT q_arr[NCH];
    #pragma unroll
    for (int c = 0; c < NCH; c++) {
        load(q_arr[c], g.Qg, coord<>{tok, warpid, 0, c});
    }

    // ---- SKEW: WG1 one phase behind. ----
    if (wg == 1) {
        __builtin_amdgcn_s_barrier();
    }

    for (int j = 0; j < nt; j++) {
        const int cur = j % RING;

        // ===== Phase A: gather tile j+2 (this group's half) + drain tile j =====
        // topk prefetch 1 iter ahead of the KV gather (async DMA -> drained with KV, no forced vmcnt(0))
        if (j + 3 < nt) {
            load_topk_tile_dma<NT,TILE_K,TK_RING>(topk_all, g.Tkg, tok, j + 3);
        }
        if (j + 2 < nt) {
            const int nxt = (j + 2) % RING;
            if (wg == 0) GATHER_HALF(ks[nxt], j + 2);
            if (wg == 1) GATHER_HALF(ks[nxt], j + 2);
            SB();
            asm volatile("s_waitcnt vmcnt(%0)" :: "i"(2 * NDMA));
            SB();
        } else {
            __builtin_amdgcn_s_waitcnt(0);
            SB();
        }
        // (no A-end barrier: gather folds into QK) 

        // ===== Cluster 1: gather+QK → full s =====
        ST_ s;
        zero(s);
        SB();
        {
            constexpr int RPC_K = TK_SUB * DC / 512;
            constexpr int WKT = (RPC_K + 4) < 15 ? (RPC_K + 4) : 15;
            constexpr int NI = NSUB * NCH;
            KcT kb0, kb1;                       // DOUBLE BUFFER: prefetch next chunk's ds_read under mma
            { typename KS::template subtile<TK_SUB, DC> s0(ks[cur], {0, 0}); load(kb0, s0); }
            #pragma unroll
            for (int i = 0; i < NI; i++) {
                const int rb = i / NCH, c = i % NCH;
                if (i + 1 < NI) {
                    const int nrb = (i + 1) / NCH, nc = (i + 1) % NCH;
                    typename KS::template subtile<TK_SUB, DC> sn(ks[cur], {nrb, nc});
                    load((i & 1) ? kb0 : kb1, sn);
                    asm volatile("s_waitcnt lgkmcnt(%0)" :: "i"(WKT));
                } else {
                    asm volatile("s_waitcnt lgkmcnt(0)");
                }
                SB();
                auto& s_rb = subtile_inplace<TK_SUB>(s, rb);
                mma_ABt(s_rb, (i & 1) ? kb1 : kb0, q_arr[c], s_rb);
                SB();
            }
        }
        // (no B-end: softmax folds into the gather+QK cluster)

        // ===== Phase C: softmax → pop =====
        mul(s, s, SCALE);
        SB();
        {
          int tkv[NB*4];
          {
            const int grp = lane / 16;
            const int* tk = topk_all + (j%TK_RING)*TILE_K;
            #pragma unroll
            for (int m = 0; m < NB*4; m++) tkv[m] = tk[4*grp + (m & 3) + 16*(m >> 2)];
          }
          asm volatile("s_waitcnt lgkmcnt(0)");
          float bias[NB*4];
          #pragma unroll
          for (int m = 0; m < NB*4; m++)
            bias[m] = __int_as_float(tkv[m] & (int)0xFF800000);
          #pragma unroll
          for (int bt = 0; bt < NB; bt++)
            #pragma unroll
            for (int p = 0; p < 2; p++) {
              s.tiles[bt][0].data[p].x += bias[bt*4 + 2*p    ];
              s.tiles[bt][0].data[p].y += bias[bt*4 + 2*p + 1];
            }
          SB();
        }
        // FIXED-MAX fast path (flydsl): bound = tile-0 colmax; alpha==1 -> no per-tile acc rescale.
        if (j == 0) col_max(m_i, s, m_i);
        SB();
        sub_col(s, s, m_i);
        exp2(s, s);
        SB();
        col_sum(l_i, s, l_i);
        SB();
        PbT pb;
        copy(pb, s);
        SB();
        PopT pop;
        p16_to_p32(pop, pb);
        SB();
        if (wg == 0) __builtin_amdgcn_s_barrier(); SB();   // Bc: WG0 only, before PV

        // ===== Phase D: PV =====
        {
            constexpr int RPC_V = TK_SUB * DC / 256;
            constexpr int WV = RPC_V < 15 ? RPC_V : 15;
            constexpr int NI = NSUB * NCH;
            VcT vb0, vb1;                       // DOUBLE BUFFER
            { typename KS::template subtile<TK_SUB, DC> v0(ks[cur], {0, 0}); load(vb0, v0); }
            #pragma unroll
            for (int i = 0; i < NI; i++) {
                const int rb = i / NCH, c = i % NCH;
                if (i + 1 < NI) {
                    const int nrb = (i + 1) / NCH, nc = (i + 1) % NCH;
                    typename KS::template subtile<TK_SUB, DC> vn(ks[cur], {nrb, nc});
                    load((i & 1) ? vb0 : vb1, vn);
                    asm volatile("s_waitcnt lgkmcnt(%0)" :: "i"(WV));
                } else {
                    asm volatile("s_waitcnt lgkmcnt(0)");
                }
                SB();
                auto& pop_rb = subtile_inplace<TK_SUB>(pop, rb);
                auto& acc_c  = subtile_inplace<DC>(acc, c);
                mma_AtB(acc_c, (i & 1) ? vb1 : vb0, pop_rb, acc_c);
                SB();
            }
        }
        SB();
        if (wg == 1) __builtin_amdgcn_s_barrier(); SB();   // Bd: WG1 only, after PV
    }

    // (E1b) issue sink load right before the skew-balance barrier so the barrier wait hides its HBM latency
    typename ST_::row_vec sink_h;
    if (HAS_SINK) load(sink_h, g.Sg, coord<>{0,0,warpid,0});
    // ---- epilogue skew balance: WG0 does the extra barrier WG1 did in the prologue ----
    if (wg == 0) __builtin_amdgcn_s_barrier();

    typename ST_::row_vec l_tot, afix, m_fin;
    if (HAS_SINK) {
        __builtin_amdgcn_s_waitcnt(0);          // (E1) sink_h issued at kernel top; long since landed
        mul(sink_h, sink_h, LOG2E);
        max(m_fin, m_i, sink_h);
        sub(afix, m_i, m_fin);
        exp2(afix, afix);
        sub(sink_h, sink_h, m_fin);
        exp2(sink_h, sink_h);
        mul(l_tot, l_i, afix);
        add(l_tot, l_tot, sink_h);
    } else {
        copy(m_fin, m_i);
        ones(afix);
        copy(l_tot, l_i);
    }
    // (epilogue fold, Kyle) combine the sink-rescale (afix) and 1/l_tot into ONE row_vec scale,
    //   then a SINGLE full-acc pass. Drops the separate div_col full-acc pass.
    typename ST_::row_vec scal;
    div(scal, afix, l_tot);        // scal = afix / l_tot   (16-elem row_vec — cheap)
    mul_col(acc, acc, scal);       // ONE pass over acc[512,16]
    OtT acc_t;
    transpose(acc_t, acc);
    store(g.Og, acc_t, coord<>{tok, warpid, 0, 0});
    typename ST_::row_vec lse;
    log(lse, l_tot);
    mul(m_fin, m_fin, LN2);
    add(lse, lse, m_fin);
    store(g.Lg, lse, coord<>{tok,0,warpid,0});
}


// ---- host launcher + python binding (mirrors kernels/attn/gqa) ----
static void dispatch(globals g) {
    size_t shmem = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)dsa_mla_fwd, hipFuncAttributeMaxDynamicSharedMemorySize, shmem);
    dsa_mla_fwd<<<g.grid(), g.block(), shmem>>>(g);
}

#ifndef DSA_NO_PYBIND
PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "DSA-V4 sparse-MLA forward (occ-2, gfx950) — tk_kernel";
    kittens::py::bind_function<dispatch, globals>(m, "dispatch",
        &globals::Qg, &globals::KVg, &globals::Tkg, &globals::Sg, &globals::Og, &globals::Lg);
}
#endif
