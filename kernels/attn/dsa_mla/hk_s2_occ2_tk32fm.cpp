// hk_s2_occ2_tk32fm.cpp — DSA-V4 Stage-2 fwd, OCC-2 independent 8-wave. STAGING-ONLY sub-tiling.
//
// Key vs occ2b: sub-tile TK_SUB is applied ONLY to the QK/PV mma OPERAND STAGING (K/V register chunks),
// NOT to the softmax. `s` and the online softmax stay on the FULL TILE_K tile → mask/col_max/exp/col_sum/
// mul_col(acc) run ONCE per gather-tile (no NSUB× rescale doubling that made occ2b slow), while the K/V
// register staging stays [TK_SUB, DC] (small → fits occ-2 VGPR).
//   QK: for rb in NSUB: mma_ABt(subtile<TK_SUB>(s, rb), k_c[TK_SUB,DC], q_arr[c])  -> fills full s[TILE_K,QB]
//   softmax ONCE on full s.
//   PV: for rb in NSUB: mma_AtB(acc_c, v_c[TK_SUB,DC], subtile<TK_SUB>(pop, rb))    -> contracts all keys
#include "kittens.cuh"
#include "sub_topk.cuh"
#include "sub_gather.cuh"
using namespace kittens;
#define NO_TKV_HOIST 1   // tkv read in mask (un-hoisted): occ-2 2nd wave hides its latency

#ifndef TILE_K
#define TILE_K 32
#endif
#ifndef TK_SUB
#define TK_SUB 32          // mma staging sub-tile (keys). softmax stays on full TILE_K.
#endif
#ifndef DC
#define DC 64
#endif
constexpr int QB = 16, D = 512;
constexpr int NSUB = TILE_K / TK_SUB;      // key-row staging blocks per mma phase
constexpr int NCH  = D / DC;               // D-chunks
constexpr int NB   = TILE_K / 16;          // 16-row tiles in the FULL s tile (mask/tkv granularity)
constexpr float LOG2E = 1.4426950408889634f, LN2 = 0.6931471805599453f;
#ifndef NW
#define NW 8
#endif
#define NT (kittens::WARP_THREADS * NW)
#ifndef NTILES
#define NTILES 4
#endif
#ifndef MINWAVES
#define MINWAVES 2
#endif

using QcT = rt<bf16,  QB,     DC, row_l, rt_16x32_s>;   // resident Q chunk (full D: q_arr[NCH])
using KcT = rt<bf16,  TK_SUB, DC, row_l, rt_16x32_s>;   // streamed K staging sub-chunk
using VcT = rt<bf16,  TK_SUB, DC, col_l, rt_32x16_s>;   // streamed V staging sub-chunk
using ST_ = rt<float, TILE_K, QB, col_l, rt_16x16_s>;   // FULL s tile
using PbT = rt<bf16,  TILE_K, QB, col_l, rt_16x16_s>;   // FULL P (bf16)
using PopT= rt<bf16,  TILE_K, QB, col_l, rt_32x16_s>;   // FULL P (mma operand layout)

// In-register rt_16x16 -> rt_32x16 (bf16, col_l) reshuffle via lane-swap net (NO LDS roundtrip).
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
using OT  = rt<float, D,      QB, col_l, rt_16x16_s>;   // acc [512,16] full head
using OtT = rt<float, QB,     D, row_l, rt_16x16_s>;
using KS  = st_bf<TILE_K, D, st_32x32_s>;

struct g_t {
    gl<bf16,-1,-1,-1,-1> Qg, KVg, Og;
    gl<int, -1,-1,-1,-1> Tkg;
    gl<float,-1,-1,-1,-1> Sg, Lg;
    int n_tiles, kv_rows;
    float scale;
    int has_sink;
};

__launch_bounds__(NT,MINWAVES)
__global__ void hk_s2_occ2_tk32fm(const g_t g){
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    KS (&ks)[2] = al.allocate<KS, 2>();
    __shared__ int topk_all[NTILES * TILE_K];

    const int warpid = kittens::warpid();
    const int lane = threadIdx.x % kittens::WARP_THREADS;
    const int tok = blockIdx.x;
    const int nt = g.n_tiles;

    QcT q_arr[NCH];              // Q resident (64 VGPR), full D as DC-chunks, invariant across tiles.
    #pragma unroll
    for (int c = 0; c < NCH; c++) load(q_arr[c], g.Qg, coord<>{0, warpid, 0, c});
    __builtin_amdgcn_s_waitcnt(0);

    OT acc;
    zero(acc);
    typename ST_::row_vec m_i, l_i, m_new, alpha;
    neg_infty(m_i);
    zero(l_i);

    #define SB() __builtin_amdgcn_sched_barrier(0)
    load_topk<NT>(topk_all, g.Tkg, tok, nt, TILE_K);
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    constexpr int NDMA = gather_ndma<NT, KS>();
    gather_kv_async<NT>(ks[0], g.KVg, topk_all, 0, g.kv_rows);
    SB();

    for (int j = 0; j < nt; j++) {
        const int cur = j & 1, nxt = (j + 1) & 1;
        if (j + 1 < nt) {
            gather_kv_async<NT>(ks[nxt], g.KVg, topk_all + (j+1)*TILE_K, 0, g.kv_rows);
            SB();
            asm volatile("s_waitcnt vmcnt(%0)" :: "i"(NDMA));
            SB();
        } else {
            __builtin_amdgcn_s_waitcnt(0);
            SB();
        }
        __builtin_amdgcn_s_barrier();
        SB();

        ST_ s;
        zero(s);
        SB();
#ifndef NO_TKV_HOIST
        // HOIST tkv read for the FULL tile's TILE_K keys (latency hides under QK; costs 16 VGPR live in QK).
        int tkv[NB*4];
        {
          const int grp = lane / 16;
          const int* tk = topk_all + j*TILE_K;
          #pragma unroll
          for (int m = 0; m < NB*4; m++) tkv[m] = tk[4*grp + (m & 3) + 16*(m >> 2)];
        }
        SB();
#endif

        // ---- QK: fill FULL s. K staging = single buffer under NO_KDBUF (frees 16 VGPR; overlap comes
        //      from the occ-2 2nd wave / staggering, not intra-warp double-buffer). ----
#if defined(DBUF) && !defined(NO_KDBUF)
        {
            constexpr int RPC_K = TK_SUB * DC / 512;
            constexpr int WKT = (RPC_K + 4) < 15 ? (RPC_K + 4) : 15;
            constexpr int NI = NSUB * NCH;
            KcT kb0, kb1;
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
#else
        // single-buffer, SINGLE flattened (rb,c) loop (mirror the DBUF path) so the compiler sees one
        // linear stream + SB() fences and keeps exactly ONE k_c live (nested double-#pragma-unroll let it
        // pipeline into many buffers -> spill).
        {
            constexpr int NI = NSUB * NCH;
            KcT k_c;                       // ONE buffer, hoisted OUT of the loop -> forces reuse (single buffer),
            #pragma unroll                 //   stops the compiler allocating a fresh k_c per unrolled iteration.
            for (int i = 0; i < NI; i++) {
                const int rb = i / NCH, c = i % NCH;
                { typename KS::template subtile<TK_SUB, DC> ksub(ks[cur], {rb, c}); load(k_c, ksub); }
                asm volatile("s_waitcnt lgkmcnt(0)");
                SB();
                auto& s_rb = subtile_inplace<TK_SUB>(s, rb);
                mma_ABt(s_rb, k_c, q_arr[c], s_rb);
                SB();
            }
        }
#endif

        // ---- softmax ONCE on the full s tile ----
        mul(s, s, g.scale);
        SB();
        {
#ifdef NO_TKV_HOIST
          // un-hoisted: read topk HERE (after QK) so tkv is NOT live during QK -> QK peak -16 VGPR.
          int tkv[NB*4];
          {
            const int grp = lane / 16;
            const int* tk = topk_all + j*TILE_K;
            #pragma unroll
            for (int m = 0; m < NB*4; m++) tkv[m] = tk[4*grp + (m & 3) + 16*(m >> 2)];
          }
          asm volatile("s_waitcnt lgkmcnt(0)");
#endif
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
        // FIXED-MAX fast path (flydsl): bound = tile-0 colmax, set once. alpha==1 for every tile
        // so the per-tile mul_col(acc) rescale is GONE. Exact by softmax shift-invariance; LSE uses
        // m_i (=bound) so it stays correct. Numerically safe: scaled scores are small (~+-2 log2).
        if (j == 0) col_max(m_i, s, m_i);   // m_i was -inf -> becomes tile-0 colmax (the fixed ref)
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

        // ---- PV: contract all keys. ONE double-buffer over flattened (rb,c) -> 2 V buffers total. ----
#ifdef DBUF
        {
            constexpr int RPC_V = TK_SUB * DC / 256;
            constexpr int WV = RPC_V < 15 ? RPC_V : 15;
            constexpr int NI = NSUB * NCH;
            VcT vb0, vb1;
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
#else
        // single-buffer, SINGLE flattened (rb,c) loop -> one v_c live.
        {
            constexpr int NI = NSUB * NCH;
            VcT v_c;                       // ONE buffer, hoisted OUT of the loop -> single buffer.
            #pragma unroll
            for (int i = 0; i < NI; i++) {
                const int rb = i / NCH, c = i % NCH;
                { typename KS::template subtile<TK_SUB, DC> vsub(ks[cur], {rb, c}); load(v_c, vsub); }
                asm volatile("s_waitcnt lgkmcnt(0)");
                SB();
                auto& pop_rb = subtile_inplace<TK_SUB>(pop, rb);
                auto& acc_c  = subtile_inplace<DC>(acc, c);
                mma_AtB(acc_c, v_c, pop_rb, acc_c);
                SB();
            }
        }
#endif
        SB();
        __syncthreads();
        SB();
    }

    typename ST_::row_vec l_tot, afix, m_fin;
    if (g.has_sink) {
        typename ST_::row_vec sink;
        load(sink, g.Sg, coord<>{0,0,warpid,0});
        __builtin_amdgcn_s_waitcnt(0);
        mul(sink, sink, LOG2E);
        max(m_fin, m_i, sink);
        sub(afix, m_i, m_fin);
        exp2(afix, afix);
        sub(sink, sink, m_fin);
        exp2(sink, sink);
        mul(l_tot, l_i, afix);
        add(l_tot, l_tot, sink);
    } else {
        copy(m_fin, m_i);
        ones(afix);
        copy(l_tot, l_i);
    }
    mul_col(acc, acc, afix);
    div_col(acc, acc, l_tot);
    OtT acc_t;
    transpose(acc_t, acc);
    store(g.Og, acc_t, coord<>{0, warpid, 0, 0});
    typename ST_::row_vec lse;
    log(lse, l_tot);
    mul(m_fin, m_fin, LN2);
    add(lse, lse, m_fin);
    store(g.Lg, lse, coord<>{0,0,warpid,0});
}
