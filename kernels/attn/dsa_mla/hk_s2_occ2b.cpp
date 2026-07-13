// hk_s2_occ2b.cpp — DSA-V4 Stage-2 fwd, dense-512, OCC-2 via INDEPENDENT 8-wave + TWO-LEVEL TILING.
//
// Design (OCC2_DESIGN_V2_INDEPENDENT.md): each of NW=8 warps does a WHOLE head, exactly like occ-1 — NO
// D-split, NO cross-wave S-combine. 8 warps/CTA = 2 waves/SIMD = occ-2. The 2 waves free-run and hide each
// other's mma-feed stalls; zero cross-wave sync added (only the shared per-token ks gather barrier).
//
// TWO-LEVEL TILING (the VGPR-fit key, from the "chunked GEMM" correction):
//   - OUTER (gather tile): gather TILE_K keys HBM->LDS (ks), kept LARGE (64) for gather amortization.
//   - INNER (compute sub-tile): loop TK_SUB keys at a time LDS->reg->mma->online-softmax->acc.
//   The staged register operands are [TK_SUB, DC] and s/P are [TK_SUB, QB] -> VGPR set by TK_SUB, NOT TILE_K.
//   acc[512,16]=128 VGPR (full head, over D_V, independent of TILE_K/TK_SUB) stays VGPR.
//   => footprint ~acc(128)+Q(64)+small(TK_SUB) -> occ-2 at a large gather TILE_K.
//
// Online softmax runs per SUB-tile (flash granularity = TK_SUB); m_i/l_i/acc persist across sub AND gather loops.
#include "kittens.cuh"
#include "sub_topk.cuh"
#include "sub_gather.cuh"
using namespace kittens;

#ifndef TILE_K
#define TILE_K 64          // gather tile (LDS rows)
#endif
#ifndef TK_SUB
#define TK_SUB 32          // compute sub-tile (register/mma). Must be a multiple of 32 (rt_32x16 P reshuffle).
#endif
#ifndef DC
#define DC 64
#endif
constexpr int QB = 16, D = 512;
constexpr int NSUB = TILE_K / TK_SUB;      // compute sub-tiles per gather tile
constexpr int NCH  = D / DC;               // D-chunks
constexpr int NB   = TK_SUB / 16;          // 16-row tiles in the s sub-tile (mask/tkv granularity)
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
using KcT = rt<bf16,  TK_SUB, DC, row_l, rt_16x32_s>;   // streamed K sub-chunk
using VcT = rt<bf16,  TK_SUB, DC, col_l, rt_32x16_s>;   // streamed V sub-chunk
using ST_ = rt<float, TK_SUB, QB, col_l, rt_16x16_s>;
using PbT = rt<bf16,  TK_SUB, QB, col_l, rt_16x16_s>;
using PopT= rt<bf16,  TK_SUB, QB, col_l, rt_32x16_s>;

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
using OT  = rt<float, D,      QB, col_l, rt_16x16_s>;   // acc [512,16] full head — VGPR-resident
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
__global__ void hk_s2_occ2b(const g_t g){
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    KS (&ks)[2] = al.allocate<KS, 2>();
    __shared__ int topk_all[NTILES * TILE_K];

    const int tid = threadIdx.x;
    const int warpid = kittens::warpid();   // 0..NW-1: this warp's head
    const int lane = tid % kittens::WARP_THREADS;
    const int tok = blockIdx.x;
    const int nt = g.n_tiles;

    // Q resident, full D as DC-chunks (invariant across tiles)
    QcT q_arr[NCH];
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
        __builtin_amdgcn_s_barrier();   // gather cross-warp: ks[cur] visible before QK
        SB();

        // ---- INNER: compute sub-tiles of TK_SUB keys over the gathered TILE_K tile ----
        // NOT unrolled: unrolling NSUB duplicates the whole QK+softmax+PV body -> register blow-up/spill.
        for (int sb = 0; sb < NSUB; sb++) {
            ST_ s;
            zero(s);
            SB();
            // HOIST tkv read for THIS sub-tile's TK_SUB keys (topk_all + j*TILE_K + sb*TK_SUB).
            int tkv[NB*4];
            {
              const int grp = lane / 16;
              const int* tk = topk_all + j*TILE_K + sb*TK_SUB;
              #pragma unroll
              for (int m = 0; m < NB*4; m++) tkv[m] = tk[4*grp + (m & 3) + 16*(m >> 2)];
            }
            SB();
#ifdef DBUF
            {
              constexpr int RPC_K = TK_SUB * DC / 512;
              constexpr int WK = RPC_K < 15 ? RPC_K : 15;
              constexpr int WKT = (WK + 4) < 15 ? (WK + 4) : 15;
              KcT kb0, kb1;
              { typename KS::template subtile<TK_SUB, DC> s0(ks[cur], {sb, 0}); load(kb0, s0); }
              #pragma unroll
              for (int c = 0; c < NCH; c++) {
                  if (c + 1 < NCH) {
                      typename KS::template subtile<TK_SUB, DC> sn(ks[cur], {sb, c + 1});
                      load((c & 1) ? kb0 : kb1, sn);
                      asm volatile("s_waitcnt lgkmcnt(%0)" :: "i"(WKT));
                  } else {
                      asm volatile("s_waitcnt lgkmcnt(0)");
                  }
                  SB();
                  mma_ABt(s, (c & 1) ? kb1 : kb0, q_arr[c], s);
                  SB();
              }
            }
#else
            #pragma unroll
            for (int c = 0; c < NCH; c++) {
                KcT k_c;
                typename KS::template subtile<TK_SUB, DC> ksub(ks[cur], {sb, c});
                load(k_c, ksub);
                asm volatile("s_waitcnt lgkmcnt(0)");
                SB();
                mma_ABt(s, k_c, q_arr[c], s);
                SB();
            }
#endif
            mul(s, s, g.scale);
            SB();
            // ---- additive -inf mask (topk & 0xFF800000) for this sub-tile's keys ----
            {
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

            col_max(m_new, s, m_i);
            SB();
            sub(alpha, m_i, m_new);
            exp2(alpha, alpha);
            SB();
            sub_col(s, s, m_new);
            exp2(s, s);
            SB();
            mul(l_i, l_i, alpha);
            col_sum(l_i, s, l_i);
            SB();
            mul_col(acc, acc, alpha);       // online rescale (per sub-tile) — in-place VGPR
            SB();

            PbT pb;
            copy(pb, s);
            SB();
            PopT pop;
            p16_to_p32(pop, pb);
            SB();

            // ---- PV: stream D_V-chunks; contraction over this sub-tile's TK_SUB keys ----
#ifdef DBUF
            {
              constexpr int RPC_V = TK_SUB * DC / 256;
              constexpr int WV = RPC_V < 15 ? RPC_V : 15;
              VcT vb0, vb1;
              { typename KS::template subtile<TK_SUB, DC> v0(ks[cur], {sb, 0}); load(vb0, v0); }
              #pragma unroll
              for (int c = 0; c < NCH; c++) {
                  if (c + 1 < NCH) {
                      typename KS::template subtile<TK_SUB, DC> vn(ks[cur], {sb, c + 1});
                      load((c & 1) ? vb0 : vb1, vn);
                      asm volatile("s_waitcnt lgkmcnt(%0)" :: "i"(WV));
                  } else {
                      asm volatile("s_waitcnt lgkmcnt(0)");
                  }
                  SB();
                  auto& acc_c = subtile_inplace<DC>(acc, c);
                  mma_AtB(acc_c, (c & 1) ? vb1 : vb0, pop, acc_c);
                  SB();
              }
            }
#else
            #pragma unroll
            for (int c = 0; c < NCH; c++) {
                VcT v_c;
                typename KS::template subtile<TK_SUB, DC> vsub(ks[cur], {sb, c});
                load(v_c, vsub);
                asm volatile("s_waitcnt lgkmcnt(0)");
                SB();
                auto& acc_c = subtile_inplace<DC>(acc, c);
                mma_AtB(acc_c, v_c, pop, acc_c);
                SB();
            }
#endif
            copy(m_i, m_new);
            SB();
        }
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
    store(g.Og, acc_t, coord<>{0, warpid, 0, 0});   // full head [16,512]
    typename ST_::row_vec lse;
    log(lse, l_tot);
    mul(m_fin, m_fin, LN2);
    add(lse, lse, m_fin);
    store(g.Lg, lse, coord<>{0,0,warpid,0});
}
