// hk_s2_occ2.cpp — DSA-V4 Stage-2 fwd, dense-512, OCC-2 [4x2] cooperative, SUBTILE-STREAMED K/V.
//
// Structure (ONE CTA per token, 8 warps = 4 M-groups x 2 sub-warps => 2 waves/SIMD = occ-2):
//   mg  = warpid>>1  (0..3): query-head this warp serves (QB=16 queries).      [same as occ-1 warpid]
//   sub = warpid&1   (0/1) : which D-half this warp owns ([sub*256 : sub*256+256]).
//
//   QK  split-K : each sub reduces ITS D-half -> partial s[TILE_K,QB].
//   COMBINE     : s_full = s_sub0 + s_sub1  (fp32, hand-rolled LDS, 2-barrier). Both subs end with full s.
//   softmax     : both subs run the SAME softmax on full s (identical m_i/l_i/alpha/P) -> no broadcast needed.
//   PV  split-N : each sub owns its D_V-half -> acc[256,16] fp32 (64 VGPR).  acc STAYS VGPR (no AGPR ferry).
//
// Why split-K (not redundant full-QK in both subs): keeps TOTAL matrix work constant across the 2 waves
//   (2*(QK/2)+2*(PV/2) = QK+PV). Redundant-QK would be 1.5x the mma -> bad exactly at occ-2 where the two
//   waves share ONE matrix core. Why fp32 combine: bf16 rounding of a partial score (~2.5 raw) risks the
//   O ~1e-4 bar. Why hand-rolled LDS (not HK store/load): HK's tile store/load has no fp32 path (bf16/fp8
//   only); both subs' lane L hold identical (key,query) elements, so a plain float[lane*16+local] works.
//
// occ-2 gate (validated skeleton, OCC2_FEASIBILITY_RESULTS.md): [4x2] streamed => VGPR<=256, AGPR=0, occ 2.
// LDS: ks[2]=st_bf<64,512>x2 = 128KB + Scomb[4][1024]f = 16KB + topk ~= 149KB (<160KB/CU) -> 1 CTA/CU -> occ-2.
#include "kittens.cuh"
#include "sub_topk.cuh"
#include "sub_gather.cuh"
using namespace kittens;

#ifndef TILE_K
#define TILE_K 64
#endif
#ifndef DC
#define DC 64
#endif
constexpr int QB = 16, D = 512, DH = 256;   // DH = per-sub D / D_V half
constexpr int NC = DH / DC;                 // D-chunks per sub-warp's half (=4 @ DC=64)
constexpr float LOG2E = 1.4426950408889634f, LN2 = 0.6931471805599453f;
#ifndef NW
#define NW 8                                // 8 warps = [4x2]
#endif
#define NT (kittens::WARP_THREADS * NW)     // 512 threads
constexpr int NW_MG = NW / 2;               // 4 M-groups (query-heads per CTA)
#ifndef NTILES
#define NTILES 4
#endif
#ifndef MINWAVES
#define MINWAVES 2                          // ask the compiler for occ-2
#endif

using QcT = rt<bf16,  QB,     DC, row_l, rt_16x32_s>;   // resident Q chunk (sub's half)
using KcT = rt<bf16,  TILE_K, DC, row_l, rt_16x32_s>;   // streamed K chunk
using VcT = rt<bf16,  TILE_K, DC, col_l, rt_32x16_s>;   // streamed V chunk
using ST_ = rt<float, TILE_K, QB, col_l, rt_16x16_s>;
using PbT = rt<bf16,  TILE_K, QB, col_l, rt_16x16_s>;
using PopT= rt<bf16,  TILE_K, QB, col_l, rt_32x16_s>;

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
using OT  = rt<float, DH,     QB, col_l, rt_16x16_s>;   // acc [256,16] — this sub's D_V half, VGPR-resident
using OtT = rt<float, QB,     DH, row_l, rt_16x16_s>;   // [16,256]
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
__global__ void hk_s2_occ2(const g_t g){
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    KS (&ks)[2] = al.allocate<KS, 2>();
#if defined(SYM_COMBINE)
    // symmetric 1-barrier combine: both waves write disjoint slots, 1 barrier, both read+add in parallel.
    #ifdef SYM_FP32
    using SCT = float;                                        // 32KB (needs the LDS to fit)
    #else
    using SCT = __hip_bfloat16;                               // 16KB (bf16-rounds the partner's partial)
    #endif
    __shared__ SCT Scomb[NW_MG][2][TILE_K * QB];             // per (mg,dh): lane*16+local
#else
    __shared__ float Scomb[NW_MG][TILE_K * QB];      // 2-barrier fp32 combine scratch (per mg: lane*16+local), 16KB
#endif
    __shared__ int topk_all[NTILES * TILE_K];

    const int warpid = kittens::warpid();
    const int mg  = warpid >> 1;       // 0..3 query-head
    const int dh = warpid & 1;         // 0/1 D-half (named dh: `sub` shadows HK's sub() softmax op)
    const int lane = threadIdx.x % kittens::WARP_THREADS;
    const int tok = blockIdx.x;
    const int nt = g.n_tiles;

    // Q resident as DC-wide chunks. REDUNDANT_QK: full-D (both waves do full QK, NO S-combine).
    // else split-K: this sub's D-half only (+ S-combine).
#ifdef REDUNDANT_QK
    constexpr int NQ = D / DC;                     // full D (=8 @ DC64)
    #define QKCOL(c) (c)
#else
    constexpr int NQ = NC;                         // this sub's half (=4)
    #define QKCOL(c) (dh*NC + (c))
#endif
    QcT q_arr[NQ];
    #pragma unroll
    for (int c = 0; c < NQ; c++) load(q_arr[c], g.Qg, coord<>{0, mg, 0, QKCOL(c)});
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

        // ---- QK split-K: streamed over THIS sub's D-half (NC chunks) ----
        ST_ s;
        zero(s);
        SB();
        // HOIST tile-j topk read (16 ints via 4 ds_read_b128), keep in lgkmcnt slack through the mma loop.
        int tkv[16];
        {
          const int grp = lane / 16;
          const int* tk = topk_all + j*TILE_K;
          #pragma unroll
          for (int m = 0; m < 16; m++) tkv[m] = tk[4*grp + (m & 3) + 16*(m >> 2)];
        }
        SB();
#ifdef DBUF
        {
          constexpr int RPC_K = TILE_K * DC / 512;
          constexpr int WK = RPC_K < 15 ? RPC_K : 15;
          constexpr int WKT = (WK + 4) < 15 ? (WK + 4) : 15;
          KcT kb0, kb1;
          { typename KS::template subtile<TILE_K, DC> s0(ks[cur], {0, QKCOL(0)}); load(kb0, s0); }
          #pragma unroll
          for (int c = 0; c < NQ; c++) {
              if (c + 1 < NQ) {
                  typename KS::template subtile<TILE_K, DC> sn(ks[cur], {0, QKCOL(c + 1)});
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
        for (int c = 0; c < NQ; c++) {
            KcT k_c;
            typename KS::template subtile<TILE_K, DC> ksub(ks[cur], {0, QKCOL(c)});
            load(k_c, ksub);
            asm volatile("s_waitcnt lgkmcnt(0)");
            SB();
            mma_ABt(s, k_c, q_arr[c], s);
            SB();
        }
#endif

        // ---- S-COMBINE: s_full = s_sub0 + s_sub1 (fp32, hand-rolled LDS, 2-barrier) ----
        // sub0 lane L and sub1 lane L hold IDENTICAL (key,query) elements in the same reg slots -> index by
        // (lane, local) only. local = bt*4 + p*2 + xy, bt in [0,height), p in [0,2), xy in {x,y}.
#if defined(REDUNDANT_QK)
        // no S-combine: both waves already hold the FULL-D score s (full QK done redundantly). Nothing to do.
#elif defined(SYM_COMBINE)
        // symmetric 1-barrier: each wave writes its partial (bf16) to its own slot, one barrier, then each
        // reads the OTHER wave's partial and adds (my partial stays fp32; only the other's is bf16-rounded).
        {
          SCT* MY = &Scomb[mg][dh][0];
          #pragma unroll
          for (int bt = 0; bt < ST_::height; bt++)
            #pragma unroll
            for (int p = 0; p < 2; p++) {
              MY[lane*16 + bt*4 + p*2 + 0] = (SCT)s.tiles[bt][0].data[p].x;
              MY[lane*16 + bt*4 + p*2 + 1] = (SCT)s.tiles[bt][0].data[p].y;
            }
          asm volatile("s_waitcnt lgkmcnt(0)");   // LDS stores must LAND before the barrier (else cross-wave race)
          SB();
          __syncthreads();
          SB();
          SCT* OT_ = &Scomb[mg][1-dh][0];
          #pragma unroll
          for (int bt = 0; bt < ST_::height; bt++)
            #pragma unroll
            for (int p = 0; p < 2; p++) {
              s.tiles[bt][0].data[p].x += (float)OT_[lane*16 + bt*4 + p*2 + 0];
              s.tiles[bt][0].data[p].y += (float)OT_[lane*16 + bt*4 + p*2 + 1];
            }
          SB();
        }
#elif !defined(NO_COMBINE)
        {
          float* SC = &Scomb[mg][0];
          if (dh == 1) {
            #pragma unroll
            for (int bt = 0; bt < ST_::height; bt++)
              #pragma unroll
              for (int p = 0; p < 2; p++) {
                SC[lane*16 + bt*4 + p*2 + 0] = s.tiles[bt][0].data[p].x;
                SC[lane*16 + bt*4 + p*2 + 1] = s.tiles[bt][0].data[p].y;
              }
          }
          asm volatile("s_waitcnt lgkmcnt(0)");   // dh==1 stores must LAND before barrier A
          SB();
          __syncthreads();                        // barrier A: sub1's partial visible
          SB();
          if (dh == 0) {
            #pragma unroll
            for (int bt = 0; bt < ST_::height; bt++)
              #pragma unroll
              for (int p = 0; p < 2; p++) {
                s.tiles[bt][0].data[p].x += SC[lane*16 + bt*4 + p*2 + 0];
                s.tiles[bt][0].data[p].y += SC[lane*16 + bt*4 + p*2 + 1];
                SC[lane*16 + bt*4 + p*2 + 0] = s.tiles[bt][0].data[p].x;   // write full back
                SC[lane*16 + bt*4 + p*2 + 1] = s.tiles[bt][0].data[p].y;
              }
          }
          asm volatile("s_waitcnt lgkmcnt(0)");   // dh==0's full-s stores must LAND before barrier B
          SB();
          __syncthreads();                        // barrier B: full s visible
          SB();
          if (dh == 1) {
            #pragma unroll
            for (int bt = 0; bt < ST_::height; bt++)
              #pragma unroll
              for (int p = 0; p < 2; p++) {
                s.tiles[bt][0].data[p].x = SC[lane*16 + bt*4 + p*2 + 0];
                s.tiles[bt][0].data[p].y = SC[lane*16 + bt*4 + p*2 + 1];
              }
          }
          SB();
        }
#else
        __syncthreads();   // NO_COMBINE ablation: keep one barrier for a fair-ish cadence (INCORRECT math)
        SB();
#endif

        mul(s, s, g.scale);
        SB();
        // ---- additive -inf mask via topk & 0xFF800000 (valid idx <2^23 -> 0; -1 sentinel -> -inf) ----
        {
          float bias[16];
          #pragma unroll
          for (int m = 0; m < 16; m++)
            bias[m] = __int_as_float(tkv[m] & (int)0xFF800000);
          #pragma unroll
          for (int bt = 0; bt < 4; bt++)
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
        mul_col(acc, acc, alpha);       // online rescale — in-place VGPR (no accvgpr ferry)
        SB();

        PbT pb;
        copy(pb, s);                    // s(fp32) -> pb(bf16), rt_16x16
        SB();
        PopT pop;
        p16_to_p32(pop, pb);            // in-register rt_16x16 -> rt_32x16 (permlane, NO LDS roundtrip)
        SB();

        // ---- PV split-N: this sub owns its D_V half; stream V in NC chunks -> acc chunks [64,16] ----
#ifdef DBUF
        {
          constexpr int RPC_V = TILE_K * DC / 256;
          constexpr int WV = RPC_V < 15 ? RPC_V : 15;
          VcT vb0, vb1;
          { typename KS::template subtile<TILE_K, DC> v0(ks[cur], {0, dh*NC + 0}); load(vb0, v0); }
          #pragma unroll
          for (int c = 0; c < NC; c++) {
              if (c + 1 < NC) {
                  typename KS::template subtile<TILE_K, DC> vn(ks[cur], {0, dh*NC + c + 1});
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
        for (int c = 0; c < NC; c++) {
            VcT v_c;
            typename KS::template subtile<TILE_K, DC> vsub(ks[cur], {0, dh*NC + c});
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
        __syncthreads();
        SB();
    }

    typename ST_::row_vec l_tot, afix, m_fin;
    if (g.has_sink) {
        typename ST_::row_vec sink;
        load(sink, g.Sg, coord<>{0,0,mg,0});
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
    store(g.Og, acc_t, coord<OtT>{0, mg, 0, dh});  // this sub's D_V half; coord<OtT> => unit_coord scales c*256

    if (dh == 0) {                                 // LSE is per-query (same in both subs) -> write once
        typename ST_::row_vec lse;
        log(lse, l_tot);
        mul(m_fin, m_fin, LN2);
        add(lse, lse, m_fin);
        store(g.Lg, lse, coord<>{0,0,mg,0});
    }
}
