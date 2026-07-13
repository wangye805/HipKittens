// hk_s2_occ1_stream.cpp — DSA-V4 Stage-2 fwd, dense-512, occ-1 NW=4, SUBTILE-STREAMED K/V reads.
// Same async-DMA double-buffer as hk_s2_occ1_db, but LDS->register reads are chunked over D (width DC):
//   - QK: loop D in DC-wide chunks; load k_c[TILE_K,DC] from a shared col-subtile, mma-accumulate into s.
//         Q held resident as DC-wide chunks q_arr[NCH] (same total regs, but sliceable for the chunk mma).
//   - PV: loop D-chunks; load v_c[TILE_K,DC], take subtile_inplace<DC>(acc,c) row-slice, one mma_AtB each.
// GOAL: never hold the whole K[32,512]/V[32,512] in VGPR -> frees VGPR so acc lives in VGPR (not AGPR),
//       killing the per-tile AGPR<->VGPR ferry of the online rescale (mul_col(acc,...)).
// Gather (HBM->LDS) is UNCHANGED — we still DMA the whole tile into LDS, only the reads are chunked.
#include "kittens.cuh"
#include "sub_topk.cuh"
#include "sub_gather.cuh"
using namespace kittens;

#ifndef TILE_K
#define TILE_K 32
#endif
#ifndef DC
#define DC 64
#endif
constexpr int QB = 16, D = 512;
constexpr int NCH = D / DC;                 // number of D-chunks
constexpr float LOG2E = 1.4426950408889634f, LN2 = 0.6931471805599453f;
#ifndef NW
#define NW 4
#endif
#define NT (kittens::WARP_THREADS * NW)
#ifndef NTILES
#define NTILES 4
#endif
#ifndef MINWAVES
#define MINWAVES 1
#endif

using QcT = rt<bf16,  QB,     DC, row_l, rt_16x32_s>;   // resident Q chunk
using KcT = rt<bf16,  TILE_K, DC, row_l, rt_16x32_s>;   // streamed K chunk
using VcT = rt<bf16,  TILE_K, DC, col_l, rt_32x16_s>;   // streamed V chunk
using ST_ = rt<float, TILE_K, QB, col_l, rt_16x16_s>;
using PbT = rt<bf16,  TILE_K, QB, col_l, rt_16x16_s>;
using PopT= rt<bf16,  TILE_K, QB, col_l, rt_32x16_s>;

// In-register rt_16x16 -> rt_32x16 (bf16, col_l) reshuffle via lane-swap net (NO LDS roundtrip).
// derived+verified: per (i,k): t=permlane32_swap(tile[2i].d[k], tile[2i+1].d[k]); {d[k],d[k+2]}=permlane16_swap(t.x,t.y).
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
using OT  = rt<float, D,      QB, col_l, rt_16x16_s>;   // acc [512,16] — now VGPR-resident
using OtT = rt<float, QB,     D, row_l, rt_16x16_s>;
using KS  = st_bf<TILE_K, D, st_32x32_s>;
using MSV = sv_fl<TILE_K>;   // per-key mask as a shared vector (register-mask like Leon; NO [TILE_K,QB] tile,
                             // NO cross-warp __syncthreads -> no vmcnt/lgkmcnt drain of the gather)

struct g_t {
    gl<bf16,-1,-1,-1,-1> Qg, KVg, Og;
    gl<int, -1,-1,-1,-1> Tkg;
    gl<float,-1,-1,-1,-1> Sg, Lg;
    int n_tiles, kv_rows;
    float scale;
    int has_sink;
};

__launch_bounds__(NT,MINWAVES)
__global__ void hk_s2_occ1_stream(const g_t g){
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    KS (&ks)[2] = al.allocate<KS, 2>();
    // (mask is now register-only via v_cndmask — no LDS mask vector needed)
    auto (&ps)[NW] = al.allocate<st_bf<TILE_K, QB, st_32x16_s>, NW>();
    __shared__ int topk_all[NTILES * TILE_K];

    const int tid = threadIdx.x;
    const int warpid = kittens::warpid();
    const int tok = blockIdx.x;
    const int nt = g.n_tiles;

    // Q resident as DC-wide chunks (loaded once; invariant across tiles)
#ifndef QSTREAM
    QcT q_arr[NCH];
    #pragma unroll
    for (int c = 0; c < NCH; c++) load(q_arr[c], g.Qg, coord<>{0, warpid, 0, c});
    __builtin_amdgcn_s_waitcnt(0);
#endif
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

#ifdef GATHER_INTERLEAVE
    uint32_t g_goff[NDMA]; kittens::i32x4 g_srsrc; uint32_t g_lds; bool g_have = false;
#endif
    for (int j = 0; j < nt; j++) {
        const int cur = j & 1, nxt = (j + 1) & 1;
#ifdef GATHER_INTERLEAVE
        asm volatile("s_waitcnt vmcnt(0)");   // drain THIS tile's gather (issued during prev PV / before loop)
        SB();
        __builtin_amdgcn_s_barrier();         // ks[cur] visible before QK
        SB();
        // phase1 for NEXT tile's gather: addr calc + topk reads now; DMAs issued interleaved in PV below
        if (j + 1 < nt) {
            gather_phase1<NT>(ks[nxt], g.KVg, topk_all + (j+1)*TILE_K, 0, g.kv_rows, g_goff, g_srsrc, g_lds);
            g_have = true;
        } else g_have = false;
        SB();
#else
        if (j + 1 < nt) {
            gather_kv_async<NT>(ks[nxt], g.KVg, topk_all + (j+1)*TILE_K, 0, g.kv_rows);
            SB();
            asm volatile("s_waitcnt vmcnt(%0)" :: "i"(NDMA));
            SB();
        } else {
            __builtin_amdgcn_s_waitcnt(0);
            SB();
        }
        __builtin_amdgcn_s_barrier();   // bare barrier: gather cross-warp (ks[cur] visible before QK). NO
        SB();                           // fill_mask/__syncthreads here anymore -> nothing drains the gather.
#endif

        // ---- QK: streamed over D chunks ----
        ST_ s;
        zero(s);
        SB();
        // HOIST tile-j topk read: issue now (16 ints via 4 ds_read_b128), keep in lgkmcnt slack through the
        // mma loop so its ~160cyc latency hides under QK; consumed at the mask below with NO exposed wait.
        int tkv[16];
        {
          const int grp = (threadIdx.x % kittens::WARP_THREADS) / 16;
          const int* tk = topk_all + j*TILE_K;
          #pragma unroll
          for (int m = 0; m < 16; m++) tkv[m] = tk[4*grp + (m & 3) + 16*(m >> 2)];
        }
        SB();
#ifdef DBUF
        // 2-deep VGPR double buffer: prefetch K chunk c+1 (LDS->VGPR) while mma'ing chunk c.
        // partial s_waitcnt lgkmcnt(WKT) keeps next chunk's reads AND the 4 hoisted topk reads in flight.
        {
          constexpr int RPC_K = TILE_K * DC / 512;   // ds_read_b128 count per K chunk
          constexpr int WK = RPC_K < 15 ? RPC_K : 15; // lgkmcnt caps at 15 on gfx950 (clamp; >=RPC_K drain still correct)
          constexpr int WKT = (WK + 4) < 15 ? (WK + 4) : 15;  // +4 topk b128 kept in flight (never partial-drained)
          KcT kb0, kb1;
          { typename KS::template subtile<TILE_K, DC> s0(ks[cur], {0, 0}); load(kb0, s0); }
          #pragma unroll
          for (int c = 0; c < NCH; c++) {
              if (c + 1 < NCH) {
                  typename KS::template subtile<TILE_K, DC> sn(ks[cur], {0, c + 1});
                  load((c & 1) ? kb0 : kb1, sn);                     // issue next (async)
                  asm volatile("s_waitcnt lgkmcnt(%0)" :: "i"(WKT)); // drain current K, keep next K + topk in flight
              } else {
                  asm volatile("s_waitcnt lgkmcnt(0)");              // final drain: topk now resident (mma-hidden)
              }
              SB();
              mma_ABt(s, (c & 1) ? kb1 : kb0, q_arr[c], s);         // overlaps next load
              SB();
#ifdef GATHER_INTERLEAVE
              if (g_have) gather_phase2_slice<NT, KS>(g_srsrc, g_lds, g_goff, 2*c, 2*c + 2);  // 2 DMAs/chunk, early (completion hides)
              SB();
#endif
          }
        }
#else
        #pragma unroll
        for (int c = 0; c < NCH; c++) {
            KcT k_c;
            typename KS::template subtile<TILE_K, DC> ksub(ks[cur], {0, c});   // rowcol in subtile-count units
            load(k_c, ksub);
#ifdef QSTREAM
            QcT q_c;
            load(q_c, g.Qg, coord<>{0, warpid, 0, c});
            asm volatile("s_waitcnt lgkmcnt(0)");
            SB();
            mma_ABt(s, k_c, q_c, s);
#else
            asm volatile("s_waitcnt lgkmcnt(0)");
            SB();
            mma_ABt(s, k_c, q_arr[c], s);
#endif
            SB();
        }
#endif
        mul(s, s, g.scale);
        SB();
        // ---- register mask via v_cndmask: per-lane set S=-inf where topk<0. No msv, no LDS round-trip. ----
        // verified layout: lane L owns head L%16, keys {4*(L/16) + w + 16*bt}, element(bt,p,xy) -> w=2p+xy.
        // topk read stays HERE for now (one exposed lgkmcnt); later hoist into QK to hide it.
        {
          // additive -inf mask: bias = topk & 0xFF800000. valid idx (0..4095 < 2^23) -> 0; -1 sentinel -> -inf.
          // topk (tkv) was HOISTED into the QK loop -> already resident, no exposed lgkmcnt here.
          // single v_and_b32 per element; NO vcc/cndmask -> no serialization; scale-mul fuses into the apply.
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
        mul_col(acc, acc, alpha);       // online rescale — now in-place VGPR (no accvgpr ferry)
        SB();

        PbT pb;
        copy(pb, s);                    // s(fp32) -> pb(bf16), rt_16x16
        SB();
        PopT pop;
#ifdef REG_RESHUFFLE
        p16_to_p32(pop, pb);            // in-register rt_16x16 -> rt_32x16 (permlane, NO LDS roundtrip)
        SB();
#else
        store(ps[warpid], pb);
        SB();
        asm volatile("s_waitcnt lgkmcnt(0)");
        SB();
#ifndef NO_PV_BARRIER
        __builtin_amdgcn_s_barrier();
        SB();
#endif
        load(pop, ps[warpid]);
        SB();
        asm volatile("s_waitcnt lgkmcnt(0)");
        SB();
#endif

        // ---- PV: streamed over D chunks (output-row slices of acc) ----
#ifdef DBUF
        {
          constexpr int RPC_V = TILE_K * DC / 256;   // ds_read_b64_tr count per V chunk (2x the K b128)
          constexpr int WV = RPC_V < 15 ? RPC_V : 15; // lgkmcnt caps at 15
          VcT vb0, vb1;
          { typename KS::template subtile<TILE_K, DC> v0(ks[cur], {0, 0}); load(vb0, v0); }
          #pragma unroll
          for (int c = 0; c < NCH; c++) {
              if (c + 1 < NCH) {
                  typename KS::template subtile<TILE_K, DC> vn(ks[cur], {0, c + 1});
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
            typename KS::template subtile<TILE_K, DC> vsub(ks[cur], {0, c});   // rowcol in subtile-count units
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
