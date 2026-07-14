// hk_s2_occ1_stream_asm_mode.cpp — ASSEMBLY-MODE variant of hk_s2_occ1_stream (Q parked in AGPR).
// Goal: run the QK mma in art tiles so Q lives in AGPR, freeing ~64 VGPR (relieves the QK peak).
// After QK, bridge art s -> normal rt and reuse the proven normal-rt softmax + PV path.
//
// STATUS (WIP): s-art + bridge (art_s_to_rt) + normal-rt PV are wired. TWO blockers found, both rooting
// in "how do operands reach AGPR/art":
//  BLOCKER 1 (K art-load): art shared->register load only supports ST st_16x16_s/st_16x32_s -> rt_16x32_s.
//    The gather LDS `ks` is st_32x32_s (gather DMA + V ds_read_b64_tr) -> load<N,M>(k_art, ks_sub) =
//    "Unsupported shape". Worked around here with art_k_from_rt (load normal rt, copy->art via up2p).
//  BLOCKER 2 (Q into AGPR): a direct global buffer_load into AGPR (load<3>(qN, g.Qg, ...)) emits
//    `buffer_load_dwordx4 a[..]` which gfx950 rejects ("invalid operand for instruction", macros.cuh:473).
//    buffer_load targets VGPR only. The bwd gets Q into AGPR via global->shared (G::load) then
//    shared->AGPR (ds_read CAN write AGPR). up2p also can't write AGPR (v_mov only) and there is no
//    v_accvgpr_write macro in this HK -> the ONLY route to AGPR is the ds_read shared art-load.
// => To actually park Q in AGPR we must stage Q in an st_16x32 LDS per warp and ds_read it into AGPR.
//    Per-warp Q[16,512]=16KB x4 warps = 64KB on top of ks (st_32x32<64,512>x2=128KB) -> LDS budget must
//    be checked; may need sequential per-warp staging (barriers, one-time prologue) or a smaller stage.
// NEXT: replace the load<3> global loads below with global->shared(st_16x32)->ds_read-into-AGPR for Q;
//    and (blocker 1) either keep art_k_from_rt or add an st_16x32 K stage. The old working
//    hk_s2_occ1_stream.cpp is kept intact.
//
// ---- original header ----
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

// ===================== ASSEMBLY MODE (Q parked in AGPR) =====================
// The QK mma runs in assembly-mode art tiles so Q lives in AGPR (freeing ~64 VGPRs that q_arr used,
// which relieves the QK VGPR peak that forced the acc<->AGPR ferry). After the QK, we BRIDGE the art
// s (fp32) back to a normal rt and reuse the existing (proven) normal-rt softmax + PV path unchanged.
// Fixed config for asm mode: TILE_K=64, DC=64, QB=16, D=512 -> NCH=8.
// Register map:  Q 8 chunks -> AGPR a[0:63] (range 256..319);  K chunk -> VGPR v[64:95];  s -> v[96:111].
static_assert(TILE_K==64 && DC==64 && QB==16 && D==512, "asm mode fixed to TILE_K=64,DC=64,QB=16,D=512");
typedef uint32_t u2v __attribute__((ext_vector_type(2)));
// Q lives as ONE full-width [QB,D] art tile in AGPR a[0:63] (16 k-tiles). K chunk -> v[64:95], s -> v[96:111].
using Qf_r = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<256,319>>,4>;  // Q[16,512] whole (16 tiles)
using Kc_r = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<64,95>>,4>;    // K[64,64] rt_16x32 = 8 tiles
using Sc_r = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<96,111>>,4>;   // s[64,16] rt_16x16 = 4 tiles
using Qf_art = art<bf16,  QB, D,  row_l, rt_16x32_s, Qf_r>;
using Kc_art = art<bf16,  TILE_K, DC, row_l, rt_16x32_s, Kc_r>;
using Sc_art = art<float, TILE_K, QB, col_l, rt_16x16_s, Sc_r>;

// chunked QK: s += K_chunk . Qfull[:, chunk c]^T. mma_ABt_base with INDEPENDENT A/B ranges so the
// streamed K chunk (width 2 = 2 k-tiles) aligns to Qfull's global k-tiles [2c, 2c+1] (M=0).
template<int c>
__device__ static inline void qk_chunk(Sc_art&, const Kc_art&){
    [&]<std::size_t... Ns>(std::index_sequence<Ns...>){ ([&]<std::size_t N>(){
        { using rA=ducks::art::get_nth_range_t<Kc_art::register_ranges, N*2+0>;
          using rB=ducks::art::get_nth_range_t<Qf_art::register_ranges, 2*c+0>;
          using rCD=ducks::art::get_nth_range_t<Sc_art::register_ranges, N>;
          mma_ABt_base<typename Sc_art::shape, bf16, rA, rB, rCD, rCD>(); }
        { using rA=ducks::art::get_nth_range_t<Kc_art::register_ranges, N*2+1>;
          using rB=ducks::art::get_nth_range_t<Qf_art::register_ranges, 2*c+1>;
          using rCD=ducks::art::get_nth_range_t<Sc_art::register_ranges, N>;
          mma_ABt_base<typename Sc_art::shape, bf16, rA, rB, rCD, rCD>(); }
    }.template operator()<Ns>(),...); }(std::make_index_sequence<TILE_K/16>{});
}

// bridge art s (fp32 rt_16x16, col_l) -> normal rt ST_ (same logical layout), element-wise via p2up.
__device__ static inline void art_s_to_rt(const Sc_art&, ST_& d){
    using SR = Sc_art::register_ranges;
    [&]<std::size_t... Ts>(std::index_sequence<Ts...>){ ([&]<std::size_t T>(){
        constexpr int lo = ducks::art::get_nth_range_t<SR,T>::lo;
        *reinterpret_cast<uint32_t*>(&d.tiles[T][0].data[0]) = macros::v_mov_b32_p2up<lo+0>();
        *reinterpret_cast<uint32_t*>(&d.tiles[T][0].data[1]) = macros::v_mov_b32_p2up<lo+1>();
        *reinterpret_cast<uint32_t*>(&d.tiles[T][0].data[2]) = macros::v_mov_b32_p2up<lo+2>();
        *reinterpret_cast<uint32_t*>(&d.tiles[T][0].data[3]) = macros::v_mov_b32_p2up<lo+3>();
    }.template operator()<Ts>(),...); }(std::make_index_sequence<ST_::height*ST_::width>{});
}
// bridge normal rt K (rt_16x32, row_l) -> art K (same shape) via up2p (VGPR write). Option (a) until the
// gather LDS can feed the art shared-load natively. tiles[n][m] -> art range index n*width+m.
__device__ static inline void art_k_from_rt(const KcT& src, Kc_art&){
    using KR = Kc_art::register_ranges;
    constexpr int W = KcT::width;
    [&]<std::size_t... Ts>(std::index_sequence<Ts...>){ ([&]<std::size_t T>(){
        constexpr int lo = ducks::art::get_nth_range_t<KR,T>::lo;
        constexpr int n = T / W, m = T % W;
        macros::v_mov_b32_up2p<lo+0>(*reinterpret_cast<const uint32_t*>(&src.tiles[n][m].data[0]));
        macros::v_mov_b32_up2p<lo+1>(*reinterpret_cast<const uint32_t*>(&src.tiles[n][m].data[1]));
        macros::v_mov_b32_up2p<lo+2>(*reinterpret_cast<const uint32_t*>(&src.tiles[n][m].data[2]));
        macros::v_mov_b32_up2p<lo+3>(*reinterpret_cast<const uint32_t*>(&src.tiles[n][m].data[3]));
    }.template operator()<Ts>(),...); }(std::make_index_sequence<KcT::height*KcT::width>{});
}
// ===========================================================================

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

    // Q parked in AGPR (art): ONE full-width [QB,D] load from global (warpid must be UNIFORM or the
    // buffer resource lands in VGPRs -> invalid). K/s art clobbered for the QK mma.
    ducks::art::clobber<Qf_r>(); ducks::art::clobber<Kc_r>(); ducks::art::clobber<Sc_r>();
    Qf_art qfull;
    const int uw = __builtin_amdgcn_readfirstlane(warpid);
    load(qfull, g.Qg, coord<>{0,uw,0,0}, coord<>{0,0,0,0});
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

        // ---- QK in assembly-mode art: Q in AGPR, K streamed into VGPR art, s art ----
        Sc_art sart;
        zero(sart);
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
        // stream K chunk c: load into a NORMAL rt (st_32x32 native), bridge to art K, chunked mma vs Qfull.
        Kc_art k_art;
        #define QKC(c) { \
            KcT k_c; \
            typename KS::template subtile<TILE_K, DC> ksub(ks[cur], {0, (c)}); \
            load(k_c, ksub); \
            asm volatile("s_waitcnt lgkmcnt(0)"); SB(); \
            art_k_from_rt(k_c, k_art); SB(); \
            qk_chunk<(c)>(sart, k_art); SB(); }
        QKC(0) QKC(1) QKC(2) QKC(3) QKC(4) QKC(5) QKC(6) QKC(7)
        #undef QKC
        // bridge art s (fp32) -> normal rt; downstream softmax + PV are the proven normal-rt path.
        ST_ s;
        art_s_to_rt(sart, s);
        SB();
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
