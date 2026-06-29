// hk_fwd5.cpp — CROSS-TILE SWP attempt (WIP). Carry S_prev; QK(t+1)∥softmax+PV(t) to hide latency.
// STATUS: correct at nt<=8, but a race/bug at nt=36 (256 bad, mostly deterministic) + spills 28 VGPR (S_prev
//   +S_cur carry over budget) -> 11ms (SLOWER than hk_fwd4 8.8ms). The latency-hiding could help (my kernel
//   is latency-bound ~7ms) but only if the spill is removed (register relief) AND the high-nt bug fixed.
//   Shelved pending register relief. hk_fwd4 = the stable deliverable.
// hk_fwd4.cpp — DSA V4 sparse-MLA fwd, HK gfx950, SOFTWARE-PIPELINED (toward gluon early-gather 3.03ms).
// ⚠⚠ WIP / NOT CORRECT YET. The async gather (raw_buffer_load_lds, reverse-swizzle) is VALIDATED in
// isolation (gcheck), but dropping it into the full kernel trips the codegen KNIFE-EDGE (VGPR=256, 0 spill):
// single-buffer async = nondeterministic partial NaN; 2-buffer = deterministic all-NaN. SAME root cause as
// the int4 sync gather — any faster-than-scalar gather perturbs the fragile single-buffer scheduling.
// STABILIZING THE PIPELINE NEEDS (next, careful): (a) subtile the PV (load V in D_V-chunks via the
// shared->reg col_offset load + acc subtile_inplace) to drop peak VGPR below 256; AND/OR (b) explicit
// __builtin_amdgcn_sched_group_barrier(MFMA/VALU) to pin softmax(t)∥QK(t+1) so codegen can't wander.
// hk_fwd3.cpp (scalar gather) remains the STABLE/correct kernel. This file is the pipeline scaffold.
// Built on hk_fwd3's validated keys-as-rows compute. 2-buffer ASYNC: prefetch tile t+1 (raw_buffer_load_lds,
// reverse-swizzle) into the other buffer while computing tile t. All topk preloaded to shared.
#include "kittens.cuh"
using namespace kittens;

constexpr int QB = 16, TILE_K = 32, D_V = 512, D_ROPE = 64, D_QK = D_V + D_ROPE;
constexpr float LOG2E = 1.4426950408889634f;
#ifndef NW
#define NW 4
#endif
#ifndef NTILES
#define NTILES 36
#endif
#define NT (kittens::WARP_THREADS * NW)

// Async global->LDS gather: KV[topk[g_row]] -> dst, lane-consecutive LDS (M0) so use REVERSE swizzle
// (validated in gcheck). Issues DMAs; caller waits via s_waitcnt vmcnt.
template<int N_THREADS, ducks::st::all ST, ducks::gl::all GL>
__device__ inline void gather_async(ST& dst, const GL& kv, const int* topk, int col_off, int t_kv) {
    using Tp = typename ST::dtype;
    constexpr int SUBR = ST::underlying_subtile_rows, SUBC = ST::underlying_subtile_cols;
    constexpr int SPR = ST::underlying_subtiles_per_row, SUBB = ST::underlying_subtile_bytes;
    constexpr int RB = ST::underlying_subtile_row_bytes;
    constexpr int bpt = ST::underlying_subtile_bytes_per_thread;   // 16
    constexpr int bpw = bpt * kittens::WARP_THREADS;
    constexpr int totalB = ST::rows * ST::cols * (int)sizeof(Tp);
    constexpr int nw = N_THREADS / kittens::WARP_THREADS;
    constexpr int mpt = (totalB + bpt * N_THREADS - 1) / (bpt * N_THREADS);
    const int row_stride = kv.template stride<2>();
    Tp* gbase = (Tp*)&kv[coord<>{0, 0, 0, 0}];
    const int laneid = kittens::laneid(); const int warpid = kittens::warpid() % nw;
    i32x4 srsrc = make_srsrc(gbase, (uint32_t)((size_t)t_kv * row_stride * sizeof(Tp)));
    const uintptr_t lds_base = reinterpret_cast<uintptr_t>(&dst.data[0]) + warpid * bpw;
    #pragma unroll
    for (int i = 0; i < mpt; i++) {
        const int lbo = laneid * bpt + warpid * bpw + i * nw * bpw;
        if (lbo >= totalB) continue;
        const int sid = lbo / SUBB, srow = sid / SPR, scol = sid % SPR, so = lbo % SUBB;
        const int row = so / RB, col = (so % RB) / (int)sizeof(Tp);
        const uint32_t sw = dst.swizzle({row, col});
        const int g_row = (sw / RB) + srow * SUBR;
        const int g_col = (sw % RB) / (int)sizeof(Tp) + scol * SUBC;
        int pr = topk[g_row]; if (pr < 0) pr = 0;
        uint32_t goff = (uint32_t)(((size_t)pr * row_stride + col_off + g_col) * sizeof(Tp));
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_base + i * nw * bpw);
        llvm_amdgcn_raw_buffer_load_lds(srsrc, lds_ptr, bpt, goff, 0, 0, (int)coherency::cache_all);
    }
}

using QlT = rt<bf16,  QB,     D_V,    row_l, rt_16x32_s>;
using QrT = rt<bf16,  QB,     D_ROPE, row_l, rt_16x32_s>;
using KlT = rt<bf16,  TILE_K, D_V,    row_l, rt_16x32_s>;
using KrT = rt<bf16,  TILE_K, D_ROPE, row_l, rt_16x32_s>;
using VT  = rt<bf16,  TILE_K, D_V,    col_l, rt_32x16_s>;
using ST  = rt<float, TILE_K, QB,     col_l, rt_16x16_s>;
using PbT = rt<bf16,  TILE_K, QB,     col_l, rt_16x16_s>;
using PopT= rt<bf16,  TILE_K, QB,     col_l, rt_32x16_s>;
using OT  = rt<float, D_V,    QB,     col_l, rt_16x16_s>;
using OtT = rt<float, QB,     D_V,    row_l, rt_16x16_s>;
using KlS = st_bf<TILE_K, D_V,    st_32x32_s>;
using KrS = st_bf<TILE_K, D_ROPE, st_32x32_s>;
// # of raw_buffer_load_lds instructions (vmcnt increments) a gather_async<N_THREADS> issues for tile ST.
template<int N_THREADS, typename ST> __device__ constexpr int gdma() {
    constexpr int bpt = ST::underlying_subtile_bytes_per_thread;
    constexpr int totalB = ST::rows * ST::cols * (int)sizeof(typename ST::dtype);
    return (totalB + bpt * N_THREADS - 1) / (bpt * N_THREADS);
}
using MbT = rt<bf16, TILE_K, QB, col_l, rt_16x16_s>;
using MtS = st_bf<TILE_K, QB, st_16x16_s>;

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
    gl<int, -1,-1,-1,-1> Tkg;     // [T,1,NTILES,TILE_K] topk
    gl<float,-1,-1,-1,-1> Sg;     // [1,1,HG*NW,QB] sink
    int n_tiles, t_kv;
    float scale;
    int has_sink;
};

__launch_bounds__(NT,1)
__global__ void hk_fwd5(const g_t g){
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    KlS (&ks)[2]  = al.allocate<KlS, 2>();
    KrS (&krs)[2] = al.allocate<KrS, 2>();
    auto (&ps)[NW] = al.allocate<st_bf<TILE_K, QB, st_32x16_s>, NW>();
#ifdef ENABLE_MASK
    MtS (&mts)[NW] = al.allocate<MtS, NW>();
#endif
    __shared__ int topk_all[NTILES * TILE_K];

    const int tid = threadIdx.x;
    const int warpid = kittens::warpid();
    const int HG = gridDim.y;
    const int qo_row = blockIdx.x * HG + blockIdx.y;
    const int sink_row = blockIdx.y * NW + warpid;
    const int tok = blockIdx.x;
    const int nt = g.n_tiles;

    QlT q_l; load(q_l, g.Qlg, coord<>{qo_row,warpid,0,0});
    QrT q_r; load(q_r, g.Qrg, coord<>{qo_row,warpid,0,0});

    // preload ALL topk for this token into shared (so loop has only gather DMAs on vmcnt)
    for (int i = tid; i < nt * TILE_K; i += NT) topk_all[i] = g.Tkg[coord<>{tok,0,i/TILE_K,i%TILE_K}];
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();

    OT acc; zero(acc);
    typename ST::row_vec m_i, l_i, m_new, alpha;
    neg_infty(m_i); zero(l_i);

    // CROSS-TILE SWP (gluon early-gather structure). Carry S_prev = QK(tile t); each loop iter does
    // QK(tile t+1) [MFMA] and softmax+PV(tile t) [VALU+MFMA] as INDEPENDENT work -> they overlap.
    // late-V (v_l loaded inside softmax_pv, after qk's k_l is dead) keeps VGPR<256 -> stable w/o heavy SB.
    #define SB() __builtin_amdgcn_sched_barrier(0)
    constexpr int NDMA = gdma<NT,KlS>() + gdma<NT,KrS>();

    auto qk = [&](ST& Sout, int buf, const int* tk) {
        zero(Sout);
        KlT k_l; load(k_l, ks[buf]); KrT k_r; load(k_r, krs[buf]);
        asm volatile("s_waitcnt lgkmcnt(0)"); SB();
        mma_ABt(Sout, k_l, q_l, Sout); SB();
        mma_ABt(Sout, k_r, q_r, Sout); SB();
        mul(Sout, Sout, g.scale);
#ifdef ENABLE_MASK
        fill_mask<kittens::WARP_THREADS>(mts[warpid], tk); asm volatile("s_waitcnt lgkmcnt(0)");
        MbT mtb; load(mtb, mts[warpid]); asm volatile("s_waitcnt lgkmcnt(0)");
        typename MbT::col_vec cvb; row_max(cvb, mtb);
        typename ST::col_vec cv; copy(cv, cvb); add_row(Sout, Sout, cv);
#endif
    };
    auto softmax_pv = [&](ST& Sp, int vbuf) {
        col_max(m_new, Sp, m_i);
        sub(alpha, m_i, m_new); exp2(alpha, alpha);
        sub_col(Sp, Sp, m_new); exp2(Sp, Sp);
        mul(l_i, l_i, alpha); col_sum(l_i, Sp, l_i);
        mul_col(acc, acc, alpha);
        PbT pb; copy(pb, Sp);
        store(ps[warpid], pb); asm volatile("s_waitcnt lgkmcnt(0)"); SB();
        PopT pop; load(pop, ps[warpid]); asm volatile("s_waitcnt lgkmcnt(0)"); SB();
        VT v_l; load(v_l, ks[vbuf]); asm volatile("s_waitcnt lgkmcnt(0)"); SB();
        mma_AtB(acc, v_l, pop, acc); SB();
        copy(m_i, m_new);
    };

    // prologue: gather tile 0 + tile 1 (assumes nt>=2)
    gather_async<NT>(ks[0],  g.KVg, topk_all,          0,   g.t_kv);
    gather_async<NT>(krs[0], g.KVg, topk_all,          D_V, g.t_kv);
    gather_async<NT>(ks[1],  g.KVg, topk_all + TILE_K, 0,   g.t_kv);
    gather_async<NT>(krs[1], g.KVg, topk_all + TILE_K, D_V, g.t_kv);
    SB();
    asm volatile("s_waitcnt vmcnt(%0)" :: "i"(NDMA)); SB();   // drain tile0, tile1 in flight
    __syncthreads(); SB();
    ST S_prev; qk(S_prev, 0, topk_all);                       // S_prev = QK(tile 0)

    for (int t = 0; t < nt - 1; t++) {
        const int cur = t & 1, nxt = (t + 1) & 1;
        asm volatile("s_waitcnt vmcnt(0)"); SB();             // tile t+1 (buf nxt) gather done
        __syncthreads(); SB();
        ST S_cur; qk(S_cur, nxt, topk_all + (t+1)*TILE_K);    // QK(tile t+1) [MFMA]
        softmax_pv(S_prev, cur);                              // softmax+PV(tile t) [overlaps QK]
        __syncthreads(); SB();                                // all warps done reading buf cur
        if (t + 2 < nt) {                                     // gather tile t+2 into freed buf cur (async)
            gather_async<NT>(ks[cur],  g.KVg, topk_all + (t+2)*TILE_K, 0,   g.t_kv);
            gather_async<NT>(krs[cur], g.KVg, topk_all + (t+2)*TILE_K, D_V, g.t_kv); SB();
        }
        copy(S_prev, S_cur);                                  // promote
    }
    softmax_pv(S_prev, (nt - 1) & 1);                         // drain: softmax+PV(tile nt-1)

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
