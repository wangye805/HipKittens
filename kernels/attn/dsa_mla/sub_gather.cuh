// sub_gather.cuh — submodule 2: async KV gather HBM -> LDS via raw_buffer_load_lds (global->LDS DMA).
// Indexed by topk (already in LDS, from submodule 1). NO register hop, ASYNC (vmcnt-tracked -> enables
// double-buffer overlap). Ported from hk_fwd4's gather_async (validated in gcheck, worst 0.0, NW=1/4).
//
// raw_buffer_load_lds writes LDS LANE-CONSECUTIVELY, so it does NOT follow HK's swizzle map. To land the
// gathered data in the swizzled slots HK's load/mma expect, we REVERSE-swizzle: for each lane-consecutive
// LDS byte slot, inverse-map to which logical (g_row=k, g_col=d) it must hold, then set that lane's global
// voffset = topk[g_row]*row_stride + col_off + g_col.  Caller drains with s_waitcnt vmcnt(N).
#pragma once
#include "kittens.cuh"

// kv_rows = total KV row count (== srsrc buffer range in rows); NOT a tile index. Must span all rows the
// topk can index, else OOB DMA reads return 0 (silent all-zero gather).
template<int N_THREADS, kittens::ducks::st::all ST, kittens::ducks::gl::all GL>
__device__ inline void gather_kv_async(ST& dst, const GL& kv, const int* topk, int col_off, int kv_rows) {
    using namespace kittens;
    using Tp = typename ST::dtype;
    constexpr int SUBR = ST::underlying_subtile_rows, SUBC = ST::underlying_subtile_cols;
    constexpr int SPR  = ST::underlying_subtiles_per_row, SUBB = ST::underlying_subtile_bytes;
    constexpr int RB   = ST::underlying_subtile_row_bytes;
    constexpr int bpt  = ST::underlying_subtile_bytes_per_thread;      // 16 (int4)
    constexpr int bpw  = bpt * kittens::WARP_THREADS;
    constexpr int totalB = ST::rows * ST::cols * (int)sizeof(Tp);
    constexpr int nw   = N_THREADS / kittens::WARP_THREADS;
    constexpr int mpt  = (totalB + bpt * N_THREADS - 1) / (bpt * N_THREADS);
    const int row_stride = kv.template stride<2>();
    Tp* gbase = (Tp*)&kv[coord<>{0, 0, 0, 0}];
    const int laneid = kittens::laneid(); const int warpid = kittens::warpid() % nw;
    i32x4 srsrc = make_srsrc(gbase, (uint32_t)((size_t)kv_rows * row_stride * sizeof(Tp)));  // full KV range
    const uint32_t lds_base = (uint32_t)(reinterpret_cast<uintptr_t>(&dst.data[0])) + warpid * bpw;
    // Phase 1: issue ALL topk LDS reads (pipelined) + compute per-slot g_col; defer goff (depends on pr).
    int pr_a[mpt]; int gcol_a[mpt];
    #pragma unroll
    for (int i = 0; i < mpt; i++) {
        const int lbo = laneid * bpt + warpid * bpw + i * nw * bpw;
        if (lbo >= totalB) { pr_a[i] = 0; gcol_a[i] = 0; continue; }
        const int sid = lbo / SUBB, srow = sid / SPR, scol = sid % SPR, so = lbo % SUBB;
        const int row = so / RB, col = (so % RB) / (int)sizeof(Tp);
        const uint32_t sw = dst.swizzle({row, col});                   // inverse-map through the swizzle (involution)
        const int g_row = (sw / RB) + srow * SUBR;                     // logical k
        gcol_a[i] = (sw % RB) / (int)sizeof(Tp) + scol * SUBC;         // logical d
        int pr = topk[g_row]; if (pr < 0) pr = 0;                      // safe gather (-1 padding) -- LDS read issued here
        pr_a[i] = pr;
    }
    asm volatile("s_waitcnt lgkmcnt(0)");                              // drain ALL topk reads ONCE (not per-iteration)
    // Phase 2: issue all DMAs. Pass the LDS dest via lds_ptr (HK global_to_shared style) and let the
    // COMPILER manage/schedule m0 -- NO hand-written `asm volatile s_mov m0`, which was a scheduling
    // barrier that pinned each DMA and prevented the compiler from pipelining them (the 48cyc serialization).
    #pragma unroll
    for (int i = 0; i < mpt; i++) {
        const int lbo = laneid * bpt + warpid * bpw + i * nw * bpw;
        if (lbo >= totalB) continue;
        uint32_t goff = (uint32_t)(((size_t)pr_a[i] * row_stride + col_off + gcol_a[i]) * sizeof(Tp));
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_base + (uint32_t)(i * nw * bpw));
        llvm_amdgcn_raw_buffer_load_lds(srsrc, lds_ptr, bpt, goff, 0, 0, (int)coherency::cache_all);
    }
}

// ---- SPLIT gather for interleaving into QK/PV ----
// phase1: reverse-swizzle addr calc + topk LDS reads -> fills goff[mpt]; returns srsrc + lds_base. (lgkmcnt)
template<int N_THREADS, kittens::ducks::st::all ST, kittens::ducks::gl::all GL>
__device__ inline void gather_phase1(ST& dst, const GL& kv, const int* topk, int col_off, int kv_rows,
                                     uint32_t* goff_out, kittens::i32x4& srsrc_out, uint32_t& lds_base_out) {
    using namespace kittens;
    using Tp = typename ST::dtype;
    constexpr int SUBR = ST::underlying_subtile_rows, SUBC = ST::underlying_subtile_cols;
    constexpr int SPR  = ST::underlying_subtiles_per_row, SUBB = ST::underlying_subtile_bytes;
    constexpr int RB   = ST::underlying_subtile_row_bytes;
    constexpr int bpt  = ST::underlying_subtile_bytes_per_thread;
    constexpr int bpw  = bpt * kittens::WARP_THREADS;
    constexpr int totalB = ST::rows * ST::cols * (int)sizeof(Tp);
    constexpr int nw   = N_THREADS / kittens::WARP_THREADS;
    constexpr int mpt  = (totalB + bpt * N_THREADS - 1) / (bpt * N_THREADS);
    const int row_stride = kv.template stride<2>();
    Tp* gbase = (Tp*)&kv[coord<>{0, 0, 0, 0}];
    const int laneid = kittens::laneid(); const int warpid = kittens::warpid() % nw;
    srsrc_out = make_srsrc(gbase, (uint32_t)((size_t)kv_rows * row_stride * sizeof(Tp)));
    lds_base_out = (uint32_t)(reinterpret_cast<uintptr_t>(&dst.data[0])) + warpid * bpw;
    int pr_a[mpt]; int gcol_a[mpt];
    #pragma unroll
    for (int i = 0; i < mpt; i++) {
        const int lbo = laneid * bpt + warpid * bpw + i * nw * bpw;
        if (lbo >= totalB) { pr_a[i] = 0; gcol_a[i] = 0; continue; }
        const int sid = lbo / SUBB, srow = sid / SPR, scol = sid % SPR, so = lbo % SUBB;
        const int row = so / RB, col = (so % RB) / (int)sizeof(Tp);
        const uint32_t sw = dst.swizzle({row, col});
        const int g_row = (sw / RB) + srow * SUBR;
        gcol_a[i] = (sw % RB) / (int)sizeof(Tp) + scol * SUBC;
        int pr = topk[g_row]; if (pr < 0) pr = 0;
        pr_a[i] = pr;
    }
    asm volatile("s_waitcnt lgkmcnt(0)");
    #pragma unroll
    for (int i = 0; i < mpt; i++)
        goff_out[i] = (uint32_t)(((size_t)pr_a[i] * row_stride + col_off + gcol_a[i]) * sizeof(Tp));
}

// phase2 slice: issue DMAs for units [u0,u1) from precomputed goff[] (vmcnt only, NO lgkmcnt).
template<int N_THREADS, kittens::ducks::st::all ST>
__device__ inline void gather_phase2_slice(const kittens::i32x4& srsrc, uint32_t lds_base,
                                           const uint32_t* goff, int u0, int u1) {
    using namespace kittens;
    constexpr int bpt = ST::underlying_subtile_bytes_per_thread;
    constexpr int bpw = bpt * kittens::WARP_THREADS;
    constexpr int nw  = N_THREADS / kittens::WARP_THREADS;
    constexpr int totalB = ST::rows * ST::cols * (int)sizeof(typename ST::dtype);
    const int laneid = kittens::laneid(); const int warpid = kittens::warpid() % nw;
    #pragma unroll
    for (int i = u0; i < u1; i++) {
        const int lbo = laneid * bpt + warpid * bpw + i * nw * bpw;
        if (lbo >= totalB) continue;
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_base + (uint32_t)(i * nw * bpw));
        llvm_amdgcn_raw_buffer_load_lds(srsrc, lds_ptr, bpt, goff[i], 0, 0, (int)coherency::cache_all);
    }
}

// # of raw_buffer_load_lds DMAs this issues (vmcnt increments) — for the caller's s_waitcnt vmcnt(N).
template<int N_THREADS, typename ST> __device__ constexpr int gather_ndma() {
    constexpr int bpt = ST::underlying_subtile_bytes_per_thread;
    constexpr int totalB = ST::rows * ST::cols * (int)sizeof(typename ST::dtype);
    return (totalB + bpt * N_THREADS - 1) / (bpt * N_THREADS);
}

// ---- topk tile HBM->LDS via raw_buffer_load_lds DMA (async, vmcnt-tracked, NO register hop, NO forced
// vmcnt(0) drain). topk is a LINEAR int array -> lane-consecutive DMA write lands correctly (no swizzle).
// TK ints = one warp's worth (gfx950 warp=64 lanes, TK=32 -> lanes 0..TK-1 active). Only warp 0 issues.
template<int N_THREADS, int TK, int RING>
__device__ inline void load_topk_tile_dma(int* ring, const kittens::gl<int,-1,-1,-1,-1>& Tkg, int tok, int tile) {
    using namespace kittens;
    constexpr int nw = N_THREADS / kittens::WARP_THREADS;
    const int warpid = kittens::warpid() % nw;
    if (warpid != 0) return;                                  // TK=32 ints => single warp
    const int laneid = kittens::laneid();
    int* dst   = ring + (tile % RING) * TK;                  // LDS destination (linear)
    int* gbase = (int*)&Tkg[coord<>{tok, 0, tile, 0}];       // HBM base of this tile's TK contiguous ints
    i32x4 srsrc = make_srsrc(gbase, (uint32_t)(TK * sizeof(int)));
    uint32_t lds_base = (uint32_t)(reinterpret_cast<uintptr_t>(dst));
    as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_base);     // M0 base; HW writes LDS[base + lane*bpt]
    if (laneid < TK) {
        llvm_amdgcn_raw_buffer_load_lds(srsrc, lds_ptr, (int)sizeof(int),
                                        (uint32_t)(laneid * sizeof(int)), 0, 0, (int)coherency::cache_all);
    }
}
