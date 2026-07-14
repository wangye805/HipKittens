// sub_topk.cuh — submodule 1: topk indices HBM -> LDS (deep-prefetch all NTILES*TILE_K once).
// Coalesced int load; the whole topk list for this token lives in shared so the tile loop has no
// per-tile topk-load latency and the async KV gather can index it directly.
#pragma once
#include "kittens.cuh"

// Load topk[tile, k] for tile in [0,n_tiles), k in [0,TK) from Tkg[{tok,0,tile,k}] into smem[tile*TK + k].
template<int N_THREADS>
__device__ inline void load_topk(int* smem, const kittens::gl<int,-1,-1,-1,-1>& Tkg,
                                 int tok, int n_tiles, int TK) {
    const int total = n_tiles * TK;
    for (int i = threadIdx.x; i < total; i += N_THREADS) {
        smem[i] = Tkg[kittens::coord<>{tok, 0, i / TK, i % TK}];
    }
}

// load ONE topk tile [tile, 0..TK) into a ring slot (per-tile prefetch). Named (not a macro) so the
// ISA->source resolver gives it its own frame in the ATT viewer.
template<int N_THREADS, int TK, int RING>
__device__ inline void load_topk_tile(int* ring, const kittens::gl<int,-1,-1,-1,-1>& Tkg, int tok, int tile) {
    int* dst = ring + (tile % RING) * TK;
    #pragma unroll
    for (int k = threadIdx.x; k < TK; k += N_THREADS)
        dst[k] = Tkg[kittens::coord<>{tok, 0, tile, k}];
}
