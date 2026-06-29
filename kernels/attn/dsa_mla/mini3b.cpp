// mini3b.cpp — keys-as-rows with NATURAL operands via mma_ABt (NO transpose, NO transposed gather).
// QK: mma_ABt(S, K, Q) with K[TILE_K,D_V] row_l, Q[QB,D_V] row_l -> S[TILE_K,QB] keys-as-rows.
//     S[k,h] = sum_d K[k,d]*Q[h,d].  Both operands NATURAL (no transpose, no transposed gather).
// V  = the SAME K data re-loaded as col_l rt_32x16 (the gluon "V=permute" win, one gather two loads).
// PV: mma_AtB(acc, V, P) as in mini3.  Single-tile, NW M-split.  Gather-free (host natural K/Q).
#include "kittens.cuh"
using namespace kittens;

constexpr int QB = 16, TILE_K = 32, D_V = 512, D_ROPE = 64;
constexpr float LOG2E = 1.4426950408889634f;
#ifndef NW
#define NW 4
#endif
#define NT (kittens::WARP_THREADS * NW)

using QlT = rt<bf16,  QB,     D_V,    row_l, rt_16x32_s>;  // Q_lora natural [16,512]  (QK B)
using QrT = rt<bf16,  QB,     D_ROPE, row_l, rt_16x32_s>;  // Q_rope natural [16,64]   (QK B)
using KlT = rt<bf16,  TILE_K, D_V,    row_l, rt_16x32_s>;  // K_lora natural [32,512]  (QK A)
using KrT = rt<bf16,  TILE_K, D_ROPE, row_l, rt_16x32_s>;  // K_rope natural [32,64]   (QK A)
using VT  = rt<bf16,  TILE_K, D_V,    col_l, rt_32x16_s>;  // V_lora = same K data col_l (PV A)
using ST  = rt<float, TILE_K, QB,     col_l, rt_16x16_s>;
using PbT = rt<bf16,  TILE_K, QB,     col_l, rt_16x16_s>;
using PopT= rt<bf16,  TILE_K, QB,     col_l, rt_32x16_s>;
using OT  = rt<float, D_V,    QB,     col_l, rt_16x16_s>;
using OtT = rt<float, QB,     D_V,    row_l, rt_16x16_s>;

struct g_t {
    gl<bf16,-1,-1,-1,-1> Qlg, Qrg, Klg, Krg, Og;  // Klg also serves as V (re-load col_l)
    float scale;
};

__launch_bounds__(NT,1)
__global__ void mini3b(const g_t g){
    const int warpid = kittens::warpid();
    QlT q_l; load(q_l, g.Qlg, coord<>{0,warpid,0,0});
    QrT q_r; load(q_r, g.Qrg, coord<>{0,warpid,0,0});
    KlT k_l; load(k_l, g.Klg, coord<>{0,0,0,0});
    KrT k_r; load(k_r, g.Krg, coord<>{0,0,0,0});
    VT  v_l; load(v_l, g.Klg, coord<>{0,0,0,0});   // SAME data as k_l, col_l layout
    __builtin_amdgcn_s_waitcnt(0);

    ST s; zero(s);
    mma_ABt(s, k_l, q_l, s);        // S = K @ Q^T  (contract D_V=512)
    mma_ABt(s, k_r, q_r, s);        // contract D_ROPE=64
    mul(s, s, g.scale);

    typename ST::row_vec m, l;
    col_max(m, s);
    sub_col(s, s, m); exp2(s, s);
    col_sum(l, s);

    PbT pb; copy(pb, s);
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    auto (&ps)[NW] = al.allocate<st_bf<TILE_K, QB, st_32x16_s>, NW>();
    store(ps[warpid], pb);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    PopT pop; load(pop, ps[warpid]);
    __builtin_amdgcn_s_waitcnt(0);

    OT acc; zero(acc);
    mma_AtB(acc, v_l, pop, acc);
    div_col(acc, acc, l);

    OtT acc_t; transpose(acc_t, acc);
    store(g.Og, acc_t, coord<>{0,warpid,0,0});
}
