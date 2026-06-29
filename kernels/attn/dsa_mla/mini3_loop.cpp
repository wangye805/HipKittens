// mini3_loop.cpp — keys-as-rows ONLINE-SOFTMAX multi-tile validator (gather-free, NW M-split).
// Extends mini3.cpp with the running m_i/l_i/acc-rescale loop over NTILES tiles.
// K/V for all tiles are host-supplied (depth-indexed); gather + pipeline come in hk_fwd3.
#include "kittens.cuh"
using namespace kittens;

constexpr int QB = 16, TILE_K = 32, D_V = 512, D_ROPE = 64;
constexpr float LOG2E = 1.4426950408889634f;
#ifndef NW
#define NW 4
#endif
#ifndef NTILES
#define NTILES 4
#endif
#define NT (kittens::WARP_THREADS * NW)

using QlT = rt<bf16,  D_V,    QB,     col_l, rt_32x16_s>;
using QrT = rt<bf16,  D_ROPE, QB,     col_l, rt_32x16_s>;
using KlT = rt<bf16,  D_V,    TILE_K, col_l, rt_32x16_s>;
using KrT = rt<bf16,  D_ROPE, TILE_K, col_l, rt_32x16_s>;
using VT  = rt<bf16,  TILE_K, D_V,    col_l, rt_32x16_s>;
using ST  = rt<float, TILE_K, QB,     col_l, rt_16x16_s>;
using PbT = rt<bf16,  TILE_K, QB,     col_l, rt_16x16_s>;
using PopT= rt<bf16,  TILE_K, QB,     col_l, rt_32x16_s>;
using OT  = rt<float, D_V,    QB,     col_l, rt_16x16_s>;
using OtT = rt<float, QB,     D_V,    row_l, rt_16x16_s>;

struct g_t {
    gl<bf16,-1,-1,-1,-1> QlTg, QrTg, KlTg, KrTg, Vg, Og;
    float scale; int nt;
};

__launch_bounds__(NT,1)
__global__ void mini3l(const g_t g){
    const int warpid = kittens::warpid();
    QlT q_l; load(q_l, g.QlTg, coord<>{0,warpid,0,0});
    QrT q_r; load(q_r, g.QrTg, coord<>{0,warpid,0,0});
    __builtin_amdgcn_s_waitcnt(0);

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    auto (&ps)[NW] = al.allocate<st_bf<TILE_K, QB, st_32x16_s>, NW>();

    typename ST::row_vec m_i, l_i, m_new, alpha;
    neg_infty(m_i); zero(l_i);
    OT acc; zero(acc);

    for(int j=0; j<g.nt; j++){
        KlT k_l; load(k_l, g.KlTg, coord<>{0,j,0,0});
        KrT k_r; load(k_r, g.KrTg, coord<>{0,j,0,0});
        VT  v_l; load(v_l, g.Vg,   coord<>{0,j,0,0});
        __builtin_amdgcn_s_waitcnt(0);

        ST s; zero(s);
        mma_AtB(s, k_l, q_l, s);
        mma_AtB(s, k_r, q_r, s);
        mul(s, s, g.scale);

        col_max(m_new, s, m_i);          // m_new = max(m_i, colmax(s))
        sub(alpha, m_i, m_new); exp2(alpha, alpha);   // alpha = exp2(m_i - m_new)
        sub_col(s, s, m_new); exp2(s, s);             // P = exp2(s - m_new)
        mul(l_i, l_i, alpha);
        col_sum(l_i, s, l_i);            // l_i = alpha*l_i + colsum(P)
        mul_col(acc, acc, alpha);        // rescale running acc

        PbT pb; copy(pb, s);
        store(ps[warpid], pb);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        PopT pop; load(pop, ps[warpid]);
        __builtin_amdgcn_s_waitcnt(0);

        mma_AtB(acc, v_l, pop, acc);
        copy(m_i, m_new);
    }

    div_col(acc, acc, l_i);
    OtT acc_t; transpose(acc_t, acc);
    store(g.Og, acc_t, coord<>{0,warpid,0,0});
}
