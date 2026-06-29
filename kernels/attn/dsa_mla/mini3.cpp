// mini3.cpp — keys-as-rows single-tile validator (gqa DISCIPLINE + gluon STRUCTURE).
//
// Orientation = gqa keys-as-rows (HK-NATIVE, no vec-layout conversions):
//   S/att = [TILE_K keys, QB heads] col_l.  col_max -> row_vec (per-head);
//   sub_col / col_sum / div_col all consume that SAME row_vec.  One layout family.
//
// QK  = two mma_AtB (S += K_T^T @ Q_T) :   K_T,[D_V,TILE_K] col_l ; Q_T,[D_V,QB] col_l
//          mma_AtB(D,A,B)=A^T@B  ->  S[k,h] = sum_d K_T[d,k]*Q_T[d,h] = sum_d K[k,d]*Q[h,d]
// PV  = mma_AtB(acc, V, P) :   V=[TILE_K,D_V] col_l (= K natural) ; P=[TILE_K,QB] col_l
//          acc[d,h] = sum_k V[k,d]*P[k,h]
//
// HK mma 16x16x32:  mma_AtB base wants A=rt_32x16, B=rt_32x16, D=rt_16x16 (mfma161632).
//   S is rt_16x16-based; PV's P operand must be rt_32x16-based -> the "roundtrip":
//   convert P(rt_16x16)->rt_32x16 via an LDS store/load (the proven BUG-1 mechanism).
//
// This validator is GATHER-FREE (host supplies K already transposed + V natural) and
// SINGLE-TILE (no online softmax loop) to isolate the compute/layout correctness.
// NW=1 by default (one warp owns the QB=16 heads).
#include "kittens.cuh"
using namespace kittens;

constexpr int QB = 16, TILE_K = 32, D_V = 512, D_ROPE = 64;
constexpr float LOG2E = 1.4426950408889634f;
#ifndef NW
#define NW 1
#endif
#define NT (kittens::WARP_THREADS * NW)

// ---- operand tiles (all col_l, base rt_32x16 for the mma operands) ----
using QlT = rt<bf16,  D_V,    QB,     col_l, rt_32x16_s>;  // Q_lora^T  [512,16]  (B)
using QrT = rt<bf16,  D_ROPE, QB,     col_l, rt_32x16_s>;  // Q_rope^T  [64,16]   (B)
using KlT = rt<bf16,  D_V,    TILE_K, col_l, rt_32x16_s>;  // K_lora^T  [512,32]  (A)
using KrT = rt<bf16,  D_ROPE, TILE_K, col_l, rt_32x16_s>;  // K_rope^T  [64,32]   (A)
using VT  = rt<bf16,  TILE_K, D_V,    col_l, rt_32x16_s>;  // V_lora    [32,512]  (A, = K natural)
// ---- S / softmax / acc (rt_16x16-based) ----
using ST  = rt<float, TILE_K, QB,     col_l, rt_16x16_s>;  // S/P float [32,16]   (D)
using PbT = rt<bf16,  TILE_K, QB,     col_l, rt_16x16_s>;  // P bf16    [32,16]
using PopT= rt<bf16,  TILE_K, QB,     col_l, rt_32x16_s>;  // P operand [32,16]   (B)
using OT  = rt<float, D_V,    QB,     col_l, rt_16x16_s>;  // acc       [512,16]  (D)
using OtT = rt<float, QB,     D_V,    row_l, rt_16x16_s>;  // acc^T for store [16,512]

struct g_t {
    gl<bf16,-1,-1,-1,-1> QlTg, QrTg, KlTg, KrTg, Vg, Og;
    float scale;
};

__launch_bounds__(NT,1)
__global__ void mini3(const g_t g){
    const int warpid = kittens::warpid();

    // M-split: each warp owns QB=16 heads. Per-warp Q/O index goes in the DEPTH coord
    // dim (dim-1) — the proven HK convention (col-dim {0,0,0,warpid} silently fails for
    // col-tile>=2). K/V are shared across the BLOCK_H heads of this token (coord 0).
    QlT q_l; load(q_l, g.QlTg, coord<>{0,warpid,0,0});
    QrT q_r; load(q_r, g.QrTg, coord<>{0,warpid,0,0});
    KlT k_l; load(k_l, g.KlTg, coord<>{0,0,0,0});
    KrT k_r; load(k_r, g.KrTg, coord<>{0,0,0,0});
    VT  v_l; load(v_l, g.Vg,   coord<>{0,0,0,0});
    __builtin_amdgcn_s_waitcnt(0);

    // ---- QK : S = K_T^T @ Q_T  (two contractions) ----
    ST s; zero(s);
    mma_AtB(s, k_l, q_l, s);        // contract D_V=512
    mma_AtB(s, k_r, q_r, s);        // contract D_ROPE=64
    mul(s, s, g.scale);

    // ---- softmax over keys (rows) -> per-head row_vec ----
    typename ST::row_vec m, l;
    col_max(m, s);                  // m[h] = max_k s[k,h]
    sub_col(s, s, m);               // s -= m   (per column/head)
    exp2(s, s);
    col_sum(l, s);                  // l[h] = sum_k s[k,h]

    // ---- P -> bf16, then rt_16x16 -> rt_32x16 via LDS roundtrip ----
    PbT pb; copy(pb, s);
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    // one roundtrip buffer per warp (each warp holds different heads)
    auto (&ps)[NW] = al.allocate<st_bf<TILE_K, QB, st_32x16_s>, NW>();
    store(ps[warpid], pb);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    PopT pop; load(pop, ps[warpid]);
    __builtin_amdgcn_s_waitcnt(0);

    // ---- PV : acc = V^T @ P ----
    OT acc; zero(acc);
    mma_AtB(acc, v_l, pop, acc);    // acc[d,h] = sum_k V[k,d]*P[k,h]
    div_col(acc, acc, l);           // normalize per head

    // ---- store acc^T = [QB, D_V] (per-warp in depth coord) ----
    OtT acc_t; transpose(acc_t, acc);
    store(g.Og, acc_t, coord<>{0,warpid,0,0});
}
