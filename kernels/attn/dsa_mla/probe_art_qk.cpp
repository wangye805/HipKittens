// Piece 1: full-contraction QK in art with Q resident in AGPR, K double-buffered in VGPR.
// s[64,16] = sum_c K_c[64,64] . Q_c[16,64]^T  (2 D-chunks here; 8 in the real kernel).
// Validate: mma reads a[] for every Q chunk; NO accvgpr ferries for K (Q out of VGPR => no overflow).
#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <random>
using namespace kittens;
#define HC(x) do{hipError_t e=(x);if(e){printf("err %s\n",hipGetErrorString(e));return 1;}}while(0)

using S_ranges  = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<0,15>>, 4>;    // s [64,16] fp32: 4 tiles
using K0_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<16,47>>, 4>;   // K [64,64] bf16: 8 tiles VGPR
using K1_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<48,79>>, 4>;
using Q0_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<256,263>>, 4>; // Q [16,64] bf16: 2 tiles AGPR a[0:7]
using Q1_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<264,271>>, 4>; // a[8:15]

using S_art  = art<float, 64, 16, col_l, rt_16x16_s, S_ranges>;
using K0_art = art<bf16,  64, 64, row_l, rt_16x32_s, K0_ranges>;
using K1_art = art<bf16,  64, 64, row_l, rt_16x32_s, K1_ranges>;
using Q0_art = art<bf16,  16, 64, row_l, rt_16x32_s, Q0_ranges>;
using Q1_art = art<bf16,  16, 64, row_l, rt_16x32_s, Q1_ranges>;

__global__ void probe(gl<bf16,-1,-1,-1,-1> gK, gl<bf16,-1,-1,-1,-1> gQ, gl<float,-1,-1,-1,-1> gS){
    extern __shared__ int __shm[]; shared_allocator al((int*)&__shm[0]);
    st_bf<64,128,st_16x32_s> &Ks = al.allocate<st_bf<64,128,st_16x32_s>>();
    st_bf<16,128,st_16x32_s> &Qs = al.allocate<st_bf<16,128,st_16x32_s>>();
    load(Ks, gK, coord<>{0,0,0,0}); load(Qs, gQ, coord<>{0,0,0,0});
    __builtin_amdgcn_s_waitcnt(0); __syncthreads();

    ducks::art::clobber<S_ranges>();
    ducks::art::clobber<K0_ranges>(); ducks::art::clobber<K1_ranges>();
    ducks::art::clobber<Q0_ranges>(); ducks::art::clobber<Q1_ranges>();
    S_art s; K0_art k0; K1_art k1; Q0_art q0; Q1_art q1;
    // Q resident (loaded once, into AGPR): slice chunk 0 = cols[0:64], chunk 1 = cols[64:128]
    { auto qsub0 = subtile_inplace<16,64>(Qs, {0,0}); uint32_t a=get_address(q0,qsub0); load<0,0>(q0,qsub0,a); load<0,1>(q0,qsub0,a); }
    { auto qsub1 = subtile_inplace<16,64>(Qs, {0,1}); uint32_t a=get_address(q1,qsub1); load<0,0>(q1,qsub1,a); load<0,1>(q1,qsub1,a); }
    // K chunks (VGPR)
    { auto ks0 = subtile_inplace<64,64>(Ks, {0,0}); uint32_t a=get_address(k0,ks0); load<0,0>(k0,ks0,a); load<0,1>(k0,ks0,a); load<0,2>(k0,ks0,a); load<0,3>(k0,ks0,a); load<0,4>(k0,ks0,a); load<0,5>(k0,ks0,a); load<0,6>(k0,ks0,a); load<0,7>(k0,ks0,a); }
    { auto ks1 = subtile_inplace<64,64>(Ks, {0,1}); uint32_t a=get_address(k1,ks1); load<0,0>(k1,ks1,a); load<0,1>(k1,ks1,a); load<0,2>(k1,ks1,a); load<0,3>(k1,ks1,a); load<0,4>(k1,ks1,a); load<0,5>(k1,ks1,a); load<0,6>(k1,ks1,a); load<0,7>(k1,ks1,a); }
    __builtin_amdgcn_s_waitcnt(0);
    zero(s);
    mma_ABt(s, k0, q0, s);
    mma_ABt(s, k1, q1, s);
    // force-use s so it isn't DCE'd (real correctness comes with art col_max in piece 2)
    ((float*)&gS[coord<>{0,0,0,0}])[threadIdx.x] = 0; // placeholder store; ISA is the validation here
}
int main(){ printf("compiled - inspect ISA for a[] Q operands + no accvgpr K ferries\n"); return 0; }
