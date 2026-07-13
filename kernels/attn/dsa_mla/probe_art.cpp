// Probe: validate HK 'art' assembly-mode QK with Q pinned to AGPR, K/s in VGPR.
// s[64,16] = K[64,32] . Q[16,32]^T  (contraction=32). Stage operands through shared (bwd pattern).
#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <random>
using namespace kittens;
#define HC(x) do{hipError_t e=(x);if(e){printf("err %s\n",hipGetErrorString(e));return 1;}}while(0)

using S_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<64,79>>, 4>;   // [64,16] fp32: 4 tiles -> 4 VGPR sub-ranges
using K_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<80,95>>, 4>;   // [64,32] bf16: 4 tiles
using Q_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<256,259>>, 4>; // [16,32] bf16: 1 tile -> AGPR a[0:3]

using S_art = art<float, 64, 16, col_l, rt_16x16_s, S_ranges>;
using K_art = art<bf16,  64, 32, row_l, rt_16x32_s, K_ranges>;
using Q_art = art<bf16,  16, 32, row_l, rt_16x32_s, Q_ranges>;

__global__ void probe(gl<bf16,-1,-1,-1,-1> gK, gl<bf16,-1,-1,-1,-1> gQ, gl<float,-1,-1,-1,-1> gS){
    extern __shared__ int __shm[]; shared_allocator al((int*)&__shm[0]);
    st_bf<64,32,st_16x32_s> &Ks = al.allocate<st_bf<64,32,st_16x32_s>>();
    st_bf<16,32,st_16x32_s> &Qs = al.allocate<st_bf<16,32,st_16x32_s>>();
    st_fl<64,16,st_16x16_s> &Ss = al.allocate<st_fl<64,16,st_16x16_s>>();
    load(Ks, gK, coord<>{0,0,0,0}); load(Qs, gQ, coord<>{0,0,0,0});
    __builtin_amdgcn_s_waitcnt(0); __syncthreads();

    ducks::art::clobber<S_ranges>(); ducks::art::clobber<K_ranges>(); ducks::art::clobber<Q_ranges>();
    S_art s; K_art k; Q_art q;
    uint32_t ka = get_address(k, Ks); load<0,0>(k,Ks,ka); load<0,1>(k,Ks,ka); load<0,2>(k,Ks,ka); load<0,3>(k,Ks,ka);
    uint32_t qa = get_address(q, Qs); load<0,0>(q,Qs,qa);
    __builtin_amdgcn_s_waitcnt(0);
    zero(s);
    mma_ABt(s, k, q, s);
    // (store elided for ISA probe)
    __builtin_amdgcn_s_waitcnt(0); __syncthreads();
    store(gS, Ss, coord<>{0,0,0,0});
}

int main(){
    std::mt19937 rng(0); std::uniform_real_distribution<float> U(-1,1);
    std::vector<float> K(64*32), Q(16*32); for(auto&x:K)x=U(rng); for(auto&x:Q)x=U(rng);
    std::vector<__hip_bfloat16> Kb(64*32),Qb(16*32);
    for(size_t i=0;i<Kb.size();i++)Kb[i]=(__hip_bfloat16)K[i]; for(size_t i=0;i<Qb.size();i++)Qb[i]=(__hip_bfloat16)Q[i];
    __hip_bfloat16 *dK,*dQ; float* dS;
    HC(hipMalloc(&dK,2*64*32)); HC(hipMalloc(&dQ,2*16*32)); HC(hipMalloc(&dS,4*64*16));
    HC(hipMemcpy(dK,Kb.data(),2*64*32,hipMemcpyHostToDevice)); HC(hipMemcpy(dQ,Qb.data(),2*16*32,hipMemcpyHostToDevice));
    gl<bf16,-1,-1,-1,-1> gK(dK,1,1,64,32), gQ(dQ,1,1,16,32); gl<float,-1,-1,-1,-1> gS(dS,1,1,64,16);
    hipFuncSetAttribute((void*)probe, hipFuncAttributeMaxDynamicSharedMemorySize, 32768);
    probe<<<1,64,32768>>>(gK,gQ,gS); HC(hipDeviceSynchronize());
    std::vector<float> S(64*16); HC(hipMemcpy(S.data(),dS,4*64*16,hipMemcpyDeviceToHost));
    double worst=0;
    for(int kk=0;kk<64;kk++)for(int qq=0;qq<16;qq++){
        double ref=0; for(int c=0;c<32;c++) ref += (double)(float)Kb[kk*32+c]*(double)(float)Qb[qq*32+c];
        double d=std::fabs(S[kk*16+qq]-ref); if(d>worst)worst=d;
    }
    printf("art QK worst abs err = %.5f  %s\n", worst, worst<0.05?"PASS":"FAIL");
    return 0;
}
