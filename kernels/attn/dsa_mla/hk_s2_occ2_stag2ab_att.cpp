// hk_s2_occ2_stag2ab_att.cpp — minimal single-launch driver for ATT capture of the streamed kernel.
#include "hk_s2_occ2_stag2ab.cpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <random>
#define HC(x) do{hipError_t e=(x);if(e){printf("err %s\n",hipGetErrorString(e));return 1;}}while(0)
#ifndef GRID_N
#define GRID_N 128
#endif
int main(){
    const int BH=QB*NW, NKK=NTILES*TILE_K, T_KV=4096;
    const float gs=(1.f/std::sqrt((float)D))*LOG2E;
    std::mt19937 rng(0); std::uniform_real_distribution<float> U(-1,1); std::uniform_int_distribution<int> Ti(0,T_KV-1);
    std::vector<float> Q(BH*D),KV(T_KV*D),sink(BH); for(auto&x:Q)x=U(rng)*0.1f; for(auto&x:KV)x=U(rng)*0.1f; for(auto&x:sink)x=U(rng)*0.5f;
    std::vector<int> topk((size_t)GRID_N*NKK); for(auto&t:topk)t=Ti(rng);
    std::vector<__hip_bfloat16> Qb(BH*D),KVb(T_KV*D);
    for(size_t i=0;i<Qb.size();i++)Qb[i]=(__hip_bfloat16)Q[i]; for(size_t i=0;i<KVb.size();i++)KVb[i]=(__hip_bfloat16)KV[i];
    __hip_bfloat16 *dQ,*dKV,*dO; int* dT; float *dS,*dL;
    HC(hipMalloc(&dQ,sizeof(__hip_bfloat16)*BH*D)); HC(hipMalloc(&dKV,sizeof(__hip_bfloat16)*T_KV*D));
    HC(hipMalloc(&dO,sizeof(__hip_bfloat16)*BH*D)); HC(hipMalloc(&dT,sizeof(int)*topk.size()));
    HC(hipMalloc(&dS,sizeof(float)*BH)); HC(hipMalloc(&dL,sizeof(float)*BH));
    HC(hipMemcpy(dQ,Qb.data(),sizeof(__hip_bfloat16)*BH*D,hipMemcpyHostToDevice));
    HC(hipMemcpy(dKV,KVb.data(),sizeof(__hip_bfloat16)*T_KV*D,hipMemcpyHostToDevice));
    HC(hipMemcpy(dT,topk.data(),sizeof(int)*topk.size(),hipMemcpyHostToDevice));
    HC(hipMemcpy(dS,sink.data(),sizeof(float)*BH,hipMemcpyHostToDevice));
    g_t g{ gl<bf16,-1,-1,-1,-1>(dQ,1,NW,QB,D), gl<bf16,-1,-1,-1,-1>(dKV,1,1,T_KV,D),
           gl<bf16,-1,-1,-1,-1>(dO,1,NW,QB,D), gl<int,-1,-1,-1,-1>(dT,GRID_N,1,NTILES,TILE_K),
           gl<float,-1,-1,-1,-1>(dS,1,1,NW,QB), gl<float,-1,-1,-1,-1>(dL,1,1,NW,QB),
           NTILES, T_KV, gs, 1 };
    size_t shmem=4*sizeof(KS);   // tight: actual ks[2] at TILE_K=32 = 64KB
    HC(hipFuncSetAttribute((void*)hk_s2_occ2_stag2ab, hipFuncAttributeMaxDynamicSharedMemorySize, shmem));
    hk_s2_occ2_stag2ab<<<dim3(GRID_N),dim3(NT),shmem>>>(g);   // single launch for ATT
    HC(hipDeviceSynchronize());
    printf("done GRID_N=%d NTILES=%d TILE_K=%d DC=%d\n", GRID_N, NTILES, TILE_K, DC);
    return 0;
}
