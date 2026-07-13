// sub_gather_test.cpp — unit test for submodule 2 (gather_kv_async): async DMA gather -> HK load -> compare.
// gather KV[topk] into a swizzled KS via raw_buffer_load_lds, drain vmcnt, HK-load K row_l, store, compare
// to KV[topk][d]. Validates the reverse-swizzle lands data where HK's load/mma read it.
#include "sub_topk.cuh"
#include "sub_gather.cuh"
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <random>
using namespace kittens;
static inline float bf(float x){return (float)(__hip_bfloat16)x;}
#define HC(x) do{hipError_t e=(x);if(e){printf("err %s\n",hipGetErrorString(e));return 1;}}while(0)
#ifndef TILE_K
#define TILE_K 32
#endif
constexpr int D = 512;
#define NT 256   // 4 warps cooperate on the gather

using KS  = st_bf<TILE_K, D, st_32x32_s>;
using KlT = rt<bf16, TILE_K, D, row_l, rt_16x32_s>;

struct g_t { gl<bf16,-1,-1,-1,-1> KVg, Kdbg; gl<int,-1,-1,-1,-1> Tkg; int t_kv; };

__global__ void k_gather(const g_t g){
    __shared__ alignment_dummy __shm[sizeof(KS)/sizeof(alignment_dummy)+16];
    shared_allocator al((int*)&__shm[0]);
    KS &ks = al.allocate<KS>();
    __shared__ int topk[TILE_K];
    load_topk<NT>(topk, g.Tkg, 0, 1, TILE_K);          // submodule 1: topk HBM->LDS (1 tile)
    __syncthreads();
    gather_kv_async<NT>(ks, g.KVg, topk, 0, g.t_kv);   // submodule 2: async DMA gather
    asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    KlT k; load(k, ks); __builtin_amdgcn_s_waitcnt(0); // HK load (row_l) -> proves data is where mma reads
    store(g.Kdbg, k, coord<>{0,0,0,0});                 // Kdbg[k,d] = K[k,d]
}

int main(int argc,char**argv){
    int seed=argc>1?atoi(argv[1]):0; int ninv=argc>2?atoi(argv[2]):0; const int T_KV=4096;
    std::mt19937 rng(seed); std::uniform_real_distribution<float> U(-1,1); std::uniform_int_distribution<int> Ti(0,T_KV-1);
    std::vector<float> KV(T_KV*D); for(auto&x:KV)x=U(rng)*0.1f;
    std::vector<int> topk(TILE_K); for(auto&t:topk)t=Ti(rng);
    for(int i=0;i<ninv&&i<TILE_K;i++) topk[TILE_K-1-i]=-1;
    std::vector<__hip_bfloat16> KVb(T_KV*D); for(size_t i=0;i<KVb.size();i++)KVb[i]=(__hip_bfloat16)KV[i];
    __hip_bfloat16 *dKV,*dK; int* dT;
    HC(hipMalloc(&dKV,sizeof(__hip_bfloat16)*T_KV*D)); HC(hipMalloc(&dK,sizeof(__hip_bfloat16)*TILE_K*D)); HC(hipMalloc(&dT,sizeof(int)*TILE_K));
    HC(hipMemcpy(dKV,KVb.data(),sizeof(__hip_bfloat16)*T_KV*D,hipMemcpyHostToDevice));
    HC(hipMemcpy(dT,topk.data(),sizeof(int)*TILE_K,hipMemcpyHostToDevice));
    g_t g{ gl<bf16,-1,-1,-1,-1>(dKV,1,1,T_KV,D), gl<bf16,-1,-1,-1,-1>(dK,1,1,TILE_K,D), gl<int,-1,-1,-1,-1>(dT,1,1,1,TILE_K), T_KV };  // t_kv = KV row count (srsrc range), NOT a tile index
    k_gather<<<dim3(1),dim3(NT)>>>(g); HC(hipDeviceSynchronize());
    std::vector<__hip_bfloat16> K(TILE_K*D); HC(hipMemcpy(K.data(),dK,sizeof(__hip_bfloat16)*TILE_K*D,hipMemcpyDeviceToHost));
    int bad=0,nan=0; float worst=0;
    for(int k=0;k<TILE_K;k++){ int pr=topk[k]<0?0:topk[k];
        for(int d=0;d<D;d++){ float ref=bf(KV[pr*D+d]), hk=(float)K[k*D+d];
            if(!(hk==hk)){nan++;continue;} float e=std::fabs(hk-ref); if(e>worst)worst=e; if(e>0.001f)bad++; } }
    printf("SUB_GATHER TILE_K=%d seed=%d ninv=%d: K worst=%.5f bad=%d/%d nan=%d  %s\n",
           TILE_K,seed,ninv,worst,bad,D*TILE_K,nan,(bad==0&&nan==0)?"PASS":"FAIL");
    return 0;
}
