// sub_topk_test.cpp — unit test for submodule 1 (load_topk): HBM->LDS->HBM roundtrip vs input.
#include "sub_topk.cuh"
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdio>
#include <random>
using namespace kittens;
#define HC(x) do{hipError_t e=(x);if(e){printf("err %s\n",hipGetErrorString(e));return 1;}}while(0)
#ifndef NTILES
#define NTILES 36
#endif
#ifndef TILE_K
#define TILE_K 32
#endif
#define NT 256

// kernel: token=blockIdx.x; load its topk to LDS, then write LDS back out to Ochk[token, tile*TK+k].
__global__ void k_load_topk(const gl<int,-1,-1,-1,-1> Tkg, gl<int,-1,-1,-1,-1> Ochk, int n_tiles, int TK){
    __shared__ int smem[NTILES * TILE_K];
    const int tok = blockIdx.x;
    load_topk<NT>(smem, Tkg, tok, n_tiles, TK);
    __syncthreads();
    const int total = n_tiles * TK;
    for (int i = threadIdx.x; i < total; i += NT) Ochk[coord<>{tok, 0, i / TK, i % TK}] = smem[i];
}

int main(int argc,char**argv){
    int seed=argc>1?atoi(argv[1]):0; const int T=8, T_KV=4096;   // T tokens
    std::mt19937 rng(seed); std::uniform_int_distribution<int> Ti(-1,T_KV-1);  // include -1 (invalid)
    std::vector<int> topk(T*NTILES*TILE_K); for(auto&t:topk)t=Ti(rng);
    int *dT,*dO; HC(hipMalloc(&dT,sizeof(int)*topk.size())); HC(hipMalloc(&dO,sizeof(int)*topk.size()));
    HC(hipMemset(dO,0x7f,sizeof(int)*topk.size()));   // poison
    HC(hipMemcpy(dT,topk.data(),sizeof(int)*topk.size(),hipMemcpyHostToDevice));
    gl<int,-1,-1,-1,-1> Tkg(dT,T,1,NTILES,TILE_K), Ochk(dO,T,1,NTILES,TILE_K);
    k_load_topk<<<dim3(T),dim3(NT)>>>(Tkg, Ochk, NTILES, TILE_K); HC(hipDeviceSynchronize());
    std::vector<int> out(topk.size()); HC(hipMemcpy(out.data(),dO,sizeof(int)*topk.size(),hipMemcpyDeviceToHost));
    int bad=0; for(size_t i=0;i<topk.size();i++) if(out[i]!=topk[i]) bad++;
    printf("SUB_TOPK T=%d NTILES=%d TILE_K=%d seed=%d: bad=%d/%zu  %s\n",
           T,NTILES,TILE_K,seed,bad,topk.size(),bad==0?"PASS":"FAIL");
    return 0;
}
