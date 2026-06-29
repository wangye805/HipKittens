// gcheck.cpp — isolate the gather: gather KV[topk] -> shared -> load row_l -> store, compare.
#include "kittens.cuh"
using namespace kittens;
constexpr int TILE_K=32, D_V=512, D_QK=576;
#ifndef NW
#define NW 1
#endif
#define NT (kittens::WARP_THREADS*NW)

template<int N_THREADS, ducks::st::all ST, ducks::gl::all GL>
__device__ inline void gather_to_shared(ST& dst, const GL& kv, const int* topk, int col_off){
    using Tp = typename ST::dtype;
    constexpr int SUBR=ST::underlying_subtile_rows, SUBC=ST::underlying_subtile_cols;
    constexpr int SPR=ST::underlying_subtiles_per_row, SUBB=ST::underlying_subtile_bytes;
    constexpr int rows=ST::rows, cols=ST::cols, total=rows*cols;
    const int row_stride=kv.template stride<2>();
    Tp* gbase=(Tp*)&kv[coord<>{0,0,0,0}];
    char* lds=(char*)&dst.data[0];
    const int t=threadIdx.x;
    for(int e=t;e<total;e+=N_THREADS){
        const int k=e/cols, d=e%cols;
        int pr=topk[k]; if(pr<0)pr=0;
        Tp val=gbase[(size_t)pr*row_stride+col_off+d];
        const int sub_id=(k/SUBR)*SPR+(d/SUBC);
        const uint32_t off=sub_id*SUBB+dst.swizzle({k%SUBR,d%SUBC});
        *(Tp*)(lds+off)=val;
    }
}
using KlS=st_bf<TILE_K,D_V,st_32x32_s>;
using KlT=rt<bf16,TILE_K,D_V,row_l,rt_16x32_s>;
using VT =rt<bf16,TILE_K,D_V,col_l,rt_32x16_s>;     // V (col_l) from same shared tile
using VtT=rt<bf16,D_V,TILE_K,row_l,rt_16x32_s>;     // V^T (transpose of col_l) for store
struct g_t{ gl<bf16,-1,-1,-1,-1> KVg, Og, OVg; gl<int,-1,-1,-1,-1> Tkg; };
__launch_bounds__(NT,1)
__global__ void gck(const g_t g){
    extern __shared__ alignment_dummy __shm[]; shared_allocator al((int*)&__shm[0]);
    KlS &ks=al.allocate<KlS>();
    __shared__ int topk[TILE_K];
    int tid=threadIdx.x;
    if(tid<TILE_K) topk[tid]=g.Tkg[coord<>{0,0,0,tid}];
    __syncthreads();
    gather_to_shared<NT>(ks,g.KVg,topk,0);
    __builtin_amdgcn_s_waitcnt(0); __syncthreads();
    KlT k_l; load(k_l,ks); __builtin_amdgcn_s_waitcnt(0);
    store(g.Og,k_l,coord<>{0,0,0,0});   // Og=[TILE_K,D_V]
    VT v_l; load(v_l,ks); __builtin_amdgcn_s_waitcnt(0);
    VtT vt; transpose(vt,v_l);
    store(g.OVg,vt,coord<>{0,0,0,0});    // OVg=[D_V,TILE_K]  OVg[d][k]=KV[topk[k]][d]
}
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <random>
static inline __hip_bfloat16 f2b(float x){return (__hip_bfloat16)x;}
static inline float b2f(__hip_bfloat16 x){return (float)x;}
#define HC(x) do{hipError_t e=(x);if(e){printf("err %s\n",hipGetErrorString(e));return 1;}}while(0)
int main(int argc,char**argv){
    int seed=argc>1?atoi(argv[1]):0; const int T_KV=4096;
    std::mt19937 rng(seed); std::uniform_real_distribution<float> U(-1,1);
    std::uniform_int_distribution<int> Ti(0,T_KV-1);
    std::vector<float> KV(T_KV*D_QK); for(auto&x:KV)x=U(rng);
    std::vector<int> topk(TILE_K); for(auto&t:topk)t=Ti(rng);
    std::vector<__hip_bfloat16> KVb(KV.size()); for(size_t i=0;i<KV.size();i++)KVb[i]=f2b(KV[i]);
    __hip_bfloat16 *dKV,*dO,*dOV; int* dT;
    HC(hipMalloc(&dKV,sizeof(__hip_bfloat16)*T_KV*D_QK));
    HC(hipMalloc(&dO,sizeof(__hip_bfloat16)*TILE_K*D_V));
    HC(hipMalloc(&dOV,sizeof(__hip_bfloat16)*D_V*TILE_K));
    HC(hipMalloc(&dT,sizeof(int)*TILE_K));
    {std::vector<__hip_bfloat16> s(TILE_K*D_V,(__hip_bfloat16)-999.f);
     HC(hipMemcpy(dO,s.data(),sizeof(__hip_bfloat16)*TILE_K*D_V,hipMemcpyHostToDevice));
     HC(hipMemcpy(dOV,s.data(),sizeof(__hip_bfloat16)*D_V*TILE_K,hipMemcpyHostToDevice));}
    HC(hipMemcpy(dKV,KVb.data(),sizeof(__hip_bfloat16)*T_KV*D_QK,hipMemcpyHostToDevice));
    HC(hipMemcpy(dT,topk.data(),sizeof(int)*TILE_K,hipMemcpyHostToDevice));
    g_t g{ gl<bf16,-1,-1,-1,-1>(dKV,1,1,T_KV,D_QK),
           gl<bf16,-1,-1,-1,-1>(dO,1,1,TILE_K,D_V),
           gl<bf16,-1,-1,-1,-1>(dOV,1,1,D_V,TILE_K),
           gl<int,-1,-1,-1,-1>(dT,1,1,1,TILE_K) };
    size_t shmem=65536;
    HC(hipFuncSetAttribute((void*)gck,hipFuncAttributeMaxDynamicSharedMemorySize,shmem));
    gck<<<dim3(1),dim3(NT),shmem>>>(g); HC(hipDeviceSynchronize());
    std::vector<__hip_bfloat16> Ob(TILE_K*D_V), OVb(D_V*TILE_K);
    HC(hipMemcpy(Ob.data(),dO,sizeof(__hip_bfloat16)*TILE_K*D_V,hipMemcpyDeviceToHost));
    HC(hipMemcpy(OVb.data(),dOV,sizeof(__hip_bfloat16)*D_V*TILE_K,hipMemcpyDeviceToHost));
    int bad=0,nanc=0,badv=0,nanv=0; float worst=0,worstv=0;
    for(int k=0;k<TILE_K;k++)for(int d=0;d<D_V;d++){
        float ref=(float)f2b(KV[topk[k]*D_QK+d]); float hk=b2f(Ob[k*D_V+d]);
        if(!(hk==hk))nanc++; else { float e=std::fabs(hk-ref); if(e>worst)worst=e; if(e>0.001f)bad++; }
        float hv=b2f(OVb[d*TILE_K+k]);
        if(!(hv==hv))nanv++; else { float e=std::fabs(hv-ref); if(e>worstv)worstv=e; if(e>0.001f)badv++; }
    }
    printf("GCHECK NW=%d seed=%d: K row_l worst=%.5f bad=%d nan=%d | V col_l worst=%.5f bad=%d nan=%d %s\n",
           NW,seed,worst,bad,nanc,worstv,badv,nanv,
           (bad==0&&nanc==0&&badv==0&&nanv==0)?"PASS":"FAIL");
    return 0;
}
