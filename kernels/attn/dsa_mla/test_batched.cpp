// test_batched.cpp — multi-token correctness for the BATCHED dsa_mla_fwd kernel (bf16-faithful CPU ref).
// Validates per-token Q/O/LSE indexing. Build: hipcc ... -DDSA_NO_PYBIND -DNTILES=<n> test_batched.cpp
#define DSA_NO_PYBIND
#include "kernel.cpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <random>
static inline __hip_bfloat16 f2b(float x){return (__hip_bfloat16)x;}
static inline float b2f(__hip_bfloat16 x){return (float)x;}
static inline float bf(float x){return (float)(__hip_bfloat16)x;}
#define HC(x) do{hipError_t e=(x);if(e){printf("err %s\n",hipGetErrorString(e));return 1;}}while(0)

int main(int argc,char**argv){
    int S=argc>1?atoi(argv[1]):4, seed=argc>2?atoi(argv[2]):0, ninv=argc>3?atoi(argv[3]):0;
    const int BH=QB*NW, NKK=NTILES*TILE_K, T_KV=4096;
    const float scale_nat=1.f/std::sqrt((float)D);
    std::mt19937 rng(seed); std::uniform_real_distribution<float> U(-1,1); std::uniform_int_distribution<int> Ti(0,T_KV-1);
    std::vector<float> Q((size_t)S*BH*D),KV(T_KV*D),sink(BH);
    for(auto&x:Q)x=U(rng)*0.1f; for(auto&x:KV)x=U(rng)*0.1f; for(auto&x:sink)x=U(rng)*0.5f;
    std::vector<int> topk((size_t)S*NKK); for(auto&t:topk)t=Ti(rng);
    for(int s=0;s<S;s++) for(int i=0;i<ninv&&i<NKK;i++) topk[(size_t)s*NKK + NKK-1-i]=-1;   // per-token invalids
    auto mkb=[&](const std::vector<float>&f){std::vector<__hip_bfloat16> b(f.size());
        for(size_t i=0;i<f.size();i++)b[i]=f2b(f[i]); return b;};
    auto Qb=mkb(Q),KVb=mkb(KV);
    __hip_bfloat16 *dQ,*dKV,*dO; int* dT; float *dS,*dL;
    HC(hipMalloc(&dQ,sizeof(__hip_bfloat16)*(size_t)S*BH*D)); HC(hipMalloc(&dKV,sizeof(__hip_bfloat16)*T_KV*D));
    HC(hipMalloc(&dO,sizeof(__hip_bfloat16)*(size_t)S*BH*D)); HC(hipMalloc(&dT,sizeof(int)*(size_t)S*NKK));
    HC(hipMalloc(&dS,sizeof(float)*BH)); HC(hipMalloc(&dL,sizeof(float)*(size_t)S*BH));
    { std::vector<__hip_bfloat16> sV((size_t)S*BH*D,(__hip_bfloat16)-999.f); HC(hipMemcpy(dO,sV.data(),sizeof(__hip_bfloat16)*(size_t)S*BH*D,hipMemcpyHostToDevice)); }
    HC(hipMemcpy(dQ,Qb.data(),sizeof(__hip_bfloat16)*(size_t)S*BH*D,hipMemcpyHostToDevice));
    HC(hipMemcpy(dKV,KVb.data(),sizeof(__hip_bfloat16)*T_KV*D,hipMemcpyHostToDevice));
    HC(hipMemcpy(dT,topk.data(),sizeof(int)*(size_t)S*NKK,hipMemcpyHostToDevice));
    HC(hipMemcpy(dS,sink.data(),sizeof(float)*BH,hipMemcpyHostToDevice));
    // globals: Qg/Og [S,NW,QB,D], KVg [1,1,T_KV,D], Tkg [S,1,NTILES,TILE_K], Sg [1,1,NW,QB], Lg [S,1,NW,QB]
    globals g{ gl<bf16,-1,-1,-1,-1>(dQ,S,NW,QB,D), gl<bf16,-1,-1,-1,-1>(dKV,1,1,T_KV,D),
               gl<int,-1,-1,-1,-1>(dT,S,1,NTILES,TILE_K), gl<float,-1,-1,-1,-1>(dS,1,1,NW,QB),
               gl<bf16,-1,-1,-1,-1>(dO,S,NW,QB,D), gl<float,-1,-1,-1,-1>(dL,S,1,NW,QB) };
    size_t shmem=g.dynamic_shared_memory();
    HC(hipFuncSetAttribute((void*)dsa_mla_fwd, hipFuncAttributeMaxDynamicSharedMemorySize, shmem));
    dsa_mla_fwd<<<g.grid(),g.block(),shmem>>>(g); HC(hipDeviceSynchronize());
    std::vector<__hip_bfloat16> Ob((size_t)S*BH*D); std::vector<float> Lo((size_t)S*BH);
    HC(hipMemcpy(Ob.data(),dO,sizeof(__hip_bfloat16)*(size_t)S*BH*D,hipMemcpyDeviceToHost));
    HC(hipMemcpy(Lo.data(),dL,sizeof(float)*(size_t)S*BH,hipMemcpyDeviceToHost));
    int nanc=0,g3=0; double esum=0; long ecnt=0; float worst=0,lworst=0;
    for(int s=0;s<S;s++) for(int h=0;h<BH;h++){
        const float* Qh=&Q[((size_t)s*BH+h)*D]; const int* tk=&topk[(size_t)s*NKK];
        std::vector<float> sc(NKK); float m=-1e30f;
        for(int kk=0;kk<NKK;kk++){int kr=tk[kk]; if(kr<0){sc[kk]=-1e30f;continue;} float d=0;
            for(int q=0;q<D;q++) d+=bf(Qh[q])*bf(KV[(size_t)kr*D+q]); sc[kk]=d*scale_nat; if(sc[kk]>m)m=sc[kk];}
        float l=0; for(int kk=0;kk<NKK;kk++){sc[kk]=(sc[kk]<=-1e29f)?0.f:std::exp(sc[kk]-m); l+=sc[kk];}
        float denom=HAS_SINK?l+std::exp(sink[h]-m):l; float lse_ref=m+std::log(denom);
        float le=std::fabs(lse_ref-Lo[(size_t)s*BH+h]); if(le>lworst)lworst=le;
        for(int v=0;v<D;v++){float a=0; for(int kk=0;kk<NKK;kk++){int kr=tk[kk]; if(kr<0)continue; a+=bf(sc[kk])*bf(KV[(size_t)kr*D+v]);}
            float o=a/denom,hk=b2f(Ob[((size_t)s*BH+h)*D+v]); if(!(hk==hk)){nanc++;continue;}
            float e=std::fabs(hk-o); if(e>worst)worst=e; esum+=e; ecnt++; if(e>0.03f)g3++;}
    }
    printf("DSA_MLA_FWD batched S=%d NTILES=%d topk=%d HAS_SINK=%d ninv=%d seed=%d: O worst=%.5f mean=%.6f >.03=%d nan=%d | LSE worst=%.6f  %s\n",
           S,NTILES,NKK,HAS_SINK,ninv,seed,worst,ecnt?esum/ecnt:0,g3,nanc,lworst,(g3==0&&nanc==0&&lworst<0.01f)?"PASS":"FAIL");
    return 0;
}
