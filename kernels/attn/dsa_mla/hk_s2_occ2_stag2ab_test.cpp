// hk_s2_occ2_stag2ab_test.cpp — correctness for the 3-buffer ring kernel (step 2). Same CPU ref as tk32sb.
#include "hk_s2_occ2_stag2ab.cpp"
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
    int seed=argc>1?atoi(argv[1]):0, has_sink=argc>2?atoi(argv[2]):1, ninv=argc>3?atoi(argv[3]):0;
    const int BH=QB*NW, NKK=NTILES*TILE_K, T_KV=4096;
    const float scale_nat=1.f/std::sqrt((float)D), gs=scale_nat*LOG2E;
    std::mt19937 rng(seed); std::uniform_real_distribution<float> U(-1,1); std::uniform_int_distribution<int> Ti(0,T_KV-1);
    std::vector<float> Q(BH*D),KV(T_KV*D),sink(BH);
    for(auto&x:Q)x=U(rng)*0.1f; for(auto&x:KV)x=U(rng)*0.1f; for(auto&x:sink)x=U(rng)*0.5f;
    std::vector<int> topk(NKK); for(auto&t:topk)t=Ti(rng);
    for(int i=0;i<ninv&&i<NKK;i++) topk[NKK-1-i]=-1;
    auto mkb=[&](const std::vector<float>&f){std::vector<__hip_bfloat16> b(f.size());
        for(size_t i=0;i<f.size();i++)b[i]=f2b(f[i]); return b;};
    auto Qb=mkb(Q),KVb=mkb(KV);
    __hip_bfloat16 *dQ,*dKV,*dO; int* dT; float *dS,*dL;
    HC(hipMalloc(&dQ,sizeof(__hip_bfloat16)*BH*D)); HC(hipMalloc(&dKV,sizeof(__hip_bfloat16)*T_KV*D));
    HC(hipMalloc(&dO,sizeof(__hip_bfloat16)*BH*D)); HC(hipMalloc(&dT,sizeof(int)*NKK));
    HC(hipMalloc(&dS,sizeof(float)*BH)); HC(hipMalloc(&dL,sizeof(float)*BH));
    {std::vector<__hip_bfloat16> sV(BH*D,(__hip_bfloat16)-999.f); HC(hipMemcpy(dO,sV.data(),sizeof(__hip_bfloat16)*BH*D,hipMemcpyHostToDevice));}
    HC(hipMemcpy(dQ,Qb.data(),sizeof(__hip_bfloat16)*BH*D,hipMemcpyHostToDevice));
    HC(hipMemcpy(dKV,KVb.data(),sizeof(__hip_bfloat16)*T_KV*D,hipMemcpyHostToDevice));
    HC(hipMemcpy(dT,topk.data(),sizeof(int)*NKK,hipMemcpyHostToDevice));
    HC(hipMemcpy(dS,sink.data(),sizeof(float)*BH,hipMemcpyHostToDevice));
    g_t g{ gl<bf16,-1,-1,-1,-1>(dQ,1,NW,QB,D), gl<bf16,-1,-1,-1,-1>(dKV,1,1,T_KV,D),
           gl<bf16,-1,-1,-1,-1>(dO,1,NW,QB,D), gl<int,-1,-1,-1,-1>(dT,1,1,NTILES,TILE_K),
           gl<float,-1,-1,-1,-1>(dS,1,1,NW,QB), gl<float,-1,-1,-1,-1>(dL,1,1,NW,QB),
           NTILES, T_KV, gs, has_sink };
    size_t shmem=4*sizeof(KS);   // 3-buffer ring at TILE_K=32 = 96KB
    HC(hipFuncSetAttribute((void*)hk_s2_occ2_stag2ab, hipFuncAttributeMaxDynamicSharedMemorySize, shmem));
    hk_s2_occ2_stag2ab<<<dim3(1),dim3(NT),shmem>>>(g); HC(hipDeviceSynchronize());
    std::vector<__hip_bfloat16> Ob(BH*D); std::vector<float> Lo(BH);
    HC(hipMemcpy(Ob.data(),dO,sizeof(__hip_bfloat16)*BH*D,hipMemcpyDeviceToHost));
    HC(hipMemcpy(Lo.data(),dL,sizeof(float)*BH,hipMemcpyDeviceToHost));
    int nanc=0,g2=0,g3=0; double esum=0; long ecnt=0; float worst=0,lworst=0;
    for(int h=0;h<BH;h++){
        std::vector<float> s(NKK); float m=-1e30f;
        for(int kk=0;kk<NKK;kk++){int kr=topk[kk]; if(kr<0){s[kk]=-1e30f;continue;} float d=0;
            for(int q=0;q<D;q++) d+=bf(Q[h*D+q])*bf(KV[kr*D+q]); s[kk]=d*scale_nat; if(s[kk]>m)m=s[kk];}
        float l=0; for(int kk=0;kk<NKK;kk++){s[kk]=(s[kk]<=-1e29f)?0.f:std::exp(s[kk]-m); l+=s[kk];}
        float denom=has_sink?l+std::exp(sink[h]-m):l; float lse_ref=m+std::log(denom);
        float le=std::fabs(lse_ref-Lo[h]); if(le>lworst)lworst=le;
        for(int v=0;v<D;v++){float a=0; for(int kk=0;kk<NKK;kk++){int kr=topk[kk]; if(kr<0)continue; a+=bf(s[kk])*bf(KV[kr*D+v]);}
            float o=a/denom,hk=b2f(Ob[h*D+v]); if(!(hk==hk)){nanc++;continue;}
            float e=std::fabs(hk-o); if(e>worst)worst=e; esum+=e; ecnt++; if(e>0.02f)g2++; if(e>0.03f)g3++;}
    }
    printf("HK_S2_OCC2_RING DC=%d nt=%d sink=%d ninv=%d seed=%d: O worst=%.5f mean=%.6f >.02=%d >.03=%d nan=%d | LSE worst=%.6f  %s\n",
           DC,NTILES,has_sink,ninv,seed,worst,ecnt?esum/ecnt:0,g2,g3,nanc,lworst,(g3==0&&nanc==0&&lworst<0.01f)?"PASS":"FAIL");
    return 0;
}
