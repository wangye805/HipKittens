// hk_fwd5_test.cpp — self-contained correctness for hk_fwd5 (gather + loop + sink).
// Builds KV[T_KV,D_QK], random VALID topk[NTILES,TILE_K], per-warp Q, per-head sink.
// bf16-faithful CPU ref (gathers KV[topk]); no -ffast-math.
#include "hk_fwd5.cpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <random>
static inline __hip_bfloat16 f2b(float x){return (__hip_bfloat16)x;}
static inline float b2f(__hip_bfloat16 x){return (float)x;}
static inline float bf(float x){return (float)(__hip_bfloat16)x;}
#define HC(x) do{hipError_t e=(x);if(e){printf("err %s\n",hipGetErrorString(e));return 1;}}while(0)

#ifndef NTILES
#define NTILES 4
#endif

int main(int argc,char**argv){
    int seed=argc>1?atoi(argv[1]):0;
    int has_sink=argc>2?atoi(argv[2]):1;
    int ninv=argc>3?atoi(argv[3]):0;   // number of topk entries to set to -1 (invalid)
    const int BH=QB*NW, NKK=NTILES*TILE_K, T_KV=4096;
    float gs=(1.f/std::sqrt((float)D_QK))*LOG2E;
    std::mt19937 rng(seed); std::uniform_real_distribution<float> U(-1,1);
    std::uniform_int_distribution<int> Ti(0,T_KV-1);

    std::vector<float> Ql(BH*D_V),Qr(BH*D_ROPE),KV(T_KV*D_QK),sink(BH);
    for(auto&x:Ql)x=U(rng)*0.1f; for(auto&x:Qr)x=U(rng)*0.1f;
    for(auto&x:KV)x=U(rng)*0.1f; for(auto&x:sink)x=U(rng)*0.5f;
    std::vector<int> topk(NTILES*TILE_K);
    for(auto&t:topk)t=Ti(rng);
    for(int i=0;i<ninv && i<(int)topk.size();i++) topk[topk.size()-1-i]=-1;  // contiguous -1 TAIL (real DSA warmup pattern; large ninv -> whole trailing tiles all-invalid)

    // per-warp Q [NW,QB,D_V]/[NW,QB,D_ROPE]
    std::vector<float> Qlw(NW*QB*D_V),Qrw(NW*QB*D_ROPE);
    for(int w=0;w<NW;w++)for(int hh=0;hh<QB;hh++){int h=w*QB+hh;
        for(int d=0;d<D_V;d++)   Qlw[w*QB*D_V+hh*D_V+d]=Ql[h*D_V+d];
        for(int d=0;d<D_ROPE;d++)Qrw[w*QB*D_ROPE+hh*D_ROPE+d]=Qr[h*D_ROPE+d];}
    // per-warp sink [NW,QB]
    std::vector<float> sinkw(NW*QB);
    for(int w=0;w<NW;w++)for(int hh=0;hh<QB;hh++) sinkw[w*QB+hh]=sink[w*QB+hh];

    auto mkb=[&](const std::vector<float>&f){std::vector<__hip_bfloat16> b(f.size());
        for(size_t i=0;i<f.size();i++)b[i]=f2b(f[i]); return b;};
    auto Qlb=mkb(Qlw),Qrb=mkb(Qrw),KVb=mkb(KV);

    __hip_bfloat16 *dQl,*dQr,*dKV,*dO; int* dT; float* dS; float* dD;
    HC(hipMalloc(&dQl,sizeof(__hip_bfloat16)*NW*QB*D_V));
    HC(hipMalloc(&dQr,sizeof(__hip_bfloat16)*NW*QB*D_ROPE));
    HC(hipMalloc(&dKV,sizeof(__hip_bfloat16)*T_KV*D_QK));
    HC(hipMalloc(&dO, sizeof(__hip_bfloat16)*BH*D_V));
    HC(hipMalloc(&dT, sizeof(int)*NTILES*TILE_K));
    HC(hipMalloc(&dS, sizeof(float)*NW*QB));
    HC(hipMalloc(&dD, sizeof(float)*NW*QB*TILE_K));
    {std::vector<__hip_bfloat16> sV(BH*D_V,(__hip_bfloat16)-999.f);
     HC(hipMemcpy(dO,sV.data(),sizeof(__hip_bfloat16)*BH*D_V,hipMemcpyHostToDevice));}
    HC(hipMemcpy(dQl,Qlb.data(),sizeof(__hip_bfloat16)*NW*QB*D_V,hipMemcpyHostToDevice));
    HC(hipMemcpy(dQr,Qrb.data(),sizeof(__hip_bfloat16)*NW*QB*D_ROPE,hipMemcpyHostToDevice));
    HC(hipMemcpy(dKV,KVb.data(),sizeof(__hip_bfloat16)*T_KV*D_QK,hipMemcpyHostToDevice));
    HC(hipMemcpy(dT, topk.data(),sizeof(int)*NTILES*TILE_K,hipMemcpyHostToDevice));
    HC(hipMemcpy(dS, sinkw.data(),sizeof(float)*NW*QB,hipMemcpyHostToDevice));

    g_t g{ gl<bf16,-1,-1,-1,-1>(dQl,1,NW,QB,D_V),
           gl<bf16,-1,-1,-1,-1>(dQr,1,NW,QB,D_ROPE),
           gl<bf16,-1,-1,-1,-1>(dKV,1,1,T_KV,D_QK),
           gl<bf16,-1,-1,-1,-1>(dO,1,NW,QB,D_V),
           gl<int,-1,-1,-1,-1>(dT,1,1,NTILES,TILE_K),
           gl<float,-1,-1,-1,-1>(dS,1,1,NW,QB),
           NTILES, T_KV, gs, has_sink };

    size_t shmem=131072;
    HC(hipFuncSetAttribute((void*)hk_fwd5, hipFuncAttributeMaxDynamicSharedMemorySize, shmem));
    hk_fwd5<<<dim3(1),dim3(NT),shmem>>>(g);
    HC(hipDeviceSynchronize());

    std::vector<__hip_bfloat16> Ob(BH*D_V);
    HC(hipMemcpy(Ob.data(),dO,sizeof(__hip_bfloat16)*BH*D_V,hipMemcpyDeviceToHost));
#ifdef DBGS
    {std::vector<float> Sd(NW*QB*TILE_K);
     HC(hipMemcpy(Sd.data(),dD,sizeof(float)*NW*QB*TILE_K,hipMemcpyDeviceToHost));
     int snan=0; float sworst=0;
     for(int h=0;h<BH;h++)for(int k=0;k<TILE_K;k++){ int kr=topk[k]; float d=0;
         for(int q=0;q<D_V;q++)   d+=bf(Ql[h*D_V+q])*bf(KV[kr*D_QK+q]);
         for(int q=0;q<D_ROPE;q++)d+=bf(Qr[h*D_ROPE+q])*bf(KV[kr*D_QK+D_V+q]);
         float ref=d*gs, hk=Sd[h*TILE_K+k];
         if(!(hk==hk))snan++; else { float e=std::fabs(hk-ref); if(e>sworst)sworst=e; } }
     printf("  [DBG S tile0] nan=%d worst=%.5f\n", snan, sworst);}
#endif

    int nanc=0,g2=0,g3=0,g5=0; double esum=0; long ecnt=0; float worst=0;
    int badhead[256]={0}; int worst_h=-1,worst_v=-1; float worst_hk=0,worst_ref=0;
    for(int h=0;h<BH;h++){
        std::vector<float> s(NKK); float m=-1e30f;
        for(int kk=0;kk<NKK;kk++){ int kr=topk[kk];
            if(kr<0){ s[kk]=-1e30f; continue; }   // invalid key masked out
            float d=0;
            for(int q=0;q<D_V;q++)   d+=bf(Ql[h*D_V+q])*bf(KV[kr*D_QK+q]);
            for(int q=0;q<D_ROPE;q++)d+=bf(Qr[h*D_ROPE+q])*bf(KV[kr*D_QK+D_V+q]);
            s[kk]=d*gs; if(s[kk]>m)m=s[kk]; }
        float l=0; for(int kk=0;kk<NKK;kk++){ s[kk]=(s[kk]<=-1e29f)?0.f:std::exp2(s[kk]-m); l+=s[kk]; }
        float afix=1.f, l_tot=l;
        if(has_sink){ float mf=std::fmax(m,sink[h]); afix=std::exp2(m-mf);
                      l_tot=l*afix+std::exp2(sink[h]-mf); }
        for(int v=0;v<D_V;v++){ float a=0;
            for(int kk=0;kk<NKK;kk++){ int kr=topk[kk]; if(kr<0)continue; a+=bf(s[kk])*bf(KV[kr*D_QK+v]); }
            float o=a*afix/l_tot, hk=b2f(Ob[h*D_V+v]);   // Og layout [NW,QB,D_V]=[BH,D_V]
            if(!(hk==hk)){nanc++;continue;}
            float e=std::fabs(hk-o); if(e>worst){worst=e;worst_h=h;worst_v=v;worst_hk=hk;worst_ref=o;} esum+=e; ecnt++;
            if(e>0.02f)g2++; if(e>0.03f){g3++; badhead[h]++;} if(e>0.05f)g5++; }
    }
    { int nbh=0,firstbad=-1,lastbad=-1; for(int h=0;h<BH;h++) if(badhead[h]){nbh++; if(firstbad<0)firstbad=h; lastbad=h;}
      printf("  [DBG] worst at h=%d v=%d hk=%.4f ref=%.4f | bad-heads=%d (first=%d last=%d) | v-of-worst%%16=%d\n",
             worst_h,worst_v,worst_hk,worst_ref,nbh,firstbad,lastbad,worst_v%16); }
    printf("HK_FWD5 NW=%d nt=%d sink=%d ninv=%d seed=%d: worst=%.5f mean=%.6f >.02=%d >.03=%d >.05=%d nan=%d %s\n",
           NW, NTILES, has_sink, ninv, seed, worst, esum/ecnt, g2, g3, g5, nanc, (g3==0&&nanc==0)?"PASS":"FAIL");
    return 0;
}
