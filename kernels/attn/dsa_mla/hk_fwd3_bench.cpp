// hk_fwd3_bench.cpp — full-grid benchmark for hk_fwd3 vs gluon (T=4096,H=128,TOPK=1152).
// grid=(T, H/BLOCK_H); times the kernel with hipEvents; spot-checks a few (token,head) outputs
// vs a bf16-faithful CPU ref to validate the grid indexing. Build with -DENABLE_MASK for the
// production (masked) path. No -ffast-math.
#include "hk_fwd3.cpp"
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
#define NTILES 36
#endif
#ifndef ATT_T
#define ATT_T 4096
#endif
#ifndef ATT_H
#define ATT_H 128
#endif

int main(int argc,char**argv){
    int seed = argc>1?atoi(argv[1]):0;
    const int T=ATT_T, H=ATT_H, HG=H/(QB*NW), TKV=ATT_T, TOPK=NTILES*TILE_K;
    if (H % (QB*NW)) { printf("H must be multiple of BLOCK_H=%d\n", QB*NW); return 1; }
    float gs=(1.f/std::sqrt((float)D_QK))*LOG2E;
    std::mt19937 rng(seed); std::uniform_real_distribution<float> U(-1,1);

    printf("CONFIG T=%d H=%d HG=%d TOPK=%d nt=%d D_V=%d D_QK=%d NW=%d\n", T,H,HG,TOPK,NTILES,D_V,D_QK,NW);

    // host buffers
    std::vector<float> Ql((size_t)T*H*D_V), Qr((size_t)T*H*D_ROPE), KV((size_t)TKV*D_QK), sink(H);
    for(auto&x:Ql)x=U(rng)*0.1f; for(auto&x:Qr)x=U(rng)*0.1f;
    for(auto&x:KV)x=U(rng)*0.1f; for(auto&x:sink)x=U(rng)*0.5f;
    // causal topk: token t selects min(TOPK, t+1) random indices in [0,t]; rest -1 (warmup pattern)
    std::vector<int> topk((size_t)T*TOPK);
    for(int t=0;t<T;t++){ int navail=t+1; std::uniform_int_distribution<int> Di(0,t<0?0:t);
        for(int k=0;k<TOPK;k++) topk[(size_t)t*TOPK+k] = (k<navail)? Di(rng) : -1; }

    auto mkb=[&](const std::vector<float>&f){std::vector<__hip_bfloat16> b(f.size());
        for(size_t i=0;i<f.size();i++)b[i]=f2b(f[i]); return b;};
    auto Qlb=mkb(Ql),Qrb=mkb(Qr),KVb=mkb(KV);
    std::vector<float> sinkw(H); for(int i=0;i<H;i++) sinkw[i]=sink[i];

    __hip_bfloat16 *dQl,*dQr,*dKV,*dO; int* dT; float* dS; float* dDdummy;
    HC(hipMalloc(&dQl,sizeof(__hip_bfloat16)*(size_t)T*H*D_V));
    HC(hipMalloc(&dQr,sizeof(__hip_bfloat16)*(size_t)T*H*D_ROPE));
    HC(hipMalloc(&dKV,sizeof(__hip_bfloat16)*(size_t)TKV*D_QK));
    HC(hipMalloc(&dO, sizeof(__hip_bfloat16)*(size_t)T*H*D_V));
    HC(hipMalloc(&dDdummy, sizeof(float)*64));
    HC(hipMalloc(&dT, sizeof(int)*(size_t)T*TOPK));
    HC(hipMalloc(&dS, sizeof(float)*H));
    HC(hipMemcpy(dQl,Qlb.data(),sizeof(__hip_bfloat16)*(size_t)T*H*D_V,hipMemcpyHostToDevice));
    HC(hipMemcpy(dQr,Qrb.data(),sizeof(__hip_bfloat16)*(size_t)T*H*D_ROPE,hipMemcpyHostToDevice));
    HC(hipMemcpy(dKV,KVb.data(),sizeof(__hip_bfloat16)*(size_t)TKV*D_QK,hipMemcpyHostToDevice));
    HC(hipMemcpy(dT, topk.data(),sizeof(int)*(size_t)T*TOPK,hipMemcpyHostToDevice));
    HC(hipMemcpy(dS, sinkw.data(),sizeof(float)*H,hipMemcpyHostToDevice));

    g_t g{ gl<bf16,-1,-1,-1,-1>(dQl,(size_t)T*HG,NW,QB,D_V),
           gl<bf16,-1,-1,-1,-1>(dQr,(size_t)T*HG,NW,QB,D_ROPE),
           gl<bf16,-1,-1,-1,-1>(dKV,1,1,TKV,D_QK),
           gl<bf16,-1,-1,-1,-1>(dO,(size_t)T*HG,NW,QB,D_V),
           gl<int,-1,-1,-1,-1>(dT,T,1,NTILES,TILE_K),
           gl<float,-1,-1,-1,-1>(dS,1,1,HG*NW,QB),
           gl<float,-1,-1,-1,-1>(dDdummy,1,1,1,64),
           NTILES, TKV, gs, 1 };

    size_t shmem=65536;
    HC(hipFuncSetAttribute((void*)hk_fwd3, hipFuncAttributeMaxDynamicSharedMemorySize, shmem));
    dim3 grid(T, HG), block(NT);

    // warmup + timing
    for(int i=0;i<5;i++) hk_fwd3<<<grid,block,shmem>>>(g);
    HC(hipDeviceSynchronize());
    hipEvent_t a,b; hipEventCreate(&a); hipEventCreate(&b);
    const int IT=20;
    hipEventRecord(a);
    for(int i=0;i<IT;i++) hk_fwd3<<<grid,block,shmem>>>(g);
    hipEventRecord(b); HC(hipEventSynchronize(b));
    float ms=0; hipEventElapsedTime(&ms,a,b); ms/=IT;
    printf("HK_FWD3 BENCH: %.4f ms/iter  (grid %dx%d, %d programs)  [gluon ref 3.10 ms]\n",
           ms, T, HG, T*HG);

    // ---- correctness spot-check on a few (token,head) ----
    std::vector<__hip_bfloat16> Ob((size_t)T*H*D_V);
    HC(hipMemcpy(Ob.data(),dO,sizeof(__hip_bfloat16)*(size_t)T*H*D_V,hipMemcpyDeviceToHost));
    int checks[][2] = {{0,0},{0,63},{100,5},{1000,64},{4000,127},{2047,100}};
    int nbad=0,nnan=0; float worst=0;
    for(auto&c:checks){ int t=c[0], h=c[1]; if(h>=H||t>=T) continue;
        int navail=(t+1<TOPK)?t+1:TOPK; float cw=0; int cb=0;
        std::vector<float> s(TOPK); float m=-1e30f;
        for(int k=0;k<TOPK;k++){ int kr=topk[(size_t)t*TOPK+k];
            if(kr<0){ s[k]=-1e30f; continue; } float d=0;
            for(int q=0;q<D_V;q++)   d+=bf(Ql[((size_t)t*H+h)*D_V+q])*bf(KV[(size_t)kr*D_QK+q]);
            for(int q=0;q<D_ROPE;q++)d+=bf(Qr[((size_t)t*H+h)*D_ROPE+q])*bf(KV[(size_t)kr*D_QK+D_V+q]);
            s[k]=d*gs; if(s[k]>m)m=s[k]; }
        float l=0; for(int k=0;k<TOPK;k++){ s[k]=(s[k]<=-1e29f)?0.f:std::exp2(s[k]-m); l+=s[k]; }
        float mf=std::fmax(m,sink[h]), afix=std::exp2(m-mf), ltot=l*afix+std::exp2(sink[h]-mf);
        for(int v=0;v<D_V;v++){ float a2=0;
            for(int k=0;k<TOPK;k++){ int kr=topk[(size_t)t*TOPK+k]; if(kr<0)continue; a2+=bf(s[k])*bf(KV[(size_t)kr*D_QK+v]); }
            float o=a2*afix/ltot, hk=b2f(Ob[((size_t)t*H+h)*D_V+v]);
            if(!(hk==hk)){nnan++;continue;} float e=std::fabs(hk-o);
            if(e>worst)worst=e; if(e>cw)cw=e; if(e>0.03f){nbad++;cb++;} }
        printf("  check t=%d h=%d navail=%d: worst=%.5f bad=%d\n", t, h, navail, cw, cb);
    }
    printf("SPOTCHECK: worst=%.5f nbad=%d nan=%d %s\n", worst, nbad, nnan, (nbad==0&&nnan==0)?"PASS":"FAIL");
    return 0;
}
