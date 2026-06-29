// mini3b_test.cpp — natural-operand (mma_ABt) keys-as-rows validator.
#include "mini3b.cpp"
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
    int seed=argc>1?atoi(argv[1]):0;
    const int D_QK=D_V+D_ROPE, BH=QB*NW;
    float gs=(1.f/std::sqrt((float)D_QK))*LOG2E;
    std::mt19937 rng(seed); std::uniform_real_distribution<float> U(-1,1);

    std::vector<float> Ql(BH*D_V),Qr(BH*D_ROPE),Kl(TILE_K*D_V),Kr(TILE_K*D_ROPE);
    for(auto&x:Ql)x=U(rng)*0.1f; for(auto&x:Qr)x=U(rng)*0.1f;
    for(auto&x:Kl)x=U(rng)*0.1f; for(auto&x:Kr)x=U(rng)*0.1f;

    // per-warp Q natural [NW,QB,D_V] / [NW,QB,D_ROPE]
    std::vector<float> Qlw(NW*QB*D_V),Qrw(NW*QB*D_ROPE);
    for(int w=0;w<NW;w++)for(int hh=0;hh<QB;hh++){int h=w*QB+hh;
        for(int d=0;d<D_V;d++)   Qlw[w*QB*D_V+hh*D_V+d]=Ql[h*D_V+d];
        for(int d=0;d<D_ROPE;d++)Qrw[w*QB*D_ROPE+hh*D_ROPE+d]=Qr[h*D_ROPE+d];}

    auto mkb=[&](const std::vector<float>&f){std::vector<__hip_bfloat16> b(f.size());
        for(size_t i=0;i<f.size();i++)b[i]=f2b(f[i]); return b;};
    auto Qlb=mkb(Qlw),Qrb=mkb(Qrw),Klb=mkb(Kl),Krb=mkb(Kr);

    __hip_bfloat16 *dQl,*dQr,*dKl,*dKr,*dO;
    HC(hipMalloc(&dQl,sizeof(__hip_bfloat16)*NW*QB*D_V));
    HC(hipMalloc(&dQr,sizeof(__hip_bfloat16)*NW*QB*D_ROPE));
    HC(hipMalloc(&dKl,sizeof(__hip_bfloat16)*TILE_K*D_V));
    HC(hipMalloc(&dKr,sizeof(__hip_bfloat16)*TILE_K*D_ROPE));
    HC(hipMalloc(&dO, sizeof(__hip_bfloat16)*BH*D_V));
    {std::vector<__hip_bfloat16> sV(BH*D_V,(__hip_bfloat16)-999.f);
     HC(hipMemcpy(dO,sV.data(),sizeof(__hip_bfloat16)*BH*D_V,hipMemcpyHostToDevice));}
    HC(hipMemcpy(dQl,Qlb.data(),sizeof(__hip_bfloat16)*NW*QB*D_V,hipMemcpyHostToDevice));
    HC(hipMemcpy(dQr,Qrb.data(),sizeof(__hip_bfloat16)*NW*QB*D_ROPE,hipMemcpyHostToDevice));
    HC(hipMemcpy(dKl,Klb.data(),sizeof(__hip_bfloat16)*TILE_K*D_V,hipMemcpyHostToDevice));
    HC(hipMemcpy(dKr,Krb.data(),sizeof(__hip_bfloat16)*TILE_K*D_ROPE,hipMemcpyHostToDevice));

    g_t g{ gl<bf16,-1,-1,-1,-1>(dQl,1,NW,QB,D_V),
           gl<bf16,-1,-1,-1,-1>(dQr,1,NW,QB,D_ROPE),
           gl<bf16,-1,-1,-1,-1>(dKl,1,1,TILE_K,D_V),
           gl<bf16,-1,-1,-1,-1>(dKr,1,1,TILE_K,D_ROPE),
           gl<bf16,-1,-1,-1,-1>(dO,1,NW,QB,D_V), gs };

    size_t shmem=16384;
    HC(hipFuncSetAttribute((void*)mini3b, hipFuncAttributeMaxDynamicSharedMemorySize, shmem));
    mini3b<<<dim3(1),dim3(NT),shmem>>>(g);
    HC(hipDeviceSynchronize());

    std::vector<__hip_bfloat16> Ob(BH*D_V);
    HC(hipMemcpy(Ob.data(),dO,sizeof(__hip_bfloat16)*BH*D_V,hipMemcpyDeviceToHost));

    int nanc=0,g2=0,g3=0,g5=0; double esum=0; long ecnt=0; float worst=0;
    for(int w=0;w<NW;w++)for(int hh=0;hh<QB;hh++){int h=w*QB+hh;
        float s[TILE_K], m=-1e30f;
        for(int k=0;k<TILE_K;k++){ float d=0;
            for(int q=0;q<D_V;q++)   d+=bf(Ql[h*D_V+q])*bf(Kl[k*D_V+q]);
            for(int q=0;q<D_ROPE;q++)d+=bf(Qr[h*D_ROPE+q])*bf(Kr[k*D_ROPE+q]);
            s[k]=d*gs; if(s[k]>m)m=s[k]; }
        float l=0; for(int k=0;k<TILE_K;k++){ s[k]=std::exp2(s[k]-m); l+=s[k]; }
        for(int v=0;v<D_V;v++){ float a=0;
            for(int k=0;k<TILE_K;k++) a+=bf(s[k])*bf(Kl[k*D_V+v]);
            float o=a/l, hk=b2f(Ob[w*QB*D_V+hh*D_V+v]);
            if(!(hk==hk)){nanc++;continue;}
            float e=std::fabs(hk-o); if(e>worst)worst=e; esum+=e; ecnt++;
            if(e>0.02f)g2++; if(e>0.03f)g3++; if(e>0.05f)g5++; }
    }
    printf("MINI3B NW=%d seed=%d: worst=%.5f mean=%.6f >.02=%d >.03=%d >.05=%d nan=%d %s\n",
           NW, seed, worst, esum/ecnt, g2, g3, g5, nanc, (g3==0&&nanc==0)?"PASS":"FAIL");
    return 0;
}
