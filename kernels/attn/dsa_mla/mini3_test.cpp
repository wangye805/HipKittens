// mini3_test.cpp — correctness harness for mini3.cpp (keys-as-rows single-tile fwd).
// M-split aware: BLOCK_H = QB*NW heads; per-warp Q/O laid out with NW in dim-1 (depth).
// K/V are shared across the BLOCK_H heads. bf16-faithful CPU ref; no -ffast-math.
#include "mini3.cpp"
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
    int seed = argc>1?atoi(argv[1]):0;
    const int D_QK = D_V + D_ROPE;
    const int BH = QB*NW;
    float gs = (1.f/std::sqrt((float)D_QK))*LOG2E;
    std::mt19937 rng(seed); std::uniform_real_distribution<float> U(-1,1);

    // natural layouts
    std::vector<float> Ql(BH*D_V), Qr(BH*D_ROPE), Kl(TILE_K*D_V), Kr(TILE_K*D_ROPE);
    for(auto&x:Ql)x=U(rng)*0.1f; for(auto&x:Qr)x=U(rng)*0.1f;
    for(auto&x:Kl)x=U(rng)*0.1f; for(auto&x:Kr)x=U(rng)*0.1f;

    // per-warp transposed Q: [NW, D_V, QB] and [NW, D_ROPE, QB]
    std::vector<float> QlT(NW*D_V*QB), QrT(NW*D_ROPE*QB);
    for(int w=0;w<NW;w++)for(int hh=0;hh<QB;hh++){ int h=w*QB+hh;
        for(int d=0;d<D_V;d++)    QlT[w*D_V*QB   + d*QB + hh]=Ql[h*D_V+d];
        for(int d=0;d<D_ROPE;d++) QrT[w*D_ROPE*QB+ d*QB + hh]=Qr[h*D_ROPE+d]; }
    // shared transposed K + natural V
    std::vector<float> KlT(D_V*TILE_K), KrT(D_ROPE*TILE_K);
    for(int k=0;k<TILE_K;k++){ for(int d=0;d<D_V;d++)   KlT[d*TILE_K+k]=Kl[k*D_V+d];
                               for(int d=0;d<D_ROPE;d++)KrT[d*TILE_K+k]=Kr[k*D_ROPE+d]; }
    std::vector<float>& V = Kl;  // V = K_lora natural [TILE_K, D_V]

    auto mkb=[&](const std::vector<float>&f){std::vector<__hip_bfloat16> b(f.size());
        for(size_t i=0;i<f.size();i++)b[i]=f2b(f[i]); return b;};
    auto QlTb=mkb(QlT),QrTb=mkb(QrT),KlTb=mkb(KlT),KrTb=mkb(KrT),Vb=mkb(V);

    __hip_bfloat16 *dQlT,*dQrT,*dKlT,*dKrT,*dV,*dO;
    HC(hipMalloc(&dQlT,sizeof(__hip_bfloat16)*NW*D_V*QB));
    HC(hipMalloc(&dQrT,sizeof(__hip_bfloat16)*NW*D_ROPE*QB));
    HC(hipMalloc(&dKlT,sizeof(__hip_bfloat16)*D_V*TILE_K));
    HC(hipMalloc(&dKrT,sizeof(__hip_bfloat16)*D_ROPE*TILE_K));
    HC(hipMalloc(&dV,  sizeof(__hip_bfloat16)*TILE_K*D_V));
    HC(hipMalloc(&dO,  sizeof(__hip_bfloat16)*BH*D_V));
    {std::vector<__hip_bfloat16> sV(BH*D_V,(__hip_bfloat16)-999.f);
     HC(hipMemcpy(dO,sV.data(),sizeof(__hip_bfloat16)*BH*D_V,hipMemcpyHostToDevice));}
    HC(hipMemcpy(dQlT,QlTb.data(),sizeof(__hip_bfloat16)*NW*D_V*QB,hipMemcpyHostToDevice));
    HC(hipMemcpy(dQrT,QrTb.data(),sizeof(__hip_bfloat16)*NW*D_ROPE*QB,hipMemcpyHostToDevice));
    HC(hipMemcpy(dKlT,KlTb.data(),sizeof(__hip_bfloat16)*D_V*TILE_K,hipMemcpyHostToDevice));
    HC(hipMemcpy(dKrT,KrTb.data(),sizeof(__hip_bfloat16)*D_ROPE*TILE_K,hipMemcpyHostToDevice));
    HC(hipMemcpy(dV,  Vb.data(),  sizeof(__hip_bfloat16)*TILE_K*D_V,hipMemcpyHostToDevice));

    g_t g{ gl<bf16,-1,-1,-1,-1>(dQlT,1,NW,D_V,QB),
           gl<bf16,-1,-1,-1,-1>(dQrT,1,NW,D_ROPE,QB),
           gl<bf16,-1,-1,-1,-1>(dKlT,1,1,D_V,TILE_K),
           gl<bf16,-1,-1,-1,-1>(dKrT,1,1,D_ROPE,TILE_K),
           gl<bf16,-1,-1,-1,-1>(dV,1,1,TILE_K,D_V),
           gl<bf16,-1,-1,-1,-1>(dO,1,NW,QB,D_V), gs };

    size_t shmem = 16384;
    HC(hipFuncSetAttribute((void*)mini3, hipFuncAttributeMaxDynamicSharedMemorySize, shmem));
    mini3<<<dim3(1),dim3(NT),shmem>>>(g);
    HC(hipDeviceSynchronize());

    std::vector<__hip_bfloat16> Ob(BH*D_V);
    HC(hipMemcpy(Ob.data(),dO,sizeof(__hip_bfloat16)*BH*D_V,hipMemcpyDeviceToHost));

    int bad=0,nanc=0,g2=0,g3=0,g5=0; double esum=0; long ecnt=0; float worst=0;
    for(int w=0;w<NW;w++)for(int hh=0;hh<QB;hh++){ int h=w*QB+hh;
        float s[TILE_K], m=-1e30f;
        for(int k=0;k<TILE_K;k++){ float d=0;
            for(int q=0;q<D_V;q++)    d+=bf(Ql[h*D_V+q])*bf(Kl[k*D_V+q]);
            for(int q=0;q<D_ROPE;q++) d+=bf(Qr[h*D_ROPE+q])*bf(Kr[k*D_ROPE+q]);
            s[k]=d*gs; if(s[k]>m)m=s[k]; }
        float l=0; for(int k=0;k<TILE_K;k++){ s[k]=std::exp2(s[k]-m); l+=s[k]; }
        for(int v=0;v<D_V;v++){ float a=0;
            for(int k=0;k<TILE_K;k++) a+=bf(s[k])*bf(Kl[k*D_V+v]);
            float o=a/l;
            float hk=b2f(Ob[w*QB*D_V + hh*D_V + v]);   // Og layout [NW,QB,D_V]
            if(!(hk==hk)){nanc++;continue;}
            float e=std::fabs(hk-o); if(e>worst)worst=e; esum+=e; ecnt++;
            if(e>0.02f)bad++; if(e>0.02f)g2++; if(e>0.03f)g3++; if(e>0.05f)g5++; }
    }
    printf("MINI3 NW=%d seed=%d: worst=%.5f mean=%.6f >.02=%d >.03=%d >.05=%d nan=%d %s\n",
           NW, seed, worst, esum/ecnt, g2, g3, g5, nanc, (g3==0&&nanc==0)?"PASS":"FAIL");
    return 0;
}
