// Piece 4 (refined): art only for QK (Q->AGPR). Bridge art s(fp32) -> normal rt, then reuse the
// EXISTING normal-rt PV path (copy->p16_to_p32->mma_AtB). Validates the realistic integration.
#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <random>
using namespace kittens;
#define HC(x) do{hipError_t e=(x);if(e){printf("err %s\n",hipGetErrorString(e));return 1;}}while(0)
typedef uint32_t u2 __attribute__((ext_vector_type(2)));
constexpr int DP=32;

using S_r=ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<64,79>>,4>;   // fp32 rt_16x16, 4 tiles
using K_r=ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<80,95>>,4>;   // bf16 rt_16x32, 4 tiles
using Q_r=ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<256,259>>,4>; // bf16 rt_16x32, 1 tile -> AGPR
using S_art=art<float,64,16,col_l,rt_16x16_s,S_r>;
using K_art=art<bf16, 64,DP,row_l,rt_16x32_s,K_r>;
using Q_art=art<bf16, 16,DP,row_l,rt_16x32_s,Q_r>;
// normal rt for the PV path (VGPR, compiler-allocated) — identical to the current kernel's types
using St  =rt<float,64,16,col_l,rt_16x16_s>;
using PbT =rt<bf16, 64,16,col_l,rt_16x16_s>;
using PopT=rt<bf16, 64,16,col_l,rt_32x16_s>;
using Vt  =rt<bf16, 64,64,col_l,rt_32x16_s>;
using AccT=rt<float,64,16,col_l,rt_16x16_s>;

__device__ static inline void p16_to_p32(PopT &dst, const PbT &src){
    #pragma unroll
    for(int i=0;i<PopT::height;i++)
        #pragma unroll
        for(int k=0;k<2;k++){
            uint32_t A=*reinterpret_cast<const uint32_t*>(&src.tiles[2*i][0].data[k]);
            uint32_t B=*reinterpret_cast<const uint32_t*>(&src.tiles[2*i+1][0].data[k]);
            u2 t=__builtin_amdgcn_permlane32_swap(A,B,false,true);
            u2 r=__builtin_amdgcn_permlane16_swap(t.x,t.y,false,true);
            *reinterpret_cast<uint32_t*>(&dst.tiles[i][0].data[k])   = r.x;
            *reinterpret_cast<uint32_t*>(&dst.tiles[i][0].data[k+2]) = r.y;
        }
}
// bridge: art s (fp32 rt_16x16) -> normal rt St (same logical layout), element-wise via p2up.
__device__ inline void art_to_rt(const S_art&, St& d){
    using SR=S_art::register_ranges;
    [&]<std::size_t... Ts>(std::index_sequence<Ts...>){ ([&]<std::size_t T>(){
        constexpr int lo=ducks::art::get_nth_range_t<SR,T>::lo;
        *reinterpret_cast<uint32_t*>(&d.tiles[T][0].data[0])=macros::v_mov_b32_p2up<lo+0>();
        *reinterpret_cast<uint32_t*>(&d.tiles[T][0].data[1])=macros::v_mov_b32_p2up<lo+1>();
        *reinterpret_cast<uint32_t*>(&d.tiles[T][0].data[2])=macros::v_mov_b32_p2up<lo+2>();
        *reinterpret_cast<uint32_t*>(&d.tiles[T][0].data[3])=macros::v_mov_b32_p2up<lo+3>();
    }.template operator()<Ts>(),...); }(std::make_index_sequence<St::height*St::width>{});
}

__global__ void probe(gl<bf16,-1,-1,-1,-1> gK, gl<bf16,-1,-1,-1,-1> gQ, gl<bf16,-1,-1,-1,-1> gV, gl<float,-1,-1,-1,-1> gO){
    extern __shared__ int __shm[]; shared_allocator al((int*)&__shm[0]);
    st_bf<64,DP,st_16x32_s> &Ks=al.allocate<st_bf<64,DP,st_16x32_s>>();
    st_bf<16,DP,st_16x32_s> &Qs=al.allocate<st_bf<16,DP,st_16x32_s>>();
    st_bf<64,64,st_32x32_s> &Vs=al.allocate<st_bf<64,64,st_32x32_s>>();
    load(Ks,gK,coord<>{0,0,0,0}); load(Qs,gQ,coord<>{0,0,0,0}); load(Vs,gV,coord<>{0,0,0,0}); __builtin_amdgcn_s_waitcnt(0); __syncthreads();
    ducks::art::clobber<S_r>(); ducks::art::clobber<K_r>(); ducks::art::clobber<Q_r>();
    S_art s; K_art k; Q_art q;
    { auto x=subtile_inplace<16,DP>(Qs,{0,0}); uint32_t a=get_address(q,x); load<0,0>(q,x,a); }
    { auto x=subtile_inplace<64,DP>(Ks,{0,0}); uint32_t a=get_address(k,x); load<0,0>(k,x,a); load<1,0>(k,x,a); load<2,0>(k,x,a); load<3,0>(k,x,a); }
    __builtin_amdgcn_s_waitcnt(0);
    zero(s); mma_ABt(s,k,q,s);          // QK in art: Q -> AGPR
    St s_n; art_to_rt(s, s_n);          // bridge art -> normal rt
#ifdef BRIDGE_ONLY
    store(gO, s_n, coord<>{0,0,0,0});   // validate the bridge alone (s = K.Q^T)
#else
    PbT pb; copy(pb, s_n);              // fp32 -> bf16 (existing rt copy)
    PopT pop; p16_to_p32(pop, pb);      // existing in-register reshuffle
    Vt v;
#ifndef NOLOAD
    { auto vsub = subtile_inplace<64,64>(Vs,{0,0}); load(v, vsub); }
#endif
    AccT acc; zero(acc);
#ifndef NOMMA
    mma_AtB(acc, v, pop, acc);   // normal-rt PV
#endif
    store(gO, acc, coord<>{0,0,0,0});
#endif
}
int main(){
    std::mt19937 rng(3); std::uniform_real_distribution<float> U(-1,1);
    std::vector<float> K(64*DP),Q(16*DP),V(64*64); for(auto&x:K)x=U(rng); for(auto&x:Q)x=U(rng); for(auto&x:V)x=U(rng);
    std::vector<__hip_bfloat16> Kb(64*DP),Qb(16*DP),Vb(64*64);
    for(size_t i=0;i<Kb.size();i++)Kb[i]=(__hip_bfloat16)K[i]; for(size_t i=0;i<Qb.size();i++)Qb[i]=(__hip_bfloat16)Q[i]; for(size_t i=0;i<Vb.size();i++)Vb[i]=(__hip_bfloat16)V[i];
    __hip_bfloat16 *dK,*dQ,*dV; float* dO; HC(hipMalloc(&dK,2*64*DP)); HC(hipMalloc(&dQ,2*16*DP)); HC(hipMalloc(&dV,2*64*64)); HC(hipMalloc(&dO,4*64*16));
    HC(hipMemcpy(dK,Kb.data(),2*64*DP,hipMemcpyHostToDevice)); HC(hipMemcpy(dQ,Qb.data(),2*16*DP,hipMemcpyHostToDevice)); HC(hipMemcpy(dV,Vb.data(),2*64*64,hipMemcpyHostToDevice));
    gl<bf16,-1,-1,-1,-1> gK(dK,1,1,64,DP), gQ(dQ,1,1,16,DP), gV(dV,1,1,64,64); gl<float,-1,-1,-1,-1> gO(dO,1,1,64,16);
    hipFuncSetAttribute((void*)probe, hipFuncAttributeMaxDynamicSharedMemorySize, 65536);
    probe<<<1,64,65536>>>(gK,gQ,gV,gO); HC(hipDeviceSynchronize());
    std::vector<float> Ov(64*16); HC(hipMemcpy(Ov.data(),dO,4*64*16,hipMemcpyDeviceToHost));
    double worst=0;
#ifdef BRIDGE_ONLY
    printf("BRIDGE_ONLY: kernel ran without GPU fault\n"); return 0;
#endif
    for(int d=0;d<64;d++) for(int qi=0;qi<16;qi++){ double o=0;
        for(int key=0;key<64;key++){ double sv=0; for(int c=0;c<DP;c++) sv+=(double)(float)Kb[key*DP+c]*(double)(float)Qb[qi*DP+c];
            float pbf=(float)(__hip_bfloat16)(float)sv; o+=(double)(float)Vb[key*64+d]*(double)pbf; }
        worst=fmax(worst,fabs(Ov[d*16+qi]-o)); }
    printf("art->rt bridge + normal PV worst=%.4f  %s\n", worst, worst<0.1?"PASS":"FAIL");
    return 0;
}
