#ifndef QLO
#define QLO 256
#endif
#ifndef WW
#define WW 0
#endif
#ifndef CC
#define CC 0
#endif
// Isolate the Q global->AGPR art load. K from shared (proven), Q from GLOBAL into AGPR (the test).
// s = K.Q^T via art mma_ABt; validate col_max/col_sum vs CPU. Iterate coord/axis until it matches.
#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <random>
using namespace kittens;
#define HC(x) do{hipError_t e=(x);if(e){printf("err %s\n",hipGetErrorString(e));return 1;}}while(0)
typedef uint32_t u2 __attribute__((ext_vector_type(2)));
constexpr int DP=64;   // K/Q inner dim (2 mma k-steps)

using S_r=ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<64,79>>,4>;   // fp32 rt_16x16, 4 tiles
using K_r=ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<80,111>>,4>;  // bf16 rt_16x32, 8 tiles (K[64,64])
using Qfull_r=ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<256,319>>,4>; // whole Q[16,512] -> AGPR (16 tiles)
using Q3_r=ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<QLO,QLO+7>>,4>;
using S_art=art<float,64,16,col_l,rt_16x16_s,S_r>;
using K_art=art<bf16, 64,DP,row_l,rt_16x32_s,K_r>;
using Qfull_art=art<bf16, 16,512,row_l,rt_16x32_s,Qfull_r>;
using Q3_art=art<bf16, 16,DP,row_l,rt_16x32_s,Q3_r>;

using KcT = rt<bf16, 64, DP, row_l, rt_16x32_s>;   // normal rt K (like the kernel)
__device__ static inline void art_k_from_rt(const KcT& src, K_art&){
    using KR = K_art::register_ranges; constexpr int W = KcT::width;
    [&]<std::size_t... Ts>(std::index_sequence<Ts...>){ ([&]<std::size_t T>(){
        constexpr int lo = ducks::art::get_nth_range_t<KR,T>::lo; constexpr int n=T/W, m=T%W;
        macros::v_mov_b32_up2p<lo+0>(*reinterpret_cast<const uint32_t*>(&src.tiles[n][m].data[0]));
        macros::v_mov_b32_up2p<lo+1>(*reinterpret_cast<const uint32_t*>(&src.tiles[n][m].data[1]));
        macros::v_mov_b32_up2p<lo+2>(*reinterpret_cast<const uint32_t*>(&src.tiles[n][m].data[2]));
        macros::v_mov_b32_up2p<lo+3>(*reinterpret_cast<const uint32_t*>(&src.tiles[n][m].data[3]));
    }.template operator()<Ts>(),...); }(std::make_index_sequence<KcT::height*KcT::width>{});
}
template<typename ART> __device__ float art_col_sum(const ART&){
    using RR=typename ART::register_ranges; float acc=0.f;
    [&]<std::size_t... Rs>(std::index_sequence<Rs...>){ ([&]<std::size_t R>(){
        constexpr int lo=ducks::art::get_nth_range_t<RR,R>::lo;
        acc+=__uint_as_float(macros::v_mov_b32_p2up<lo+0>()); acc+=__uint_as_float(macros::v_mov_b32_p2up<lo+1>());
        acc+=__uint_as_float(macros::v_mov_b32_p2up<lo+2>()); acc+=__uint_as_float(macros::v_mov_b32_p2up<lo+3>());
    }.template operator()<Rs>(),...); }(std::make_index_sequence<ART::height*ART::width>{});
    u2 r=__builtin_amdgcn_permlane32_swap(__float_as_uint(acc),__float_as_uint(acc),false,true); acc=__uint_as_float(r.x)+__uint_as_float(r.y);
    u2 s=__builtin_amdgcn_permlane16_swap(__float_as_uint(acc),__float_as_uint(acc),false,true); acc=__uint_as_float(s.x)+__uint_as_float(s.y);
    return acc;
}

__global__ void probe(gl<bf16,-1,-1,-1,-1> gK, gl<bf16,-1,-1,-1,-1> gQ, gl<float,-1,-1,-1,-1> gO){
    extern __shared__ int __shm[]; shared_allocator al((int*)&__shm[0]);
    st_bf<64,DP,st_32x32_s> &Ks=al.allocate<st_bf<64,DP,st_32x32_s>>();
    load(Ks,gK,coord<>{0,0,0,0}); __builtin_amdgcn_s_waitcnt(0); __syncthreads();
    ducks::art::clobber<S_r>(); ducks::art::clobber<K_r>(); ducks::art::clobber<Qfull_r>();
    S_art s; K_art k; Qfull_art qfull; Q3_art q;  // q aliases chunk-3 sub-range of qfull
#ifdef KBRIDGE
    // K via normal-rt load + art_k_from_rt bridge (the kernel's path under test)
    { KcT kc; auto x=subtile_inplace<64,DP>(Ks,{0,0}); load(kc, x); asm volatile("s_waitcnt lgkmcnt(0)"); art_k_from_rt(kc, k); }
#else
    // K from shared art-load (proven path)
    { auto x=subtile_inplace<64,DP>(Ks,{0,0}); uint32_t a=get_address(k,x); load<0,0>(k,x,a); load<0,1>(k,x,a); load<1,0>(k,x,a); load<1,1>(k,x,a); load<2,0>(k,x,a); load<2,1>(k,x,a); load<3,0>(k,x,a); load<3,1>(k,x,a); }
#endif
    // Q from GLOBAL into AGPR: load WHOLE [16,512] once (full width, like bwd).
    load(qfull, gQ, coord<>{0,WW,0,0}, coord<>{0,0,0,0});
    __builtin_amdgcn_s_waitcnt(0);
    // Custom chunked QK: s += K_chunk . Qfull[:, chunk c]^T via mma_ABt_base with INDEPENDENT A/B ranges.
    // K_art [64,64]=8 tiles (h4xw2): tile (N,klocal)=N*2+klocal. Qfull [16,512]=16 tiles (h1xw16): global
    // k-tile = 2*c+klocal (M=0). s [64,16]=4 tiles (h4xw1): tile N. c = chunk index (CC).
    zero(s);
    constexpr int c = CC;
    [&]<std::size_t... Ns>(std::index_sequence<Ns...>){ ([&]<std::size_t N>(){
        { using rA=ducks::art::get_nth_range_t<K_art::register_ranges, N*2+0>;
          using rB=ducks::art::get_nth_range_t<Qfull_art::register_ranges, 2*c+0>;
          using rCD=ducks::art::get_nth_range_t<S_art::register_ranges, N>;
          mma_ABt_base<typename S_art::shape, bf16, rA, rB, rCD, rCD>(); }
        { using rA=ducks::art::get_nth_range_t<K_art::register_ranges, N*2+1>;
          using rB=ducks::art::get_nth_range_t<Qfull_art::register_ranges, 2*c+1>;
          using rCD=ducks::art::get_nth_range_t<S_art::register_ranges, N>;
          mma_ABt_base<typename S_art::shape, bf16, rA, rB, rCD, rCD>(); }
    }.template operator()<Ns>(),...); }(std::make_index_sequence<4>{});
    ((float*)&gO[coord<>{0,0,0,0}])[threadIdx.x] = art_col_sum(s);  // sum over keys per query? col_sum over rows(keys) -> per query q=L%16
}
int main(){
    // Q layout matches the kernel: [1, ND=4 warps, 16 queries, DW=512 cols]. We load warp W=2, chunk C=3.
    constexpr int ND=4, DW=512, W=WW, C=CC;
    std::mt19937 rng(2); std::uniform_real_distribution<float> U(-1,1);
    std::vector<float> K(64*DP), Qfull(ND*16*DW); for(auto&x:K)x=U(rng); for(auto&x:Qfull)x=U(rng);
    std::vector<__hip_bfloat16> Kb(64*DP), Qb(ND*16*DW);
    for(size_t i=0;i<Kb.size();i++)Kb[i]=(__hip_bfloat16)K[i]; for(size_t i=0;i<Qb.size();i++)Qb[i]=(__hip_bfloat16)Qfull[i];
    // CPU Q slice used by the mma = Qb[W, qi, C*64 + c]
    auto Qsl=[&](int qi,int c)->float{ return (float)Qb[((size_t)W*16 + qi)*DW + C*DP + c]; };
    __hip_bfloat16 *dK,*dQ; float* dO; HC(hipMalloc(&dK,2*64*DP)); HC(hipMalloc(&dQ,2*ND*16*DW)); HC(hipMalloc(&dO,4*64));
    HC(hipMemcpy(dK,Kb.data(),2*64*DP,hipMemcpyHostToDevice)); HC(hipMemcpy(dQ,Qb.data(),2*ND*16*DW,hipMemcpyHostToDevice));
    gl<bf16,-1,-1,-1,-1> gK(dK,1,1,64,DP), gQ(dQ,1,ND,16,DW); gl<float,-1,-1,-1,-1> gO(dO,1,1,1,64);
    hipFuncSetAttribute((void*)probe, hipFuncAttributeMaxDynamicSharedMemorySize, 65536);
    probe<<<1,64,65536>>>(gK,gQ,gO); HC(hipDeviceSynchronize());
    std::vector<float> Ov(64); HC(hipMemcpy(Ov.data(),dO,4*64,hipMemcpyDeviceToHost));
    for(int wcpu=0; wcpu<ND; wcpu++) for(int ccpu=0; ccpu<DW/DP; ccpu++){
      double w2=0;
      for(int L=0;L<64;L++){ int qi=L%16; double d=0;
        for(int key=0;key<64;key++){ double v=0; for(int c=0;c<DP;c++) v+=(double)(float)Kb[key*DP+c]*(double)(float)Qb[((size_t)wcpu*16+qi)*DW + ccpu*DP + c]; d+=v; }
        w2=fmax(w2,fabs(Ov[L]-d)); }
      if(w2<0.1) printf("  MATCH: kernel actually read warp=%d chunk=%d (asked W=%d C=%d)\n", wcpu, ccpu, WW, CC);
    }
    double worst=0;
    for(int L=0;L<64;L++){ int qi=L%16; double d=0; for(int key=0;key<64;key++){ double v=0; for(int c=0;c<DP;c++) v+=(double)(float)Kb[key*DP+c]*(double)Qsl(qi,c); d+=v; } worst=fmax(worst,fabs(Ov[L]-d)); }
    printf("Q-global-load QK col-sum worst=%.4f  %s\n", worst, worst<0.1?"PASS":"FAIL");
    return 0;
}
