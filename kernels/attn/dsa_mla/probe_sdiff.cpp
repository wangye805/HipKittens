// Element-wise compare: art-QK s (Q/K in AGPR, chunked mma_ABt_base) vs normal-QK s (VGPR, mma_ABt),
// full D=512, same Q,K. Isolates the per-element art-vs-normal QK difference behind the LSE bias.
#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <random>
using namespace kittens;
#define HC(x) do{hipError_t e=(x);if(e){printf("err %s\n",hipGetErrorString(e));return 1;}}while(0)
constexpr int QB=16, D=512, TILE_K=64, DC=64, NCH=D/DC;

template<int GPR> __device__ __forceinline__ void v_accvgpr_write(uint32_t val){
    asm volatile("v_accvgpr_write_b32 a[%0], %1" :: "n"(GPR-256), "v"(val)); }

using Qf_r=ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<256,319>>,4>;
using Kc_r=ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<320,351>>,4>;
using Sc_r=ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<352,367>>,4>;
using Qf_art=art<bf16,QB,D,row_l,rt_16x32_s,Qf_r>;
using Kc_art=art<bf16,TILE_K,DC,row_l,rt_16x32_s,Kc_r>;
using Sc_art=art<float,TILE_K,QB,col_l,rt_16x16_s,Sc_r>;
using KcT=rt<bf16,TILE_K,DC,row_l,rt_16x32_s>;
using QcT=rt<bf16,QB,DC,row_l,rt_16x32_s>;
using ST_=rt<float,TILE_K,QB,col_l,rt_16x16_s>;
using KS=st_bf<TILE_K,D,st_32x32_s>;

__device__ static inline void art_k_from_rt(const KcT& src, Kc_art&){
    using KR=Kc_art::register_ranges; constexpr int W=KcT::width;
    [&]<std::size_t...Ts>(std::index_sequence<Ts...>){ ([&]<std::size_t T>(){
        constexpr int lo=ducks::art::get_nth_range_t<KR,T>::lo; constexpr int n=T/W,m=T%W;
        v_accvgpr_write<lo+0>(*reinterpret_cast<const uint32_t*>(&src.tiles[n][m].data[0]));
        v_accvgpr_write<lo+1>(*reinterpret_cast<const uint32_t*>(&src.tiles[n][m].data[1]));
        v_accvgpr_write<lo+2>(*reinterpret_cast<const uint32_t*>(&src.tiles[n][m].data[2]));
        v_accvgpr_write<lo+3>(*reinterpret_cast<const uint32_t*>(&src.tiles[n][m].data[3]));
    }.template operator()<Ts>(),...); }(std::make_index_sequence<KcT::height*KcT::width>{});
}
__device__ static inline void art_s_to_rt(const Sc_art&, ST_& d){
    using SR=Sc_art::register_ranges;
    [&]<std::size_t...Ts>(std::index_sequence<Ts...>){ ([&]<std::size_t T>(){
        constexpr int lo=ducks::art::get_nth_range_t<SR,T>::lo;
        *reinterpret_cast<uint32_t*>(&d.tiles[T][0].data[0])=macros::v_mov_b32_p2up<lo+0>();
        *reinterpret_cast<uint32_t*>(&d.tiles[T][0].data[1])=macros::v_mov_b32_p2up<lo+1>();
        *reinterpret_cast<uint32_t*>(&d.tiles[T][0].data[2])=macros::v_mov_b32_p2up<lo+2>();
        *reinterpret_cast<uint32_t*>(&d.tiles[T][0].data[3])=macros::v_mov_b32_p2up<lo+3>();
    }.template operator()<Ts>(),...); }(std::make_index_sequence<ST_::height*ST_::width>{});
}
template<int c> __device__ static inline void qk_chunk(Sc_art&, const Kc_art&){
    [&]<std::size_t...Ns>(std::index_sequence<Ns...>){ ([&]<std::size_t N>(){
        { using rA=ducks::art::get_nth_range_t<Kc_art::register_ranges,N*2+0>; using rB=ducks::art::get_nth_range_t<Qf_art::register_ranges,2*c+0>; using rCD=ducks::art::get_nth_range_t<Sc_art::register_ranges,N>;
          if constexpr(c==0) mma_ABt_base_zero_accum<typename Sc_art::shape,bf16,rA,rB,rCD>(); else mma_ABt_base<typename Sc_art::shape,bf16,rA,rB,rCD,rCD>(); }
        { using rA=ducks::art::get_nth_range_t<Kc_art::register_ranges,N*2+1>; using rB=ducks::art::get_nth_range_t<Qf_art::register_ranges,2*c+1>; using rCD=ducks::art::get_nth_range_t<Sc_art::register_ranges,N>;
          mma_ABt_base<typename Sc_art::shape,bf16,rA,rB,rCD,rCD>(); }
    }.template operator()<Ns>(),...); }(std::make_index_sequence<TILE_K/16>{});
}

typedef uint32_t u2 __attribute__((ext_vector_type(2)));
template<typename ART> __device__ float art_col_max(const ART&){
    using RR=typename ART::register_ranges; float acc=-1e30f;
    [&]<std::size_t...Rs>(std::index_sequence<Rs...>){ ([&]<std::size_t R>(){ constexpr int lo=ducks::art::get_nth_range_t<RR,R>::lo;
        acc=fmaxf(acc,__uint_as_float(macros::v_mov_b32_p2up<lo+0>())); acc=fmaxf(acc,__uint_as_float(macros::v_mov_b32_p2up<lo+1>()));
        acc=fmaxf(acc,__uint_as_float(macros::v_mov_b32_p2up<lo+2>())); acc=fmaxf(acc,__uint_as_float(macros::v_mov_b32_p2up<lo+3>()));
    }.template operator()<Rs>(),...); }(std::make_index_sequence<ART::height*ART::width>{});
    u2 r=__builtin_amdgcn_permlane32_swap(__float_as_uint(acc),__float_as_uint(acc),false,true); acc=fmaxf(__uint_as_float(r.x),__uint_as_float(r.y));
    u2 s=__builtin_amdgcn_permlane16_swap(__float_as_uint(acc),__float_as_uint(acc),false,true); acc=fmaxf(__uint_as_float(s.x),__uint_as_float(s.y)); return acc; }
template<typename ART> __device__ float art_denom(const ART&, float m, float sc){
    using RR=typename ART::register_ranges; float acc=0.f;
    [&]<std::size_t...Rs>(std::index_sequence<Rs...>){ ([&]<std::size_t R>(){ constexpr int lo=ducks::art::get_nth_range_t<RR,R>::lo;
        acc+=exp2f(sc*(__uint_as_float(macros::v_mov_b32_p2up<lo+0>())-m)); acc+=exp2f(sc*(__uint_as_float(macros::v_mov_b32_p2up<lo+1>())-m));
        acc+=exp2f(sc*(__uint_as_float(macros::v_mov_b32_p2up<lo+2>())-m)); acc+=exp2f(sc*(__uint_as_float(macros::v_mov_b32_p2up<lo+3>())-m));
    }.template operator()<Rs>(),...); }(std::make_index_sequence<ART::height*ART::width>{});
    u2 r=__builtin_amdgcn_permlane32_swap(__float_as_uint(acc),__float_as_uint(acc),false,true); acc=__uint_as_float(r.x)+__uint_as_float(r.y);
    u2 s=__builtin_amdgcn_permlane16_swap(__float_as_uint(acc),__float_as_uint(acc),false,true); acc=__uint_as_float(s.x)+__uint_as_float(s.y); return acc; }

__global__ void probe_norm(gl<bf16,-1,-1,-1,-1> gQ, gl<bf16,-1,-1,-1,-1> gK, gl<float,-1,-1,-1,-1> gSn){
    extern __shared__ int __shm[]; shared_allocator al((int*)&__shm[0]);
    KS &ks=al.allocate<KS>();
    load(ks,gK,coord<>{0,0,0,0}); __builtin_amdgcn_s_waitcnt(0); __syncthreads();
    QcT q_arr[NCH];
    #pragma unroll
    for(int c=0;c<NCH;c++) load(q_arr[c], gQ, coord<>{0,0,0,c});
    __builtin_amdgcn_s_waitcnt(0);
    ST_ s_norm; zero(s_norm);
    #pragma unroll
    for(int c=0;c<NCH;c++){ KcT k_c; typename KS::template subtile<TILE_K,DC> ksub(ks,{0,c}); load(k_c,ksub); asm volatile("s_waitcnt lgkmcnt(0)"); mma_ABt(s_norm,k_c,q_arr[c],s_norm); }
    store(gSn, s_norm, coord<>{0,0,0,0});
}
__global__ void probe_art(gl<bf16,-1,-1,-1,-1> gQ, gl<bf16,-1,-1,-1,-1> gK, gl<float,-1,-1,-1,-1> gSa){
    extern __shared__ int __shm[]; shared_allocator al((int*)&__shm[0]);
    KS &ks=al.allocate<KS>();
    load(ks,gK,coord<>{0,0,0,0}); __builtin_amdgcn_s_waitcnt(0); __syncthreads();
    ducks::art::clobber<Qf_r>(); ducks::art::clobber<Kc_r>(); ducks::art::clobber<Sc_r>();
    Qf_art qfull; Kc_art k_art; Sc_art sart;
    const int uw=__builtin_amdgcn_readfirstlane(0);
    load(qfull, gQ, coord<>{0,uw,0,0}, coord<>{0,0,0,0});
    __builtin_amdgcn_s_waitcnt(0);
    #define QKC(c) { KcT k_c; typename KS::template subtile<TILE_K,DC> ksub(ks,{0,(c)}); load(k_c,ksub); asm volatile("s_waitcnt lgkmcnt(0)"); art_k_from_rt(k_c,k_art); qk_chunk<(c)>(sart,k_art); }
    QKC(0) QKC(1) QKC(2) QKC(3) QKC(4) QKC(5) QKC(6) QKC(7)
    #undef QKC
    // denom straight from art s: m=col_max, denom=sum exp2(sc*(s-m)); store per lane (query=lane%16)
    const float sc=0.1f;
    float m=art_col_max(sart); float dn=art_denom(sart, m, sc);
    ((float*)&gSa[coord<>{0,0,0,0}])[threadIdx.x]=dn;
}
int main(){
    std::mt19937 rng(7); std::uniform_real_distribution<float> U(-1,1);
    std::vector<float> Q(QB*D), K(TILE_K*D); for(auto&x:Q)x=U(rng); for(auto&x:K)x=U(rng);
    std::vector<__hip_bfloat16> Qb(QB*D), Kb(TILE_K*D);
    for(size_t i=0;i<Qb.size();i++)Qb[i]=(__hip_bfloat16)Q[i]; for(size_t i=0;i<Kb.size();i++)Kb[i]=(__hip_bfloat16)K[i];
    __hip_bfloat16 *dQ,*dK; float *dSn,*dSa; HC(hipMalloc(&dQ,2*QB*D)); HC(hipMalloc(&dK,2*TILE_K*D)); HC(hipMalloc(&dSn,4*TILE_K*QB)); HC(hipMalloc(&dSa,4*TILE_K*QB));
    HC(hipMemcpy(dQ,Qb.data(),2*QB*D,hipMemcpyHostToDevice)); HC(hipMemcpy(dK,Kb.data(),2*TILE_K*D,hipMemcpyHostToDevice));
    gl<bf16,-1,-1,-1,-1> gQ(dQ,1,1,QB,D), gK(dK,1,1,TILE_K,D); gl<float,-1,-1,-1,-1> gSn(dSn,1,1,TILE_K,QB), gSa(dSa,1,1,TILE_K,QB);
    hipFuncSetAttribute((void*)probe_norm, hipFuncAttributeMaxDynamicSharedMemorySize, 65536);
    hipFuncSetAttribute((void*)probe_art, hipFuncAttributeMaxDynamicSharedMemorySize, 65536);
    probe_art<<<1,64,65536>>>(gQ,gK,gSa); HC(hipDeviceSynchronize());
    std::vector<float> Sa(TILE_K*QB); HC(hipMemcpy(Sa.data(),dSa,4*TILE_K*QB,hipMemcpyDeviceToHost));
    // art denom per lane (query=lane%16) vs CPU denom(query) = sum exp2(0.1*(s[key][q]-max_key s))
    const float sc=0.1f; double dworst=0, dbias=0;
    for(int L=0;L<64;L++){ int q=L%16; double mx=-1e30; for(int key=0;key<TILE_K;key++){ double t=0; for(int c=0;c<D;c++)t+=(double)(float)Kb[key*D+c]*(double)(float)Qb[q*D+c]; if(t>mx)mx=t; }
        double dcpu=0; for(int key=0;key<TILE_K;key++){ double t=0; for(int c=0;c<D;c++)t+=(double)(float)Kb[key*D+c]*(double)(float)Qb[q*D+c]; dcpu+=exp2((double)sc*(t-mx)); }
        double lk=Sa[L]; double ld=log((double)lk)-log(dcpu); dworst=fmax(dworst,fabs(ld)); dbias+=ld; }
    printf("art denom vs CPU: worst |log ratio|=%.5f  mean signed log=%+.5f  (Sa[0]=%.3f)\n", dworst, dbias/64, Sa[0]);
    return 0;
    std::vector<float> Sn(TILE_K*QB);
    // element-wise art vs normal
    double worst_an=0, sum_an=0, sum_signed=0; int n=TILE_K*QB;
    for(int i=0;i<n;i++){ double d=Sa[i]-Sn[i]; worst_an=fmax(worst_an,fabs(d)); sum_an+=fabs(d); sum_signed+=d; }
    // both vs CPU (element layout: gSn[key][q] after store -> [TILE_K,QB] row-major)
    double worst_ncpu=0, worst_acpu=0;
    for(int key=0;key<TILE_K;key++) for(int q=0;q<QB;q++){ double t=0; for(int c=0;c<D;c++) t+=(double)(float)Kb[key*D+c]*(double)(float)Qb[q*D+c];
        worst_ncpu=fmax(worst_ncpu,fabs(Sn[key*QB+q]-t)); worst_acpu=fmax(worst_acpu,fabs(Sa[key*QB+q]-t)); }
    // CPU s[0][q] for q=0..3
    double cpu00=0; for(int c=0;c<D;c++) cpu00+=(double)(float)Kb[0*D+c]*(double)(float)Qb[0*D+c];
    printf("samples: Sn[0..3]=%.3f %.3f %.3f %.3f | Sa[0..3]=%.3f %.3f %.3f %.3f | cpu[0][0]=%.3f\n",
           Sn[0],Sn[1],Sn[2],Sn[3], Sa[0],Sa[1],Sa[2],Sa[3], cpu00);
    printf("s: art-vs-normal worst=%.5f mean|d|=%.6f signed_mean=%+.6f | normal-vs-CPU worst=%.4f  art-vs-CPU worst=%.4f\n",
           worst_an, sum_an/n, sum_signed/n, worst_ncpu, worst_acpu);
    return 0;
}
