// Piece 3: softmax core in art. QK-art -> m=col_max(s) -> denom=sum exp2(scale*(s-m)) -> compare CPU.
// Validates scale + sub(m) + exp2 + col_sum fused over the art s registers (per-lane m from col_max).
#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <random>
#include <utility>
using namespace kittens;
#define HC(x) do{hipError_t e=(x);if(e){printf("err %s\n",hipGetErrorString(e));return 1;}}while(0)
typedef uint32_t u2 __attribute__((ext_vector_type(2)));

template<typename ART>
__device__ float art_col_max(const ART&){
    using RR = typename ART::register_ranges; float acc=-1e30f;
    [&]<std::size_t... Rs>(std::index_sequence<Rs...>){ ([&]<std::size_t R>(){
        constexpr int lo=ducks::art::get_nth_range_t<RR,R>::lo;
        acc=fmaxf(acc,__uint_as_float(macros::v_mov_b32_p2up<lo+0>())); acc=fmaxf(acc,__uint_as_float(macros::v_mov_b32_p2up<lo+1>()));
        acc=fmaxf(acc,__uint_as_float(macros::v_mov_b32_p2up<lo+2>())); acc=fmaxf(acc,__uint_as_float(macros::v_mov_b32_p2up<lo+3>()));
    }.template operator()<Rs>(),...); }(std::make_index_sequence<ART::height*ART::width>{});
    u2 r=__builtin_amdgcn_permlane32_swap(__float_as_uint(acc),__float_as_uint(acc),false,true); acc=fmaxf(__uint_as_float(r.x),__uint_as_float(r.y));
    u2 s=__builtin_amdgcn_permlane16_swap(__float_as_uint(acc),__float_as_uint(acc),false,true); acc=fmaxf(__uint_as_float(s.x),__uint_as_float(s.y));
    return acc;
}
// softmax denom = sum over 64 keys of exp2(scale*(s - m)); m = per-lane full col_max.
template<typename ART>
__device__ float art_softmax_denom(const ART&, float m, float scale){
    using RR = typename ART::register_ranges; float acc=0.f;
    [&]<std::size_t... Rs>(std::index_sequence<Rs...>){ ([&]<std::size_t R>(){
        constexpr int lo=ducks::art::get_nth_range_t<RR,R>::lo;
        acc+=exp2f(scale*(__uint_as_float(macros::v_mov_b32_p2up<lo+0>())-m)); acc+=exp2f(scale*(__uint_as_float(macros::v_mov_b32_p2up<lo+1>())-m));
        acc+=exp2f(scale*(__uint_as_float(macros::v_mov_b32_p2up<lo+2>())-m)); acc+=exp2f(scale*(__uint_as_float(macros::v_mov_b32_p2up<lo+3>())-m));
    }.template operator()<Rs>(),...); }(std::make_index_sequence<ART::height*ART::width>{});
    u2 r=__builtin_amdgcn_permlane32_swap(__float_as_uint(acc),__float_as_uint(acc),false,true); acc=__uint_as_float(r.x)+__uint_as_float(r.y);
    u2 s=__builtin_amdgcn_permlane16_swap(__float_as_uint(acc),__float_as_uint(acc),false,true); acc=__uint_as_float(s.x)+__uint_as_float(s.y);
    return acc;
}

using S_ranges=ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<64,79>>,4>;
using K0_ranges=ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<80,111>>,4>;
using K1_ranges=ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<112,143>>,4>;
using Q0_ranges=ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<256,263>>,4>;
using Q1_ranges=ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<264,271>>,4>;
using S_art=art<float,64,16,col_l,rt_16x16_s,S_ranges>;
using K0_art=art<bf16,64,64,row_l,rt_16x32_s,K0_ranges>; using K1_art=art<bf16,64,64,row_l,rt_16x32_s,K1_ranges>;
using Q0_art=art<bf16,16,64,row_l,rt_16x32_s,Q0_ranges>; using Q1_art=art<bf16,16,64,row_l,rt_16x32_s,Q1_ranges>;

__global__ void probe(gl<bf16,-1,-1,-1,-1> gK, gl<bf16,-1,-1,-1,-1> gQ, gl<float,-1,-1,-1,-1> gL, float scale){
    extern __shared__ int __shm[]; shared_allocator al((int*)&__shm[0]);
    st_bf<64,128,st_16x32_s> &Ks=al.allocate<st_bf<64,128,st_16x32_s>>();
    st_bf<16,128,st_16x32_s> &Qs=al.allocate<st_bf<16,128,st_16x32_s>>();
    load(Ks,gK,coord<>{0,0,0,0}); load(Qs,gQ,coord<>{0,0,0,0}); __builtin_amdgcn_s_waitcnt(0); __syncthreads();
    ducks::art::clobber<S_ranges>(); ducks::art::clobber<K0_ranges>(); ducks::art::clobber<K1_ranges>(); ducks::art::clobber<Q0_ranges>(); ducks::art::clobber<Q1_ranges>();
    S_art s; K0_art k0; K1_art k1; Q0_art q0; Q1_art q1;
    { auto q=subtile_inplace<16,64>(Qs,{0,0}); uint32_t a=get_address(q0,q); load<0,0>(q0,q,a); load<0,1>(q0,q,a); }
    { auto q=subtile_inplace<16,64>(Qs,{0,1}); uint32_t a=get_address(q1,q); load<0,0>(q1,q,a); load<0,1>(q1,q,a); }
    { auto k=subtile_inplace<64,64>(Ks,{0,0}); uint32_t a=get_address(k0,k); load<0,0>(k0,k,a); load<0,1>(k0,k,a); load<1,0>(k0,k,a); load<1,1>(k0,k,a); load<2,0>(k0,k,a); load<2,1>(k0,k,a); load<3,0>(k0,k,a); load<3,1>(k0,k,a); }
    { auto k=subtile_inplace<64,64>(Ks,{0,1}); uint32_t a=get_address(k1,k); load<0,0>(k1,k,a); load<0,1>(k1,k,a); load<1,0>(k1,k,a); load<1,1>(k1,k,a); load<2,0>(k1,k,a); load<2,1>(k1,k,a); load<3,0>(k1,k,a); load<3,1>(k1,k,a); }
    __builtin_amdgcn_s_waitcnt(0);
    zero(s); mma_ABt(s,k0,q0,s); mma_ABt(s,k1,q1,s);
    float m = art_col_max(s);
    float l = art_softmax_denom(s, m, scale);
    ((float*)&gL[coord<>{0,0,0,0}])[threadIdx.x] = l;
}
int main(){
    std::mt19937 rng(0); std::uniform_real_distribution<float> U(-1,1);
    std::vector<float> K(64*128),Q(16*128); for(auto&x:K)x=U(rng); for(auto&x:Q)x=U(rng);
    std::vector<__hip_bfloat16> Kb(64*128),Qb(16*128);
    for(size_t i=0;i<Kb.size();i++)Kb[i]=(__hip_bfloat16)K[i]; for(size_t i=0;i<Qb.size();i++)Qb[i]=(__hip_bfloat16)Q[i];
    float scale=0.12f;
    __hip_bfloat16 *dK,*dQ; float* dL; HC(hipMalloc(&dK,2*64*128)); HC(hipMalloc(&dQ,2*16*128)); HC(hipMalloc(&dL,4*64));
    HC(hipMemcpy(dK,Kb.data(),2*64*128,hipMemcpyHostToDevice)); HC(hipMemcpy(dQ,Qb.data(),2*16*128,hipMemcpyHostToDevice));
    gl<bf16,-1,-1,-1,-1> gK(dK,1,1,64,128), gQ(dQ,1,1,16,128); gl<float,-1,-1,-1,-1> gL(dL,1,1,1,64);
    hipFuncSetAttribute((void*)probe, hipFuncAttributeMaxDynamicSharedMemorySize, 65536);
    probe<<<1,64,65536>>>(gK,gQ,gL,scale); HC(hipDeviceSynchronize());
    std::vector<float> Lv(64); HC(hipMemcpy(Lv.data(),dL,4*64,hipMemcpyDeviceToHost));
    double worst=0;
    for(int L=0;L<64;L++){ int q=L%16; double mx=-1e30; for(int key=0;key<64;key++){ double v=0; for(int c=0;c<128;c++) v+=(double)(float)Kb[key*128+c]*(double)(float)Qb[q*128+c]; if(v>mx)mx=v; }
        double d=0; for(int key=0;key<64;key++){ double v=0; for(int c=0;c<128;c++) v+=(double)(float)Kb[key*128+c]*(double)(float)Qb[q*128+c]; d+=exp2((double)scale*(v-mx)); }
        worst=fmax(worst,fabs(Lv[L]-d)); }
    printf("art softmax denom worst=%.4f  %s\n", worst, worst<0.05?"PASS":"FAIL");
    return 0;
}
