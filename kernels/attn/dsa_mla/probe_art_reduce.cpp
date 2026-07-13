// Piece 2 impl+test: art col_max / col_sum, validated on QK-art output (Q->AGPR).
// s[64,16] = K0.Q0^T + K1.Q1^T (art, Q in AGPR) -> art_col_max/col_sum -> store per-lane -> compare CPU.
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

// ---- art reductions over col_l fp32 s: local reduce over the lane's regs, then permlane32+16 butterfly ----
template<typename ART>
__device__ float art_col_max(const ART&){
    using RR = typename ART::register_ranges;
    float acc = -1e30f;
    [&]<std::size_t... Rs>(std::index_sequence<Rs...>){
        ([&]<std::size_t R>(){
            constexpr int lo = ducks::art::get_nth_range_t<RR, R>::lo;
            acc = fmaxf(acc, __uint_as_float(macros::v_mov_b32_p2up<lo+0>()));
            acc = fmaxf(acc, __uint_as_float(macros::v_mov_b32_p2up<lo+1>()));
            acc = fmaxf(acc, __uint_as_float(macros::v_mov_b32_p2up<lo+2>()));
            acc = fmaxf(acc, __uint_as_float(macros::v_mov_b32_p2up<lo+3>()));
        }.template operator()<Rs>(), ...);
    }(std::make_index_sequence<ART::height*ART::width>{});
    u2 r32 = __builtin_amdgcn_permlane32_swap(__float_as_uint(acc),__float_as_uint(acc),false,true);
    acc = fmaxf(__uint_as_float(r32.x), __uint_as_float(r32.y));
    u2 r16 = __builtin_amdgcn_permlane16_swap(__float_as_uint(acc),__float_as_uint(acc),false,true);
    acc = fmaxf(__uint_as_float(r16.x), __uint_as_float(r16.y));
    return acc;
}
template<typename ART>
__device__ float art_col_sum(const ART&){
    using RR = typename ART::register_ranges;
    float acc = 0.f;
    [&]<std::size_t... Rs>(std::index_sequence<Rs...>){
        ([&]<std::size_t R>(){
            constexpr int lo = ducks::art::get_nth_range_t<RR, R>::lo;
            acc += __uint_as_float(macros::v_mov_b32_p2up<lo+0>());
            acc += __uint_as_float(macros::v_mov_b32_p2up<lo+1>());
            acc += __uint_as_float(macros::v_mov_b32_p2up<lo+2>());
            acc += __uint_as_float(macros::v_mov_b32_p2up<lo+3>());
        }.template operator()<Rs>(), ...);
    }(std::make_index_sequence<ART::height*ART::width>{});
    u2 r32 = __builtin_amdgcn_permlane32_swap(__float_as_uint(acc),__float_as_uint(acc),false,true);
    acc = __uint_as_float(r32.x) + __uint_as_float(r32.y);
    u2 r16 = __builtin_amdgcn_permlane16_swap(__float_as_uint(acc),__float_as_uint(acc),false,true);
    acc = __uint_as_float(r16.x) + __uint_as_float(r16.y);
    return acc;
}

using S_ranges  = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<64,79>>, 4>;
using K0_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<80,111>>, 4>;
using K1_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<112,143>>, 4>;
using Q0_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<256,263>>, 4>;
using Q1_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<264,271>>, 4>;
using S_art  = art<float, 64, 16, col_l, rt_16x16_s, S_ranges>;
using K0_art = art<bf16,  64, 64, row_l, rt_16x32_s, K0_ranges>;
using K1_art = art<bf16,  64, 64, row_l, rt_16x32_s, K1_ranges>;
using Q0_art = art<bf16,  16, 64, row_l, rt_16x32_s, Q0_ranges>;
using Q1_art = art<bf16,  16, 64, row_l, rt_16x32_s, Q1_ranges>;

__global__ void probe(gl<bf16,-1,-1,-1,-1> gK, gl<bf16,-1,-1,-1,-1> gQ, gl<float,-1,-1,-1,-1> gM, gl<float,-1,-1,-1,-1> gL){
    extern __shared__ int __shm[]; shared_allocator al((int*)&__shm[0]);
    st_bf<64,128,st_16x32_s> &Ks = al.allocate<st_bf<64,128,st_16x32_s>>();
    st_bf<16,128,st_16x32_s> &Qs = al.allocate<st_bf<16,128,st_16x32_s>>();
    load(Ks, gK, coord<>{0,0,0,0}); load(Qs, gQ, coord<>{0,0,0,0});
    __builtin_amdgcn_s_waitcnt(0); __syncthreads();
    ducks::art::clobber<S_ranges>(); ducks::art::clobber<K0_ranges>(); ducks::art::clobber<K1_ranges>();
    ducks::art::clobber<Q0_ranges>(); ducks::art::clobber<Q1_ranges>();
    S_art s; K0_art k0; K1_art k1; Q0_art q0; Q1_art q1;
    { auto q=subtile_inplace<16,64>(Qs,{0,0}); uint32_t a=get_address(q0,q); load<0,0>(q0,q,a); load<0,1>(q0,q,a); }
    { auto q=subtile_inplace<16,64>(Qs,{0,1}); uint32_t a=get_address(q1,q); load<0,0>(q1,q,a); load<0,1>(q1,q,a); }
    { auto k=subtile_inplace<64,64>(Ks,{0,0}); uint32_t a=get_address(k0,k); load<0,0>(k0,k,a); load<0,1>(k0,k,a); load<0,2>(k0,k,a); load<0,3>(k0,k,a); load<0,4>(k0,k,a); load<0,5>(k0,k,a); load<0,6>(k0,k,a); load<0,7>(k0,k,a); }
    { auto k=subtile_inplace<64,64>(Ks,{0,1}); uint32_t a=get_address(k1,k); load<0,0>(k1,k,a); load<0,1>(k1,k,a); load<0,2>(k1,k,a); load<0,3>(k1,k,a); load<0,4>(k1,k,a); load<0,5>(k1,k,a); load<0,6>(k1,k,a); load<0,7>(k1,k,a); }
    __builtin_amdgcn_s_waitcnt(0);
    zero(s); mma_ABt(s,k0,q0,s); mma_ABt(s,k1,q1,s);
    float m = art_col_max(s);
    float l = art_col_sum(s);
    ((float*)&gM[coord<>{0,0,0,0}])[threadIdx.x] = m;
    ((float*)&gL[coord<>{0,0,0,0}])[threadIdx.x] = l;
}
int main(){
    std::mt19937 rng(0); std::uniform_real_distribution<float> U(-1,1);
    std::vector<float> K(64*128), Q(16*128); for(auto&x:K)x=U(rng); for(auto&x:Q)x=U(rng);
    std::vector<__hip_bfloat16> Kb(64*128),Qb(16*128);
    for(size_t i=0;i<Kb.size();i++)Kb[i]=(__hip_bfloat16)K[i]; for(size_t i=0;i<Qb.size();i++)Qb[i]=(__hip_bfloat16)Q[i];
    __hip_bfloat16 *dK,*dQ; float *dM,*dL;
    HC(hipMalloc(&dK,2*64*128)); HC(hipMalloc(&dQ,2*16*128)); HC(hipMalloc(&dM,4*64)); HC(hipMalloc(&dL,4*64));
    HC(hipMemcpy(dK,Kb.data(),2*64*128,hipMemcpyHostToDevice)); HC(hipMemcpy(dQ,Qb.data(),2*16*128,hipMemcpyHostToDevice));
    gl<bf16,-1,-1,-1,-1> gK(dK,1,1,64,128), gQ(dQ,1,1,16,128); gl<float,-1,-1,-1,-1> gM(dM,1,1,1,64), gL(dL,1,1,1,64);
    hipFuncSetAttribute((void*)probe, hipFuncAttributeMaxDynamicSharedMemorySize, 65536);
    probe<<<1,64,65536>>>(gK,gQ,gM,gL); HC(hipDeviceSynchronize());
    std::vector<float> M(64),Lv(64); HC(hipMemcpy(M.data(),dM,4*64,hipMemcpyDeviceToHost)); HC(hipMemcpy(Lv.data(),dL,4*64,hipMemcpyDeviceToHost));
    // CPU: s[key][q]=sum_c K[key][c]*Q[q][c] (contraction 128); col_max/col_sum over 64 keys per query q
    double worstM=0,worstL=0;
    for(int L=0;L<64;L++){ int q=L%16;
        double mx=-1e30,sm=0;
        for(int key=0;key<64;key++){ double s=0; for(int c=0;c<128;c++) s+=(double)(float)Kb[key*128+c]*(double)(float)Qb[q*128+c]; if(s>mx)mx=s; sm+=s; }
        worstM=fmax(worstM,fabs(M[L]-mx)); worstL=fmax(worstL,fabs(Lv[L]-sm));
    }
    printf("art_col_max worst=%.4f  art_col_sum worst=%.4f  %s\n", worstM, worstL, (worstM<0.1&&worstL<0.5)?"PASS":"FAIL");
    return 0;
}
