// Piece 4 investigation: PV mma_AtB in art. acc[64,16]=V[64,64]^T.P[64,16], contraction over 64 keys.
// RESOLVED: (1) mma_AtB requires ALL operands col_l. (2) P operand must be rt_32x16 (mma K=32
//   contraction) -- rt_16x16 rejected (size/K). (3) art load CANNOT target rt_32x16 from shared
//   ("Unsupported shape", shared_to_register.cuh:108/191) -> P32 must be built by the in-register
//   permlane reshuffle from s (rt_16x16 bf16), i.e. the REG_RESHUFFLE p16->p32 recipe already
//   proven in hk_s2_occ1_stream. So piece 4 = art reshuffle(s->P32) + mma_AtB(acc,V,P32) + rescale.
// This probe loads P as rt_32x16 to isolate the mma shape check; blocked by (3) -> kept as the
// design record (compiles only up to the load). Real path validated at integration.
#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <random>
using namespace kittens;
#define HC(x) do{hipError_t e=(x);if(e){printf("err %s\n",hipGetErrorString(e));return 1;}}while(0)

// try: acc[64,16] col_l rt_16x16, V[64,64] col_l, P[64,16] col_l rt_16x16
typedef uint32_t u2 __attribute__((ext_vector_type(2)));
using ACC_r=ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<64,79>>,4>;   // 16 fp32 = 4 tiles rt_16x16
using V_r  =ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<80,111>>,4>;   // 64x64 bf16 rt_16x32 = 8 tiles x4 = 32 regs
using P_r  =ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<112,119>>,4>;  // 64x16 bf16 rt_32x16 = 2 tiles x4 = 8 regs
using ACC_art=art<float,64,16,col_l,rt_16x16_s,ACC_r>;
using V_art  =art<bf16, 64,64,col_l,rt_16x32_s,V_r>;
using P_art  =art<bf16, 64,16,col_l,rt_32x16_s,P_r>;

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

__global__ void probe(gl<bf16,-1,-1,-1,-1> gV, gl<bf16,-1,-1,-1,-1> gP, gl<float,-1,-1,-1,-1> gO){
    extern __shared__ int __shm[]; shared_allocator al((int*)&__shm[0]);
    st_bf<64,64,st_16x32_s> &Vs=al.allocate<st_bf<64,64,st_16x32_s>>();
    st_bf<64,16,st_32x16_s> &Ps=al.allocate<st_bf<64,16,st_32x16_s>>();
    load(Vs,gV,coord<>{0,0,0,0}); load(Ps,gP,coord<>{0,0,0,0}); __builtin_amdgcn_s_waitcnt(0); __syncthreads();
    ducks::art::clobber<ACC_r>(); ducks::art::clobber<V_r>(); ducks::art::clobber<P_r>();
    ACC_art acc; V_art v; P_art p;
    { auto x=subtile_inplace<64,64>(Vs,{0,0}); uint32_t a=get_address(v,x); load<0,0>(v,x,a); load<0,1>(v,x,a); load<1,0>(v,x,a); load<1,1>(v,x,a); load<2,0>(v,x,a); load<2,1>(v,x,a); load<3,0>(v,x,a); load<3,1>(v,x,a); }
    { auto x=subtile_inplace<64,16>(Ps,{0,0}); uint32_t a=get_address(p,x); load<0,0>(p,x,a); load<1,0>(p,x,a); }
    __builtin_amdgcn_s_waitcnt(0);
    zero(acc); mma_AtB(acc, v, p, acc);
    ((float*)&gO[coord<>{0,0,0,0}])[threadIdx.x] = art_col_sum(acc);  // sum over DC per query (q=L%16)
}
int main(){
    std::mt19937 rng(1); std::uniform_real_distribution<float> U(-1,1);
    std::vector<float> V(64*64), P(64*16); for(auto&x:V)x=U(rng); for(auto&x:P)x=U(rng);
    std::vector<__hip_bfloat16> Vb(64*64), Pb(64*16);
    for(size_t i=0;i<Vb.size();i++)Vb[i]=(__hip_bfloat16)V[i]; for(size_t i=0;i<Pb.size();i++)Pb[i]=(__hip_bfloat16)P[i];
    __hip_bfloat16 *dV,*dP; float* dO; HC(hipMalloc(&dV,2*64*64)); HC(hipMalloc(&dP,2*64*16)); HC(hipMalloc(&dO,4*64));
    HC(hipMemcpy(dV,Vb.data(),2*64*64,hipMemcpyHostToDevice)); HC(hipMemcpy(dP,Pb.data(),2*64*16,hipMemcpyHostToDevice));
    gl<bf16,-1,-1,-1,-1> gV(dV,1,1,64,64), gP(dP,1,1,64,16); gl<float,-1,-1,-1,-1> gO(dO,1,1,1,64);
    hipFuncSetAttribute((void*)probe, hipFuncAttributeMaxDynamicSharedMemorySize, 65536);
    probe<<<1,64,65536>>>(gV,gP,gO); HC(hipDeviceSynchronize());
    std::vector<float> Ov(64); HC(hipMemcpy(Ov.data(),dO,4*64,hipMemcpyDeviceToHost));
    double worst=0;
    for(int L=0;L<64;L++){ int q=L%16; double s=0; for(int d=0;d<64;d++){ double o=0; for(int key=0;key<64;key++) o+=(double)(float)Vb[key*64+d]*(double)(float)Pb[key*16+q]; s+=o; } worst=fmax(worst,fabs(Ov[L]-s)); }
    printf("art PV mma_AtB (V rt_16x32, P rt_32x16) col-sum worst=%.4f  %s\n", worst, worst<0.05?"PASS":"FAIL");
    return 0;
}
