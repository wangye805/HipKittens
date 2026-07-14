#include "kittens.cuh"
using namespace kittens;
// Decisive: HK's buffer_load macro (known-good for VGPR) with GPR_START=256 -> does HBM->AGPR assemble?
__global__ void k(const int* p, int* out){
    uint64_t base=(uint64_t)p;
    buffer_resource br = make_buffer_resource(base, 1u<<20, 0x00020000);
    uint32_t voff=threadIdx.x*16;
#ifndef TRY_AGPR
    macros::buffer_load_dwordx4<0>(br, voff, 0, 0);    // -> v[0:3]  (known good)
#else
    macros::buffer_load_dwordx4<256>(br, voff, 0, 0);  // -> a[0:3]  (the question)
#endif
    out[threadIdx.x]=0;
}
int main(){ return 0; }
