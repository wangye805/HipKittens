#include "kittens.cuh"
using namespace kittens;

constexpr int D = 64;
constexpr int WARP_SIZE_KV = 64;
constexpr int BLOCK_SIZE_KV = 256;
constexpr int DOT_SLICE_QO = 16;
#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)
using G = kittens::group<NUM_WARPS>;

struct globals {
    gl<bf16,  -1, -1, -1, -1> Kg;    // (1,1,256,64)  K  (KV x D)
    gl<bf16,  -1, -1, -1, -1> dPg;   // (1,1,256,16)  dP (KV x Q)
    gl<float, -1, -1, -1, -1> dQog;  // (1,64,1,16) [b][warp*Q][h][Dslice]
    dim3 grid()  { return dim3(1, 1, 1); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 100000; }
};

__launch_bounds__(NUM_THREADS, 1)
__global__ __attribute__((amdgpu_num_vgpr(29))) void dq_mma_ker(const globals g) {
    const int warpid = kittens::warpid();

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE_KV, D, st_16x16_s> (&K_j_smem) = al.allocate<st_bf<BLOCK_SIZE_KV, D, st_16x16_s>>();
    st_bf<BLOCK_SIZE_KV, DOT_SLICE_QO, st_16x16_swizzled_s> (&attn_i_smem) = al.allocate<st_bf<BLOCK_SIZE_KV, DOT_SLICE_QO, st_16x16_swizzled_s>>();

    G::load<1, false>(K_j_smem,  g.Kg,  {0, 0, 0, 0});
    G::load<1, false>(attn_i_smem, g.dPg, {0, 0, 0, 0});
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
    __syncthreads();

    using K_ranges  = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<256, 287>>, 4>;
    using dS_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<30, 61>>, 4>;
    using dQ_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<110, 113>>, 4>;
    ducks::art::clobber<K_ranges>();
    ducks::art::clobber<dS_ranges>();
    ducks::art::clobber<dQ_ranges>();

    art<bf16, 256, 16, col_l, rt_32x16_4_s, K_ranges>  K_j_col;
    art<bf16, 256, 16, col_l, rt_32x16_4_s, dS_ranges> dP_col_T;
    art<float, 16, 16, col_l, rt_16x16_s, dQ_ranges> dQ_i_T;
    art<float, 16, 16, row_l, rt_16x16_s, ducks::art::transpose_2d<dQ_ranges, 1, 1>> dQ_i;

    const uint32_t K_j_col_addr = [&] {
        const int laneid = kittens::laneid();
        const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<256, 16>(K_j_smem, {0, warpid}).data[0]);
        const int row_offset = (laneid % 16) / 4 + (laneid / 16) * 4;
        const int col_offset = ((laneid % 4) * 4);
        const int lane_byte_offset = (row_offset * 16 + col_offset) * sizeof(bf16);
        return src_ptr + lane_byte_offset;
    }();
    const uint32_t dP_col_T_addr = get_address(dP_col_T, attn_i_smem);

    load<0, 0>(K_j_col, subtile_inplace<256, 16>(K_j_smem, {0, warpid}), K_j_col_addr);
    load<1, 0>(K_j_col, subtile_inplace<256, 16>(K_j_smem, {0, warpid}), K_j_col_addr);
    load<2, 0>(K_j_col, subtile_inplace<256, 16>(K_j_smem, {0, warpid}), K_j_col_addr);
    load<3, 0>(K_j_col, subtile_inplace<256, 16>(K_j_smem, {0, warpid}), K_j_col_addr);
    load<4, 0>(K_j_col, subtile_inplace<256, 16>(K_j_smem, {0, warpid}), K_j_col_addr);
    load<5, 0>(K_j_col, subtile_inplace<256, 16>(K_j_smem, {0, warpid}), K_j_col_addr);
    load<6, 0>(K_j_col, subtile_inplace<256, 16>(K_j_smem, {0, warpid}), K_j_col_addr);
    load<7, 0>(K_j_col, subtile_inplace<256, 16>(K_j_smem, {0, warpid}), K_j_col_addr);
    load<0, 0>(dP_col_T, attn_i_smem, dP_col_T_addr);
    load<1, 0>(dP_col_T, attn_i_smem, dP_col_T_addr);
    load<2, 0>(dP_col_T, attn_i_smem, dP_col_T_addr);
    load<3, 0>(dP_col_T, attn_i_smem, dP_col_T_addr);
    load<4, 0>(dP_col_T, attn_i_smem, dP_col_T_addr);
    load<5, 0>(dP_col_T, attn_i_smem, dP_col_T_addr);
    load<6, 0>(dP_col_T, attn_i_smem, dP_col_T_addr);
    load<7, 0>(dP_col_T, attn_i_smem, dP_col_T_addr);
    asm volatile("s_waitcnt lgkmcnt(0)");

    mma_AtB<0, 0, 0>(dQ_i_T, K_j_col, dP_col_T);
    mma_AtB<0, 0, 1>(dQ_i_T, K_j_col, dP_col_T, dQ_i_T);
    mma_AtB<0, 0, 2>(dQ_i_T, K_j_col, dP_col_T, dQ_i_T);
    mma_AtB<0, 0, 3>(dQ_i_T, K_j_col, dP_col_T, dQ_i_T);
    mma_AtB<0, 0, 4>(dQ_i_T, K_j_col, dP_col_T, dQ_i_T);
    mma_AtB<0, 0, 5>(dQ_i_T, K_j_col, dP_col_T, dQ_i_T);
    mma_AtB<0, 0, 6>(dQ_i_T, K_j_col, dP_col_T, dQ_i_T);
    mma_AtB<0, 0, 7>(dQ_i_T, K_j_col, dP_col_T, dQ_i_T);

    store<1>(g.dQog, dQ_i_T, {0, 0, 0, 0}, {0, warpid, 0, 0});
}

void dispatch(globals g) {
    unsigned long mem = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)dq_mma_ker, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    dq_mma_ker<<<g.grid(), g.block(), mem>>>(g);
}

#include "pyutils/pyutils.cuh"
PYBIND11_MODULE(dq_mma_test, m) {
    m.doc() = "isolated dQ mma test";
    py::bind_function<dispatch>(m, "dispatch",
        &globals::Kg, &globals::dPg, &globals::dQog);
}
