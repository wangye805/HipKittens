/**
 * @file
 * @brief MXFP8 block scale loading and packing utilities.
 *
 * Provides functions for staging E8M0 block scales in LDS and packing them
 * into fp8e8m0_4 registers for use with scaled MFMA instructions.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../memory/util/util.cuh"   // llvm_amdgcn_raw_buffer_load_lds, make_srsrc

namespace kittens {

/**
 * @brief Load iteration-major packed E8M0 scales from global memory into LDS.
 *
 * First 256 threads each load one uint32 (4 packed E8M0 bytes) for A and B.
 * A scales are placed at smem[0..1023], B scales at smem[1024..2047].
 *
 * @param smem_scales LDS buffer, must be >= 2048 bytes.
 * @param scale_A_iter Iteration-major A scales: [k_iter * M + row] as uint32.
 * @param scale_B_iter Iteration-major B scales: [k_iter * N + row] as uint32.
 * @param block_m Starting row offset for A within the current block.
 * @param block_n Starting row offset for B within the current block.
 * @param k_iter Current K iteration index.
 * @param M_dim M dimension of the matrix.
 * @param N_dim N dimension of the matrix.
 */
__device__ __forceinline__ void load_scales_to_lds(
    uint8_t *smem_scales,
    const uint32_t *__restrict__ scale_A_iter,
    const uint32_t *__restrict__ scale_B_iter,
    int block_m, int block_n, int k_iter, int M_dim, int N_dim) {
    int tid = threadIdx.x;
    if (tid < 256) {
        uint32_t sa = scale_A_iter[k_iter * M_dim + block_m + tid];
        uint32_t sb = scale_B_iter[k_iter * N_dim + block_n + tid];
        *(uint32_t *)&smem_scales[tid * 4] = sa;
        *(uint32_t *)&smem_scales[1024 + tid * 4] = sb;
    }
}

/**
 * @brief Async DMA-prefetch of E8M0 scales from global memory directly into LDS.
 *
 * Uses buffer_load_dwordx4 ... lds (HBM → LDS direct, no VGPR temps). Compared
 * to load_scales_to_lds, this primitive:
 *   - Doesn't insert a compiler-generated s_waitcnt vmcnt(N) inline (no VGPR
 *     dependency chain). The wave issues and continues immediately.
 *   - Uses 0 temp VGPRs per thread (vs ~4-6 for load_scales_to_lds).
 *   - Distributes work to 2 warps (warp 0 for A, warp 4 for B), one per slot,
 *     vs load_scales_to_lds's 4 warps (warps 0-3 all on slot 0).
 *
 * Layout in LDS is identical to load_scales_to_lds:
 *   smem_scales[0..1023]:    A scales (256 dwords, indexed by scale-row)
 *   smem_scales[1024..2047]: B scales (256 dwords)
 *
 * Caller is responsible for ensuring an s_waitcnt lgkmcnt(0) drain happens
 * between this prefetch and any subsequent read of smem_scales (e.g. via
 * pack_scales). The DMA's LDS-write portion is tracked by lgkmcnt.
 *
 * Per-warp instruction count: 1 buffer_load_dwordx4 lds on warps 0 and 4.
 * Other warps execute s_cbranch_execz and skip.
 *
 * @param smem_scales LDS buffer, must be >= 2048 bytes.
 * @param scale_A_iter Iteration-major A scales: [k_iter * M + row] as uint32.
 * @param scale_B_iter Iteration-major B scales: [k_iter * N + row] as uint32.
 * @param block_m Starting row offset for A within the current block.
 * @param block_n Starting row offset for B within the current block.
 * @param k_iter Current K iteration index.
 * @param M_dim M dimension of the matrix.
 * @param N_dim N dimension of the matrix.
 */
__device__ __forceinline__ void prefetch_scales_to_lds_dma(
    uint8_t *smem_scales,
    const uint32_t *__restrict__ scale_A_iter,
    const uint32_t *__restrict__ scale_B_iter,
    int block_m, int block_n, int k_iter, int M_dim, int N_dim) {

    // Each warp's buffer_load_dwordx4 with lds modifier reads 16 bytes per lane
    // from HBM and writes them contiguously to LDS at (lds_base + lane*16).
    // 64 lanes × 16 bytes = 1024 bytes per warp issue — exactly the size of
    // one scale array (256 uint32 = 1024 bytes).
    //
    // Warp 0 loads A scales (1024 B → smem_scales[0..1023]).
    // Warp 4 loads B scales (1024 B → smem_scales[1024..2047]).
    // Other warps skip.

    int warp_id = threadIdx.x / WARP_THREADS;     // 0..7
    int lane_id = threadIdx.x % WARP_THREADS;     // 0..63

    if (warp_id == 0) {
        // A scales: source = scale_A_iter + k_iter * M_dim + block_m
        const uint32_t *src = scale_A_iter + (size_t)k_iter * M_dim + block_m;
        // Build buffer resource: 1024 bytes range (4-byte format, 256 records).
        i32x4 srd = make_srsrc((const void*)src, 1024);
        // LDS destination: smem_scales[0..1023]
        uintptr_t lds_addr = reinterpret_cast<uintptr_t>(smem_scales);
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);
        // Per-lane voffset = lane_id * 16 (each lane reads 16 contiguous bytes)
        llvm_amdgcn_raw_buffer_load_lds(
            srd,
            lds_ptr,
            16,                  // bytes per lane
            lane_id * 16,        // per-lane voffset (VGPR)
            0,                   // soffset
            0,                   // instruction offset (compile-time)
            0                    // aux / cache coherency
        );
    } else if (warp_id == 4) {
        // B scales: source = scale_B_iter + k_iter * N_dim + block_n
        const uint32_t *src = scale_B_iter + (size_t)k_iter * N_dim + block_n;
        i32x4 srd = make_srsrc((const void*)src, 1024);
        // LDS destination: smem_scales[1024..2047]
        uintptr_t lds_addr = reinterpret_cast<uintptr_t>(smem_scales + 1024);
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);
        llvm_amdgcn_raw_buffer_load_lds(
            srd,
            lds_ptr,
            16,
            lane_id * 16,
            0,
            0,
            0
        );
    }
}

/**
 * @brief Pack 4 E8M0 scale bytes from LDS into one fp8e8m0_4 register.
 *
 * Each lane (r16 = laneid%16, k_sub = laneid/16) loads 4 dwords from
 * consecutive 16-row groups, then uses v_perm_b32 to extract byte k_sub
 * from each, producing the packed scale register for scaled MFMA.
 *
 * @param smem_scales LDS pointer to scale region.
 * @param lds_base Byte offset within smem_scales (0 for A, 1024 for B).
 * @param row_offset Starting row within the scale region (warp's tile offset).
 * @return fp8e8m0_4 with 4 scale bytes packed for MFMA opsel.
 */
__device__ __forceinline__ fp8e8m0_4 pack_scales(
    const uint8_t *smem_scales, int lds_base, int row_offset) {
    int lid   = laneid();
    int r16   = lid % 16;
    int k_sub = lid / 16;

    const uint32_t *s32 = (const uint32_t *)(smem_scales + lds_base);
    uint32_t w0 = s32[row_offset + 0 * 16 + r16];
    uint32_t w1 = s32[row_offset + 1 * 16 + r16];
    uint32_t w2 = s32[row_offset + 2 * 16 + r16];
    uint32_t w3 = s32[row_offset + 3 * 16 + r16];

    uint32_t sel = 0x0C0C0000u | (k_sub << 8) | (4u + k_sub);
    uint32_t lo  = __builtin_amdgcn_perm(w0, w1, sel);
    uint32_t hi  = __builtin_amdgcn_perm(w2, w3, sel);

    return (fp8e8m0_4)(lo | (hi << 16));
}
} // namespace kittens
