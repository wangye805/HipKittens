// Piece 2 TEST (spec) — art col_max/col_sum, and it doubles as the QK-art NUMERIC check.
// Flow: QK in art (Q->AGPR) -> s[64,16] art -> art_col_max(m, s) / art_col_sum(l, s)
//       -> store m,l (small row_vecs) -> compare to CPU col_max/col_sum of K.Q^T.
// This sidesteps the "can't store col_l fp32 s" blocker: the reduction output is the storable observable.
//
// STATUS: this is the RED test. art_col_max/art_col_sum are NOT yet implemented (the fwd needs them;
// the bwd sidesteps reductions via saved LSE). Implement per the approach below, then this must PASS.
//
// ---- art_col_max/col_sum implementation approach (from assembly/maps.cuh unary_map pattern) ----
// s is art<float,64,16,col_l,rt_16x16_s,S_ranges>: height=4, width=1, 4 fp32/base-tile-reg-group.
// Per lane: 16 fp32 (4 base tiles x 4). col_max reduces over the 64 keys -> per-query (col) max.
//  (1) LOCAL: compile-time iterate S_ranges (index_sequence, like unary_map) emitting v_max_f32 on the
//      ranged registers (op::op<range::lo>()) -> 1 accum/lane.  [need a macros::v_max_f32<a,b,d> macro;
//      if absent, inline-asm "v_max_f32 v[%0],v[%1],v[%2]" with range::lo immediates.]
//  (2) CROSS-LANE: permlane32_swap then permlane16_swap butterfly on accum (same as the rt patch) ->
//      full col_max in all 4 group-lanes. No LDS, no broadcast.
//  col_sum = same with v_add_f32 (and start from 0). Output = a row_vec (or plain per-lane float).
#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <random>
using namespace kittens;
#define HC(x) do{hipError_t e=(x);if(e){printf("err %s\n",hipGetErrorString(e));return 1;}}while(0)

// (reuse the validated 2-chunk QK-art from probe_art_qk: Q0/Q1 in AGPR, K0/K1 VGPR, s VGPR)
using S_ranges  = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<0,15>>, 4>;
using K0_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<16,47>>, 4>;
using K1_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<48,79>>, 4>;
using Q0_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<256,263>>, 4>;
using Q1_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<264,271>>, 4>;
using S_art  = art<float, 64, 16, col_l, rt_16x16_s, S_ranges>;
using K0_art = art<bf16,  64, 64, row_l, rt_16x32_s, K0_ranges>;
using K1_art = art<bf16,  64, 64, row_l, rt_16x32_s, K1_ranges>;
using Q0_art = art<bf16,  16, 64, row_l, rt_16x32_s, Q0_ranges>;
using Q1_art = art<bf16,  16, 64, row_l, rt_16x32_s, Q1_ranges>;

// TODO(piece2): implement — art_col_max(float& m, const S_art& s); art_col_sum(float& l, const S_art& s);
// then: mma_ABt QK -> art_col_max(m,s); art_col_sum(l,s); store m,l per-lane; compare CPU. Must PASS.

int main(){
    printf("RED: art_col_max/art_col_sum not yet implemented — see approach in header. "
           "When implemented, this test validates QK-art + reduction numerically.\n");
    return 0;
}
