// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <string.h>

#include "dma.h"
#include "encoding.h"
#include "runtime.h"
#include "synchronization.h"
#include "builtins_v2.h"

#include "data_encoder_f16.h"

#include "baremetal/mempool_checks.h"
#include "baremetal/mempool_matmul_f16.h"
#include "baremetal/mempool_layernorm_f16.h"
#include "baremetal/mempool_softmax_f16.h"

/*
======================
Parameters and defines

INNER: When defined runs inner product based matmul.
OUTER: When defined runs outer product based matmul.
*/
#define OUTER

__fp16 matrix_input[dim_s * dim_e]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));
__fp16 matrix_Wqkv[dim_e * dim_e]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));
__fp16 matrix_qkv[dim_s * dim_e]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));
__fp16 matrix_k[dim_s * dim_e]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));
__fp16 matrix_a[dim_s * dim_s]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));
__fp16 matrix_s[dim_s * dim_s]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));
__fp16 matrix_o[dim_s * dim_h]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));
__fp16 matrix_result[dim_s * dim_e]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  //uint32_t dim_h = dim_e / num_heads;
  uint32_t num_heads = dim_e / dim_h;
  
  // Initialize barrier and synchronize
  mempool_barrier_init(core_id);

  
#if defined(OUTER)
  mempool_start_benchmark();
// QKV GENERATION -----------------
// 1) Generate matrix K
  // Initialize matrices
  if (core_id == 0) {
    // Store matrices from l2 to l1 memory
    dma_memcpy_blocking(matrix_input, l2_Input,
                        (dim_s * dim_e) * sizeof(int16_t));
    dma_memcpy_blocking(matrix_Wqkv, l2_Wk,
                        (dim_e * dim_e) * sizeof(int16_t));
  }
  mempool_barrier(num_cores);

  // Matmul based on outer product
  matmul_4x2_parallel_outer_f16vec(matrix_input, matrix_Wqkv, matrix_qkv, dim_s, 
                             dim_e, dim_e, core_id, num_cores);
  mempool_barrier(num_cores);

  // Store matrices
  if (core_id == 0) {
    // Store resulting matrix from l1 to l2 memory
    dma_memcpy_blocking(l2_K, matrix_qkv, (dim_s * dim_e) * sizeof(int16_t));
  }
  mempool_barrier(num_cores);

// 2) Generate matrix Q
  // Initialize matrices
  if (core_id == 0) {
    // Store matrix from l2 to l1 memory (replace previous l2_Wk with l2_Wq)
    dma_memcpy_blocking(matrix_Wqkv, l2_Wq,
                        (dim_e * dim_e) * sizeof(int16_t));
  }
  mempool_barrier(num_cores);

  // Matmul based on inner product
  matmul_4x2_parallel_outer_f16vec(matrix_input, matrix_Wqkv, matrix_qkv, dim_s, 
                             dim_e, dim_e, core_id, num_cores);
  mempool_barrier(num_cores);

  // Store matrices
  if (core_id == 0) {
    // Store resulting matrix from l1 to l2 memory
    dma_memcpy_blocking(l2_Q, matrix_qkv, (dim_s * dim_e) * sizeof(int16_t));
  }
  mempool_barrier(num_cores);

// 3) Generate matrix V
  // Initialize matrices
  if (core_id == 0) {
    dma_memcpy_blocking(matrix_Wqkv, l2_Wv,
                        (dim_e * dim_e) * sizeof(int16_t));
  }
  mempool_barrier(num_cores);

  // Matmul based on inner product
  matmul_4x2_parallel_outer_f16vec(matrix_input, matrix_Wqkv, matrix_qkv, dim_s, 
                             dim_e, dim_e, core_id, num_cores);
  mempool_barrier(num_cores);
  
  // Store matrices
  if (core_id == 0) {
    // Store resulting matrix from l1 to l2 memory
    dma_memcpy_blocking(l2_V, matrix_qkv, (dim_s * dim_e) * sizeof(int16_t));
  }
  mempool_barrier(num_cores);

// ATTENTION HEAD SPLITTING -----------
  // Number of heads = 16
  // Process attention heads in parallel
  uint32_t core_per_head = num_cores / num_heads;
  uint32_t head_id = core_id / core_per_head;
  uint32_t local_core_id = core_id % core_per_head;

  uint32_t offset = head_id * dim_h;

  // ATTENTION MATRIX GENERATION --------
  if (core_id == 0) {
    dma_memcpy_blocking(matrix_qkv, l2_Q,
                        (dim_s * dim_e) * sizeof(int16_t));
  }
  mempool_barrier(num_cores);
  // TRANSPOSE
    // Focus is on the speed-up gained, not the correctness
    // Use the same matrix K for simplicity
  if (core_id == 0) {
    dma_memcpy_blocking(matrix_k, l2_K,
                        (dim_s * dim_e) * sizeof(int16_t));
  }
  mempool_barrier(num_cores);

  __fp16* q_head = matrix_qkv + offset;        // shape: [dim_s][dim_h]
  __fp16* k_head = matrix_k   + offset;        // shape: [dim_s][dim_h]

  // Output for the current attention head
  __fp16* o_head = matrix_o + offset;          // shape: [dim_s][dim_h]

  // Matmul based on inner product
  matmul_4x2_parallel_outer_f16vec(q_head, k_head, matrix_a, dim_s, dim_h,
                             dim_s, local_core_id, core_per_head);
  mempool_barrier(num_cores);

  // SOFTMAX APPLICATION --------------
  softmax_parallel_2x4_f16vec(matrix_a, matrix_s, dim_s, dim_s,
                             local_core_id, core_per_head);
  mempool_barrier(num_cores);

  if (core_id == 0) {
    dma_memcpy_blocking(matrix_qkv, l2_V,
                        (dim_s * dim_e) * sizeof(int16_t));
  }
  mempool_barrier(num_cores);

  __fp16* v_head = matrix_qkv + offset;

  // OUTPUT MATRIX GENERATION -----------
    // Matmul based on inner product
    matmul_4x2_parallel_outer_f16vec(matrix_s, v_head, matrix_o, dim_s, dim_s,
                               dim_e, local_core_id, core_per_head);
    mempool_barrier(num_cores);

// SCALE ----------------------------
  if (core_id == 0) {
    dma_memcpy_blocking(matrix_Wqkv, l2_Wo,
                        (dim_e * dim_e) * sizeof(int16_t));
  }
  mempool_barrier(num_cores);
  // Matmul based on inner product
  matmul_4x2_parallel_outer_f16vec(o_head, matrix_Wqkv, matrix_s,
                 dim_s, dim_e, dim_e, core_id, num_cores);
  mempool_barrier(num_cores);

// NORMALIZATION ---------------------
  layernorm_parallel_2x4_f16vec(matrix_s, matrix_result, dim_s,
                                dim_e, core_id, num_cores);

  mempool_stop_benchmark(); 
#endif 


  //mempool_check_f8(matrix_result, l2_Result, dim_s * dim_e, 0x34, 1); // tol = 0.25 = 0x34 (__fp8)
  
  mempool_barrier(num_cores);
  return 0;
}