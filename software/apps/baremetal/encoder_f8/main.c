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

#include "data_encoder_f8.h"

#include "baremetal/mempool_checks.h"
#include "baremetal/mempool_matmul_f8.h"
#include "baremetal/mempool_layernorm_f8.h"
#include "baremetal/mempool_softmax_f8.h"

/*
======================
Parameters and defines

INNER: When defined runs inner product based matmul.
OUTER: When defined runs outer product based matmul.
*/
#define OUTER

__fp8 matrix_input[dim_s * dim_e]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));
__fp8 matrix_Wqkv[dim_e * dim_h]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));
__fp8 matrix_qkv[dim_s * dim_h]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));
__fp8 matrix_a[dim_s * dim_s]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));
__fp8 matrix_upscale[dim_s * dim_s]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  uint32_t dim_h = dim_e / num_heads;
  
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
                        (dim_s * dim_e) * sizeof(int8_t));
    dma_memcpy_blocking(matrix_Wqkv, l2_Wk,
                        (dim_e * dim_e) * sizeof(int8_t));
  }
  mempool_barrier(num_cores);

  // Matmul based on outer product
  matmul_4x4_parallel_outer_f8vec(matrix_input, matrix_Wqkv, matrix_qkv, dim_s, 
                             dim_e, dim_e, core_id, num_cores);
  mempool_barrier(num_cores);

  // Store matrices
  if (core_id == 0) {
  	// Store resulting matrix from l1 to l2 memory
    dma_memcpy_blocking(l2_K, matrix_qkv, (dim_s * dim_e) * sizeof(int8_t));
  }
  mempool_barrier(num_cores);

// 2) Generate matrix Q
  // Initialize matrices
  if (core_id == 0) {
  	// Store matrix from l2 to l1 memory (replace previous l2_Wk with l2_Wq)
    dma_memcpy_blocking(matrix_Wqkv, l2_Wq,
                        (dim_e * dim_e) * sizeof(int8_t));
  }
  mempool_barrier(num_cores);

  // Matmul based on inner product
  matmul_4x4_parallel_outer_f8vec(matrix_input, matrix_Wqkv, matrix_qkv, dim_s, 
                             dim_e, dim_e, core_id, num_cores);
  mempool_barrier(num_cores);

  // Store matrices
  if (core_id == 0) {
  	// Store resulting matrix from l1 to l2 memory
    dma_memcpy_blocking(l2_Q, matrix_qkv, (dim_s * dim_e) * sizeof(int8_t));
  }
  mempool_barrier(num_cores);

// 3) Generate matrix V
  // Initialize matrices
  if (core_id == 0) {
    dma_memcpy_blocking(matrix_Wqkv, l2_Wv,
                        (dim_e * dim_e) * sizeof(int8_t));
  }
  mempool_barrier(num_cores);

  // Matmul based on inner product
  matmul_4x4_parallel_outer_f8vec(matrix_image, matrix_Wqkv, matrix_qkv, dim_s, 
                             dim_e, dim_e, core_id, num_cores);
  mempool_barrier(num_cores);
  
  // Store matrices
  if (core_id == 0) {
  	// Store resulting matrix from l1 to l2 memory
    dma_memcpy_blocking(l2_V, matrix_qkv, (dim_s * dim_e) * sizeof(int8_t));
  }
  mempool_barrier(num_cores);

// ATTENTION HEAD SPLITTING -----------
  // Number of heads = 16
  // Process attention heads in parallel
  if (core_id < num_heads) {

  // ATTENTION MATRIX GENERATION --------
    if (core_id == 0) {
      dma_memcpy_blocking(matrix_qkv, l2_Q,
                          (dim_s * dim_e) * sizeof(int8_t));
    }
    mempool_barrier(num_cores);

    uint32_t core_per_head = num_cores / num_heads;

    // TRANSPOSE
      // Focus is on the speed-up gained, not the accuracy / correctness
      // Use the same matrix K for simplicity

    // Matmul based on inner product
    matmul_4x4_parallel_inner_f8vec(matrix_qkv, matrix_k, matrix_a, dim_s, dim_h,
                               dim_s, core_id, core_per_head);
    mempool_barrier(num_cores);

  // SOFTMAX APPLICATION --------------
    softmax_parallel_f8vec(matrix_a, matrix_s, dim_s, dim_s,
                               core_id, core_per_head);
    mempool_barrier(num_cores);

  // OUTPUT MATRIX GENERATION -----------
    // Matmul based on inner product
    matmul_4x4_parallel_inner_f8vec(matrix_s, matrix_v, matrix_o, dim_s, dim_s,
                               dim_h, core_id, core_per_head);
    mempool_barrier(num_cores);

  }

// SCALE ----------------------------
  // Matmul based on inner product
  matmul_4x4_parallel_inner_f8vec(matrix_o, matrix_Wo, matrix_o_scaled,
                 dim_s, dim_h, dim_s, core_id, num_cores);
  mempool_barrier(num_cores);

 // NORMALIZATION ---------------------
  layernorm_parallel_f8vec(matrix_o_scaled, matrix_result, dim_s,
                                dim_h, core_id, num_cores);


  mempool_stop_benchmark(); 
#endif 


  //mempool_check_f8(matrix_c, l2_Result, dim_s * dim_h, 0x34, 0); // tol = 0.25 = 0x34 (__fp8)
  
  mempool_barrier(num_cores);
  return 0;
}