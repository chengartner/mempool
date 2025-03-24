// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "builtins_v2.h"

#define AXPYF8VEC_UNROLLED4_LOOP                                               \
  {                                                                            \
    x0123 = (*(v4b *)&in_x[i]);                                                \                                              \
    x4567 = (*(v4b *)&in_x[i + 4]);                                            \
    y0123 = (*(v4b *)&in_y[i]);                                                \
    y4567 = (*(v4b *)&in_y[i + 4]);                                            \
    asm volatile("vfmac.h %[y0123], %[x0123], %[aaaa];"                        \
                 "vfmac.h %[y4567], %[x4567], %[aaaa];"                        \
                 : [y0123] "+&r"(y0123), [y4567] "+&r"(y4567),                 \
                 : [x0123] "r"(x0123), [x4567] "r"(x4567),                     \
                   [aaaa] "r"(aaaa));                                          \
    (*(v4b *)&in_y[i]) = y0123;                                                \                                              \
    (*(v4b *)&in_y[i + 4]) = y4567;                                            \
  }

/* Parallel dot-product with loop unrolling */
/* Load and stores only in local memory */
void axpy_f8vecp_local_unrolled4(uint32_t a, __fp8 *in_x, __fp8 *in_y, 
								 uint32_t Len) {

  uint32_t core_id = mempool_get_core_id();

  uint32_t aaaa = (a << 24U) | (a << 16U) | (a << 8U) | a;
  v4b x0123, x4567;
  v4b y0123, y4567;
  for (uint32_t i = 4 * core_id * BANKING_FACTOR; i < Len; i += 4 * NUM_BANKS) {
    AXPYF8VEC_UNROLLED4_LOOP;
  }
  // Barrier synchronization
  mempool_log_barrier(2, core_id); // TODO 2 or 4??

  return;
}
