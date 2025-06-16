// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/* This library applies a softmax over an entire matrix.
 * A is the M x N input matrix, B stores the resulting matrix
 */

#pragma once
#include "builtins_v2.h"

void softmax_4x4_parallel_f8vec(const __fp8 *__restrict__ A,
                                __fp8 *__restrict__ B, uint32_t M,
                                uint32_t N, uint32_t core_id,
                                uint32_t numThreads) {
  
  uint32_t i = 0; // loop counter for M
  uint32_t j = 0; // loop counter for N

  const unsigned ShuffleMask0 = 0x04040404; // [a b c d] => [d d d d]
  const unsigned ShuffleMask1 = 0x05050505; // [a b c d] => [c c c c]
  const unsigned ShuffleMask2 = 0x06060606; // [a b c d] => [b b b b]
  const unsigned ShuffleMask3 = 0x07070707; // [a b c d] => [a a a a]

  v4b max;

  for (i = core_id * 4; i < N; i += numThreads * 4) {
    for (j = 0; j < M; j += 4) {

      v4b aVec0 = *(v4b *)&(A[i * N + j]);        // aVec0 = [a03 a02 a01 a00]
      v4b aVec1 = *(v4b *)&(A[(i + 1) * N + j]);  // aVec1 = [a13 a12 a11 a10]
      v4b aVec2 = *(v4b *)&(A[(i + 2) * N + j]);  // aVec2 = [a23 a22 a21 a20]
      v4b aVec3 = *(v4b *)&(A[(i + 3) * N + j]);  // aVec3 = [a33 a32 a31 a30]

      v4b max01, max23, vmax;
      v4b pmax0, pmax1, pmax2, pmax3; // Potential max elements

      asm volatile(
        // Find max value
        "vfmax.b %[max01], %[aVec0], %[aVec1];"
        "vfmax.b %[max23], %[aVec2], %[aVec3];"
        "vfmax.b %[vmax], %[max01], %[max23];"
//        // Broadcast each element across vector lanes for full reduction
//        "pv.shuffle2.b %[pmax0], %[vmax], %[ShuffleMask0];"
//        "pv.shuffle2.b %[pmax1], %[vmax], %[ShuffleMask1];"
//        "pv.shuffle2.b %[pmax2], %[vmax], %[ShuffleMask2];"
//        "pv.shuffle2.b %[pmax3], %[vmax], %[ShuffleMask3];"
//        "vfmax.b %[max01], %[pmax0], %[pmax1];"
//        "vfmax.b %[max23], %[pmax2], %[pmax3];"
//        "vfmax.b %[vmax], %[max01], %[max23];"
//        "vfmac.b %[max], %[vmax], 1;"
        : [max01] "+&r"(max01), [max23] "+&r"(max23), [vmax] "+&r"(vmax),
          [pmax0] "+&r"(pmax0), [pmax1] "+&r"(pmax1), [pmax2] "+&r"(pmax2), 
          [pmax3] "+&r"(pmax3), [max] "+&r"(max)
        : [aVec0] "r"(aVec0), [aVec1] "r"(aVec1), [aVec2] "r"(aVec2), [aVec3] "r"(aVec3),
          [ShuffleMask0] "r"(ShuffleMask0), [ShuffleMask1] "r"(ShuffleMask1),
          [ShuffleMask2] "r"(ShuffleMask2), [ShuffleMask3] "r"(ShuffleMask3));
        
    }
  }  
  
  for (j = core_id * 4; j < N; j += numThreads * 4) {
    for (i = 0; i < M; i += 4) {
      
      v4b aVec0 = *(v4b *)&(A[i * N + j]);
      v4b aVec1 = *(v4b *)&(A[(i + 1) * N + j]);
      v4b aVec2 = *(v4b *)&(A[(i + 2) * N + j]);
      v4b aVec3 = *(v4b *)&(A[(i + 3) * N + j]);
      v4b x0, x1, x2, x3; 
      v4b xSq0, xSq1, xSq2, xSq3;
      v4b exp0, exp1, exp2, exp3;

      // Compute x' = x - max(x)
      asm volatile(
        "vfsub.b %[x0], %[aVec0], %[max];"
        "vfsub.b %[x1], %[aVec1], %[max];"
        "vfsub.b %[x2], %[aVec2], %[max];"
        "vfsub.b %[x3], %[aVec3], %[max];"
        : [x0] "+&r"(x0), [x1] "+&r"(x1), [x2] "+&r"(x2), [x3] "+&r"(x3)
        : [aVec0] "r"(aVec0), [aVec1] "r"(aVec1), [aVec2] "r"(aVec2), [aVec3] "r"(aVec3), 
          [max] "r"(max));

      // Approximate exp with Taylor Series
      asm volatile(
        "vfmul.b %[xSq0], %[x0], %[x0];"  // x^2
        "vfmul.b %[xSq1], %[x1], %[x1];"
        "vfmul.b %[xSq2], %[x2], %[x2];"
        "vfmul.b %[xSq3], %[x3], %[x3];"
        "vfmul.b %[xSq0], %[xSq0], 0.5;"  // 0.5*x^2
        "vfmul.b %[xSq1], %[xSq1], 0.5;"
        "vfmul.b %[xSq2], %[xSq2], 0.5;"
        "vfmul.b %[xSq3], %[xSq3], 0.5;"
        "vfmac.b %[xSq0], %[x0], 1.0;"    // 1*x + 0.5*x^2
        "vfmac.b %[xSq1], %[x1], 1.0;"
        "vfmac.b %[xSq2], %[x2], 1.0;"
        "vfmac.b %[xSq3], %[x3], 1.0;"
        "vfadd.b %[exp0], %[xSq0], 1.0;"   // exp = exp(x') = exp(x - max(x))
        "vfadd.b %[exp1], %[xSq1], 1.0;"
        "vfadd.b %[exp2], %[xSq2], 1.0;"
        "vfadd.b %[exp3], %[xSq3], 1.0;"
        : [xSq0] "+&r"(xSq0), [xSq1] "+&r"(xSq1), [xSq2] "+&r"(xSq2), [xSq3] "+&r"(xSq3), 
          [exp0] "+&r"(exp0), [exp1] "+&r"(exp1), [exp2] "+&r"(exp2), [exp3] "+&r"(exp3)
        : [x0] "r"(x0), [x1] "r"(x1), [x2] "r"(x2), [x3] "r"(x3));

      v4b vSum, sum0, sum1, sum2, sum3;
      // Compute sum_exp = sum(exp(x - max_x))
      asm volatile(
        "vfadd.b %[vSum], %[exp0], %[exp1];"     //Tree reduce to scalar
        "vfadd.b %[vSum], %[vSum], %[exp2];"
        "vfadd.b %[vSum], %[vSum], %[exp3];"
        "pv.shuffle2.b %[sum0], %[vSum], %[ShuffleMask0];"
        "pv.shuffle2.b %[sum1], %[vSum], %[ShuffleMask1];"
        "pv.shuffle2.b %[sum2], %[vSum], %[ShuffleMask2];"
        "pv.shuffle2.b %[sum3], %[vSum], %[ShuffleMask3];"
        "vfadd.b %[vSum], %[sum0], %[sum1];"
        "vfadd.b %[vSum], %[vSum], %[sum2];"
        "vfadd.b %[vSum], %[vSum], %[sum3];"
        : [vSum] "+&r"(vSum), [sum0] "+&r"(sum0), [sum1] "+&r"(sum1),
          [sum2] "+&r"(sum2), [sum3] "+&r"(sum3)
        : [exp0] "r"(exp0), [exp1] "r"(exp1), [exp2] "r"(exp2), [exp3] "r"(exp3));

      v4b res0, res1, res2, res3;
      // Compute softmax = exp / sum_exp
      asm volatile(
        "vfdiv.b %[res0], %[exp0], %[vSum];"
        "vfdiv.b %[res1], %[exp1], %[vSum];"
        "vfdiv.b %[res2], %[exp2], %[vSum];"
        "vfdiv.b %[res3], %[exp3], %[vSum];"
        : [res0] "+&r"(res0), [res1] "+&r"(res1), [res2] "+&r"(res2), [res3] "+&r"(res3)
        : [exp0] "r"(exp0), [vSum] "r"(vSum));
      
      (*(v4b *)&B[i * N + j]) = res0;
      (*(v4b *)&B[(i + 1) * N + j]) = res1;
      (*(v4b *)&B[(i + 2) * N + j]) = res2;
      (*(v4b *)&B[(i + 3) * N + j]) = res3;
    }
  } 
}