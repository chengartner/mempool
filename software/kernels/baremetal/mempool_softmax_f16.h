// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/* This library applies a softmax over an entire matrix.
 * A is the M x N input matrix, B stores the resulting matrix
 */

#pragma once
#include "builtins_v2.h"

void softmax_parallel_f16vec(const __fp16 *__restrict__ A,
                                __fp16 *__restrict__ B, uint32_t M,
                                uint32_t N, uint32_t core_id,
                                uint32_t numThreads) {
  
  uint32_t i = 0; // loop counter for M
  uint32_t j = 0; // loop counter for N

  const unsigned ShuffleMask1 = 0x00020002; // [a b] => [a a]
  const unsigned ShuffleMask0 = 0x00030003; // [a b] => [b b]

  float Half = 0.5f;
  float One = 1.0f;
  float init = 0.5f;
  v2h vHalf, vOne, max_init;
  asm volatile(
    "vfcpka.h.s %[vHalf], %[Half], %[Half];"
    "vfcpka.h.s %[vOne], %[One], %[One];"
    "vfcpka.h.s %[max_init], %[init], %[init];"
    : [vHalf] "+&r"(vHalf), [vOne] "+&r"(vOne), [max_init] "+&r"(max_init)
    : [Half] "r"(Half), [One] "r"(One), [init] "r"(init));

  for (i = core_id * 2; i < M; i += numThreads * 2) {
    
  // 1) Find the row-wise maximum value
    v2h max0 = max_init;
    v2h max1 = max_init;

    for (j = 0; j < N; j += 4) {

      v2h aVec00 = *(v2h *)&(A[i * N + j]);         // aVec00 = [a01 a00]
      v2h aVec01 = *(v2h *)&(A[i * N + j+2]);       // aVec01 = [a03 a02]
      v2h aVec10 = *(v2h *)&(A[(i + 1) * N + j]);   // aVec10 = [a11 a10]
      v2h aVec11 = *(v2h *)&(A[(i + 1) * N + j+2]); // aVec11 = [a13 a12]

      v2h temp0, temp1;

      asm volatile(
        // Find max value (no full reduction)
        "vfmax.h %[temp0], %[aVec00], %[aVec01];"
        "vfmax.h %[temp1], %[aVec10], %[aVec11];"
        "vfmax.h %[max0], %[max0], %[temp0];"
        "vfmax.h %[max1], %[max1], %[temp1];"
        : [max0] "+&r"(max0), [max1] "+&r"(max1), [temp0] "+&r"(temp0), 
          [temp1] "+&r"(temp1)
        : [aVec00] "r"(aVec00), [aVec01] "r"(aVec01), [aVec10] "r"(aVec10), 
          [aVec11] "r"(aVec11));
    }
    
    // Potential max elements:
    v2h pmax00, pmax01; 
    v2h pmax10, pmax11; 
    
    asm volatile(
      // Broadcast each element across vector lanes for full reduction
      "pv.shuffle2.h %[pmax00], %[max0], %[ShuffleMask0];"
      "pv.shuffle2.h %[pmax01], %[max0], %[ShuffleMask1];"
      "pv.shuffle2.h %[pmax10], %[max1], %[ShuffleMask0];"
      "pv.shuffle2.h %[pmax11], %[max1], %[ShuffleMask1];"
      "vfmax.h %[max0], %[pmax00], %[pmax01];"
      "vfmax.h %[max1], %[pmax10], %[pmax11];"
      : [max0] "+&r"(max0), [max1] "+&r"(max1),
        [pmax00] "+&r"(pmax00), [pmax01] "+&r"(pmax01),
        [pmax10] "+&r"(pmax10), [pmax11] "+&r"(pmax11)
      : [ShuffleMask0] "r"(ShuffleMask0), [ShuffleMask1] "r"(ShuffleMask1));


  // 2) Compute exp(x - max)
    v2h vSum0 = (v2h)0.0f;
    v2h vSum1 = (v2h)0.0f;

    for (j = 0; j < N; j += 4) {
      
      v2h aVec00 = *(v2h *)&(A[i * N + j]);
      v2h aVec01 = *(v2h *)&(A[i * N + j+2]);
      v2h aVec10 = *(v2h *)&(A[(i + 1) * N + j]);
      v2h aVec11 = *(v2h *)&(A[(i + 1) * N + j+2]);
      
      v2h x00, x01, x10, x11; 
      // Compute x' = x - max(x)
      asm volatile(
        "vfsub.h %[x00], %[aVec00], %[max0];"
        "vfsub.h %[x01], %[aVec01], %[max0];"
        "vfsub.h %[x10], %[aVec10], %[max1];"
        "vfsub.h %[x11], %[aVec11], %[max1];"
        : [x00] "+&r"(x00), [x01] "+&r"(x01), [x10] "+&r"(x10), [x11] "+&r"(x11)
        : [aVec00] "r"(aVec00), [aVec01] "r"(aVec01), [aVec10] "r"(aVec10), 
          [aVec11] "r"(aVec11), [max0] "r"(max0), [max1] "r"(max1));

      v2h xSq00, xSq01, xSq10, xSq11;
      v2h exp00, exp01, exp10, exp11;
      v2h temp0, temp1;

      // Approximate exp with Taylor Series
      asm volatile(
        "vfmul.h %[xSq00], %[x00], %[x00];"      // x^2
        "vfmul.h %[xSq01], %[x01], %[x01];"
        "vfmul.h %[xSq10], %[x10], %[x10];"
        "vfmul.h %[xSq11], %[x11], %[x11];"
        "vfmul.h %[xSq00], %[xSq00], %[vHalf];"  // 0.5*x^2
        "vfmul.h %[xSq01], %[xSq01], %[vHalf];"
        "vfmul.h %[xSq10], %[xSq10], %[vHalf];"
        "vfmul.h %[xSq11], %[xSq11], %[vHalf];"
        "vfmac.h %[xSq00], %[x00], %[vOne];"    // 1*x + 0.5*x^2
        "vfmac.h %[xSq01], %[x01], %[vOne];"
        "vfmac.h %[xSq10], %[x10], %[vOne];"
        "vfmac.h %[xSq11], %[x11], %[vOne];"
        "vfadd.h %[exp00], %[xSq00], %[vOne];"   // exp = exp(x') = exp(x - max(x))
        "vfadd.h %[exp01], %[xSq01], %[vOne];"
        "vfadd.h %[exp10], %[xSq10], %[vOne];"
        "vfadd.h %[exp11], %[xSq11], %[vOne]"
        : [xSq00] "+&r"(xSq00), [xSq01] "+&r"(xSq01), [xSq10] "+&r"(xSq10), [xSq11] "+&r"(xSq11), 
          [exp00] "+&r"(exp00), [exp01] "+&r"(exp01), [exp10] "+&r"(exp10), [exp11] "+&r"(exp11),
          [vSum0] "+&r"(vSum0), [vSum1] "+&r"(vSum1), [temp0] "+&r"(temp0), [temp1] "+&r"(temp1) 
        : [x00] "r"(x00), [x01] "r"(x01), [x10] "r"(x10), [x11] "r"(x11),
          [vHalf] "r"(vHalf), [vOne] "r"(vOne));

      // Add exponents to row-wise sum
      asm volatile(
        "vfadd.h %[temp0], %[exp00], %[exp01];"
        "vfadd.h %[temp1], %[exp10], %[exp11];"
        "vfadd.h %[vSum0], %[vSum0], %[temp0];"
        "vfadd.h %[vSum1], %[vSum1], %[temp1];"
        : [vSum0] "+&r"(vSum0), [vSum1] "+&r"(vSum1), [temp0] "+&r"(temp0), [temp1] "+&r"(temp1) 
        : [exp00] "r"(exp00), [exp01] "r"(exp01), [exp10] "r"(exp10), [exp11] "r"(exp11));

      // Store temporary variables
      (*(v2h *)&B[i * N + j]) = exp00;
      (*(v2h *)&B[i * N + j+2]) = exp01;
      (*(v2h *)&B[(i + 1) * N + j]) = exp10;
      (*(v2h *)&B[(i + 1) * N + j+2]) = exp11;
    }

    v2h sum00, sum01;
    v2h sum10, sum11;
    asm volatile(
      // Broadcast each element across vector lanes for full reduction
      "pv.shuffle2.h %[sum00], %[vSum0], %[ShuffleMask0];"
      "pv.shuffle2.h %[sum01], %[vSum0], %[ShuffleMask1];"
      "pv.shuffle2.h %[sum10], %[vSum1], %[ShuffleMask0];"
      "pv.shuffle2.h %[sum11], %[vSum1], %[ShuffleMask1];"
      "vfadd.h %[vSum0], %[sum00], %[sum01];"
      "vfadd.h %[vSum1], %[sum10], %[sum11];"
      : [vSum0] "+&r"(vSum0), [vSum1] "+&r"(vSum1), [sum00] "+&r"(sum00), [sum01] "+&r"(sum01),
        [sum10] "+&r"(sum10), [sum11] "+&r"(sum11)
      : [ShuffleMask0] "r"(ShuffleMask0), [ShuffleMask1] "r"(ShuffleMask1));


  // 3) Divide by the row-wise sum
    for (j = 0; j < N; j += 4) {
      v2h bVec00 = *(v2h *)&(B[i * N + j]);
      v2h bVec01 = *(v2h *)&(B[i * N + j+2]);
      v2h bVec10 = *(v2h *)&(B[(i + 1) * N + j]);
      v2h bVec11 = *(v2h *)&(B[(i + 1) * N + j+2]);

      v2h res00, res01, res10, res11;
      // Compute softmax = exp / sum_exp
      asm volatile(
        "vfdiv.h %[res00], %[bVec00], %[vSum0];"
        "vfdiv.h %[res01], %[bVec01], %[vSum0];"
        "vfdiv.h %[res10], %[bVec10], %[vSum1];"
        "vfdiv.h %[res11], %[bVec11], %[vSum1];"
        : [res00] "+&r"(res00), [res01] "+&r"(res01), [res10] "+&r"(res10), 
          [res11] "+&r"(res11)
        : [bVec00] "r"(bVec00), [bVec01] "r"(bVec01), [bVec10] "r"(bVec10), 
          [bVec11] "r"(bVec11), [vSum0] "r"(vSum0), [vSum1] "r"(vSum1));
      
      (*(v2h *)&B[i * N + j]) = res00;
      (*(v2h *)&B[i * N + j+2]) = res01;
      (*(v2h *)&B[(i + 1) * N + j]) = res10;
      (*(v2h *)&B[(i + 1) * N + j+2]) = res11;
    }
  }
}