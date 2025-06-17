// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/* This library applies a softmax over an entire matrix.
 * A is the M x N input matrix, B stores the resulting matrix
 */

#pragma once
#include "builtins_v2.h"

void softmax_parallel_f8vec(const __fp8 *__restrict__ A,
                                __fp8 *__restrict__ B, uint32_t M,
                                uint32_t N, uint32_t core_id,
                                uint32_t numThreads) {
  
  uint32_t i = 0; // loop counter for M
  uint32_t j = 0; // loop counter for N

  const unsigned ShuffleMask0 = 0x04040404; // [a b c d] => [d d d d]
  const unsigned ShuffleMask1 = 0x05050505; // [a b c d] => [c c c c]
  const unsigned ShuffleMask2 = 0x06060606; // [a b c d] => [b b b b]
  const unsigned ShuffleMask3 = 0x07070707; // [a b c d] => [a a a a]

  float Half = 0.5f;
  float One = 1.0f;
  float init = 0.5f;
  v4b vHalf, vOne, max_init;

  asm volatile(
    "vfcpka.b.s %[vHalf], %[Half], %[Half];"
    "vfcpka.b.s %[vOne], %[One], %[One];"
    "vfcpka.b.s %[max_init], %[init], %[init];"
    "vfcpkb.b.s %[vHalf], %[Half], %[Half];"
    "vfcpkb.b.s %[vOne], %[One], %[One];"
    "vfcpkb.b.s %[max_init], %[init], %[init];"
    : [vHalf] "+&r"(vHalf), [vOne] "+&r"(vOne), [max_init] "+&r"(max_init)
    : [Half] "r"(Half), [One] "r"(One), [init] "r"(init));

  for (i = core_id * 2; i < M; i += numThreads * 2) {
    
  // 1) Find the row-wise maximum value
    v4b max0 = max_init;
    v4b max1 = max_init;

    for (j = 0; j < N; j += 8) {

      v4b aVec00 = *(v4b *)&(A[i * N + j]);         // aVec00 = [a03 a02 a01 a00]
      v4b aVec01 = *(v4b *)&(A[i * N + j+4]);       // aVec01 = [a07 a06 a05 a04]
      v4b aVec10 = *(v4b *)&(A[(i + 1) * N + j]);   // aVec10 = [a13 a12 a11 a10]
      v4b aVec11 = *(v4b *)&(A[(i + 1) * N + j+4]); // aVec11 = [a17 a16 a15 a14]

      v4b temp0, temp1;

      asm volatile(
        // Find max value (no full reduction)
        "vfmax.b %[temp0], %[aVec00], %[aVec01];"
        "vfmax.b %[temp1], %[aVec10], %[aVec11];"
        "vfmax.b %[max0], %[max0], %[temp0];"
        "vfmax.b %[max1], %[max1], %[temp1];"
        : [max0] "+&r"(max0), [max1] "+&r"(max1), [temp0] "+&r"(temp0), 
          [temp1] "+&r"(temp1)
        : [aVec00] "r"(aVec00), [aVec01] "r"(aVec01), [aVec10] "r"(aVec10), 
          [aVec11] "r"(aVec11));
    }
    
    // Potential max elements:
    v4b pmax00, pmax01, pmax02, pmax03; 
    v4b pmax10, pmax11, pmax12, pmax13; 
    v4b temp00, temp01, temp10, temp11;
    
    asm volatile(
      // Broadcast each element across vector lanes for full reduction
      "pv.shuffle2.b %[pmax00], %[max0], %[ShuffleMask0];"
      "pv.shuffle2.b %[pmax01], %[max0], %[ShuffleMask1];"
      "pv.shuffle2.b %[pmax02], %[max0], %[ShuffleMask2];"
      "pv.shuffle2.b %[pmax03], %[max0], %[ShuffleMask3];"
      "pv.shuffle2.b %[pmax10], %[max1], %[ShuffleMask0];"
      "pv.shuffle2.b %[pmax11], %[max1], %[ShuffleMask1];"
      "pv.shuffle2.b %[pmax12], %[max1], %[ShuffleMask2];"
      "pv.shuffle2.b %[pmax13], %[max1], %[ShuffleMask3];"
      "vfmax.b %[temp00], %[pmax00], %[pmax01];"
      "vfmax.b %[temp01], %[pmax02], %[pmax03];"
      "vfmax.b %[temp10], %[pmax10], %[pmax11];"
      "vfmax.b %[temp11], %[pmax12], %[pmax13];"
      "vfmax.b %[max0], %[temp00], %[temp01];"
      "vfmax.b %[max1], %[temp10], %[temp11];"
      : [max0] "+&r"(max0), [max1] "+&r"(max1), [temp00] "+&r"(temp00), [temp01] "+&r"(temp01),
        [temp10] "+&r"(temp10), [temp11] "+&r"(temp11),
        [pmax00] "+&r"(pmax00), [pmax01] "+&r"(pmax01), [pmax02] "+&r"(pmax02), 
        [pmax03] "+&r"(pmax03), [pmax10] "+&r"(pmax10), [pmax11] "+&r"(pmax11), 
        [pmax12] "+&r"(pmax12), [pmax13] "+&r"(pmax13)
      : [ShuffleMask0] "r"(ShuffleMask0), [ShuffleMask1] "r"(ShuffleMask1),
        [ShuffleMask2] "r"(ShuffleMask2), [ShuffleMask3] "r"(ShuffleMask3));

    dump_try(*(uint32_t*)&max0);
    dump_try(*(uint32_t*)&max1);

  // 2) Compute exp(x - max)
    v4b vSum0 = (v4b)0.0f;
    v4b vSum1 = (v4b)0.0f;

    for (j = 0; j < N; j += 8) {
      
      v4b aVec00 = *(v4b *)&(A[i * N + j]);         // aVec00 = [a03 a02 a01 a00]
      v4b aVec01 = *(v4b *)&(A[i * N + j+4]);       // aVec01 = [a07 a06 a05 a04]
      v4b aVec10 = *(v4b *)&(A[(i + 1) * N + j]);   // aVec10 = [a13 a12 a11 a10]
      v4b aVec11 = *(v4b *)&(A[(i + 1) * N + j+4]); // aVec11 = [a17 a16 a15 a14]
      
      v4b x00, x01, x10, x11; 
      // Compute x' = x - max(x)
      asm volatile(
        "vfsub.b %[x00], %[aVec00], %[max0];"
        "vfsub.b %[x01], %[aVec01], %[max0];"
        "vfsub.b %[x10], %[aVec10], %[max1];"
        "vfsub.b %[x11], %[aVec11], %[max1];"
        : [x00] "+&r"(x00), [x01] "+&r"(x01), [x10] "+&r"(x10), [x11] "+&r"(x11)
        : [aVec00] "r"(aVec00), [aVec01] "r"(aVec01), [aVec10] "r"(aVec10), 
          [aVec11] "r"(aVec11), [max0] "r"(max0), [max1] "r"(max1));

      v4b xSq00, xSq01, xSq10, xSq11;
      v4b exp00, exp01, exp10, exp11;
      v4b temp0, temp1;

      // Approximate exp with Taylor Series
      asm volatile(
        "vfmul.b %[xSq00], %[x00], %[x00];"      // x^2
        "vfmul.b %[xSq01], %[x01], %[x01];"
        "vfmul.b %[xSq10], %[x10], %[x10];"
        "vfmul.b %[xSq11], %[x11], %[x11];"
        "vfmul.b %[xSq00], %[xSq00], %[vHalf];"  // 0.5*x^2
        "vfmul.b %[xSq01], %[xSq01], %[vHalf];"
        "vfmul.b %[xSq10], %[xSq10], %[vHalf];"
        "vfmul.b %[xSq11], %[xSq11], %[vHalf];"
        "vfmac.b %[xSq00], %[x00], %[vOne];"    // 1*x + 0.5*x^2
        "vfmac.b %[xSq01], %[x01], %[vOne];"
        "vfmac.b %[xSq10], %[x10], %[vOne];"
        "vfmac.b %[xSq11], %[x11], %[vOne];"
        "vfadd.b %[exp00], %[xSq00], %[vOne];"   // exp = exp(x') = exp(x - max(x))
        "vfadd.b %[exp01], %[xSq01], %[vOne];"
        "vfadd.b %[exp10], %[xSq10], %[vOne];"
        "vfadd.b %[exp11], %[xSq11], %[vOne]"
        : [xSq00] "+&r"(xSq00), [xSq01] "+&r"(xSq01), [xSq10] "+&r"(xSq10), [xSq11] "+&r"(xSq11), 
          [exp00] "+&r"(exp00), [exp01] "+&r"(exp01), [exp10] "+&r"(exp10), [exp11] "+&r"(exp11),
          [vSum0] "+&r"(vSum0), [vSum1] "+&r"(vSum1), [temp0] "+&r"(temp0), [temp1] "+&r"(temp1) 
        : [x00] "r"(x00), [x01] "r"(x01), [x10] "r"(x10), [x11] "r"(x11),
          [vHalf] "r"(vHalf), [vOne] "r"(vOne));

      // Add exponents to row-wise sum
      asm volatile(
        "vfadd.b %[temp0], %[exp00], %[exp01];"
        "vfadd.b %[temp1], %[exp10], %[exp11];"
        "vfadd.b %[vSum0], %[vSum0], %[temp0];"
        "vfadd.b %[vSum1], %[vSum1], %[temp1];"
        : [vSum0] "+&r"(vSum0), [vSum1] "+&r"(vSum1), [temp0] "+&r"(temp0), [temp1] "+&r"(temp1) 
        : [exp00] "r"(exp00), [exp01] "r"(exp01), [exp10] "r"(exp10), [exp11] "r"(exp11));

      dump_try(*(uint32_t*)&exp00);

      // Store temporary variables
      (*(v4b *)&B[i * N + j]) = exp00;
      (*(v4b *)&B[i * N + j+4]) = exp01;
      (*(v4b *)&B[(i + 1) * N + j]) = exp10;
      (*(v4b *)&B[(i + 1) * N + j+4]) = exp11;
    }

    v4b sum00, sum01, sum02, sum03;
    v4b sum10, sum11, sum12, sum13;
    asm volatile(
      // Broadcast each element across vector lanes for full reduction
      "pv.shuffle2.b %[sum00], %[vSum0], %[ShuffleMask0];"
      "pv.shuffle2.b %[sum01], %[vSum0], %[ShuffleMask1];"
      "pv.shuffle2.b %[sum02], %[vSum0], %[ShuffleMask2];"
      "pv.shuffle2.b %[sum03], %[vSum0], %[ShuffleMask3];"
      "pv.shuffle2.b %[sum10], %[vSum1], %[ShuffleMask0];"
      "pv.shuffle2.b %[sum11], %[vSum1], %[ShuffleMask1];"
      "pv.shuffle2.b %[sum12], %[vSum1], %[ShuffleMask2];"
      "pv.shuffle2.b %[sum13], %[vSum1], %[ShuffleMask3];"
      "vfadd.b %[vSum0], %[sum00], %[sum01];"
      "vfadd.b %[vSum1], %[sum10], %[sum11];"
      "vfadd.b %[vSum0], %[vSum0], %[sum02];"
      "vfadd.b %[vSum1], %[vSum1], %[sum12];"
      "vfadd.b %[vSum0], %[vSum0], %[sum03];"
      "vfadd.b %[vSum1], %[vSum1], %[sum13];"
      : [vSum0] "+&r"(vSum0), [vSum1] "+&r"(vSum1), [sum00] "+&r"(sum00), [sum01] "+&r"(sum01),
        [sum02] "+&r"(sum02), [sum03] "+&r"(sum03), [sum10] "+&r"(sum10), [sum11] "+&r"(sum11),
        [sum12] "+&r"(sum12), [sum13] "+&r"(sum13)
      : [ShuffleMask0] "r"(ShuffleMask0), [ShuffleMask1] "r"(ShuffleMask1),
        [ShuffleMask2] "r"(ShuffleMask2), [ShuffleMask3] "r"(ShuffleMask3));

    dump_try(*(uint32_t*)&vSum0);

  // 3) Divide by the row-wise sum
    for (j = 0; j < N; j += 8) {
      v4b bVec00 = *(v4b *)&(B[i * N + j]);
      v4b bVec01 = *(v4b *)&(B[i * N + j+4]);
      v4b bVec10 = *(v4b *)&(B[(i + 1) * N + j]);
      v4b bVec11 = *(v4b *)&(B[(i + 1) * N + j+4]);

      v4b res00, res01, res10, res11;
      // Compute softmax = exp / sum_exp
      asm volatile(
        "vfdiv.b %[res00], %[bVec00], %[vSum0];"
        "vfdiv.b %[res01], %[bVec01], %[vSum0];"
        "vfdiv.b %[res10], %[bVec10], %[vSum1];"
        "vfdiv.b %[res11], %[bVec11], %[vSum1];"
        : [res00] "+&r"(res00), [res01] "+&r"(res01), [res10] "+&r"(res10), 
          [res11] "+&r"(res11)
        : [bVec00] "r"(bVec00), [bVec01] "r"(bVec01), [bVec10] "r"(bVec10), 
          [bVec11] "r"(bVec11), [vSum0] "r"(vSum0), [vSum1] "r"(vSum1));
      
      (*(v4b *)&B[i * N + j]) = res00;
      (*(v4b *)&B[i * N + j+4]) = res01;
      (*(v4b *)&B[(i + 1) * N + j]) = res10;
      (*(v4b *)&B[(i + 1) * N + j+4]) = res11;
    }
  }
}