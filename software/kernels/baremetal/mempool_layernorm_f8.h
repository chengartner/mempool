// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/* This library implements a normalization matrix based on the LayerNorm.
 * A is the M x N input matrix, B stores the normalized A matrix
 */

#pragma once
#include "builtins_v2.h"


void layernorm_parallel_f8vec(const __fp8 *__restrict__ A,
                                __fp8 *__restrict__ B, uint32_t M,
                                uint32_t N, uint32_t core_id,
                                uint32_t numThreads) {
  
  uint32_t i = 0; // loop counter for M
  uint32_t j = 0; // loop counter for N

  float InvN = 1.0f / (float)N;
  v4b vInvN;

  asm volatile(
  	"vfcpka.b.s %[vInvN], %[InvN], %[InvN];"
    "vfcpkb.b.s %[vInvN], %[InvN], %[InvN];"
  	: [vInvN] "+&r"(vInvN)
  	: [InvN] "r"(InvN));

  const unsigned ShuffleMask0 = 0x04040404; // [a b c d] => [d d d d]
  const unsigned ShuffleMask1 = 0x05050505; // [a b c d] => [c c c c]
  const unsigned ShuffleMask2 = 0x06060606; // [a b c d] => [b b b b]
  const unsigned ShuffleMask3 = 0x07070707; // [a b c d] => [a a a a]

  for (i = core_id * 2; i < M; i += numThreads * 2) {

  	v4b vSum0 = (v4b)0.0f, vSum1 = (v4b)0.0f;
  	v4b vSumSq0 = (v4b)0.0f, vSumSq1 = (v4b)0.0f;

    for (j = 0; j < N; j += 8) {

      v4b aVec00 = *(v4b *)&(A[i * N + j]);        // aVec0 = [a03 a02 a01 a00]
      v4b aVec01 = *(v4b *)&(A[i * N + j+4]);
      v4b aVec10 = *(v4b *)&(A[(i + 1) * N + j]);  // aVec1 = [a13 a12 a11 a10]
      v4b aVec11 = *(v4b *)&(A[(i + 1) * N + j+4]);
      
      v4b aVecSq00, aVecSq01, aVecSq10, aVecSq11;

      asm volatile(
        // Accumulate sum(x)
        "vfadd.b %[vSum0], %[vSum0], %[aVec00];"
        "vfadd.b %[vSum1], %[vSum1], %[aVec10];"
        "vfmul.b %[aVecSq00], %[aVec00], %[aVec00];"
        "vfmul.b %[aVecSq01], %[aVec01], %[aVec01];"
        "vfmul.b %[aVecSq10], %[aVec10], %[aVec10];"
        "vfmul.b %[aVecSq11], %[aVec11], %[aVec11];"
        "vfadd.b %[vSum0], %[vSum0], %[aVec01];"
        "vfadd.b %[vSum1], %[vSum1], %[aVec11];"
        // Accumulate sum(x^2)
        "vfadd.b %[vSumSq0], %[vSumSq0], %[aVecSq00];"
        "vfadd.b %[vSumSq1], %[vSumSq1], %[aVecSq10];"
        "vfadd.b %[vSumSq0], %[vSumSq0], %[aVecSq01];"
        "vfadd.b %[vSumSq1], %[vSumSq1], %[aVecSq11];"
        : [vSum0] "+&r"(vSum0), [vSum1] "+&r"(vSum1), [vSumSq0] "+&r"(vSumSq0),
          [vSumSq1] "+&r"(vSumSq1), [aVecSq00] "+&r"(aVecSq00), 
          [aVecSq01] "+&r"(aVecSq01),  [aVecSq10] "+&r"(aVecSq10), [aVecSq11] "+&r"(aVecSq11)
        : [aVec00] "r"(aVec00), [aVec01] "r"(aVec01), [aVec10] "r"(aVec10),
          [aVec11] "r"(aVec11));
    }

    v4b sum00, sum01, sum02, sum03;
    v4b sum10, sum11, sum12, sum13;
    // Reduce vector sum to one sum (replicated 4 times)
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
      "pv.shuffle2.b %[sum00], %[vSumSq0], %[ShuffleMask0];"
      "pv.shuffle2.b %[sum01], %[vSumSq0], %[ShuffleMask1];"
      "pv.shuffle2.b %[sum02], %[vSumSq0], %[ShuffleMask2];"
      "pv.shuffle2.b %[sum03], %[vSumSq0], %[ShuffleMask3];"
      "pv.shuffle2.b %[sum10], %[vSumSq1], %[ShuffleMask0];"
      "pv.shuffle2.b %[sum11], %[vSumSq1], %[ShuffleMask1];"
      "pv.shuffle2.b %[sum12], %[vSumSq1], %[ShuffleMask2];"
      "pv.shuffle2.b %[sum13], %[vSumSq1], %[ShuffleMask3];"
      "vfadd.b %[vSumSq0], %[sum00], %[sum01];"
      "vfadd.b %[vSumSq1], %[sum10], %[sum11];"
      "vfadd.b %[vSumSq0], %[vSum0], %[sum02];"
      "vfadd.b %[vSumSq1], %[vSum1], %[sum12];"
      "vfadd.b %[vSumSq0], %[vSum0], %[sum03];"
      "vfadd.b %[vSumSq1], %[vSum1], %[sum13];"
      : [vSum0] "+&r"(vSum0), [vSum1] "+&r"(vSum1), [vSumSq0] "+&r"(vSumSq0),
        [vSumSq1] "+&r"(vSumSq1), [sum00] "+&r"(sum00), [sum01] "+&r"(sum01),
        [sum02] "+&r"(sum02), [sum03] "+&r"(sum03), [sum10] "+&r"(sum10), [sum11] "+&r"(sum11),
        [sum12] "+&r"(sum12), [sum13] "+&r"(sum13)
      : [ShuffleMask0] "r"(ShuffleMask0), [ShuffleMask1] "r"(ShuffleMask1),
        [ShuffleMask2] "r"(ShuffleMask2), [ShuffleMask3] "r"(ShuffleMask3));

    v4b vMean0, vMean1, vVar0, vVar1, vStd0, vStd1, vMeanSq0, vMeanSq1;
    
    asm volatile(
      // Compute the mean: mean = sum / M
      "vfmul.b %[vMean0], %[vSum0], %[vInvN];"
      "vfmul.b %[vMean1], %[vSum1], %[vInvN];"
      // Compute E[x^2]
      "vfmul.b %[vVar0], %[vSumSq0], %[vInvN];"
      "vfmul.b %[vVar1], %[vSumSq1], %[vInvN];"
      // Compute mean^2
      "vfmul.b %[vMeanSq0], %[vMean0], %[vMean0];"
      "vfmul.b %[vMeanSq1], %[vMean1], %[vMean1];"
      // Compute the variance: var = E[x^2] - mean^2
      "vfsub.b %[vVar0], %[vVar0], %[vMeanSq0];"
      "vfsub.b %[vVar1], %[vVar1], %[vMeanSq1];"
      // Compute the square root of Var: std = sqrt(var)
      "vfsqrt.b %[vStd0], %[vVar0];"
      "vfsqrt.b %[vStd1], %[vVar1];"
      : [vMean0] "+&r"(vMean0), [vMean1] "+&r"(vMean1), [vVar0] "+&r"(vVar0), 
        [vVar1] "+&r"(vVar1), [vStd0] "+&r"(vStd0), [vStd1] "+&r"(vStd1),
    	  [vInvN] "+&r"(vInvN), [vMeanSq0] "+&r"(vMeanSq0), [vMeanSq1] "+&r"(vMeanSq1)
      : [vSum0] "r"(vSum0), [vSum1] "r"(vSum1), [vSumSq0] "r"(vSumSq0),
        [vSumSq1] "r"(vSumSq1));

    dump_try(*(uint32_t*)&vVar0);
    dump_try(*(uint32_t*)&vStd0);

    for (j = 0; j < N; j += 8) {

      v4b aVec00 = *(v4b *)&(A[i * N + j]);        // aVec0 = [a03 a02 a01 a00]
      v4b aVec01 = *(v4b *)&(A[i * N + j+4]);
      v4b aVec10 = *(v4b *)&(A[(i + 1) * N + j]);  // aVec1 = [a13 a12 a11 a10]
      v4b aVec11 = *(v4b *)&(A[(i + 1) * N + j+4]);
    	
      v4b aNorm00, aNorm01, aNorm10, aNorm11;

      asm volatile(
    	// Compute: aVec0 - vMean
    	"vfsub.b %[aNorm00], %[aVec00], %[vMean0];"
    	"vfsub.b %[aNorm01], %[aVec01], %[vMean0];"
      "vfsub.b %[aNorm10], %[aVec10], %[vMean1];"
      "vfsub.b %[aNorm11], %[aVec11], %[vMean1];"
    	// Compute: aNorm0 / vStd
    	"vfdiv.b %[aNorm00], %[aNorm00], %[vStd0];"
    	"vfdiv.b %[aNorm01], %[aNorm01], %[vStd0];"
      "vfdiv.b %[aNorm10], %[aNorm10], %[vStd1];"
      "vfdiv.b %[aNorm11], %[aNorm11], %[vStd1];"
        : [aNorm00] "+&r"(aNorm00), [aNorm01] "+&r"(aNorm01), 
          [aNorm10] "+&r"(aNorm10), [aNorm11] "+&r"(aNorm11)
        : [aVec00] "r"(aVec00), [aVec01] "r"(aVec01),
          [aVec10] "r"(aVec10), [aVec11] "r"(aVec11),
          [vMean0] "r"(vMean0), [vMean1] "r"(vMean1), [vStd0] "r"(vStd0),
          [vStd1] "r"(vStd1));
      
      (*(v4b *)&B[i * N + j]) = aNorm00;
      (*(v4b *)&B[i * N + j+4]) = aNorm01;
      (*(v4b *)&B[(i + 1) * N + j]) = aNorm10;
      (*(v4b *)&B[(i + 1) * N + j+4]) = aNorm11;
    }
  }
}