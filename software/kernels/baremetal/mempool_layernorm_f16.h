// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/* This library implements a normalization matrix based on the LayerNorm.
 * A is the M x N input matrix, B stores the normalized A matrix
 */

#pragma once
#include "builtins_v2.h"


void layernorm_parallel_f16vec(const __fp16 *__restrict__ A,
                                __fp16 *__restrict__ B, uint32_t M,
                                uint32_t N, uint32_t core_id,
                                uint32_t numThreads) {
  
  uint32_t i = 0; // loop counter for M
  uint32_t j = 0; // loop counter for N

  float InvN = 1.0f / (float)N;
  v2h vInvN;

  asm volatile(
  	"vfcpka.h.s %[vInvN], %[InvN], %[InvN];"
  	: [vInvN] "+&r"(vInvN)
  	: [InvN] "r"(InvN));

  for (i = core_id * 2; i < M; i += numThreads * 2) {

    v2h vSum0 = (v2h)0.0f, vSum1 = (v2h)0.0f;
  	v2h vSumSq0 = (v2h)0.0f, vSumSq1 = (v2h)0.0f;

    for (j = 0; j < N; j += 4) {

      v2h aVec00 = *(v2h *)&(A[i * N + j]);
      v2h aVec10 = *(v2h *)&(A[(i + 1) * N + j]);
      v2h aVec01 = *(v2h *)&(A[i * N + j+2]);
      v2h aVec11 = *(v2h *)&(A[(i + 1) * N + j+2]);
      
      v2h aVecSq00, aVecSq01, aVecSq10, aVecSq11;

      asm volatile(
        // 1) Accumulate sum(x): Accumulate aVeci0 and aVeci1 in vSumi
        // 2) Accumulate sum(x^2): Square aVecji and then accumulate in vSumSqi
        "vfadd.h %[vSum0], %[vSum0], %[aVec00];"
        "vfadd.h %[vSum1], %[vSum1], %[aVec10];"
        "vfmul.h %[aVecSq00], %[aVec00], %[aVec00];"
        "vfmul.h %[aVecSq10], %[aVec10], %[aVec10];"
        "vfmul.h %[aVecSq01], %[aVec01], %[aVec01];"
        "vfmul.h %[aVecSq11], %[aVec11], %[aVec11];"
        "vfadd.h %[vSumSq0], %[vSumSq0], %[aVecSq00];"
        "vfadd.h %[vSumSq1], %[vSumSq1], %[aVecSq10];"
        "vfadd.h %[vSum0], %[vSum0], %[aVec01];"
        "vfadd.h %[vSum1], %[vSum1], %[aVec11];"
        "vfadd.h %[vSumSq0], %[vSumSq0], %[aVecSq01];"
        "vfadd.h %[vSumSq1], %[vSumSq1], %[aVecSq11];"
        : [vSum0] "+&r"(vSum0), [vSum1] "+&r"(vSum1), [vSumSq0] "+&r"(vSumSq0),
          [vSumSq1] "+&r"(vSumSq1), [aVecSq00] "+&r"(aVecSq00), 
          [aVecSq01] "+&r"(aVecSq01),  [aVecSq10] "+&r"(aVecSq10), [aVecSq11] "+&r"(aVecSq11)
        : [aVec00] "r"(aVec00), [aVec01] "r"(aVec01), [aVec10] "r"(aVec10),
          [aVec11] "r"(aVec11));
    }

    unsigned ShuffleMask1 = 0x00020002; // [a b] => [a a]
    unsigned ShuffleMask0 = 0x00030003; // [a b] => [b b]
    v2h sum00, sum01, sum10, sum11;
    v2h sumSq00, sumSq01, sumSq10, sumSq11;
    // Reduce vector sum to one sum (replicated 2 times)
    asm volatile(
      // Broadcast each element across vector lanes for full reduction
      "pv.shuffle2.h %[sum00], %[vSum0], %[ShuffleMask0];"
      "pv.shuffle2.h %[sum01], %[vSum0], %[ShuffleMask1];"
      "pv.shuffle2.h %[sum10], %[vSum1], %[ShuffleMask0];"
      "pv.shuffle2.h %[sum11], %[vSum1], %[ShuffleMask1];"
      "pv.shuffle2.h %[sumSq00], %[vSumSq0], %[ShuffleMask0];"
      "pv.shuffle2.h %[sumSq01], %[vSumSq0], %[ShuffleMask1];"
      "pv.shuffle2.h %[sumSq10], %[vSumSq1], %[ShuffleMask0];"
      "pv.shuffle2.h %[sumSq11], %[vSumSq1], %[ShuffleMask1];"
      "vfadd.h %[vSum0], %[sum00], %[sum01];"
      "vfadd.h %[vSum1], %[sum10], %[sum11];"
      "vfadd.h %[vSumSq0], %[sumSq00], %[sumSq01];"
      "vfadd.h %[vSumSq1], %[sumSq10], %[sumSq11];"
      : [vSum0] "+&r"(vSum0), [vSum1] "+&r"(vSum1), [vSumSq0] "+&r"(vSumSq0),
        [vSumSq1] "+&r"(vSumSq1), [sum00] "+&r"(sum00), [sum01] "+&r"(sum01),
        [sum10] "+&r"(sum10), [sum11] "+&r"(sum11), [sumSq00] "+&r"(sumSq00), [sumSq01] "+&r"(sumSq01), 
        [sumSq10] "+&r"(sumSq10), [sumSq11] "+&r"(sumSq11)
      : [ShuffleMask0] "r"(ShuffleMask0), [ShuffleMask1] "r"(ShuffleMask1));

    v2h vMean0, vMean1, vVar0, vVar1, vStd0, vStd1, vMeanSq0, vMeanSq1;
    
    asm volatile(
      // Compute the mean: mean = sum / M
      "vfmul.h %[vMean0], %[vSum0], %[vInvN];"
      "vfmul.h %[vMean1], %[vSum1], %[vInvN];"
      // Compute E[x^2]
      "vfmul.h %[vVar0], %[vSumSq0], %[vInvN];"
      "vfmul.h %[vVar1], %[vSumSq1], %[vInvN];"
      // Compute mean^2
      "vfmul.h %[vMeanSq0], %[vMean0], %[vMean0];"
      "vfmul.h %[vMeanSq1], %[vMean1], %[vMean1];"
      // Compute the variance: var = E[x^2] - mean^2
      "vfsub.h %[vVar0], %[vVar0], %[vMeanSq0];"
      "vfsub.h %[vVar1], %[vVar1], %[vMeanSq1];"
      // Compute the square root of Var: std = sqrt(var)
      "vfsqrt.h %[vStd0], %[vVar0];"
      "vfsqrt.h %[vStd1], %[vVar1];"
      : [vMean0] "+&r"(vMean0), [vMean1] "+&r"(vMean1), [vVar0] "+&r"(vVar0), 
        [vVar1] "+&r"(vVar1), [vStd0] "+&r"(vStd0), [vStd1] "+&r"(vStd1),
    	  [vInvN] "+&r"(vInvN), [vMeanSq0] "+&r"(vMeanSq0), [vMeanSq1] "+&r"(vMeanSq1)
      : [vSum0] "r"(vSum0), [vSum1] "r"(vSum1), [vSumSq0] "r"(vSumSq0),
        [vSumSq1] "r"(vSumSq1));


    for (j = 0; j < N; j += 4) {

      v2h aVec00 = *(v2h *)&(A[i * N + j]);        // aVec0 = [a03 a02 a01 a00]
      v2h aVec01 = *(v2h *)&(A[i * N + j+2]);
      v2h aVec10 = *(v2h *)&(A[(i + 1) * N + j]);  // aVec1 = [a13 a12 a11 a10]
      v2h aVec11 = *(v2h *)&(A[(i + 1) * N + j+2]);
    	
      v2h aNorm00, aNorm01, aNorm10, aNorm11;

      asm volatile(
    	  // Compute: aVec0 - vMean
    	  "vfsub.h %[aNorm00], %[aVec00], %[vMean0];"
    	  "vfsub.h %[aNorm01], %[aVec01], %[vMean0];"
        "vfsub.h %[aNorm10], %[aVec10], %[vMean1];"
        "vfsub.h %[aNorm11], %[aVec11], %[vMean1];"
    	  // Compute: aNorm0 / vStd
    	  "vfdiv.h %[aNorm00], %[aNorm00], %[vStd0];"
    	  "vfdiv.h %[aNorm01], %[aNorm01], %[vStd0];"
        "vfdiv.h %[aNorm10], %[aNorm10], %[vStd1];"
        "vfdiv.h %[aNorm11], %[aNorm11], %[vStd1];"
        : [aNorm00] "+&r"(aNorm00), [aNorm01] "+&r"(aNorm01), 
          [aNorm10] "+&r"(aNorm10), [aNorm11] "+&r"(aNorm11)
        : [aVec00] "r"(aVec00), [aVec01] "r"(aVec01),
          [aVec10] "r"(aVec10), [aVec11] "r"(aVec11),
          [vMean0] "r"(vMean0), [vMean1] "r"(vMean1), [vStd0] "r"(vStd0),
          [vStd1] "r"(vStd1));
      
      (*(v2h *)&B[i * N + j]) = aNorm00;
      (*(v2h *)&B[i * N + j+2]) = aNorm01;
      (*(v2h *)&B[(i + 1) * N + j]) = aNorm10;
      (*(v2h *)&B[(i + 1) * N + j+2]) = aNorm11;
    }
  }
}