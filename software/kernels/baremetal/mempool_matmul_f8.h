// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/* This library implements the matrix multiplication in multiple different ways.
 * The functions all follow the following format:
 *
 * A is an M x N matrix, B is a N x P matrix, and C is a M x P matrix
 * C = AB
 */

#pragma once
#include "builtins_v2.h"

void matmul_4x2_parallel_f8vec(const __fp8 *__restrict__ A,
                                const __fp8 *__restrict__ B,
                                __fp8 *__restrict__ C, uint32_t M,
                                uint32_t N, uint32_t P, uint32_t core_id,
                                uint32_t numThreads) {

  uint32_t i = 0; // loop counter for M
  uint32_t j = 0; // loop counter for N
  uint32_t k = 0; // loop counter for P

  for (k = core_id * 2; k < P; k += numThreads * 4) {
    for (i = 0; i < M; i += 4) {
      float volatile sum00 = 0.0f;
      float volatile sum01 = 0.0f;
      float volatile sum02 = 0.0f;
      float volatile sum03 = 0.0f;
      float volatile sum10 = 0.0f;
      float volatile sum11 = 0.0f;
      float volatile sum12 = 0.0f;
      float volatile sum13 = 0.0f;
      float volatile sum20 = 0.0f;
      float volatile sum21 = 0.0f;
      float volatile sum22 = 0.0f;
      float volatile sum23 = 0.0f;
      float volatile sum30 = 0.0f;
      float volatile sum31 = 0.0f;
      float volatile sum32 = 0.0f;
      float volatile sum33 = 0.0f;
      for (j = 0; j < N; j += 4) {

        v4b aVec0 = *(v4b *)&(A[i * N + j]);
        v4b aVec1 = *(v4b *)&(A[(i + 1) * N + j]);
        v4b aVec2 = *(v4b *)&(A[(i + 2) * N + j]);
        v4b aVec3 = *(v4b *)&(A[(i + 3) * N + j]);
        v4b bVecTemp0 = *(v4b *)&(B[j * P + k]);
        v4b bVecTemp1 = *(v4b *)&(B[(j + 1) * P + k]);
        v4b bVec0, bVec1, bVec2, bVec3;
        unsigned TempH, TempL;
        asm volatile(
            "pv.extract.b %[TempH], %[bVecTemp0], 3;"
            "pv.extract.b %[TempL], %[bVecTemp1], 3;"
            "pv.pack %[bVec0], %[TempL], %[TempH];"     // bVec0 = [b00 b10]
            "pv.extract.b %[TempH], %[bVecTemp0], 2;"
            "pv.extract.b %[TempL], %[bVecTemp1], 2;"
            "pv.pack %[bVec1], %[TempL], %[TempH];"     // bVec1 = [b01 b11]
            "pv.extract.b %[TempH], %[bVecTemp0], 1;"
            "pv.extract.b %[TempL], %[bVecTemp1], 1;"
            "pv.pack %[bVec2], %[TempL], %[TempH];"     // bVec2 = [b02 b12]
            "pv.extract.b %[TempH], %[bVecTemp0], 0;"
            "pv.extract.b %[TempL], %[bVecTemp1], 0;"
            "pv.pack %[bVec3], %[TempL], %[TempH];"     // bVec3 = [b03 b13]
            "vfdotpex.s.b %[sum00], %[aVec0], %[bVec0];"
            "vfdotpex.s.b %[sum01], %[aVec0], %[bVec1];"
            "vfdotpex.s.b %[sum02], %[aVec0], %[bVec2];"
            "vfdotpex.s.b %[sum03], %[aVec0], %[bVec3];"
            "vfdotpex.s.b %[sum10], %[aVec1], %[bVec0];"
            "vfdotpex.s.b %[sum11], %[aVec1], %[bVec1];"
            "vfdotpex.s.b %[sum12], %[aVec1], %[bVec2];"
            "vfdotpex.s.b %[sum13], %[aVec1], %[bVec3];"
            "vfdotpex.s.b %[sum20], %[aVec2], %[bVec0];"
            "vfdotpex.s.b %[sum21], %[aVec2], %[bVec1];"
            "vfdotpex.s.b %[sum22], %[aVec2], %[bVec2];"
            "vfdotpex.s.b %[sum23], %[aVec2], %[bVec3];"
            "vfdotpex.s.b %[sum30], %[aVec3], %[bVec0];"
            "vfdotpex.s.b %[sum31], %[aVec3], %[bVec1];"
            "vfdotpex.s.b %[sum32], %[aVec3], %[bVec2];"
            "vfdotpex.s.b %[sum32], %[aVec3], %[bVec3];"
            : [sum00] "+&r"(sum00), [sum01] "+&r"(sum01), [sum02] "+&r"(sum02), [sum03] "+&r"(sum03),
              [sum10] "+&r"(sum10), [sum11] "+&r"(sum11), [sum12] "+&r"(sum12), [sum13] "+&r"(sum13),
              [sum20] "+&r"(sum20), [sum21] "+&r"(sum21), [sum22] "+&r"(sum22), [sum23] "+&r"(sum23),
              [sum30] "+&r"(sum30), [sum31] "+&r"(sum31), [sum32] "+&r"(sum32), [sum33] "+&r"(sum33),
              [bVec0] "=&r"(bVec0), [bVec1] "+&r"(bVec1), [bVec2] "+&r"(bVec2), [bVec3] "+&r"(bVec3),
              [TempH] "+&r"(TempH), [TempL] "+&r"(TempL)
            : [aVec0] "r"(aVec0), [aVec1] "r"(aVec1), [aVec2] "r"(aVec2),
              [aVec3] "r"(aVec3), [bVecTemp0] "r"(bVecTemp0),
              [bVecTemp1] "r"(bVecTemp1)
            :);
      }
      v4b res0, res1, res2, res3;
      asm volatile("pv.packhi.b %[res0], %[sum03], %[sum02];"
                   "pv.packlo.b %[res0], %[sum01], %[sum00];"
                   "pv.packhi.b %[res1], %[sum13], %[sum12];"
                   "pv.packlo.b %[res1], %[sum11], %[sum10];"
                   "pv.packhi.b %[res1], %[sum23], %[sum22];"
                   "pv.packlo.b %[res1], %[sum21], %[sum20];"
                   "pv.packhi.b %[res1], %[sum33], %[sum32];"
                   "pv.packlo.b %[res1], %[sum31], %[sum30];"
                   : [res0] "=&r"(res0), [res1] "=&r"(res1), [res2] "=&r"(res2),
                     [res3] "=&r"(res3)
                   : [sum00] "r"(sum00), [sum01] "r"(sum01), [sum02] "r"(sum02), [sum03] "r"(sum03),
                     [sum10] "r"(sum10), [sum11] "r"(sum11), [sum12] "r"(sum12), [sum13] "r"(sum13),
                     [sum20] "r"(sum20), [sum21] "r"(sum21), [sum22] "r"(sum22), [sum23] "r"(sum23),
                     [sum30] "r"(sum30), [sum31] "r"(sum31), [sum32] "r"(sum32), [sum33] "r"(sum33)
                   :);
      (*(v4b *)&C[i * P + k]) = res0;
      (*(v4b *)&C[(i + 1) * P + k]) = res1;
      (*(v4b *)&C[(i + 2) * P + k]) = res2;
      (*(v4b *)&C[(i + 3) * P + k]) = res3;
    }
  }
}
