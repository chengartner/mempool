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

// Matmul based on outer product between rows of A and cols of B
  // Use 4 cols of A (store 4 elements each) and 4 rows of B (store 4 elements each)
void matmul_4x4_parallel_outer_f8vec(const __fp8 *__restrict__ A,
                                const __fp8 *__restrict__ B,
                                __fp8 *__restrict__ C, uint32_t M,
                                uint32_t N, uint32_t P, uint32_t core_id,
                                uint32_t numThreads) {

  uint32_t i = 0; // loop counter for M
  uint32_t j = 0; // loop counter for N
  uint32_t k = 0; // loop counter for P

  const unsigned ShuffleMask0 = 0x04040404; // [a b c d] => [d d d d]
  const unsigned ShuffleMask1 = 0x05050505; // [a b c d] => [c c c c]
  const unsigned ShuffleMask2 = 0x06060606; // [a b c d] => [b b b b]
  const unsigned ShuffleMask3 = 0x07070707; // [a b c d] => [a a a a]

  for (k = core_id * 4; k < P; k += numThreads * 4) {
    for (i = 0; i < M; i += 4) {
      v4b sum0 = (v4b)0.0f;
      v4b sum1 = (v4b)0.0f;
      v4b sum2 = (v4b)0.0f;
      v4b sum3 = (v4b)0.0f;
      
      for (j = 0; j < N; j += 4) {

        v4b aVec0 = *(v4b *)&(A[i * N + j]);        // aVec0 = [a03 a02 a01 a00]
        v4b aVec1 = *(v4b *)&(A[(i + 1) * N + j]);  // aVec1 = [a13 a12 a11 a10]
        v4b aVec2 = *(v4b *)&(A[(i + 2) * N + j]);  // aVec2 = [a23 a22 a21 a20]
        v4b aVec3 = *(v4b *)&(A[(i + 3) * N + j]);  // aVec3 = [a33 a32 a31 a30]
        //v4b aVec00 = (v4b)0.0f, aVec01 = (v4b)0.0f, aVec02 = (v4b)0.0f, aVec03 = (v4b)0.0f; 
        //v4b aVec10 = (v4b)0.0f, aVec11 = (v4b)0.0f, aVec12 = (v4b)0.0f, aVec13 = (v4b)0.0f;
        //v4b aVec20 = (v4b)0.0f, aVec21 = (v4b)0.0f, aVec22 = (v4b)0.0f, aVec23 = (v4b)0.0f;
        //v4b aVec30 = (v4b)0.0f, aVec31 = (v4b)0.0f, aVec32 = (v4b)0.0f, aVec33 = (v4b)0.0f;
        v4b aVec00, aVec01, aVec02, aVec03;
        v4b aVec10, aVec11, aVec12, aVec13;
        v4b aVec20, aVec21, aVec22, aVec23;
        v4b aVec30, aVec31, aVec32, aVec33;

        asm volatile(
            "pv.shuffle2.b %[aVec00], %[aVec0], %[ShuffleMask0];" // aVec00 = [a00 a00 a00 a00]
            "pv.shuffle2.b %[aVec01], %[aVec0], %[ShuffleMask1];" // aVec01 = [a01 a01 a01 a01]
            "pv.shuffle2.b %[aVec02], %[aVec0], %[ShuffleMask2];" // aVec02 = [a02 a02 a02 a02]
            "pv.shuffle2.b %[aVec03], %[aVec0], %[ShuffleMask3];" // aVec03 = [a03 a03 a03 a03]
            "pv.shuffle2.b %[aVec10], %[aVec1], %[ShuffleMask0];" // aVec10 = [a10 a10 a10 a10]
            "pv.shuffle2.b %[aVec11], %[aVec1], %[ShuffleMask1];" // aVec11 = [a11 a11 a11 a11]
            "pv.shuffle2.b %[aVec12], %[aVec1], %[ShuffleMask2];" // aVec12 = [a12 a12 a12 a12]
            "pv.shuffle2.b %[aVec13], %[aVec1], %[ShuffleMask3];" // aVec13 = [a13 a13 a13 a13]
            "pv.shuffle2.b %[aVec20], %[aVec2], %[ShuffleMask0];" // aVec20 = [a20 a20 a20 a20]
            "pv.shuffle2.b %[aVec21], %[aVec2], %[ShuffleMask1];" // aVec21 = [a21 a21 a21 a21]
            "pv.shuffle2.b %[aVec22], %[aVec2], %[ShuffleMask2];" // aVec22 = [a22 a22 a22 a22]
            "pv.shuffle2.b %[aVec23], %[aVec2], %[ShuffleMask3];" // aVec23 = [a23 a23 a23 a23]
            "pv.shuffle2.b %[aVec30], %[aVec3], %[ShuffleMask0];" // aVec30 = [a30 a30 a30 a30]
            "pv.shuffle2.b %[aVec31], %[aVec3], %[ShuffleMask1];" // aVec31 = [a31 a31 a31 a31]
            "pv.shuffle2.b %[aVec32], %[aVec3], %[ShuffleMask2];" // aVec32 = [a32 a32 a32 a32]
            "pv.shuffle2.b %[aVec33], %[aVec3], %[ShuffleMask3];" // aVec33 = [a33 a33 a33 a33]
            : [aVec00] "+&r"(aVec00), [aVec01] "+&r"(aVec01), [aVec02] "+&r"(aVec02), [aVec03] "+&r"(aVec03),
              [aVec10] "+&r"(aVec10), [aVec11] "+&r"(aVec11), [aVec12] "+&r"(aVec12), [aVec13] "+&r"(aVec13),
              [aVec20] "+&r"(aVec20), [aVec21] "+&r"(aVec21), [aVec22] "+&r"(aVec22), [aVec23] "+&r"(aVec23),
              [aVec30] "+&r"(aVec30), [aVec31] "+&r"(aVec31), [aVec32] "+&r"(aVec32), [aVec33] "+&r"(aVec33)
            : [aVec0] "r"(aVec0), [aVec1] "r"(aVec1), [aVec2] "r"(aVec2), [aVec3] "r"(aVec3),
              [ShuffleMask0] "r"(ShuffleMask0), [ShuffleMask1] "r"(ShuffleMask1),
              [ShuffleMask2] "r"(ShuffleMask2), [ShuffleMask3] "r"(ShuffleMask3));

        v4b bVec0 = *(v4b *)&(B[j * P + k]);        // bVec0 = [b03 b02 b01 b00]
        v4b bVec1 = *(v4b *)&(B[(j + 1) * P + k]);  // bVec1 = [b13 b12 b11 b10]
        v4b bVec2 = *(v4b *)&(B[(j + 2) * P + k]);  // bVec2 = [b23 b22 b21 b20]
        v4b bVec3 = *(v4b *)&(B[(j + 3) * P + k]);  // bVec3 = [b33 b32 b31 b30]

        asm volatile(
            "vfmac.b %[sum0], %[aVec00], %[bVec0];"   // res0 += a00*b00 a00*b01 a00*b02 a00*b03
            "vfmac.b %[sum1], %[aVec10], %[bVec0];"   // res1 += a10*b00 a10*b01 a10*b02 a10*b03
            "vfmac.b %[sum2], %[aVec20], %[bVec0];"   // res2 += a20*b00 a20*b01 a20*b02 a20*b03
            "vfmac.b %[sum3], %[aVec30], %[bVec0];"   // res3 += a30*b00 a30*b01 a30*b02 a30*b03
            
            "vfmac.b %[sum0], %[aVec01], %[bVec1];"   // res0 += a01*b10 a01*b11 a01*b12 a01*b13
            "vfmac.b %[sum1], %[aVec11], %[bVec1];"   // res1 += a11*b10 a11*b11 a11*b12 a11*b13
            "vfmac.b %[sum2], %[aVec21], %[bVec1];"   // res2 += a21*b10 a21*b11 a21*b12 a21*b13
            "vfmac.b %[sum3], %[aVec31], %[bVec1];"   // res3 += a31*b10 a31*b11 a31*b12 a31*b13

            "vfmac.b %[sum0], %[aVec02], %[bVec2];"   // res0 += a02*b20 a02*b21 a02*b22 a02*b23
            "vfmac.b %[sum1], %[aVec12], %[bVec2];"   // res1 += a12*b20 a12*b21 a12*b22 a12*b23
            "vfmac.b %[sum2], %[aVec22], %[bVec2];"   // res2 += a22*b20 a22*b21 a22*b22 a22*b23
            "vfmac.b %[sum3], %[aVec32], %[bVec2];"   // res3 += a32*b20 a32*b21 a32*b22 a32*b23

            "vfmac.b %[sum0], %[aVec03], %[bVec3];"   // res0 += a03*b30 a03*b31 a03*b32 a03*b33
            "vfmac.b %[sum1], %[aVec13], %[bVec3];"   // res1 += a13*b30 a13*b31 a13*b32 a13*b33
            "vfmac.b %[sum2], %[aVec23], %[bVec3];"   // res2 += a23*b30 a23*b31 a23*b32 a23*b33
            "vfmac.b %[sum3], %[aVec33], %[bVec3];"   // res3 += a33*b30 a33*b31 a33*b32 a33*b33            
            : [sum0] "+&r"(sum0), [sum1] "+&r"(sum1), [sum2] "+&r"(sum2), [sum3] "+&r"(sum3)
            : [aVec00] "r"(aVec00), [aVec01] "r"(aVec01), [aVec02] "r"(aVec02), [aVec03] "r"(aVec03),
              [aVec10] "r"(aVec10), [aVec11] "r"(aVec11), [aVec12] "r"(aVec12), [aVec13] "r"(aVec13),
              [aVec20] "r"(aVec20), [aVec21] "r"(aVec21), [aVec22] "r"(aVec22), [aVec23] "r"(aVec23),
              [aVec30] "r"(aVec30), [aVec31] "r"(aVec31), [aVec32] "r"(aVec32), [aVec33] "r"(aVec33),
              [bVec0] "r"(bVec0), [bVec1] "r"(bVec1), [bVec2] "r"(bVec2), [bVec3] "r"(bVec3));
      }

      //dump_try(*(uint32_t*)&sum0);

      (*(v4b *)&C[i * P + k]) = sum0;
      (*(v4b *)&C[(i + 1) * P + k]) = sum1;
      (*(v4b *)&C[(i + 2) * P + k]) = sum2;
      (*(v4b *)&C[(i + 3) * P + k]) = sum3;
    }
  }
}


// Matmul based on inner product between rows of A and cols of B
  // Use 4 rows of A (store 4 elements each) and 4 cols of B (store 4 elements each)
void matmul_4x4_parallel_inner_f8vec(const __fp8 *__restrict__ A,
                                const __fp8 *__restrict__ B,
                                __fp8 *__restrict__ C, uint32_t M,
                                uint32_t N, uint32_t P, uint32_t core_id,
                                uint32_t numThreads) {

  unsigned ShuffleMask1 = 0x05070406; // [a b c d] => [c a d b]

  for (uint32_t k = core_id * 4; k < P; k += numThreads * 4) {
    for (uint32_t i = 0; i < M; i += 4) {
//      float volatile sum00 = 0.0f;
//      float volatile sum01 = 0.0f;
//      float volatile sum02 = 0.0f;
//      float volatile sum03 = 0.0f;
//      float volatile sum10 = 0.0f;
//      float volatile sum11 = 0.0f;
//      float volatile sum12 = 0.0f;
//      float volatile sum13 = 0.0f;
//      float volatile sum20 = 0.0f;
//      float volatile sum21 = 0.0f;
//      float volatile sum22 = 0.0f;
//      float volatile sum23 = 0.0f;
//      float volatile sum30 = 0.0f;
//      float volatile sum31 = 0.0f;
//      float volatile sum32 = 0.0f;
//      float volatile sum33 = 0.0f;

      float register sum00 = 0.0f;
      float register sum01 = 0.0f;
      float register sum02 = 0.0f;
      float register sum03 = 0.0f;
      float register sum10 = 0.0f;
      float register sum11 = 0.0f;
      float register sum12 = 0.0f;
      float register sum13 = 0.0f;
      float register sum20 = 0.0f;
      float register sum21 = 0.0f;
      float register sum22 = 0.0f;
      float register sum23 = 0.0f;
      float register sum30 = 0.0f;
      float register sum31 = 0.0f;
      float register sum32 = 0.0f;
      float register sum33 = 0.0f;
      for (uint32_t j = 0; j < N; j += 4) {

        v4b bVec0 = *(v4b *)&(B[j * P + k]);        // bVecTemp0 = [b03 b02 b01 b00]
        v4b bVec1 = *(v4b *)&(B[(j + 1) * P + k]);  // bVecTemp1 = [b13 b12 b11 b10]
        v4b bVec2 = *(v4b *)&(B[(j + 2) * P + k]);  // bVecTemp2 = [b23 b22 b21 b20]
        v4b bVec3 = *(v4b *)&(B[(j + 3) * P + k]);  // bVecTemp3 = [b33 b32 b31 b30]
        
        // Use aVec0, ..., aVec3 as temporary variables for col creation
        v4b aVec0, aVec1, aVec2, aVec3;
        
        // Create the cols of matrix B
        // bVecTemp holds the row, bVec store the col
        // Note: pv.pack.h packs the upper 2 bytes of the source registers
        // Note: pv.pack packs the lower 2 bytes of the source registers
        asm volatile(
            "pv.pack.h %[aVec2], %[bVec0], %[bVec1];"  // aVec2 = [b03 b02 b13 b12]
            "pv.pack.h %[aVec3], %[bVec2], %[bVec3];"  // aVec3 = [b23 b22 b33 b32]
            "pv.pack %[aVec0], %[bVec0], %[bVec1];"    // aVec0 = [b01 b00 b11 b10]
            "pv.pack %[aVec1], %[bVec2], %[bVec3];"    // aVec1 = [b21 b20 b31 b30]

            "pv.shuffle2.b %[aVec2], %[aVec2], %[ShuffleMask1];" // aVec2 = [b13 b03 b12 b02]
            "pv.shuffle2.b %[aVec3], %[aVec3], %[ShuffleMask1];" // aVec3 = [b33 b23 b32 b22]            
            "pv.shuffle2.b %[aVec0], %[aVec0], %[ShuffleMask1];" // aVec0 = [b11 b01 b10 b00]
            "pv.shuffle2.b %[aVec1], %[aVec1], %[ShuffleMask1];" // aVec1 = [b31 b21 b30 b20]

            "pv.pack.h %[bVec3], %[aVec3], %[aVec2];" // bVec3 = [b33 b23 b13 b03]
            "pv.pack %[bVec2], %[aVec3], %[aVec2];"   // bVec2 = [b32 b22 b12 b02]
            "pv.pack.h %[bVec1], %[aVec1], %[aVec0];" // bVec1 = [b31 b21 b11 b01]
            "pv.pack %[bVec0], %[aVec1], %[aVec0];"   // bVec0 = [b30 b20 b10 b00]

            : [bVec0] "+&r"(bVec0), [bVec1] "+&r"(bVec1), [bVec2] "+&r"(bVec2), [bVec3] "+&r"(bVec3),
              [aVec0] "+&r"(aVec0), [aVec1] "+&r"(aVec1), [aVec2] "+&r"(aVec2), [aVec3] "+&r"(aVec3)
            : [ShuffleMask1] "r"(ShuffleMask1));

        aVec0 = *(v4b *)&(A[i * N + j]);            // aVec0 = [a03 a02 a01 a00]
        aVec1 = *(v4b *)&(A[(i + 1) * N + j]);      // aVec1 = [a13 a12 a11 a10]
        aVec2 = *(v4b *)&(A[(i + 2) * N + j]);      // aVec1 = [a13 a12 a11 a10]
        aVec3 = *(v4b *)&(A[(i + 3) * N + j]);      // aVec1 = [a13 a12 a11 a10]

        asm volatile(
            "vfdotpexa.s.b %[sum03], %[aVec0], %[bVec3];"   // c03 += a00*b03 + a01*b13
            "vfdotpexa.s.b %[sum02], %[aVec0], %[bVec2];"
            "vfdotpexa.s.b %[sum01], %[aVec0], %[bVec1];"
            "vfdotpexa.s.b %[sum00], %[aVec0], %[bVec0];"
            "vfdotpexa.s.b %[sum13], %[aVec1], %[bVec3];"
            "vfdotpexa.s.b %[sum12], %[aVec1], %[bVec2];"
            "vfdotpexa.s.b %[sum11], %[aVec1], %[bVec1];"
            "vfdotpexa.s.b %[sum10], %[aVec1], %[bVec0];"
            "vfdotpexa.s.b %[sum23], %[aVec2], %[bVec3];"
            "vfdotpexa.s.b %[sum22], %[aVec2], %[bVec2];"
            "vfdotpexa.s.b %[sum21], %[aVec2], %[bVec1];"
            "vfdotpexa.s.b %[sum20], %[aVec2], %[bVec0];"
            "vfdotpexa.s.b %[sum33], %[aVec3], %[bVec3];"
            "vfdotpexa.s.b %[sum32], %[aVec3], %[bVec2];"
            "vfdotpexa.s.b %[sum31], %[aVec3], %[bVec1];"
            "vfdotpexa.s.b %[sum30], %[aVec3], %[bVec0];"

            "vfdotpexb.s.b %[sum03], %[aVec0], %[bVec3];"   // c03 += a02*b23 + a03*b33
            "vfdotpexb.s.b %[sum02], %[aVec0], %[bVec2];"
            "vfdotpexb.s.b %[sum01], %[aVec0], %[bVec1];"
            "vfdotpexb.s.b %[sum00], %[aVec0], %[bVec0];"
            "vfdotpexb.s.b %[sum13], %[aVec1], %[bVec3];"
            "vfdotpexb.s.b %[sum12], %[aVec1], %[bVec2];"
            "vfdotpexb.s.b %[sum11], %[aVec1], %[bVec1];"
            "vfdotpexb.s.b %[sum10], %[aVec1], %[bVec0];"
            "vfdotpexb.s.b %[sum23], %[aVec2], %[bVec3];"
            "vfdotpexb.s.b %[sum22], %[aVec2], %[bVec2];"
            "vfdotpexb.s.b %[sum21], %[aVec2], %[bVec1];"
            "vfdotpexb.s.b %[sum20], %[aVec2], %[bVec0];"
            "vfdotpexb.s.b %[sum33], %[aVec3], %[bVec3];"
            "vfdotpexb.s.b %[sum32], %[aVec3], %[bVec2];"
            "vfdotpexb.s.b %[sum31], %[aVec3], %[bVec1];"
            "vfdotpexb.s.b %[sum30], %[aVec3], %[bVec0];"
            : [sum00] "+&r"(sum00), [sum01] "+&r"(sum01), [sum02] "+&r"(sum02), [sum03] "+&r"(sum03),
              [sum10] "+&r"(sum10), [sum11] "+&r"(sum11), [sum12] "+&r"(sum12), [sum13] "+&r"(sum13),
              [sum20] "+&r"(sum20), [sum21] "+&r"(sum21), [sum22] "+&r"(sum22), [sum23] "+&r"(sum23),
              [sum30] "+&r"(sum30), [sum31] "+&r"(sum31), [sum32] "+&r"(sum32), [sum33] "+&r"(sum33)
            : [aVec0] "r"(aVec0), [aVec1] "r"(aVec1), [aVec2] "r"(aVec2), [aVec3] "r"(aVec3),
              [bVec0] "r"(bVec0), [bVec1] "r"(bVec1), [bVec2] "r"(bVec2), [bVec3] "r"(bVec3));
      }

      v4b res0, res1, res2, res3;
      asm volatile("vfcpka.b.s %[res0], %[sum00], %[sum01];"
                   "vfcpkb.b.s %[res0], %[sum02], %[sum03];"
                   "vfcpka.b.s %[res1], %[sum10], %[sum11];"
                   "vfcpkb.b.s %[res1], %[sum12], %[sum13];"
                   "vfcpka.b.s %[res2], %[sum20], %[sum21];"
                   "vfcpkb.b.s %[res2], %[sum22], %[sum23];"
                   "vfcpka.b.s %[res3], %[sum30], %[sum31];"
                   "vfcpkb.b.s %[res3], %[sum32], %[sum33];"
                   : [res0] "=&r"(res0), [res1] "=&r"(res1), [res2] "=&r"(res2), [res3] "=&r"(res3)
                   : [sum00] "r"(sum00), [sum01] "r"(sum01), [sum02] "r"(sum02), [sum03] "r"(sum03),
                     [sum10] "r"(sum10), [sum11] "r"(sum11), [sum12] "r"(sum12), [sum13] "r"(sum13),
                     [sum20] "r"(sum20), [sum21] "r"(sum21), [sum22] "r"(sum22), [sum23] "r"(sum23),
                     [sum30] "r"(sum30), [sum31] "r"(sum31), [sum32] "r"(sum32), [sum33] "r"(sum33));
      
      (*(v4b *)&C[i * P + k]) = res0;
      (*(v4b *)&C[(i + 1) * P + k]) = res1;
      (*(v4b *)&C[(i + 2) * P + k]) = res2;
      (*(v4b *)&C[(i + 3) * P + k]) = res3;
    }
  }
}


// Matmul based on inner product between rows of A and cols of B
  // Expands matrix elements from fp8 (input) to fp32 (output)
void matmul_ex_4x4_parallel_inner_f8vec(const __fp8 *__restrict__ A,
                                const __fp8 *__restrict__ B,
                                float *__restrict__ C, uint32_t M,
                                uint32_t N, uint32_t P, uint32_t core_id,
                                uint32_t numThreads) {

  uint32_t i = 0; // loop counter for M
  uint32_t j = 0; // loop counter for N
  uint32_t k = 0; // loop counter for P

  for (k = core_id * 4; k < P; k += numThreads * 4) {
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

        v4b bVecTemp0 = *(v4b *)&(B[j * P + k]);        // bVecTemp0 = [b03 b02 b01 b00]
        v4b bVecTemp1 = *(v4b *)&(B[(j + 1) * P + k]);  // bVecTemp1 = [b13 b12 b11 b10]
        v4b bVecTemp2 = *(v4b *)&(B[(j + 2) * P + k]);  // bVecTemp2 = [b23 b22 b21 b20]
        v4b bVecTemp3 = *(v4b *)&(B[(j + 3) * P + k]);  // bVecTemp3 = [b33 b32 b31 b30]
        v4b aVec0 = *(v4b *)&(A[i * N + j]);            // aVec0 = [a03 a02 a01 a00]
        v4b aVec1 = *(v4b *)&(A[(i + 1) * N + j]);      // aVec1 = [a13 a12 a11 a10]
        v4b aVec2 = *(v4b *)&(A[(i + 2) * N + j]);      // aVec1 = [a13 a12 a11 a10]
        v4b aVec3 = *(v4b *)&(A[(i + 3) * N + j]);      // aVec1 = [a13 a12 a11 a10]
        v4b bVec0 = (v4b)0.0f, bVec1 = (v4b)0.0f, bVec2 = (v4b)0.0f, bVec3 = (v4b)0.0f;
        v4b bVecLow0 = (v4b)0.0f, bVecLow1 = (v4b)0.0f, bVecLow2 = (v4b)0.0f, bVecLow3 = (v4b)0.0f;
        unsigned ShuffleMask1 = 0x05070406; // [a b c d] => [c a d b]
        unsigned ShuffleMask2 = 0x05040706; // [a b c d] => [c d a b]
        
        // Create the cols of matrix B
        // bVecTemp holds the row, bVec store the col
        // Note: pv.pack.h packs the upper 2 bytes of the source register
        // Note: pv.pack packs the lower 2 bytes of the source register
        asm volatile(
            "pv.pack.h %[bVec2], %[bVecTemp0], %[bVecTemp1];"    // bVec2 = [b03 b02 b13 b12]
            "pv.pack.h %[bVecLow2], %[bVecTemp2], %[bVecTemp3];" // bVecLow2 = [b23 b22 b33 b32]
            "pv.pack %[bVec0], %[bVecTemp0], %[bVecTemp1];"    // bVec0 = [b01 b00 b11 b10]
            "pv.pack %[bVecLow0], %[bVecTemp2], %[bVecTemp3];" // bVecLow0 = [b21 b20 b31 b30]

            "pv.shuffle2.b %[bVec2], %[bVec2], %[ShuffleMask1];"       // bVec2 = [b13 b03 b12 b02]
            "pv.shuffle2.b %[bVecLow2], %[bVecLow2], %[ShuffleMask1];" // bVecLow2 = [b33 b23 b32 b22]            
            "pv.shuffle2.b %[bVec0], %[bVec0], %[ShuffleMask1];"       // bVec0 = [b11 b01 b10 b00]
            "pv.shuffle2.b %[bVecLow0], %[bVecLow0], %[ShuffleMask1];" // bVecLow0 = [b31 b21 b30 b20]

            "pv.extract.h %[bVec3], %[bVec2], 1;"         // bVec3 = [0 0 b13 b03]
            "pv.extract.h %[bVec2], %[bVec2], 0;"         // bVec2 = [0 0 b12 b02]
            "pv.extract.h %[bVecLow3], %[bVecLow2], 1;"   // bVecLow3 = [0 0 b33 b23]
            "pv.extract.h %[bVecLow2], %[bVecLow2], 0;"   // bVecLow2 = [0 0 b32 b22]

            "pv.extract.h %[bVec1], %[bVec0], 1;"          // bVec1 = [0 0 b11 b01]
            "pv.extract.h %[bVec0], %[bVec0], 0;"          // bVec0 = [0 0 b10 b00]
            "pv.extract.h %[bVecLow1], %[bVecLow0], 1;"    // bVecLow1 = [0 0 b31 b21]
            "pv.extract.h %[bVecLow0], %[bVecLow0], 0;"    // bVecLow0 = [0 0 b30 b20]

            "pv.pack %[bVec3], %[bVecLow3], %[bVec3];" // bVec3 = [b33 b23 b13 b03]
            "pv.pack %[bVec2], %[bVecLow2], %[bVec2];" // bVec2 = [b32 b22 b12 b02]
            "pv.pack %[bVec1], %[bVecLow1], %[bVec1];" // bVec1 = [b31 b21 b11 b01]
            "pv.pack %[bVec0], %[bVecLow0], %[bVec0];" // bVec3 = [b30 b20 b10 b00]

            : [bVec0] "+&r"(bVec0), [bVec1] "+&r"(bVec1), [bVec2] "+&r"(bVec2), [bVec3] "+&r"(bVec3),
              [bVecLow0] "+&r"(bVecLow0), [bVecLow1] "+&r"(bVecLow1), [bVecLow2] "+&r"(bVecLow2), [bVecLow3] "+&r"(bVecLow3),
              [bVecTemp0] "+&r"(bVecTemp0), [bVecTemp1] "+&r"(bVecTemp1), [bVecTemp2] "+&r"(bVecTemp2),
              [bVecTemp3] "+&r"(bVecTemp3)
            : [aVec0] "r"(aVec0), [aVec1] "r"(aVec1), 
              [ShuffleMask1] "r"(ShuffleMask1), [ShuffleMask2] "r"(ShuffleMask2));

        asm volatile(
            "vfdotpexa.s.b %[sum03], %[aVec0], %[bVec3];"   // c03 += a00*b03 + a01*b13
            "vfdotpexa.s.b %[sum02], %[aVec0], %[bVec2];"
            "vfdotpexa.s.b %[sum01], %[aVec0], %[bVec1];"
            "vfdotpexa.s.b %[sum00], %[aVec0], %[bVec0];"
            "vfdotpexa.s.b %[sum13], %[aVec1], %[bVec3];"
            "vfdotpexa.s.b %[sum12], %[aVec1], %[bVec2];"
            "vfdotpexa.s.b %[sum11], %[aVec1], %[bVec1];"
            "vfdotpexa.s.b %[sum10], %[aVec1], %[bVec0];"
            "vfdotpexa.s.b %[sum23], %[aVec2], %[bVec3];"
            "vfdotpexa.s.b %[sum22], %[aVec2], %[bVec2];"
            "vfdotpexa.s.b %[sum21], %[aVec2], %[bVec1];"
            "vfdotpexa.s.b %[sum20], %[aVec2], %[bVec0];"
            "vfdotpexa.s.b %[sum33], %[aVec3], %[bVec3];"
            "vfdotpexa.s.b %[sum32], %[aVec3], %[bVec2];"
            "vfdotpexa.s.b %[sum31], %[aVec3], %[bVec1];"
            "vfdotpexa.s.b %[sum30], %[aVec3], %[bVec0];"

            "vfdotpexb.s.b %[sum03], %[aVec0], %[bVec3];"   // c03 += a02*b23 + a03*b33
            "vfdotpexb.s.b %[sum02], %[aVec0], %[bVec2];"
            "vfdotpexb.s.b %[sum01], %[aVec0], %[bVec1];"
            "vfdotpexb.s.b %[sum00], %[aVec0], %[bVec0];"
            "vfdotpexb.s.b %[sum13], %[aVec1], %[bVec3];"
            "vfdotpexb.s.b %[sum12], %[aVec1], %[bVec2];"
            "vfdotpexb.s.b %[sum11], %[aVec1], %[bVec1];"
            "vfdotpexb.s.b %[sum10], %[aVec1], %[bVec0];"
            "vfdotpexb.s.b %[sum23], %[aVec2], %[bVec3];"
            "vfdotpexb.s.b %[sum22], %[aVec2], %[bVec2];"
            "vfdotpexb.s.b %[sum21], %[aVec2], %[bVec1];"
            "vfdotpexb.s.b %[sum20], %[aVec2], %[bVec0];"
            "vfdotpexb.s.b %[sum33], %[aVec3], %[bVec3];"
            "vfdotpexb.s.b %[sum32], %[aVec3], %[bVec2];"
            "vfdotpexb.s.b %[sum31], %[aVec3], %[bVec1];"
            "vfdotpexb.s.b %[sum30], %[aVec3], %[bVec0];"
            : [sum00] "+&r"(sum00), [sum01] "+&r"(sum01), [sum02] "+&r"(sum02), [sum03] "+&r"(sum03),
              [sum10] "+&r"(sum10), [sum11] "+&r"(sum11), [sum12] "+&r"(sum12), [sum13] "+&r"(sum13),
              [sum20] "+&r"(sum20), [sum21] "+&r"(sum21), [sum22] "+&r"(sum22), [sum23] "+&r"(sum23),
              [sum30] "+&r"(sum30), [sum31] "+&r"(sum31), [sum32] "+&r"(sum32), [sum33] "+&r"(sum33)
            : [aVec0] "r"(aVec0), [aVec1] "r"(aVec1), [aVec2] "r"(aVec2), [aVec3] "r"(aVec3),
              [bVec0] "r"(bVec0), [bVec1] "r"(bVec1), [bVec2] "r"(bVec2), [bVec3] "r"(bVec3));
      }
      
      C[i * P + k] = sum00;
      C[i * P + (k + 1)] = sum01;
      C[i * P + (k + 2)] = sum02;
      C[i * P + (k + 3)] = sum03;
      C[(i + 1) * P + k] = sum10;
      C[(i + 1) * P + (k + 1)] = sum11;
      C[(i + 1) * P + (k + 2)] = sum12;
      C[(i + 1) * P + (k + 3)] = sum13;
      C[(i + 2) * P + k] = sum20;
      C[(i + 2) * P + (k + 1)] = sum21;
      C[(i + 2) * P + (k + 2)] = sum22;
      C[(i + 2) * P + (k + 3)] = sum23;
      C[(i + 3) * P + k] = sum30;
      C[(i + 3) * P + (k + 1)] = sum31;
      C[(i + 3) * P + (k + 2)] = sum32;
      C[(i + 3) * P + (k + 3)] = sum33;
    }
  }
}

