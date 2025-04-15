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

/*
void matmul_2x4_parallel_inner_f8vec(const __fp8 *__restrict__ A,
                                const __fp8 *__restrict__ B,
                                __fp8 *__restrict__ C, uint32_t M,
                                uint32_t N, uint32_t P, uint32_t core_id,
                                uint32_t numThreads) {
  
  uint32_t i = 0; // loop counter for M
  uint32_t j = 0; // loop counter for N
  uint32_t k = 0; // loop counter for P

  for (k = core_id * 4; k < P; k += numThreads * 4) {
    for (i = 0; i < M; i += 2) {
      float volatile sum00 = 0.0f;
      float volatile sum01 = 0.0f;
      float volatile sum02 = 0.0f;
      float volatile sum03 = 0.0f;
      float volatile sum10 = 0.0f;
      float volatile sum11 = 0.0f;
      float volatile sum12 = 0.0f;
      float volatile sum13 = 0.0f;
      float volatile sumTemp = 0.0f;
      for (j = 0; j < N; j += 4) {

        v4b aVec0 = *(v4b *)&(A[i * N + j]);            // aVec0 = [a03 a02 a01 a00]
        //dump_try(*(uint32_t*)&aVec0);
        v4b aVec1 = *(v4b *)&(A[(i + 1) * N + j]);
        v4b bVecTemp0 = *(v4b *)&(B[j * P + k]);        // bVecTemp0 = [b03 b02 b01 b00]
        v4b bVecTemp1 = *(v4b *)&(B[(j + 1) * P + k]);  // bVecTemp1 = [b13 b12 b11 b10]
        v4b bVecTemp2 = *(v4b *)&(B[(j + 2) * P + k]);  // bVecTemp2 = [b23 b22 b21 b20]
        v4b bVecTemp3 = *(v4b *)&(B[(j + 3) * P + k]);  // bVecTemp3 = [b33 b32 b31 b30]
        v4b bVec0, bVec1, bVec2, bVec3, bVecLow0, bVecLow1, bVecLow2, bVecLow3;
        unsigned TempShuffle1 = 0x05070406; // [a b c d] => [c a d b]
        unsigned TempShuffle2 = 0x05040706; // [a b c d] => [c d a b]

        asm volatile(
            "pv.pack.h %[bVec2], %[bVecTemp0], %[bVecTemp1];"    // bVec2 = [b03 b02 b13 b12]
            "pv.shuffle2.b %[bVec2], %[bVec2], %[TempShuffle1];" // bVec2 = [b13 b03 b12 b02]
            "pv.extract.h %[bVec3], %[bVec2], 1;" // bVec3 = [0 0 b13 b03]
            "pv.shuffle2.b %[bVec3], %[bVec3], %[TempShuffle2];" // bVec3 = [b13 b03 0 0]
            "pv.extract.h %[bVec2], %[bVec2], 0;" // bVec2 = [0 0 b12 b02]
            "pv.shuffle2.b %[bVec2], %[bVec2], %[TempShuffle2];" // bVec3 = [b12 b02 0 0]
            "pv.shuffle2.b %[bVecTemp0], %[bVecTemp0], %[TempShuffle2];" // bVecTemp0 = [b01 b00 b03 b02]
            "pv.shuffle2.b %[bVecTemp1], %[bVecTemp1], %[TempShuffle2];" // bVecTemp1 = [b11 b10 b13 b12]
            "pv.pack.h %[bVec0], %[bVecTemp0], %[bVecTemp1];"    // bVec0 = [b01 b00 b11 b10]
            "pv.shuffle2.b %[bVec0], %[bVec0], %[TempShuffle1];" // bVec0 = [b11 b01 b10 b00]
            "pv.extract.h %[bVec1], %[bVec0], 1;" // bVec1 = [0 0 b11 b01]
            "pv.shuffle2.b %[bVec1], %[bVec1], %[TempShuffle2];" // bVec3 = [b11 b01 0 0]
            "pv.extract.h %[bVec0], %[bVec0], 0;" // bVec0 = [0 0 b10 b00]
            "pv.shuffle2.b %[bVec0], %[bVec0], %[TempShuffle2];" // bVec3 = [b10 b00 0 0]

            "pv.pack.h %[bVecLow2], %[bVecTemp2], %[bVecTemp3];"       // bVecLow2 = [b23 b22 b33 b32]
            "pv.shuffle2.b %[bVecLow2], %[bVecLow2], %[TempShuffle1];" // bVecLow2 = [b33 b23 b32 b22]
            "pv.extract.h %[bVecLow3], %[bVecLow2], 1;" // bVecLow3 = [0 0 b33 b23]
            "pv.shuffle2.b %[bVecLow3], %[bVecLow3], %[TempShuffle2];" // bVecLow3 = [b33 b23 0 0]            
            "pv.pack.h %[bVec3], %[bVecLow3], %[bVec3];" // bVec3 = [b33 b23 b13 b03]
            "pv.extract.h %[bVecLow2], %[bVecLow2], 0;"  // bVecLow2 = [0 0 b32 b22]
            "pv.shuffle2.b %[bVecLow2], %[bVecLow2], %[TempShuffle2];" // bVecLow2 = [b32 b22 0 0]            
            "pv.pack.h %[bVec2], %[bVecLow2], %[bVec2];" // bVec2 = [b32 b22 b12 b02]
            "pv.shuffle2.b %[bVecTemp2], %[bVecTemp2], %[TempShuffle2];" // bVecTemp0 = [b21 b20 b23 b22]
            "pv.shuffle2.b %[bVecTemp3], %[bVecTemp3], %[TempShuffle2];" // bVecTemp1 = [b31 b30 b33 b32]
            "pv.pack.h %[bVecLow0], %[bVecTemp2], %[bVecTemp3];"       // bVecLow0 = [b21 b20 b31 b30]
            "pv.shuffle2.b %[bVecLow0], %[bVecLow0], %[TempShuffle1];" // bVecLow0 = [b31 b21 b30 b20]
            "pv.extract.h %[bVecLow1], %[bVecLow0], 1;" // bVecLow1 = [0 0 b31 b21]
            "pv.shuffle2.b %[bVecLow1], %[bVecLow1], %[TempShuffle2];" // bVecLow1 = [b31 b21 0 0]            
            "pv.pack.h %[bVec1], %[bVecLow1], %[bVec1];" // bVec1 = [b31 b21 b11 b01]
            "pv.extract.h %[bVecLow0], %[bVecLow0], 0;" // bVecLow0 = [0 0 b30 b20]
            "pv.shuffle2.b %[bVecLow0], %[bVecLow0], %[TempShuffle2];" // bVecLow0 = [b30 b20 0 0]            
            "pv.pack.h %[bVec0], %[bVecLow0], %[bVec0];" // bVec3 = [b30 b20 b10 b00]
            : [bVec0] "+&r"(bVec0), [bVec1] "+&r"(bVec1), [bVec2] "+&r"(bVec2), [bVec3] "+&r"(bVec3),
              [bVecLow0] "+&r"(bVecLow0), [bVecLow1] "+&r"(bVecLow1), [bVecLow2] "+&r"(bVecLow2), [bVecLow3] "+&r"(bVecLow3),
              [bVecTemp0] "+&r"(bVecTemp0), [bVecTemp1] "+&r"(bVecTemp1), [bVecTemp2] "+&r"(bVecTemp2),
              [bVecTemp3] "+&r"(bVecTemp3)
            : [aVec0] "r"(aVec0), [aVec1] "r"(aVec1), 
              [TempShuffle1] "r"(TempShuffle1), [TempShuffle2] "r"(TempShuffle2));

        dump_try(*(uint32_t*)&bVec3);
        dump_try(*(uint32_t*)&bVec2);
        dump_try(*(uint32_t*)&bVec1);
        dump_try(*(uint32_t*)&bVec0);
        //dump_try(*(uint32_t*)&sum00);
        //dump_try(*(uint32_t*)&bVecLow0);

        asm volatile(
            "vfdotpexa.s.b %[sum00], %[aVec0], %[bVec0];"   // res0 = a00*b00 + a01*b10
            "vfdotpexb.s.b %[sumTemp], %[aVec0], %[bVec0];" // res1 = a02*b20 + a03*b30
            "fadd.s %[sum00], %[sum00], %[sumTemp];"        // sum00 = res0 + res1
            "vfdotpexa.s.b %[sum01], %[aVec0], %[bVec1];"
            "vfdotpexb.s.b %[sumTemp], %[aVec0], %[bVec1];"
            "fadd.s %[sum01], %[sum01], %[sumTemp];"
            "vfdotpexa.s.b %[sum02], %[aVec0], %[bVec2];"
            "vfdotpexb.s.b %[sumTemp], %[aVec0], %[bVec2];"
            "fadd.s %[sum02], %[sum02], %[sumTemp];"            
            "vfdotpexa.s.b %[sum03], %[aVec0], %[bVec3];"
            "vfdotpexb.s.b %[sumTemp], %[aVec0], %[bVec3];"
            "fadd.s %[sum03], %[sum03], %[sumTemp];"            
            "vfdotpexa.s.b %[sum10], %[aVec1], %[bVec0];"
            "vfdotpexb.s.b %[sumTemp], %[aVec1], %[bVec0];"
            "fadd.s %[sum10], %[sum03], %[sumTemp];"
            "vfdotpexa.s.b %[sum11], %[aVec1], %[bVec1];"
            "vfdotpexb.s.b %[sumTemp], %[aVec1], %[bVec1];"
            "fadd.s %[sum11], %[sum03], %[sumTemp];"
            "vfdotpexa.s.b %[sum12], %[aVec1], %[bVec2];"
            "vfdotpexb.s.b %[sumTemp], %[aVec1], %[bVec2];"
            "fadd.s %[sum12], %[sum03], %[sumTemp];"
            "vfdotpexa.s.b %[sum13], %[aVec1], %[bVec3];"
            "vfdotpexb.s.b %[sumTemp], %[aVec1], %[bVec3];"
            "fadd.s %[sum13], %[sum03], %[sumTemp];"
            : [sum00] "+&r"(sum00), [sum01] "+&r"(sum01), [sum02] "+&r"(sum02), [sum03] "+&r"(sum03),
              [sum10] "+&r"(sum10), [sum11] "+&r"(sum11), [sum12] "+&r"(sum12), [sum13] "+&r"(sum13),
              [sumTemp] "+&r"(sumTemp)
            : [aVec0] "r"(aVec0), [aVec1] "r"(aVec1), 
              [bVec0] "r"(bVec0), [bVec1] "r"(bVec1), [bVec2] "r"(bVec2), [bVec3] "r"(bVec3));
        
        dump_try(*(uint32_t*)&sum00);
        //dump_try(*(uint32_t*)&aVec1);
        //dump_try(*(uint32_t*)&sum00);
        //dump_try(*(uint32_t*)&sum01);
        //dump_try(*(uint32_t*)&sumTemp);
      }
    }
  }
}
*/



void matmul_2x4_parallel_inner_f8vec(const __fp8 *__restrict__ A,
                                const __fp8 *__restrict__ B,
                                __fp8 *__restrict__ C, uint32_t M,
                                uint32_t N, uint32_t P, uint32_t core_id,
                                uint32_t numThreads) {

  uint32_t i = 0; // loop counter for M
  uint32_t j = 0; // loop counter for N
  uint32_t k = 0; // loop counter for P

  for (k = core_id * 4; k < P; k += numThreads * 4) {
    for (i = 0; i < M; i += 2) {
      float volatile sum00 = 0.0f;
      float volatile sum01 = 0.0f;
      float volatile sum02 = 0.0f;
      float volatile sum03 = 0.0f;
      float volatile sum10 = 0.0f;
      float volatile sum11 = 0.0f;
      float volatile sum12 = 0.0f;
      float volatile sum13 = 0.0f;
      for (j = 0; j < N; j += 4) {

        v4b aVec0 = *(v4b *)&(A[i * N + j]);            // aVec0 = [a03 a02 a01 a00]
        //dump_try(*(uint32_t*)&aVec0);
        v4b aVec1 = *(v4b *)&(A[(i + 1) * N + j]);
        v4b bVecTemp0 = *(v4b *)&(B[j * P + k]);        // bVecTemp0 = [b03 b02 b01 b00]
        v4b bVecTemp1 = *(v4b *)&(B[(j + 1) * P + k]);  // bVecTemp1 = [b13 b12 b11 b10]
        v4b bVecTemp2 = *(v4b *)&(B[(j + 2) * P + k]);  // bVecTemp2 = [b23 b22 b21 b20]
        v4b bVecTemp3 = *(v4b *)&(B[(j + 3) * P + k]);  // bVecTemp3 = [b33 b32 b31 b30]
        dump_try(*(uint32_t*)&bVecTemp0);
        v4b bVec0, bVec1, bVec2, bVec3, bVecLow0, bVecLow1, bVecLow2, bVecLow3;
        unsigned TempShuffle1 = 0x05070406; // [a b c d] => [c a d b]
        unsigned TempShuffle2 = 0x05040706; // [a b c d] => [c d a b]
        asm volatile(
            "pv.pack.h %[bVec2], %[bVecTemp0], %[bVecTemp1];"    // bVec2 = [b03 b02 b13 b12]
            "pv.shuffle2.b %[bVec2], %[bVec2], %[TempShuffle1];" // bVec2 = [b13 b03 b12 b02]
            "pv.extract.h %[bVec3], %[bVec2], 1;" // bVec3 = [0 0 b13 b03]
            "pv.shuffle2.b %[bVec3], %[bVec3], %[TempShuffle2];" // bVec3 = [b13 b03 0 0]
            "pv.extract.h %[bVec2], %[bVec2], 0;" // bVec2 = [0 0 b12 b02]
            "pv.shuffle2.b %[bVec2], %[bVec2], %[TempShuffle2];" // bVec3 = [b12 b02 0 0]
            "pv.shuffle2.b %[bVecTemp0], %[bVecTemp0], %[TempShuffle2];" // bVecTemp0 = [b01 b00 b03 b02]
            "pv.shuffle2.b %[bVecTemp1], %[bVecTemp1], %[TempShuffle2];" // bVecTemp1 = [b11 b10 b13 b12]
            "pv.pack.h %[bVec0], %[bVecTemp0], %[bVecTemp1];"    // bVec0 = [b01 b00 b11 b10]
            "pv.shuffle2.b %[bVec0], %[bVec0], %[TempShuffle1];" // bVec0 = [b11 b01 b10 b00]
            "pv.extract.h %[bVec1], %[bVec0], 1;" // bVec1 = [0 0 b11 b01]
            "pv.shuffle2.b %[bVec1], %[bVec1], %[TempShuffle2];" // bVec3 = [b11 b01 0 0]
            "pv.extract.h %[bVec0], %[bVec0], 0;" // bVec0 = [0 0 b10 b00]
            "pv.shuffle2.b %[bVec0], %[bVec0], %[TempShuffle2];" // bVec3 = [b10 b00 0 0]

            "pv.pack.h %[bVecLow2], %[bVecTemp2], %[bVecTemp3];"       // bVecLow2 = [b23 b22 b33 b32]
            "pv.shuffle2.b %[bVecLow2], %[bVecLow2], %[TempShuffle1];" // bVecLow2 = [b33 b23 b32 b22]
            "pv.extract.h %[bVecLow3], %[bVecLow2], 1;" // bVecLow3 = [0 0 b33 b23]
            "pv.shuffle2.b %[bVecLow3], %[bVecLow3], %[TempShuffle2];" // bVecLow3 = [b33 b23 0 0]            
            "pv.pack.h %[bVec3], %[bVecLow3], %[bVec3];" // bVec3 = [b33 b23 b13 b03]
            "pv.extract.h %[bVecLow2], %[bVecLow2], 0;"  // bVecLow2 = [0 0 b32 b22]
            "pv.shuffle2.b %[bVecLow2], %[bVecLow2], %[TempShuffle2];" // bVecLow2 = [b32 b22 0 0]            
            "pv.pack.h %[bVec2], %[bVecLow2], %[bVec2];" // bVec2 = [b32 b22 b12 b02]
            "pv.shuffle2.b %[bVecTemp2], %[bVecTemp2], %[TempShuffle2];" // bVecTemp0 = [b21 b20 b23 b22]
            "pv.shuffle2.b %[bVecTemp3], %[bVecTemp3], %[TempShuffle2];" // bVecTemp1 = [b31 b30 b33 b32]
            "pv.pack.h %[bVecLow0], %[bVecTemp2], %[bVecTemp3];"       // bVecLow0 = [b21 b20 b31 b30]
            "pv.shuffle2.b %[bVecLow0], %[bVecLow0], %[TempShuffle1];" // bVecLow0 = [b31 b21 b30 b20]
            "pv.extract.h %[bVecLow1], %[bVecLow0], 1;" // bVecLow1 = [0 0 b31 b21]
            "pv.shuffle2.b %[bVecLow1], %[bVecLow1], %[TempShuffle2];" // bVecLow1 = [b31 b21 0 0]            
            "pv.pack.h %[bVec1], %[bVecLow1], %[bVec1];" // bVec1 = [b31 b21 b11 b01]
            "pv.extract.h %[bVecLow0], %[bVecLow0], 0;" // bVecLow0 = [0 0 b30 b20]
            "pv.shuffle2.b %[bVecLow0], %[bVecLow0], %[TempShuffle2];" // bVecLow0 = [b30 b20 0 0]            
            "pv.pack.h %[bVec0], %[bVecLow0], %[bVec0];" // bVec3 = [b30 b20 b10 b00]
            : [bVec0] "+&r"(bVec0), [bVec1] "+&r"(bVec1), [bVec2] "+&r"(bVec2), [bVec3] "+&r"(bVec3),
              [bVecLow0] "+&r"(bVecLow0), [bVecLow1] "+&r"(bVecLow1), [bVecLow2] "+&r"(bVecLow2), [bVecLow3] "+&r"(bVecLow3),
              [bVecTemp0] "+&r"(bVecTemp0), [bVecTemp1] "+&r"(bVecTemp1), [bVecTemp2] "+&r"(bVecTemp2),
              [bVecTemp3] "+&r"(bVecTemp3)
            : [aVec0] "r"(aVec0), [aVec1] "r"(aVec1), 
              [TempShuffle1] "r"(TempShuffle1), [TempShuffle2] "r"(TempShuffle2));

        //dump_try(*(uint32_t*)&bVec0);
        //dump_try(*(uint32_t*)&bVec1);
        //dump_try(*(uint32_t*)&bVec2);
        //dump_try(*(uint32_t*)&bVec3);

        asm volatile(
            "vfdotpexa.s.b %[sum00], %[aVec0], %[bVec0];"   // res0 += a00*b00 + a01*b10
            "vfdotpexb.s.b %[sum00], %[aVec0], %[bVec0];"   // res0 += a02*b20 + a03*b30
            "vfdotpexa.s.b %[sum01], %[aVec0], %[bVec1];"
            "vfdotpexb.s.b %[sum01], %[aVec0], %[bVec1];"
            "vfdotpexa.s.b %[sum02], %[aVec0], %[bVec2];"
            "vfdotpexb.s.b %[sum02], %[aVec0], %[bVec2];"
            "vfdotpexa.s.b %[sum03], %[aVec0], %[bVec3];"
            "vfdotpexb.s.b %[sum03], %[aVec0], %[bVec3];"
            "vfdotpexa.s.b %[sum10], %[aVec1], %[bVec0];"
            "vfdotpexb.s.b %[sum10], %[aVec1], %[bVec0];"
            "vfdotpexa.s.b %[sum11], %[aVec1], %[bVec1];"
            "vfdotpexb.s.b %[sum11], %[aVec1], %[bVec1];"
            "vfdotpexa.s.b %[sum12], %[aVec1], %[bVec2];"
            "vfdotpexb.s.b %[sum12], %[aVec1], %[bVec2];"
            "vfdotpexa.s.b %[sum13], %[aVec1], %[bVec3];"
            "vfdotpexb.s.b %[sum13], %[aVec1], %[bVec3];"
            : [sum00] "+&r"(sum00), [sum01] "+&r"(sum01), [sum02] "+&r"(sum02), [sum03] "+&r"(sum03),
              [sum10] "+&r"(sum10), [sum11] "+&r"(sum11), [sum12] "+&r"(sum12), [sum13] "+&r"(sum13)
            : [aVec0] "r"(aVec0), [aVec1] "r"(aVec1), 
              [bVec0] "r"(bVec0), [bVec1] "r"(bVec1), [bVec2] "r"(bVec2), [bVec3] "r"(bVec3));
      }

      //dump_try(*(uint32_t*)&sum00);
      //dump_try(*(uint32_t*)&sum01);
      //dump_try(*(uint32_t*)&sum02);
      //dump_try(*(uint32_t*)&sum03);

      v4b res0, res1;
      asm volatile("vfcpka.b.s %[res0], %[sum00], %[sum01];"
                   "vfcpkb.b.s %[res0], %[sum02], %[sum03];"
                   "vfcpka.b.s %[res1], %[sum10], %[sum11];"
                   "vfcpkb.b.s %[res1], %[sum12], %[sum13];"
                   : [res0] "=&r"(res0), [res1] "=&r"(res1)
                   : [sum00] "r"(sum00), [sum01] "r"(sum01), [sum02] "r"(sum02), [sum03] "r"(sum03),
                     [sum10] "r"(sum10), [sum11] "r"(sum11), [sum12] "r"(sum12), [sum13] "r"(sum13));
      
      //dump_try(*(uint32_t*)&res0);

      (*(v4b *)&C[i * P + k]) = res0;
      (*(v4b *)&C[(i + 1) * P + k]) = res1;
    }
  }
}


/*
void matmul_4x2_parallel_inner_f8vec(const __fp8 *__restrict__ A,
                                const __fp8 *__restrict__ B,
                                __fp8 *__restrict__ C, uint32_t M,
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
      for (j = 0; j < N; j += 2) {

        v4b aVec0 = *(v4b *)&(A[i * N + j*2]);
        v4b aVec1 = *(v4b *)&(A[(i + 1) * N + j*2]);
        v4b aVec2 = *(v4b *)&(A[(i + 2) * N + j*2]);
        v4b aVec3 = *(v4b *)&(A[(i + 3) * N + j*2]);
        v4b bVecTemp0 = *(v4b *)&(B[j * P + k]);        // bVecTemp0 = [b00 b01 b02 b03]
        v4b bVecTemp1 = *(v4b *)&(B[(j + 1) * P + k]);  // bVecTemp1 = [b10 b11 b12 b13]
        v4b bVec0, bVec1, bVec2, bVec3;
        unsigned TempShuffle1 = 0b0110 0100 0111 0101; // [a b c d] => [b d a c]
        unsigned TempShuffle2 = 0b0101 0100 0111 0110; // [a b c d] => [c d a b]
        asm volatile(
            "pv.pack.h %[bVec2], %[bVecTemp0], %[bVecTemp1];"    // bVec2 = [b02 b03 b12 b13]
            "pv.shuffle2.b %[bVec2], %[bVec2], %[TempShuffle1];" // bVec2 = [b03 b13 b02 b12]
            "pv.extract.h %[bVec3], %[bVec2], 1;" // bVec3 = [b03 b13]
            "pv.extract.h %[bVec2], %[bVec2], 0;" // bVec2 = [b02 b12]
            "pv.shuffle2.b %[bVecTemp0], %[bVecTemp0], %[TempShuffle2];" // bVecTemp0 = [b02 b03 b00 b01]
            "pv.shuffle2.b %[bVecTemp1], %[bVecTemp1], %[TempShuffle2];" // bVecTemp1 = [b12 b13 b10 b11]
            "pv.pack.h %[bVec0], %[bVecTemp0], %[bVecTemp1];"    // bVec0 = [b00 b01 b10 b11]
            "pv.shuffle2.b %[bVec0], %[bVec0], %[TempShuffle1];" // bVec0 = [b01 b11 b00 b10]
            "pv.extract.h %[bVec1], %[bVec0], 1;" // bVec1 = [b01 b11]
            "pv.extract.h %[bVec0], %[bVec0], 0;" // bVec0 = [b00 b10]

            "vfdotpexa.s.b %[sum00], %[aVec0], %[bVec0];"
            "vfdotpexa.s.b %[sum01], %[aVec0], %[bVec1];"
            "vfdotpexa.s.b %[sum02], %[aVec0], %[bVec2];"
            "vfdotpexa.s.b %[sum03], %[aVec0], %[bVec3];"
            "vfdotpexa.s.b %[sum10], %[aVec1], %[bVec0];"
            "vfdotpexa.s.b %[sum11], %[aVec1], %[bVec1];"
            "vfdotpexa.s.b %[sum12], %[aVec1], %[bVec2];"
            "vfdotpexa.s.b %[sum13], %[aVec1], %[bVec3];"
            "vfdotpexa.s.b %[sum20], %[aVec2], %[bVec0];"
            "vfdotpexa.s.b %[sum21], %[aVec2], %[bVec1];"
            "vfdotpexa.s.b %[sum22], %[aVec2], %[bVec2];"
            "vfdotpexa.s.b %[sum23], %[aVec2], %[bVec3];"
            "vfdotpexa.s.b %[sum30], %[aVec3], %[bVec0];"
            "vfdotpexa.s.b %[sum31], %[aVec3], %[bVec1];"
            "vfdotpexa.s.b %[sum32], %[aVec3], %[bVec2];"
            "vfdotpexa.s.b %[sum33], %[aVec3], %[bVec3];"
            : [sum00] "+&r"(sum00), [sum01] "+&r"(sum01), [sum02] "+&r"(sum02), [sum03] "+&r"(sum03),
              [sum10] "+&r"(sum10), [sum11] "+&r"(sum11), [sum12] "+&r"(sum12), [sum13] "+&r"(sum13),
              [sum20] "+&r"(sum20), [sum21] "+&r"(sum21), [sum22] "+&r"(sum22), [sum23] "+&r"(sum23),
              [sum30] "+&r"(sum30), [sum31] "+&r"(sum31), [sum32] "+&r"(sum32), [sum33] "+&r"(sum33),
              [bVec0] "+&r"(bVec0), [bVec1] "+&r"(bVec1), [bVec2] "+&r"(bVec2), [bVec3] "+&r"(bVec3),
              [bVecTemp0] "+&r"(bVecTemp0), [bVecTemp1] "+&r"(bVecTemp1)
            : [aVec0] "r"(aVec0), [aVec1] "r"(aVec1), [aVec2] "r"(aVec2), [aVec3] "r"(aVec3), 
              [TempShuffle1] "r"(TempShuffle1), [TempShuffle2] "r"(TempShuffle2));
      }
      v4b res0, res1, res2, res3;
      asm volatile("vfcpka.b.s %[res0], %[sum03], %[sum02];"
                   "vfcpkb.b.s %[res0], %[sum01], %[sum00];"
                   "vfcpka.b.s %[res1], %[sum13], %[sum12];"
                   "vfcpkb.b.s %[res1], %[sum11], %[sum10];"
                   "vfcpka.b.s %[res2], %[sum23], %[sum22];"
                   "vfcpkb.b.s %[res2], %[sum21], %[sum20];"
                   "vfcpka.b.s %[res3], %[sum33], %[sum32];"
                   "vfcpkb.b.s %[res3], %[sum31], %[sum30];"
                   : [res0] "=&r"(res0), [res1] "=&r"(res1), [res2] "=&r"(res2),
                     [res3] "=&r"(res3)
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
*/


/*
// Uses the inner product
void matmul_4x2_parallel_inner_f8(const __fp8 *__restrict__ A,
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
            "vfdotpexa.s.b %[sum00], %[aVec0], %[bVec0];"
            "vfdotpexa.s.b %[sum01], %[aVec0], %[bVec1];"
            "vfdotpexa.s.b %[sum02], %[aVec0], %[bVec2];"
            "vfdotpexa.s.b %[sum03], %[aVec0], %[bVec3];"
            "vfdotpexa.s.b %[sum10], %[aVec1], %[bVec0];"
            "vfdotpexa.s.b %[sum11], %[aVec1], %[bVec1];"
            "vfdotpexa.s.b %[sum12], %[aVec1], %[bVec2];"
            "vfdotpexa.s.b %[sum13], %[aVec1], %[bVec3];"
            "vfdotpexa.s.b %[sum20], %[aVec2], %[bVec0];"
            "vfdotpexa.s.b %[sum21], %[aVec2], %[bVec1];"
            "vfdotpexa.s.b %[sum22], %[aVec2], %[bVec2];"
            "vfdotpexa.s.b %[sum23], %[aVec2], %[bVec3];"
            "vfdotpexa.s.b %[sum30], %[aVec3], %[bVec0];"
            "vfdotpexa.s.b %[sum31], %[aVec3], %[bVec1];"
            "vfdotpexa.s.b %[sum32], %[aVec3], %[bVec2];"
            "vfdotpexa.s.b %[sum32], %[aVec3], %[bVec3];"
            : [sum00] "+&r"(sum00), [sum01] "+&r"(sum01), [sum02] "+&r"(sum02), [sum03] "+&r"(sum03),
              [sum10] "+&r"(sum10), [sum11] "+&r"(sum11), [sum12] "+&r"(sum12), [sum13] "+&r"(sum13),
              [sum20] "+&r"(sum20), [sum21] "+&r"(sum21), [sum22] "+&r"(sum22), [sum23] "+&r"(sum23),
              [sum30] "+&r"(sum30), [sum31] "+&r"(sum31), [sum32] "+&r"(sum32), [sum33] "+&r"(sum33),
              [bVec0] "+&r"(bVec0), [bVec1] "+&r"(bVec1), [bVec2] "+&r"(bVec2), [bVec3] "+&r"(bVec3),
              [TempH] "+&r"(TempH), [TempL] "+&r"(TempL)
            : [aVec0] "r"(aVec0), [aVec1] "r"(aVec1), [aVec2] "r"(aVec2),
              [aVec3] "r"(aVec3), [bVecTemp0] "r"(bVecTemp0),
              [bVecTemp1] "r"(bVecTemp1));
      }
      v4b res0, res1, res2, res3;
      asm volatile("vfcpka.b.s %[res0], %[sum03], %[sum02];"
                   "vfcpkb.b.s %[res0], %[sum01], %[sum00];"
                   "vfcpka.b.s %[res1], %[sum13], %[sum12];"
                   "vfcpkb.b.s %[res1], %[sum11], %[sum10];"
                   "vfcpka.b.s %[res2], %[sum23], %[sum22];"
                   "vfcpkb.b.s %[res2], %[sum21], %[sum20];"
                   "vfcpka.b.s %[res3], %[sum33], %[sum32];"
                   "vfcpkb.b.s %[res3], %[sum31], %[sum30];"
                   : [res0] "=&r"(res0), [res1] "=&r"(res1), [res2] "=&r"(res2),
                     [res3] "=&r"(res3)
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
*/