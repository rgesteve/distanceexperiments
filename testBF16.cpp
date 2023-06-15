//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <immintrin.h>
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <cstdlib>

using namespace std;

float op1_f32[16];
float op2_f32[16];
#if 0
float op3_f32[16];
float op4_f32[16];
#endif
float res_f32[16];
float res_comp_f32[16];

#if 0
void foo() {
     // Choose some sample values for arrays
   float v = sqrt(2);
   for (int i = 0; i < 16; i++)
   {
       op1_f32[i]  = v;
       op2_f32[i]  = (float)i;
#if 0
       op3_f32[i]  = 0.0;
#endif
       res_f32[i]   = 0.0;
       // Compute result of dot product operation using float32 (for comparison with bf16)
       res_comp_f32[i] =  2.0 * op1_f32[i] * op1_f32[i] + res_f32[i]; // the "2.0" is because we're assuming both operands have the same content
   }

   // Display input values
   cout << endl;
   printf("INPUT TO BF16 INSTRUCTION: \n");
   printf("  First float32 vector, each containing values: \n");
   for (int j = 15; j >= 0; j--){
       cout << "  "<<op1_f32[j]<<" ";
   }
   cout << endl<<endl;
   printf("  Second float32 vector, each containing values: \n");
   for (int j = 15; j >= 0; j--){
       cout << "  "<<op2_f32[j]<<" ";
   }
   cout << endl<<endl;
   printf("  One float32 vector (input/output vector), containing values: \n");
   for (int j = 15; j >= 0; j--){
       cout <<"  "<< res_f32[j]<<" ";
   }
   cout << endl;
}

void bar() {
   // Display results
   cout << endl;
   printf("RESULTS OF DOT PRODUCT USING BF16 INSTRUCTION: \n");
   for (int j = 15; j >= 0; j--){
       cout << res_f32[j]<<" ";
   }
   cout << endl;

   cout << endl;
   printf("RESULTS OF DOT PRODUCT USING FLOAT32 INSTRUCTIONS : \n");
   for (int j = 15; j >= 0; j--){
       cout << res_comp_f32[j] <<" ";
   }
   cout << endl;
   cout << endl;
}

// void dot_fp16(float op1_f32[16], float op2_f32[16], float op3_f32[16], float res_f32[16], float res_comp_f32[16])
void dot_fp16(float dummy[16])
{
   // register variables
   // Load 16 float32 values into registers (data does not need to be aligned on any particular boundary)
   __m512 v1_f32 =_mm512_loadu_ps(op1_f32);
   __m512 v2_f32 =_mm512_loadu_ps(op2_f32);
   __m512 vr_f32 =_mm512_loadu_ps(res_f32);

#if 0
   // Convert two float32 registers (16 values each) to one BF16 register #1 (32 values)
   __m512bh v1_f16 = _mm512_cvtne2ps_pbh(v1_f32, v2_f32);

   // Convert two float32 registers (16 values each) to one BF16 register #2 (32 values)
   __m512bh v2_f16 = _mm512_cvtne2ps_pbh(v1_f32, v2_f32);
#endif
   __m512bh v1_f16 = _mm512_cvtne2ps_pbh(v1_f32, v1_f32);
   __m512bh v2_f16 = _mm512_cvtne2ps_pbh(v2_f32, v2_f32);

   // FMA: Performs dot product of BF16 registers #1 and #2. Accumulate result into one float32 output register
   vr_f32 = _mm512_dpbf16_ps(vr_f32, v1_f16, v2_f16);
   //vr_f32 = _mm512_dpbf16_ps(v3_f32, v1_f16, v2_f16);

   // Copy output register to memory (memory address does not need to be aligned on any particular boundary)
   _mm512_storeu_ps((void *) res_f32, vr_f32);

   float red = _mm512_reduce_add_ps(vr_f32);
   cout << "The reduction is: [" << red << "]" << endl;
}
#endif

float dotProductBF16(const float* a, const float* b, int size)
{
  float resfp32[16];
  float zeros[16];

  for (int i = 0; i < 16; i++) {
       resfp32[i]   = 0.0;
       zeros[i] = 0.0;
  }

   // register variables
   // Load 16 float32 values into registers (data does not need to be aligned on any particular boundary)
  __m512 v1_f32 =_mm512_loadu_ps(a);
  __m512 v2_f32 =_mm512_loadu_ps(b);
  __m512 vr_f32 =_mm512_loadu_ps(resfp32);
  __m512 vr_zeros =_mm512_loadu_ps(zeros);

#if 0
   // Convert two float32 registers (16 values each) to one BF16 register #1 (32 values)
   __m512bh v1_f16 = _mm512_cvtne2ps_pbh(v1_f32, v2_f32);

   // Convert two float32 registers (16 values each) to one BF16 register #2 (32 values)
   __m512bh v2_f16 = _mm512_cvtne2ps_pbh(v1_f32, v2_f32);
#endif
   __m512bh v1_f16 = _mm512_cvtne2ps_pbh(v1_f32, vr_zeros);
   __m512bh v2_f16 = _mm512_cvtne2ps_pbh(v2_f32, vr_zeros);


   // FMA: Performs dot product of BF16 registers #1 and #2. Accumulate result into one float32 output register
   vr_f32 = _mm512_dpbf16_ps(vr_f32, v1_f16, v2_f16);
   //vr_f32 = _mm512_dpbf16_ps(v3_f32, v1_f16, v2_f16);

#if 0
   // Copy output register to memory (memory address does not need to be aligned on any particular boundary)
   _mm512_storeu_ps((void *) resfp32, vr_f32);
#endif

   float red = _mm512_reduce_add_ps(vr_f32);
   return red;
}



