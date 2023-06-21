#include <iostream>
#include <iomanip>
#include <cmath>
#include <cfenv>
#include <cstdlib>

#include <immintrin.h>

#include <benchmark/benchmark.h>

float dotProductFP16(const _Float16* a, const _Float16* b, size_t size)
{
  _Float16 zeros[size];
  for (int i = 0; i < size; i++) {
    zeros[i] = (_Float16)0.f;
  }

  __m512h in1_zmm = _mm512_loadu_ph(a);
  __m512h in2_zmm = _mm512_loadu_ph(b);
  __m512h v_zeros = _mm512_loadu_ph(zeros);
  __m512h out_zmm = _mm512_fmadd_ph(in1_zmm, in2_zmm, v_zeros);
  _Float16 res = _mm512_reduce_add_ph(out_zmm);

  return (float)(res);
}

void BM_loop_dp_fp16(benchmark::State& state)
{
  using namespace std;
  _Float16 h = 3.14f;
  _Float16 arr1[32], arr2[32], zeros[32];

  for (int i = 0; i < 32; i++) {
    arr1[i] = (_Float16)i;
    //arr2[i] = (_Float16)(i + 1);
    //arr2[i] = (_Float16)1.f;
    arr2[i] = (_Float16)2.f;
    zeros[i] = (_Float16)0.f;
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(dotProductFP16(arr1, arr2, 32));
  }
}

BENCHMARK(BM_loop_dp_fp16);
BENCHMARK_MAIN();

#if 0
int main()
{
  using namespace std;
  _Float16 h = 3.14f;
  _Float16 arr1[32], arr2[32], zeros[32];

  for (int i = 0; i < 32; i++) {
    arr1[i] = (_Float16)i;
    //arr2[i] = (_Float16)(i + 1);
    //arr2[i] = (_Float16)1.f;
    arr2[i] = (_Float16)2.f;
    zeros[i] = (_Float16)0.f;
  }

  float res = dotProductFP16(arr1, arr2, 32);

  cout << "Result: [" << res << "]" << endl;
  
  return EXIT_SUCCESS;
}
#endif
