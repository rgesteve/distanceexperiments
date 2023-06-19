#include <iostream>
#include <iomanip>
#include <cmath>
#include <cfenv>
#include <cstdlib>

#include <immintrin.h>

int main()
{
  using namespace std;
  // half h = 3.14f;
  _Float16 h = 3.14f;
  _Float16 arr1[32], arr2[32], zeros[32];

  for (int i = 0; i < 32; i++) {
    arr1[i] = (_Float16)i;
    //arr2[i] = (_Float16)(i + 1);
    //arr2[i] = (_Float16)1.f;
    arr2[i] = (_Float16)2.f;
    zeros[i] = (_Float16)0.f;
  }

  __m512h in1_zmm = _mm512_loadu_ph(arr1);
  __m512h in2_zmm = _mm512_loadu_ph(arr2);
  __m512h v_zeros = _mm512_loadu_ph(zeros);
  __m512h out_zmm = _mm512_fmadd_ph(in1_zmm, in2_zmm, v_zeros);
  _Float16 res = _mm512_reduce_add_ph(out_zmm);

  cout << "Result: [" << (float)(res) << "]" << endl;
  
  return EXIT_SUCCESS;
}
