#include <iostream>
#include <vector>

using namespace std;

float cosine_similarity(const vector<_Float16>& v1, const vector<_Float16>& v2) {
  // Check if the vectors have the same size
  if (v1.size() != v2.size()) {
    cout << "The vectors must have the same size" << endl;
    return -1;
  }

  // Calculate the norm of v1 using the AVX intrinsic vaddps
  __m256 v1_simd = _mm256_loadu_ps((const float*)v1.data());
  __m256 norm1_simd = _mm256_dpbf16_ps(v1_simd, v1_simd, 0xF);
  float norm1 = _mm256_cvtph_ps(norm1_simd);

  // Calculate the norm of v2 using the AVX intrinsic vaddps
  __m256 v2_simd = _mm256_loadu_ps((const float*)v2.data());
  __m256 norm2_simd = _mm256_dpbf16_ps(v2_simd, v2_simd, 0xF);
  float norm2 = _mm256_cvtph_ps(norm2_simd);

  // Calculate the dot product of v1 and v2 using the AVX intrinsic vdpbf16_ps
  __m256 dot_product_simd = _mm256_dpbf16_ps(v1_simd, v2_simd, 0xF);
  float dot_product = _mm256_cvtph_ps(dot_product_simd);

  // Calculate the cosine similarity
  float similarity = dot_product / (norm1 * norm2);

  return similarity;
}

int test_fp16() {
  // Create two vectors
  vector<_Float16> v1 = {1.0f, 2.0f, 3.0f};
  vector<_Float16> v2 = {4.0f, 5.0f, 6.0f};

  // Calculate the cosine similarity
  float similarity = cosine_similarity(v1, v2);

  // Print the result
  cout << "The cosine similarity between v1 and v2 is " << similarity << endl;

  return 0;
}
