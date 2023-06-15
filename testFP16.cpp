#include <immintrin.h>

using namespace std;

#if 0
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
#endif

#if 0
#include <immintrin.h>

__m512h createM512h(const float* src) {
    // Load the floats into a 512-bit wide register
    __m512 vec_fp32 = _mm512_loadu_ps(src);

    // Convert the floats to fp16
    __m512h vec_fp16 = _mm512_castps_ph(vec_fp32);

    return vec_fp16;
}

int main() {
    // Example usage
    float floats[16] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};

    __m512h m512h_val = createM512h(floats);

    // Access the converted fp16 values
    uint16_t fp16_vals[32];
    _mm512_storeu_si512((__m512i*)fp16_vals, (__m512i)m512h_val);

    // Print the converted fp16 values
    for (int i = 0; i < 16; ++i) {
        printf("fp16_vals[%d]: %f\n", i, (float)fp16_vals[i]);
    }

    return 0;
}
#endif

float dotProductFP16(const float* a, const float* b, int size) {
    // Make sure the size is a multiple of 16 for AVX-512
    int size_aligned = (size + 15) & ~15;

    // Create vectors for the inputs
    __m512 vec_a = _mm512_loadu_ps(a);
    __m512 vec_b = _mm512_loadu_ps(b);

    // Perform the dot product using AVX-512 intrinsics
    __m512 prod = _mm512_mul_ps(vec_a, vec_b);

    // Horizontal add of the dot product results
    __m256 sum = _mm512_castps512_ps256(prod);
    __m128 hsum1 = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf32x4_ps(sum, 1));
    __m128 hsum2 = _mm_add_ps(hsum1, _mm_movehl_ps(hsum1, hsum1));
    __m128 hsum3 = _mm_add_ss(hsum2, _mm_shuffle_ps(hsum2, hsum2, 0x55));

    // Extract the dot product result
    float result;
    _mm_store_ss(&result, hsum3);

    return result;
}
