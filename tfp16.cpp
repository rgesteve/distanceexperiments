#if 0
#include <cstdio>
#include <stdint.h>  // this is for `uint16_t`, but that can be done using `unsigned short`s instead
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
#else
#include <stdio.h>
#include <immintrin.h>

void createM512h(__m512h* dst, const float* src) {
    // Load the floats into a 512-bit wide register
    __m512 vec_fp32 = _mm512_loadu_ps(src);

    // Convert the floats to fp16
    __m512h vec_fp16 = _mm512_castps_ph(vec_fp32);

    // Store the converted fp16 values in the destination array
    _mm512_storeu_si512((__m512i*)dst, (__m512i)vec_fp16);
}

int main() {
    // Example usage
    float floats[16] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};

    __m512h m512h_vals[16];
    createM512h(m512h_vals, floats);

    // Print the converted fp16 values
    for (int i = 0; i < 16; ++i) {
        unsigned short* fp16_vals = (unsigned short*)&m512h_vals[i];
        printf("fp16_vals[%d]: %f\n", i, (float)fp16_vals[0]);
    }

    return 0;
}

#endif
