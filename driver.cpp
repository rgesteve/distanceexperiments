#include <iostream>
#include <vector>
#include <iostream>
#include <cstdlib>

#include "dp.h"

using namespace std;

float dotProductCPU(const float*a, const float*b, int size) {
  float result = 0.f;
  for (int i = 0; i < size; i++) {
    result += a[i] * b[i];
  }
  return result;
}

int main() {
    // Test the dot product function
    float a[16] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    float b[16] = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f, 32.0f};

    cout << "Hello, world!" << endl;

    float result = dotProductFP16(a, b, 16);
    printf("Dot Product: %f\n", result);

    float result1 = dotProductBF16(a, b, 16);
    printf("Dot Product: %f\n", result1);

    float result2 = dotProductCPU(a, b, 16);
    printf("Dot Product (on the CPU): %f\n", result2);

    return EXIT_SUCCESS;
}



