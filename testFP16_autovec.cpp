//void dot(_Float16 *a, _Float16 *b, _Float16 *restrict c) {
void dot(_Float16 *a, _Float16 *b, _Float16 c) {
  for (int i = 0; i < 32; ++i)
    c+= a[i] * b[i];
}

// clang-16 -S -emit-llvm -march=sapphirerapids -O2 -mprefer-vector-width=512
