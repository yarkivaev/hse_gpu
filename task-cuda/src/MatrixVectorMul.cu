#include <MatrixVectorMul.cuh>

__global__
void MatrixVectorMul(int height, int width, float* matrix, float* vector, float* result) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= height) {
    return;
  }
  float sum = 0.f;
  for (int j = 0; j < width; ++j) {
    sum += matrix[row * width + j] * vector[j];
  }
  result[row] = sum;
}
