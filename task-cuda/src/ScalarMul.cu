#include <ScalarMul.cuh>
#include <CommonKernels.cuh>

__global__
void ScalarMulBlock(int numElements, float* vector1, float* vector2, float* result) {
  float sum = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements; i += blockDim.x * gridDim.x) {
    sum += vector1[i] * vector2[i];
  }
  sum = blockReduceSum(sum);
  if (threadIdx.x == 0) {
    result[blockIdx.x] = sum;
  }
}
