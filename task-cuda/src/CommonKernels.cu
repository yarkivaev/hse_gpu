#include <CommonKernels.cuh>

__global__ void ReduceSumKernel(int numElements, float* data, float* result) {
  float sum = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements; i += blockDim.x * gridDim.x) {
    sum += data[i];
  }
  sum = blockReduceSum(sum);
  if (threadIdx.x == 0) {
    result[blockIdx.x] = sum;
  }
}

__global__ void ScalarMulDoubleBlockReduce(int numElements, float* vector1, float* vector2, float* partial) {
  float sum = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements; i += blockDim.x * gridDim.x) {
    sum += vector1[i] * vector2[i];
  }
  sum = blockReduceSum(sum);
  if (threadIdx.x == 0) {
    partial[blockIdx.x] = sum;
  }
}

__global__ void ScalarMulSecondBlockReduce(int numBlocks, float* partial, float* result) {
  float sum = 0.f;
  for (int i = threadIdx.x; i < numBlocks; i += blockDim.x) {
    sum += partial[i];
  }
  sum = blockReduceSum(sum);
  if (threadIdx.x == 0) {
    *result = sum;
  }
}
