#include <ScalarMulRunner.cuh>
#include <ScalarMul.cuh>
#include <CommonKernels.cuh>
#include <cuda_runtime.h>
#include <vector>

static float reducePartial(float* dPartial, int count, int blockSize, float* dOnes, float* dTmp) {
  while (count > 1) {
    int next = (count + blockSize - 1) / blockSize;
    if (next == 1) {
      ScalarMulBlock<<<1, blockSize>>>(count, dPartial, dOnes, dPartial);
      cudaDeviceSynchronize();
      break;
    }
    ScalarMulBlock<<<next, blockSize>>>(count, dPartial, dOnes, dTmp);
    cudaDeviceSynchronize();
    float* swap = dPartial;
    dPartial = dTmp;
    dTmp = swap;
    count = next;
  }
  float out = 0.f;
  cudaMemcpy(&out, dPartial, sizeof(float), cudaMemcpyDeviceToHost);
  return out;
}

float ScalarMulSumPlusReduction(int numElements, float* vector1, float* vector2, int blockSize) {
  int grid = (numElements + blockSize - 1) / blockSize;
  float* dPartial = nullptr;
  cudaMalloc(&dPartial, (size_t)grid * sizeof(float));
  float* dTmp = nullptr;
  cudaMalloc(&dTmp, (size_t)grid * sizeof(float));
  float* dOnes = nullptr;
  cudaMalloc(&dOnes, (size_t)grid * sizeof(float));
  std::vector<float> ones((size_t)grid, 1.f);
  cudaMemcpy(dOnes, ones.data(), (size_t)grid * sizeof(float), cudaMemcpyHostToDevice);
  ScalarMulBlock<<<grid, blockSize>>>(numElements, vector1, vector2, dPartial);
  cudaDeviceSynchronize();
  float result = reducePartial(dPartial, grid, blockSize, dOnes, dTmp);
  cudaFree(dPartial);
  cudaFree(dTmp);
  cudaFree(dOnes);
  return result;
}

float ScalarMulTwoReductions(int numElements, float* vector1, float* vector2, int blockSize) {
  int grid = (numElements + blockSize - 1) / blockSize;
  float* dPartial = nullptr;
  cudaMalloc(&dPartial, (size_t)grid * sizeof(float));
  float* dResult = nullptr;
  cudaMalloc(&dResult, sizeof(float));
  ScalarMulDoubleBlockReduce<<<grid, blockSize>>>(numElements, vector1, vector2, dPartial);
  cudaDeviceSynchronize();
  ScalarMulSecondBlockReduce<<<1, blockSize>>>(grid, dPartial, dResult);
  cudaDeviceSynchronize();
  float result = 0.f;
  cudaMemcpy(&result, dResult, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(dPartial);
  cudaFree(dResult);
  return result;
}
