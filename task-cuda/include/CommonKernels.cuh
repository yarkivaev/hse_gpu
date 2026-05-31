#pragma once

__device__ __forceinline__ float warpReduceSum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
  __shared__ float shared[32];
  __shared__ float total;
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  val = warpReduceSum(val);
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : 0.f;
  if (wid == 0) {
    val = warpReduceSum(val);
  }
  if (threadIdx.x == 0) {
    total = val;
  }
  __syncthreads();
  return total;
}

__global__ void ReduceSumKernel(int numElements, float* data, float* result);

__global__ void ScalarMulDoubleBlockReduce(int numElements, float* vector1, float* vector2, float* partial);

__global__ void ScalarMulSecondBlockReduce(int numBlocks, float* partial, float* result);
