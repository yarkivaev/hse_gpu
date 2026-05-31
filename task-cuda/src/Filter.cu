#include <Filter.cuh>
#include <cuda_runtime.h>

namespace {

__device__ unsigned int gSyncCount;
__device__ unsigned int gSyncSense;

__device__ void blockExclusiveScan(float* shared, int n) {
  int offset = 1;
  for (int d = n >> 1; d > 0; d >>= 1) {
    __syncthreads();
    if (threadIdx.x < d) {
      int ai = offset * (2 * threadIdx.x + 1) - 1;
      int bi = offset * (2 * threadIdx.x + 2) - 1;
      shared[bi] += shared[ai];
    }
    offset <<= 1;
  }
  if (threadIdx.x == 0) {
    shared[n - 1] = 0.f;
  }
  for (int d = 1; d < n; d <<= 1) {
    offset >>= 1;
    __syncthreads();
    if (threadIdx.x < d) {
      int ai = offset * (2 * threadIdx.x + 1) - 1;
      int bi = offset * (2 * threadIdx.x + 2) - 1;
      float t = shared[ai];
      shared[ai] = shared[bi];
      shared[bi] += t;
    }
  }
  __syncthreads();
}

__device__ void gridBarrier() {
  __threadfence();
  __syncthreads();
  __shared__ unsigned int startSense;
  if (threadIdx.x == 0) {
    startSense = gSyncSense;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned int ticket = atomicAdd(&gSyncCount, 1u);
    if (ticket + 1u == gridDim.x) {
      gSyncCount = 0u;
      atomicAdd(&gSyncSense, 1u);
    }
  }
  __syncthreads();
  unsigned int localStart = startSense;
  while (atomicAdd(&gSyncSense, 0u) == localStart) {
  }
  __threadfence();
  __syncthreads();
}

__device__ void filterParallel(
    int numElements,
    float* array,
    OperationFilterType type,
    float threshold,
    float* result,
    float* auxArray1,
    float* auxArray2) {
  int chunk = blockDim.x;
  int begin = blockIdx.x * chunk;
  int end = begin + chunk;
  if (end > numElements) {
    end = numElements;
  }
  extern __shared__ float sharedFlags[];
  for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
    bool keep = (type == GT) ? (array[i] > threshold) : (array[i] < threshold);
    auxArray1[i] = keep ? 1.f : 0.f;
  }
  __syncthreads();
  if (threadIdx.x < chunk) {
    sharedFlags[threadIdx.x] = (begin + threadIdx.x < end) ? auxArray1[begin + threadIdx.x] : 0.f;
  }
  blockExclusiveScan(sharedFlags, chunk);
  if (begin + threadIdx.x < end) {
    auxArray2[begin + threadIdx.x] = sharedFlags[threadIdx.x];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (begin < numElements) {
      int last = end - 1;
      result[blockIdx.x] = auxArray2[last] + auxArray1[last];
    } else {
      result[blockIdx.x] = 0.f;
    }
  }
  gridBarrier();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int blocks = gridDim.x;
    float run = 0.f;
    for (int b = 0; b < blocks; ++b) {
      float next = run + result[b];
      result[b] = run;
      run = next;
    }
  }
  gridBarrier();
  if (begin < numElements) {
    float blockOffset = result[blockIdx.x];
    for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
      auxArray2[i] += blockOffset;
    }
    __syncthreads();
    for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
      if (auxArray1[i] > 0.f) {
        result[(int)auxArray2[i]] = array[i];
      }
    }
  }
  gridBarrier();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    float count = auxArray2[numElements - 1] + auxArray1[numElements - 1];
    auxArray2[0] = count;
  }
}

__global__ void FilterMapKernel(
    int numElements,
    float* array,
    OperationFilterType type,
    float threshold,
    float* flags) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements; i += blockDim.x * gridDim.x) {
    bool keep = (type == GT) ? (array[i] > threshold) : (array[i] < threshold);
    flags[i] = keep ? 1.f : 0.f;
  }
}

__global__ void FilterBlockScanKernel(
    int numElements,
    float* flags,
    float* scanOut,
    float* blockSums) {
  int chunk = blockDim.x;
  int begin = blockIdx.x * chunk;
  if (begin >= numElements) {
    return;
  }
  int end = begin + chunk;
  if (end > numElements) {
    end = numElements;
  }
  extern __shared__ float sharedFlags[];
  if (threadIdx.x < chunk) {
    sharedFlags[threadIdx.x] = (begin + threadIdx.x < end) ? flags[begin + threadIdx.x] : 0.f;
  }
  blockExclusiveScan(sharedFlags, chunk);
  if (begin + threadIdx.x < end) {
    scanOut[begin + threadIdx.x] = sharedFlags[threadIdx.x];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int last = end - 1;
    blockSums[blockIdx.x] = scanOut[last] + flags[last];
  }
}

__global__ void FilterPrefixBlockSums(int numBlocks, float* blockSums) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  float run = 0.f;
  for (int b = 0; b < numBlocks; ++b) {
    float next = run + blockSums[b];
    blockSums[b] = run;
    run = next;
  }
}

__global__ void FilterAddOffsetsKernel(
    int numElements,
    float* scanOut,
    float* blockOffsets) {
  int chunk = blockDim.x;
  int begin = blockIdx.x * chunk;
  if (begin >= numElements) {
    return;
  }
  int end = begin + chunk;
  if (end > numElements) {
    end = numElements;
  }
  float offset = blockOffsets[blockIdx.x];
  for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
    scanOut[i] += offset;
  }
}

__global__ void FilterScatterKernel(
    int numElements,
    float* array,
    float* flags,
    float* scanOut,
    float* result) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements; i += blockDim.x * gridDim.x) {
    if (flags[i] > 0.f) {
      result[(int)scanOut[i]] = array[i];
    }
  }
}

__global__ void FilterWriteCountKernel(int numElements, float* flags, float* scanOut, float* auxArray2) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  float count = (numElements == 0) ? 0.f : scanOut[numElements - 1] + flags[numElements - 1];
  auxArray2[0] = count;
}

}

static void filterLaunchHost(
    int numElements,
    float* array,
    OperationFilterType type,
    float threshold,
    float* result,
    float* auxArray1,
    float* auxArray2,
    int blockSize) {
  if (numElements <= 0) {
    float zero = 0.f;
    cudaMemcpy(auxArray2, &zero, sizeof(float), cudaMemcpyHostToDevice);
    return;
  }
  int grid = (numElements + blockSize - 1) / blockSize;
  size_t sharedBytes = (size_t)blockSize * sizeof(float);
  FilterMapKernel<<<grid, blockSize>>>(numElements, array, type, threshold, auxArray1);
  FilterBlockScanKernel<<<grid, blockSize, sharedBytes>>>(numElements, auxArray1, auxArray2, result);
  FilterPrefixBlockSums<<<1, 1>>>(grid, result);
  FilterAddOffsetsKernel<<<grid, blockSize>>>(numElements, auxArray2, result);
  FilterScatterKernel<<<grid, blockSize>>>(numElements, array, auxArray1, auxArray2, result);
  FilterWriteCountKernel<<<1, 1>>>(numElements, auxArray1, auxArray2, auxArray2);
}

void FilterRun(
    int numElements,
    float* array,
    OperationFilterType type,
    float* value,
    float* result,
    float* auxArray1,
    float* auxArray2,
    int blockSize) {
  float threshold = 0.f;
  cudaMemcpy(&threshold, value, sizeof(float), cudaMemcpyDeviceToHost);
  filterLaunchHost(numElements, array, type, threshold, result, auxArray1, auxArray2, blockSize);
}

__global__ void Filter(
    int numElements,
    float* array,
    OperationFilterType type,
    float* value,
    float* result,
    float* auxArray1,
    float* auxArray2) {
  if (numElements <= 0) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      auxArray2[0] = 0.f;
    }
    return;
  }
  filterParallel(numElements, array, type, *value, result, auxArray1, auxArray2);
}
