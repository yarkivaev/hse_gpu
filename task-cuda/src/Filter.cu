#include <Filter.cuh>

namespace {

__device__ unsigned int gSyncCount;
__device__ unsigned int gSyncSense;

__device__ void gridBarrier() {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned int ticket = atomicAdd(&gSyncCount, 1u);
    if (ticket + 1u == gridDim.x) {
      gSyncCount = 0u;
      atomicAdd(&gSyncSense, 1u);
    } else {
      unsigned int sense = gSyncSense;
      while (sense == gSyncSense) {
      }
    }
  }
  __syncthreads();
}

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
