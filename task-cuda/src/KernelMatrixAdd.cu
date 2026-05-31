#include <KernelMatrixAdd.cuh>

__global__ void KernelMatrixAdd(int height, int width, int pitch, float* A, float* B, float* result) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= height || col >= width) {
    return;
  }
  float* rowA = (float*)((char*)A + row * pitch);
  float* rowB = (float*)((char*)B + row * pitch);
  float* rowR = (float*)((char*)result + row * pitch);
  rowR[col] = rowA[col] + rowB[col];
}
