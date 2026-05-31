#include <MatrixMul.cuh>

#define TILE 16

__global__
void MatrixMul(int heightA, int widthA, int widthB, float* matrixA, float* matrixB, float* matrixResult) {
  __shared__ float tileA[TILE][TILE];
  __shared__ float tileB[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float sum = 0.f;
  int tiles = (widthA + TILE - 1) / TILE;
  for (int t = 0; t < tiles; ++t) {
    int aCol = t * TILE + threadIdx.x;
    int aRow = row;
    tileA[threadIdx.y][threadIdx.x] = (aRow < heightA && aCol < widthA)
        ? matrixA[aRow * widthA + aCol] : 0.f;
    int bRow = t * TILE + threadIdx.y;
    int bCol = col;
    tileB[threadIdx.y][threadIdx.x] = (bRow < widthA && bCol < widthB)
        ? matrixB[bRow * widthB + bCol] : 0.f;
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
      sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < heightA && col < widthB) {
    matrixResult[row * widthB + col] = sum;
  }
}
