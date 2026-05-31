#include <CosineVector.cuh>
#include <ScalarMulRunner.cuh>
#include <cmath>

float CosineVector(int numElements, float* vector1, float* vector2, int blockSize) {
  float dot = ScalarMulSumPlusReduction(numElements, vector1, vector2, blockSize);
  float norm1 = ScalarMulSumPlusReduction(numElements, vector1, vector1, blockSize);
  float norm2 = ScalarMulSumPlusReduction(numElements, vector2, vector2, blockSize);
  return dot / (sqrtf(norm1) * sqrtf(norm2));
}
