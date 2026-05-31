#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <MatrixMul.cuh>

static void fail(const char* msg) {
  fprintf(stderr, "%s\n", msg);
  exit(1);
}

static void checkCuda(cudaError_t err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "cuda: %s\n", cudaGetErrorString(err));
    exit(1);
  }
}

static float runOnce(int n) {
  int widthA = n;
  int heightA = n;
  int widthB = n;
  size_t aBytes = (size_t)heightA * (size_t)widthA * sizeof(float);
  size_t bBytes = (size_t)widthA * (size_t)widthB * sizeof(float);
  size_t cBytes = (size_t)heightA * (size_t)widthB * sizeof(float);
  std::vector<float> hA((size_t)n * n), hB((size_t)n * n);
  for (int i = 0; i < n * n; ++i) {
    hA[i] = (float)(i % 7);
    hB[i] = (float)(i % 5);
  }
  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  checkCuda(cudaMalloc(&dA, aBytes));
  checkCuda(cudaMalloc(&dB, bBytes));
  checkCuda(cudaMalloc(&dC, cBytes));
  checkCuda(cudaMemcpy(dA, hA.data(), aBytes, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(dB, hB.data(), bBytes, cudaMemcpyHostToDevice));
  dim3 block(16, 16);
  dim3 grid((widthB + 15) / 16, (heightA + 15) / 16);
  MatrixMul<<<grid, block>>>(heightA, widthA, widthB, dA, dB, dC);
  checkCuda(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  checkCuda(cudaEventCreate(&start));
  checkCuda(cudaEventCreate(&stop));
  checkCuda(cudaEventRecord(start));
  MatrixMul<<<grid, block>>>(heightA, widthA, widthB, dA, dB, dC);
  checkCuda(cudaEventRecord(stop));
  checkCuda(cudaEventSynchronize(stop));
  float ms = 0.f;
  checkCuda(cudaEventElapsedTime(&ms, start, stop));
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return ms;
}

static void doCheck() {
  const int n = 64;
  std::vector<float> hA((size_t)n * n), hB((size_t)n * n), hC((size_t)n * n), ref((size_t)n * n);
  for (int i = 0; i < n * n; ++i) {
    hA[i] = (float)(i % 11);
    hB[i] = (float)(i % 9);
  }
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      ref[r * n + c] = 0.f;
      for (int k = 0; k < n; ++k) {
        ref[r * n + c] += hA[r * n + k] * hB[k * n + c];
      }
    }
  }
  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  checkCuda(cudaMalloc(&dA, hA.size() * sizeof(float)));
  checkCuda(cudaMalloc(&dB, hB.size() * sizeof(float)));
  checkCuda(cudaMalloc(&dC, hC.size() * sizeof(float)));
  checkCuda(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));
  dim3 block(16, 16);
  dim3 grid((n + 15) / 16, (n + 15) / 16);
  MatrixMul<<<grid, block>>>(n, n, n, dA, dB, dC);
  checkCuda(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < ref.size(); ++i) {
    if (hC[i] != ref[i]) {
      fail("matmul mismatch");
    }
  }
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  printf("PASS\n");
}

static void doBenchmark(const char* path) {
  FILE* f = stdout;
  if (path != nullptr) {
    f = fopen(path, "w");
    if (f == nullptr) {
      fail("cannot open output");
    }
  }
  fprintf(f, "mode,n,block,ms\n");
  for (int n : {128, 256, 512, 1024, 2048}) {
    float ms = runOnce(n);
    fprintf(f, "size,%d,16,%.4f\n", n, ms);
  }
  int n = 512;
  for (int t : {8, 16, 32}) {
    float ms = runOnce(n);
    fprintf(f, "block,%d,%d,%.4f\n", n, t, ms);
  }
  if (path != nullptr) {
    fclose(f);
  }
}

int main(int argc, char** argv) {
  bool checkFlag = false;
  bool benchFlag = false;
  const char* out = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--check") == 0) {
      checkFlag = true;
    } else if (strcmp(argv[i], "--benchmark") == 0) {
      benchFlag = true;
    } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
      out = argv[++i];
    }
  }
  if (!checkFlag && !benchFlag) {
    checkFlag = true;
  }
  if (checkFlag) {
    doCheck();
  }
  if (benchFlag) {
    doBenchmark(out);
  }
  return 0;
}
