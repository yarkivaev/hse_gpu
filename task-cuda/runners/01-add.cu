#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "KernelAdd.cuh"

static void fail(const char* msg) {
  fprintf(stderr, "%s\n", msg);
  exit(1);
}

static void check(cudaError_t err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "cuda: %s\n", cudaGetErrorString(err));
    exit(1);
  }
}

static float runOnce(int n, int blockSize) {
  size_t bytes = (size_t)n * sizeof(float);
  std::vector<float> hx(n), hy(n);
  for (int i = 0; i < n; ++i) {
    hx[i] = (float)(i % 97) * 0.01f;
    hy[i] = (float)(i % 53) * 0.02f;
  }
  float *dx = nullptr, *dy = nullptr, *dr = nullptr;
  check(cudaMalloc(&dx, bytes));
  check(cudaMalloc(&dy, bytes));
  check(cudaMalloc(&dr, bytes));
  check(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
  check(cudaMemcpy(dy, hy.data(), bytes, cudaMemcpyHostToDevice));
  int grid = (n + blockSize - 1) / blockSize;
  KernelAdd<<<grid, blockSize>>>(n, dx, dy, dr);
  check(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  check(cudaEventCreate(&start));
  check(cudaEventCreate(&stop));
  check(cudaEventRecord(start));
  KernelAdd<<<grid, blockSize>>>(n, dx, dy, dr);
  check(cudaEventRecord(stop));
  check(cudaEventSynchronize(stop));
  float ms = 0.f;
  check(cudaEventElapsedTime(&ms, start, stop));
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(dx);
  cudaFree(dy);
  cudaFree(dr);
  return ms;
}

static void doCheck() {
  const int n = 10000;
  const int blockSize = 256;
  size_t bytes = (size_t)n * sizeof(float);
  std::vector<float> hx(n), hy(n), hr(n), ref(n);
  for (int i = 0; i < n; ++i) {
    hx[i] = (float)i * 0.1f;
    hy[i] = (float)(n - i) * 0.1f;
    ref[i] = hx[i] + hy[i];
  }
  float *dx = nullptr, *dy = nullptr, *dr = nullptr;
  check(cudaMalloc(&dx, bytes));
  check(cudaMalloc(&dy, bytes));
  check(cudaMalloc(&dr, bytes));
  check(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
  check(cudaMemcpy(dy, hy.data(), bytes, cudaMemcpyHostToDevice));
  int grid = (n + blockSize - 1) / blockSize;
  KernelAdd<<<grid, blockSize>>>(n, dx, dy, dr);
  check(cudaMemcpy(hr.data(), dr, bytes, cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    if (hr[i] != ref[i]) {
      fail("add mismatch");
    }
  }
  cudaFree(dx);
  cudaFree(dy);
  cudaFree(dr);
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
  const int blockSize = 256;
  fprintf(f, "mode,n,block,ms\n");
  for (int p = 10; p <= 24; ++p) {
    int n = 1 << p;
    float ms = runOnce(n, blockSize);
    fprintf(f, "size,%d,%d,%.4f\n", n, blockSize, ms);
  }
  for (int bs : {32, 64, 128, 256, 512, 1024}) {
    int n = 1 << 20;
    float ms = runOnce(n, bs);
    fprintf(f, "block,%d,%d,%.4f\n", n, bs, ms);
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
