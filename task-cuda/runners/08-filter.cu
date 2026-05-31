#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <Filter.cuh>

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

static void launchFilter(
    int n,
    int blockSize,
    float* dArray,
    OperationFilterType type,
    float* dVal,
    float* dResult,
    float* dAux1,
    float* dAux2) {
  int grid = (n + blockSize - 1) / blockSize;
  if (grid < 1) {
    grid = 1;
  }
  size_t shared = (size_t)blockSize * sizeof(float);
  Filter<<<grid, blockSize, shared>>>(n, dArray, type, dVal, dResult, dAux1, dAux2);
}

static std::vector<float> cpuFilter(const std::vector<float>& in, OperationFilterType type, float threshold) {
  std::vector<float> out;
  for (float v : in) {
    bool keep = (type == GT) ? (v > threshold) : (v < threshold);
    if (keep) {
      out.push_back(v);
    }
  }
  return out;
}

static float runFilter(int n, int blockSize, OperationFilterType type, float threshold) {
  std::vector<float> h(n);
  for (int i = 0; i < n; ++i) {
    h[i] = (float)(i % 50) - 25.f;
  }
  float *dArray = nullptr, *dResult = nullptr, *dAux1 = nullptr, *dAux2 = nullptr, *dVal = nullptr;
  checkCuda(cudaMalloc(&dArray, (size_t)n * sizeof(float)));
  checkCuda(cudaMalloc(&dResult, (size_t)n * sizeof(float)));
  checkCuda(cudaMalloc(&dAux1, (size_t)n * sizeof(float)));
  checkCuda(cudaMalloc(&dAux2, (size_t)n * sizeof(float)));
  checkCuda(cudaMalloc(&dVal, sizeof(float)));
  checkCuda(cudaMemcpy(dArray, h.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(dVal, &threshold, sizeof(float), cudaMemcpyHostToDevice));
  launchFilter(n, blockSize, dArray, type, dVal, dResult, dAux1, dAux2);
  checkCuda(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  checkCuda(cudaEventCreate(&start));
  checkCuda(cudaEventCreate(&stop));
  checkCuda(cudaEventRecord(start));
  launchFilter(n, blockSize, dArray, type, dVal, dResult, dAux1, dAux2);
  checkCuda(cudaEventRecord(stop));
  checkCuda(cudaEventSynchronize(stop));
  float ms = 0.f;
  checkCuda(cudaEventElapsedTime(&ms, start, stop));
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(dArray);
  cudaFree(dResult);
  cudaFree(dAux1);
  cudaFree(dAux2);
  cudaFree(dVal);
  return ms;
}

static void doCheckType(OperationFilterType type, float threshold, int blockSize) {
  const int n = 1000;
  std::vector<float> h(n);
  for (int i = 0; i < n; ++i) {
    h[i] = (float)(i % 50) - 25.f;
  }
  std::vector<float> ref = cpuFilter(h, type, threshold);
  float *dArray = nullptr, *dResult = nullptr, *dAux1 = nullptr, *dAux2 = nullptr, *dVal = nullptr;
  checkCuda(cudaMalloc(&dArray, (size_t)n * sizeof(float)));
  checkCuda(cudaMalloc(&dResult, (size_t)n * sizeof(float)));
  checkCuda(cudaMalloc(&dAux1, (size_t)n * sizeof(float)));
  checkCuda(cudaMalloc(&dAux2, (size_t)n * sizeof(float)));
  checkCuda(cudaMalloc(&dVal, sizeof(float)));
  checkCuda(cudaMemcpy(dArray, h.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(dVal, &threshold, sizeof(float), cudaMemcpyHostToDevice));
  launchFilter(n, blockSize, dArray, type, dVal, dResult, dAux1, dAux2);
  checkCuda(cudaDeviceSynchronize());
  float count = 0.f;
  checkCuda(cudaMemcpy(&count, dAux2, sizeof(float), cudaMemcpyDeviceToHost));
  std::vector<float> out((size_t)n);
  checkCuda(cudaMemcpy(out.data(), dResult, (size_t)count * sizeof(float), cudaMemcpyDeviceToHost));
  if ((int)count != (int)ref.size()) {
    fail("filter count mismatch");
  }
  for (size_t i = 0; i < ref.size(); ++i) {
    if (out[i] != ref[i]) {
      fail("filter value mismatch");
    }
  }
  cudaFree(dArray);
  cudaFree(dResult);
  cudaFree(dAux1);
  cudaFree(dAux2);
  cudaFree(dVal);
}

static void doCheck() {
  const int blockSize = 256;
  doCheckType(GT, 0.f, blockSize);
  doCheckType(LT, 0.f, blockSize);
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
  for (int p = 10; p <= 20; ++p) {
    int n = 1 << p;
    float ms = runFilter(n, blockSize, GT, 0.f);
    fprintf(f, "size,%d,%d,%.4f\n", n, blockSize, ms);
  }
  int n = 1 << 18;
  for (int bs : {32, 64, 128, 256, 512}) {
    float ms = runFilter(n, bs, GT, 0.f);
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
