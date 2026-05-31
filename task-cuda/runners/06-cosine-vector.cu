#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <CosineVector.cuh>

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

static float cpuCosine(const std::vector<float>& a, const std::vector<float>& b) {
  float dot = 0.f, n1 = 0.f, n2 = 0.f;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    n1 += a[i] * a[i];
    n2 += b[i] * b[i];
  }
  return dot / (sqrtf(n1) * sqrtf(n2));
}

static float runBench(int n, int blockSize) {
  std::vector<float> h1(n), h2(n);
  for (int i = 0; i < n; ++i) {
    h1[i] = (float)(i % 97);
    h2[i] = (float)(i % 53);
  }
  float *d1 = nullptr, *d2 = nullptr;
  checkCuda(cudaMalloc(&d1, (size_t)n * sizeof(float)));
  checkCuda(cudaMalloc(&d2, (size_t)n * sizeof(float)));
  checkCuda(cudaMemcpy(d1, h1.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d2, h2.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
  CosineVector(n, d1, d2, blockSize);
  cudaEvent_t start, stop;
  checkCuda(cudaEventCreate(&start));
  checkCuda(cudaEventCreate(&stop));
  checkCuda(cudaEventRecord(start));
  CosineVector(n, d1, d2, blockSize);
  checkCuda(cudaEventRecord(stop));
  checkCuda(cudaEventSynchronize(stop));
  float ms = 0.f;
  checkCuda(cudaEventElapsedTime(&ms, start, stop));
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d1);
  cudaFree(d2);
  return ms;
}

static void doCheck() {
  const int n = 50000;
  const int blockSize = 256;
  std::vector<float> h1(n), h2(n);
  for (int i = 0; i < n; ++i) {
    h1[i] = (float)i;
    h2[i] = (float)(i * 2);
  }
  float ref = cpuCosine(h1, h2);
  float *d1 = nullptr, *d2 = nullptr;
  checkCuda(cudaMalloc(&d1, (size_t)n * sizeof(float)));
  checkCuda(cudaMalloc(&d2, (size_t)n * sizeof(float)));
  checkCuda(cudaMemcpy(d1, h1.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d2, h2.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
  float got = CosineVector(n, d1, d2, blockSize);
  if (fabsf(got - ref) > 1e-4f) {
    fail("cosine mismatch");
  }
  for (int i = 0; i < n; ++i) {
    h2[i] = (float)(-i);
  }
  checkCuda(cudaMemcpy(d2, h2.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
  got = CosineVector(n, d1, d2, blockSize);
  if (fabsf(got + 1.f) > 1e-4f) {
    fail("cosine collinear mismatch");
  }
  cudaFree(d1);
  cudaFree(d2);
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
    float ms = runBench(n, blockSize);
    fprintf(f, "size,%d,%d,%.4f\n", n, blockSize, ms);
  }
  int n = 1 << 18;
  for (int bs : {32, 64, 128, 256, 512}) {
    float ms = runBench(n, bs);
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
