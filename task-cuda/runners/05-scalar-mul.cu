#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <ScalarMulRunner.cuh>

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

static float cpuDot(const std::vector<float>& a, const std::vector<float>& b) {
  float s = 0.f;
  for (size_t i = 0; i < a.size(); ++i) {
    s += a[i] * b[i];
  }
  return s;
}

static float benchMethod(float (*fn)(int, float*, float*, int), int n, int blockSize) {
  std::vector<float> h1(n), h2(n);
  for (int i = 0; i < n; ++i) {
    h1[i] = (float)(i % 97) * 0.01f;
    h2[i] = (float)(i % 53) * 0.02f;
  }
  float *d1 = nullptr, *d2 = nullptr;
  checkCuda(cudaMalloc(&d1, (size_t)n * sizeof(float)));
  checkCuda(cudaMalloc(&d2, (size_t)n * sizeof(float)));
  checkCuda(cudaMemcpy(d1, h1.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d2, h2.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
  fn(n, d1, d2, blockSize);
  cudaEvent_t start, stop;
  checkCuda(cudaEventCreate(&start));
  checkCuda(cudaEventCreate(&stop));
  checkCuda(cudaEventRecord(start));
  fn(n, d1, d2, blockSize);
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

static bool closeEnough(float got, float expected) {
  float scale = fmaxf(fmaxf(fabsf(got), fabsf(expected)), 1.f);
  return fabsf(got - expected) <= 1e-4f * scale + 0.5f;
}

static void doCheck() {
  const int n = 100000;
  const int blockSize = 256;
  std::vector<float> h1(n), h2(n);
  for (int i = 0; i < n; ++i) {
    h1[i] = (float)i * 0.001f;
    h2[i] = (float)(n - i) * 0.001f;
  }
  float ref = cpuDot(h1, h2);
  float *d1 = nullptr, *d2 = nullptr;
  checkCuda(cudaMalloc(&d1, (size_t)n * sizeof(float)));
  checkCuda(cudaMalloc(&d2, (size_t)n * sizeof(float)));
  checkCuda(cudaMemcpy(d1, h1.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d2, h2.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
  float a = ScalarMulSumPlusReduction(n, d1, d2, blockSize);
  float b = ScalarMulTwoReductions(n, d1, d2, blockSize);
  if (!closeEnough(a, ref) || !closeEnough(b, ref)) {
    fail("scalar mul mismatch");
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
  fprintf(f, "mode,n,block,ms,method\n");
  for (int p = 10; p <= 20; ++p) {
    int n = 1 << p;
    float ms1 = benchMethod(ScalarMulSumPlusReduction, n, blockSize);
    float ms2 = benchMethod(ScalarMulTwoReductions, n, blockSize);
    fprintf(f, "size,%d,%d,%.4f,sumplus\n", n, blockSize, ms1);
    fprintf(f, "size,%d,%d,%.4f,two\n", n, blockSize, ms2);
  }
  int n = 1 << 18;
  for (int bs : {32, 64, 128, 256, 512}) {
    float ms1 = benchMethod(ScalarMulSumPlusReduction, n, bs);
    float ms2 = benchMethod(ScalarMulTwoReductions, n, bs);
    fprintf(f, "block,%d,%d,%.4f,sumplus\n", n, bs, ms1);
    fprintf(f, "block,%d,%d,%.4f,two\n", n, bs, ms2);
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
