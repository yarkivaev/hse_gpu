#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <MatrixVectorMul.cuh>

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

static float runOnce(int height, int width, int blockSize) {
  size_t mBytes = (size_t)height * (size_t)width * sizeof(float);
  size_t vBytes = (size_t)width * sizeof(float);
  size_t rBytes = (size_t)height * sizeof(float);
  std::vector<float> hm((size_t)height * width), hv(width);
  for (size_t i = 0; i < hm.size(); ++i) {
    hm[i] = (float)(i % 31) * 0.1f;
  }
  for (int i = 0; i < width; ++i) {
    hv[i] = (float)(i % 17) * 0.2f;
  }
  float *dm = nullptr, *dv = nullptr, *dr = nullptr;
  checkCuda(cudaMalloc(&dm, mBytes));
  checkCuda(cudaMalloc(&dv, vBytes));
  checkCuda(cudaMalloc(&dr, rBytes));
  checkCuda(cudaMemcpy(dm, hm.data(), mBytes, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(dv, hv.data(), vBytes, cudaMemcpyHostToDevice));
  int grid = (height + blockSize - 1) / blockSize;
  MatrixVectorMul<<<grid, blockSize>>>(height, width, dm, dv, dr);
  checkCuda(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  checkCuda(cudaEventCreate(&start));
  checkCuda(cudaEventCreate(&stop));
  checkCuda(cudaEventRecord(start));
  MatrixVectorMul<<<grid, blockSize>>>(height, width, dm, dv, dr);
  checkCuda(cudaEventRecord(stop));
  checkCuda(cudaEventSynchronize(stop));
  float ms = 0.f;
  checkCuda(cudaEventElapsedTime(&ms, start, stop));
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(dm);
  cudaFree(dv);
  cudaFree(dr);
  return ms;
}

static void doCheck() {
  const int height = 256;
  const int width = 128;
  const int blockSize = 256;
  std::vector<float> hm((size_t)height * width), hv(width), hr(height), ref(height);
  for (int c = 0; c < width; ++c) {
    hv[c] = (float)c * 0.5f;
  }
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      hm[r * width + c] = (float)(r + c);
    }
  }
  for (int r = 0; r < height; ++r) {
    ref[r] = 0.f;
    for (int c = 0; c < width; ++c) {
      ref[r] += hm[r * width + c] * hv[c];
    }
  }
  float *dm = nullptr, *dv = nullptr, *dr = nullptr;
  checkCuda(cudaMalloc(&dm, hm.size() * sizeof(float)));
  checkCuda(cudaMalloc(&dv, hv.size() * sizeof(float)));
  checkCuda(cudaMalloc(&dr, hr.size() * sizeof(float)));
  checkCuda(cudaMemcpy(dm, hm.data(), hm.size() * sizeof(float), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(dv, hv.data(), hv.size() * sizeof(float), cudaMemcpyHostToDevice));
  int grid = (height + blockSize - 1) / blockSize;
  MatrixVectorMul<<<grid, blockSize>>>(height, width, dm, dv, dr);
  checkCuda(cudaMemcpy(hr.data(), dr, hr.size() * sizeof(float), cudaMemcpyDeviceToHost));
  for (int r = 0; r < height; ++r) {
    if (hr[r] != ref[r]) {
      fail("gemv mismatch");
    }
  }
  cudaFree(dm);
  cudaFree(dv);
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
  for (int n : {256, 512, 1024, 2048, 4096}) {
    float ms = runOnce(n, n, blockSize);
    fprintf(f, "size,%d,%d,%.4f\n", n, blockSize, ms);
  }
  int n = 2048;
  for (int bs : {32, 64, 128, 256, 512}) {
    float ms = runOnce(n, n, bs);
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
