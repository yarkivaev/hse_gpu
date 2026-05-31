#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <KernelMatrixAdd.cuh>

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

static float runOnce(int n, dim3 block) {
  size_t pitchBytes = 0;
  float *dA = nullptr, *dB = nullptr, *dR = nullptr;
  std::vector<float> hA((size_t)n * n), hB((size_t)n * n);
  checkCuda(cudaMallocPitch((void**)&dA, &pitchBytes, (size_t)n * sizeof(float), (size_t)n));
  checkCuda(cudaMallocPitch((void**)&dB, &pitchBytes, (size_t)n * sizeof(float), (size_t)n));
  checkCuda(cudaMallocPitch((void**)&dR, &pitchBytes, (size_t)n * sizeof(float), (size_t)n));
  for (int i = 0; i < n * n; ++i) {
    hA[i] = (float)(i % 17);
    hB[i] = (float)(i % 13);
  }
  checkCuda(cudaMemcpy2D(dA, pitchBytes, hA.data(), (size_t)n * sizeof(float),
      (size_t)n * sizeof(float), (size_t)n, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy2D(dB, pitchBytes, hB.data(), (size_t)n * sizeof(float),
      (size_t)n * sizeof(float), (size_t)n, cudaMemcpyHostToDevice));
  dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
  KernelMatrixAdd<<<grid, block>>>(n, n, (int)pitchBytes, dA, dB, dR);
  checkCuda(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  checkCuda(cudaEventCreate(&start));
  checkCuda(cudaEventCreate(&stop));
  checkCuda(cudaEventRecord(start));
  KernelMatrixAdd<<<grid, block>>>(n, n, (int)pitchBytes, dA, dB, dR);
  checkCuda(cudaEventRecord(stop));
  checkCuda(cudaEventSynchronize(stop));
  float ms = 0.f;
  checkCuda(cudaEventElapsedTime(&ms, start, stop));
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dR);
  return ms;
}

static void doCheck() {
  const int n = 512;
  dim3 block(16, 16);
  size_t pitchBytes = 0;
  float *dA = nullptr, *dB = nullptr, *dR = nullptr;
  checkCuda(cudaMallocPitch((void**)&dA, &pitchBytes, (size_t)n * sizeof(float), (size_t)n));
  checkCuda(cudaMallocPitch((void**)&dB, &pitchBytes, (size_t)n * sizeof(float), (size_t)n));
  checkCuda(cudaMallocPitch((void**)&dR, &pitchBytes, (size_t)n * sizeof(float), (size_t)n));
  std::vector<float> hA((size_t)n * n), hB((size_t)n * n), hR((size_t)n * n);
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      hA[r * n + c] = (float)(r + c);
      hB[r * n + c] = (float)(r - c);
      hR[r * n + c] = hA[r * n + c] + hB[r * n + c];
    }
  }
  checkCuda(cudaMemcpy2D(dA, pitchBytes, hA.data(), (size_t)n * sizeof(float),
      (size_t)n * sizeof(float), (size_t)n, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy2D(dB, pitchBytes, hB.data(), (size_t)n * sizeof(float),
      (size_t)n * sizeof(float), (size_t)n, cudaMemcpyHostToDevice));
  dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
  KernelMatrixAdd<<<grid, block>>>(n, n, (int)pitchBytes, dA, dB, dR);
  std::vector<float> out((size_t)n * n);
  checkCuda(cudaMemcpy2D(out.data(), (size_t)n * sizeof(float), dR, pitchBytes,
      (size_t)n * sizeof(float), (size_t)n, cudaMemcpyDeviceToHost));
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      if (out[r * n + c] != hR[r * n + c]) {
        fail("matrix add mismatch");
      }
    }
  }
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dR);
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
  dim3 block(16, 16);
  fprintf(f, "mode,n,block,ms\n");
  for (int n : {256, 512, 1024, 2048, 4096}) {
    float ms = runOnce(n, block);
    fprintf(f, "size,%d,%d,%.4f\n", n, block.x, ms);
  }
  int n = 1024;
  for (int bx : {8, 16, 32}) {
    dim3 b(bx, bx);
    float ms = runOnce(n, b);
    fprintf(f, "block,%d,%d,%.4f\n", n, bx, ms);
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
