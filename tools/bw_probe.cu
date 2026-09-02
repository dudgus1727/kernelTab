// **실효 메모리 대역폭**을 실측한다. ECC 오버헤드가 포함된 값이다.
//
// 왜 필요한가
// -----------
// 이 하네스는 대역폭을 `버스 폭 x 2(DDR) x 관측 클럭` **하나의 경로**로만
// 계산한다 (계산 경로가 둘이면 다른 GPU 에서 조용히 어긋난다). 그 값은
// 물리 상한이고 **ECC 오버헤드를 모른다.**
//
// 데이터센터 GPU 는 ECC 가 기본으로 켜져 있고 끄지 않는다 (실무 조건이다).
// 그래서 도달 가능한 대역폭은 계산값보다 낮다. 그 차이를 알고 있어야
// `frac_of_peak` 나 roofline 을 읽을 때 오해하지 않는다.
//
// ⚠️ 이 값을 `known.json` 이나 `env.json` 의 계산 경로에 **넣지 마라.**
//    기록용이다 — 계산 경로는 하나로 유지한다.
//
//   nvcc -arch=sm_90a -O3 tools/bw_probe.cu -o bw_probe && ./bw_probe
#include <cstdio>
#include <cuda_runtime.h>

__global__ void k_read(const float4 *__restrict__ src, size_t n, float *out) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  float4 acc = make_float4(0, 0, 0, 0);
  for (; i < n; i += stride) {
    float4 v = src[i];
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  if (acc.x == 1e30f) *out = acc.x + acc.y + acc.z + acc.w;  // 제거 방지
}

__global__ void k_copy(const float4 *__restrict__ src, float4 *__restrict__ dst,
                       size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) dst[i] = src[i];
}

__global__ void k_write(float4 *__restrict__ dst, size_t n, float v) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  float4 x = make_float4(v, v, v, v);
  for (; i < n; i += stride) dst[i] = x;
}

int main() {
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  // L2(H100 60 MB)보다 훨씬 커야 한다. 안 그러면 캐시 대역폭을 잰다.
  const size_t BYTES = 4ull << 30;
  const size_t n4 = BYTES / sizeof(float4);
  float4 *a, *b; float *sink;
  if (cudaMalloc(&a, BYTES) != cudaSuccess ||
      cudaMalloc(&b, BYTES) != cudaSuccess ||
      cudaMalloc(&sink, 4) != cudaSuccess) {
    printf("cudaMalloc 실패 (%zu MB x2 필요)\n", BYTES >> 20); return 1;
  }
  cudaMemset(a, 1, BYTES);
  int blocks = p.multiProcessorCount * 32, threads = 256;
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

  printf("%s  L2 %d MB  버퍼 %zu MB  (ECC 상태는 nvidia-smi 로 확인)\n",
         p.name, p.l2CacheSize >> 20, BYTES >> 20);
  printf("  %-10s %10s %12s  %s\n", "패턴", "ms", "GB/s", "움직인 바이트");

  struct { const char *tag; double factor; } cases[] = {
      {"read", 1.0}, {"write", 1.0}, {"copy(r+w)", 2.0}};
  for (int c = 0; c < 3; ++c) {
    double best = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
      cudaEventRecord(e0);
      if (c == 0) k_read<<<blocks, threads>>>(a, n4, sink);
      else if (c == 1) k_write<<<blocks, threads>>>(b, n4, 1.f);
      else k_copy<<<blocks, threads>>>(a, b, n4);
      cudaEventRecord(e1); cudaEventSynchronize(e1);
      float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
      if (ms < best) best = ms;
    }
    double moved = BYTES * cases[c].factor;
    printf("  %-10s %10.3f %12.1f  %.1f GB\n", cases[c].tag, best,
           moved / (best * 1e-3) / 1e9, moved / 1e9);
  }
  cudaFree(a); cudaFree(b); cudaFree(sink);
  return 0;
}
