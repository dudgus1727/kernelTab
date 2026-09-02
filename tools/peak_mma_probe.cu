// FP32 누산 반감을 **직접 실측**한다 — known.json 의 fma_f16_per_clk_per_sm 근거.
//
//   nvcc -std=c++17 -arch=sm_89 -O3 tools/peak_mma_probe.cu -o peak_mma_probe
//   ./peak_mma_probe            # stdout 에 JSON 한 줄
//
// ## 왜 필요한가
//
// GeForce Ada / Blackwell 은 FP16 입력 + **FP32 누산**에서 처리량이 절반이다
// (256 FMA/clk/SM). A6000 같은 프로 SKU 는 반감이 없다 (512). 이 하네스는
// ElementAccumulator=float 를 쓰므로 **반감된 쪽**이 맞는 값이고, 512 를 쓰면
// 실효 피크가 2 배가 되어 ridge point 와 is_memory_bound 가 전 형상에서 틀린다.
//
// 데이터시트로는 확정할 수 없다 — 헤드라인이 희소(2:4) 기준이거나 FP16 누산
// 기준인 경우가 많다. 5090 은 이 프로브로 확정했고(비율 1.983), 4090 도
// 같은 방법으로 확인한다.
//
// ## 방법
//
// 메모리 접근이 없는 mma.sync 루프. 누산 체인을 4 개 독립으로 두어 지연이
// 아니라 **처리량**을 재게 한다 (한 체인이면 latency-bound 라 피크가 안 나온다).
// 두 모드의 유일한 차이는 누산 타입이다.
//
//   f32 누산:  mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//   f16 누산:  mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
//
// mma 하나 = 16x8x16 = 2048 MAC = 4096 flop.
//
// ⚠️ 관측 클럭으로 역산한 FMA/clk/SM 이 두 모드에서 정수 256/512 에 가까운지
//    확인한다. 안 맞으면 클럭이 흔들렸거나 점유율이 모자란 것이다.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK(x)                                                              \
  do {                                                                        \
    cudaError_t _e = (x);                                                     \
    if (_e != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e),     \
              __FILE__, __LINE__);                                            \
      return 1;                                                               \
    }                                                                         \
  } while (0)


// --- 캠페인 조건 스탬프 -----------------------------------------------------
// ⛔ 이 프로브를 **캠페인 조건 밖**(호스트 nvcc / native libcuda)에서 돌리면
//    나온 값은 캠페인 값이 아니다. 그리고 그것은 **조용히 틀린다** — 프로브가
//    돌고 값이 나오므로 결과만 보고는 구분할 방법이 없다.
//    실제로 그렇게 잰 눈금 하나가 무효가 됐다 (2026-09-02, H100 세션).
//
//    ★ 축은 nvcc 버전이 아니라 **libcuda** 다. 이벤트 눈금·런치 오버헤드·
//      드라이버 경로가 전부 거기 달렸다. 호스트 native 와 이미지 compat 은
//      다른 드라이버다 (4090: 580.173.02 vs 610.43.02).
#define _KT_STR(x) #x
#define KT_STR(x) _KT_STR(x)
#define KT_NVCC_VERSION \
  KT_STR(__CUDACC_VER_MAJOR__) "." KT_STR(__CUDACC_VER_MINOR__) "." \
  KT_STR(__CUDACC_VER_BUILD__)
static int kt_driver_api_version(void) {
  int v = 0;
  cudaDriverGetVersion(&v);
  return v;
}

#define CHAINS 4

__global__ void mma_f32_accum(float *sink, int iters) {
  uint32_t a0 = 0x3C003C00u ^ threadIdx.x, a1 = 0x3C003C00u, a2 = a0, a3 = a1;
  uint32_t b0 = 0x3C003C00u ^ (threadIdx.x << 1), b1 = 0x3C003C00u;
  float c[CHAINS][4];
#pragma unroll
  for (int k = 0; k < CHAINS; ++k)
    c[k][0] = c[k][1] = c[k][2] = c[k][3] = 0.f;

  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int k = 0; k < CHAINS; ++k) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c[k][0]), "+f"(c[k][1]), "+f"(c[k][2]), "+f"(c[k][3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }
  if (sink) {   // 런타임에 nullptr 이므로 실행되지 않지만 최적화는 막는다
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < CHAINS; ++k) s += c[k][0] + c[k][1] + c[k][2] + c[k][3];
    sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
  }
}

__global__ void mma_f16_accum(float *sink, int iters) {
  uint32_t a0 = 0x3C003C00u ^ threadIdx.x, a1 = 0x3C003C00u, a2 = a0, a3 = a1;
  uint32_t b0 = 0x3C003C00u ^ (threadIdx.x << 1), b1 = 0x3C003C00u;
  uint32_t c[CHAINS][2];
#pragma unroll
  for (int k = 0; k < CHAINS; ++k) c[k][0] = c[k][1] = 0u;

  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int k = 0; k < CHAINS; ++k) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
          "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
          : "+r"(c[k][0]), "+r"(c[k][1])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }
  if (sink) {
    uint32_t s = 0u;
#pragma unroll
    for (int k = 0; k < CHAINS; ++k) s += c[k][0] + c[k][1];
    sink[blockIdx.x * blockDim.x + threadIdx.x] = (float)s;
  }
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

template <typename F>
static double best_tflops(F kern, int sm, int blocks_per_sm, int threads,
                          int iters, int reps) {
  const int blocks = sm * blocks_per_sm;
  const int warps = blocks * (threads / 32);
  // mma 하나 = 4096 flop, 워프당 CHAINS 개 x iters 회
  const double flops = (double)warps * CHAINS * iters * 4096.0;
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  kern<<<blocks, threads>>>(nullptr, 64);         // 워밍업
  cudaDeviceSynchronize();
  std::vector<double> tf;
  for (int r = 0; r < reps; ++r) {
    cudaEventRecord(a);
    kern<<<blocks, threads>>>(nullptr, iters);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    if (ms > 0) tf.push_back(flops / (ms * 1e-3) / 1e12);
  }
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  if (tf.empty()) return 0.0;
  return *std::max_element(tf.begin(), tf.end());   // 피크이므로 최댓값
}

int main(int argc, char **argv) {
  int iters = (argc > 1) ? atoi(argv[1]) : 20000;
  int reps = (argc > 2) ? atoi(argv[2]) : 15;
  CHECK(cudaSetDevice(0));
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  const int sm = prop.multiProcessorCount;
  const int threads = 256;
  const int bps = 4;            // SM 당 블록. 점유율을 충분히 준다

  double f32 = best_tflops(mma_f32_accum, sm, bps, threads, iters, reps);
  double f16 = best_tflops(mma_f16_accum, sm, bps, threads, iters, reps);
  CHECK(cudaDeviceSynchronize());

  int clk_khz = 0;
  cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);

  // FMA/clk/SM = TFLOP/s / (SM x 2 flop x clock)
  printf("{\"gpu\":\"%s\",\"sm_count\":%d,\"iters\":%d,\"reps\":%d,"
         "\"tflops_f32_accum\":%.2f,\"tflops_f16_accum\":%.2f,"
         "\"ratio_f16_over_f32\":%.4f,\"clock_khz_attr\":%d,"
         "\"driver_api_version\":%d,\"nvcc_version\":\"%s\"}\n",
         prop.name, sm, iters, reps, f32, f16,
         (f32 > 0 ? f16 / f32 : 0.0), clk_khz,
         kt_driver_api_version(), KT_NVCC_VERSION);
  return 0;
}
