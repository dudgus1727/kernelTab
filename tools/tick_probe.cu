// CUDA 이벤트 타이머의 **눈금(양자)** 을 재기 위한 원시 표본 수집기.
//
// 왜 필요한가
// -----------
// `core/noise.py` 의 노이즈 바닥은 `max(sigma_rel(t), tick_ms / t)` 다.
// 14 us 커널에서 통계 노이즈가 2.7 % 여도 이벤트 눈금 하나가 7.3 % 면,
// 같은 눈금에 떨어진 두 config 는 **시간이 문자 그대로 동일하게 기록**된다.
// 눈금을 모르면 없는 순위를 만든다 (docs/baselines.md 2026-08-20).
//
// 눈금은 GPU/드라이버마다 다르다: A6000 1024 ns / 4090 32 ns / 5090 16 ns.
//
// 이 프로그램은 **판정하지 않는다.** 서로 다른 길이의 커널을 여러 번 재서
// 원시 ms 값을 그대로 뱉는다. 판정은 `tools/tick_report.py` 가 한다 —
// 수집과 판정을 나눠야 같은 표본으로 여러 후보를 검증할 수 있다.
//
//   nvcc -arch=sm_90a -O3 tools/tick_probe.cu -o tick_probe
//   ./tick_probe --reps 600 > /tmp/ticks.csv
//   python3 tools/tick_report.py /tmp/ticks.csv
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>


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

// 정확히 `cycles` 만큼 도는 커널. 메모리를 건드리지 않아 클럭 외의
// 변동 요인이 없다.
__global__ void spin(long long cycles, int *sink) {
  long long t0 = clock64();
  while (clock64() - t0 < cycles) { }
  if (threadIdx.x == 1u << 30) *sink = 1;   // 최적화 제거 방지
}

int main(int argc, char **argv) {
  int reps = 600;
  for (int i = 1; i < argc; ++i)
    if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);

  const int dev = 0;
  cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
  int *sink; cudaMalloc(&sink, 4);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

  // 목표 길이 (us). 짧은 쪽은 눈금이 크게 보이고, 긴 쪽은 **큰 배수에서도
  // 격자가 유지되는가**를 본다 (5090 에서 절대값 검증이 실패한 자리).
  const double targets_us[] = {2, 8, 30, 120, 500, 2000};
  const int nt = sizeof(targets_us) / sizeof(targets_us[0]);

  // ⛔ `cudaDeviceProp::clockRate` 는 **CUDA 13 에서 제거됐다.** 이 이미지
  //    (nvcc 13.3.73)에서는 컴파일 자체가 안 된다:
  //        error: class "cudaDeviceProp" has no member "clockRate"
  //    `cudaDeviceGetAttribute` 는 남아 있으므로 그쪽을 쓴다.
  //
  //    ★ 12.x 에서는 **컴파일이 통과한다.** 그래서 호스트 툴킷으로 빌드해
  //      재면 모른 채 넘어간다 — H100 세션이 정확히 그렇게 32 ns 를 보고했다
  //      (호스트 nvcc 12.8). 눈금은 드라이버/툴킷의 성질이므로 **캠페인
  //      이미지 안에서 빌드해 재라.** 두 세션이 같은 자리를 따로 밟았다.
  int clock_khz = 0;
  cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev);
  if (clock_khz <= 0) {           // 조용히 0 으로 진행하지 않는다
    fprintf(stderr, "cudaDevAttrClockRate 를 읽지 못했다\n");
    return 1;
  }
  printf("# gpu=%s sm_%d%d clock_khz=%d reps=%d "
         "driver_api_version=%d nvcc_version=%s\n",
         p.name, p.major, p.minor, clock_khz, reps,
         kt_driver_api_version(), KT_NVCC_VERSION);
  printf("target_us,elapsed_ms\n");

  for (int t = 0; t < nt; ++t) {
    // clockRate 는 부스트 최대치라 실제보다 짧게 나올 수 있다 — 목표 길이는
    // 근사면 충분하다. 눈금은 길이와 무관하다.
    long long cyc = (long long)(targets_us[t] * 1e-6 * clock_khz * 1000.0);
    if (cyc < 1) cyc = 1;
    for (int w = 0; w < 20; ++w) spin<<<1, 32>>>(cyc, sink);   // 워밍업
    cudaDeviceSynchronize();
    for (int i = 0; i < reps; ++i) {
      cudaEventRecord(e0);
      spin<<<1, 32>>>(cyc, sink);
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
      printf("%g,%.9f\n", targets_us[t], ms);   // ★ 원시값. 반올림하지 않는다
    }
  }
  cudaFree(sink);
  return 0;
}
