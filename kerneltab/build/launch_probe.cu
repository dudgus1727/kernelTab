// 빈 커널 런치 오버헤드 측정.
//
// measure/ 의 본 측정 프로토콜과 동일한 방식(L2 flush -> event bracket)으로
// "아무것도 하지 않는 커널"을 재봤을 때 나오는 시간이 곧 측정 바닥값이다.
// GEMM 측정치가 이 값의 3배 미만이면 커널 성능이 아니라 런치 오버헤드를
// 재고 있는 것이므로 status="below_launch_overhead" 로 표시한다.
//
// 출력: stdout 에 JSON 한 줄.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK(x)                                                               \
  do {                                                                         \
    cudaError_t _e = (x);                                                      \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e),      \
              __FILE__, __LINE__);                                             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void nop_kernel() {}

static double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main() {
  int dev = 0;
  CHECK(cudaSetDevice(dev));
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, dev));

  const int sm = prop.multiProcessorCount;
  const size_t l2 = (size_t)prop.l2CacheSize;
  const size_t flush_bytes = l2 * 2 + (1u << 20);

  void *flush_buf = nullptr;
  CHECK(cudaMalloc(&flush_buf, flush_bytes));
  CHECK(cudaMemset(flush_buf, 0, flush_bytes));

  cudaEvent_t ev_a, ev_b;
  CHECK(cudaEventCreate(&ev_a));
  CHECK(cudaEventCreate(&ev_b));

  // 워밍업 + 컨텍스트/모듈 로딩
  for (int i = 0; i < 1000; ++i) nop_kernel<<<1, 32>>>();
  CHECK(cudaDeviceSynchronize());

  const int N = 10000;

  // 1) 백투백 비동기 런치 처리량 (런치 큐잉 비용의 하한)
  double async_small_ms, async_grid_ms;
  {
    CHECK(cudaEventRecord(ev_a));
    for (int i = 0; i < N; ++i) nop_kernel<<<1, 32>>>();
    CHECK(cudaEventRecord(ev_b));
    CHECK(cudaEventSynchronize(ev_b));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, ev_a, ev_b));
    async_small_ms = ms / N;

    CHECK(cudaEventRecord(ev_a));
    for (int i = 0; i < N; ++i) nop_kernel<<<2 * sm, 128>>>();
    CHECK(cudaEventRecord(ev_b));
    CHECK(cudaEventSynchronize(ev_b));
    CHECK(cudaEventElapsedTime(&ms, ev_a, ev_b));
    async_grid_ms = ms / N;
  }

  // 2) 런치 + 동기화 왕복 지연
  double sync_rt_ms;
  {
    const int M = 2000;
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    std::vector<double> v;
    v.reserve(M);
    for (int i = 0; i < M; ++i) {
      CHECK(cudaEventRecord(t0));
      nop_kernel<<<1, 32>>>();
      CHECK(cudaEventRecord(t1));
      CHECK(cudaEventSynchronize(t1));
      float ms = 0;
      CHECK(cudaEventElapsedTime(&ms, t0, t1));
      v.push_back(ms);
    }
    sync_rt_ms = median(v);
    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));
  }

  // 3) 본 측정과 동일한 프로토콜: L2 flush -> event bracket -> sync
  //    이것이 results.jsonl 의 time_ms 와 직접 비교 가능한 바닥값이다.
  double bracketed_small_ms, bracketed_grid_ms;
  {
    const int M = 2000;
    std::vector<double> a, b;
    a.reserve(M);
    b.reserve(M);
    for (int i = 0; i < M; ++i) {
      CHECK(cudaMemsetAsync(flush_buf, i & 0xff, flush_bytes));
      CHECK(cudaEventRecord(ev_a));
      nop_kernel<<<1, 32>>>();
      CHECK(cudaEventRecord(ev_b));
      CHECK(cudaEventSynchronize(ev_b));
      float ms = 0;
      CHECK(cudaEventElapsedTime(&ms, ev_a, ev_b));
      a.push_back(ms);

      CHECK(cudaMemsetAsync(flush_buf, i & 0xff, flush_bytes));
      CHECK(cudaEventRecord(ev_a));
      nop_kernel<<<2 * sm, 128>>>();
      CHECK(cudaEventRecord(ev_b));
      CHECK(cudaEventSynchronize(ev_b));
      CHECK(cudaEventElapsedTime(&ms, ev_a, ev_b));
      b.push_back(ms);
    }
    bracketed_small_ms = median(a);
    bracketed_grid_ms = median(b);
  }

  printf(
      "{\"launch_async_small_ms\": %.9f, \"launch_async_grid_ms\": %.9f, "
      "\"launch_sync_roundtrip_ms\": %.9f, "
      "\"launch_bracketed_small_ms\": %.9f, \"launch_bracketed_grid_ms\": %.9f, "
      "\"l2_flush_bytes\": %zu, \"sm_count\": %d}\n",
      async_small_ms, async_grid_ms, sync_rt_ms, bracketed_small_ms,
      bracketed_grid_ms, flush_bytes, sm);

  CHECK(cudaEventDestroy(ev_a));
  CHECK(cudaEventDestroy(ev_b));
  CHECK(cudaFree(flush_buf));
  return 0;
}
