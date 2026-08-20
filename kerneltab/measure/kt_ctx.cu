// libkt_ctx.so — 측정 컨텍스트와 프로토콜.
//
// 여기 한 곳에만 프로토콜이 있다. 커널 .so 는 런치만 한다.
//   - 버퍼 소유 및 재사용 (형상이 바뀔 때만 재할당)
//   - 입력 생성: 1/4 배수 값. fp16 에서 정확히 표현되고 곱/합이 fp32 에서
//     오차 없이 누적되므로, 정답 비교가 "부동소수점 잡음" 이 아니라 진짜
//     버그(예: alignment 오용)만 잡아낸다.
//   - cuBLAS 참조 (정답 + 참조 성능)
//   - split-K parallel 리덕션
//   - 반복 수 결정 / L2 flush / CUDA event 타이밍 / IQR 이상치 제거
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kt_abi.h"

namespace {

struct Buf {
  void *ptr = nullptr;
  size_t bytes = 0;

  cudaError_t ensure(size_t need) {
    if (need <= bytes) return cudaSuccess;
    if (ptr) cudaFree(ptr);
    ptr = nullptr;
    bytes = 0;
    cudaError_t e = cudaMalloc(&ptr, need);
    if (e != cudaSuccess) return e;
    bytes = need;
    return cudaSuccess;
  }
  void free_() {
    if (ptr) cudaFree(ptr);
    ptr = nullptr;
    bytes = 0;
  }
};

struct Ctx {
  int device = 0;
  cublasHandle_t cublas = nullptr;
  cudaEvent_t ev_a = nullptr, ev_b = nullptr;

  Buf A, B, C, D, Dref, ws, flush;
  size_t flush_bytes = 0;

  int M = 0, N = 0, K = 0;
  double ref_absmax = 0.0;

  std::string err;
};

// ---------------------------------------------------------------------------
// 입력 생성
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned hash_u32(unsigned x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

// {-1, -0.75, ..., 0.75} — fp16 에서 정확, 곱도 정확(1/16 배수),
// K <= 2^20 까지 fp32 누산도 정확.
__global__ void fill_kernel(__half *p, size_t n, unsigned seed) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) {
    unsigned h = hash_u32(unsigned(i) ^ seed);
    int q = int(h & 7u) - 4;  // [-4, 3]
    p[i] = __float2half(float(q) * 0.25f);
  }
}

// ---------------------------------------------------------------------------
// split-K parallel 리덕션: workspace[slices][mn] (fp16) -> D (fp16)
// 부분합 저장은 fp16(= ElementC)이지만 누산은 fp32 로 한다.
// ---------------------------------------------------------------------------
__global__ void reduce_kernel(__half *__restrict__ D,
                              const __half *__restrict__ W, int slices,
                              size_t mn) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  for (; i < mn; i += stride) {
    float acc = 0.f;
    for (int s = 0; s < slices; ++s) acc += __half2float(W[size_t(s) * mn + i]);
    D[i] = __float2half(acc);
  }
}

// ---------------------------------------------------------------------------
// 오차: max|D - Dref| 와 max|Dref| (음이 아닌 float 은 비트패턴 순서가
// unsigned 정수 순서와 같아서 atomicMax 를 그대로 쓸 수 있다)
// ---------------------------------------------------------------------------
__global__ void diff_kernel(const __half *__restrict__ D,
                            const __half *__restrict__ R, size_t n,
                            unsigned *max_diff, unsigned *max_ref) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  float ld = 0.f, lr = 0.f;
  for (; i < n; i += stride) {
    float d = __half2float(D[i]);
    float r = __half2float(R[i]);
    ld = fmaxf(ld, fabsf(d - r));
    lr = fmaxf(lr, fabsf(r));
  }
  atomicMax(max_diff, __float_as_uint(ld));
  atomicMax(max_ref, __float_as_uint(lr));
}

inline int grid_for(size_t n, int block, int cap = 4096) {
  size_t g = (n + block - 1) / block;
  return int(g < size_t(cap) ? (g ? g : 1) : size_t(cap));
}

#define CK(ctx, expr)                                                     \
  do {                                                                    \
    cudaError_t _e = (expr);                                              \
    if (_e != cudaSuccess) {                                              \
      (ctx)->err = std::string(#expr) + ": " + cudaGetErrorString(_e);    \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// D = A(row MxK) * B(col KxN), row-major 출력.
// column-major 인 cuBLAS 로는 D^T[N,M] = B^T * A 로 계산한다.
inline cublasStatus_t cublas_gemm(Ctx *c, cudaStream_t stream) {
  const float alpha = 1.f, beta = 0.f;
  cublasSetStream(c->cublas, stream);
  return cublasGemmEx(c->cublas, CUBLAS_OP_T, CUBLAS_OP_N, c->N, c->M, c->K,
                      &alpha, c->B.ptr, CUDA_R_16F, c->K, c->A.ptr, CUDA_R_16F,
                      c->K, &beta, c->D.ptr, CUDA_R_16F, c->N,
                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// ---------------------------------------------------------------------------
// 통계
// ---------------------------------------------------------------------------
double quantile(const std::vector<double> &s, double q) {
  if (s.empty()) return 0.0;
  double pos = q * (double(s.size()) - 1.0);
  size_t lo = size_t(pos);
  size_t hi = std::min(lo + 1, s.size() - 1);
  double f = pos - double(lo);
  return s[lo] * (1.0 - f) + s[hi] * f;
}

void summarize(std::vector<double> t, double iqr_k, KtMeasure *out) {
  out->n_reps = int(t.size());
  std::sort(t.begin(), t.end());
  double q1 = quantile(t, 0.25), q3 = quantile(t, 0.75);
  double iqr = q3 - q1;
  double lo = q1 - iqr_k * iqr, hi = q3 + iqr_k * iqr;

  std::vector<double> kept;
  kept.reserve(t.size());
  for (double v : t)
    if (v >= lo && v <= hi) kept.push_back(v);
  if (kept.empty()) kept = t;

  out->n_kept = int(kept.size());
  out->outlier_frac =
      double(out->n_reps - out->n_kept) / double(std::max(1, out->n_reps));
  out->time_ms = quantile(kept, 0.5);
  out->time_min_ms = kept.front();
  out->time_max_ms = kept.back();

  double mean = 0;
  for (double v : kept) mean += v;
  mean /= double(kept.size());
  double var = 0;
  for (double v : kept) var += (v - mean) * (v - mean);
  out->time_std_ms =
      kept.size() > 1 ? std::sqrt(var / double(kept.size() - 1)) : 0.0;
}

// 측정 대상 1회 = GEMM (+ parallel 이면 리덕션). 같은 스트림에 순서대로 넣는다.
inline int issue(Ctx *c, KtLaunchFn fn, void *handle, int reduce_slices,
                 cudaStream_t stream) {
  int st = fn(handle, stream);
  if (st != 0) return st;
  if (reduce_slices > 0) {
    size_t mn = size_t(c->M) * size_t(c->N);
    reduce_kernel<<<grid_for(mn, 256), 256, 0, stream>>>(
        static_cast<__half *>(c->D.ptr), static_cast<const __half *>(c->ws.ptr),
        reduce_slices, mn);
  }
  return 0;
}

}  // namespace

extern "C" {

void *kt_ctx_create(int device) {
  Ctx *c = new Ctx();
  c->device = device;
  if (cudaSetDevice(device) != cudaSuccess) {
    delete c;
    return nullptr;
  }
  if (cublasCreate(&c->cublas) != CUBLAS_STATUS_SUCCESS) {
    delete c;
    return nullptr;
  }
  cublasSetMathMode(c->cublas, CUBLAS_TENSOR_OP_MATH);
  cudaEventCreate(&c->ev_a);
  cudaEventCreate(&c->ev_b);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  c->flush_bytes = size_t(prop.l2CacheSize) * 2 + (1u << 20);
  if (c->flush.ensure(c->flush_bytes) != cudaSuccess) {
    delete c;
    return nullptr;
  }
  cudaMemset(c->flush.ptr, 0, c->flush_bytes);
  return c;
}

void kt_ctx_destroy(void *ctxp) {
  Ctx *c = static_cast<Ctx *>(ctxp);
  if (!c) return;
  c->A.free_();
  c->B.free_();
  c->C.free_();
  c->D.free_();
  c->Dref.free_();
  c->ws.free_();
  c->flush.free_();
  if (c->cublas) cublasDestroy(c->cublas);
  if (c->ev_a) cudaEventDestroy(c->ev_a);
  if (c->ev_b) cudaEventDestroy(c->ev_b);
  delete c;
}

const char *kt_ctx_last_error(void *ctxp) {
  Ctx *c = static_cast<Ctx *>(ctxp);
  return c ? c->err.c_str() : "null ctx";
}

int kt_ctx_prepare_problem(void *ctxp, int M, int N, int K) {
  Ctx *c = static_cast<Ctx *>(ctxp);
  if (c->M == M && c->N == N && c->K == K) return 0;  // 이미 준비됨
  c->err.clear();

  size_t mk = size_t(M) * K, kn = size_t(K) * N, mn = size_t(M) * N;
  CK(c, c->A.ensure(mk * 2));
  CK(c, c->B.ensure(kn * 2));
  CK(c, c->C.ensure(mn * 2));
  CK(c, c->D.ensure(mn * 2));
  CK(c, c->Dref.ensure(mn * 2));

  c->M = M;
  c->N = N;
  c->K = K;

  fill_kernel<<<grid_for(mk, 256), 256>>>(static_cast<__half *>(c->A.ptr), mk,
                                          0x1234u);
  fill_kernel<<<grid_for(kn, 256), 256>>>(static_cast<__half *>(c->B.ptr), kn,
                                          0x9abcu);
  CK(c, cudaMemset(c->C.ptr, 0, mn * 2));
  CK(c, cudaMemset(c->D.ptr, 0, mn * 2));
  CK(c, cudaDeviceSynchronize());

  // cuBLAS 참조를 D 에 만든 뒤 Dref 로 복사한다.
  if (cublas_gemm(c, nullptr) != CUBLAS_STATUS_SUCCESS) {
    c->err = "cublasGemmEx (reference) failed";
    return -1;
  }
  CK(c, cudaDeviceSynchronize());
  CK(c, cudaMemcpy(c->Dref.ptr, c->D.ptr, mn * 2, cudaMemcpyDeviceToDevice));

  // max|Dref|
  Buf tmp;
  CK(c, tmp.ensure(2 * sizeof(unsigned)));
  CK(c, cudaMemset(tmp.ptr, 0, 2 * sizeof(unsigned)));
  diff_kernel<<<grid_for(mn, 256), 256>>>(
      static_cast<const __half *>(c->Dref.ptr),
      static_cast<const __half *>(c->Dref.ptr), mn,
      static_cast<unsigned *>(tmp.ptr), static_cast<unsigned *>(tmp.ptr) + 1);
  unsigned h[2];
  CK(c, cudaMemcpy(h, tmp.ptr, sizeof(h), cudaMemcpyDeviceToHost));
  tmp.free_();
  float ref;
  std::memcpy(&ref, &h[1], 4);
  c->ref_absmax = double(ref);
  return 0;
}

int kt_ctx_ensure_workspace(void *ctxp, size_t bytes) {
  Ctx *c = static_cast<Ctx *>(ctxp);
  if (bytes == 0) return 0;
  CK(c, c->ws.ensure(bytes));
  return 0;
}

int kt_ctx_buffers(void *ctxp, KtBuffers *out, int parallel_mode) {
  Ctx *c = static_cast<Ctx *>(ctxp);
  out->A = c->A.ptr;
  out->B = c->B.ptr;
  out->C = c->C.ptr;
  // parallel 모드에서는 GEMM 의 D 가 부분합 workspace 다 (CUTLASS 프로파일러와
  // 동일). 리덕션이 그것을 읽어 진짜 D 를 만든다.
  out->D = parallel_mode ? c->ws.ptr : c->D.ptr;
  out->workspace = c->ws.ptr;
  out->workspace_bytes = c->ws.bytes;
  return 0;
}

int kt_ctx_zero_d(void *ctxp) {
  Ctx *c = static_cast<Ctx *>(ctxp);
  CK(c, cudaMemset(c->D.ptr, 0, size_t(c->M) * c->N * 2));
  return 0;
}

double kt_ctx_ref_absmax(void *ctxp) {
  return static_cast<Ctx *>(ctxp)->ref_absmax;
}

double kt_ctx_max_rel_error(void *ctxp) {
  Ctx *c = static_cast<Ctx *>(ctxp);
  size_t mn = size_t(c->M) * size_t(c->N);
  Buf tmp;
  if (tmp.ensure(2 * sizeof(unsigned)) != cudaSuccess) return -1.0;
  cudaMemset(tmp.ptr, 0, 2 * sizeof(unsigned));
  diff_kernel<<<grid_for(mn, 256), 256>>>(
      static_cast<const __half *>(c->D.ptr),
      static_cast<const __half *>(c->Dref.ptr), mn,
      static_cast<unsigned *>(tmp.ptr), static_cast<unsigned *>(tmp.ptr) + 1);
  unsigned h[2];
  if (cudaMemcpy(h, tmp.ptr, sizeof(h), cudaMemcpyDeviceToHost) != cudaSuccess) {
    tmp.free_();
    return -1.0;
  }
  tmp.free_();
  float dmax, rmax;
  std::memcpy(&dmax, &h[0], 4);
  std::memcpy(&rmax, &h[1], 4);
  if (rmax <= 0.f) return double(dmax);
  return double(dmax) / double(rmax);
}

int kt_ctx_run_once(void *ctxp, KtLaunchFn fn, void *handle,
                    int reduce_slices) {
  Ctx *c = static_cast<Ctx *>(ctxp);
  c->err.clear();
  CK(c, cudaMemset(c->D.ptr, 0, size_t(c->M) * c->N * 2));
  int st = issue(c, fn, handle, reduce_slices, nullptr);
  if (st != 0) {
    c->err = "kt_launch returned cutlass status " + std::to_string(st);
    return st;
  }
  CK(c, cudaDeviceSynchronize());
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    c->err = std::string("kernel launch: ") + cudaGetErrorString(e);
    return -1;
  }
  return 0;
}

// 프로토콜 본체. cuBLAS 측정도 같은 루프를 쓴다 (fn == nullptr).
static int measure_impl(Ctx *c, KtLaunchFn fn, void *handle, int reduce_slices,
                        const KtProtocol *proto, KtMeasure *out) {
  c->err.clear();
  std::memset(out, 0, sizeof(*out));

  auto one = [&](cudaStream_t s) -> int {
    if (fn) return issue(c, fn, handle, reduce_slices, s);
    return cublas_gemm(c, s) == CUBLAS_STATUS_SUCCESS ? 0 : -1;
  };

  // 1) 짧은 예비 실행으로 1회 소요 시간을 추정한다.
  //
  //    예전에는 무조건 max(3, min_warmup) = 10 회였다. 491 ms 짜리 커널이면
  //    **추정에만 4.9 초**다. 그래서 먼저 3 회만 돌려 재고, 그것이
  //    probe_budget_ms 보다 짧을 때만 표본을 늘린다.
  //
  //    3 회를 하한으로 두는 이유: 1 회째는 디바이스 모듈 로드가 섞이므로
  //    (docs/measurement_drift.md) 그 값으로 n_reps 를 정하면 안 된다.
  const int probe_floor = 3;
  const int probe_max = std::max(probe_floor, proto->min_warmup);
  int probe = probe_floor;
  float probe_ms = 0;
  CK(c, cudaEventRecord(c->ev_a));
  for (int i = 0; i < probe; ++i) {
    int st = one(nullptr);
    if (st != 0) {
      c->err = "launch failed (probe), status " + std::to_string(st);
      return st ? st : -1;
    }
  }
  CK(c, cudaEventRecord(c->ev_b));
  CK(c, cudaEventSynchronize(c->ev_b));
  CK(c, cudaEventElapsedTime(&probe_ms, c->ev_a, c->ev_b));

  if (proto->probe_budget_ms > 0 && probe_ms < proto->probe_budget_ms
      && probe < probe_max) {
    // 시간이 남으면 표본을 늘려 추정을 다듬는다. 짧은 커널은 예전과 같은
    // 횟수까지 간다.
    double per = double(probe_ms) / double(probe);
    int want = per > 0 ? int(proto->probe_budget_ms / per) : probe_max;
    int extra = std::min(probe_max, std::max(probe, want)) - probe;
    if (extra > 0) {
      CK(c, cudaEventRecord(c->ev_a));
      for (int i = 0; i < extra; ++i) {
        int st = one(nullptr);
        if (st != 0) {
          c->err = "launch failed (probe2), status " + std::to_string(st);
          return st ? st : -1;
        }
      }
      CK(c, cudaEventRecord(c->ev_b));
      CK(c, cudaEventSynchronize(c->ev_b));
      float more = 0;
      CK(c, cudaEventElapsedTime(&more, c->ev_a, c->ev_b));
      probe_ms += more;
      probe += extra;
    }
  }
  cudaError_t le = cudaGetLastError();
  if (le != cudaSuccess) {
    c->err = std::string("probe launch: ") + cudaGetErrorString(le);
    return -1;
  }
  double per_ms = double(probe_ms) / double(probe);

  // 2) 총 시간 예산으로 반복 수를 정한다.
  //    min_reps = clamp(ceil(min_total_ms / t), floor, cap)
  //    n_reps   = clamp(target_ms / t,          min_reps, max_reps)
  int min_reps = proto->min_reps_floor;
  if (per_ms > 0) {
    int by_budget = int(std::ceil(proto->min_total_ms / per_ms));
    min_reps = std::max(proto->min_reps_floor,
                        std::min(proto->min_reps_cap, by_budget));
  }
  int n_reps = min_reps;
  if (per_ms > 0) {
    double want = proto->target_ms / per_ms;
    n_reps = int(want < 1.0 ? 1.0 : want);
  }
  n_reps = std::max(min_reps, std::min(proto->max_reps, n_reps));

  // 3) 워밍업: 본 측정 반복 수의 warmup_frac 또는 최소 min_warmup.
  //    여기에 **시간 상한**을 씌운다 (warmup_budget_ms). 상한이지 하한이
  //    아니므로 짧은 커널은 전혀 줄지 않는다 — 캐시/클럭 상태가 중요한
  //    쪽은 거기이고, 낭비는 전부 느린 꼬리에 있었다.
  //
  //    ⚠️ 프로세스 시작 시의 워밍업(rehearse.py, 20 초)과 다른 것이다.
  //    그쪽은 메모리 클럭 램프업용이고 여기는 작업당 캐시 워밍업이다.
  //    그쪽 하한은 건드리지 않는다.
  int warm = std::max(proto->min_warmup, int(proto->warmup_frac * n_reps));
  if (proto->warmup_budget_ms > 0 && per_ms > 0) {
    int by_time = int(proto->warmup_budget_ms / per_ms);
    int floor_reps = std::max(1, proto->warmup_reps_floor);
    warm = std::min(warm, std::max(floor_reps, by_time));
  }
  out->n_probe = probe;
  out->n_warmup = warm;
  out->overhead_ms = double(probe_ms) + double(warm) * per_ms;
  for (int i = 0; i < warm; ++i) {
    if (one(nullptr) != 0) {
      c->err = "launch failed (warmup)";
      return -1;
    }
  }
  CK(c, cudaDeviceSynchronize());

  // 4) 본 측정. 매 iteration 앞에서 L2 를 flush 한다.
  std::vector<double> t;
  t.reserve(n_reps);
  for (int i = 0; i < n_reps; ++i) {
    CK(c, cudaMemsetAsync(c->flush.ptr, i & 0xff, c->flush_bytes));
    CK(c, cudaEventRecord(c->ev_a));
    if (one(nullptr) != 0) {
      c->err = "launch failed (timed)";
      return -1;
    }
    CK(c, cudaEventRecord(c->ev_b));
    CK(c, cudaEventSynchronize(c->ev_b));
    float ms = 0;
    CK(c, cudaEventElapsedTime(&ms, c->ev_a, c->ev_b));
    t.push_back(double(ms));
  }
  le = cudaGetLastError();
  if (le != cudaSuccess) {
    c->err = std::string("timed launch: ") + cudaGetErrorString(le);
    return -1;
  }

  summarize(std::move(t), proto->iqr_k, out);
  return 0;
}

int kt_ctx_measure(void *ctxp, KtLaunchFn fn, void *handle, int reduce_slices,
                   const KtProtocol *proto, KtMeasure *out) {
  return measure_impl(static_cast<Ctx *>(ctxp), fn, handle, reduce_slices,
                      proto, out);
}

int kt_ctx_measure_cublas(void *ctxp, const KtProtocol *proto,
                          KtMeasure *out) {
  Ctx *c = static_cast<Ctx *>(ctxp);
  int rc = measure_impl(c, nullptr, nullptr, 0, proto, out);
  // cuBLAS 측정이 D 를 덮어썼으므로 참조와 동일한 상태다. 문제없음.
  return rc;
}

}  // extern "C"

extern "C" int kt_abi_version(void) { return KT_ABI_VERSION; }

extern "C" int kt_abi_sizeof(int which) {
  switch (which) {
    case KT_ABI_PROBLEM:  return int(sizeof(KtProblem));
    case KT_ABI_BUFFERS:  return int(sizeof(KtBuffers));
    case KT_ABI_PROTOCOL: return int(sizeof(KtProtocol));
    case KT_ABI_MEASURE:  return int(sizeof(KtMeasure));
    default: return -1;
  }
}
