// kerneltab C ABI — 커널 .so 와 컨텍스트 .so 가 공유하는 구조체.
//
// 설계: 커널 .so 는 "GemmUniversal 인스턴스화 + 런치 래퍼" 만 담는다.
// 측정 프로토콜(워밍업, 반복 수 결정, L2 flush, event 타이밍, IQR)은
// 전부 libkt_ctx.so 한 곳에 있다. 그래야
//   (1) 수천 개 TU 에 프로토콜이 복제되지 않고
//   (2) 프로토콜을 고칠 때 커널을 다시 빌드하지 않아도 된다.
// Python 은 두 .so 를 dlopen 하고 kt_launch 의 함수 포인터를 ctx 에 넘긴다.
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KtProblem {
  int M, N, K;
  int split_k;
  int split_k_mode;  // 0 = serial (kGemm), 1 = parallel (kGemmSplitKParallel)
} KtProblem;

typedef struct KtBuffers {
  void *A;
  void *B;
  void *C;
  void *D;          // parallel 이면 부분합 workspace 를 가리킨다
  void *workspace;  // serial: 세마포어 / parallel: 부분합 버퍼(= D)
  size_t workspace_bytes;
} KtBuffers;

typedef struct KtInfo {
  char kernel_id[128];
  int num_regs;             // cudaFuncGetAttributes::numRegs
  size_t smem_static;       // cudaFuncGetAttributes::sharedSizeBytes (CUTLASS 는 dynamic 을 쓰므로 보통 0)
  size_t smem_dynamic;      // sizeof(GemmKernel::SharedStorage) — 실제 사용량
  size_t local_bytes;       // localSizeBytes (스필이 여기 나타난다)
  size_t const_bytes;
  int max_threads_per_block;
  int threads;              // GemmKernel::kThreadCount
  int max_blocks_per_sm;    // cudaOccupancyMaxActiveBlocksPerMultiprocessor
  int cutlass_max_blocks;   // GemmUniversalBase::maximum_active_blocks()
} KtInfo;

// ---- 커널 .so 가 내보내는 심볼 -------------------------------------------
int kt_info(KtInfo *out);
size_t kt_workspace_bytes(const KtProblem *p);
int kt_grid_k(const KtProblem *p);        // 실제 K 슬라이스 수 (요청값과 다를 수 있다)
void kt_grid_shape(const KtProblem *p, int out_xyz[3]);  // 런치 grid (x,y,z)
void kt_tiled_shape(const KtProblem *p, int out_mnk[3]); // 논리 타일 격자 (m,n,k)
int kt_can_implement(const KtProblem *p); // 0 = ok, 그 외 = cutlass::Status
int kt_prepare(const KtProblem *p, const KtBuffers *b, void **handle);
int kt_launch(void *handle, void *stream);  // 비동기 런치 1회
void kt_release(void *handle);
const char *kt_status_string(int status);

typedef int (*KtLaunchFn)(void *handle, void *stream);

// ---- 측정 프로토콜 --------------------------------------------------------
typedef struct KtProtocol {
  double target_ms;      // 총 측정 시간 목표 (기본 20)
  double min_total_ms;   // 최소 총 측정 시간 (기본 3)
  int min_reps_floor;    // 반복 수 하한 (기본 5). IQR 사분위 계산의 최소 표본
  int min_reps_cap;      // 하한 규칙의 상한 (기본 30)
  int max_reps;          // 1000
  double warmup_frac;    // 0.2
  int min_warmup;        // 10
  double iqr_k;          // 1.5
} KtProtocol;

// 반복 수 결정 규칙 (kt_ctx.cu):
//   min_reps = clamp(ceil(min_total_ms / t), min_reps_floor, min_reps_cap)
//   n_reps   = clamp(target_ms / t,          min_reps,        max_reps)
// 고정 하한(예전의 min_reps=30)을 쓰면 느린 커널에서 최소 반복 수가 시간을
// 지배한다 (20ms 커널 x 30회 = 작업당 0.6초). 총 시간 예산으로 통일하면
// 어떤 커널이든 최소 min_total_ms 는 측정하되 그 이상은 낭비하지 않는다.

typedef struct KtMeasure {
  double time_ms;      // IQR 제거 후 중앙값
  double time_std_ms;
  double time_min_ms;
  double time_max_ms;
  int n_reps;
  int n_kept;
  double outlier_frac;
} KtMeasure;

// ---- libkt_ctx.so ---------------------------------------------------------
void *kt_ctx_create(int device);
void kt_ctx_destroy(void *ctx);

// 형상이 바뀔 때만 재할당/재초기화하고 cuBLAS 참조 결과를 계산한다.
int kt_ctx_prepare_problem(void *ctx, int M, int N, int K);
int kt_ctx_ensure_workspace(void *ctx, size_t bytes);
int kt_ctx_buffers(void *ctx, KtBuffers *out, int parallel_mode);
int kt_ctx_zero_d(void *ctx);

// launch_fn / handle 은 커널 .so 에서 얻은 것. reduce_slices > 0 이면
// 매 iteration 마다 GEMM 직후 split-K 리덕션을 같은 스트림에서 실행한다
// (리덕션이 빠지면 parallel 모드 시간이 serial 과 비교 불가능해진다).
int kt_ctx_measure(void *ctx, KtLaunchFn launch_fn, void *handle,
                   int reduce_slices, const KtProtocol *proto, KtMeasure *out);
int kt_ctx_measure_cublas(void *ctx, const KtProtocol *proto, KtMeasure *out);

// 한 번만 실행해서 결과를 남긴다 (정확도 검증용).
int kt_ctx_run_once(void *ctx, KtLaunchFn launch_fn, void *handle, int reduce_slices);

// max|D - D_ref| / max|D_ref|
double kt_ctx_max_rel_error(void *ctx);
double kt_ctx_ref_absmax(void *ctx);

const char *kt_ctx_last_error(void *ctx);

#ifdef __cplusplus
}
#endif
