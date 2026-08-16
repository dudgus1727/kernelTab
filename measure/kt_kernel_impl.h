// 생성된 커널 .cu 의 뒤에 붙는 고정 구현부.
// 앞쪽에서 `KtGemm`, `KtElementAccumulator`, `KT_KERNEL_ID` 가 정의되어 있다.
//
// 여기에 측정 루프를 두지 않는 것은 의도적이다. 프로토콜은 libkt_ctx.so 에만
// 있고 이 파일은 "인자 구성 + 런치" 만 한다.
#pragma once

#include <cstring>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"

#include "kt_abi.h"

namespace kt {

using GemmKernel = typename KtGemm::GemmKernel;

struct Handle {
  KtGemm gemm;
};

inline cutlass::gemm::GemmUniversalMode mode_of(int m) {
  return m ? cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel
           : cutlass::gemm::GemmUniversalMode::kGemm;
}

inline typename KtGemm::Arguments make_args(const KtProblem *p,
                                            const KtBuffers *b) {
  const int64_t mn = int64_t(p->M) * int64_t(p->N);
  // A: row-major  MxK -> lda = K
  // B: col-major  KxN -> ldb = K
  // C/D: row-major MxN -> ldc = ldd = N
  return typename KtGemm::Arguments(
      mode_of(p->split_k_mode),
      cutlass::gemm::GemmCoord(p->M, p->N, p->K),
      p->split_k,
      // alpha = 1, beta = 0.
      // beta == 0 이므로 is_source_needed() == false -> C 를 읽지 않는다.
      // serial split-K 의 partition > 0 에서는 set_k_partition() 이 beta 를
      // 1 로 바꿔 이전 부분합을 정상 누적한다.
      {KtElementAccumulator(1.0f), KtElementAccumulator(0.0f)},
      b->A, b->B, b->C, b->D,
      int64_t(p->M) * p->K,  // batch_stride_A
      int64_t(p->K) * p->N,  // batch_stride_B
      mn,                    // batch_stride_C
      mn,                    // batch_stride_D  (parallel 모드의 슬라이스 간격)
      int64_t(p->K), int64_t(p->K), int64_t(p->N), int64_t(p->N));
}

}  // namespace kt

extern "C" {

const char *kt_status_string(int status) {
  return cutlassGetStatusString(static_cast<cutlass::Status>(status));
}

int kt_info(KtInfo *out) {
  std::memset(out, 0, sizeof(*out));
  std::strncpy(out->kernel_id, KT_KERNEL_ID, sizeof(out->kernel_id) - 1);

  const void *fn =
      reinterpret_cast<const void *>(&cutlass::Kernel2<kt::GemmKernel>);

  out->threads = int(kt::GemmKernel::kThreadCount);
  out->smem_dynamic = sizeof(typename kt::GemmKernel::SharedStorage);

  // 48KB 를 넘는 dynamic smem 은 opt-in 이 필요하다. occupancy 질의 전에 켠다.
  if (out->smem_dynamic >= (48u << 10)) {
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         int(out->smem_dynamic));
  }

  cudaFuncAttributes attr;
  cudaError_t e = cudaFuncGetAttributes(&attr, fn);
  if (e != cudaSuccess) return -int(e);
  out->num_regs = attr.numRegs;
  out->smem_static = attr.sharedSizeBytes;
  out->local_bytes = attr.localSizeBytes;
  out->const_bytes = attr.constSizeBytes;
  out->max_threads_per_block = attr.maxThreadsPerBlock;

  int blocks = 0;
  e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, out->threads,
                                                    out->smem_dynamic);
  out->max_blocks_per_sm = (e == cudaSuccess) ? blocks : -1;
  out->cutlass_max_blocks = KtGemm::maximum_active_blocks();
  return 0;
}

size_t kt_workspace_bytes(const KtProblem *p) {
  KtBuffers b;
  std::memset(&b, 0, sizeof(b));
  return KtGemm::get_workspace_size(kt::make_args(p, &b));
}

int kt_grid_k(const KtProblem *p) {
  KtBuffers b;
  std::memset(&b, 0, sizeof(b));
  // grid.z == grid_tiled_shape.k() == CUTLASS 가 실제로 만드는 K 슬라이스 수.
  return int(KtGemm::get_grid_shape(kt::make_args(p, &b)).z);
}

int kt_can_implement(const KtProblem *p) {
  KtBuffers b;
  std::memset(&b, 0, sizeof(b));
  return int(KtGemm::can_implement(kt::make_args(p, &b)));
}

int kt_prepare(const KtProblem *p, const KtBuffers *b, void **handle) {
  kt::Handle *h = new kt::Handle();
  cutlass::Status st = h->gemm.initialize(kt::make_args(p, b), b->workspace);
  if (st != cutlass::Status::kSuccess) {
    delete h;
    *handle = nullptr;
    return int(st);
  }
  *handle = h;
  return 0;
}

int kt_launch(void *handle, void *stream) {
  kt::Handle *h = static_cast<kt::Handle *>(handle);
  return int(h->gemm.run(static_cast<cudaStream_t>(stream)));
}

void kt_release(void *handle) { delete static_cast<kt::Handle *>(handle); }

}  // extern "C"
