// stages=1 이 컴파일만 되는 것이 아니라 **수치적으로 맞는가**.
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/gemm.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cstdio>
#include <cmath>

template <int STAGES>
int run(const char* tag) {
  using Gemm = cutlass::gemm::device::GemmUniversal<
      cutlass::half_t, cutlass::layout::RowMajor,
      cutlass::half_t, cutlass::layout::ColumnMajor,
      cutlass::half_t, cutlass::layout::RowMajor,
      float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
      STAGES, 8, 8>;

  int M = 256, N = 256, K = 128;
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> A({M, K});
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B({K, N});
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> C({M, N}), D({M, N}), Ref({M, N});
  cutlass::reference::host::TensorFillRandomUniform(A.host_view(), 2024, 2, -2, 0);
  cutlass::reference::host::TensorFillRandomUniform(B.host_view(), 2025, 2, -2, 0);
  cutlass::reference::host::TensorFill(C.host_view(), cutlass::half_t(0));
  A.sync_device(); B.sync_device(); C.sync_device(); D.sync_device();

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, {1.0f, 0.0f},
      A.device_data(), B.device_data(), C.device_data(), D.device_data(),
      0, 0, 0, 0, A.stride(0), B.stride(0), C.stride(0), D.stride(0)};
  Gemm op;
  auto st = op.can_implement(args);
  if (st != cutlass::Status::kSuccess) { printf("%s can_implement FAIL\n", tag); return 1; }
  if (op.initialize(args) != cutlass::Status::kSuccess) { printf("%s init FAIL\n", tag); return 1; }
  if (op() != cutlass::Status::kSuccess) { printf("%s run FAIL\n", tag); return 1; }
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("%s sync FAIL %s\n", tag, cudaGetErrorString(e)); return 1; }
  D.sync_host();

  cutlass::reference::host::Gemm<cutlass::half_t, cutlass::layout::RowMajor,
      cutlass::half_t, cutlass::layout::ColumnMajor,
      cutlass::half_t, cutlass::layout::RowMajor, float, float> ref;
  ref({M, N, K}, 1.0f, A.host_ref(), B.host_ref(), 0.0f, C.host_ref(), Ref.host_ref(), float(0));
  bool ok = cutlass::reference::host::TensorEquals(D.host_view(), Ref.host_view());
  double maxrel = 0; int nbad = 0;
  for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) {
    double d = float(D.at({i, j})), r0 = float(Ref.at({i, j}));
    double den = std::abs(r0) > 1e-6 ? std::abs(r0) : 1.0;
    double rel = std::abs(d - r0) / den;
    if (rel > 1e-3) ++nbad;
    if (rel > maxrel) maxrel = rel;
  }
  printf("%s  can_implement=OK  실행=OK  수치 일치=%s  최대상대오차=%.3g  "
         "틀린 원소 %d/%d\n", tag, ok ? "YES" : "NO", maxrel, nbad, M * N);
  return ok ? 0 : 1;
}

int main() {
  int r = 0;
  r |= run<1>("stages=1");
  r |= run<2>("stages=2");
  r |= run<3>("stages=3");
  return r;
}
