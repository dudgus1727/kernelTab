// GemmUniversal 과 호환되는 horizontal(= N 방향 우선) 래스터 스위즐.
//
// CUTLASS 의 `GemmHorizontalThreadblockSwizzle` 은 get_tile_offset(GemmCoord)
// 시그니처라 `kernel::GemmUniversal` (get_tile_offset(int log_tile) 로 호출)
// 과 컴파일되지 않는다. 구버전 device::Gemm 경로 전용이다.
//
// 래스터 방향은 L2 재사용에 직접 영향을 주는 진짜 config 축이고
// (SM90 3.x API 에서는 raster_order = along_m | along_n 로 명시된다),
// 이 축을 잃으면 나중에 SM80 <-> SM90 전이 실험이 불가능해진다.
// 그래서 CUTLASS 의 horizontal 동작을 그대로 GemmUniversal 인터페이스에
// 맞춰 다시 쓴다: grid = (n_tiles, m_tiles, k), offset = (blockIdx.y, blockIdx.x, blockIdx.z).
//
// 비교 대상인 GemmIdentityThreadblockSwizzle<1> 은
// grid = (m_tiles, n_tiles, k), offset = (blockIdx.x, blockIdx.y, blockIdx.z) 로
// M 방향 우선 래스터다. 둘의 차이가 곧 래스터 방향의 효과다.
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

namespace kt {

struct HorizontalThreadblockSwizzle {
  CUTLASS_HOST_DEVICE
  HorizontalThreadblockSwizzle() {}

  CUTLASS_HOST_DEVICE
  static cutlass::gemm::GemmCoord get_tiled_shape(
      cutlass::gemm::GemmCoord problem_size,
      cutlass::gemm::GemmCoord tile_size, int split_k_slices) {
    return cutlass::gemm::GemmCoord(
        (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
        (problem_size.n() + tile_size.n() - 1) / tile_size.n(), split_k_slices);
  }

  CUTLASS_HOST_DEVICE
  static int get_log_tile(cutlass::gemm::GemmCoord /*tiled_shape*/) { return 0; }

  CUTLASS_HOST_DEVICE
  static dim3 get_grid_shape(cutlass::gemm::GemmCoord tiled_shape) {
    return dim3(tiled_shape.n(), tiled_shape.m(), tiled_shape.k());
  }

  CUTLASS_DEVICE
  static cutlass::gemm::GemmCoord get_tile_offset(int /*log_tile*/) {
    return cutlass::gemm::GemmCoord{
        cutlass::gemm::threadblock::RematerializeBlockIdxY(),
        cutlass::gemm::threadblock::RematerializeBlockIdxX(),
        cutlass::gemm::threadblock::RematerializeBlockIdxZ()};
  }

  CUTLASS_DEVICE
  static cutlass::gemm::GemmCoord get_tile_offset(
      cutlass::gemm::GemmCoord /*tiled_shape*/) {
    return get_tile_offset(0);
  }
};

}  // namespace kt
