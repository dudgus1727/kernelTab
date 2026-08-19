"""핵심 데이터클래스.

설계 원칙
---------
* frozen + slots dataclass. 뜨거운 경로(수십만 조합 열거)에서 쓰이므로
  Pydantic 같은 런타임 검증 계층을 두지 않는다.
* 측정된 시간은 이 객체들에 절대 넣지 않는다. 나중에 config를 입력으로
  받는 예측 함수가 정답을 훔쳐보는 것을 구조적으로 막기 위함이다.
* KernelConfig 는 아키텍처 공통 필드 + `ext` (아키텍처 전용 확장)로
  나뉜다. 공통 필드만으로 물리 피처(waves, tail_waste, mainloop_iters)를
  계산할 수 있어야 아키텍처 간 전이 실험이 성립한다.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "Hardware",
    "KernelConfig",
    "Problem",
    "RuntimeConfig",
    "Sm80Ext",
]


@dataclass(frozen=True, slots=True)
class Problem:
    """GEMM 형상. D[MxN] = A[MxK] @ B[KxN]."""

    M: int
    N: int
    K: int
    dtype: str = "f16"
    acc_dtype: str = "f32"
    layout_a: str = "row"
    layout_b: str = "col"
    layout_c: str = "row"


@dataclass(frozen=True, slots=True)
class Hardware:
    """런타임에 감지된 GPU 스펙.

    peak_tflops_f16 / bandwidth_gbps 를 제외한 모든 값은
    CUDA driver API 로 자동 획득한다 (core.hardware.detect_hardware).
    """

    name: str
    arch: str  # "sm_86"
    sm_count: int
    smem_per_block: int  # opt-in 최대치 (dynamic smem 포함)
    max_threads_per_sm: int
    regs_per_sm: int
    peak_tflops_f16: float  # dense FP16 입력, FP32 누산 기준
    bandwidth_gbps: float
    l2_bytes: int


@dataclass(frozen=True, slots=True)
class KernelConfig:
    """커널 하나를 결정하는 값 = 컴파일 단위.

    여기 있는 값이 바뀌면 커널을 다시 빌드해야 한다.
    alignment 는 탐색 축이 아니라 (형상, 레이아웃)에서 유도되는 값이지만
    템플릿 인자이므로 커널 정체성에 포함된다.
    """

    tile_m: int
    tile_n: int
    tile_k: int
    align_a: int
    align_b: int
    align_c: int
    arch: str  # "sm_86" 등
    ext: object  # Sm80Ext | (미래) Sm90Ext


@dataclass(frozen=True, slots=True)
class Sm80Ext:
    """SM80/86/89 (CUTLASS 2.x API) 전용 확장 필드."""

    warp_m: int
    warp_n: int
    warp_k: int
    stages: int
    swizzle_type: str  # "identity" | "horizontal"
    swizzle_n: int  # 1,2,4,8 (horizontal 이면 1)


# SM90 백엔드를 추가할 때 쓸 확장. 지금은 구현하지 않는다.
# @dataclass(frozen=True, slots=True)
# class Sm90Ext:
#     cluster_m: int; cluster_n: int
#     mainloop_schedule: str      # cooperative | pingpong | auto
#     epilogue_schedule: str
#     tile_scheduler: str         # persistent | stream_k
#     raster_order: str           # along_m | along_n
#     stages: int | None          # None = auto


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """런타임 Arguments 로 전달되는 값 = 재컴파일 불필요.

    split_k / split_k_mode 는 CUTLASS 2.x 에서 GemmUniversal::Arguments 의
    필드이지 템플릿 인자가 아니다. 이것을 KernelConfig 에 넣으면 빌드해야
    할 커널 수가 SPLIT_K x SPLIT_MODE 배로 늘어난다.
    """

    split_k: int
    split_k_mode: str  # "serial" | "parallel"
