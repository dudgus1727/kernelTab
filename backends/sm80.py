"""SM80/86/89 백엔드 — CUTLASS 2.x GEMM API (device::GemmUniversal).

3.x API(CollectiveBuilder / GemmUniversalAdapter / CuTe)는 SM90 이상 전용이라
sm_86 에서 동작하지 않는다. CUTLASS 최신 릴리스를 쓰되 API 계열은 2.x 다.

ArchTag 는 sm_86 에서도 `arch::Sm80` 이 정상이다. ArchTag 는 "이 기능을
지원하는 최소 SM"을 뜻하고 Sm86 태그는 2.x GEMM 경로에 존재하지 않는다.
실제 타겟 SM 은 nvcc -arch 플래그가 결정한다.
"""

from __future__ import annotations

from math import ceil

from core.types import Hardware, KernelConfig, Problem, RuntimeConfig, Sm80Ext

# ---------------------------------------------------------------------------
# 탐색 축
# ---------------------------------------------------------------------------
TB_TILES = [
    (32, 64), (32, 128), (64, 64), (64, 128), (64, 256), (128, 64),
    (128, 128), (128, 256), (256, 64), (256, 128), (256, 256),
]
TB_K = [32, 64]
WARP_TILES = [
    (16, 64), (32, 32), (32, 64), (64, 32), (64, 64),
    (64, 128), (128, 64), (128, 128),
]
STAGES = [2, 3, 4, 5, 6, 7, 8]
SWIZZLE = [
    ("identity", 1), ("identity", 2), ("identity", 4), ("identity", 8),
    ("horizontal", 1),
]

# SPLIT_K 에 3, 6, 12 가 들어있는 것은 의도적이다.
# A6000 의 SM 개수 84 = 2^2 x 3 x 7 이라 3의 배수 split-K 가 wave 정렬에
# 유리할 수 있다는 가설을 검증하기 위함이다. {1,2,4,8} 로 줄이지 말 것.
SPLIT_K = [1, 2, 3, 4, 6, 8, 12, 16]
SPLIT_MODE = ["serial", "parallel"]

# mma.m16n8k16 (fp16 입력, fp32 누산)
INSTRUCTION_SHAPE = (16, 8, 16)

#: 스레드당 누산기 레지스터 상한. 초과하면 스필이 확정적이다.
MAX_ACCUM_REGS_PER_THREAD = 200

#: 에필로그 smem 의 열 방향 패딩. CUTLASS DefaultEpilogueTensorOp 의
#: Padding = MatrixShape<0, 64 / sizeof_bits<ElementAccumulator> * 4> 에서 유도.
#: ElementAccumulator = float (32bit) 이므로 64/32*4 = 8.
_EPILOGUE_PAD_COLS = 8
_ACC_BYTES = 4


def warp_k_options(tile_k: int) -> list[int]:
    """K 방향 warp 분할. 기본은 tile_k (분할 없음)."""
    return [tile_k] if tile_k <= 32 else [tile_k, tile_k // 2]


class Sm80Backend:
    arch_family = ("sm_80", "sm_86", "sm_89")

    # -- 열거 -------------------------------------------------------------
    def enumerate_ext(self, hw: Hardware) -> list:
        out = []
        for tm, tn in TB_TILES:
            for tk in TB_K:
                for wm, wn in WARP_TILES:
                    for wk in warp_k_options(tk):
                        for swz_type, swz_n in SWIZZLE:
                            for st in STAGES:
                                out.append((
                                    (tm, tn, tk),
                                    Sm80Ext(
                                        warp_m=wm, warp_n=wn, warp_k=wk,
                                        stages=st,
                                        swizzle_type=swz_type, swizzle_n=swz_n,
                                    ),
                                ))
        return out

    def enumerate_runtime(self, p: Problem, cfg: KernelConfig) -> list[RuntimeConfig]:
        out = []
        for sk in SPLIT_K:
            for mode in SPLIT_MODE:
                rc = RuntimeConfig(split_k=sk, split_k_mode=mode)
                if self.is_valid_runtime(rc, p, cfg):
                    out.append(rc)
        return out

    # -- 자원 -------------------------------------------------------------
    def smem_bytes(self, cfg: KernelConfig, dtype_bytes: int) -> int:
        """kernel::GemmUniversal::SharedStorage = union(mainloop, epilogue).

        mainloop (MmaBase::SharedStorage, SmemPadding = <0,0> for tensor op):
            A: (tile_m) x (tile_k * stages)
            B: (tile_k * stages) x (tile_n)
        epilogue (EpilogueBase::SharedStorage, element = ElementAccumulator):
            rows = warps_m * kRowsPerIteration(8) * warps_k * kFragmentsPerIteration
            cols = tile_n + Padding(8)
            kFragmentsPerIteration = 2 if warps_k == 1 else 1
        """
        e: Sm80Ext = cfg.ext  # type: ignore[assignment]
        mainloop = e.stages * cfg.tile_k * (cfg.tile_m + cfg.tile_n) * dtype_bytes

        warps_m = cfg.tile_m // e.warp_m
        warps_n = cfg.tile_n // e.warp_n
        warps_k = cfg.tile_k // e.warp_k
        frags = 2 if warps_k == 1 else 1
        rows = warps_m * 8 * warps_k * frags
        cols = warps_n * e.warp_n + _EPILOGUE_PAD_COLS
        epilogue = rows * cols * _ACC_BYTES

        return max(mainloop, epilogue)

    def expected_hmma(
        self,
        cfg: KernelConfig,
        rc: RuntimeConfig | None = None,
        p: Problem | None = None,
    ) -> int:
        e: Sm80Ext = cfg.ext  # type: ignore[assignment]
        im, in_, ik = INSTRUCTION_SHAPE
        per_warp_kgroup = (e.warp_m // im) * (e.warp_n // in_)
        kgroups = e.warp_k // ik
        static_per_warp = per_warp_kgroup * kgroups
        if rc is None or p is None:
            # SASS 정적 카운트와 비교할 값 (워프 하나가 메인루프 한 번을
            # 완전히 펼쳤을 때 나오는 HMMA 수).
            return static_per_warp
        # 동적 총 개수: 문제 전체 grid 에서 실행되는 HMMA 수.
        warps_per_tb = (cfg.tile_m // e.warp_m) * (cfg.tile_n // e.warp_n) * (
            cfg.tile_k // e.warp_k
        )
        tbs = (
            ceil(p.M / cfg.tile_m) * ceil(p.N / cfg.tile_n) * rc.split_k
        )
        k_iters = ceil(p.K / (cfg.tile_k * rc.split_k))
        return tbs * warps_per_tb * static_per_warp * k_iters

    # -- 유효성 -----------------------------------------------------------
    def explain_kernel(
        self, cfg: KernelConfig, hw: Hardware, dtype_bytes: int
    ) -> str | None:
        """컴파일/실행 가능성만 본다. 느릴 것 같다는 이유로 거르지 않는다."""
        e: Sm80Ext = cfg.ext  # type: ignore[assignment]

        if cfg.tile_m % e.warp_m or cfg.tile_n % e.warp_n:
            return "tile_divisible_by_warp"

        n_warps = (cfg.tile_m // e.warp_m) * (cfg.tile_n // e.warp_n)
        if n_warps not in (4, 8):
            return "warp_count_4_or_8"

        im, in_, _ = INSTRUCTION_SHAPE
        if e.warp_m % im or e.warp_n % in_:
            return "warp_shape_vs_instruction"

        if cfg.tile_k % e.warp_k or e.warp_k % INSTRUCTION_SHAPE[2]:
            return "warp_k_divisible"

        if self.smem_bytes(cfg, dtype_bytes) > hw.smem_per_block:
            return "smem_capacity"

        if (e.warp_m * e.warp_n) / 32 > MAX_ACCUM_REGS_PER_THREAD:
            return "accumulator_registers"

        if e.swizzle_type == "horizontal" and e.swizzle_n != 1:
            return "horizontal_swizzle_n"

        return None

    def is_valid_kernel(self, cfg: KernelConfig, hw: Hardware, dtype_bytes: int) -> bool:
        return self.explain_kernel(cfg, hw, dtype_bytes) is None

    # -- split-K 의미론 (CUTLASS 2.x 실제 동작) ---------------------------
    #
    # kernel/params_universal_base.h : init_grid_tiled_shape()
    #   cacheline_elements   = 128 / sizeof(Element)              (fp16 -> 64)
    #   cacheline_needed     = (A row-major && K % 64 == 0)
    #                          || (B col-major && K % 64 == 0)
    #   kAlignK              = max(8, cacheline_needed ? 64 : 1)
    #   gemm_k_size          = round_up(ceil_div(K, split_k), kAlignK)
    #   grid_tiled_shape.k() = ceil_div(K, gemm_k_size)      <-- 실제 슬라이스 수
    #
    # 즉 CUTLASS 는 K 가 split_k 로 나누어떨어지지 않아도 알아서 자른다.
    # 다만 요청한 split_k 와 실제 슬라이스 수가 다를 수 있고(예: K=512, sk=12
    # -> 실제 8), 그러면 라벨과 실측 대상이 어긋나 같은 것을 다른 이름으로
    # 두 번 재게 된다. 그래서 "요청 == 실제" 를 유효성 조건으로 쓴다.
    @staticmethod
    def align_k(p: Problem, dtype_bytes: int = 2) -> int:
        cacheline_elems = 128 // dtype_bytes
        contiguous_k = (p.layout_a == "row") or (p.layout_b == "col")
        needed = contiguous_k and (p.K % cacheline_elems == 0)
        return max(8, cacheline_elems if needed else 1)

    def gemm_k_size(self, p: Problem, rc: RuntimeConfig, dtype_bytes: int = 2) -> int:
        ak = self.align_k(p, dtype_bytes)
        per = -(-p.K // rc.split_k)  # ceil_div
        return ((per + ak - 1) // ak) * ak

    def effective_split_k(
        self, p: Problem, rc: RuntimeConfig, dtype_bytes: int = 2
    ) -> int:
        gks = self.gemm_k_size(p, rc, dtype_bytes)
        return -(-p.K // gks) if gks else 1

    def workspace_bytes(
        self, p: Problem, cfg: KernelConfig, rc: RuntimeConfig, elem_c_bytes: int = 2
    ) -> int:
        """GemmUniversalBase::get_workspace_size() 와 동일한 계산."""
        k = self.effective_split_k(p, rc)
        if rc.split_k_mode == "parallel":
            return elem_c_bytes * p.M * p.N * k
        if k > 1:  # serial split-K 는 타일별 세마포어만 필요
            return 4 * ceil(p.M / cfg.tile_m) * ceil(p.N / cfg.tile_n)
        return 0

    def is_valid_runtime(
        self, rc: RuntimeConfig, p: Problem, cfg: KernelConfig
    ) -> bool:
        from core.config import alignments_for

        # 슬라이스 하나가 최소한 K 타일 하나는 담아야 한다.
        if rc.split_k * cfg.tile_k > p.K:
            return False
        # 요청한 split_k 를 CUTLASS 가 실제로 만들어내는가.
        if self.effective_split_k(p, rc) != rc.split_k:
            return False
        # split_k == 1 이면 serial/parallel 이 같은 커널이다. 중복 제거.
        if rc.split_k == 1 and rc.split_k_mode != "serial":
            return False
        # alignment 는 형상에서 유도되는 값이므로 형상과 커널이 일치해야 한다.
        if (cfg.align_a, cfg.align_b, cfg.align_c) != alignments_for(p):
            return False
        return True

    # -- 식별자 / 코드 생성 ------------------------------------------------
    def kernel_id(self, cfg: KernelConfig) -> str:
        e: Sm80Ext = cfg.ext  # type: ignore[assignment]
        sw = "id" if e.swizzle_type == "identity" else "hz"
        return (
            f"{cfg.arch.replace('_', '')}"
            f"_tb{cfg.tile_m}x{cfg.tile_n}x{cfg.tile_k}"
            f"_w{e.warp_m}x{e.warp_n}x{e.warp_k}"
            f"_st{e.stages}"
            f"_sw{sw}{e.swizzle_n}"
            f"_a{cfg.align_a}{cfg.align_b}{cfg.align_c}"
        )

    def emit_cpp(self, cfg: KernelConfig) -> str:
        e: Sm80Ext = cfg.ext  # type: ignore[assignment]
        kid = self.kernel_id(cfg)
        if e.swizzle_type == "identity":
            swizzle = (
                f"cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<{e.swizzle_n}>"
            )
        else:
            swizzle = "cutlass::gemm::threadblock::GemmHorizontalThreadblockSwizzle"

        im, in_, ik = INSTRUCTION_SHAPE
        return f"""// 자동 생성 — 수정하지 말 것. backends/sm80.py:emit_cpp()
// kernel_id: {kid}
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#define KT_KERNEL_ID "{kid}"

using KtElementA = cutlass::half_t;
using KtElementB = cutlass::half_t;
using KtElementC = cutlass::half_t;
using KtElementAccumulator = float;

using KtLayoutA = cutlass::layout::{_layout(cfg, 'a')};
using KtLayoutB = cutlass::layout::{_layout(cfg, 'b')};
using KtLayoutC = cutlass::layout::{_layout(cfg, 'c')};

// ScaleType::Default + 런타임 beta=0.
//   - beta==0 이면 is_source_needed()==false 라서 C 를 읽지 않는다 (목표).
//   - serial split-K 는 set_k_partition() 이 partition>0 에서 beta=1 로 바꿔
//     부분합을 정상 누적한다. OnlyAlphaScaling 은 source 를 영원히 안 읽어서
//     serial split-K 결과가 틀리고, NoBetaScaling 은 partition 0 에서도 C 를
//     읽어 목표(‘C 를 안 읽어 더 빠르다’)에 반한다.
// EpilogueOutputOp 의 Count 는 C 의 alignment 와 같아야 한다.
using KtEpilogue = cutlass::epilogue::thread::LinearCombination<
    KtElementC,
    {cfg.align_c},
    KtElementAccumulator,
    KtElementAccumulator,
    cutlass::epilogue::thread::ScaleType::Default>;

using KtGemm = cutlass::gemm::device::GemmUniversal<
    KtElementA, KtLayoutA,
    KtElementB, KtLayoutB,
    KtElementC, KtLayoutC,
    KtElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<{cfg.tile_m}, {cfg.tile_n}, {cfg.tile_k}>,
    cutlass::gemm::GemmShape<{e.warp_m}, {e.warp_n}, {e.warp_k}>,
    cutlass::gemm::GemmShape<{im}, {in_}, {ik}>,
    KtEpilogue,
    {swizzle},
    {e.stages},
    {cfg.align_a},
    {cfg.align_b},
    cutlass::arch::OpMultiplyAdd>;

// KT_NO_IMPL 로 컴파일하면 타입만 인스턴스화한다 (smem/자원 검증용).
#ifndef KT_NO_IMPL
#include "kt_kernel_impl.h"
#endif
"""


def _layout(cfg: KernelConfig, which: str) -> str:
    # KernelConfig 는 레이아웃을 들고 있지 않다. 이번 단계의 레이아웃은
    # Problem 의 기본값(A row, B col, C row)으로 고정이며, 레이아웃을 축으로
    # 열 때 KernelConfig 에 필드를 추가하고 여기를 바꾼다.
    return {"a": "RowMajor", "b": "ColumnMajor", "c": "RowMajor"}[which]
