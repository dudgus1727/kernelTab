"""alignment 계산과 아키텍처 무관 열거 로직.

alignment 는 탐색 축이 아니다. 형상과 레이아웃에서 유도되는 값이며,
낮은 alignment 는 트레이드오프가 아니라 단순 열등이므로(더 좁은 벡터 접근)
형상이 허용하는 최댓값을 쓴다. 다만 CUTLASS 템플릿 인자라 커널 빌드에는
영향을 주므로 KernelConfig 에 포함된다.

이 모듈은 backend.ext 의 내부 구조를 절대 들여다보지 않는다.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from backends.base import Backend
from core.types import KernelConfig, Problem, RuntimeConfig

__all__ = [
    "DTYPE_BYTES",
    "dtype_bytes",
    "alignments_for",
    "alignment_combos",
    "enumerate_kernels",
    "enumerate_kernels_with_funnel",
    "enumerate_runtimes",
]

DTYPE_BYTES = {"f16": 2, "bf16": 2, "f32": 4, "tf32": 4, "f8": 1, "s8": 1}


def dtype_bytes(dtype: str) -> int:
    try:
        return DTYPE_BYTES[dtype]
    except KeyError:
        raise ValueError(f"알 수 없는 dtype: {dtype}") from None


def alignments_for(p: Problem) -> tuple[int, int, int]:
    """(align_a, align_b, align_c) — 원소 개수 단위.

    레이아웃에 따라 alignment 가 걸리는 차원이 다르다. 벡터화는 항상
    '연속(contiguous) 차원'에 걸리므로:
      A row-major   -> 연속 차원 K
      A column-major-> 연속 차원 M
      B column-major-> 연속 차원 K
      B row-major   -> 연속 차원 N
      C row-major   -> 연속 차원 N
      C column-major-> 연속 차원 M
    """

    def maxal(n: int) -> int:
        for a in (8, 4, 2, 1):
            if n % a == 0:
                return a
        return 1

    al_a = maxal(p.K if p.layout_a == "row" else p.M)
    al_b = maxal(p.K if p.layout_b == "col" else p.N)
    al_c = maxal(p.N if p.layout_c == "row" else p.M)
    return al_a, al_b, al_c


def alignment_combos(problems: Iterable[Problem]) -> list[tuple[int, int, int]]:
    """형상 그리드에 실제로 등장하는 alignment 조합만 추린다.

    커널 빌드 범위를 이 집합으로 제한한다. (8,4,2,1)^3 = 64 조합을 전부
    빌드하면 대부분 아무 형상에도 쓰이지 않는다.
    """
    return sorted({alignments_for(p) for p in problems})


def enumerate_kernels(
    hw,
    backend: Backend,
    align_combos: Iterable[tuple[int, int, int]],
    dtype: str = "f16",
) -> list[KernelConfig]:
    kernels, _ = enumerate_kernels_with_funnel(hw, backend, align_combos, dtype)
    return kernels


def enumerate_kernels_with_funnel(
    hw,
    backend: Backend,
    align_combos: Iterable[tuple[int, int, int]],
    dtype: str = "f16",
) -> tuple[list[KernelConfig], Counter]:
    """유효 커널 목록 + 제약별 탈락 집계.

    funnel 은 alignment 와 무관하므로 첫 alignment 조합에 대해서만 센다.
    """
    nb = dtype_bytes(dtype)
    combos = list(align_combos)
    ext_list = backend.enumerate_ext(hw)

    kernels: list[KernelConfig] = []
    funnel: Counter = Counter()
    for idx, (al_a, al_b, al_c) in enumerate(combos):
        for (tm, tn, tk), ext in ext_list:
            cfg = KernelConfig(
                tile_m=tm, tile_n=tn, tile_k=tk,
                align_a=al_a, align_b=al_b, align_c=al_c,
                arch=hw.arch, ext=ext,
            )
            reason = backend.explain_kernel(cfg, hw, nb)
            if idx == 0:
                funnel[reason or "VALID"] += 1
            if reason is None:
                kernels.append(cfg)
    return kernels, funnel


def enumerate_runtimes(
    backend: Backend, p: Problem, cfg: KernelConfig
) -> list[RuntimeConfig]:
    return backend.enumerate_runtime(p, cfg)
