"""파생 지표 — 측정 없이 계산 가능한 물리 피처.

이 함수들은 결과 JSONL 에 저장하지 않는다. 계산이 공짜이고, 계산식에
버그가 발견됐을 때 원본 측정을 다시 하지 않고 재계산하면 되기 때문이다.
scripts/export.py 가 분석 시점에 계산해 parquet 에 넣는다.

★ 이 모듈의 함수는 cfg.ext 를 절대 참조하지 않는다.
   공통 필드(tile_m/n/k)와 hw, rc 만으로 계산되어야 SM80 에서 학습한
   피처를 SM90 데이터에 그대로 적용하는 전이 실험이 성립한다.
   smem 이 필요하면 backend.smem_bytes() 에 위임한다.
"""

from __future__ import annotations

from math import ceil

from core.config import dtype_bytes
from core.types import Hardware, KernelConfig, Problem, RuntimeConfig

__all__ = [
    "grid_tiles",
    "waves",
    "tail_waste",
    "mainloop_iters",
    "arith_intensity",
    "ridge_point",
    "is_memory_bound",
    "flops",
    "bytes_moved",
    "tail_m_frac",
    "tail_n_frac",
]


def flops(p: Problem) -> float:
    return 2.0 * p.M * p.N * p.K


def bytes_moved(p: Problem) -> float:
    """이상적인(재사용 완벽) 최소 트래픽.

    C 는 beta=0 이라 읽지 않고 쓰기만 한다. 출력 원소 크기는 입력과 같은
    dtype 을 쓴다고 가정한다 (이 하네스는 fp16 in / fp16 out).
    """
    nb = dtype_bytes(p.dtype)
    return float((p.M * p.K + p.K * p.N + p.M * p.N) * nb)


def arith_intensity(p: Problem) -> float:
    """FLOP / byte. 형상만으로 결정되며 config 와 무관하다."""
    return flops(p) / bytes_moved(p)


def ridge_point(hw: Hardware) -> float:
    """roofline 의 무릎 지점 (FLOP/byte)."""
    return (hw.peak_tflops_f16 * 1e12) / (hw.bandwidth_gbps * 1e9)


def is_memory_bound(p: Problem, hw: Hardware) -> bool:
    return arith_intensity(p) < ridge_point(hw)


def grid_tiles(p: Problem, cfg: KernelConfig, rc: RuntimeConfig) -> int:
    """런치되는 threadblock 총 개수."""
    return ceil(p.M / cfg.tile_m) * ceil(p.N / cfg.tile_n) * rc.split_k


def waves(
    p: Problem,
    hw: Hardware,
    cfg: KernelConfig,
    rc: RuntimeConfig,
    blocks_per_sm: int = 1,
) -> float:
    """GPU 를 몇 번 '가득 채워야' 하는가.

    blocks_per_sm 기본값 1 은 core/shapes.py 의 층 C 가 M 을 역산할 때 쓰는
    정의와 같다. 빌드 후에는 kernels.jsonl 의 max_blocks_per_sm 을 넘겨
    occupancy 를 반영한 waves 를 얻을 수 있다 (export.py 가 그렇게 한다).
    """
    return grid_tiles(p, cfg, rc) / (hw.sm_count * max(1, blocks_per_sm))


def tail_waste(
    p: Problem,
    hw: Hardware,
    cfg: KernelConfig,
    rc: RuntimeConfig,
    blocks_per_sm: int = 1,
) -> float:
    """마지막 wave 에서 노는 SM 슬롯의 비율 (0 이면 완전 정렬).

    0.9 wave 든 1.1 wave 든 GPU 는 wave 단위로 시간을 쓰므로, 올림한
    wave 수 대비 낭비분을 본다.
    """
    w = waves(p, hw, cfg, rc, blocks_per_sm)
    if w <= 0:
        return 0.0
    full = ceil(w)
    return (full - w) / full


def mainloop_iters(p: Problem, cfg: KernelConfig, rc: RuntimeConfig) -> int:
    """threadblock 하나가 도는 K 루프 횟수."""
    return ceil(p.K / (cfg.tile_k * rc.split_k))


def tail_m_frac(p: Problem, cfg: KernelConfig) -> float:
    """M 방향 마지막 타일에서 실제로 쓰이는 행의 비율."""
    r = p.M % cfg.tile_m
    return 1.0 if r == 0 else r / cfg.tile_m


def tail_n_frac(p: Problem, cfg: KernelConfig) -> float:
    r = p.N % cfg.tile_n
    return 1.0 if r == 0 else r / cfg.tile_n
