"""빌드된 커널 행(`kernels.jsonl`)에 대한 술어 — **한 곳에만 둔다.**

## 왜 이 모듈이 있는가

`launchable` 판정이 **여섯 곳에 따로** 있었다.

| 어디 | 결측일 때 |
|---|---|
| `scripts/rehearse.py::launchable()` | `True` |
| `scripts/export.py` (`launchable` 컬럼) | `None` |
| `scripts/validate_table.py` | 통과시킴 |
| `scripts/check_correctness.py` | 인라인 |
| `scripts/recheck_stability.py` | 인라인 |
| `scripts/verify_clock_lock.py` | 인라인 |

같은 식(`regs * threads <= regs_per_sm`)인데 **결측 처리가 서로 달랐다.**
`docs/decisions.md` 13 의 일곱 번째와 같은 구조다 — 같은 클래스가 여러
경로에 따로 존재하면, 한 곳을 고쳐도 나머지가 남는다.
"""

from __future__ import annotations

__all__ = ["launchable", "regs_total_per_block", "threads_per_block"]


def threads_per_block(row: dict) -> int | None:
    """`kernels.jsonl` 행의 블록당 스레드 수. 없으면 타일/워프에서 계산한다."""
    t = row.get("threads")
    if t:
        return int(t)
    tile, ext = row.get("tile"), row.get("ext")
    if not tile or not ext:
        return None
    try:
        return (tile["m"] // ext["warp_m"]) * (tile["n"] // ext["warp_n"]) \
            * (tile["k"] // ext["warp_k"]) * 32
    except (KeyError, TypeError, ZeroDivisionError):
        return None


def regs_total_per_block(row: dict) -> int | None:
    r, t = row.get("regs_per_thread"), threads_per_block(row)
    return r * t if r and t else None


def launchable(row: dict, regs_per_sm: int) -> bool:
    """런치 가능한가.

    `cutlass::Kernel2` 에는 `__launch_bounds__` 가 없어서 ptxas 가 스레드 수에
    맞춰 레지스터를 제한하지 않는다. 그 결과 `regs_per_thread * threads` 가
    블록당 레지스터 파일을 넘는 커널이 만들어지고, **빌드는 성공하는데 런치가
    항상 실패한다** (CUTLASS status 7). 실제로 런치해 보고 에러를 받기보다
    미리 판정하는 편이 명확하고 빠르다.

    결측이면 `True` 다 — "모른다" 를 "못 쓴다" 로 바꾸면 멀쩡한 커널이 조용히
    빠진다. 판정하려면 `regs_total_per_block()` 이 `None` 인지 먼저 보라.
    """
    tot = regs_total_per_block(row)
    return True if tot is None else tot <= regs_per_sm
