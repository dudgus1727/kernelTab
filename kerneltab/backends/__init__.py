"""백엔드 레지스트리.

호출부는 절대 backends.cutlass_v2 을 직접 import 하지 않는다. 반드시
get_backend(hw.arch) 를 통한다. 지원하지 않는 아키텍처에서 조용히 잘못
동작하는 대신 명확히 실패하게 만들기 위함이다.
"""

from __future__ import annotations

from kerneltab.backends.base import Backend, UnsupportedArch

__all__ = ["Backend", "UnsupportedArch", "get_backend"]


def get_backend(arch: str) -> Backend:
    # CUTLASS 2.x 경로가 성립하는 arch. sm_120 은 세대가 다르지만 같은 API
    # 계열이고 수치까지 일치하는 것을 확인했다 (decisions.md).
    if arch in ("sm_80", "sm_86", "sm_89", "sm_120"):
        from kerneltab.backends.cutlass_v2 import CutlassV2Backend

        return CutlassV2Backend()
    if arch in ("sm_90", "sm_100"):
        raise NotImplementedError(
            f"{arch} 백엔드 미구현 (향후 backends/cutlass_v3.py).\n"
            "  3.x API 는 SM90 '전용' 이 아니라 '이상용' 이다 — 파일 이름도\n"
            "  아키텍처가 아니라 API 계열로 짓는다.")
    raise UnsupportedArch(arch)
