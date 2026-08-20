"""백엔드 레지스트리.

호출부는 절대 backends.sm80 을 직접 import 하지 않는다. 반드시
get_backend(hw.arch) 를 통한다. 지원하지 않는 아키텍처에서 조용히 잘못
동작하는 대신 명확히 실패하게 만들기 위함이다.
"""

from __future__ import annotations

from kerneltab.backends.base import Backend, UnsupportedArch

__all__ = ["Backend", "UnsupportedArch", "get_backend"]


def get_backend(arch: str) -> Backend:
    if arch in ("sm_80", "sm_86", "sm_89"):
        from kerneltab.backends.sm80 import Sm80Backend

        return Sm80Backend()
    if arch in ("sm_90", "sm_100"):
        raise NotImplementedError(f"{arch} 백엔드 미구현 (향후 backends/sm90.py)")
    raise UnsupportedArch(arch)
