"""kerneltab — CUTLASS GEMM (형상 x config) -> 성능 표 측정 하네스.

`core` / `build` / `measure` 를 **최상위 이름으로 두지 않는다.** `build` 는
PyPI 에 실제로 존재하는 패키지(`python -m build`)이고 `core`/`measure` 도
흔한 이름이라, 소비 프로젝트와 충돌하면 `ImportError` 가 아니라 **다른
모듈이 조용히 import 된다.** 가장 나쁜 실패 방식이다 (`docs/packaging.md`).

    from kerneltab.core.features import waves
    from kerneltab.backends import get_backend
"""

__version__ = "0.2.0"
