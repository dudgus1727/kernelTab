"""Backend Protocol.

아키텍처마다 다른 것(warp tile / stages / cluster / schedule, smem 공식,
C++ 코드 생성)은 전부 백엔드 뒤에 숨긴다. core/, build/, measure/ 는
Sm80Ext 같은 구체 타입을 절대 직접 참조하지 않고 이 Protocol 로만 통신한다.

smem_bytes 가 백엔드에 있는 이유: SM90 은 TMA 배리어와 epilogue 스테이징이
추가되어 SM80 의 `stages * tile_k * (tile_m + tile_n) * dtype_bytes` 공식이
맞지 않는다.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.types import Hardware, KernelConfig, Problem, RuntimeConfig


class UnsupportedArch(RuntimeError):
    """지원하지 않는 아키텍처. 조용히 잘못 동작하지 말고 명확히 실패한다."""


@runtime_checkable
class Backend(Protocol):
    #: 이 백엔드가 담당하는 arch 문자열들
    arch_family: tuple[str, ...]

    # -- 열거 -------------------------------------------------------------
    def enumerate_ext(self, hw: Hardware) -> list:
        """((tile_m, tile_n, tile_k), ext) 쌍의 목록을 돌려준다.

        tile 은 KernelConfig 의 공통 필드이지만 유효한 warp 분할이 tile 에
        종속되므로 백엔드가 tile 과 ext 를 함께 생성한다. 호출부(core.config)
        는 ext 의 내부를 들여다보지 않고 그대로 KernelConfig 에 실어 나른다.
        """
        ...

    def enumerate_runtime(self, p: Problem, cfg: KernelConfig) -> list[RuntimeConfig]:
        """이 (형상, 커널)에 대해 유효한 런타임 config 전부.

        Protocol 명세에 원래 없던 메서드지만, split-K 축의 값 집합이
        아키텍처마다 다르므로(SM90 은 stream-K 가 추가된다) 백엔드에 둔다.
        """
        ...

    # -- 유효성 -----------------------------------------------------------
    def is_valid_kernel(self, cfg: KernelConfig, hw: Hardware, dtype_bytes: int) -> bool:
        """컴파일/실행 가능성만 판정한다. 성능 판단은 절대 하지 않는다."""
        ...

    def explain_kernel(
        self, cfg: KernelConfig, hw: Hardware, dtype_bytes: int
    ) -> str | None:
        """유효하면 None, 아니면 최초로 걸린 제약의 이름. funnel 집계용."""
        ...

    def is_valid_runtime(self, rc: RuntimeConfig, p: Problem, cfg: KernelConfig) -> bool:
        ...

    # -- 자원/코드 --------------------------------------------------------
    def smem_bytes(self, cfg: KernelConfig, dtype_bytes: int) -> int:
        """커널이 실제로 잡는 static shared memory 바이트 수."""
        ...

    def emit_cpp(self, cfg: KernelConfig) -> str:
        """KernelConfig -> CUTLASS C++ 인스턴스화 소스."""
        ...

    def kernel_id(self, cfg: KernelConfig) -> str:
        """커널의 안정적인 문자열 식별자. 파일명과 JSONL 조인 키로 쓴다."""
        ...

    def expected_hmma(
        self,
        cfg: KernelConfig,
        rc: RuntimeConfig | None = None,
        p: Problem | None = None,
    ) -> int:
        """기대 HMMA 개수.

        rc/p 가 없으면 SASS 정적 카운트와 비교할 '워프 하나가 메인루프를
        한 번 완전히 펼쳤을 때의 HMMA 수'.
        rc/p 가 있으면 문제 전체에서 실행되는 동적 HMMA 총 개수.
        """
        ...
