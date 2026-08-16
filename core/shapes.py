"""형상 그리드 (층별).

각 층은 서로 다른 것을 배우기 위해 존재한다. 층을 합치면 어떤 축이
무엇을 설명하는지 사후에 분리할 수 없으므로 생성 함수를 분리해 둔다.
"""

from __future__ import annotations

from math import ceil

from core.types import Hardware, Problem

__all__ = [
    "shapes_layer_a",
    "shapes_layer_b",
    "shapes_layer_c",
    "shapes_layer_d",
    "shapes_layer_e",
    "all_layers",
    "all_shapes",
]

# Llama 7B 계열 선형층의 (N, K). GPU 와 무관하게 고정이다.
_LLAMA_NK = [
    (4096, 4096),    # o_proj
    (12288, 4096),   # qkv (fused)
    (11008, 4096),   # gate/up
    (4096, 11008),   # down
]
_LLAMA_M = [1, 8, 32, 128, 256, 512, 1024, 2048, 4096, 8192]


def shapes_layer_a() -> list[Problem]:
    """실제 워크로드. (N,K)는 Llama 7B 계열. GPU 무관 고정."""
    return [Problem(m, n, k) for (n, k) in _LLAMA_NK for m in _LLAMA_M]


def shapes_layer_b() -> list[Problem]:
    """K 변화. mainloop 반복 수와 split-K 상한을 탐색.

    이 층이 없으면 stages 와 warp_k 에 대해 아무것도 배울 수 없다.
    K 가 고정이면 mainloop 깊이가 고정이라 파이프라인 단수의 효과가
    형상 축과 완전히 교락(confound)된다.
    """
    out = [Problem(1024, 4096, k) for k in (128, 256, 512, 1024, 2048, 4096, 8192, 16384)]
    out += [Problem(128, 4096, k) for k in (256, 1024, 4096, 16384)]
    return out


def shapes_layer_c(hw: Hardware) -> list[Problem]:
    """어려운 wave 구간. 목표 waves 에서 M 을 역산한다.

    M 을 상수로 박으면 GPU 마다 sm_count 가 달라 같은 M 이 전혀 다른 물리적
    상황(예: 0.7 wave vs 1.3 wave)을 의미하게 되어 GPU 간 비교가 깨진다.
    waves 를 고정하고 M 을 역산해야 전이 실험이 성립한다.
    """
    TARGET_WAVES = [0.3, 0.5, 0.76, 1.2, 1.5, 2.3, 3.05, 4.5, 6.7]
    N = K = 4096
    n_tiles = ceil(N / 128)

    ms: list[int] = []
    for w in TARGET_WAVES:
        m_tiles = round(w * hw.sm_count / n_tiles)
        m = max(1, m_tiles) * 128
        ms.append(m)
    # 2의 거듭제곱이 아닌 M — 타일 경계에 정확히 떨어지지 않는 상황
    ms += [1000, 1500, 3000]

    seen: set[int] = set()
    out: list[Problem] = []
    for m in ms:
        if m in seen:
            continue
        seen.add(m)
        out.append(Problem(m, N, K))
    return out


def shapes_layer_d() -> list[Problem]:
    """정렬 엣지케이스. alignment 가 8 미만이 되는 형상.

    alignments_for() 가 실제로 동작하는지, 그리고 낮은 alignment 커널이
    빌드/실행되는지를 검증한다.
    """
    return [
        Problem(1024, 4096, 4100),   # align_a/b = 4
        Problem(1024, 4096, 4098),   # align_a/b = 2
        Problem(1024, 4096, 4097),   # align_a/b = 1
        Problem(1024, 4100, 4096),   # align_c = 4
        Problem(1024, 4098, 4096),   # align_c = 2
    ]


def shapes_layer_e() -> list[Problem]:
    """정방형 대조군. 문헌 벤치마크와 직접 비교 가능한 기준선."""
    return [Problem(n, n, n) for n in (512, 1024, 2048, 4096, 8192)]


def all_layers(hw: Hardware) -> dict[str, list[Problem]]:
    return {
        "a_workload": shapes_layer_a(),
        "b_kvary": shapes_layer_b(),
        "c_waves": shapes_layer_c(hw),
        "d_alignment": shapes_layer_d(),
        "e_square": shapes_layer_e(),
    }


def all_shapes(hw: Hardware) -> list[Problem]:
    """층 전체를 합치고 중복 제거. 층 간 중복이 존재한다(의도된 것)."""
    seen: set[Problem] = set()
    out: list[Problem] = []
    for probs in all_layers(hw).values():
        for p in probs:
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
    return out
