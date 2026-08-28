"""형상 그리드 (층별).

각 층은 서로 다른 것을 배우기 위해 존재한다. 층을 합치면 어떤 축이
무엇을 설명하는지 사후에 분리할 수 없으므로 생성 함수를 분리해 둔다.
"""

from __future__ import annotations

from math import ceil

from kerneltab.core.types import Hardware, Problem

__all__ = [
    "all_layers",
    "all_shapes",
    "shapes_layer_a",
    "shapes_layer_b",
    "shapes_layer_c",
    "shapes_layer_d",
    "shapes_layer_e",
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

    ## M 이 4096 / 2048 인 이유 (2026-08-28 변경, 이전엔 1024 / 128)

    RTX 5090 에서 **작은 K 끝이 런치 오버헤드 아래로 내려갔다.**
    `below_launch_overhead` 문턱은 `3 x launch_bracketed_grid_ms` 이고
    5090 에서 12.29 us 다. 옛 M 으로는 네 칸이 그 아래였다:

        M=1024: K=128 -> 5.5 us,  K=256 -> 9.2 us
        M= 128: K=256 -> 1.8 us,  K=1024 -> 5.5 us   (SOL 하한 기준)

    M 을 올려 전부 문턱 위로 올렸다 (최소 18.3 us). **이 층의 목적은
    K 축이고 M 은 부수적**이므로 목적이 보존된다.

    두 행을 둔 이유였던 M 대비는 8 배 -> 2 배로 줄지만, 그 역할은
    **층 A 가 M 을 17 개 스윕하며 이미 담당한다.** 그리고 A6000 캠페인에서
    난이도 상위 5 개 중 4 개가 이 층이었고 난이도와 log K 의 상관이
    -0.71 이었다 — **어려움의 원천은 M 대비가 아니라 작은 K 다.**
    K 네 칸을 잃는 것보다 M 대비를 줄이는 쪽이 싸다.

    ⚠️ A6000 캠페인(`c63710df` / `828baa64`)은 **옛 M(1024 / 128)로 쟀다.**
    그 그리드는 번들에 박혀 있으므로 소급 변경되지 않는다. 전이 실험에서는
    `common_shapes_only` 가 이 차이를 걸러낸다.
    """
    out = [Problem(4096, 4096, k) for k in (128, 256, 512, 1024, 2048, 4096, 8192, 16384)]
    out += [Problem(2048, 4096, k) for k in (256, 1024, 4096, 16384)]
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
    """정방형 대조군. 문헌 벤치마크와 직접 비교 가능한 기준선.

    ## 사다리를 2048~16384 로 민 이유 (2026-08-28, 이전엔 512~8192)

    RTX 5090 의 `below_launch_overhead` 문턱(12.29 us) 아래로 저(低)단
    두 칸이 내려갔다 — `512^3` 은 SOL 1.15 us (문턱의 9 %),
    `1024^3` 은 9.17 us (75 %).

    **75 % 짜리를 남기는 것이 9 % 짜리보다 나쁘다.** 9 % 는 어떤 config
    로도 문턱 위로 못 올라가 통째로 결측이 되지만, 75 % 는 **느린 config 는
    살아남고 빠른 config 만 잘린다.** 정답 쪽만 검열되므로, 남은 값으로
    순위를 매기면 **틀린 답이 나오는데 그것이 결측으로 보이지 않는다.**

    칸이 5 -> 4 로 줄지만 이 층은 대조군이라 감수할 만하다. `16384^3` 의
    버퍼는 A/B/C/D 합쳐 2.1 GB 로 32 GB 에 여유가 있다.
    """
    return [Problem(n, n, n) for n in (2048, 4096, 8192, 16384)]


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
