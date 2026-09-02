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


#: ★ `(N,K) = (4096,4096)` 열에서 이 M 미만은 제외한다 (2026-09-02, H100).
#:
#: 이 열은 B 가 4096x4096 = 33.5 MB 로 네 열 중 가장 작다. 작은 M 에서는
#: 그 33.5 MB 를 읽는 시간이 지배하는데, H100 NVL(4.0 TB/s)에서는 **8.4 us**
#: 로 `below_launch_overhead` 문턱(10.75 us = 3 x 3.58 us)보다 짧다.
#:
#: 문턱의 78~82 % 는 **가장 나쁜 구간**이다 — 느린 config 는 살아남고 빠른
#: config 만 잘려 정답 쪽만 검열된다. 통째로 결측인 편이 차라리 낫다.
#:
#: 나머지 세 열은 B 가 더 커서 M=1 에서도 문턱의 209~233 % 다. 즉 디코드
#: 구간(M=1~128) 자체는 **세 열에서 그대로 보존된다.**
#:
#: ⚠️ 층 A 는 GPU 무관 고정이므로 이 제외는 모든 GPU 에 적용된다 (층 B/E 를
#:    5090 때 옮긴 것과 같은 처리다). A6000/5090 번들은 소급 변경되지 않고
#:    `common_shapes_only` 가 차이를 걸러낸다.
_M_FLOOR_4096x4096 = 256


def shapes_layer_a() -> list[Problem]:
    """실제 워크로드. (N,K)는 Llama 7B 계열. GPU 무관 고정."""
    return [Problem(m, n, k)
            for (n, k) in _LLAMA_NK
            for m in _LLAMA_M
            if not ((n, k) == (4096, 4096) and m < _M_FLOOR_4096x4096)]


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

    ## M 을 한 칸 더 올렸다 (2026-09-02, H100 NVL. 이전엔 4096 / 2048)

    같은 일이 H100 에서 다시 일어났다. 문턱이 **10.752 us** (5090 12.288)
    로 더 낮은데도 두 칸이 아래로 내려갔다:

        4096x4096x128  -> 8.86 us (문턱의 82 %, memory)
        2048x4096x256  -> 6.88 us (문턱의 64 %, compute)

    M 을 8192 / 4096 으로 올려 최소 17.5 us(163 %)로 만들었다. **M 대비는
    2 배로 유지**되고, 이 층의 목적인 K 축은 그대로다.

    ⚠️ A6000 캠페인(`c63710df` / `828baa64`)은 **옛 M(1024 / 128)로 쟀다.**
    그 그리드는 번들에 박혀 있으므로 소급 변경되지 않는다. 전이 실험에서는
    `common_shapes_only` 가 이 차이를 걸러낸다.
    """
    out = [Problem(8192, 4096, k) for k in (128, 256, 512, 1024, 2048, 4096, 8192, 16384)]
    out += [Problem(4096, 4096, k) for k in (256, 1024, 4096, 16384)]
    return out


def shapes_layer_c(hw: Hardware) -> list[Problem]:
    """어려운 wave 구간. 목표 waves 에서 M 을 역산한다.

    M 을 상수로 박으면 GPU 마다 sm_count 가 달라 같은 M 이 전혀 다른 물리적
    상황(예: 0.7 wave vs 1.3 wave)을 의미하게 되어 GPU 간 비교가 깨진다.
    waves 를 고정하고 M 을 역산해야 전이 실험이 성립한다.

    ⚠️ **타일이 하나(m_tiles=1)로 떨어지는 목표는 버린다** — 목표 waves 를
    맞추지 못하고, 그 형상이 런치 오버헤드 문턱 아래로 내려간다 (아래 주석).
    """
    TARGET_WAVES = [0.3, 0.5, 0.76, 1.2, 1.5, 2.3, 3.05, 4.5, 6.7]
    N = K = 4096
    n_tiles = ceil(N / 128)

    ms: list[int] = []
    for w in TARGET_WAVES:
        m_tiles = round(w * hw.sm_count / n_tiles)
        # ⛔ 타일이 하나뿐이면 목표 waves 를 애초에 못 맞춘다 — 실제 waves 는
        #    n_tiles/sm_count 로 고정된다 (H100: 목표 0.3 -> 실제 0.24,
        #    A6000: 목표 0.3·0.5 -> 둘 다 실제 0.38). 게다가 그 형상
        #    (M=128, N=K=4096)이 H100 에서 `below_launch_overhead` 문턱의
        #    82 % 로 내려간다 — 빠른 config 만 잘리는 가장 나쁜 구간이다.
        #    (2026-09-02. 층 A 의 같은 형상 제외와 한 쌍이다)
        if m_tiles < 2:
            continue
        ms.append(m_tiles * 128)
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
