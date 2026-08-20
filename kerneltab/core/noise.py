"""측정 노이즈 바닥 — **형상마다 다르다** (고정 1% 를 쓰면 안 된다).

## 왜 필요한가

33시간 캠페인의 앵커(조합 12개 x 312회 재측정)에서:

| 커널 시간 | 상대 표준편차 | 33시간 변동폭 |
|---|---:|---:|
| > 1 ms | **0.044%** | 0.11 ~ 1.41% |
| < 0.1 ms | **1.541%** | 1.82 ~ 7.14% |

**35 배 차이**다. 단일 숫자(예전에 쓰던 "0.05%")는 이걸 뭉갠다. 뭉갠 채로
정답 집합을 1 % 로 자르면, 15 us 커널에서는 **노이즈를 신호로 학습**하게
된다 — 그 크기에서 1 % 는 재현되지 않는 차이다.

이미 세 방향에서 같은 결론이 나왔다:

* 난이도: M=1/8 형상이 1.11~1.15 (선택이 거의 무의미)
* 순위 안정성: `1x12288x4096` 은 최적 대비 5 % 이내 config 가 **2,501개**
* 측정 재현성: 이 모듈

세 번째가 앞 둘의 **원인 일부**다. 5 % 이내 2,501개가 진짜 성능 유사성인지
측정 노이즈인지 구분하려면 이 값이 필요하다.

## 모델

    sigma_rel(t) = SIGMA_ABS_MS / t + SIGMA_REL

절대 성분(타이머/런치 지터)과 상대 성분(클럭/열 미세 변동)의 합이다.
앵커의 절대 sigma 가 커널 시간과 거의 무관하게 0.26~0.47 us 로 나오는 것이
절대 성분의 증거다.

계수는 **강건 추정**이다. 최소제곱은 이상치 하나(1.67 ms 조합의 5.6 us)에
끌려 작은 커널의 노이즈를 40 % 과대평가했다. 작은 t 구간에서 절대 성분의
중앙값을, 큰 t 구간에서 상대 성분의 중앙값을 따로 잡았다.

**약간 보수적인 쪽으로 치우쳐 있다.** 허용치는 노이즈를 과소평가하는 것보다
과대평가하는 편이 안전하다 — 과소평가하면 노이즈를 신호로 배운다.
"""

from __future__ import annotations

import itertools

__all__ = [
    "EVENT_TICK_MS",
    "SIGMA_ABS_MS",
    "SIGMA_REL",
    "coefficients",
    "noise_floor",
    "noise_floor_ms",
    "resolvable",
    "sigma_rel",
    "tick_ms_observed",
    "tick_pct",
]

#: 절대 지터 (ms). 커널 시간과 무관한 타이머/런치 성분.
#: 앵커 중 t < 0.06 ms 조합들의 절대 표준편차 중앙값.
#: CUDA 이벤트 타이머의 관측 양자 (ms). A6000 에서 서로 다른 측정값의
#: 최소 간격이 정확히 이 값이었다 (0.014336 - 0.013312).
#:
#: ⚠️ **이것보다 작은 차이는 분해할 수 없다.** 14 us 커널에서 한 눈금은
#: **7.3 %** 다. 짧은 앵커의 중앙값 차이를 % 로만 보면 눈금 하나가 큰
#: 이상 신호처럼 보인다 — 실제로 슬라이스 start/end 차이 -6.49 % 가
#: **0.84 눈금**이었다.
#:
#: 다른 GPU 에서는 다를 수 있다. `tick_ms_observed()` 로 확인하라.
EVENT_TICK_MS = 0.001024

SIGMA_ABS_MS = 0.000374

#: 상대 성분. 앵커 중 t > 1 ms 조합들의 상대 표준편차 중앙값.
SIGMA_REL = 0.00044

#: 이 값들이 나온 조건. 다른 GPU 에서는 반드시 다시 재야 한다.
PROVENANCE = {
    "gpu": "NVIDIA RTX A6000",
    "env_hash": "c63710df",
    "source": "results/anchors.jsonl — 앵커 12조합 x 312회, 33시간",
    "method": "강건 추정 (작은 t 에서 절대, 큰 t 에서 상대)",
}


def coefficients() -> dict:
    """`BUNDLE.json` 에 실어 소비 쪽이 재계산 없이 쓰게 한다."""
    return {"sigma_abs_ms": SIGMA_ABS_MS, "sigma_rel": SIGMA_REL,
            "tick_ms": EVENT_TICK_MS,
            "model": "noise_floor(t) = max(sigma_abs_ms/t + sigma_rel, "
                     "tick_ms/t)",
            "model_note": (
                "tick_ms 는 CUDA 이벤트 타이머의 양자다. 이보다 작은 차이는 "
                "분해할 수 없다 — 같은 눈금에 떨어진 두 config 는 시간이 "
                "문자 그대로 동일하게 기록된다. 14us 에서 통계 노이즈는 "
                "2.7% 인데 눈금은 7.3% 라 분해능이 지배한다 (경계 ~1.5ms)."),
            **PROVENANCE}


def tick_pct(time_ms: float) -> float:
    """이 시간에서 타이머 눈금 하나가 몇 %인가.

    **분해 한계**다. 이보다 작은 차이는 측정으로 구분할 수 없다.
    """
    return EVENT_TICK_MS / time_ms if time_ms > 0 else float("inf")


def tick_ms_observed(values) -> float | None:
    """관측된 값들에서 양자를 추정한다. 서로 다른 값의 최소 간격."""
    xs = sorted({round(float(v), 9) for v in values})
    gaps = [b - a for a, b in itertools.pairwise(xs) if b - a > 1e-9]
    return min(gaps) if gaps else None


def sigma_rel(time_ms: float) -> float:
    """**통계적** 노이즈 모델만. 앵커의 반복 측정에서 적합했다.

    관측된 산포와 비교할 때는 이것을 쓴다 (`core/anchors.py` 의 `모델` 열).
    "두 값을 구분할 수 있는가" 에는 `noise_floor()` 를 써라 — 거기에는
    타이머 분해능이 더 들어간다.
    """
    if not time_ms or time_ms <= 0:
        return SIGMA_REL
    return SIGMA_ABS_MS / time_ms + SIGMA_REL


def noise_floor(time_ms: float) -> float:
    """두 측정값을 **구분할 수 있는 최소 상대 차이** (0~1).

    통계적 노이즈와 **타이머 분해능** 중 큰 쪽이다.

        noise_floor(t) = max( sigma_rel(t),  tick_pct(t) )

    ⚠️ **분해능 항이 없으면 짧은 커널에서 모델이 과소평가한다.** 14 us 에서
    `sigma_rel` 은 2.7 % 인데 눈금 하나는 **7.3 %** 다. 같은 눈금에 떨어진
    두 config 는 시간이 **문자 그대로 동일**하게 기록되므로, 2.7 % 기준으로
    "구분된다" 고 판정하면 없는 순위를 만든다.

    | t | `sigma_rel` | `tick_pct` | 지배하는 쪽 |
    |---|---|---|---|
    | 14 us | 2.7 % | **7.3 %** | 분해능 |
    | 56 us | 0.71 % | **1.8 %** | 분해능 |
    | 0.5 ms | 0.12 % | **0.20 %** | 분해능 |
    | 1.5 ms | 0.069 % | 0.068 % | 비슷 |
    | 4 ms | **0.053 %** | 0.026 % | 통계 |

    경계는 약 1.5 ms 다. 그보다 짧으면 **분해능이 지배한다.**
    """
    if not time_ms or time_ms <= 0:
        # 분해능 항은 t 로 나누므로 의미가 없다. 옛 계약대로 상대 성분만.
        return SIGMA_REL
    return max(sigma_rel(time_ms), tick_pct(time_ms))


def noise_floor_ms(time_ms: float) -> float:
    """절대 단위 (ms)."""
    return noise_floor(time_ms) * time_ms


def resolvable(t_a: float, t_b: float, k: float = 2.0) -> bool:
    """두 측정값의 차이가 노이즈로 설명되지 않는가.

    `k` 는 몇 시그마로 볼지. 기본 2 시그마.
    """
    if not (t_a and t_b):
        return False
    s = k * (noise_floor_ms(t_a) ** 2 + noise_floor_ms(t_b) ** 2) ** 0.5
    return abs(t_a - t_b) > s
