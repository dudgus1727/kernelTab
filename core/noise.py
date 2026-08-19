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

__all__ = ["SIGMA_ABS_MS", "SIGMA_REL", "noise_floor_ms", "noise_floor",
           "coefficients", "resolvable"]

#: 절대 지터 (ms). 커널 시간과 무관한 타이머/런치 성분.
#: 앵커 중 t < 0.06 ms 조합들의 절대 표준편차 중앙값.
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
            "model": "sigma_rel(t) = sigma_abs_ms / t + sigma_rel",
            **PROVENANCE}


def noise_floor(time_ms: float) -> float:
    """이 정도 시간의 커널을 반복 측정했을 때의 **상대** 표준편차 (0~1).

    두 config 의 시간 차이가 이것보다 작으면 **구분할 수 없다.**
    """
    if not time_ms or time_ms <= 0:
        return SIGMA_REL
    return SIGMA_ABS_MS / time_ms + SIGMA_REL


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
