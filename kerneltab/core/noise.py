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
import warnings
from dataclasses import dataclass


class NoiseCoefRequired(TypeError):
    """노이즈 계수를 안 넘겼다. 기본값으로 조용히 A6000 눈금을 쓰지 않는다."""

    def __init__(self, where: str):
        super().__init__(
            f"{where} 에 노이즈 계수(NoiseCoef)를 넘겨야 한다.\n"
            "  이 값은 **측정 조건마다 다르다** — 특히 타이머 양자(tick_ms).\n"
            "  번들이 있으면:  Bundle(...).coef   또는  noise.from_bundle(info)\n"
            "  A6000 캠페인을 분석하는 것이 확실하면:  noise.A6000_MEASURED\n"
            "  노이즈와 무관한 고정 허용치를 쓰려면:  tol=0.01 (근거를 남길 것)")


class NoiseCoefWarning(UserWarning):
    """번들이 계수를 다 들고 있지 않아 A6000 값으로 채웠다."""


__all__ = [
    "A6000_MEASURED",
    "EVENT_TICK_MS",
    "SIGMA_ABS_MS",
    "SIGMA_REL",
    "NoiseCoef",
    "NoiseCoefRequired",
    "NoiseCoefWarning",
    "coef_from_observed",
    "coefficients",
    "from_bundle",
    "tick_ms_observed",
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


@dataclass(frozen=True)
class NoiseCoef:
    """한 측정 조건의 노이즈 계수. **어디서 왔는지(`source`)를 반드시 들고 다닌다.**

    ⛔ 모듈 전역 함수로 두었더니 `core/table.py::answer_set()` 이 그것을
    호출했고, 그러면 **4090/H100 번들에서도 A6000 눈금을 조용히 쓴다.**
    `Bundle.tick_ms` 를 안 거치므로 그 경고조차 안 난다.

    그래서 계수를 **주입 필수**로 바꿨다. 기본값을 조용히 쓰는 것보다
    명시를 요구하는 쪽이 안전하다 — 특히 기본값이 다른 GPU 에서 틀릴 때.
    """

    sigma_abs_ms: float
    sigma_rel: float
    tick_ms: float
    #: 이 계수가 어디서 나왔는가. 비워 둘 수 없다.
    source: str

    def __post_init__(self):
        if not self.source:
            raise ValueError(
                "NoiseCoef.source 가 비었다. 이 계수가 어느 측정 조건에서 "
                "나왔는지 적어라 — 다른 GPU 에 쓰면 정답 집합이 틀린다.")

    # -- 성분 -------------------------------------------------------------
    def sigma(self, time_ms: float) -> float:
        """**통계적** 노이즈만. 관측된 산포와 비교할 때 쓴다."""
        if not time_ms or time_ms <= 0:
            return self.sigma_rel
        return self.sigma_abs_ms / time_ms + self.sigma_rel

    def tick_pct(self, time_ms: float) -> float:
        """타이머 눈금 하나가 몇 %인가. **분해 한계**다."""
        if not time_ms or time_ms <= 0:
            return float("inf")
        return self.tick_ms / time_ms

    def floor(self, time_ms: float) -> float:
        """두 측정값을 **구분할 수 있는 최소 상대 차이**.

            floor(t) = max( sigma(t),  tick_ms / t )

        ⚠️ 분해능 항이 없으면 짧은 커널에서 과소평가한다. 14 us 에서
        `sigma` 는 2.7 % 인데 눈금 하나는 **7.3 %** 다. 같은 눈금에 떨어진
        두 config 는 시간이 **문자 그대로 동일**하게 기록된다.
        경계는 약 1.5 ms — 그보다 짧으면 분해능이 지배한다.
        """
        if not time_ms or time_ms <= 0:
            return self.sigma_rel
        return max(self.sigma(time_ms), self.tick_pct(time_ms))

    def floor_ms(self, time_ms: float) -> float:
        return self.floor(time_ms) * time_ms

    def resolvable(self, t_a: float, t_b: float, k: float = 2.0) -> bool:
        """두 측정값의 차이가 노이즈로 설명되지 않는가. `k` 시그마."""
        if not (t_a and t_b):
            return False
        s = k * (self.floor_ms(t_a) ** 2 + self.floor_ms(t_b) ** 2) ** 0.5
        return abs(t_a - t_b) > s

    def as_dict(self) -> dict:
        return {"sigma_abs_ms": self.sigma_abs_ms, "sigma_rel": self.sigma_rel,
                "tick_ms": self.tick_ms, "source": self.source}


#: A6000 캠페인에서 잰 계수. **다른 GPU 에 쓰면 틀린다.**
#:
#: 이것을 쓰려면 **이름을 직접 써야 한다** — 그 행위가 곧 "이 조건임을
#: 알고 쓴다" 는 표시다. 기본값으로 숨어 있으면 아무도 확인하지 않는다.
A6000_MEASURED = NoiseCoef(
    sigma_abs_ms=SIGMA_ABS_MS, sigma_rel=SIGMA_REL, tick_ms=EVENT_TICK_MS,
    source="A6000 / c63710df 앵커 12조합 x 312회, 33시간")


def from_bundle(info: dict) -> NoiseCoef:
    """`BUNDLE.json` 에서 계수를 만든다. **번들이 계수를 들고 다녀야 한다.**

    `tick_ms` 가 없으면(schema_version 1) 경고하고 A6000 관측치로 채운다 —
    조용히 채우면 다른 GPU 번들에서 틀린 눈금을 쓰게 된다.
    """
    c = (info.get("noise_floor") or {}) if info else {}
    bid = (info or {}).get("bundle_id", "?")
    tick = c.get("tick_ms")
    if not tick:
        warnings.warn(
            f"번들 {bid} 에 noise_floor.tick_ms 가 없다 "
            f"(schema_version {(info or {}).get('schema_version', 1)}).\n"
            f"  A6000 관측치 {EVENT_TICK_MS} ms 로 대체한다 — "
            "**다른 GPU 라면 틀린 값이다.**\n"
            "  타이머 분해능 없이 채점하면 짧은 형상에서 노이즈를 신호로 "
            "배운다.",
            NoiseCoefWarning, stacklevel=2)
        tick = EVENT_TICK_MS
    return NoiseCoef(
        sigma_abs_ms=c.get("sigma_abs_ms", SIGMA_ABS_MS),
        sigma_rel=c.get("sigma_rel", SIGMA_REL),
        tick_ms=float(tick),
        source=(f"번들 {bid} / 계수 출처 env_hash="
                f"{str(c.get('env_hash', '?'))[:8]}"))


#: 관측 추정 눈금이 이 범위 밖이면 못 믿는다 (A6000 눈금 대비 배수).
TICK_PLAUSIBLE = (0.2, 5.0)


def coef_from_observed(values, name: str) -> NoiseCoef:
    """관측된 시간에서 타이머 양자를 추정해 `NoiseCoef` 를 만든다.

    통계항(`sigma_abs_ms`, `sigma_rel`)은 A6000 앵커 값을 쓴다 — 그쪽은
    조건 간 차이가 작고, 짧은 형상에서 지배하는 것은 눈금 항이다.

    ⚠️ **입력이 원시 반복 측정값일 때만 맞는다.** 짝수 개의 중앙값은
    보간되어 눈금 격자에서 벗어나고, 그러면 최소 간격이 부동소수 잡음까지
    내려간다 — 실제로 번들 표(980,915행)에서 `0.0 ns` 가 나왔다.
    그래서 추정치가 A6000 눈금의 `TICK_PLAUSIBLE` 배 밖이면 **버리고**
    A6000 값으로 돌아가며, 그 사실을 `source` 에 남긴다.
    """
    base = A6000_MEASURED
    est = tick_ms_observed(values)
    lo, hi = TICK_PLAUSIBLE
    if est and lo * base.tick_ms <= est <= hi * base.tick_ms:
        return NoiseCoef(base.sigma_abs_ms, base.sigma_rel, est,
                         f"{name} 관측 눈금 {est * 1e6:.1f}ns + A6000 통계항")
    why = (f"추정 {est * 1e6:.3f}ns 가 A6000 눈금의 "
           f"{est / base.tick_ms:.2g}배") if est else "추정 실패"
    return NoiseCoef(base.sigma_abs_ms, base.sigma_rel, base.tick_ms,
                     f"{name}: A6000 눈금 {base.tick_ms * 1e6:.1f}ns "
                     f"(관측 추정 버림 — {why})")


def tick_ms_observed(values) -> float | None:
    """관측된 값들에서 양자를 추정한다. 서로 다른 값의 최소 간격."""
    xs = sorted({round(float(v), 9) for v in values})
    gaps = [b - a for a, b in itertools.pairwise(xs) if b - a > 1e-9]
    return min(gaps) if gaps else None
