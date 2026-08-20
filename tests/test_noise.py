"""노이즈 바닥이 **형상마다 다르게** 동작하는가.

고정 1 % 로 되돌아가면 여기서 걸린다. 33시간 앵커에서 크기별 재현성이
35 배 차이났고, 뭉개면 작은 형상에서 노이즈를 신호로 배운다.
"""
from __future__ import annotations

import itertools

import pytest

from kerneltab.core import noise
from kerneltab.core.noise import (
    SIGMA_ABS_MS,
    SIGMA_REL,
    coefficients,
    noise_floor,
    noise_floor_ms,
    resolvable,
)


class TestModel:
    def test_small_kernels_are_much_noisier(self):
        """15 us 커널이 3 ms 커널보다 훨씬 시끄럽다 — 이게 요점이다."""
        assert noise_floor(0.015) > 20 * noise_floor(3.0)

    def test_matches_measured_anchors(self):
        """앵커 실측과 자릿수가 맞는가 (33시간 x 312회).

        ⚠️ **통계 모델(`sigma_rel`)로 비교한다.** `noise_floor` 에는
        타이머 분해능이 더 들어가 있어서 짧은 커널에서 훨씬 크다 — 관측된
        산포와 비교할 값이 아니다.
        """
        from kerneltab.core.noise import sigma_rel
        # 0.0143 ms -> 실측 2.93%,  2.9164 ms -> 실측 0.018%
        assert 0.015 < sigma_rel(0.0143) < 0.045
        assert 0.0003 < sigma_rel(2.9164) < 0.0015

    def test_짧은_커널에서는_분해능이_지배한다(self):
        """★ 통계 모델만 쓰면 **과소평가**한다.

        14 us 에서 sigma_rel 은 2.7 % 인데 눈금 하나는 7.3 % 다. 같은 눈금에
        떨어진 두 config 는 시간이 **문자 그대로 동일**하게 기록되므로
        2.7 % 기준으로 "구분된다" 고 하면 없는 순위를 만든다.
        """
        from kerneltab.core.noise import sigma_rel, tick_pct
        t = 0.0143
        assert tick_pct(t) > sigma_rel(t)
        assert noise_floor(t) == pytest.approx(tick_pct(t))

    def test_긴_커널에서는_통계가_지배한다(self):
        from kerneltab.core.noise import sigma_rel, tick_pct
        t = 4.0
        assert sigma_rel(t) > tick_pct(t)
        assert noise_floor(t) == pytest.approx(sigma_rel(t))

    def test_monotone_decreasing(self):
        ts = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 20.0]
        v = [noise_floor(t) for t in ts]
        assert all(a > b for a, b in itertools.pairwise(v))

    def test_floor_never_zero(self):
        """아무리 긴 커널도 상대 성분이 남는다."""
        assert noise_floor(1e6) >= SIGMA_REL

    def test_absolute_form(self):
        t = 0.02
        assert noise_floor_ms(t) == pytest.approx(noise_floor(t) * t)

    @pytest.mark.parametrize("bad", [0, None, -1.0])
    def test_degenerate_time(self, bad):
        assert noise_floor(bad) == SIGMA_REL


class TestResolvable:
    def test_one_percent_not_resolvable_on_tiny_kernels(self):
        """**이 테스트가 이 모듈의 존재 이유다.**

        15 us 커널에서 1 % 차이는 재현되지 않는다. 고정 1 % 허용치를
        쓰면 그 형상에서 노이즈를 정답/오답으로 가르게 된다.
        """
        t = 0.0143
        assert not resolvable(t, t * 1.01)

    def test_one_percent_resolvable_on_ms_kernels(self):
        t = 2.9164
        assert resolvable(t, t * 1.01)

    def test_crossover_is_around_half_ms(self):
        """1 % 를 구분할 수 있게 되는 지점. 문서에 인용하는 값이다."""
        assert not resolvable(0.1, 0.1 * 1.01)
        assert resolvable(0.5, 0.5 * 1.01)

    def test_identical_times_not_resolvable(self):
        assert not resolvable(1.0, 1.0)


class TestCoefficientsTravel:
    def test_coefficients_carry_provenance(self):
        """계수만 옮기면 안 된다 — 어느 GPU 에서 잰 것인지 같이 가야 한다."""
        c = coefficients()
        assert c["sigma_abs_ms"] == SIGMA_ABS_MS
        assert c["sigma_rel"] == SIGMA_REL
        for k in ("gpu", "env_hash", "source", "model"):
            assert c.get(k), f"{k} 가 계수와 함께 전달되지 않는다"


class TestTimerTick:
    """★ 타이머 눈금보다 작은 차이는 **분해할 수 없다**.

    G-7 중간 점검에서 슬라이스 내 이동이 `-6.49%` 로 나왔다. 음수(= start 가
    느림)라 **냉시작 신호처럼 보였다** — 워밍업을 줄인 직후라 정확히
    우려하던 자리였다. 확인해 보니 **눈금의 0.84 배**였다.

        start  값 [13.312, 13.312, 13.312, 13.312]   <- 전부 같은 눈금
        end    값 [12.288, 12.448, 13.312]
        눈금 1.024us = 13.312us 의 7.69%

    start 가 전부 정확히 같은 것이 양자화의 서명이다. 열이나 캐시 거동이면
    그럴 수 없다.
    """

    def test_짧은_커널에서_눈금이_크다(self):
        assert noise.tick_pct(0.013312) == pytest.approx(0.0769, abs=1e-3)
        assert noise.tick_pct(0.0143) == pytest.approx(0.0716, abs=1e-3)

    def test_긴_커널에서는_무시할_수_있다(self):
        assert noise.tick_pct(2.9) < 0.001

    def test_실측_사례가_눈금_하나_미만이다(self):
        start, end = 0.013312, 0.012448
        moved = abs(end / start - 1)
        assert moved == pytest.approx(0.0649, abs=1e-3)
        assert moved < noise.tick_pct(start), (
            "이 차이는 눈금보다 작다 — 분해할 수 없다")

    def test_관측값에서_양자를_추정한다(self):
        vals = [0.013312, 0.013312, 0.014336, 0.012288, 0.014336]
        assert noise.tick_ms_observed(vals) == pytest.approx(0.001024, abs=1e-9)

    def test_값이_하나뿐이면_추정_못_한다(self):
        assert noise.tick_ms_observed([0.5, 0.5]) is None

    def test_0_이하는_무한대(self):
        assert noise.tick_pct(0) == float("inf")
