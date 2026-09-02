"""노이즈 바닥이 **형상마다 다르게** 동작하는가.

고정 1 % 로 되돌아가면 여기서 걸린다. 33시간 앵커에서 크기별 재현성이
35 배 차이났고, 뭉개면 작은 형상에서 노이즈를 신호로 배운다.
"""
from __future__ import annotations

import itertools

import pytest

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

from kerneltab.core import noise
from kerneltab.core.noise import (
    A6000_MEASURED,
    SIGMA_ABS_MS,
    SIGMA_REL,
    coefficients,
)

# 계수는 **주입**한다 (C-1). 모듈 전역 함수는 지웠다 — 4090/H100 번들에서
# A6000 눈금을 조용히 쓰던 경로였다. 이 파일은 A6000 계수의 성질을 검사하니
# 그것을 이름으로 집어 온다.
noise_floor = A6000_MEASURED.floor
noise_floor_ms = A6000_MEASURED.floor_ms
resolvable = A6000_MEASURED.resolvable
sigma_rel = A6000_MEASURED.sigma
tick_pct = A6000_MEASURED.tick_pct


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
        # 0.0143 ms -> 실측 2.93%,  2.9164 ms -> 실측 0.018%
        assert 0.015 < sigma_rel(0.0143) < 0.045
        assert 0.0003 < sigma_rel(2.9164) < 0.0015

    def test_짧은_커널에서는_분해능이_지배한다(self):
        """★ 통계 모델만 쓰면 **과소평가**한다.

        14 us 에서 sigma_rel 은 2.7 % 인데 눈금 하나는 7.3 % 다. 같은 눈금에
        떨어진 두 config 는 시간이 **문자 그대로 동일**하게 기록되므로
        2.7 % 기준으로 "구분된다" 고 하면 없는 순위를 만든다.
        """
        t = 0.0143
        assert tick_pct(t) > sigma_rel(t)
        assert noise_floor(t) == pytest.approx(tick_pct(t))

    def test_긴_커널에서는_통계가_지배한다(self):
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
        assert tick_pct(0.013312) == pytest.approx(0.0769, abs=1e-3)
        assert tick_pct(0.0143) == pytest.approx(0.0716, abs=1e-3)

    def test_긴_커널에서는_무시할_수_있다(self):
        assert tick_pct(2.9) < 0.001

    def test_실측_사례가_눈금_하나_미만이다(self):
        start, end = 0.013312, 0.012448
        moved = abs(end / start - 1)
        assert moved == pytest.approx(0.0649, abs=1e-3)
        assert moved < tick_pct(start), (
            "이 차이는 눈금보다 작다 — 분해할 수 없다")

    def test_관측값에서_양자를_추정한다(self):
        vals = [0.013312, 0.013312, 0.014336, 0.012288, 0.014336]
        assert noise.tick_ms_observed(vals) == pytest.approx(0.001024, abs=1e-9)

    def test_값이_하나뿐이면_추정_못_한다(self):
        assert noise.tick_ms_observed([0.5, 0.5]) is None

    def test_0_이하는_무한대(self):
        assert tick_pct(0) == float("inf")


# --------------------------------------------------------------------------
# D-6 (2026-09-01) — 번들이 A6000 계수를 싣던 문제
# --------------------------------------------------------------------------

class TestCoefFromAnchors:
    """계수는 **그 캠페인의 앵커에서** 나와야 한다.

    `scripts/bundle.py::_noise_coefficients()` 가 docstring 은 "앵커에서 잰"
    이라면서 `noise.coefficients()` — A6000 모듈 상수 — 를 돌려줬다. 그대로
    5090 번들에 실렸으면 짧은 형상의 정답 집합이 **1,477 배** 넓어진다
    (`8x4096x4096` 에서 정답 1 개 대신 1,477 개).
    """

    def _rows(self, times_by_group):
        return [{"kernel_id": k, "problem": {"M": m, "N": 4096, "K": 4096},
                 "time_ms": t}
                for (k, m), ts in times_by_group.items() for t in ts]

    def test_짧은_긴_앵커에서_각각_뽑는다(self):
        from kerneltab.core import noise

        # 짧은 조합은 절대 산포가, 긴 조합은 상대 산포가 지배하도록 만든다.
        rows = self._rows({
            ("k_short_a", 512): [0.100000, 0.100016, 0.100032],
            ("k_short_b", 512): [0.110000, 0.110016, 0.110032],
            ("k_long_a", 4096): [1.000000, 1.000800, 1.001600],
            ("k_long_b", 4096): [1.200000, 1.200960, 1.201920],
        })
        c = noise.coef_from_anchors(rows, "테스트")
        assert c.tick_ms == pytest.approx(16e-6, rel=0.01)
        assert c.sigma_abs_ms > 0
        assert c.sigma_rel > 0
        assert "테스트" in c.source and "앵커" in c.source

    def test_앵커가_부족하면_A6000_으로_대체하지_않고_실패한다(self):
        from kerneltab.core import noise

        with pytest.raises(noise.NoiseCoefUnavailable):
            noise.coef_from_anchors([], "빈 앵커")

    def test_눈금이_물리적_범위_밖이면_실패한다(self):
        """⛔ 대체하지 않는다. 조용한 대체가 D-6 을 만들었다."""
        from kerneltab.core import noise

        rows = self._rows({
            ("a", 512): [1e-9, 2e-9, 3e-9],      # 눈금 1e-9 ms = 1 fs
            ("b", 4096): [4e-9, 5e-9, 6e-9],
        })
        with pytest.raises(noise.NoiseCoefUnavailable):
            noise.coef_from_anchors(rows, "말도 안 되는 눈금")

    def test_TICK_PLAUSIBLE_은_절대_범위여야_한다(self):
        """A6000 눈금의 배수로 정의하면 32 ns 가 범위 밖이 된다."""
        from kerneltab.core import noise

        lo, hi = noise.TICK_PLAUSIBLE_MS
        for tick_ns in (16, 32, 1024):       # 5090 / 초기추정 / A6000
            assert lo <= tick_ns * 1e-6 <= hi, (
                f"{tick_ns} ns 가 허용 범위 밖이다 — "
                "범위를 첫 환경의 값 배수로 정의하지 마라 (decisions 25)")


class TestTickGrid:
    """눈금은 **격자를 설명하는지** 확인해서 고른다."""

    def test_부동소수_잡음을_정수_ns_로_스냅한다(self):
        from kerneltab.core import noise

        # 16 ns 격자 위의 값들 + 최소 간격에 실린 미세 잡음
        xs = [16e-6 * n for n in (1, 2, 3, 5, 8, 45000)]
        tick, cover = noise.tick_grid(xs)
        assert tick == pytest.approx(16e-6, rel=1e-9)
        assert cover == 1.0

    def test_거친_격자를_선호한다(self):
        """8 ns 는 16 ns 격자도 전부 설명한다 — 그래도 16 을 골라야 한다."""
        from kerneltab.core import noise

        xs = [16e-6 * n for n in (1, 2, 3, 4, 7, 11)]
        tick, cover = noise.tick_grid(xs)
        assert tick == pytest.approx(16e-6, rel=1e-9)


def test_bundle_이_모듈_상수를_그대로_싣지_않는다():
    """`_noise_coefficients()` 가 `noise.coefficients()` 를 돌려주면 안 된다."""
    src = (REPO / "scripts" / "bundle.py").read_text()
    fn = src[src.index("def _noise_coefficients("):]
    fn = fn[:fn.index("\ndef ", 1)]
    assert "coef_from_anchors" in fn, (
        "번들이 이 캠페인의 앵커에서 계수를 뽑지 않는다 — "
        "A6000 값이 실리면 정답 집합이 짧은 형상에서 1,000 배 넓어진다 (D-6)")
    assert "return noise.coefficients()" not in fn


class TestOddMultiple:
    """★ 눈금 추정을 **위에서** 조인다 (부분표본 재추정은 아래에서 조인다).

    후보가 진짜 눈금의 짝수배면 그 후보의 홀수 배수인 값이 존재할 수 없다.
    H100 에서 16 ns 가 설명력 100 % 인데 홀수 배수 0 % 라 기각됐다.
    """

    def test_진짜_눈금은_홀수_배수를_갖는다(self):
        from kerneltab.core.noise import odd_multiple_frac
        q = 32e-6
        vals = [q * n for n in (3, 5, 7, 9, 11, 100, 101)]
        assert odd_multiple_frac(vals, q) > 0.5

    def test_절반_후보는_홀수_배수가_없다(self):
        """32 ns 격자 위의 값은 16 ns 격자로도 전부 설명되지만 전부 짝수배다."""
        from kerneltab.core.noise import odd_multiple_frac, tick_grid
        q = 32e-6
        vals = [q * n for n in (3, 5, 7, 9, 11, 100, 101)]
        assert tick_grid(vals)[1] == 1.0            # 설명력만으로는 못 가른다
        assert odd_multiple_frac(vals, q / 2) == 0.0

    def test_격자_밖_값은_세지_않는다(self):
        from kerneltab.core.noise import odd_multiple_frac
        q = 32e-6
        assert odd_multiple_frac([q * 2.5], q) == 0.0

    def test_두_함수가_같은_허용오차를_쓴다(self):
        """다르면 한쪽이 격자로 인정한 값을 다른 쪽이 세지 않는다."""
        import inspect

        from kerneltab.core import noise
        src = inspect.getsource(noise)
        assert src.count("TICK_ON_GRID_TOL") >= 3, (
            "격자 허용오차가 상수로 공유되지 않는다")
        assert "< 0.02) / len(" not in src, "허용오차를 다시 박아 넣었다"
