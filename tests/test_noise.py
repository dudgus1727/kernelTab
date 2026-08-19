"""노이즈 바닥이 **형상마다 다르게** 동작하는가.

고정 1 % 로 되돌아가면 여기서 걸린다. 33시간 앵커에서 크기별 재현성이
35 배 차이났고, 뭉개면 작은 형상에서 노이즈를 신호로 배운다.
"""
from __future__ import annotations

import pytest

from core.noise import (SIGMA_ABS_MS, SIGMA_REL, coefficients, noise_floor,
                        noise_floor_ms, resolvable)


class TestModel:
    def test_small_kernels_are_much_noisier(self):
        """15 us 커널이 3 ms 커널보다 훨씬 시끄럽다 — 이게 요점이다."""
        assert noise_floor(0.015) > 20 * noise_floor(3.0)

    def test_matches_measured_anchors(self):
        """앵커 실측과 자릿수가 맞는가 (33시간 x 312회)."""
        # 0.0143 ms -> 실측 2.93%,  2.9164 ms -> 실측 0.018%
        assert 0.015 < noise_floor(0.0143) < 0.045
        assert 0.0003 < noise_floor(2.9164) < 0.0015

    def test_monotone_decreasing(self):
        ts = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 20.0]
        v = [noise_floor(t) for t in ts]
        assert all(a > b for a, b in zip(v, v[1:]))

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
