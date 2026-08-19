"""형상 그리드.

층 C 는 **waves 를 고정하고 M 을 역산**한다. GPU 마다 sm_count 가 다르므로
같은 M 이 다른 물리적 상황을 의미하기 때문이다. 그 역산이 실제로 hw 에
반응하는지 확인하는 것이 이 파일의 핵심이다.
"""

from core.shapes import (
    all_layers,
    all_shapes,
    shapes_layer_a,
    shapes_layer_b,
    shapes_layer_c,
    shapes_layer_d,
    shapes_layer_e,
)


class TestLayerCounts:
    def test_documented_counts(self, hw_a6000):
        assert len(shapes_layer_a()) == 40      # (N,K) 4종 x M 10종
        assert len(shapes_layer_b()) == 12      # 8 + 4
        assert len(shapes_layer_c(hw_a6000)) == 11
        assert len(shapes_layer_d()) == 5
        assert len(shapes_layer_e()) == 5

    def test_layer_sum_and_unique(self, hw_a6000):
        layers = all_layers(hw_a6000)
        assert sum(len(v) for v in layers.values()) == 73
        assert len(all_shapes(hw_a6000)) == 66     # 층 간 중복 7개 제거

    def test_all_shapes_dedups(self, hw_a6000):
        shapes = all_shapes(hw_a6000)
        assert len(shapes) == len(set(shapes))


class TestLayerCIsHardwareDerived:
    """★ 층 C 가 sm_count 에 반응하지 않으면 역산이 동작하지 않는 것이다."""

    def test_different_sm_count_gives_different_m(self, hw_a6000, hw_other):
        m_84 = [p.M for p in shapes_layer_c(hw_a6000)]     # SM 84
        m_128 = [p.M for p in shapes_layer_c(hw_other)]    # SM 128
        assert m_84 != m_128, "sm_count 를 바꿔도 M 이 같다 — 역산 미동작"

    def test_m_scales_with_sm_count(self, hw_a6000, hw_other):
        """SM 이 많을수록 같은 waves 를 채우는 데 더 큰 M 이 필요하다."""
        # 2의 거듭제곱이 아닌 고정 M (1000/1500/3000) 은 제외하고 비교
        fixed = {1000, 1500, 3000}
        m_84 = sorted(m for m in {p.M for p in shapes_layer_c(hw_a6000)}
                      if m not in fixed)
        m_128 = sorted(m for m in {p.M for p in shapes_layer_c(hw_other)}
                       if m not in fixed)
        assert max(m_128) > max(m_84)

    def test_waves_targets_are_hit(self, hw_a6000, mk_cfg):
        """역산된 M 이 실제로 목표 waves 근처를 만드는지."""
        from core import features as F
        from core.types import RuntimeConfig
        cfg = mk_cfg(tile=(128, 128, 32))
        got = sorted({round(F.waves(p, hw_a6000, cfg, RuntimeConfig(1, "serial")), 2)
                      for p in shapes_layer_c(hw_a6000)})
        # 목표 [0.3, 0.5, 0.76, 1.2, 1.5, 2.3, 3.05, 4.5, 6.7] 부근을 덮는가
        assert min(got) < 0.6 and max(got) > 6.0
        assert len(got) >= 8

    def test_fixed_m_values_present(self, hw_a6000):
        """2의 거듭제곱이 아닌 M 이 포함되어야 한다 (타일 경계 비정렬)."""
        ms = {p.M for p in shapes_layer_c(hw_a6000)}
        assert {1000, 1500, 3000} <= ms

    def test_c_dedups_internally(self, hw_a6000):
        shapes = shapes_layer_c(hw_a6000)
        assert len({p.M for p in shapes}) == len(shapes)


class TestHardwareIndependentLayers:
    """층 A/B/D/E 는 hw 와 무관하게 항상 같아야 한다."""

    def test_a_b_d_e_are_constant(self, hw_a6000, hw_other):
        assert shapes_layer_a() == shapes_layer_a()
        assert shapes_layer_b() == shapes_layer_b()
        assert shapes_layer_d() == shapes_layer_d()
        assert shapes_layer_e() == shapes_layer_e()
        # 이 함수들은 hw 를 인자로 받지도 않는다 (시그니처로 강제됨)
        import inspect
        for fn in (shapes_layer_a, shapes_layer_b, shapes_layer_d, shapes_layer_e):
            assert not inspect.signature(fn).parameters

    def test_layer_a_is_llama(self):
        nk = {(p.N, p.K) for p in shapes_layer_a()}
        assert nk == {(4096, 4096), (12288, 4096), (11008, 4096), (4096, 11008)}
        assert {p.M for p in shapes_layer_a()} == {
            1, 8, 32, 128, 256, 512, 1024, 2048, 4096, 8192}

    def test_layer_b_varies_k(self):
        """이 층이 없으면 stages 와 warp_k 에 대해 아무것도 배울 수 없다."""
        ks = {p.K for p in shapes_layer_b()}
        assert len(ks) >= 8
        assert min(ks) == 128 and max(ks) == 16384

    def test_layer_e_is_square(self):
        assert all(p.M == p.N == p.K for p in shapes_layer_e())


class TestLayerD:
    def test_covers_all_low_alignments(self):
        from core.config import alignments_for
        got = {alignments_for(p) for p in shapes_layer_d()}
        assert got == {(4, 4, 8), (2, 2, 8), (1, 1, 8), (8, 8, 4), (8, 8, 2)}

    def test_no_full_alignment_shape(self):
        """층 D 는 alignment 엣지케이스 전용이다. (8,8,8) 이 섞이면 목적이 흐려진다."""
        from core.config import alignments_for
        assert all(alignments_for(p) != (8, 8, 8) for p in shapes_layer_d())
