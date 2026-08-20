"""파생 물리 피처.

★ 가장 중요한 불변식: **이 함수들은 `cfg.ext` 를 참조하면 안 된다.**
공통 필드(tile_m/n/k)와 hw, rc 만으로 계산되어야 SM80 에서 학습한 것을
SM90 에 적용하는 전이 실험이 성립한다. `ext=None` 테스트가 그것을 강제한다.
"""
import math

import pytest

from kerneltab.core import features as F
from kerneltab.core.types import KernelConfig, Problem, RuntimeConfig

SK1 = RuntimeConfig(1, "serial")


class TestWavesAndTail:
    def test_waves_hand_computed(self, hw_a6000, mk_cfg):
        """M=N=1024, tile 128x128 -> 8x8 = 64 타일, SM 84 -> 64/84."""
        cfg = mk_cfg(tile=(128, 128, 32))
        w = F.waves(Problem(1024, 1024, 4096), hw_a6000, cfg, SK1)
        assert w == pytest.approx(64 / 84)
        assert w == pytest.approx(0.7619047619)

    def test_tail_waste_hand_computed(self, hw_a6000, mk_cfg):
        """0.762 wave -> 올림 1 wave 중 0.238 이 논다."""
        cfg = mk_cfg(tile=(128, 128, 32))
        t = F.tail_waste(Problem(1024, 1024, 4096), hw_a6000, cfg, SK1)
        assert t == pytest.approx((1 - 64 / 84) / 1)
        assert t == pytest.approx(0.2380952381)

    def test_waves_scales_with_split_k(self, hw_a6000, mk_cfg):
        cfg = mk_cfg(tile=(128, 128, 32))
        p = Problem(1024, 1024, 4096)
        w1 = F.waves(p, hw_a6000, cfg, SK1)
        w6 = F.waves(p, hw_a6000, cfg, RuntimeConfig(6, "serial"))
        assert w6 == pytest.approx(w1 * 6)

    def test_waves_uses_hw_sm_count(self, hw_a6000, hw_other, mk_cfg):
        """sm_count 를 하드코딩하면 이 테스트가 잡는다."""
        cfg = mk_cfg(tile=(128, 128, 32))
        p = Problem(1024, 1024, 4096)
        assert F.waves(p, hw_a6000, cfg, SK1) != F.waves(p, hw_other, cfg, SK1)
        assert F.waves(p, hw_other, cfg, SK1) == pytest.approx(64 / 128)

    def test_tail_waste_zero_when_aligned(self, hw_a6000, mk_cfg):
        """타일 수가 SM 수의 배수면 낭비가 0 이다."""
        cfg = mk_cfg(tile=(128, 128, 32))
        # 84 타일 = 정확히 1 wave
        p = Problem(128 * 84, 128, 4096)
        assert F.waves(p, hw_a6000, cfg, SK1) == pytest.approx(1.0)
        assert F.tail_waste(p, hw_a6000, cfg, SK1) == pytest.approx(0.0)

    def test_blocks_per_sm_divides_waves(self, hw_a6000, mk_cfg):
        cfg = mk_cfg(tile=(128, 128, 32))
        p = Problem(1024, 1024, 4096)
        assert F.waves(p, hw_a6000, cfg, SK1, blocks_per_sm=2) == pytest.approx(
            F.waves(p, hw_a6000, cfg, SK1) / 2)


class TestMainloopIters:
    def test_hand_computed(self, mk_cfg):
        """K=4096, tile_k=64, split_k=6 -> ceil(4096/384) = 11."""
        cfg = mk_cfg(tile=(128, 128, 64))
        n = F.mainloop_iters(Problem(1024, 1024, 4096), cfg,
                             RuntimeConfig(6, "serial"))
        assert n == math.ceil(4096 / (64 * 6)) == 11

    def test_no_split(self, mk_cfg):
        cfg = mk_cfg(tile=(128, 128, 32))
        assert F.mainloop_iters(Problem(1, 1, 4096), cfg, SK1) == 128

    def test_residue_rounds_up(self, mk_cfg):
        """K 가 tile_k 로 안 나뉘면 올림한다 (CUTLASS 가 잔여를 처리한다)."""
        cfg = mk_cfg(tile=(128, 128, 32))
        assert F.mainloop_iters(Problem(1, 1, 4100), cfg, SK1) == 129


class TestArithIntensity:
    @pytest.mark.parametrize("n", [512, 1024, 2048, 4096, 8192])
    def test_square_converges_to_n_over_3(self, n):
        """정방형이면 2n^3 / (3n^2 * 2byte) = n/3."""
        assert F.arith_intensity(Problem(n, n, n)) == pytest.approx(n / 3)

    def test_skinny_is_memory_bound(self, hw_a6000):
        """M=1 은 극단적으로 메모리 바운드다."""
        assert F.arith_intensity(Problem(1, 4096, 4096)) < 3.0
        assert F.is_memory_bound(Problem(1, 4096, 4096), hw_a6000)

    def test_dtype_affects_bytes(self):
        f16 = F.arith_intensity(Problem(1024, 1024, 1024, dtype="f16"))
        f32 = F.arith_intensity(Problem(1024, 1024, 1024, dtype="f32"))
        assert f16 == pytest.approx(f32 * 2)


class TestRoofline:
    def test_ridge_point_uses_effective_peak(self, hw_a6000):
        """실효 피크/대역폭으로 계산되어야 한다. 스펙 값(154.8/768)이면 201.6."""
        rp = F.ridge_point(hw_a6000)
        assert rp == pytest.approx(116.1e12 / 729.7e9)
        assert rp == pytest.approx(159.106, abs=1e-3)
        assert rp != pytest.approx(201.5, abs=1.0)   # 스펙 기준이면 안 된다

    def test_bound_classification_flips_at_ridge(self, hw_a6000):
        rp = F.ridge_point(hw_a6000)
        # AI = n/3 이므로 n = 3*rp 부근에서 갈린다
        n_lo, n_hi = int(3 * rp) - 30, int(3 * rp) + 30
        assert F.is_memory_bound(Problem(n_lo, n_lo, n_lo), hw_a6000)
        assert not F.is_memory_bound(Problem(n_hi, n_hi, n_hi), hw_a6000)

    def test_ridge_point_depends_on_hw(self, hw_a6000, hw_other):
        assert F.ridge_point(hw_a6000) != F.ridge_point(hw_other)


class TestCpAsync:
    @pytest.mark.parametrize("K,ok,nbytes", [
        (4096, True, 16), (4100, True, 8), (4098, True, 4), (4097, False, 2),
    ])
    def test_only_k4097_is_blocked(self, K, ok, nbytes):
        """cp.async 는 4/8/16 바이트만 지원한다. fp16 x alignment 1 = 2 바이트."""
        p = Problem(1024, 4096, K)
        assert F.min_access_bytes(p) == nbytes
        assert F.can_use_cp_async(p) is ok

    def test_layout_matters(self):
        """A 를 column-major 로 두면 M 이 연속이라 K 가 홀수여도 통과한다."""
        assert not F.can_use_cp_async(Problem(1024, 4096, 4097))
        assert F.can_use_cp_async(
            Problem(1024, 4096, 4097, layout_a="col", layout_b="row"))


class TestNoExtDependency:
    """★ 물리 피처가 cfg.ext 를 참조하지 않는다는 것을 구조적으로 강제한다.

    참조하면 ext=None 에서 AttributeError 가 난다. 이 불변식이 깨지면
    아키텍처 전이 실험의 전제가 무너진다.
    """

    def test_all_feature_fns_work_with_ext_none(self, hw_a6000):
        cfg = KernelConfig(tile_m=128, tile_n=128, tile_k=32,
                           align_a=8, align_b=8, align_c=8,
                           arch="sm_86", ext=None)
        p = Problem(1024, 1024, 4096)
        rc = RuntimeConfig(6, "serial")
        # 하나라도 ext 를 건드리면 여기서 터진다
        assert F.grid_tiles(p, cfg, rc) > 0
        assert F.waves(p, hw_a6000, cfg, rc) > 0
        assert 0 <= F.tail_waste(p, hw_a6000, cfg, rc) < 1
        assert F.mainloop_iters(p, cfg, rc) > 0
        assert 0 < F.tail_m_frac(p, cfg) <= 1
        assert 0 < F.tail_n_frac(p, cfg) <= 1
        assert F.arith_intensity(p) > 0
        assert F.ridge_point(hw_a6000) > 0
        assert isinstance(F.is_memory_bound(p, hw_a6000), bool)
        assert F.flops(p) > 0 and F.bytes_moved(p) > 0
        assert F.min_access_bytes(p) > 0
        assert isinstance(F.can_use_cp_async(p), bool)

    def test_features_module_does_not_import_backends(self):
        """core 가 backends 에 런타임 의존하면 레이어링이 뒤집힌 것이다.

        ⚠️ 패키지 이전에서 이 검사가 **조용히 무력화될 뻔했다.** import 가
        `from backends...` 에서 `from kerneltab.backends...` 로 바뀌면서
        `module.split(".")[0]` 이 `"kerneltab"` 을 돌려주기 시작했고,
        `"backends" not in imported` 는 항상 참이 된다. 최상위 이름에
        의존하는 검사는 네임스페이스 이전에 전부 이렇게 된다.
        그래서 `kerneltab.` 접두사를 벗겨낸 뒤 비교한다.
        """
        import ast
        import pathlib

        def top(name: str) -> str:
            parts = name.split(".")
            if parts[0] == "kerneltab":
                parts = parts[1:]
            return parts[0] if parts else ""

        src = (pathlib.Path(__file__).resolve().parent.parent
               / "kerneltab" / "core" / "features.py")
        tree = ast.parse(src.read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(top(node.module))
            elif isinstance(node, ast.Import):
                imported.update(top(a.name) for a in node.names)
        # 검사가 실제로 무언가를 보고 있는지부터 확인한다 (빈 집합이면 통과가
        # 통과가 아니다).
        assert imported, "import 를 하나도 못 읽었다 — 검사가 죽어 있다"
        assert "backends" not in imported, (
            "kerneltab/core/features.py 가 backends 를 import 한다 — 레이어링 역전")
