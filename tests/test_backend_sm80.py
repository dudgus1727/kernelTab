"""SM80 백엔드 — 제약, smem 공식, HMMA 공식, split-K 의미론.

`is_valid_kernel` 은 **컴파일/실행 가능성만** 본다. 성능으로 거르지 않는다.
각 제약마다 최소 한 개씩 통과/탈락 사례를 고정해 둔다. 제약을 건드렸을 때
무엇이 달라지는지 즉시 드러나야 한다.
"""

from backends.sm80 import (
    SPLIT_K,
    epilogue_thread_map_ok,
    mainloop_smem_thread_map_ok,
    warp_k_options,
)
from core.types import Problem, RuntimeConfig


def why(backend, cfg, hw, nb=2):
    return backend.explain_kernel(cfg, hw, nb)


class TestIsValidKernelPerConstraint:
    def test_baseline_is_valid(self, backend, hw_a6000, mk_cfg):
        assert why(backend, mk_cfg(), hw_a6000) is None

    def test_tile_not_divisible_by_warp(self, backend, hw_a6000, mk_cfg):
        cfg = mk_cfg(tile=(128, 128, 32), warp=(48, 64, 32))
        assert why(backend, cfg, hw_a6000) == "tile_divisible_by_warp"

    def test_warp_count_must_be_4_8_16(self, backend, hw_a6000, mk_cfg):
        # 128/64 * 128/64 = 4 -> 통과
        assert why(backend, mk_cfg(tile=(128, 128, 32), warp=(64, 64, 32)),
                   hw_a6000) is None
        # 128/128 * 128/128 = 1 -> 탈락
        cfg = mk_cfg(tile=(128, 128, 32), warp=(128, 128, 32))
        assert why(backend, cfg, hw_a6000) == "warp_count_mn"

    def test_warp_shape_vs_instruction(self, backend, hw_a6000, mk_cfg):
        """mma.m16n8k16 이므로 warp_m % 16, warp_n % 8 이어야 한다."""
        cfg = mk_cfg(tile=(128, 128, 32), warp=(32, 4, 32))
        assert why(backend, cfg, hw_a6000) in (
            "tile_divisible_by_warp", "warp_count_mn",
            "warp_shape_vs_instruction")

    def test_warp_k_divisible(self, backend, hw_a6000, mk_cfg):
        cfg = mk_cfg(tile=(128, 128, 32), warp=(64, 64, 24))
        assert why(backend, cfg, hw_a6000) == "warp_k_divisible"

    def test_smem_capacity(self, backend, hw_a6000, mk_cfg):
        """256x256x64 stages=8 은 어떤 GPU 에서도 smem 을 넘는다."""
        cfg = mk_cfg(tile=(256, 256, 64), warp=(64, 64, 64), stages=8)
        assert why(backend, cfg, hw_a6000) in (
            "warp_count_mn", "smem_capacity", "threads_per_block")

    def test_smem_uses_hw_limit(self, backend, hw_a6000, hw_other, mk_cfg):
        """smem 한도를 하드코딩하면 이 테스트가 잡는다 (84/101376 금지)."""
        cfg = mk_cfg(tile=(128, 256, 64), warp=(64, 64, 64), stages=4)
        a = why(backend, cfg, hw_a6000)          # 101376 한도
        b = why(backend, cfg, hw_other)          # 65536 한도
        assert (a, b) != (None, None) or a == b
        # 한도가 작은 쪽에서 먼저 걸려야 한다
        assert backend.smem_bytes(cfg, 2) > hw_other.smem_per_block
        assert b == "smem_capacity"

    def test_accumulator_registers(self, backend, hw_a6000, mk_cfg):
        """warp 128x128 -> 128*128/32 = 512 누산 레지스터 > 256.

        stages=2 로 두어 smem 제약(먼저 검사된다)에 걸리지 않게 한다.
        """
        cfg = mk_cfg(tile=(256, 256, 32), warp=(128, 128, 32), stages=2)
        assert backend.smem_bytes(cfg, 2) <= hw_a6000.smem_per_block
        assert why(backend, cfg, hw_a6000) == "accumulator_registers"

    def test_horizontal_swizzle_n_must_be_1(self, backend, hw_a6000, mk_cfg):
        cfg = mk_cfg(swizzle=("horizontal", 4))
        assert why(backend, cfg, hw_a6000) == "horizontal_swizzle_n"
        assert why(backend, mk_cfg(swizzle=("horizontal", 1)), hw_a6000) is None

    def test_cpasync_min_access(self, backend, hw_a6000, mk_cfg):
        """alignment 1 (fp16 = 2바이트) 은 multistage 불가, 2단만 가능."""
        assert why(backend, mk_cfg(stages=4, align=(1, 1, 8)),
                   hw_a6000) == "cpasync_min_access"
        assert why(backend, mk_cfg(stages=2, align=(1, 1, 8)), hw_a6000) is None
        # alignment 2 = 4바이트 -> multistage 가능
        assert why(backend, mk_cfg(stages=4, align=(2, 2, 8)), hw_a6000) is None

    def test_warp_n_128_rejected(self, backend, hw_a6000, mk_cfg):
        """실측 60/60 오답. 성능이 아니라 정확성 문제라 열거에서 뺀다."""
        cfg = mk_cfg(tile=(128, 256, 32), warp=(64, 128, 32))
        assert why(backend, cfg, hw_a6000) == "warp_n_incorrect_results"
        # mirror 인 (128,64) 는 정상 동작하므로 살아 있어야 한다
        assert why(backend, mk_cfg(tile=(256, 128, 32), warp=(128, 64, 32)),
                   hw_a6000) is None


class TestThreadMapPredicates:
    def test_mainloop_rule(self):
        """tile_m*tile_k >= 8*threads and tile_n*tile_k >= 8*threads."""
        assert mainloop_smem_thread_map_ok(128, 128, 32, 512)      # 4096 == 4096
        assert not mainloop_smem_thread_map_ok(256, 64, 32, 512)   # B: 2048 < 4096
        assert not mainloop_smem_thread_map_ok(64, 256, 32, 512)   # A: 2048 < 4096
        assert not mainloop_smem_thread_map_ok(64, 256, 64, 1024)  # A: 4096 < 8192

    def test_epilogue_rule_case_b(self):
        """warps_n*warps_k >= 8 이면 tile_n/align_c/32 >= 1 이어야 한다."""
        # tile_n=128, align_c=8 -> 128/8/32 = 0
        assert not epilogue_thread_map_ok(128, 2, 4, 2, 8)
        # align_c=2 -> 128/2/32 = 2
        assert epilogue_thread_map_ok(128, 2, 4, 2, 2)

    def test_epilogue_rule_case_a(self):
        assert epilogue_thread_map_ok(128, 2, 2, 1, 8)


class TestSmemBytes:
    def test_mainloop_formula(self, backend, mk_cfg):
        """stages * tile_k * (tile_m + tile_n) * dtype_bytes."""
        cfg = mk_cfg(tile=(128, 128, 32), warp=(64, 64, 32), stages=4)
        assert backend.smem_bytes(cfg, 2) == 4 * 32 * (128 + 128) * 2 == 65536

    def test_epilogue_can_dominate(self, backend, mk_cfg):
        """stages 가 작고 tile 이 크면 epilogue 가 이긴다."""
        cfg = mk_cfg(tile=(256, 256, 32), warp=(64, 64, 32), stages=2)
        mainloop = 2 * 32 * (256 + 256) * 2
        assert backend.smem_bytes(cfg, 2) > mainloop

    def test_align_c_changes_fragments(self, backend, mk_cfg):
        """kFragmentsPerIteration=2 는 align_c==8 특수화에서만 적용된다.

        align_c=2 커널 120개가 과대평가됐던 버그의 회귀 테스트다.
        """
        base = dict(tile=(64, 64, 32), warp=(16, 64, 32), stages=2)
        a8 = backend.smem_bytes(mk_cfg(**base, align=(8, 8, 8)), 2)
        a2 = backend.smem_bytes(mk_cfg(**base, align=(8, 8, 2)), 2)
        assert a8 == 18432      # epilogue 가 이긴다 (frags=2)
        assert a2 == 16384      # mainloop 가 이긴다 (frags=1)
        assert a2 < a8

    def test_scales_with_dtype(self, backend, mk_cfg):
        cfg = mk_cfg()
        assert backend.smem_bytes(cfg, 4) == 2 * backend.smem_bytes(cfg, 2)


class TestExpectedHmma:
    def test_static_per_warp(self, backend, mk_cfg):
        """(warp_m/16) * (warp_n/8) * (warp_k/16)."""
        assert backend.expected_hmma(mk_cfg(warp=(64, 64, 32))) == 4 * 8 * 2 == 64
        assert backend.expected_hmma(mk_cfg(warp=(16, 64, 32))) == 1 * 8 * 2 == 16
        assert backend.expected_hmma(mk_cfg(warp=(32, 32, 64))) == 2 * 4 * 4 == 32

    def test_dynamic_total_scales(self, backend, mk_cfg):
        cfg = mk_cfg(tile=(128, 128, 32), warp=(64, 64, 32))
        p, rc = Problem(1024, 1024, 4096), RuntimeConfig(1, "serial")
        small = backend.expected_hmma(cfg, rc, p)
        big = backend.expected_hmma(cfg, rc, Problem(2048, 1024, 4096))
        assert big == 2 * small


class TestSplitKSemantics:
    def test_effective_split_k_matches_cutlass_formula(self, backend):
        """kAlignK = max(8, K%64==0 ? 64 : 1); gemm_k = round_up(ceil(K/sk), kAlignK)."""
        p = Problem(1024, 1024, 4096)          # K % 64 == 0 -> kAlignK 64
        assert backend.effective_split_k(p, RuntimeConfig(3, "serial")) == 3
        assert backend.effective_split_k(p, RuntimeConfig(6, "serial")) == 6
        assert backend.effective_split_k(p, RuntimeConfig(12, "serial")) == 11

    def test_small_k_collapses(self, backend):
        p = Problem(1024, 4096, 512)
        assert backend.effective_split_k(p, RuntimeConfig(6, "serial")) == 4
        assert backend.effective_split_k(p, RuntimeConfig(16, "serial")) == 8

    def test_k_not_multiple_of_64(self, backend):
        """K=4100 -> kAlignK 8. split_k=1 도 유효해야 한다 (층 D 가 성립하려면)."""
        p = Problem(1024, 4096, 4100)
        assert backend.effective_split_k(p, RuntimeConfig(1, "serial")) == 1
        assert backend.effective_split_k(p, RuntimeConfig(2, "serial")) == 2


class TestIsValidRuntime:
    def test_mismatched_effective_split_k_rejected(self, backend, mk_cfg):
        cfg = mk_cfg(tile=(128, 128, 32))
        p = Problem(1024, 1024, 4096)
        assert backend.is_valid_runtime(RuntimeConfig(6, "serial"), p, cfg)
        assert not backend.is_valid_runtime(RuntimeConfig(12, "serial"), p, cfg)

    def test_split_k1_parallel_dropped(self, backend, mk_cfg):
        cfg = mk_cfg()
        p = Problem(1024, 1024, 4096)
        assert backend.is_valid_runtime(RuntimeConfig(1, "serial"), p, cfg)
        assert not backend.is_valid_runtime(RuntimeConfig(1, "parallel"), p, cfg)

    def test_alignment_must_match_shape(self, backend, mk_cfg):
        """★ 이 가드가 없으면 a888 커널이 K=4100 형상에 쓰여 결과가 틀린다."""
        p = Problem(1024, 4096, 4100)          # (4,4,8)
        assert not backend.is_valid_runtime(
            RuntimeConfig(1, "serial"), p, mk_cfg(align=(8, 8, 8)))
        assert backend.is_valid_runtime(
            RuntimeConfig(1, "serial"), p, mk_cfg(align=(4, 4, 8)))

    def test_slice_needs_one_k_tile(self, backend, mk_cfg):
        cfg = mk_cfg(tile=(128, 128, 64))
        assert not backend.is_valid_runtime(
            RuntimeConfig(16, "serial"), Problem(1024, 4096, 512), cfg)

    def test_split_k_axis_keeps_3_6_12(self):
        """A6000 SM 84 = 2^2*3*7 가설 검증용. 절대 줄이면 안 된다."""
        assert {3, 6, 12} <= set(SPLIT_K)


class TestWarpKOptions:
    def test_no_split_for_small_tile_k(self):
        assert warp_k_options(32) == [32]

    def test_split_for_tile_k_64(self):
        assert warp_k_options(64) == [64, 32]


class TestNoHardcodedHardware:
    """★ hw 를 바꾸면 결과가 달라져야 한다. 84/101376 이 박혀 있으면 안 된다."""

    def test_enumeration_reacts_to_hw(self, backend, hw_a6000, hw_other):
        from core.config import enumerate_kernels
        a = len(enumerate_kernels(hw_a6000, backend, [(8, 8, 8)]))
        b = len(enumerate_kernels(hw_other, backend, [(8, 8, 8)]))
        assert a != b, "smem_per_block 을 바꿔도 유효 커널 수가 같다 — 하드코딩 의심"
        assert b < a       # hw_other 는 smem 이 작다
