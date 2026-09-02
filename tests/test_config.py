"""alignment 유도와 dtype 처리.

alignment 는 탐색 축이 아니라 (형상, 레이아웃)에서 유도되는 값이다.
여기가 틀리면 커널이 잘못된 벡터 폭으로 메모리를 읽어 **조용히 틀린 결과**가
나온다. 층 D 형상은 그 유도를 검증하려고 존재한다.
"""
import pytest

from kerneltab.core.config import (
    DTYPE_BYTES,
    alignment_combos,
    alignments_for,
    dtype_bytes,
)
from kerneltab.core.types import Problem


class TestAlignmentsFor:
    @pytest.mark.parametrize("K,expected", [
        (4096, (8, 8, 8)),   # K % 8 == 0
        (4100, (4, 4, 8)),   # 4100 = 4 * 1025
        (4098, (2, 2, 8)),   # 4098 = 2 * 2049
        (4097, (1, 1, 8)),   # 홀수
    ])
    def test_layer_d_k_variants(self, K, expected):
        """층 D 의 K 변형. A/B 는 K 가 연속 차원이므로 K 로 결정된다."""
        assert alignments_for(Problem(1024, 4096, K)) == expected

    @pytest.mark.parametrize("N,expected", [
        (4096, (8, 8, 8)),
        (4100, (8, 8, 4)),
        (4098, (8, 8, 2)),
        (4097, (8, 8, 1)),
    ])
    def test_layer_d_n_variants(self, N, expected):
        """C 는 row-major 이므로 N 이 연속 차원이다."""
        assert alignments_for(Problem(1024, N, 4096)) == expected

    def test_layout_changes_which_dim_binds(self):
        """레이아웃을 바꾸면 걸리는 차원이 바뀐다 — 이것이 이 함수의 핵심이다."""
        # A row-major -> K 가 연속. K=4097 이면 align_a=1
        assert alignments_for(Problem(1024, 4096, 4097)).__getitem__(0) == 1
        # A column-major -> M 이 연속. K 가 홀수여도 M=1024 라 8
        p = Problem(1024, 4096, 4097, layout_a="col")
        assert alignments_for(p)[0] == 8
        # B column-major -> K 가 연속 / row-major -> N 이 연속
        assert alignments_for(Problem(1024, 4096, 4097, layout_b="col"))[1] == 1
        assert alignments_for(Problem(1024, 4096, 4097, layout_b="row"))[1] == 8
        # C column-major -> M 이 연속
        assert alignments_for(Problem(1023, 4097, 4096, layout_c="col"))[2] == 1
        assert alignments_for(Problem(1024, 4097, 4096, layout_c="col"))[2] == 8

    def test_max_is_8_not_more(self):
        """fp16 에서 128비트 = 8원소가 상한이다. 16 이 나오면 안 된다."""
        assert alignments_for(Problem(4096, 4096, 4096)) == (8, 8, 8)

    def test_odd_everything(self):
        assert alignments_for(Problem(1, 1, 1)) == (1, 1, 1)


class TestAlignmentCombos:
    def test_dedup_and_sorted(self):
        shapes = [Problem(1024, 4096, 4096), Problem(2048, 4096, 4096),
                  Problem(1024, 4096, 4100)]
        combos = alignment_combos(shapes)
        assert combos == [(4, 4, 8), (8, 8, 8)]

    def test_layer_d_yields_five_distinct(self):
        from kerneltab.core.shapes import shapes_layer_d
        assert len(alignment_combos(shapes_layer_d())) == 5


class TestDtypeBytes:
    def test_known(self):
        assert dtype_bytes("f16") == 2
        assert dtype_bytes("f32") == 4
        assert set(DTYPE_BYTES) >= {"f16", "bf16", "f32"}

    def test_unknown_raises(self):
        """조용히 기본값을 쓰면 alignment 가 통째로 틀린다. 반드시 예외."""
        with pytest.raises(ValueError, match="알 수 없는 dtype"):
            dtype_bytes("fp8_e4m3_but_typo")


class TestPredicateCount:
    """★ 세 번째 컴파일 제약 — `static_assert: Too many predicates.`

    H100 NVL 전수 빌드 13,975 개에서 실패 20 건이 **전부** 이 조건 하나로
    갈렸다 (TP 20 / TN 13,955 / FN 0 / FP 0). 성공 커널의 최대 접근수 64,
    실패 커널의 최소 접근수 128 로 경계가 깨끗하다.
    """

    def test_실측_실패_조합을_거른다(self):
        from kerneltab.backends.cutlass_v2 import predicate_count_ok
        # 실제 실패한 넷 (alignment 1, tile_k 64, 스레드 128)
        assert not predicate_count_ok(64, 256, 64, 128, 1, 1)
        assert not predicate_count_ok(128, 256, 64, 128, 1, 1)
        assert not predicate_count_ok(256, 64, 64, 128, 1, 1)
        assert not predicate_count_ok(256, 128, 64, 128, 1, 1)

    def test_같은_타일이라도_alignment_가_2_면_통과한다(self):
        """실측: 같은 (tile, warp, stages) 로 align>=2 인 100 건 전부 성공."""
        from kerneltab.backends.cutlass_v2 import predicate_count_ok
        for al in (2, 4, 8):
            assert predicate_count_ok(64, 256, 64, 128, al, al)
            assert predicate_count_ok(256, 128, 64, 128, al, al)

    def test_경계값(self):
        """접근수 64 는 통과, 65 부터 거부 (kPredicateWordCount <= 4)."""
        from kerneltab.backends.cutlass_v2 import MAX_PREDICATES, predicate_count_ok
        assert MAX_PREDICATES == 64
        assert predicate_count_ok(64, 64, 64, 128, 1, 1)        # 32 접근
        assert predicate_count_ok(128, 64, 64, 128, 1, 1)       # 64 접근 (경계)
        assert not predicate_count_ok(256, 64, 64, 128, 1, 1)   # 128 접근

    def test_성능_필터가_아니다(self, hw_h100, backend):
        """이 제약은 **컴파일 가능성**만 본다 — 걸리는 것은 alignment 1 뿐이다.

        alignment 8 인 정상 config 를 하나라도 자르면 성능 필터가 된 것이다.
        """
        from kerneltab.core.shapes import all_shapes
        from kerneltab.core.config import alignment_combos, enumerate_kernels_with_funnel
        combos = alignment_combos(all_shapes(hw_h100))
        _, funnel = enumerate_kernels_with_funnel(hw_h100, backend, combos, "f16")
        # funnel 은 첫 조합((1,1,8))만 센다 — 거기서만 걸려야 한다
        assert funnel.get("predicate_count", 0) > 0
        for al in ((8, 8, 8), (4, 4, 8), (2, 2, 8)):
            ks, f2 = enumerate_kernels_with_funnel(hw_h100, backend, [al], "f16")
            assert f2.get("predicate_count", 0) == 0, (
                f"align {al} 에서 predicate_count 가 걸렸다 — 이 제약은 "
                "alignment 1 전용이어야 한다")
