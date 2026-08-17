"""alignment 유도와 dtype 처리.

alignment 는 탐색 축이 아니라 (형상, 레이아웃)에서 유도되는 값이다.
여기가 틀리면 커널이 잘못된 벡터 폭으로 메모리를 읽어 **조용히 틀린 결과**가
나온다. 층 D 형상은 그 유도를 검증하려고 존재한다.
"""
import pytest

from core.config import DTYPE_BYTES, alignment_combos, alignments_for, dtype_bytes
from core.types import Problem


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
        from core.shapes import shapes_layer_d
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
