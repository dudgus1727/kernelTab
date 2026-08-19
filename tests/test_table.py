"""소비 인터페이스 — **정답 누출 방지 장치가 실제로 막는지** 검증한다.

`docs/consumer_contract.md` 는 "문서는 지켜지지 않으므로 코드로 강제한다" 고
썼다. 그렇다면 그 코드를 지키는 것도 문서여서는 안 된다. 이 파일이 그 역할을
한다.

합성 parquet 으로만 돌린다 — 실제 측정 데이터에 의존하지 않는다.
"""
import warnings

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("pandas")

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from core.table import (
    ANSWER_COLS,
    KNOWN_FEATURE_COLS,
    OUTCOME_COLS,
    SAFE_META_COLS,
    AnswerLeakError,
    assert_no_answers,
    classify_columns,
    load_for_ranking,
    load_for_scoring,
)


def _rows(env_hash="aaaa1111", n_shapes=3, n_cfg=5):
    """형상 3개 x config 5개. 일부는 status != ok."""
    out = []
    for si, (M, N, K) in enumerate([(1024, 1024, 4096), (64, 4096, 4096),
                                    (4096, 4096, 4096)][:n_shapes]):
        for ci in range(n_cfg):
            out.append({
                # 메타
                "kernel_id": f"k{ci}", "env_hash": env_hash, "arch": "sm_86",
                "cutlass_commit": "deadbeef", "nvcc_arch": "sm_86",
                "clock_locked": True,
                # 형상 / config (피처)
                "M": M, "N": N, "K": K, "dtype": "f16",
                "tile_m": 128, "tile_n": 128, "tile_k": 32,
                "align_a": 8, "align_b": 8, "align_c": 8,
                "split_k": 1 + ci, "split_k_mode": "serial",
                "ext_warp_m": 64, "ext_warp_n": 64, "ext_warp_k": 32,
                "ext_stages": 4, "ext_swizzle_type": "identity",
                "ext_swizzle_n": 8,
                "waves": 0.76 + ci, "tail_waste": 0.2, "mainloop_iters": 128,
                "arith_intensity": 400.0, "is_memory_bound": False,
                "regs_per_thread": 128, "threads": 256, "launchable": True,
                "pipeline_kind": "multistage",
                # 결과 (OUTCOME)
                "status": "ok" if ci < 4 else "numerical_fail",
                "max_rel_error": 1e-4, "actual_split_k": 1 + ci,
                "sm_clock_mhz": 1350, "gpu_temp_c": 60,
                "timestamp": "2026-08-17T00:00:00.000000Z",
                # 정답 (ANSWER)
                "time_ms": 1.0 + 0.1 * ci + si,
                "time_std_ms": 0.001, "time_min_ms": 0.99, "time_max_ms": 1.01,
                "n_reps": 30, "outlier_frac": 0.0,
                "cublas_ms": 1.05 + si,
                "tflops": 100.0 - ci, "frac_of_peak": 0.8,
                "vs_cublas": 1.05,
                # 형상 난이도 = 중앙값/최적. 정답에서 유도된 값이라
                # ANSWER_COLS 에 있다 (규칙에 노출되면 안 된다).
                "difficulty": 1.8,
            })
    return out


@pytest.fixture
def table(tmp_path):
    p = tmp_path / "table.parquet"
    pq.write_table(pa.Table.from_pylist(_rows()), p)
    return p


@pytest.fixture
def mixed_table(tmp_path):
    """측정 조건 두 종이 섞인 표."""
    p = tmp_path / "mixed.parquet"
    pq.write_table(pa.Table.from_pylist(
        _rows("aaaa1111") + _rows("bbbb2222")), p)
    return p


class TestNoAnswerLeak:
    def test_ranking_has_no_answer_cols(self, table):
        X = load_for_ranking(table)
        leaked = [c for c in ANSWER_COLS if c in X.columns]
        assert leaked == [], f"정답 컬럼이 남았다: {leaked}"

    def test_ranking_has_no_outcome_cols(self, table):
        X = load_for_ranking(table)
        assert [c for c in OUTCOME_COLS if c in X.columns] == []

    def test_ranking_keeps_features(self, table):
        X = load_for_ranking(table)
        for c in ("M", "N", "K", "tile_m", "waves", "ext_stages",
                  "split_k", "launchable", "pipeline_kind"):
            assert c in X.columns

    def test_derived_answers_are_removed_too(self, table):
        """time_ms 만 빼는 것이 가장 흔한 사고다. 유도값도 정답이다."""
        X = load_for_ranking(table)
        for c in ("tflops", "frac_of_peak", "vs_cublas", "cublas_ms"):
            assert c not in X.columns

    def test_scoring_has_answers_but_no_features(self, table):
        y = load_for_scoring(table)
        assert "time_ms" in y.columns and "cublas_ms" in y.columns
        # 피처를 함께 주지 않는다 — 그대로 규칙에 넘기는 사고를 막는다
        for c in ("waves", "tile_m", "ext_stages", "arith_intensity"):
            assert c not in y.columns

    def test_scoring_keeps_join_keys(self, table):
        y = load_for_scoring(table)
        for c in ("kernel_id", "M", "N", "K", "split_k", "split_k_mode"):
            assert c in y.columns

    def test_ranking_and_scoring_join(self, table):
        X, y = load_for_ranking(table), load_for_scoring(table)
        keys = ["kernel_id", "M", "N", "K", "split_k", "split_k_mode"]
        merged = X.merge(y, on=keys, how="inner")
        assert len(merged) == len(X) > 0


class TestAssertNoAnswers:
    def test_catches_leak(self, table):
        y = load_for_scoring(table)
        with pytest.raises(AnswerLeakError, match="정답 컬럼"):
            assert_no_answers(y)

    def test_passes_clean(self, table):
        assert assert_no_answers(load_for_ranking(table)) is None

    def test_catches_manual_readparquet(self, table):
        """로더를 우회한 경우를 잡는 것이 이 함수의 존재 이유다."""
        df = pd.read_parquet(table).drop(columns=["time_ms"])
        with pytest.raises(AnswerLeakError):
            assert_no_answers(df, where="직접 읽은 표")


class TestEnvHash:
    def test_mixed_conditions_raise_without_env_hash(self, mixed_table):
        with pytest.raises(AnswerLeakError, match="측정 조건이 2 종"):
            load_for_ranking(mixed_table)

    def test_explicit_env_hash_filters(self, mixed_table):
        X = load_for_ranking(mixed_table, env_hash="aaaa")
        assert set(X["env_hash"].unique()) == {"aaaa1111"}

    def test_single_condition_needs_no_env_hash(self, table):
        assert len(load_for_ranking(table)) > 0


class TestOkOnly:
    def test_default_filters_failures(self, table):
        X = load_for_ranking(table)
        assert len(X) == 3 * 4          # config 5개 중 1개가 numerical_fail

    def test_can_keep_failures(self, table):
        X = load_for_ranking(table, ok_only=False, keep_outcomes=True)
        assert len(X) == 3 * 5
        assert set(X["status"].unique()) == {"ok", "numerical_fail"}


class TestUnknownColumnGuard:
    """★ export.py 가 새 컬럼을 추가했는데 분류를 안 하면 조용히 샌다.

    제거 목록만 관리해서는 그 구멍을 막을 수 없다. 허용 목록과 함께 두고
    미분류 컬럼을 감지한다.
    """

    def _with_extra(self, tmp_path, name, value=1.0):
        rows = _rows()
        for r in rows:
            r[name] = value
        p = tmp_path / "extra.parquet"
        pq.write_table(pa.Table.from_pylist(rows), p)
        return p

    def test_unknown_column_warns(self, tmp_path):
        p = self._with_extra(tmp_path, "time_p99_ms")
        with pytest.warns(UserWarning, match="분류되지 않은 컬럼"):
            load_for_ranking(p)

    def test_unknown_column_is_dropped(self, tmp_path):
        """경고만 하고 노출하면 의미가 없다. 반드시 빠져야 한다."""
        p = self._with_extra(tmp_path, "time_p99_ms")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = load_for_ranking(p)
        assert "time_p99_ms" not in X.columns

    def test_can_escalate_to_error(self, tmp_path):
        p = self._with_extra(tmp_path, "secretly_the_answer")
        with pytest.raises(AnswerLeakError, match="분류되지 않은 컬럼"):
            load_for_ranking(p, unknown_columns="raise")

    def test_ignore_still_drops(self, tmp_path):
        p = self._with_extra(tmp_path, "whatever")
        X = load_for_ranking(p, unknown_columns="ignore")
        assert "whatever" not in X.columns

    def test_known_columns_do_not_warn(self, table):
        with warnings.catch_warnings():
            warnings.simplefilter("error")     # 경고가 나면 실패
            load_for_ranking(table)


class TestClassifyColumns:
    def test_partitions_are_disjoint(self):
        sets = [set(ANSWER_COLS), set(OUTCOME_COLS),
                set(KNOWN_FEATURE_COLS), set(SAFE_META_COLS)]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                assert not sets[i] & sets[j], (
                    f"컬럼이 두 분류에 동시에 있다: {sets[i] & sets[j]}")

    def test_classify_covers_synthetic_table(self, table):
        c = classify_columns(pd.read_parquet(table).columns)
        assert c["unknown"] == []
        assert set(c["answer"]) == set(ANSWER_COLS)
