"""번들 로더 — 무결성 검증과 다중 GPU 결합.

합성 번들로만 돌린다. 번들은 kernelrule 에 넘기는 배포 단위이므로,
**손상 감지**와 **공통 형상 필터**가 실제로 동작하는지가 핵심이다.
"""
import json

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("pandas")

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from core.bundle import (  # noqa: E402
    BundleError, load_bundle, load_bundles, resolve_bundle_path,
)
from core.table import ANSWER_COLS, AnswerLeakError  # noqa: E402


def _rows(env_hash, shapes, n_cfg=3):
    out = []
    for (M, N, K) in shapes:
        for ci in range(n_cfg):
            out.append({
                "kernel_id": f"k{ci}", "env_hash": env_hash, "arch": "sm_86",
                "M": M, "N": N, "K": K, "dtype": "f16",
                "tile_m": 128, "tile_n": 128, "tile_k": 32,
                "align_a": 8, "align_b": 8, "align_c": 8,
                "split_k": 1, "split_k_mode": "serial",
                "waves": 1.0 + ci, "status": "ok",
                "time_ms": 1.0 + ci, "cublas_ms": 1.2, "tflops": 90.0,
                "frac_of_peak": 0.8, "vs_cublas": 1.1,
                "time_std_ms": 0.0, "time_min_ms": 1.0, "time_max_ms": 1.0,
                "n_reps": 30, "outlier_frac": 0.0,
            })
    return out


def _make(tmp_path, bundle_id, env_hash, gpu, sm_count, shapes):
    d = tmp_path / bundle_id
    d.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(_rows(env_hash, shapes)),
                   d / "table.parquet")
    (d / "env.json").write_text(json.dumps({"env_hash": env_hash}))
    (d / "kernels.jsonl").write_text("")
    files = {}
    import hashlib
    for f in sorted(d.iterdir()):
        files[f.name] = {"bytes": f.stat().st_size,
                         "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    (d / "BUNDLE.json").write_text(json.dumps({
        "bundle_id": bundle_id, "env_hash": env_hash, "gpu_name": gpu,
        "arch": "sm_86", "sm_count": sm_count,
        "n_rows": len(shapes) * 3, "files": files,
        "shape_layers": {"a_workload": [list(s) for s in shapes]},
    }))
    return d


@pytest.fixture
def two_bundles(tmp_path):
    """형상이 일부만 겹치는 두 GPU. 층 C 가 sm_count 에서 역산되기 때문이다."""
    common = [(1024, 4096, 4096), (4096, 4096, 4096)]
    a = _make(tmp_path, "rtx-a6000-sm_86-aaaa1111", "aaaa1111",
              "NVIDIA RTX A6000", 84, common + [(384, 4096, 4096)])
    b = _make(tmp_path, "rtx-4090-sm_89-bbbb2222", "bbbb2222",
              "NVIDIA GeForce RTX 4090", 128, common + [(512, 4096, 4096)])
    return tmp_path, a, b


class TestResolve:
    def test_direct_path(self, two_bundles):
        _, a, _ = two_bundles
        assert resolve_bundle_path(a) == a.resolve()

    def test_via_env_var(self, two_bundles, monkeypatch):
        root, a, _ = two_bundles
        monkeypatch.setenv("KERNELTAB_DATASETS", str(root))
        assert resolve_bundle_path("rtx-a6000-sm_86-aaaa1111") == a.resolve()

    def test_missing_raises_with_search_path(self, tmp_path):
        with pytest.raises(BundleError, match="탐색"):
            resolve_bundle_path(tmp_path / "nope")


class TestIntegrity:
    def test_loads_clean(self, two_bundles):
        _, a, _ = two_bundles
        b = load_bundle(a)
        assert b.bundle_id == "rtx-a6000-sm_86-aaaa1111"
        assert b.info["sm_count"] == 84

    def test_detects_tampered_table(self, two_bundles):
        """표만 바꿔치기한 것을 조용히 넘기면 안 된다."""
        _, a, _ = two_bundles
        rows = _rows("aaaa1111", [(1, 1, 1)])
        pq.write_table(pa.Table.from_pylist(rows), a / "table.parquet")
        with pytest.raises(BundleError, match="sha256 불일치|크기"):
            load_bundle(a)

    def test_detects_missing_file(self, two_bundles):
        _, a, _ = two_bundles
        (a / "kernels.jsonl").unlink()
        with pytest.raises(BundleError, match="파일 없음"):
            load_bundle(a)

    def test_verify_false_skips(self, two_bundles):
        _, a, _ = two_bundles
        (a / "kernels.jsonl").unlink()
        assert load_bundle(a, verify=False).bundle_id


class TestAnswerRemoval:
    def test_ranking_drops_answers(self, two_bundles):
        _, a, _ = two_bundles
        X = load_bundle(a).ranking()
        assert [c for c in ANSWER_COLS if c in X.columns] == []

    def test_ranking_tags_provenance(self, two_bundles):
        """행마다 어느 GPU 에서 왔는지 있어야 결합 후 구분이 된다."""
        _, a, _ = two_bundles
        X = load_bundle(a).ranking()
        assert set(X["bundle_id"]) == {"rtx-a6000-sm_86-aaaa1111"}
        assert set(X["sm_count"]) == {84}

    def test_scoring_has_answers(self, two_bundles):
        _, a, _ = two_bundles
        y = load_bundle(a).scoring()
        assert "time_ms" in y.columns


class TestCombine:
    def test_union_of_columns(self, two_bundles):
        root, a, b = two_bundles
        df = load_bundles([a, b])
        assert set(df["bundle_id"]) == {"rtx-a6000-sm_86-aaaa1111",
                                        "rtx-4090-sm_89-bbbb2222"}
        assert len(df) == 9 + 9   # 형상 3개 x config 3개 x 번들 2개

    def test_common_shapes_only(self, two_bundles):
        """★ 층 C 는 sm_count 에서 M 을 역산하므로 GPU 마다 형상이 다르다.

        이 필터 없이 전이 실험을 하면 "규칙이 나빠서" 인지 "형상이 달라서"
        인지 구분할 수 없다.
        """
        root, a, b = two_bundles
        df = load_bundles([a, b], common_shapes_only=True)
        shapes = set(map(tuple, df[["M", "N", "K"]].to_numpy().tolist()))
        assert shapes == {(1024, 4096, 4096), (4096, 4096, 4096)}
        assert (384, 4096, 4096) not in shapes   # A6000 전용
        assert (512, 4096, 4096) not in shapes   # 4090 전용
        assert len(df) == 2 * 3 * 2

    def test_no_common_shapes_raises(self, tmp_path):
        a = _make(tmp_path, "x-sm_86-1111", "1111", "X", 84, [(1, 1, 1)])
        b = _make(tmp_path, "y-sm_86-2222", "2222", "Y", 84, [(2, 2, 2)])
        with pytest.raises(BundleError, match="공통 형상"):
            load_bundles([a, b], common_shapes_only=True)

    def test_duplicate_bundle_rejected(self, two_bundles):
        _, a, _ = two_bundles
        with pytest.raises(BundleError, match="중복"):
            load_bundles([a, a])

    def test_combined_ranking_still_has_no_answers(self, two_bundles):
        _, a, b = two_bundles
        df = load_bundles([a, b])
        assert [c for c in ANSWER_COLS if c in df.columns] == []

    def test_scoring_kind(self, two_bundles):
        _, a, b = two_bundles
        df = load_bundles([a, b], kind="scoring")
        assert "time_ms" in df.columns
