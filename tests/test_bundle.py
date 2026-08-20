"""번들 로더 — 무결성 검증과 다중 GPU 결합.

합성 번들로만 돌린다. 번들은 kernelrule 에 넘기는 배포 단위이므로,
**손상 감지**와 **공통 형상 필터**가 실제로 동작하는지가 핵심이다.
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("pandas")

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent

from kerneltab.core.bundle import (
    BundleError,
    load_bundle,
    load_bundles,
    resolve_bundle_path,
)
from kerneltab.core.table import ANSWER_COLS


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
              "NVIDIA RTX A6000", 84, [*common, (384, 4096, 4096)])
    b = _make(tmp_path, "rtx-4090-sm_89-bbbb2222", "bbbb2222",
              "NVIDIA GeForce RTX 4090", 128, [*common, (512, 4096, 4096)])
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
        with pytest.raises(BundleError, match=r"sha256 불일치|크기"):
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


# ---------------------------------------------------------------------------
# 라이선스 — 번들은 코드와 분리되어 유통되므로 파일이 조건을 들고 다녀야 한다
# ---------------------------------------------------------------------------


def test_license_defaults_to_cc_by(tmp_path):
    """license 필드가 없는 옛 번들도 CC BY 4.0 으로 읽힌다."""
    d = _make(tmp_path, "gpu-sm_86-aaaa1111", "aaaa1111", "GPU A", 84,
              [(512, 512, 512)])
    info = json.loads((d / "BUNDLE.json").read_text())
    assert "license" not in info          # 옛 번들
    assert load_bundle(d).license == "CC-BY-4.0"


def test_license_is_read_from_bundle_json(tmp_path):
    d = _make(tmp_path, "gpu-sm_86-bbbb2222", "bbbb2222", "GPU B", 84,
              [(512, 512, 512)])
    info = json.loads((d / "BUNDLE.json").read_text())
    info["license"] = "CC-BY-SA-4.0"
    (d / "BUNDLE.json").write_text(json.dumps(info))
    # 체크섬 대상에 BUNDLE.json 자신은 없으므로 verify 를 켜도 통과한다
    assert load_bundle(d).license == "CC-BY-SA-4.0"


def test_bundle_script_separates_tool_and_data_license():
    """도구는 Apache-2.0, 표는 CC BY 4.0. 데이터셋에 소프트웨어 라이선스를
    붙이면 이용자가 오히려 혼란스럽다."""
    from pathlib import Path as _P
    src = (_P(__file__).resolve().parent.parent / "scripts" / "bundle.py").read_text()
    assert '"license": "CC-BY-4.0"' in src
    assert '"tool_license": "Apache-2.0"' in src
    assert 'LICENSE.txt' in src           # 번들 안에 라이선스 파일을 쓴다


# ---------------------------------------------------------------------------
# 배포 경계 — 여러 조건이 섞인 표는 번들이 되면 안 된다
# ---------------------------------------------------------------------------
#
# 실제로 사고가 났다. `results/table.parquet` 은 작업 파일이라 여러 조건을
# 담는데(조건 간 비교/드리프트 분석용), bundle.py 가 그것을 **그대로 복사**해서
# 폐기한 드리프트 데이터 226,145행이 공개 릴리즈에 실렸다.
# BUNDLE.json 의 n_rows 는 필터한 값이라 980,915 로 맞았고, 그래서 아무도
# 못 알아챘다 — 통계와 파일이 어긋난 것이다.


def _mixed_bundle(tmp_path, bundle_id="gpu-sm_86-aaaa1111"):
    """두 조건이 섞인 표를 가진 번들을 만든다."""
    import hashlib

    d = tmp_path / bundle_id
    d.mkdir(parents=True)
    rows = _rows("aaaa1111", [(512, 512, 512)]) + _rows("bbbb2222", [(512, 512, 512)])
    pq.write_table(pa.Table.from_pylist(rows), d / "table.parquet")
    (d / "env.json").write_text(json.dumps({"env_hash": "aaaa1111"}))
    (d / "kernels.jsonl").write_text("")
    (d / "manifest.json").write_text("{}")
    files = {}
    for f in sorted(d.iterdir()):
        files[f.name] = {"bytes": f.stat().st_size,
                         "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    (d / "BUNDLE.json").write_text(json.dumps({
        "bundle_id": bundle_id, "env_hash": "aaaa1111",
        "gpu_name": "G", "arch": "sm_86", "sm_count": 84,
        # ★ 통계는 필터한 값. 파일은 안 걸렀다 — 이 불일치가 버그의 서명이다
        "n_rows": len([r for r in rows if r["env_hash"].startswith("aaaa1111")]),
        "files": files, "shape_layers": {},
    }))
    return d


def test_validate_bundle_rejects_mixed_env(tmp_path):
    """★ 이번 버그를 잡는 검사. 섞인 표로 번들을 만들려 하면 실패해야 한다."""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from validate_table import validate_bundle

    assert validate_bundle(_mixed_bundle(tmp_path)) != 0


def test_validate_bundle_detects_n_rows_mismatch(tmp_path):
    """n_rows 와 실제 행 수의 불일치 — 이 검사만 있었어도 잡혔다."""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from validate_table import validate_bundle

    d = _mixed_bundle(tmp_path, "gpu-sm_86-cccc3333")
    info = json.loads((d / "BUNDLE.json").read_text())
    # 조건은 하나만 남기되 n_rows 를 틀리게 둔다
    t = pq.read_table(d / "table.parquet")
    import pyarrow.compute as pc
    t = t.filter(pc.starts_with(pc.cast(t.column("env_hash"), "string"), "aaaa1111"))
    pq.write_table(t, d / "table.parquet")
    info["n_rows"] = t.num_rows + 999
    (d / "BUNDLE.json").write_text(json.dumps(info))
    assert validate_bundle(d) != 0


def test_validate_bundle_accepts_single_env(tmp_path):
    import hashlib
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from validate_table import validate_bundle

    d = tmp_path / "gpu-sm_86-dddd4444"
    d.mkdir(parents=True)
    rows = _rows("dddd4444", [(512, 512, 512)])
    pq.write_table(pa.Table.from_pylist(rows), d / "table.parquet")
    (d / "env.json").write_text(json.dumps({"env_hash": "dddd4444"}))
    (d / "kernels.jsonl").write_text("")
    (d / "manifest.json").write_text("{}")
    files = {}
    for f in sorted(d.iterdir()):
        files[f.name] = {"bytes": f.stat().st_size,
                         "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    (d / "BUNDLE.json").write_text(json.dumps({
        "bundle_id": "gpu-sm_86-dddd4444", "env_hash": "dddd4444",
        "gpu_name": "G", "arch": "sm_86", "sm_count": 84,
        "n_rows": len(rows), "files": files, "shape_layers": {},
    }))
    assert validate_bundle(d) == 0
