"""R-5 — `env_hash` 격리가 **구조로** 강제되는가.

교훈이 문서에만 있으면 지켜지지 않는다. 이 저장소가 같은 함정을 다섯 번
밟았고, `docs/decisions.md` 13번에 적어 둔 다음에도 두 번 더 밟았다.
그래서 테스트로 고정한다.
"""
from __future__ import annotations

import json
import statistics

import pytest

from core.records import (ALL, EnvHashError, aggregate_per_env, env_hashes,
                          iter_records, load_records)


@pytest.fixture
def mixed(tmp_path):
    """조건 두 종이 섞인 jsonl. 느린 쪽(폐기 구간)과 빠른 쪽."""
    p = tmp_path / "results.jsonl"
    rows = []
    for i in range(10):                       # 정상 조건: 1.0 근처
        rows.append({"env_hash": "aaaa1111" + "0" * 56, "kernel_id": f"k{i}",
                     "M": 512, "N": 512, "K": 512, "status": "ok",
                     "time_ms": 1.0 + i * 0.01})
    for i in range(10):                       # 폐기 조건: 10배 느림
        rows.append({"env_hash": "bbbb2222" + "0" * 56, "kernel_id": f"k{i}",
                     "M": 512, "N": 512, "K": 512, "status": "ok",
                     "time_ms": 10.0 + i * 0.1})
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


class TestRequiredEnvHash:
    def test_missing_env_hash_is_typeerror(self, mixed):
        """기본값이 없어야 한다. 안 쓰면 부르는 순간 죽는다."""
        with pytest.raises(TypeError):
            list(iter_records(mixed))          # type: ignore[call-arg]

    def test_empty_env_hash_rejected(self, mixed):
        """빈 문자열로 전체를 읽는 우회로를 막는다."""
        for bad in ("", None):
            with pytest.raises(EnvHashError):
                list(iter_records(mixed, bad))  # type: ignore[arg-type]

    def test_all_must_be_explicit(self, mixed):
        assert len(load_records(mixed, ALL)) == 20
        assert len(load_records(mixed, "aaaa1111")) == 10

    def test_prefix_match(self, mixed):
        """8자만 줘도 된다 — 실제 코드가 그렇게 쓴다."""
        assert len(load_records(mixed, "aaaa")) == 10
        assert len(load_records(mixed, "aaaa1111" + "0" * 56)) == 10

    def test_unknown_hash_yields_nothing(self, mixed):
        assert load_records(mixed, "cccc3333") == []

    def test_missing_file_is_empty_not_error(self, tmp_path):
        assert load_records(tmp_path / "nope.jsonl", ALL) == []


class TestAggregatePerEnv:
    def test_env_hash_is_forced_into_key(self, mixed):
        """집계 키에 env_hash 가 **강제로** 들어간다. 섞을 수 없다."""
        out = aggregate_per_env(
            load_records(mixed, ALL),
            key_fn=lambda r: (r["M"], r["N"], r["K"]),
            val_fn=lambda r: r["time_ms"],
            agg=lambda v: statistics.median(v) / min(v))
        assert len(out) == 2, "조건 두 종이 각각 따로 집계돼야 한다"
        keys = {k[0][:8] for k in out}
        assert keys == {"aaaa1111", "bbbb2222"}

    def test_mixing_would_have_inflated_the_metric(self, mixed):
        """섞으면 지표가 조용히 부풀어 오른다 — difficulty 22배 사례의 축소판."""
        rows = load_records(mixed, ALL)
        per_env = aggregate_per_env(
            rows, key_fn=lambda r: (r["M"],), val_fn=lambda r: r["time_ms"],
            agg=lambda v: statistics.median(v) / min(v))
        clean = max(per_env.values())
        # 조건을 무시하고 한 덩어리로 집계하면
        allv = [r["time_ms"] for r in rows]
        mixed_val = statistics.median(allv) / min(allv)
        assert clean < 1.2, f"조건별 난이도는 1 근처여야 한다: {clean}"
        assert mixed_val > 4.0, "섞으면 크게 부풀어야 한다 (이 테스트의 전제)"

    def test_min_n_filters_small_groups(self, mixed):
        out = aggregate_per_env(
            load_records(mixed, ALL), key_fn=lambda r: (r["kernel_id"],),
            val_fn=lambda r: r["time_ms"], agg=len, min_n=2)
        assert out == {}, "커널당 조건별 1개뿐이므로 min_n=2 면 전부 걸러진다"


class TestDiagnostics:
    def test_env_hashes_counts(self, mixed):
        h = env_hashes(mixed)
        assert {k[:8] for k in h} == {"aaaa1111", "bbbb2222"}
        assert all(v == 10 for v in h.values())


class TestCallersFilter:
    """실제 호출부가 필터를 쓰는지. 새로 추가한 사람이 빠뜨리면 여기서 걸린다."""

    @pytest.mark.parametrize("path,needle", [
        ("scripts/rehearse.py", "records.iter_records(RESULTS, env_hash)"),
        ("scripts/rehearse.py", "records.iter_records(RESULTS, env[\"env_hash\"])"),
        ("scripts/rehearse.py", "records.load_records(DRIFT, _eh)"),
        ("scripts/recheck_stability.py", "records.load_records(RESULTS, env[\"env_hash\"])"),
        ("scripts/report_phase3.py", "records.iter_records(RESULTS, env_hash)"),
        ("scripts/export.py", "records.aggregate_per_env"),
    ])
    def test_caller_uses_filtered_reader(self, path, needle):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / path).read_text()
        assert needle in src, f"{path} 가 {needle} 를 쓰지 않는다 (R-5)"
