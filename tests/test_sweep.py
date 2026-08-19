"""R-2 / R-3 — 스윕 진입점의 인터페이스와 재개 상태.

R-2: 사람용 stdout 을 파싱하면 문구를 다듬는 순간 진입점이 깨진다.
R-3: 재개 시 라운드가 0 부터 다시 시작하면 셔플 순서가 반복된다.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("sw", REPO / "scripts" / "sweep.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "LOG", tmp_path / "sweep.jsonl")
    return m


def _write(log: Path, rows: list[dict]) -> None:
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


class TestResumeState:
    def test_no_log_is_fresh_start(self, sweep):
        assert sweep.resume_state("aaaa1111", 13) == (set(), 0)

    def test_restores_round_and_done(self, sweep):
        _write(sweep.LOG, [
            {"event": "sweep_start", "env_hash": "aaaa1111"},
            {"event": "slice", "round": 0, "segment": 0, "rc": 7},
            {"event": "slice", "round": 0, "segment": 1, "rc": 0},
            {"event": "slice", "round": 1, "segment": 0, "rc": 0},
        ])
        done, rnd = sweep.resume_state("aaaa1111", 3)
        assert done == {0, 1}
        assert rnd == 1, "마지막 라운드부터 이어야 셔플 순서가 반복되지 않는다"

    def test_ignores_other_env_hash(self, sweep):
        """★ 이전 캠페인의 로그가 같은 파일에 남아 있다 (R-5 와 같은 원칙)."""
        _write(sweep.LOG, [
            {"event": "sweep_start", "env_hash": "bbbb2222"},
            {"event": "slice", "round": 5, "segment": 7, "rc": 0},
            {"event": "sweep_start", "env_hash": "aaaa1111"},
            {"event": "slice", "round": 0, "segment": 1, "rc": 0},
        ])
        done, rnd = sweep.resume_state("aaaa1111", 13)
        assert done == {1}, f"다른 조건의 세그먼트가 섞였다: {done}"
        assert rnd == 0

    def test_only_rc_done_counts_as_finished(self, sweep):
        """rc=7 은 '예산 소진, 다음으로' 다. 완료가 아니다."""
        _write(sweep.LOG, [
            {"event": "sweep_start", "env_hash": "aaaa1111"},
            {"event": "slice", "round": 0, "segment": 2, "rc": 7},
        ])
        done, _ = sweep.resume_state("aaaa1111", 13)
        assert done == set()

    def test_tolerates_corrupt_lines(self, sweep):
        sweep.LOG.write_text(
            json.dumps({"event": "sweep_start", "env_hash": "aaaa1111"}) + "\n"
            + "{쓰다 만 줄\n"
            + json.dumps({"event": "slice", "round": 2, "segment": 0, "rc": 0}) + "\n")
        done, rnd = sweep.resume_state("aaaa1111", 13)
        assert done == {0} and rnd == 2


class TestJsonInterface:
    """R-2 — sweep 은 사람용 출력을 파싱하지 않는다."""

    def test_sweep_does_not_parse_human_output(self):
        src = (REPO / "scripts" / "sweep.py").read_text()
        for bad in ('startswith("세그먼트 ")', 'startswith("작업 수:")',
                    'startswith("SEGJOBS ")'):
            assert bad not in src, f"사람용 출력을 파싱한다: {bad}"
        assert 'startswith("JSON ")' in src

    def test_rehearse_emits_json(self):
        src = (REPO / "scripts" / "rehearse.py").read_text()
        assert '"env_hash_v2"' in src and 'print("JSON "' in src

    def test_plan_rejects_env_hash_mismatch(self, sweep, monkeypatch):
        """★ 조건이 어긋난 채 스윕이 시작되면 안 된다."""
        class R:
            returncode = 0
            stdout = "JSON " + json.dumps({
                "n_segments": 2, "n_jobs": 10, "segment_kernels": 5,
                "jobs_per_segment": {"0": 5, "1": 5}, "anchors": [],
                "env_hash": "aaaa1111", "env_hash_v2": "vvvv1111"})
            stderr = ""
        monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **k: R())
        # 같으면 통과
        ok = sweep.segment_plan(5, {"env_hash": "aaaa1111", "env_hash_v2": "vvvv1111"})
        assert ok["n_segments"] == 2
        # v2 가 다르면 거부
        with pytest.raises(SystemExit) as e:
            sweep.segment_plan(5, {"env_hash": "aaaa1111", "env_hash_v2": "다름"})
        assert "env_hash_v2" in str(e.value)
        # 구 해시(재개 키)가 달라도 거부
        with pytest.raises(SystemExit):
            sweep.segment_plan(5, {"env_hash": "다름", "env_hash_v2": "vvvv1111"})

    def test_plan_fails_loudly_without_json(self, sweep, monkeypatch):
        class R:
            returncode = 0
            stdout = "세그먼트 13개 x 커널 500개\n작업 수: 980,915\n"
            stderr = ""
        monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **k: R())
        with pytest.raises(SystemExit) as e:
            sweep.segment_plan(500, {"env_hash": "a"})
        assert "JSON" in str(e.value)
