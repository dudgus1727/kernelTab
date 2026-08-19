"""R-4b — NVML 조회 실패를 **세어서 보고**하는가.

조용히 `None` 을 넣으면 `sm_clock_mhz` / `mem_clock_mhz` 가 결측으로
기록되고, 33시간 뒤에 "조건이 유지됐는가" 를 확인할 근거가 사라진다.
`docs/decisions.md` 14번의 두 번째 선택지(기록에 남긴다).
"""
from __future__ import annotations

import sys
import types

import pytest


class FakeNVMLError(Exception):
    pass


def _fake_pynvml(fail: set[str]):
    """지정한 항목만 실패하는 가짜 pynvml."""
    m = types.SimpleNamespace()
    m.NVMLError = FakeNVMLError
    m.NVML_CLOCK_SM, m.NVML_CLOCK_MEM, m.NVML_TEMPERATURE_GPU = 0, 1, 0
    m.nvmlInit = lambda: None
    m.nvmlShutdown = lambda: None
    m.nvmlDeviceGetHandleByIndex = lambda i: "H"
    m.nvmlDeviceGetHandleByUUID = lambda u: "H"

    def clock(h, which):
        name = "sm_clock" if which == 0 else "mem_clock"
        if name in fail:
            raise FakeNVMLError(f"{name} 실패")
        return 1350 if which == 0 else 7601

    m.nvmlDeviceGetClockInfo = clock
    m.nvmlDeviceGetTemperature = lambda h, w: (_ for _ in ()).throw(
        FakeNVMLError("temp")) if "temp" in fail else 55
    m.nvmlDeviceGetPowerUsage = lambda h: (_ for _ in ()).throw(
        FakeNVMLError("power")) if "power" in fail else 150000
    m.nvmlDeviceGetCurrentClocksThrottleReasons = lambda h: 0
    m.nvmlDeviceGetCurrentClocksEventReasons = lambda h: 0
    return m


@pytest.fixture
def probe_factory(monkeypatch):
    def make(fail=frozenset(), uuid=None):
        monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml(set(fail)))
        from measure.gpu_state import NvmlProbe
        return NvmlProbe(uuid=uuid)
    return make


class TestFailureCounting:
    def test_clean_run_reports_no_failure(self, probe_factory):
        p = probe_factory()
        for _ in range(5):
            p.snapshot()
        assert p.failures()["n_failures"] == {}
        assert "실패 없음" in p.report()

    def test_counts_per_field(self, probe_factory):
        p = probe_factory(fail={"sm_clock", "power"})
        for _ in range(7):
            s = p.snapshot()
        assert s["sm_clock_mhz"] is None
        assert s["mem_clock_mhz"] == 7601        # 나머지는 계속 읽힌다
        f = p.failures()
        assert f["n_failures"] == {"sm_clock": 7, "power": 7}
        assert f["n_snapshots"] == 7

    def test_report_is_loud(self, probe_factory):
        """★ '12,043회' 처럼 즉시 눈에 띄어야 한다."""
        p = probe_factory(fail={"mem_clock"})
        for _ in range(3):
            p.snapshot()
        r = p.report()
        assert "!!" in r
        assert "mem_clock" in r
        assert "3" in r
        assert "근거가 없다" in r

    def test_first_error_is_kept(self, probe_factory):
        p = probe_factory(fail={"temp"})
        p.snapshot()
        assert "temp" in p.failures()["first_error"]["temp"]


class TestUuidStrictness:
    def test_uuid_failure_raises_not_falls_back(self, monkeypatch):
        """★ 인덱스로 폴백하면 **다른 GPU** 의 클럭을 기록한다 (P-2 와 같은 문제)."""
        m = _fake_pynvml(set())

        def boom(u):
            raise FakeNVMLError("no such uuid")
        m.nvmlDeviceGetHandleByUUID = boom
        monkeypatch.setitem(sys.modules, "pynvml", m)
        from measure.gpu_state import NvmlProbe
        with pytest.raises(RuntimeError) as e:
            NvmlProbe(uuid="GPU-없음")
        assert "폴백하지 않는다" in str(e.value)


class TestWiring:
    def test_rehearse_reports_nvml_failures(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent
               / "scripts" / "rehearse.py").read_text()
        assert "probe.report()" in src
        assert "probe.failures()" in src
