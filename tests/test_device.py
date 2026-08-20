"""P-2 — GPU 선택이 UUID 기반인가, 그리고 **틀리면 시끄럽게 실패하는가**.

인덱스를 그대로 믿으면 컨테이너(`--gpus '"device=3"'` -> 안에서는 0)나
`CUDA_VISIBLE_DEVICES` 재배치 상황에서 **다른 GPU 를 측정**한다.
그건 조용히 틀린 데이터를 만든다.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kerneltab.core import device

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def fake_devices(monkeypatch):
    devs = [(0, "GPU-aaaa", "A"), (1, "GPU-bbbb", "B"), (3, "GPU-cccc", "C")]
    monkeypatch.setattr(device, "list_devices", lambda: devs)
    return devs


class TestIndexOfUuid:
    def test_finds_by_uuid_not_position(self, fake_devices):
        assert device.index_of_uuid("GPU-cccc") == 3

    def test_missing_uuid_raises(self, fake_devices):
        """★ 조용히 0 으로 폴백하면 다른 GPU 를 측정하고도 모른다."""
        with pytest.raises(device.DeviceNotFoundError) as e:
            device.index_of_uuid("GPU-zzzz")
        msg = str(e.value)
        assert "GPU-zzzz" in msg
        assert "GPU-aaaa" in msg, "보이는 GPU 목록을 알려줘야 고칠 수 있다"

    def test_empty_uuid_raises(self, fake_devices):
        with pytest.raises(device.DeviceNotFoundError):
            device.index_of_uuid("")


class TestResolveDevice:
    def test_sets_visible_devices_from_uuid(self, fake_devices, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        idx, uuid = device.resolve_device(
            {"hardware_extra": {"uuid": "GPU-cccc"}})
        assert idx == 3
        import os
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"

    def test_respects_preset_visible_devices(self, fake_devices, monkeypatch):
        """호출자가 좁혀 놓은 것을 스크립트가 되돌리면 안 된다."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        monkeypatch.setattr("kerneltab.core.hardware.device_uuid", lambda i=0: "GPU-bbbb")
        idx, uuid = device.resolve_device(
            {"hardware_extra": {"uuid": "GPU-bbbb"}})
        assert idx == 0                       # 보이는 것 중 0 번
        import os
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"   # 안 건드렸다

    def test_preset_pointing_at_wrong_gpu_raises(self, fake_devices, monkeypatch):
        """★ 미리 설정된 값이 **다른 GPU** 를 가리키면 측정하면 안 된다."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        monkeypatch.setattr("kerneltab.core.hardware.device_uuid", lambda i=0: "GPU-aaaa")
        with pytest.raises(device.DeviceNotFoundError) as e:
            device.resolve_device({"hardware_extra": {"uuid": "GPU-cccc"}})
        assert "다른 GPU" in str(e.value)

    def test_missing_uuid_in_env_raises(self, fake_devices, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        with pytest.raises(device.DeviceNotFoundError):
            device.resolve_device({"device_index": 3})


class TestSmiTarget:
    def test_prefers_uuid(self):
        assert device.smi_target(
            {"hardware_extra": {"uuid": "GPU-cccc"}, "device_index": 9}) == "GPU-cccc"

    def test_falls_back_to_index(self):
        assert device.smi_target({"device_index": 2}) == "2"


class TestCallers:
    """스크립트가 저장된 인덱스를 직접 쓰면 컨테이너에서 깨진다."""

    def test_no_script_sets_visible_from_device_index(self):
        offenders = []
        for f in sorted((REPO / "scripts").glob("*.py")):
            if f.name == "phase0_env.py":
                continue          # 여기서 --device 로 처음 정한다
            src = f.read_text()
            if 'os.environ["CUDA_VISIBLE_DEVICES"] = str(env["device_index"])' in src:
                offenders.append(f.name)
        assert not offenders, f"저장된 인덱스를 그대로 쓰는 곳: {offenders}"

    def test_telemetry_uses_uuid(self):
        src = (REPO / "scripts" / "rehearse.py").read_text()
        assert "start_telemetry(device.smi_target(env))" in src
