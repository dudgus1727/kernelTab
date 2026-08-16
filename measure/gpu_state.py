"""GPU 클럭 고정 시도와 NVML 기반 상태 스냅샷.

측정 루프 안에서는 절대 nvidia-smi 를 fork 하지 않는다. 프로세스 생성
비용(수십 ms)이 측정 대상(수백 us)보다 크기 때문에 측정 자체를 오염시킨다.
클럭/온도는 pynvml(NVML C API 바인딩)로 읽는다.

nvidia-smi 는 (a) 클럭 고정 시도, (b) 백그라운드 텔레메트리 프로세스 —
둘 다 측정 루프 밖 — 에서만 쓴다.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

__all__ = [
    "ClockLockResult",
    "try_lock_clocks",
    "reset_clocks",
    "NvmlProbe",
    "drift_check_seconds",
]


@dataclass
class ClockLockResult:
    locked: bool
    mhz: int | None
    target_mhz: int | None
    error: str | None


def _smi(args: list[str]) -> tuple[int, str]:
    p = subprocess.run(
        ["nvidia-smi", *args], capture_output=True, text=True, timeout=60
    )
    return p.returncode, (p.stdout + p.stderr).strip()


def _query(device_index: int, field: str) -> str | None:
    rc, out = _smi(
        ["-i", str(device_index), f"--query-gpu={field}", "--format=csv,noheader,nounits"]
    )
    if rc != 0 or not out:
        return None
    return out.splitlines()[0].strip()


def _pick_target_mhz(device_index: int) -> int | None:
    """고정할 SM 클럭 선택.

    GPU별 값을 박지 않는다. 기본 application clock (제조사가 지속 가능하다고
    정의한 클럭)을 우선 쓰고, 없으면 최대 SM 클럭의 85% 를 쓴다.
    """
    for field in ("clocks.default_applications.graphics", "clocks.applications.graphics"):
        v = _query(device_index, field)
        if v and v.replace(".", "").isdigit():
            return int(float(v))
    v = _query(device_index, "clocks.max.sm")
    if v and v.replace(".", "").isdigit():
        return int(float(v) * 0.85)
    return None


def try_lock_clocks(device_index: int = 0, target_mhz: int | None = None) -> ClockLockResult:
    """`nvidia-smi -lgc` 로 클럭 고정을 시도한다.

    컨테이너/클라우드에서는 권한 부족으로 실패하는 것이 정상이다.
    실패해도 측정은 계속하되, 드리프트 점검 주기를 짧게 가져간다.
    """
    tgt = target_mhz or _pick_target_mhz(device_index)
    if tgt is None:
        return ClockLockResult(False, None, None, "고정할 목표 클럭을 조회하지 못했다")

    rc, out = _smi(["-i", str(device_index), "-lgc", f"{tgt},{tgt}"])
    # nvidia-smi 는 권한 실패 시에도 rc=0 을 반환하는 경우가 있어 문구도 본다.
    failed = rc != 0 or "does not have permission" in out or "Terminating early" in out
    if failed:
        return ClockLockResult(False, None, tgt, out or f"rc={rc}")

    cur = _query(device_index, "clocks.current.sm")
    return ClockLockResult(True, int(float(cur)) if cur and cur.replace(".", "").isdigit() else tgt, tgt, None)


def reset_clocks(device_index: int = 0) -> bool:
    rc, out = _smi(["-i", str(device_index), "-rgc"])
    return rc == 0 and "does not have permission" not in out


def drift_check_seconds(clock_locked: bool) -> int:
    """클럭이 고정되지 않았으면 드리프트 점검을 3분 주기로 촘촘히 한다."""
    return 600 if clock_locked else 180


class NvmlProbe:
    """측정 루프 안에서 쓰는 저비용 클럭/온도 스냅샷."""

    def __init__(self, uuid: str | None = None, index: int = 0):
        import pynvml

        self._nvml = pynvml
        pynvml.nvmlInit()
        self.handle = None
        if uuid:
            try:
                self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid.encode())
            except Exception:
                self.handle = None
        if self.handle is None:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)

    def snapshot(self) -> dict:
        n = self._nvml
        h = self.handle
        try:
            sm = n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM)
        except Exception:
            sm = None
        try:
            mem = n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM)
        except Exception:
            mem = None
        try:
            temp = n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = None
        try:
            power = n.nvmlDeviceGetPowerUsage(h) / 1000.0
        except Exception:
            power = None
        try:
            throttle = n.nvmlDeviceGetCurrentClocksThrottleReasons(h)
        except Exception:
            try:
                throttle = n.nvmlDeviceGetCurrentClocksEventReasons(h)
            except Exception:
                throttle = None
        return {
            "sm_clock_mhz": sm,
            "mem_clock_mhz": mem,
            "gpu_temp_c": temp,
            "power_w": power,
            "throttle_reasons": throttle,
        }

    def close(self) -> None:
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass
