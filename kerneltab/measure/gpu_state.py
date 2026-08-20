"""GPU 클럭 고정 시도와 NVML 기반 상태 스냅샷.

측정 루프 안에서는 절대 nvidia-smi 를 fork 하지 않는다. 프로세스 생성
비용(수십 ms)이 측정 대상(수백 us)보다 크기 때문에 측정 자체를 오염시킨다.
클럭/온도는 pynvml(NVML C API 바인딩)로 읽는다.

nvidia-smi 는 (a) 클럭 고정 시도, (b) 백그라운드 텔레메트리 프로세스 —
둘 다 측정 루프 밖 — 에서만 쓴다.
"""

from __future__ import annotations

import subprocess
from collections import Counter
from dataclasses import dataclass

__all__ = [
    "ClockLockResult",
    "NvmlProbe",
    "drift_check_seconds",
    "reset_clocks",
    "try_lock_clocks",
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
    """측정 루프 안에서 쓰는 저비용 클럭/온도 스냅샷.

    ## 실패를 삼키되 **세어서 보고한다** (R-4b)

    NVML 조회는 측정 루프 안에서 초당 여러 번 일어난다. 한 번 실패했다고
    33시간 측정을 멈추는 것은 과하다. 그러나 **조용히 None 을 넣으면**
    `sm_clock_mhz` / `mem_clock_mhz` 가 결측으로 기록되고, 나중에
    "33시간 동안 조건이 유지됐는가" 를 확인할 근거가 사라진다.

    그래서 실패마다 세고 첫 오류를 기억한다. `report()` 를 측정 끝에
    부르면 `NVML 조회 실패 12,043회` 처럼 즉시 드러난다.
    (`docs/decisions.md` 14번 — "할 수 없으면 아무것도 안 한다" 금지,
     실패시키거나 / 기록에 남기거나 / 명시적으로 우회시킨다 중 두 번째)
    """

    def __init__(self, uuid: str | None = None, index: int = 0):
        import pynvml

        self._nvml = pynvml
        self._err = Counter()          # 필드 -> 실패 횟수
        self._first_err: dict[str, str] = {}
        self._n_snap = 0
        pynvml.nvmlInit()

        if uuid:
            # ⛔ 실패해도 인덱스로 폴백하지 않는다. 폴백하면 **다른 GPU 의**
            #    클럭을 측정 대상의 것으로 기록하게 되고, 그건 조용히 틀린
            #    데이터다 (P-2 와 같은 문제).
            try:
                self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid.encode())
            except (pynvml.NVMLError, OSError) as e:
                raise RuntimeError(
                    f"NVML 이 UUID {uuid} 인 GPU 를 찾지 못했다: {e}\n"
                    "  인덱스로 폴백하지 않는다 — 다른 GPU 의 클럭을 측정\n"
                    "  대상의 것으로 기록하게 된다.") from e
        else:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)

    def _try(self, field: str, fn):
        """NVML 한 항목. 실패하면 세고 `None` 을 돌려준다."""
        try:
            return fn()
        except (self._nvml.NVMLError, OSError, ValueError) as e:
            self._err[field] += 1
            self._first_err.setdefault(field, f"{type(e).__name__}: {e}")
            return None

    def snapshot(self) -> dict:
        n, h = self._nvml, self.handle
        self._n_snap += 1

        def throttle():
            try:
                return n.nvmlDeviceGetCurrentClocksThrottleReasons(h)
            except AttributeError:
                # 드라이버/바인딩 버전에 따라 이름이 다르다. 오류가 아니다.
                return n.nvmlDeviceGetCurrentClocksEventReasons(h)

        return {
            "sm_clock_mhz": self._try(
                "sm_clock", lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM)),
            "mem_clock_mhz": self._try(
                "mem_clock", lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM)),
            "gpu_temp_c": self._try(
                "temp", lambda: n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)),
            "power_w": self._try(
                "power", lambda: n.nvmlDeviceGetPowerUsage(h) / 1000.0),
            "throttle_reasons": self._try("throttle", throttle),
        }

    def failures(self) -> dict:
        """조회 실패 집계. 측정 기록에 실어 사후에 걸러낼 수 있게 한다."""
        return {"n_snapshots": self._n_snap,
                "n_failures": dict(self._err),
                "first_error": dict(self._first_err)}

    def report(self) -> str:
        """측정 끝에 한 줄. 실패가 있으면 **크게** 보인다."""
        if not self._err:
            return f"[NVML] 스냅샷 {self._n_snap:,}회, 조회 실패 없음"
        tot = sum(self._err.values())
        lines = [f"[NVML] !! 조회 실패 {tot:,}회 "
                 f"(스냅샷 {self._n_snap:,}회 중)"]
        for k, v in sorted(self._err.items(), key=lambda kv: -kv[1]):
            lines.append(f"         {k:12s} {v:>9,}회  {self._first_err[k][:70]}")
        lines.append("         이 필드들은 결측으로 기록됐다. 그만큼 "
                     "'조건이 유지됐는가' 를 확인할 근거가 없다.")
        return "\n".join(lines)

    def close(self) -> None:
        try:
            self._nvml.nvmlShutdown()
        except (self._nvml.NVMLError, OSError):
            pass          # 종료 시 정리 실패는 무해하다 (이미 다 썼다)
