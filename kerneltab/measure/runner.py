"""커널 .so / libkt_ctx.so 의 ctypes 바인딩.

Python 은 오케스트레이션만 한다. 타이밍 루프는 전부 libkt_ctx.so 안에 있어서
인터프리터 오버헤드가 측정에 섞이지 않는다.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

#: `kt_abi.h` 의 KT_ABI_VERSION 과 같아야 한다. 구조체를 고칠 때 함께 올린다.
KT_ABI_VERSION = 2

__all__ = [
    "DEFAULT_PROTOCOL",
    "PROTOCOL_DEFAULTS",
    "SEGMENT_DEFAULTS",
    "SOAK_DEFAULTS",
    "Ctx",
    "Kernel",
    "KtBuffersC",
    "KtMeasureC",
    "KtProblemC",
    "KtProtocolC",
    "protocol_from_env",
]


class KtProblemC(ctypes.Structure):
    _fields_ = [
        ("M", ctypes.c_int), ("N", ctypes.c_int), ("K", ctypes.c_int),
        ("split_k", ctypes.c_int), ("split_k_mode", ctypes.c_int),
    ]


class KtBuffersC(ctypes.Structure):
    _fields_ = [
        ("A", ctypes.c_void_p), ("B", ctypes.c_void_p),
        ("C", ctypes.c_void_p), ("D", ctypes.c_void_p),
        ("workspace", ctypes.c_void_p), ("workspace_bytes", ctypes.c_size_t),
    ]


class KtProtocolC(ctypes.Structure):
    _fields_ = [
        ("target_ms", ctypes.c_double),
        ("min_total_ms", ctypes.c_double),
        ("min_reps_floor", ctypes.c_int),
        ("min_reps_cap", ctypes.c_int),
        ("max_reps", ctypes.c_int),
        ("warmup_frac", ctypes.c_double),
        ("min_warmup", ctypes.c_int),
        ("iqr_k", ctypes.c_double),
        # ⚠️ 필드 순서와 타입이 kt_abi.h 의 KtProtocol 과 **정확히** 같아야
        #    한다. 어긋나면 조용히 쓰레기 값이 들어간다 — 예외가 안 난다.
        #    tests/test_protocol.py 가 sizeof 와 offset 을 고정한다.
        ("probe_budget_ms", ctypes.c_double),
        ("warmup_budget_ms", ctypes.c_double),
        ("warmup_reps_floor", ctypes.c_int),
    ]


class KtMeasureC(ctypes.Structure):
    _fields_ = [
        ("time_ms", ctypes.c_double),
        ("time_std_ms", ctypes.c_double),
        ("time_min_ms", ctypes.c_double),
        ("time_max_ms", ctypes.c_double),
        ("n_reps", ctypes.c_int),
        ("n_kept", ctypes.c_int),
        ("outlier_frac", ctypes.c_double),
        # kt_abi.h 의 KtMeasure 와 순서·타입이 정확히 같아야 한다.
        ("n_probe", ctypes.c_int),
        ("n_warmup", ctypes.c_int),
        ("overhead_ms", ctypes.c_double),
    ]




#: 열평형 소킹 파라미터. **측정 조건의 일부이므로 env.json 에 기록되어
#: env_hash 에 반영된다.** 소킹 없이 잰 데이터와 소킹 후 데이터는 서로 다른
#: 조건이며 절대 섞으면 안 된다 (2026-08-16 열 램프 구간 폐기 사건 참조).
SOAK_DEFAULTS = {
    # 기본 False. A6000 에서 열 가설이 기각됐다 (69도/230W 소킹 후 -0.07%).
    # 코드를 남기는 이유는 다른 GPU 에서는 다를 수 있어서다.
    # docs/measurement_drift.md
    "enabled": False,
    "probe_interval_s": 300,
    "stable_span": 0.003,
    "stable_runs": 3,
    "min_seconds": 45 * 60,
    "max_seconds": 180 * 60,
}

#: 드리프트 대책. 한 프로세스가 로드하는 서로 다른 커널 수를 묶고, 세그먼트를
#: 라운드 로빈으로 돈다. 드리프트의 유일한 설명 변수가 이 수이기 때문이다
#: (docs/measurement_drift.md). 측정 조건이므로 env_hash 에 들어간다.
SEGMENT_DEFAULTS = {
    "kernels": 500,          # 세그먼트당 커널 수
    "seconds": 2700,         # 한 슬라이스의 **시간 상한** (진행 배분은 작업 수)
    "anchor_kernels": 6,     # 모든 세그먼트에서 재는 고정 커널
    # 슬라이스마다 프로세스를 새로 띄우므로 **매번 워밍업이 필요하다.**
    # 측정 조건이므로 env_hash 에 들어간다 — 워밍업 유무가 다른 데이터를
    # 섞으면 안 된다.
    "warmup_seconds": 20,
}

PROTOCOL_DEFAULTS = {
    "target_ms": 20.0, "min_total_ms": 3.0, "min_reps_floor": 5,
    "min_reps_cap": 30, "max_reps": 1000, "warmup_frac": 0.2,
    "min_warmup": 10, "iqr_k": 1.5,
    # 시간 예산 — **상한**이다. 하한은 위 min_warmup/warmup_frac 이 그대로
    # 유지되므로 짧은 커널은 영향이 없다 (docs/next_campaign.md 5절).
    "probe_budget_ms": 5.0, "warmup_budget_ms": 20.0, "warmup_reps_floor": 3,
}

#: 측정 프로토콜 기본값. 총 20ms 목표, 최소 3ms 보장, 반복 5~1000회.
#: 실제 값은 env.json 의 `protocol` 에서 읽는다 (측정 조건의 일부이므로
#: env_hash 에 포함되어야 한다). protocol_from_env() 를 쓸 것.
DEFAULT_PROTOCOL = KtProtocolC(**PROTOCOL_DEFAULTS)


def protocol_from_env(env: dict) -> KtProtocolC:
    """env.json 의 protocol 로 KtProtocol 을 만든다.

    프로토콜은 측정 조건이다. env.json 에 기록해 env_hash 에 반영해야
    프로토콜이 바뀐 뒤의 측정이 예전 것을 건너뛰지 않는다.
    """
    d = dict(PROTOCOL_DEFAULTS)
    d.update(env.get("protocol") or {})
    return KtProtocolC(**{k: d[k] for k in PROTOCOL_DEFAULTS})


class Ctx:
    """libkt_ctx.so — 버퍼, cuBLAS 참조, 측정 프로토콜."""

    def __init__(self, so_path: str | Path, device: int = 0):
        self.lib = ctypes.CDLL(str(so_path))
        L = self.lib
        L.kt_ctx_create.argtypes = [ctypes.c_int]
        L.kt_ctx_create.restype = ctypes.c_void_p
        L.kt_ctx_destroy.argtypes = [ctypes.c_void_p]
        L.kt_ctx_prepare_problem.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 3
        L.kt_ctx_ensure_workspace.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        L.kt_ctx_buffers.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(KtBuffersC), ctypes.c_int]
        L.kt_ctx_zero_d.argtypes = [ctypes.c_void_p]
        L.kt_ctx_measure.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(KtProtocolC), ctypes.POINTER(KtMeasureC)]
        L.kt_ctx_measure_cublas.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(KtProtocolC),
            ctypes.POINTER(KtMeasureC)]
        L.kt_ctx_run_once.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        L.kt_ctx_max_rel_error.argtypes = [ctypes.c_void_p]
        L.kt_ctx_max_rel_error.restype = ctypes.c_double
        L.kt_ctx_ref_absmax.argtypes = [ctypes.c_void_p]
        L.kt_ctx_ref_absmax.restype = ctypes.c_double
        L.kt_ctx_last_error.argtypes = [ctypes.c_void_p]
        L.kt_ctx_last_error.restype = ctypes.c_char_p

        self._check_abi(so_path)

        self.h = L.kt_ctx_create(device)
        if not self.h:
            raise RuntimeError("kt_ctx_create 실패 (GPU 사용 가능한지 확인)")
        self._mnk: tuple[int, int, int] | None = None
        self.proto = DEFAULT_PROTOCOL

    #: `kt_abi.h` 의 KtAbiStruct 와 같은 순서.
    _ABI_STRUCTS = ((0, KtProblemC), (1, KtBuffersC),
                    (2, KtProtocolC), (3, KtMeasureC))

    def _check_abi(self, so_path) -> None:
        """`.so` 가 지금 헤더로 빌드된 것인지 확인한다.

        ⛔ 이것이 없으면 **옛 `.so` 가 조용히 붙는다.** ctypes 는 심볼만
        맞으면 되고, 구조체가 커진 만큼은 그냥 0 으로 남는다. 실제로
        워밍업 시간 예산을 넣고 `n_warmup` 이 전부 0 으로 나왔는데 원인이
        볼륨에 캐싱된 옛 `.so` 였다. 아무 오류도 나지 않았다.
        """
        L = self.lib
        if not hasattr(L, "kt_abi_version"):
            raise RuntimeError(
                f"{so_path} 에 kt_abi_version 이 없다 — **옛 빌드**다.\n"
                "  이 상태로 돌리면 새 프로토콜 필드가 조용히 무시된다.\n"
                "  지우고 다시 빌드하라: rm <artifacts>/libkt_ctx.so")
        L.kt_abi_version.restype = ctypes.c_int
        L.kt_abi_sizeof.argtypes = [ctypes.c_int]
        L.kt_abi_sizeof.restype = ctypes.c_int
        got = L.kt_abi_version()
        if got != KT_ABI_VERSION:
            raise RuntimeError(
                f"{so_path} 의 ABI 버전이 {got}, 코드는 {KT_ABI_VERSION} 이다.\n"
                "  구조체가 바뀌었는데 .so 가 옛 것이다. 다시 빌드하라.")
        bad = []
        for which, pytype in self._ABI_STRUCTS:
            c_size = L.kt_abi_sizeof(which)
            py_size = ctypes.sizeof(pytype)
            if c_size != py_size:
                bad.append(f"{pytype.__name__}: .so {c_size} vs python {py_size}")
        if bad:
            raise RuntimeError(
                "구조체 크기가 어긋난다 — 필드가 밀려 **쓰레기 값**이 들어간다.\n"
                "  " + "\n  ".join(bad))

    def set_protocol(self, env: dict) -> None:
        """env.json 의 protocol 을 이 컨텍스트의 기본값으로 삼는다."""
        self.proto = protocol_from_env(env)

    # -- 문제 준비 --------------------------------------------------------
    def prepare_problem(self, M: int, N: int, K: int) -> None:
        if self._mnk == (M, N, K):
            return
        rc = self.lib.kt_ctx_prepare_problem(self.h, M, N, K)
        if rc != 0:
            raise RuntimeError(f"prepare_problem({M},{N},{K}): {self.last_error()}")
        self._mnk = (M, N, K)

    def buffers(self, workspace_bytes: int, parallel: bool) -> KtBuffersC:
        if workspace_bytes:
            rc = self.lib.kt_ctx_ensure_workspace(self.h, workspace_bytes)
            if rc != 0:
                raise MemoryError(f"workspace {workspace_bytes}B: {self.last_error()}")
        b = KtBuffersC()
        self.lib.kt_ctx_buffers(self.h, ctypes.byref(b), 1 if parallel else 0)
        b.workspace_bytes = workspace_bytes
        return b

    # -- 측정 -------------------------------------------------------------
    def measure(self, launch_addr: int, handle: int, reduce_slices: int,
                proto: KtProtocolC | None = None) -> tuple[int, KtMeasureC]:
        proto = proto or self.proto
        m = KtMeasureC()
        rc = self.lib.kt_ctx_measure(
            self.h, ctypes.c_void_p(launch_addr), ctypes.c_void_p(handle),
            reduce_slices, ctypes.byref(proto), ctypes.byref(m))
        return rc, m

    def measure_cublas(self, proto: KtProtocolC | None = None):
        proto = proto or self.proto
        m = KtMeasureC()
        rc = self.lib.kt_ctx_measure_cublas(self.h, ctypes.byref(proto),
                                            ctypes.byref(m))
        return rc, m

    def run_once(self, launch_addr: int, handle: int, reduce_slices: int) -> int:
        return self.lib.kt_ctx_run_once(
            self.h, ctypes.c_void_p(launch_addr), ctypes.c_void_p(handle),
            reduce_slices)

    def max_rel_error(self) -> float:
        return self.lib.kt_ctx_max_rel_error(self.h)

    def ref_absmax(self) -> float:
        return self.lib.kt_ctx_ref_absmax(self.h)

    def last_error(self) -> str:
        p = self.lib.kt_ctx_last_error(self.h)
        return p.decode() if p else ""

    def close(self) -> None:
        if getattr(self, "h", None):
            self.lib.kt_ctx_destroy(self.h)
            self.h = None


class Kernel:
    """커널 .so 하나."""

    def __init__(self, so_path: str | Path):
        self.path = str(so_path)
        self.lib = ctypes.CDLL(self.path)
        L = self.lib
        L.kt_workspace_bytes.argtypes = [ctypes.POINTER(KtProblemC)]
        L.kt_workspace_bytes.restype = ctypes.c_size_t
        L.kt_grid_k.argtypes = [ctypes.POINTER(KtProblemC)]
        L.kt_grid_k.restype = ctypes.c_int
        L.kt_grid_shape.argtypes = [
            ctypes.POINTER(KtProblemC), ctypes.POINTER(ctypes.c_int * 3)]
        L.kt_tiled_shape.argtypes = [
            ctypes.POINTER(KtProblemC), ctypes.POINTER(ctypes.c_int * 3)]
        L.kt_can_implement.argtypes = [ctypes.POINTER(KtProblemC)]
        L.kt_can_implement.restype = ctypes.c_int
        L.kt_prepare.argtypes = [
            ctypes.POINTER(KtProblemC), ctypes.POINTER(KtBuffersC),
            ctypes.POINTER(ctypes.c_void_p)]
        L.kt_prepare.restype = ctypes.c_int
        L.kt_release.argtypes = [ctypes.c_void_p]
        L.kt_status_string.argtypes = [ctypes.c_int]
        L.kt_status_string.restype = ctypes.c_char_p
        self.launch_addr = ctypes.cast(L.kt_launch, ctypes.c_void_p).value

    def workspace_bytes(self, p: KtProblemC) -> int:
        return self.lib.kt_workspace_bytes(ctypes.byref(p))

    def grid_k(self, p: KtProblemC) -> int:
        return self.lib.kt_grid_k(ctypes.byref(p))

    def grid_shape(self, p: KtProblemC) -> tuple[int, int, int]:
        a = (ctypes.c_int * 3)()
        self.lib.kt_grid_shape(ctypes.byref(p), ctypes.byref(a))
        return (a[0], a[1], a[2])

    def tiled_shape(self, p: KtProblemC) -> tuple[int, int, int]:
        a = (ctypes.c_int * 3)()
        self.lib.kt_tiled_shape(ctypes.byref(p), ctypes.byref(a))
        return (a[0], a[1], a[2])

    def can_implement(self, p: KtProblemC) -> int:
        return self.lib.kt_can_implement(ctypes.byref(p))

    def status_string(self, st: int) -> str:
        s = self.lib.kt_status_string(st)
        return s.decode() if s else str(st)

    def prepare(self, p: KtProblemC, b: KtBuffersC) -> tuple[int, int]:
        h = ctypes.c_void_p()
        rc = self.lib.kt_prepare(ctypes.byref(p), ctypes.byref(b),
                                 ctypes.byref(h))
        return rc, (h.value or 0)

    def release(self, handle: int) -> None:
        if handle:
            self.lib.kt_release(ctypes.c_void_p(handle))
