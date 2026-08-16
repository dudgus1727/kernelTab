"""커널 .so / libkt_ctx.so 의 ctypes 바인딩.

Python 은 오케스트레이션만 한다. 타이밍 루프는 전부 libkt_ctx.so 안에 있어서
인터프리터 오버헤드가 측정에 섞이지 않는다.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

__all__ = [
    "KtProblemC", "KtBuffersC", "KtProtocolC", "KtMeasureC",
    "Ctx", "Kernel", "DEFAULT_PROTOCOL",
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
        ("min_reps", ctypes.c_int),
        ("max_reps", ctypes.c_int),
        ("warmup_frac", ctypes.c_double),
        ("min_warmup", ctypes.c_int),
        ("iqr_k", ctypes.c_double),
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
    ]


#: 스펙의 측정 프로토콜: 총 20ms, 최소 30회, 최대 1000회, 워밍업 20%/최소 10회
DEFAULT_PROTOCOL = KtProtocolC(
    target_ms=20.0, min_reps=30, max_reps=1000,
    warmup_frac=0.2, min_warmup=10, iqr_k=1.5,
)


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

        self.h = L.kt_ctx_create(device)
        if not self.h:
            raise RuntimeError("kt_ctx_create 실패 (GPU 사용 가능한지 확인)")
        self._mnk: tuple[int, int, int] | None = None

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
                proto: KtProtocolC = DEFAULT_PROTOCOL) -> tuple[int, KtMeasureC]:
        m = KtMeasureC()
        rc = self.lib.kt_ctx_measure(
            self.h, ctypes.c_void_p(launch_addr), ctypes.c_void_p(handle),
            reduce_slices, ctypes.byref(proto), ctypes.byref(m))
        return rc, m

    def measure_cublas(self, proto: KtProtocolC = DEFAULT_PROTOCOL):
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
