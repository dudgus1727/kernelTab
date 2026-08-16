"""하드웨어 자동 감지.

동일한 컨테이너 이미지가 A6000 / 4090 / H100 어디서 돌아도 정확해야 하므로
GPU 스펙을 파이썬 코드에 하드코딩하지 않는다.

* API로 얻는 값: arch, sm_count, smem_per_block, max_threads_per_sm,
  regs_per_sm, l2_bytes, 이름  -> CUDA Driver API (libcuda, ctypes)
* API로 못 얻는 값: peak_tflops_f16, bandwidth_gbps -> hwspec/known.json

Driver API 를 ctypes 로 직접 쓰는 이유:
  - cudaGetDeviceProperties 는 struct 레이아웃이 CUDA 버전마다 달라
    ctypes 로 재현하기 취약하다. cuDeviceGetAttribute 는 (enum -> int)
    형태라 ABI가 안정적이다.
  - torch/cupy/pycuda 같은 무거운 의존성을 끌어오지 않는다.
"""

from __future__ import annotations

import ctypes
import json
from dataclasses import asdict
from pathlib import Path

from core.types import Hardware

__all__ = [
    "NVCC_ARCH",
    "UnknownGpuError",
    "HardwareDetectionError",
    "detect_hardware",
    "nvcc_arch_flag",
    "hardware_to_dict",
    "device_uuid",
]

# ---------------------------------------------------------------------------
# 0-c. 아키텍처 -> nvcc -arch 플래그
# ---------------------------------------------------------------------------
# Hopper 이상은 'a' 접미어(sm_90a)가 없으면 TMA/WGMMA 등 3.x API 핵심 기능이
# 컴파일 단계에서 비활성화된다. 빌드 명령에 -arch=sm_86 을 하드코딩하지 말고
# 반드시 이 표를 통해 감지된 arch 로부터 생성할 것.
NVCC_ARCH = {
    "sm_80": "sm_80",
    "sm_86": "sm_86",
    "sm_89": "sm_89",
    "sm_90": "sm_90a",
    "sm_100": "sm_100a",
}

HWSPEC_PATH = Path(__file__).resolve().parent.parent / "hwspec" / "known.json"


class HardwareDetectionError(RuntimeError):
    """CUDA Driver API 호출 실패."""


class UnknownGpuError(RuntimeError):
    """known.json 에 없는 GPU. 추정하지 않고 중단한다."""


class UnsupportedArch(RuntimeError):
    """지원하지 않는 compute capability."""


# ---------------------------------------------------------------------------
# CUDA Driver API (ctypes)
# ---------------------------------------------------------------------------
# CUdevice_attribute 열거값. CUDA 초기 버전부터 고정되어 있다.
_ATTR = {
    "max_threads_per_block": 1,
    "max_shared_memory_per_block": 8,
    "warp_size": 10,
    "multiprocessor_count": 16,
    "memory_clock_rate_khz": 36,
    "global_memory_bus_width": 37,
    "l2_cache_size": 38,
    "max_threads_per_multiprocessor": 39,
    "compute_capability_major": 75,
    "compute_capability_minor": 76,
    "max_shared_memory_per_multiprocessor": 81,
    "max_registers_per_multiprocessor": 82,
    "max_shared_memory_per_block_optin": 97,
}

_lib = None


def _driver() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    for cand in ("libcuda.so.1", "libcuda.so"):
        try:
            lib = ctypes.CDLL(cand)
            break
        except OSError:
            lib = None
    if lib is None:
        raise HardwareDetectionError(
            "libcuda.so 를 찾을 수 없다. NVIDIA 드라이버가 설치된 환경인지 확인하라."
        )
    rc = lib.cuInit(0)
    if rc != 0:
        raise HardwareDetectionError(f"cuInit 실패 (CUresult={rc})")
    _lib = lib
    return lib


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise HardwareDetectionError(f"{what} 실패 (CUresult={rc})")


def _cu_device(index: int) -> ctypes.c_int:
    lib = _driver()
    n = ctypes.c_int()
    _check(lib.cuDeviceGetCount(ctypes.byref(n)), "cuDeviceGetCount")
    if n.value == 0:
        raise HardwareDetectionError("CUDA 디바이스가 보이지 않는다.")
    if index >= n.value:
        raise HardwareDetectionError(
            f"device index {index} 요청했으나 보이는 디바이스는 {n.value}개다."
        )
    dev = ctypes.c_int()
    _check(lib.cuDeviceGet(ctypes.byref(dev), index), "cuDeviceGet")
    return dev


def _attr(dev: ctypes.c_int, key: str) -> int:
    lib = _driver()
    val = ctypes.c_int()
    _check(
        lib.cuDeviceGetAttribute(ctypes.byref(val), _ATTR[key], dev),
        f"cuDeviceGetAttribute({key})",
    )
    return val.value


def _name(dev: ctypes.c_int) -> str:
    lib = _driver()
    buf = ctypes.create_string_buffer(256)
    _check(lib.cuDeviceGetName(buf, 256, dev), "cuDeviceGetName")
    return buf.value.decode("utf-8", "replace").strip()


def device_uuid(index: int = 0) -> str:
    """물리 GPU 식별자. CUDA_VISIBLE_DEVICES 재배치와 무관하게 기록용으로 쓴다."""
    lib = _driver()
    dev = _cu_device(index)
    buf = (ctypes.c_byte * 16)()
    _check(lib.cuDeviceGetUuid(buf, dev), "cuDeviceGetUuid")
    h = bytes(bytearray((b & 0xFF) for b in buf)).hex()
    return f"GPU-{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


# ---------------------------------------------------------------------------
# known.json
# ---------------------------------------------------------------------------
def _load_known() -> dict:
    if not HWSPEC_PATH.exists():
        raise HardwareDetectionError(f"{HWSPEC_PATH} 가 없다.")
    with HWSPEC_PATH.open() as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _lookup_perf(name: str) -> tuple[float, float]:
    known = _load_known()
    entry = known.get(name)
    if entry is None:
        raise UnknownGpuError(
            f"GPU '{name}' 가 {HWSPEC_PATH} 에 없다.\n"
            f"  등록된 GPU: {sorted(known)}\n"
            "  peak_tflops_f16 / bandwidth_gbps 는 추정할 수 없다. roofline 과 "
            "ridge point 계산이 여기에 직접 의존하므로 공식 스펙시트 값을 "
            "known.json 에 추가한 뒤 다시 실행하라.\n"
            "  주의: peak_tflops_f16 은 FP32 누산 기준(dense) 값이어야 한다."
        )
    try:
        return float(entry["peak_tflops_f16"]), float(entry["bandwidth_gbps"])
    except KeyError as e:  # pragma: no cover - 설정 실수
        raise UnknownGpuError(f"'{name}' 항목에 {e} 키가 없다.") from e


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------
def detect_hardware(device_index: int = 0) -> Hardware:
    """런타임에 GPU를 감지해 Hardware를 구성한다."""
    dev = _cu_device(device_index)
    major = _attr(dev, "compute_capability_major")
    minor = _attr(dev, "compute_capability_minor")
    arch = f"sm_{major}{minor}"
    name = _name(dev)
    peak_tflops, bw = _lookup_perf(name)

    return Hardware(
        name=name,
        arch=arch,
        sm_count=_attr(dev, "multiprocessor_count"),
        # opt-in 최대치. sm_86 은 101376 (99KB) 이고, non-optin 값(48KB)과
        # 다르다. CUTLASS 는 dynamic smem 으로 opt-in 영역을 쓰므로 이쪽이 맞다.
        smem_per_block=_attr(dev, "max_shared_memory_per_block_optin"),
        max_threads_per_sm=_attr(dev, "max_threads_per_multiprocessor"),
        regs_per_sm=_attr(dev, "max_registers_per_multiprocessor"),
        peak_tflops_f16=peak_tflops,
        bandwidth_gbps=bw,
        l2_bytes=_attr(dev, "l2_cache_size"),
    )


def nvcc_arch_flag(arch: str) -> str:
    """감지한 arch -> nvcc -arch 값. sm_86 을 하드코딩하지 말고 이것을 쓴다."""
    try:
        return NVCC_ARCH[arch]
    except KeyError:
        raise UnsupportedArch(
            f"{arch} 에 대한 nvcc arch 매핑이 없다. NVCC_ARCH 에 추가하라."
        ) from None


def hardware_to_dict(hw: Hardware) -> dict:
    return asdict(hw)


def extra_device_info(device_index: int = 0) -> dict:
    """Hardware 에는 넣지 않지만 env.json 기록용으로 유용한 부가 정보."""
    dev = _cu_device(device_index)
    return {
        "uuid": device_uuid(device_index),
        "warp_size": _attr(dev, "warp_size"),
        "max_threads_per_block": _attr(dev, "max_threads_per_block"),
        "smem_per_block_default": _attr(dev, "max_shared_memory_per_block"),
        "smem_per_sm": _attr(dev, "max_shared_memory_per_multiprocessor"),
        "memory_clock_khz": _attr(dev, "memory_clock_rate_khz"),
        "memory_bus_width_bits": _attr(dev, "global_memory_bus_width"),
    }
