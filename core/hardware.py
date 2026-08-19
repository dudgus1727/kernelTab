"""하드웨어 자동 감지.

동일한 컨테이너 이미지가 A6000 / 4090 / H100 어디서 돌아도 정확해야 하므로
GPU 스펙을 파이썬 코드에 하드코딩하지 않는다.

* API로 얻는 값: arch, sm_count, smem_per_block, max_threads_per_sm,
  regs_per_sm, l2_bytes, 이름  -> CUDA Driver API (libcuda, ctypes)
* API로 못 얻는 값: peak_tflops_f16 -> hwspec/known.json (없으면 중단)
* 대역폭: 버스 폭 x 2(DDR) x 메모리 클럭 으로 API 에서 유도한다.
  known.json 의 bandwidth_gbps_spec 은 교차검증용이다.

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
from dataclasses import replace as dc_replace
from pathlib import Path

from core.types import Hardware

__all__ = [
    "NVCC_ARCH",
    "HardwareDetectionError",
    "UnknownGpuError",
    "bandwidth_from_api",
    "bandwidth_reference_mhz",
    "detect_hardware",
    "device_uuid",
    "hardware_from_env",
    "hardware_to_dict",
    "known_spec",
    "nvcc_arch_flag",
    "peak_reference_mhz",
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


def peak_reference_mhz(name: str) -> int | None:
    """peak_tflops_f16 이 성립하는 SM 클럭. 클럭 고정 측정 시 보정에 쓴다."""
    entry = _load_known().get(name)
    if entry is None:
        return None
    v = entry.get("peak_tflops_f16_at_mhz")
    return int(v) if v else None


def known_spec(name: str) -> dict:
    """known.json 항목 전체 (교차검증용)."""
    return dict(_load_known().get(name) or {})


def bandwidth_reference_mhz(name: str) -> int | None:
    """bandwidth_gbps 가 성립하는 메모리 클럭. 클럭 고정 측정 시 보정에 쓴다."""
    entry = _load_known().get(name)
    if entry is None:
        return None
    v = entry.get("bandwidth_gbps_at_mem_mhz")
    return int(v) if v else None


def _lookup_peak(name: str) -> float:
    """peak_tflops_f16. **API 로 얻을 방법이 없어 known.json 에 의존한다.**

    없으면 추정하지 않고 중단한다. roofline 의 분자가 여기 직접 달려 있다.
    (대역폭은 버스 폭 x 2 x 클럭 으로 API 에서 유도되므로 표가 필요 없다.)
    """
    known = _load_known()
    entry = known.get(name)
    if entry is None:
        raise UnknownGpuError(
            f"GPU '{name}' 가 {HWSPEC_PATH} 에 없다.\n"
            f"  등록된 GPU: {sorted(known)}\n"
            "  peak_tflops_f16 은 API 로 얻을 수 없고 추정해서도 안 된다. "
            "roofline 의 분자가 여기 직접 의존하므로 공식 스펙시트 값을 "
            "known.json 에 추가한 뒤 다시 실행하라.\n"
            "  필수 키: peak_tflops_f16 (FP32 누산 기준 dense), "
            "peak_tflops_f16_at_mhz (그 값이 성립하는 SM 클럭)."
        )
    try:
        return float(entry["peak_tflops_f16"])
    except KeyError as e:  # pragma: no cover - 설정 실수
        raise UnknownGpuError(f"'{name}' 항목에 {e} 키가 없다.") from e


def bandwidth_from_api(bus_bits: int, mem_mhz: float) -> float:
    """대역폭 = 버스 폭 x 2(DDR) x 클럭. **유일한 계산 경로다.**

    GDDR6/GDDR6X/HBM 모두 nvidia-smi 가 보고하는 메모리 클럭의 2배가 데이터
    레이트다 (A6000 8001->16 Gbps, 4090 10501->21 Gbps, H100 2619->5.2 Gbps).
    known.json 의 bandwidth_gbps_spec 은 교차검증용으로만 쓴다 — 계산 경로가
    하나여야 다른 GPU 에서 표 항목을 빼먹어도 조용히 틀리지 않는다.
    """
    return bus_bits * 2 * mem_mhz * 1e6 / 8 / 1e9


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
    peak_tflops = _lookup_peak(name)
    # 대역폭은 표가 아니라 API 에서 유도한다 (P0 최대 메모리 클럭 기준).
    bw = round(bandwidth_from_api(_attr(dev, "global_memory_bus_width"),
                                  _attr(dev, "memory_clock_rate_khz") / 1000.0), 2)

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


def hardware_from_env(env: dict) -> Hardware:
    """results/env.json -> Hardware.  **모든 호출부는 이것만 쓴다.**

    `Hardware(**env["hardware"])` 를 직접 쓰면 peak_tflops_f16 이 스펙(부스트
    클럭 기준) 값이 되어, 클럭을 고정한 측정에서 roofline ridge point 가
    실제보다 높게 나오고 "이 형상이 메모리 바운드인가" 판정이 틀린다.

    GPU 마다 다른 클럭으로 고정하게 되므로, 유효 피크를 쓰지 않으면
    "메모리 바운드" 의 의미가 GPU 마다 달라져 전이 실험이 오염된다.
    """
    hw = Hardware(**env["hardware"])
    eff = env.get("peak_tflops_f16_effective")
    if eff and eff != hw.peak_tflops_f16:
        hw = dc_replace(hw, peak_tflops_f16=float(eff))
    bw = env.get("bandwidth_gbps_effective")
    if bw and bw != hw.bandwidth_gbps:
        hw = dc_replace(hw, bandwidth_gbps=float(bw))
    return hw


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
