"""공용 픽스처. GPU 를 전혀 쓰지 않는다 — 측정 중에도 안전하게 돌릴 수 있다."""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backends import get_backend  # noqa: E402
from core.types import Hardware, KernelConfig  # noqa: E402


@pytest.fixture(scope="session")
def backend():
    return get_backend("sm_86")


@pytest.fixture
def hw_a6000():
    """A6000 을 1350 MHz / 7601 MHz 로 고정했을 때의 실효 스펙."""
    return Hardware(
        name="NVIDIA RTX A6000", arch="sm_86", sm_count=84,
        smem_per_block=101376, max_threads_per_sm=1536, regs_per_sm=65536,
        peak_tflops_f16=116.1, bandwidth_gbps=729.7, l2_bytes=6291456,
    )


@pytest.fixture
def hw_other():
    """SM 개수만 다른 가상 GPU. 하드웨어 상수 하드코딩 검출용."""
    return Hardware(
        name="FAKE", arch="sm_86", sm_count=128,
        smem_per_block=65536, max_threads_per_sm=2048, regs_per_sm=65536,
        peak_tflops_f16=200.0, bandwidth_gbps=1000.0, l2_bytes=4194304,
    )


@pytest.fixture
def mk_cfg(backend):
    """KernelConfig 를 간단히 만드는 헬퍼."""
    def _mk(tile=(128, 128, 32), warp=(64, 64, 32), stages=4,
            swizzle=("identity", 8), align=(8, 8, 8), arch="sm_86"):
        ext = backend.ext_from_dict({
            "warp_m": warp[0], "warp_n": warp[1], "warp_k": warp[2],
            "stages": stages, "swizzle_type": swizzle[0],
            "swizzle_n": swizzle[1]})
        return KernelConfig(tile[0], tile[1], tile[2],
                            align[0], align[1], align[2], arch, ext)
    return _mk
