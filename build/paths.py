"""경로 해석 (CUTLASS, CUDA, 산출물 디렉토리).

컨테이너/로컬 어디서든 동작해야 하므로 경로를 코드에 박지 않고
환경변수 -> 관례적 위치 순으로 탐색한다. 최종 결정된 값은 Phase 0 에서
results/env.json 에 기록되고, 이후 단계는 env.json 을 단일 진실 소스로 쓴다.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
ARTIFACT_DIR = REPO_ROOT / "build" / "artifacts"
ENV_JSON = RESULTS_DIR / "env.json"


class PathError(RuntimeError):
    pass


def _cutlass_candidates() -> list[Path]:
    cands: list[Path] = []
    for var in ("KERNELTAB_CUTLASS_DIR", "CUTLASS_DIR", "CUTLASS_PATH"):
        v = os.environ.get(var)
        if v:
            cands.append(Path(v))
    cands += [
        Path("/opt/cutlass"),
        Path("/usr/local/cutlass"),
        REPO_ROOT.parent / "related_work" / "cutlass",
        REPO_ROOT.parent / "cutlass",
    ]
    return cands


def cutlass_dir(explicit: str | os.PathLike | None = None) -> Path:
    """CUTLASS 저장소 루트. include/cutlass/cutlass.h 존재로 검증한다."""
    cands = [Path(explicit)] if explicit else _cutlass_candidates()
    for c in cands:
        if (c / "include" / "cutlass" / "cutlass.h").exists():
            return c.resolve()
    raise PathError(
        "CUTLASS 저장소를 찾지 못했다. --cutlass 로 지정하거나 "
        "KERNELTAB_CUTLASS_DIR 환경변수를 설정하라.\n"
        f"  탐색한 경로: {[str(c) for c in cands]}"
    )


def cutlass_includes(root: Path) -> list[str]:
    return [
        f"-I{root / 'include'}",
        f"-I{root / 'tools' / 'util' / 'include'}",
    ]


def nvcc_path() -> Path:
    p = shutil.which("nvcc")
    if p:
        return Path(p)
    for c in (Path("/usr/local/cuda/bin/nvcc"), Path("/opt/cuda/bin/nvcc")):
        if c.exists():
            return c
    raise PathError("nvcc 를 찾을 수 없다. PATH 또는 CUDA_HOME 을 확인하라.")


def cuda_home() -> Path:
    v = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if v:
        return Path(v)
    return nvcc_path().parent.parent


def cuda_bin(tool: str) -> Path:
    """cuobjdump / nvdisasm 등 CUDA 툴킷 바이너리."""
    p = shutil.which(tool)
    if p:
        return Path(p)
    c = cuda_home() / "bin" / tool
    if c.exists():
        return c
    raise PathError(f"{tool} 을 찾을 수 없다 (CUDA 툴킷 설치 확인).")


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
