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


def _dir_from_env(var: str, default: Path) -> Path:
    """환경변수로 덮어쓸 수 있는 디렉토리 (수정 6).

    컨테이너에서는 `results/` 와 `build/artifacts/` 를 **볼륨으로 마운트**한다.
    이미지 안 경로와 호스트 경로가 다르므로 코드에 박으면 안 된다.

    ⚠️ **기본값은 반드시 현재 경로여야 한다.** 바꾸면 기존 산출물
    (98 만 줄, 7.4 GB 커널)을 못 찾는다. 환경변수를 안 주면 지금과 똑같이
    동작한다.
    """
    v = os.environ.get(var)
    return Path(v).expanduser().resolve() if v else default


#: 측정 산출물. `KERNELTAB_RESULTS_DIR` 로 덮어쓴다 (컨테이너 볼륨).
RESULTS_DIR = _dir_from_env("KERNELTAB_RESULTS_DIR", REPO_ROOT / "results")

#: 빌드 산출물(커널 .so 7.4 GB). `KERNELTAB_ARTIFACT_DIR` 로 덮어쓴다.
#: 아키텍처마다 다르므로 이미지에 굽지 않고 볼륨에 캐싱한다.
ARTIFACT_DIR = _dir_from_env("KERNELTAB_ARTIFACT_DIR",
                             REPO_ROOT / "build" / "artifacts")

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


def kernel_so(kernel_id: str) -> Path:
    """커널 `.so` 의 위치. **`kernels.jsonl` 의 `so_path` 를 쓰지 마라** (P-1).

    `so_path` 는 빌드한 기계의 **절대 경로**다. 컨테이너 안에서는 그 경로가
    존재하지 않고, 저장소를 옮기거나 볼륨 마운트 지점이 다르면 그대로 깨진다.
    `kernel_id` 에서 조립하면 어디서 읽든 맞는다.

    기존 `kernels.jsonl` 7,490 줄과 **100 % 일치**함을 확인했다 (불일치 0).
    옛 줄의 `so_path` 는 남아 있어도 읽는 쪽이 무시한다.
    """
    return ARTIFACT_DIR / "lib" / f"{kernel_id}.so"
