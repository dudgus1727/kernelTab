"""경로 해석 (CUTLASS, CUDA, 산출물 디렉토리).

컨테이너/로컬 어디서든 동작해야 하므로 경로를 코드에 박지 않고
환경변수 -> 관례적 위치 순으로 탐색한다. 최종 결정된 값은 Phase 0 에서
results/env.json 에 기록되고, 이후 단계는 env.json 을 단일 진실 소스로 쓴다.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

#: 패키지 루트 (`kerneltab/`). **C++ 소스가 여기 있다** — `measure/*.cu`,
#: `*.h` 는 런타임에 nvcc 로 컴파일되므로 설치된 위치를 따라가야 한다.
PKG_ROOT = Path(__file__).resolve().parent.parent

#: 저장소 루트. 산출물(`results/`, `artifacts/`)과 데이터(`hwspec/`)가 여기
#: 있다. **패키지 밖**이다 — 7.4 GB 짜리 산출물이 패키지 안에 있으면
#: `pip install` 이 그것까지 가져가려 하고, 컨테이너 볼륨 마운트 지점으로도
#: 부적절하다.
REPO_ROOT = PKG_ROOT.parent


def _dir_from_env(var: str, default: Path) -> Path:
    """환경변수로 덮어쓸 수 있는 디렉토리 (수정 6).

    컨테이너에서는 `results/` 와 `artifacts/` 를 **볼륨으로 마운트**한다.
    이미지 안 경로와 호스트 경로가 다르므로 코드에 박으면 안 된다.

    ⚠️ **기본값을 바꾸려면 디렉토리를 함께 옮겨야 한다.** 바꾸기만 하면 기존
    산출물(98 만 줄, 7.4 GB 커널)을 못 찾는다. 실제로 패키지 이전에서
    `build/artifacts` -> `artifacts` 로 한 번 옮겼고, 그때 디렉토리 이동과
    기본값 변경과 `tests/test_portability.py` 의 단언을 **같은 커밋에서**
    바꿨다. 셋 중 하나만 바꾸면 조용히 깨진다.
    """
    v = os.environ.get(var)
    return Path(v).expanduser().resolve() if v else default


#: 측정 산출물. `KERNELTAB_RESULTS_DIR` 로 덮어쓴다 (컨테이너 볼륨).
RESULTS_DIR = _dir_from_env("KERNELTAB_RESULTS_DIR", REPO_ROOT / "results")

#: 빌드 산출물(커널 .so 7.4 GB). `KERNELTAB_ARTIFACT_DIR` 로 덮어쓴다.
#: 아키텍처마다 다르므로 이미지에 굽지 않고 볼륨에 캐싱한다.
#: 패키지 **밖**이다 (`artifacts/`, 예전 `build/artifacts/` 아님).
ARTIFACT_DIR = _dir_from_env("KERNELTAB_ARTIFACT_DIR",
                             REPO_ROOT / "artifacts")

ENV_JSON = RESULTS_DIR / "env.json"

#: GPU 스펙 데이터. 패키지가 아니라 **데이터 디렉토리**다 (`__init__.py`
#: 없음). `KERNELTAB_HWSPEC_DIR` 로 덮어쓸 수 있다 — editable 설치가 아니면
#: 저장소 루트가 없을 수 있기 때문이다.
HWSPEC_DIR = _dir_from_env("KERNELTAB_HWSPEC_DIR", REPO_ROOT / "hwspec")


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
    """CUTLASS 헤더만. **커널 소스를 컴파일할 때는 `kernel_includes()` 를 써라.**"""
    return [
        f"-I{root / 'include'}",
        f"-I{root / 'tools' / 'util' / 'include'}",
    ]


def kernel_includes(root: Path) -> list[str]:
    """생성된 커널 `.cu` 를 컴파일하는 데 필요한 include 전부.

    `emit_cpp()` 결과는 `kt_swizzle.h` / `kt_abi.h` 를 include 하므로
    `kerneltab/measure/` 가 반드시 들어가야 한다. CUTLASS 경로만 주면
    **`fatal error: kt_swizzle.h: No such file or directory`** 로 죽는다.

    실제로 `check_smem.py` 가 CUTLASS 경로만 주고 있어서 40개 전부
    "BUILD FAIL / compilation terminated." 이 났다. 조립을 여러 곳에서
    따로 하면 이렇게 된다 — 그래서 여기 하나로 모은다.
    """
    return [*cutlass_includes(root), f"-I{PKG_ROOT / 'measure'}"]


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
