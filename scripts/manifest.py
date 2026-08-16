#!/usr/bin/env python3
"""재현성 매니페스트 — 컨테이너 태그와 데이터 출처 추적에 쓴다.

"이 데이터가 어떤 코드/라이브러리 조합으로 만들어졌는가" 를 한 줄로 답할 수
있어야 한다. 그 답이 `manifest_hash` 이고, 이미지 태그가 된다:

    kerneltab:cu124-<manifest_hash 앞 8자>

GPU 를 전혀 건드리지 않으므로 측정 중에도 안전하게 실행할 수 있다.

    python3 scripts/manifest.py               # 사람이 읽는 형태
    python3 scripts/manifest.py --json        # JSON 만
    python3 scripts/manifest.py --tag         # 이미지 태그만
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build import paths  # noqa: E402

#: 매니페스트 해시에 들어가는 키. **재현성에 영향을 주는 것만** 넣는다.
#: 시각/호스트명/가용 메모리처럼 실행마다 변하는 값은 절대 넣지 않는다
#: (넣으면 같은 조건에서도 해시가 매번 달라져 추적이 무의미해진다).
HASHED_KEYS = (
    "cuda_version",
    "nvcc_version",
    "cutlass_commit",
    "kerneltab_commit",
    "kerneltab_tree_hash",
    "python_version",
    "packages",
)


def run(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return p.returncode, (p.stdout + p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, repr(e)


def nvcc_info() -> tuple[str | None, str | None]:
    try:
        nvcc = paths.nvcc_path()
    except Exception:
        return None, None
    rc, out = run([str(nvcc), "--version"])
    if rc != 0:
        return None, None
    m = re.search(r"release (\d+\.\d+), V(\S+)", out)
    return (m.group(1), m.group(2)) if m else (None, None)


def git_info(root: Path) -> dict:
    """커밋 + worktree 상태. dirty 면 커밋만으로는 재현되지 않는다."""
    rc, commit = run(["git", "-C", str(root), "rev-parse", "HEAD"])
    rc2, dirty = run(["git", "-C", str(root), "status", "--porcelain"])
    if rc != 0:
        return {"commit": None, "dirty": None, "error": commit[:200]}
    return {
        "commit": commit.strip(),
        "dirty": bool(dirty.strip()) if rc2 == 0 else None,
        "dirty_files": len(dirty.strip().splitlines()) if rc2 == 0 else None,
    }


def tree_hash(root: Path) -> str:
    """추적 대상 소스의 내용 해시.

    커밋만으로는 부족하다 — 커밋하지 않은 수정이 있으면 같은 커밋이라도 다른
    코드다. 실제 파일 내용을 해싱해 그 구멍을 막는다.
    """
    h = hashlib.sha256()
    pats = ("core/*.py", "backends/*.py", "build/*.py", "build/*.cu",
            "measure/*.py", "measure/*.cu", "measure/*.h", "scripts/*.py",
            "hwspec/*.json", "pyproject.toml")
    files: list[Path] = []
    for pat in pats:
        files.extend(sorted(root.glob(pat)))
    for f in sorted(files):
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def packages() -> dict:
    """설치된 패키지 중 이 하네스가 실제로 쓰는 것들의 버전."""
    import importlib.metadata as md

    want = ("nvidia-ml-py", "pyarrow", "pandas", "numpy", "setuptools", "pip")
    out = {}
    for name in want:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            continue
    return out


def build(cutlass_dir: str | None = None) -> dict:
    release, version = nvcc_info()
    try:
        cutlass_root = paths.cutlass_dir(cutlass_dir)
    except Exception:
        cutlass_root = None

    cutlass_git = git_info(cutlass_root) if cutlass_root else {"commit": None}
    kt_git = git_info(REPO_ROOT)

    m = {
        "cuda_version": release,
        "nvcc_version": version,
        "cutlass_commit": cutlass_git.get("commit"),
        "cutlass_dirty": cutlass_git.get("dirty"),
        "cutlass_dir": str(cutlass_root) if cutlass_root else None,
        "kerneltab_commit": kt_git.get("commit"),
        "kerneltab_dirty": kt_git.get("dirty"),
        "kerneltab_tree_hash": tree_hash(REPO_ROOT),
        "python_version": platform.python_version(),
        "python_impl": platform.python_implementation(),
        "packages": packages(),
    }
    payload = {k: m.get(k) for k in HASHED_KEYS}
    m["manifest_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    m["hashed_keys"] = list(HASHED_KEYS)
    m["image_tag"] = image_tag(m)
    return m


def image_tag(m: dict) -> str:
    """kerneltab:cu124-abcd1234 형식. latest 는 쓰지 않는다."""
    cu = (m.get("cuda_version") or "unknown").replace(".", "")
    return f"kerneltab:cu{cu}-{(m.get('manifest_hash') or '')[:8]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--tag", action="store_true")
    ap.add_argument("--cutlass", default=None)
    args = ap.parse_args()

    m = build(args.cutlass)
    if args.tag:
        print(m["image_tag"])
        return 0
    if args.json:
        print(json.dumps(m, indent=2, ensure_ascii=False))
        return 0

    print("재현성 매니페스트")
    print("=" * 62)
    for k in ("cuda_version", "nvcc_version", "cutlass_commit", "cutlass_dirty",
              "kerneltab_commit", "kerneltab_dirty", "kerneltab_tree_hash",
              "python_version"):
        v = m.get(k)
        if k.endswith("tree_hash") and v:
            v = v[:16] + "..."
        print(f"  {k:22s} {v}")
    print(f"  {'packages':22s} {m['packages']}")
    print()
    print(f"  manifest_hash          {m['manifest_hash']}")
    print(f"  image_tag              {m['image_tag']}")
    if m.get("kerneltab_dirty") or m.get("cutlass_dirty"):
        print("\n  !! worktree 에 커밋되지 않은 변경이 있다. 커밋 해시만으로는")
        print("     재현되지 않으므로 tree_hash 를 함께 봐야 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
