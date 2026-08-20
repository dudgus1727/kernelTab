"""`scripts/manifest.py` — 이미지 태그가 소스를 실제로 반영하는가.

패키지 이전에서 `tree_hash` 의 glob 패턴 10개 중 **7개가 0개를 맞았다.**
`core/*.py` 가 `kerneltab/core/*.py` 로 옮겨졌기 때문이다. 그런데 해시는
여전히 그럴듯한 값이 나왔고 **오류는 없었다.** 결과는 "코드를 고쳐도 이미지
태그가 그대로" — 같은 태그, 다른 코드.

이 저장소가 네 번 밟은 "조용히 아무것도 안 하는 안전장치" 와 같은 클래스다
(`docs/decisions.md` 14).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location(
        "kt_manifest", REPO / "scripts" / "manifest.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _mod()


def test_모든_패턴이_파일을_찾는다():
    """소스가 옮겨지면 여기서 걸린다."""
    empty = [p for p in M.TREE_HASH_PATTERNS if not list(REPO.glob(p))]
    assert not empty, (
        f"tree_hash 패턴이 0개를 맞는다: {empty}. 소스가 옮겨졌다면 "
        "TREE_HASH_PATTERNS 를 함께 고쳐야 한다 — 안 그러면 코드가 바뀌어도 "
        "이미지 태그가 그대로다.")


def test_패키지_소스가_포함된다():
    """`kerneltab/` 아래 .py 가 전부 훑어져야 한다."""
    covered = {f.resolve() for p in M.TREE_HASH_PATTERNS for f in REPO.glob(p)}
    missing = [f for f in REPO.glob("kerneltab/**/*.py")
               if f.resolve() not in covered and "__pycache__" not in str(f)]
    assert not missing, f"tree_hash 가 빠뜨린 소스: {missing}"


def test_패턴이_비면_예외다(tmp_path):
    """조용히 그럴듯한 해시를 돌려주면 안 된다."""
    with pytest.raises(M.ManifestError) as e:
        M.tree_hash(tmp_path)
    assert "tree_hash" in str(e.value)


def test_내용이_바뀌면_해시가_바뀐다(tmp_path):
    for p in M.TREE_HASH_PATTERNS:
        d = tmp_path / Path(p).parent
        d.mkdir(parents=True, exist_ok=True)
        (d / Path(p).name.replace("*", "x")).write_text("a")
    h1 = M.tree_hash(tmp_path)
    (tmp_path / "pyproject.toml").write_text("b")
    assert M.tree_hash(tmp_path) != h1


def test_태그에_latest_가_없다():
    tag = M.image_tag({"cuda_version": "13.3", "manifest_hash": "0" * 64})
    assert tag == "kerneltab:cu133-00000000"
    assert "latest" not in tag
