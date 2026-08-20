"""`core/kernels.py` — 런치 가능성 술어가 **한 곳에만** 있는지 고정한다.

이 술어는 여섯 곳에 복사돼 있었고, 결측 처리가 서로 달랐다
(`True` / `None` / 통과시킴). 같은 클래스가 여러 경로에 있으면 한 곳을
고쳐도 나머지가 남는다 — `docs/decisions.md` 13 의 일곱 번째와 같다.
"""
from __future__ import annotations

import ast
import pathlib

from core import kernels

REPO = pathlib.Path(__file__).resolve().parent.parent
REGS_PER_SM = 65536


def _row(regs=None, threads=None, tile=None, ext=None):
    r = {}
    if regs is not None:
        r["regs_per_thread"] = regs
    if threads is not None:
        r["threads"] = threads
    if tile:
        r["tile"] = tile
    if ext:
        r["ext"] = ext
    return r


def test_한계_이하는_런치_가능():
    assert kernels.launchable(_row(regs=128, threads=512), REGS_PER_SM)


def test_한계_초과는_불가():
    # 실측: sm86_tb256x256x32_w64x64x32_st3_swid4_a884 는 216 x 512 = 110,592
    assert not kernels.launchable(_row(regs=216, threads=512), REGS_PER_SM)


def test_정확히_한계는_가능():
    assert kernels.launchable(_row(regs=128, threads=512), 65536)


def test_결측이면_True():
    """'모른다' 를 '못 쓴다' 로 바꾸면 멀쩡한 커널이 조용히 빠진다."""
    assert kernels.launchable(_row(), REGS_PER_SM)
    assert kernels.launchable(_row(regs=128), REGS_PER_SM)
    assert kernels.regs_total_per_block(_row()) is None


def test_threads_를_타일에서_계산한다():
    """`threads` 컬럼이 없는 옛 줄도 판정할 수 있어야 한다."""
    row = _row(regs=216,
               tile={"m": 256, "n": 256, "k": 32},
               ext={"warp_m": 64, "warp_n": 64, "warp_k": 32})
    assert kernels.threads_per_block(row) == 512
    assert kernels.regs_total_per_block(row) == 110592
    assert not kernels.launchable(row, REGS_PER_SM)


def test_망가진_행에도_예외가_안_난다():
    assert kernels.threads_per_block({"tile": {}, "ext": {}}) is None
    assert kernels.threads_per_block(
        {"tile": {"m": 128, "n": 128, "k": 32},
         "ext": {"warp_m": 0, "warp_n": 64, "warp_k": 32}}) is None


def test_인라인_복사본이_다시_생기지_않는다():
    """`regs_per_thread * threads` 를 직접 쓰는 곳이 없어야 한다.

    이것이 이 모듈의 존재 이유다. 새로 추가한 사람이 인라인으로 다시 쓰면
    여기서 걸린다.
    """
    offenders = []
    for p in list((REPO / "scripts").glob("*.py")) + \
            list((REPO / "core").glob("*.py")) + \
            list((REPO / "measure").glob("*.py")):
        if p.name == "kernels.py":
            continue
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
                continue
            names = set()
            for side in (node.left, node.right):
                if isinstance(side, ast.Subscript) and isinstance(
                        side.slice, ast.Constant):
                    names.add(side.slice.value)
                elif isinstance(side, ast.Call) and isinstance(
                        side.func, ast.Attribute) and side.func.attr == "get" \
                        and side.args and isinstance(side.args[0], ast.Constant):
                    names.add(side.args[0].value)
                elif isinstance(side, ast.Name):
                    names.add(side.id)
            if {"regs_per_thread", "threads"} <= names:
                offenders.append(f"{p.relative_to(REPO)}:{node.lineno}")
    assert not offenders, (
        "런치 가능성 판정이 다시 인라인으로 복사됐다. "
        f"core.kernels.launchable() 을 써라: {offenders}")
