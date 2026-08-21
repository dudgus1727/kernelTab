"""**계약 문서가 적어 둔 것을 코드가 지키는지** AST 로 검사한다.

이 파일이 왜 있는가 — `docs/consumer_contract.md` §9 는

    "status != ok 는 결측이 아니다. 제외하면 그 config 가 통째로 빠진다"

라고 **옳게** 적혀 있었는데, 코드는 정확히 반대로 하고 있었다
(`baseline_*.py` 가 `status == "ok"` 만 남겼다). 그 결과 61형상 전부에서
측정된 config 가 3,465 개가 아니라 **3 개**로 줄었고, 정적 top-1 이
1.115 인데 1.394 로 공개 문서에 실렸다.

**계약 문서가 옳아도 코드가 반대면 아무도 못 본다** — 문서를 믿으니까
아무도 안 본다. 계약을 적었으면 그 계약을 검사하는 테스트를 함께 써라.
`docs/decisions.md` 13 에 같은 줄이 있다.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCAN_DIRS = ("kerneltab", "scripts", "rules")


def _py_files():
    for d in SCAN_DIRS:
        root = REPO / d
        if root.is_dir():
            yield from sorted(root.rglob("*.py"))


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _where(path, node):
    return f"{path.relative_to(REPO)}:{node.lineno}"


# --------------------------------------------------------------------------
# consumer_contract §9 — status 필터
# --------------------------------------------------------------------------

#: 정당한 status 비교임을 표시하는 주석. 같은 줄이나 바로 위 4줄 안에
#: 있어야 한다. 이유를 뒤에 적는다.
STATUS_MARKER = "# status-filter:"
MARKER_LOOKBACK = 4


def _status_comparisons(tree):
    """`... status ... == "ok"` (또는 `!=`) 형태의 비교를 찾는다."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        consts = [c.value for c in [node.left, *node.comparators]
                  if isinstance(c, ast.Constant)]
        if "ok" not in consts:
            continue
        src = ast.dump(node)
        if "'status'" in src or "attr='status'" in src:
            yield node


def _marked(lines, lineno):
    """이 비교에 `# status-filter:` 표시가 붙어 있는가."""
    lo = max(0, lineno - 1 - MARKER_LOOKBACK)
    return any(STATUS_MARKER in ln for ln in lines[lo:lineno])


def test_status_ok_필터에는_근거_표시가_붙는다():
    """§9: `status != ok` 는 **결측이 아니다.** 지우면 config 가 통째로 빠진다.

    거를 때마다 `# status-filter: <이유>` 를 적게 한다. 파일 단위 허용
    목록이 아니라 **줄 단위**인 것이 요점이다 — `export.py` 는 정당한
    비교(정책 게이트)와 틀린 비교를 **같은 파일 안에** 갖고 있었다.

    표시를 붙이는 순간 리뷰 지점이 된다. 이 프로젝트가 놓친 것은 검사가
    아니라 **눈에 띄는 자리**였다.
    """
    bad = []
    for f in _py_files():
        lines = f.read_text(encoding="utf-8").splitlines()
        for node in _status_comparisons(_tree(f)):
            if not _marked(lines, node.lineno):
                bad.append(f"{_where(f, node)}  {lines[node.lineno - 1].strip()}")
    assert not bad, (
        "status 비교에 근거 표시(`# status-filter:`)가 없다:\n  "
        + "\n  ".join(bad)
        + "\n\nconsumer_contract §9: status != ok 는 결측이 아니다. "
          "제외하면 전 형상 덮개를 요구하는 집계(정적 top-k)가 무너진다 "
          "— 3,465 -> 3 개가 됐던 자리다. 거르는 것이 맞으면 이유를 적어라.")


@pytest.mark.parametrize("script", ["baseline_rule", "baseline_gbdt",
                                    "baseline_vendor"])
def test_baseline_status_기본값이_all(script):
    """`--status` 기본값이 `ok` 로 되돌아가면 여기서 걸린다."""
    tree = _tree(REPO / "scripts" / f"{script}.py")
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "add_argument"):
            continue
        names = [a.value for a in node.args if isinstance(a, ast.Constant)]
        if "--status" not in names:
            continue
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                found.append(kw.value.value)
    assert found == ["all"], (
        f"{script}.py 의 --status 기본값이 {found} 다. 'all' 이어야 한다 — "
        "기본값이 ok 면 아무도 플래그를 안 주고, 그게 baselines.md 를 "
        "틀리게 만든 경로다.")


# --------------------------------------------------------------------------
# C-1 — 노이즈 계수는 주입한다
# --------------------------------------------------------------------------

#: 지운 모듈 전역들. 이름이 되살아나면 조용히 A6000 눈금을 쓰는 경로가
#: 다시 열린다.
REMOVED_NOISE_GLOBALS = ("noise_floor", "noise_floor_ms", "sigma_rel",
                         "tick_pct", "resolvable")

#: `A6000_MEASURED` 를 이름으로 쓰는 것이 **허용된** 곳.
#: 번들이 아직 없는 자리 — 측정 직후 그 장비에서 도는 코드다.
A6000_NAME_ALLOWED = {
    "kerneltab/core/anchors.py",     # 앵커 판정은 관측 산포로 한다 (참고열)
    "scripts/measure_drift.py",      # 상대 가중치로만 쓴다
    "scripts/verify_warmup.py",      # 통계항만; 눈금은 관측에서 추정
    "scripts/compare_campaigns.py",  # 통계항만; 눈금은 관측에서 추정
}


def test_지운_노이즈_전역이_되살아나지_않는다():
    """모듈 전역 `noise_floor(t)` 는 **없어야 한다.**

    있으면 `answer_set()` 같은 채점 경로가 그것을 부르고, 4090/H100 번들을
    채점할 때 A6000 눈금을 **경고 없이** 쓴다 (`Bundle.tick_ms` 를 안 거치니
    그 경고조차 안 난다). 눈금이 2 배인 GPU 에서 정답 집합이 1.5~6.2 배
    어긋난다.
    """
    from kerneltab.core import noise

    alive = [n for n in REMOVED_NOISE_GLOBALS if hasattr(noise, n)]
    assert not alive, (
        f"core/noise.py 에 모듈 전역 {alive} 이 되살아났다. "
        "계수는 NoiseCoef 로 주입한다 — Bundle.coef / noise.from_bundle().")


def test_아무도_지운_전역을_import_하지_않는다():
    bad = []
    for f in _py_files():
        for node in ast.walk(_tree(f)):
            if isinstance(node, ast.ImportFrom) and node.module and \
                    node.module.endswith("core.noise"):
                for a in node.names:
                    if a.name in REMOVED_NOISE_GLOBALS:
                        bad.append(f"{_where(f, node)}  ({a.name})")
            elif isinstance(node, ast.Attribute) and \
                    node.attr in REMOVED_NOISE_GLOBALS and \
                    getattr(node.value, "id", "") == "noise":
                bad.append(f"{_where(f, node)}  (noise.{node.attr})")
    assert not bad, "지운 노이즈 전역을 참조한다:\n  " + "\n  ".join(bad)


def test_A6000_계수는_허용된_곳에서만_이름으로_쓴다():
    """`A6000_MEASURED` 를 새 채점 경로에서 쓰기 시작하면 걸린다.

    이름을 쓰는 것 자체는 괜찮다 — 출처가 코드에 보이니까. 다만 **어디서
    쓰는지**는 리뷰 지점이어야 한다. 번들이 있는 자리라면 `Bundle.coef` 를
    써야 한다.
    """
    bad = []
    for f in _py_files():
        rel = str(f.relative_to(REPO))
        if rel in A6000_NAME_ALLOWED or rel == "kerneltab/core/noise.py":
            continue
        for node in ast.walk(_tree(f)):
            if isinstance(node, ast.Name) and node.id == "A6000_MEASURED":
                bad.append(_where(f, node))
            elif isinstance(node, ast.Attribute) and \
                    node.attr == "A6000_MEASURED":
                bad.append(_where(f, node))
    assert not bad, (
        "A6000_MEASURED 를 허용 목록 밖에서 쓴다:\n  " + "\n  ".join(bad)
        + "\n번들이 있으면 Bundle.coef 를 써라.")
