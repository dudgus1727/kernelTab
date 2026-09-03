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
    """status 를 `"ok"` 로 거르는 자리를 찾는다.

    두 형태를 본다:

    1. `... status ... == "ok"` (또는 `!=`) — 비교
    2. **`status.get("ok")` / `status["ok"]`** — 조회

    ⛔ 2 번을 빼먹었더니 `validate_table.py` 가 `status.get("ok") / tot < 0.80`
       으로 게이트하는 것을 **다섯 번째 사례로 놓쳤다.** `ok` 비율은
       하드웨어에 의존해서(A6000 10.65 % vs 5090 22.21 %), 그것으로 게이트하면
       **더 빠른 GPU 일수록 표가 나쁘다고 판정한다.**
       검사가 잡는 형태가 좁으면 같은 실수가 다른 문법으로 되돌아온다.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            consts = [c.value for c in [node.left, *node.comparators]
                      if isinstance(c, ast.Constant)]
            if "ok" in consts:
                src = ast.dump(node)
                if "'status'" in src or "attr='status'" in src:
                    yield node
            continue
        # status.get("ok") / status.get("ok", 0)
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "ok"
                and "status" in ast.dump(node.func.value)):
            yield node
            continue
        # status["ok"]
        if (isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant)
                and node.slice.value == "ok"
                and "status" in ast.dump(node.value)):
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


# --------------------------------------------------------------------------
# C-3 — 축 목록은 안전장치다
# --------------------------------------------------------------------------

def test_백엔드를_직접_import_하지_않는다():
    """호출부는 `get_backend()` 만 쓴다.

    `check_axis_coverage.py` 가 축 목록이 필요해서 `backends.cutlass_v2` 을
    직접 import 했었다. Protocol 에 `axis_space()` 를 넣어 없앴다.
    """
    #: 백엔드 **자체를 검증하는** 스크립트. cutlass_v2 의 예측 함수가 실측과
    #: 맞는지 보는 것이 목적이라 그 함수를 직접 부를 수밖에 없다.
    allowed = {"scripts/validate_constraints.py"}
    bad = []
    for f in _py_files():
        rel = str(f.relative_to(REPO))
        if rel.startswith("kerneltab/backends/") or rel in allowed:
            continue
        for node in ast.walk(_tree(f)):
            mod = ""
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
            elif isinstance(node, ast.Import):
                mod = ",".join(a.name for a in node.names)
            if "backends.sm" in mod:
                bad.append(f"{_where(f, node)}  {mod}")
    assert not bad, (
        "백엔드를 직접 import 한다:\n  " + "\n  ".join(bad)
        + "\nget_backend(hw.arch) 를 써라. 필요한 것이 없으면 Protocol 에 "
          "메서드를 더한다 (decisions.md 12).")


def test_axis_space_가_모듈_상수와_같다():
    """`axis_space()` 가 손으로 적은 사본이 되면 걸린다.

    축 목록은 탐색 범위이면서 **안전장치**다 (`stages=1` — decisions.md
    13-b). 두 벌이 되면 한쪽만 고쳐진다.
    """
    from kerneltab.backends import cutlass_v2, get_backend

    sp = get_backend("sm_86").axis_space()
    assert sp["stages"] == set(cutlass_v2.STAGES)
    assert sp["split_k"] == set(cutlass_v2.SPLIT_K)
    assert sp["warp_tile"] == {tuple(t) for t in cutlass_v2.WARP_TILES}
    assert sp["tb_tile"] == {tuple(t) for t in cutlass_v2.TB_TILES}


def test_stages_1_은_축에_없다():
    """`stages=1` 은 컴파일도 되고 `can_implement()` 도 통과하는데 **결과가
    틀린다** (65,536 원소 중 62,674 개 불일치, 최대 상대오차 32).

    열거기의 `explain_kernel()` 도 `valid` 로 판정한다 — 이 목록이 유일한
    방어다. 근거는 `docs/axis_coverage.md`, `decisions.md` 13-b.
    """
    from kerneltab.backends import cutlass_v2

    assert 1 not in cutlass_v2.STAGES, (
        "stages=1 을 축에 넣었다. CUTLASS 2.x OpClassTensorOp 에서 수치가 "
        "틀린다 — can_implement() 는 통과시키므로 열거기로는 못 거른다.")


def test_축_덮개_KNOWN_에는_근거가_있다():
    """`KNOWN` 에 "확인했다" 만 적고 넘어가는 것을 막는다."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_cac", REPO / "scripts" / "check_axis_coverage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for key, why in mod.KNOWN.items():
        assert len(why) > 120, f"{key} 의 근거가 너무 짧다: {why!r}"
        assert "확인" in why or "불일치" in why, f"{key} 에 확인 내용이 없다"


def test_축을_순회하는_스크립트가_축을_다시_적지_않는다():
    """축 값을 리터럴로 다시 적으면 `SPLIT_K` 를 고쳐도 그쪽만 옛날에 남는다.

    실제로 밟았다. 2026-08-28 에 축 덮개 점검이 찾아낸 `split_k` 5 와 7 을
    `SPLIT_K` 에 넣었는데 `smoke_splitk.py` 는
    `for sk in (1, 2, 3, 4, 6, 8, 12, 16)` 를 그대로 들고 있었다. **스모크가
    새 축을 한 번도 안 돌리고 "이상 없음" 을 찍었다.**

    `stages=1` 사례가 보여주듯 축 목록은 탐색 범위이면서 안전장치다. 축을
    순회하는 곳은 반드시 `backend.axis_space()` 에서 읽어야 한다
    (decisions.md 12 — 같은 판정이 여러 곳에 있으면 하나는 어긋난다).
    """
    from kerneltab.backends import cutlass_v2

    axes = {
        "split_k": set(cutlass_v2.SPLIT_K),
        "stages": set(cutlass_v2.STAGES),
    }
    bad = []
    for f in _py_files():
        rel = str(f.relative_to(REPO))
        if rel.startswith("kerneltab/backends/") or rel.startswith("tests/"):
            continue
        for node in ast.walk(_tree(f)):
            if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                continue
            vals = [e.value for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, int)]
            if len(vals) < 4 or len(vals) != len(node.elts):
                continue
            for name, axis in axes.items():
                # 축과 **거의** 같은 리터럴 = 축의 사본이 낡은 것
                if set(vals) < axis and len(vals) >= len(axis) - 3:
                    bad.append(f"{_where(f, node)}  {name} 축의 옛 사본? {vals}")
    assert not bad, (
        "축 값을 리터럴로 다시 적은 곳이 있다:\n  " + "\n  ".join(bad)
        + "\n  backend.axis_space()[<축>] 에서 읽어라. 축을 늘렸을 때 "
          "여기만 옛날에 남으면 새 값이 한 번도 안 돌아간다.")


def test_드리프트_감시는_가장_짧은_형상으로_한다():
    """감시 지표를 잘못 고르면 **오염을 보고도 못 본다.**

    A6000 은 4096³ 하나로 감시하며 "+5.06 % 니 견딜 만하다" 고 판단했는데,
    같은 시각 512³ 측정은 **+1380 % 오염**돼 있었다. 런치당 상수 오버헤드는
    긴 커널에서 안 보인다.

    그 교훈으로 `DRIFT_SHAPES` 에 작은 형상을 넣었는데, **`drift_check` 은
    작은 형상을 기록만 하고 판정을 이끄는 반환값은 큰 형상이었다.** 커밋
    제목이 "작은 형상 감시" 였고 docstring 도 "돌려주는 값은 작은 형상" 이라고
    적혀 있었는데 코드가 반대였다 (2026-08-29, 5090 G-7 준비 중 발견).

    그래서 **감시 형상이 가장 짧은 것인지**를 코드로 고정한다.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_reh", REPO / "scripts" / "rehearse.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    def work(p):
        return p.M * p.N * p.K

    assert m.DRIFT_MONITOR_SHAPE in m.DRIFT_SHAPES
    smallest = min(m.DRIFT_SHAPES, key=work)
    assert m.DRIFT_MONITOR_SHAPE == smallest, (
        f"드리프트 감시 형상이 가장 짧은 것이 아니다: "
        f"{m.DRIFT_MONITOR_SHAPE} (가장 짧은 것은 {smallest}).\n"
        "  런치당 상수 오버헤드는 긴 커널에서 안 보인다 — "
        "A6000 이 4096³ 로 감시하다 512³ 의 +1380% 오염을 놓쳤다.")

    # 그리고 `drift_check` 이 그 형상의 값을 돌려주는지 소스로 확인한다.
    src = (REPO / "scripts" / "rehearse.py").read_text()
    fn = src[src.index("def drift_check("):]
    fn = fn[:fn.index("\ndef ", 1)]
    assert "return m_mon.time_ms" in fn, (
        "drift_check 이 감시 형상의 값을 돌려주지 않는다. "
        "기록만 하고 판정을 큰 형상으로 하면 감시가 둔감해진다.")


def test_드리프트_기준값이_감시와_같은_형상이다():
    """⛔ 기준값과 감시값의 **형상이 다르면** 비율이 통째로 무의미해진다.

    `drift_check()` 은 `DRIFT_MONITOR_SHAPE`(가장 짧은 것)를 돌려주는데,
    누적 드리프트의 기준값은 소킹이 남긴 `soak_ref_last_ms` 였고 그것은
    `_probe_ref()` 의 기본값 = `DRIFT_SHAPE`(가장 긴 것)로 잰 값이다.
    2048³ 대 4096³ 이면 일의 양이 8 배 다르므로:

        drift_ratio = t_drift / drift_base ~= 0.12       <- 1.0 이 아니다
        |t - base| / base ~= 87 %  >  DRIFT_ABS_WARN(8 %)

    누적 경고는 스트라이크가 아니라 **래치**라서 점검마다 다시 찍힌다 —
    10 분 주기 24 시간이면 96 회다. 그리고 `drift_ratio` 는 모든 측정
    줄에 실려 `table.parquet` 까지 나간다 (`core/table.py` 의 D-3 열).

    `DRIFT_MONITOR_SHAPE` 를 도입할 때(31d5926) 반환값만 바꾸고 기준값
    쪽을 안 고쳐서 생겼다 — `decisions.md` 3 (같은 값이 여러 곳에 살면
    하나는 어긋난다). 소킹이 기본 비활성이라 드러나지 않고 있었다.
    """
    src = (REPO / "scripts" / "rehearse.py").read_text(encoding="utf-8")

    # (a) 소킹이 감시 형상의 값을 남기는가
    soak = src[src.index("def thermal_soak("):]
    soak = soak[:soak.index("\ndef ", 1)]
    assert "soak_ref_monitor_last_ms" in soak, (
        "thermal_soak 이 감시 형상(DRIFT_MONITOR_SHAPE)으로 잰 기준값을 "
        "남기지 않는다. 긴 형상 값만 남기면 드리프트 기준으로 쓸 수 없다.")
    assert "DRIFT_MONITOR_SHAPE" in soak, (
        "thermal_soak 이 감시 형상을 아예 재지 않는다.")

    # (b) 그리고 측정 루프가 **그 값을** 기준으로 쓰는가
    assert 'drift_base = soak_info.get("soak_ref_monitor_last_ms")' in src, (
        "drift_base 가 감시 형상의 기준값에서 오지 않는다.\n"
        "  soak_ref_last_ms 는 DRIFT_SHAPE(가장 긴 형상)로 잰 값이라 "
        "감시값과 비교하면 형상이 다른 두 값의 비가 된다.")
    assert 'soak_info.get("soak_ref_last_ms")' not in src, (
        "긴 형상의 소킹 기준값을 아직 어딘가에서 드리프트 기준으로 쓴다.")


def test_누적_드리프트_경고가_상태_전환에서만_찍힌다():
    """반복되는 경고는 감시가 아니라 소음이다.

    누적 드리프트 경고는 스트라이크가 아니라 래치다 — 조건이 지속되면
    점검마다 같은 줄이 나온다. 같은 이유로 `sw_power_cap` 경고를 이미 한
    번 걷어냈다("24시간 캠페인에서 144회 오경보가 나면 진짜 클럭 풀림을
    놓친다"). 억제하는 대신 **횟수를 요약에 남긴다** — 안 그러면 "경고가
    안 떴다" 가 정상과 감시 죽음을 다 뜻하게 된다 (decisions 27).
    """
    src = (REPO / "scripts" / "rehearse.py").read_text(encoding="utf-8")
    assert "drift_abs_latched" in src, (
        "누적 드리프트 경고가 래치 상태를 들고 있지 않다 — 조건이 지속되면 "
        "점검마다 반복해서 찍힌다.")
    assert "drift_abs_overs" in src, "억제한 초과 횟수를 세지 않는다."
    # 요약과 하트비트 양쪽에 남아야 한다.
    tail = src[src.index("    print(\"\\n\" + probe.report())"):]
    assert "drift_abs_overs" in tail[:1200], (
        "억제한 경고 횟수가 슬라이스 요약에 안 나온다.")
    assert "drift_abs_overs=drift_abs_overs" in src, (
        "하트비트에 안 남는다 — 감시자가 읽을 수 없다.")


def test_재현성_보고가_흔들린_조합을_제대로_지목한다():
    """⛔ 값은 맞는데 **이름이 틀린** 보고서였다.

    `recheck_stability.py` 의 변동폭 계산이 `main()` 안에 인라인으로 있었고
    루프 변수가 `_key` 인데 결과에는 **바깥 스코프의 `key`**(측정 루프가 남긴
    마지막 조합)를 담았다. 그래서 상위 8줄이 전부 같은 이름을 달고 나왔다 —
    시간이 0.1485 ms 와 4.1667 ms 로 다른데 이름은 같았다.

    판정(`over_5pct`)은 값에서 나오므로 멀쩡했고, 그래서 게이트는 통과했다.
    **틀린 것은 "무엇이 흔들렸는가" 뿐인데 그것이 이 보고서를 읽는 이유다.**
    5 % 를 넘는 조합이 나오면 엉뚱한 커널을 쫓게 된다 (`decisions.md` 23).

    인라인이라 테스트를 붙일 자리가 없었다 — 그래서 `spread_rows()` 로 뺐다.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_recheck", REPO / "scripts" / "recheck_stability.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    samples = {
        ("kA", 64, 64, 64, 1, "serial"): [1.00, 1.00, 1.00],       # 0 %
        ("kB", 128, 128, 128, 2, "parallel"): [1.00, 1.10, 1.05],  # 10 %
        ("kC", 256, 256, 256, 4, "serial"): [2.00, 2.02, 2.01],    # 1 %
        ("kD", 512, 512, 512, 8, "serial"): [5.0],                 # 표본 부족
    }
    rows = m.spread_rows(samples)

    assert len(rows) == 3, "표본이 2개 미만인 조합은 빠져야 한다"
    assert [r[1][0] for r in rows] == ["kB", "kC", "kA"], (
        f"변동폭 순서나 이름이 틀렸다: {[(r[0], r[1][0]) for r in rows]}")
    assert abs(rows[0][0] - 0.10 / 1.05) < 1e-9
    # ★ 이름이 값과 같은 행에 붙어 있는가 (옛 버그는 여기서 전부 같았다)
    assert len({r[1][0] for r in rows}) == 3, (
        "여러 조합이 같은 이름을 달고 나온다 — 바깥 스코프의 key 를 담고 있다")


class TestDriftSummaryGroupsByKernel:
    """★ 드리프트 요약은 **커널로도 나눠야** 한다.

    같은 `env_hash` 안에서도 드리프트 프로브 커널이 바뀌면 절대 시간이
    달라진다 (세그먼트마다 자기 커널 목록에서 고른다). 나누지 않으면
    서로 다른 것을 잰 두 값의 차이가 "드리프트" 로 보고된다.

    H100 G-7 실측: 커널 A 79행 0.83 % + 커널 B 1행 -> 합치면 64.23 %.
    `f7ad211`(기준값과 감시값의 형상이 달랐다)과 같은 계열이다.
    """

    def test_요약이_커널로_나눈다(self):
        import re
        from pathlib import Path
        src = Path(__file__).resolve().parent.parent / "scripts" / "rehearse.py"
        t = src.read_text()
        blk = t[t.index("if DRIFT.exists():"):]
        blk = blk[:blk.index("tel = analyze_telemetry()")]
        assert "by_kernel" in blk, (
            "드리프트 요약이 커널로 나누지 않는다 — 다른 커널의 절대 시간이 "
            "섞여 오경보가 난다")
        # env_hash 필터도 그대로 있어야 한다 (둘 다 필요하다)
        assert "load_records(DRIFT, _eh)" in blk
        assert re.search(r"max\(by_kernel\.items\(\)", blk), (
            "행이 가장 많은 커널을 골라야 한다")

    def test_섞이면_변동폭이_부풀려진다(self):
        """수정의 근거를 수치로 고정한다."""
        a = [1.3264 + 0.0001 * i for i in range(79)]      # 커널 A
        b = [2.1785]                                       # 커널 B
        span = lambda v: (max(v) - min(v)) / (sum(v) / len(v))
        assert span(a) < 0.05                              # 진짜 드리프트
        assert span(a + b) > 0.5                           # 섞으면 오경보


class TestBufferPreallocation:
    """★ 버퍼는 **최대 크기로 미리 잡는다** — 고수위 재할당이 조건을 가른다.

    `Buf::ensure` 는 더 큰 형상을 만나면 free 후 재할당한다. 그러면 한
    프로세스 안에서 큰 형상 **전/후**의 측정 조건이 달라지고, 짧은 메모리
    바운드 커널(M<=128)이 5.4 % 흔들린다. 셔플 때문에 어느 쪽에서 재는지가
    우연히 갈리며 그 구분은 데이터에 남지 않는다.

    H100 실측 (M=1,N=11008,K=4096, sk16 serial, 79 us):
        작은 버퍼 0.0786 / 큰 버퍼 0.0829  (각 구간 안은 0.1~0.65 % 로 안정)
    G-7 5 번을 실패시킨 원인이고, 영향 행은 3.79 % (그중 M<=128 은 0.77 %).

    ⚠️ 지연 dlopen 은 **원인이 아니다** — 프로세스를 분리해 A/B 로 재서
       기각했다 (5.23/5.89 % vs 5.41/5.33 %). 같은 프로세스에서 A 다음 B 를
       돌리면 B 가 A 의 모듈 상태를 물려받아 잘못된 결론이 난다.
    """

    def test_세그먼트_시작에_최대_형상으로_잡는다(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "scripts"
               / "rehearse.py").read_text()
        i = src.index("probe = NvmlProbe(")
        blk = src[i:i + 2000]
        assert "ctx.prepare_problem(_bm, _bn, _bk)" in blk, (
            "세그먼트 시작에 최대 형상으로 선할당하지 않는다 — 큰 형상 전후의 "
            "측정 조건이 갈린다")
        assert "max(q.M for q in shapes)" in blk
        # ⛔ 리비전 2 는 절반만 덮었다 — workspace 는 같은 고수위 Buf 인데
        #    prepare_problem 이 안 잡는다. parallel split-K 에서는 세마포어가
        #    아니라 GEMM 의 D(부분합 버퍼)로 최대 8 GiB 까지 커진다.
        assert "ctx.buffers(_ws, parallel=False)" in blk, (
            "workspace 를 선할당하지 않는다 (MEASURE_PATH_REVISION 3)")

    def test_recheck_도_같은_상태에서_잰다(self):
        """★ 재현성 도구가 스윕과 다른 버퍼 상태에서 재면 안 된다.

        `recheck_stability.py` 만 선할당을 안 하면 pass1 에서 workspace 가
        가장 큰 parallel split-K 조합에 **상수 0.58 ms** 가 붙는다
        (10.67 ms 기준 5.5 %, 14.64 ms 기준 3.9 % — 비율이 아니라 절대량이
        같다). 별도 프로세스 두 회차에서 pass1 이 0.03 % 이내로 재현됐다 —
        표본이 아니라 계통이다. G-7 5 번이 이 불일치로 실패했다.
        """
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "scripts"
               / "recheck_stability.py").read_text()
        i = src.index("probe = NvmlProbe(")
        blk = src[i:i + 2500]
        assert "ctx.prepare_problem(_bm, _bn, _bk)" in blk, (
            "recheck 가 선할당을 안 한다 — 스윕과 다른 경로로 잰다")
        assert "ctx.buffers(_ws, parallel=False)" in blk, (
            "recheck 가 workspace 를 선할당하지 않는다")
        # ★ 버리는 사전 pass — 스윕은 슬라이스마다 618 개 커널을 돌린 상태에서
        #   재는데 recheck 는 차가운 프로세스에서 시작한다. 커널 하나가
        #   다른 커널을 그 프로세스 안에서 4.7 % 빠르게 만든다 (실측).
        assert "[예열]" in src, (
            "recheck 에 버리는 사전 pass 가 없다 — pass1 만 다른 장치 상태에서 "
            "재게 되고 G-7 5 번이 그 때문에 실패한다")

