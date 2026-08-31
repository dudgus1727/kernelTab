"""축 덮개 점검의 **3.x 검출기가 살아 있는가.**

이 검출기는 정상일 때 **0건**이다 — `target=CUTLASS` 는 2026-09-01 확인
시점에 CUTLASS 3.x 파라미터를 하나도 안 냈다. 0건인 감시는 자기가 살아
있는지 스스로 증명하지 못하므로, **되돌려서 잡는지**를 여기서 고정한다
(`decisions.md` 27).

심는 값은 지어낸 것이 아니라 `target=CUTLASS3` 가 RTX 5090 에 실제로 낸
`cluster(1 4)`, H100 에 낸 wgmma `instr(64 8 16)` 이다.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "check_axis_coverage.py"


def _mod():
    spec = importlib.util.spec_from_file_location("_cac", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(payload: dict, tmp_path: Path) -> subprocess.CompletedProcess:
    f = tmp_path / "vendor.json"
    f.write_text(json.dumps(payload))
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--vendor", str(f)],
        capture_output=True, text=True, cwd=REPO)


#: 벤더가 실제로 내는 2.x 추천 하나 (A6000, 1x4096x4096).
CLEAN = {
    "stages": 6, "cta": [128, 128, 32], "warp": [64, 64, 32],
    "split_k": 2, "swizzle": 1, "cta_order": 0,
    "cluster": [1, 1], "instr": [16, 8, 16],
    "raw": "layout(TN_ROW) stages(6) cta(128 128 32) warp(64 64 32) "
           "instr(16 8 16) splitK(2) swizz(1) ctaOrder(0) cluster(1 1)",
}


def test_2x_추천만_있으면_통과한다(tmp_path):
    r = _run({"_meta": {"gpu": "t"}, "1x4096x4096": [CLEAN]}, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "3.x" not in r.stdout


@pytest.mark.parametrize("field,value,why", [
    ("cluster", [1, 4], "CUTLASS3 가 RTX 5090 에 실제로 낸 cluster"),
    ("instr", [64, 8, 16], "CUTLASS3 가 H100 에 실제로 낸 wgmma instr"),
])
def test_3x_파라미터를_심으면_exit_5(tmp_path, field, value, why):
    """★ 통과와 검출은 다르다. 심어서 잡히는 것까지 확인한다."""
    bad = dict(CLEAN, **{field: value})
    r = _run({"_meta": {"gpu": "t"}, "1x4096x4096": [CLEAN, bad]}, tmp_path)
    assert r.returncode == 5, f"{why} 를 못 잡았다\n{r.stdout}{r.stderr}"
    assert "3.x" in r.stdout
    assert "baseline_vendor" in r.stdout, "대응 방법을 안 알려준다"


def test_축_구멍_4_와_3x_5_를_구분한다(tmp_path):
    """사유가 다르므로 코드도 다르다. 그리고 둘 다면 **5 가 이긴다.**"""
    hole = dict(CLEAN, stages=99)          # 축에 없는 값 (근거 없음)
    r4 = _run({"_meta": {}, "1x4096x4096": [hole]}, tmp_path)
    assert r4.returncode == 4, r4.stdout

    both = dict(hole, cluster=[1, 4])
    r5 = _run({"_meta": {}, "1x4096x4096": [both]}, tmp_path)
    assert r5.returncode == 5, (
        "공간 자체가 다르면 축 구멍 판정을 그대로 믿을 수 없다\n" + r5.stdout)


def test_옛_JSON_은_raw_에서_읽는다(tmp_path):
    """`--extract` 가 cluster 를 기록하기 전(2026-09-01) 파일도 감시된다.

    `.get("cluster")` 로만 보면 옛 파일이 **조용히 통과한다** — 그것이
    "경고가 안 뜬다" 의 나쁜 쪽이다.
    """
    old = {k: v for k, v in CLEAN.items() if k not in ("cluster", "instr")}
    assert _run({"_meta": {}, "s": [old]}, tmp_path).returncode == 0

    old_bad = dict(old, raw=old["raw"].replace("cluster(1 1)", "cluster(1 4)"))
    r = _run({"_meta": {}, "s": [old_bad]}, tmp_path)
    assert r.returncode == 5, "raw 에 있는 cluster(1 4) 를 놓쳤다\n" + r.stdout


def test_읽을_수_없으면_모른다고_말한다(tmp_path):
    """None 은 '3.x 가 아니다' 가 아니라 '모른다' 다."""
    blind = {"stages": 6, "cta": [128, 128, 32], "warp": [64, 64, 32]}
    r = _run({"_meta": {}, "s": [blind]}, tmp_path)
    assert r.returncode == 2, r.stdout
    assert "돌 수 없다" in r.stdout

    assert _mod().three_x_params(blind) is None


def test_커밋된_벤더_JSON_에_3x_가_없다():
    """저장소에 들어 있는 실제 추출물로도 확인한다."""
    files = sorted((REPO / "docs" / "baselines").glob("vendor_*.json"))
    assert files, "벤더 추출물이 없다"
    tx = _mod().three_x_params
    for f in files:
        data = json.loads(f.read_text())
        data.pop("_meta", None)
        for shape, recs in data.items():
            for rec in recs:
                got = tx(rec)
                assert got is not None, f"{f.name} {shape}: cluster/instr 불명"
                cluster, instr = got
                assert cluster in (None, (1, 1)), f"{f.name} {shape}: {cluster}"
                assert instr in (None, (16, 8, 16)), f"{f.name} {shape}: {instr}"
