"""`hwspec/known.json` 스키마.

**항목을 추가하는 시점에 걸리는 것이 실행 시점에 걸리는 것보다 낫다.**

`peak_tflops_f16` 만 넣고 `peak_tflops_f16_at_mhz` 를 빠뜨리면, 클럭을
고정해도 `phase0_env.py` 가 **조용히 스펙 피크를 그대로 쓴다.** ridge point
가 틀리고 `is_memory_bound` 가 전 형상에서 틀린다 — 그런데 아무 일도 안
일어난 것처럼 보인다 (`docs/pending_fixes.md` D-1).

RTX 5090 은 그 값(2407 MHz)이 없었으면 실효 피크가 실제 234.3 대신 209.5 가
됐다. 지금은 `phase0_env` 가 실패하지만, **여기서 먼저 걸리는 것이 낫다.**
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
KNOWN = REPO / "hwspec" / "known.json"

#: 모든 GPU 항목이 반드시 들고 있어야 하는 키와 그 이유.
REQUIRED = {
    "peak_tflops_f16":
        "roofline 의 분자. dense FP16 입력 + FP32 누산 기준이다 "
        "(데이터시트 헤드라인은 희소 기준인 경우가 많다)",
    "peak_tflops_f16_at_mhz":
        "★ 그 피크가 성립하는 SM 클럭. 없으면 클럭 고정 시 보정이 "
        "생략되어 스펙 피크가 그대로 쓰인다 (D-1)",
    "bandwidth_gbps_at_mem_mhz":
        "대역폭 교차검증의 기준 메모리 클럭",
    "bandwidth_gbps_spec":
        "데이터시트 P0 대역폭. 교차검증 전용",
    "mem_bus_bits":
        "메모리 버스 폭. 실효 대역폭은 이것 x 2 x 관측 클럭으로 계산한다",
    "source":
        "이 숫자가 어디서 왔는가. 추정하면 결과 전체가 오염된다",
}


def _entries() -> dict:
    raw = json.loads(KNOWN.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def test_known_json_이_읽힌다():
    assert _entries(), f"{KNOWN} 에 GPU 항목이 없다"


@pytest.mark.parametrize("gpu", sorted(_entries()))
def test_필수_키가_전부_있다(gpu):
    entry = _entries()[gpu]
    missing = {k: why for k, why in REQUIRED.items() if k not in entry}
    assert not missing, (
        f"'{gpu}' 에 빠진 키:\n  "
        + "\n  ".join(f"{k} — {why}" for k, why in missing.items()))


@pytest.mark.parametrize("gpu", sorted(_entries()))
def test_수치가_말이_되는가(gpu):
    e = _entries()[gpu]
    assert e["peak_tflops_f16"] > 0
    assert 100 <= e["peak_tflops_f16_at_mhz"] <= 5000, (
        f"'{gpu}' 의 기준 클럭 {e['peak_tflops_f16_at_mhz']} MHz 가 "
        "SM 클럭 범위 밖이다 — 메모리 클럭을 잘못 넣지 않았는가")
    assert e["mem_bus_bits"] in (128, 192, 256, 320, 384, 448, 512, 1024,
                                 2048, 4096, 5120, 6144, 8192), (
        f"'{gpu}' 의 버스 폭 {e['mem_bus_bits']} bit 가 통상값이 아니다")


@pytest.mark.parametrize("gpu", sorted(_entries()))
def test_대역폭_교차검증이_맞는다(gpu):
    """`bandwidth_gbps_spec` 과 `버스폭 x 2 x 기준클럭` 이 일치해야 한다.

    계산 경로가 하나여야 다른 GPU 에서 항목을 빼먹어도 조용히 틀리지 않는다.
    """
    e = _entries()[gpu]
    calc = e["mem_bus_bits"] * 2 * e["bandwidth_gbps_at_mem_mhz"] * 1e6 / 8 / 1e9
    spec = e["bandwidth_gbps_spec"]
    assert abs(calc - spec) / spec < 0.02, (
        f"'{gpu}': 버스 폭 {e['mem_bus_bits']}bit x 2 x "
        f"{e['bandwidth_gbps_at_mem_mhz']}MHz = {calc:.1f} GB/s 인데 "
        f"spec 은 {spec} GB/s 다. 둘 중 하나가 틀렸다.")


@pytest.mark.parametrize("gpu", sorted(_entries()))
def test_source_에_근거가_있다(gpu):
    """`source` 가 비어 있거나 형식적이면 추정한 값일 수 있다."""
    src = _entries()[gpu]["source"]
    assert len(src) > 40, (
        f"'{gpu}' 의 source 가 너무 짧다 ({len(src)}자). "
        "이 숫자가 어디서 나왔는지 — 계산식이든 실측이든 — 적어라. "
        "roofline 이 여기 직접 의존한다.")
