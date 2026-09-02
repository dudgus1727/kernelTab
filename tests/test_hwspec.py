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
    "peak_derivation":
        "★ peak_tflops_f16 을 만든 산식 {sm_count, fma_f16_per_clk_per_sm, "
        "boost_mhz}. 산문(source)에만 적으면 값과 어긋나도 아무도 모른다",
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


@pytest.mark.parametrize("gpu", sorted(_entries()))
def test_피크_교차검증이_맞는다(gpu):
    """`peak_tflops_f16` 과 `sm x fma x 2 x boost` 가 일치해야 한다.

    ⛔ 대역폭에는 이 교차검증이 있었는데(`mem_bus_bits`) **피크에는 없었다.**
       그래서 4090 항목의 `source` 산문이 `128 SM x 512 x 2 x 2.52 GHz`
       (= 330.3)라고 적혀 있는데 값은 165.2 인 상태가 통과했다. 값이 옳고
       산문이 틀린 경우였다 — `docs/decisions.md` 23 이 "가장 찾기 어렵다"
       고 적은 바로 그 형태다.

    산문을 grep 해서 판정하지 않는다 (`decisions.md` 24). 구조화된
    `peak_derivation` 을 두고 그것으로 검사한다.

    ⚠️ `fma_f16_per_clk_per_sm` 은 **FP32 누산 기준**이다. GeForce Ada /
       Blackwell 은 반감되어 256, 프로 SKU(A6000)는 512 다. 새 GPU 를
       넣을 때 반감 여부를 `mma.sync` 로 실측해 확정하라 — 여기서 512 와
       256 을 잘못 고르면 ridge point 가 2 배 틀린다.
    """
    e = _entries()[gpu]
    d = e["peak_derivation"]
    calc = d["sm_count"] * d["fma_f16_per_clk_per_sm"] * 2 * d["boost_mhz"] / 1e6
    peak = e["peak_tflops_f16"]
    assert abs(calc - peak) / peak < 0.01, (
        f"'{gpu}': {d['sm_count']} SM x {d['fma_f16_per_clk_per_sm']} "
        f"FMA/clk/SM x 2 flop x {d['boost_mhz']} MHz = {calc:.1f} TFLOP/s 인데 "
        f"peak_tflops_f16 은 {peak} 다. 둘 중 하나가 틀렸다.\n"
        "  2 배 어긋나면 FP32 누산 반감(GeForce = 256, 프로 SKU = 512)을 "
        "잘못 골랐을 가능성이 높다.")
    assert d["boost_mhz"] == e["peak_tflops_f16_at_mhz"], (
        f"'{gpu}': peak_derivation.boost_mhz({d['boost_mhz']}) 와 "
        f"peak_tflops_f16_at_mhz({e['peak_tflops_f16_at_mhz']}) 가 다르다. "
        "실효 피크 보정이 다른 클럭을 기준으로 계산된다.")


#: 실측으로 확인된 FMA/clk/SM 과 그 근거. **추정값을 넣지 마라.**
KNOWN_FMA = {
    256: "GeForce Ada/Blackwell — FP32 누산 반감",
    512: "A6000 프로 SKU — 반감 없음",
    1324: "H100(Hopper) mma.sync FP32 누산 실측. 반감은 없고 발행률이 다르다",
}


@pytest.mark.parametrize("gpu", sorted(_entries()))
def test_fma_가_알려진_값이다(gpu):
    """반감 여부만이 아니라 **명령 발행률**도 세대마다 다르다.

    ⛔ H100 에서 셋째 값이 나왔다. 반감은 없는데(FP32/FP16 비 1.031) 값이
       512 가 아니다 — Hopper 의 `mma.sync` 는 서브파티션당 HMMA.16816 을
       6 사이클에 하나가 상한이라 4 x 2048 / 6 = 1365 가 설계 상한이고
       FP32 누산 실측이 1324 다 (`tools/peak_mma_probe.cu`).

       ★ 이 표의 config 공간(2.x = mma.sync)이 도달할 수 있는 값이므로
       이것이 맞는 값이다. 데이터시트 피크는 wgmma(3.x)로만 나오며
       `peak_tflops_f16_datasheet` 에 기록만 한다
       (`docs/consumer_contract.md` 12 절).
    """
    fma = _entries()[gpu]["peak_derivation"]["fma_f16_per_clk_per_sm"]
    assert fma in KNOWN_FMA, (
        f"'{gpu}' 의 FMA/clk/SM 이 {fma} 다. 알려진 값은 "
        + ", ".join(f"{k}({v})" for k, v in KNOWN_FMA.items())
        + ". 새 값이면 mma.sync 실측 근거를 source 에 적고 이 목록을 넓혀라 "
        "— **추정하지 말고 tools/peak_mma_probe.cu 로 재라.**")
