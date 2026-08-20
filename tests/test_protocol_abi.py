"""ctypes 구조체가 `kt_abi.h` 의 C 구조체와 **정확히** 같은가.

어긋나면 예외가 나지 않는다. 필드가 밀려서 **쓰레기 값이 조용히** 들어가고,
프로토콜이 엉뚱한 값으로 측정된다. `docs/decisions.md` 14 의 "조용히 잘못된
값으로 진행" 이다.

정적 분석으로는 못 잡는다 — Python 쪽과 C 쪽이 따로 있기 때문이다. 그래서
**헤더를 파싱해서** 대조한다.
"""
from __future__ import annotations

import ctypes
import re
from pathlib import Path

import pytest

from kerneltab.measure.runner import KtBuffersC, KtMeasureC, KtProblemC, KtProtocolC

ABI_H = Path(__file__).resolve().parent.parent / "kerneltab" / "measure" / "kt_abi.h"

#: C 타입 -> ctypes 타입
_CTYPE = {
    "double": ctypes.c_double,
    "float": ctypes.c_float,
    "int": ctypes.c_int,
    "long long": ctypes.c_longlong,
    "size_t": ctypes.c_size_t,
    "void *": ctypes.c_void_p,
}


def _parse(name: str) -> list[tuple[str, str]]:
    """`typedef struct <name> { ... } <name>;` 에서 `(타입, 필드명)` 목록."""
    src = ABI_H.read_text()
    m = re.search(r"typedef struct " + name + r"\s*\{(.*?)\}\s*" + name + r"\s*;",
                  src, re.DOTALL)
    assert m, f"{name} 을 kt_abi.h 에서 찾지 못했다"
    out = []
    for line in m.group(1).splitlines():
        line = line.split("//")[0].strip()
        if not line or not line.endswith(";"):
            continue
        decl = line[:-1].strip()
        # `int M, N, K;` 처럼 한 줄에 여러 개 선언된 경우를 편다.
        fm = re.match(r"^(.*?)([A-Za-z_][A-Za-z0-9_,*\s]*)$", decl)
        assert fm, f"파싱 실패: {decl}"
        head, names = fm.group(1), fm.group(2)
        parts = [x.strip() for x in names.split(",")]
        # 첫 항목에는 타입이 붙어 있다: "int M" -> ("int", "M")
        first = (head + parts[0]).strip()
        fm2 = re.match(r"^(.*?)([A-Za-z_][A-Za-z0-9_]*)$", first)
        assert fm2, f"파싱 실패: {first}"
        ctype = fm2.group(1).strip()
        out.append((ctype, fm2.group(2)))
        for extra in parts[1:]:
            out.append((ctype, extra.lstrip("*").strip()))
    return out


CASES = [
    ("KtProblem", KtProblemC),
    ("KtBuffers", KtBuffersC),
    ("KtProtocol", KtProtocolC),
    ("KtMeasure", KtMeasureC),
]


@pytest.mark.parametrize(("cname", "pytype"), CASES)
def test_필드_이름과_순서가_같다(cname, pytype):
    c_fields = [f for _, f in _parse(cname)]
    py_fields = [f for f, _ in pytype._fields_]
    assert c_fields == py_fields, (
        f"{cname} 의 필드가 어긋난다. ctypes 는 조용히 밀린 값을 읽는다.\n"
        f"  kt_abi.h : {c_fields}\n"
        f"  runner.py: {py_fields}")


@pytest.mark.parametrize(("cname", "pytype"), CASES)
def test_필드_타입이_같다(cname, pytype):
    py = dict(pytype._fields_)
    bad = []
    for ctype, field in _parse(cname):
        want = _CTYPE.get(ctype)
        if want is None:
            continue                       # 배열 등 — 이름/순서 검사로 충분
        got = py.get(field)
        if got is not want:
            bad.append(f"{field}: C {ctype} vs ctypes {got}")
    assert not bad, f"{cname} 타입 불일치: {bad}"


def test_프로토콜_기본값이_전부_구조체에_있다():
    from kerneltab.measure.runner import PROTOCOL_DEFAULTS
    py = {f for f, _ in KtProtocolC._fields_}
    assert set(PROTOCOL_DEFAULTS) == py, (
        "PROTOCOL_DEFAULTS 와 KtProtocolC 필드가 다르다. 하나만 고치면 "
        "protocol_from_env() 가 KeyError 로 죽거나 필드가 0 으로 남는다.")


def test_시간_예산_필드가_있다():
    """워밍업 시간 예산화 (2026-08-20). 없으면 옛 동작으로 조용히 돌아간다."""
    py = {f for f, _ in KtProtocolC._fields_}
    assert {"probe_budget_ms", "warmup_budget_ms", "warmup_reps_floor"} <= py


def test_오버헤드_회계_필드가_있다():
    """`n_probe`/`n_warmup` 이 없으면 워밍업이 0 이 돼도 알 수 없다."""
    py = {f for f, _ in KtMeasureC._fields_}
    assert {"n_probe", "n_warmup", "overhead_ms"} <= py


def test_abi_버전이_헤더와_같다():
    """구조체를 고치면 KT_ABI_VERSION 을 **양쪽 다** 올려야 한다."""
    from kerneltab.measure.runner import KT_ABI_VERSION
    m = re.search(r"#define KT_ABI_VERSION\s+(\d+)", ABI_H.read_text())
    assert m, "kt_abi.h 에 KT_ABI_VERSION 이 없다"
    assert int(m.group(1)) == KT_ABI_VERSION, (
        f"kt_abi.h={m.group(1)} vs runner.py={KT_ABI_VERSION}. "
        "구조체를 고쳤으면 양쪽을 함께 올려라 — 안 그러면 옛 .so 가 조용히 붙는다.")


def test_abi_구조체_열거가_헤더와_같다():
    from kerneltab.measure.runner import Ctx
    src = ABI_H.read_text()
    order = re.findall(r"KT_ABI_([A-Z]+)\s*=\s*(\d+)", src)
    want = {int(v): k.lower() for k, v in order}
    got = {i: t.__name__.replace("Kt", "").replace("C", "").lower()
           for i, t in Ctx._ABI_STRUCTS}
    assert want == got, f"열거 순서가 다르다: {want} vs {got}"
