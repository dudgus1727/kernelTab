"""공용 픽스처 + **스킵 감시**. GPU 를 전혀 쓰지 않는다.

## 왜 스킵을 실패로 만드는가 (R-1)

`test_table.py` 와 `test_bundle.py` 는 최상단에서 `importorskip("pyarrow")`
를 한다. pyarrow 가 없는 환경에서는 **두 모듈 41개가 통째로 안 돌고**
요약에는 "2 skipped" 로만 나온다. 초록불이다.

하필 그 둘이 **정답 누출 방지를 검증하는 것들**이다. `test_table.py` 의
docstring 은 이렇게 시작한다 — "문서는 지켜지지 않으므로 코드로 강제한다.
그렇다면 그 코드를 지키는 것도 문서여서는 안 된다." 그런데 그 파일이
도는지가 **환경에 pyarrow 가 깔려 있느냐**에 달려 있었다. 문서로 강제하는
것과 다를 바 없다.

이 저장소가 같은 클래스를 세 번 만났다:

| 사례 | 증상 |
|---|---|
| `WARMUP_SECONDS` | 정의만 되고 안 쓰임 — 로그는 "워밍업 한다" 고 찍힘 |
| `MEM_CLOCK_MIN_FRAC` | 주석에 기준만 있고 미구현 |
| `test_table.py` | 스킵되는데 초록불 |

공통점은 **"조건이 안 맞으면 조용히 아무것도 하지 않는다"** 이다.
그래서 스킵 자체를 막지 않고 **스킵됐다는 사실을 실패로 만든다.**

의도적으로 넘기려면 `KERNELTAB_ALLOW_SKIP=1` 을 준다.
"""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backends import get_backend
from core.types import Hardware, KernelConfig

#: 이 모듈들이 안 돌면 **실패**다. 정답 누출 방지를 검증하는 것들이라
#: 조용히 스킵되면 "누출 방지 검증됨" 을 거짓으로 믿게 된다.
CRITICAL_MODULES = {"test_table.py", "test_bundle.py"}
ALLOW_SKIP_ENV = "KERNELTAB_ALLOW_SKIP"

_seen: dict[str, dict] = {}


def pytest_runtest_logreport(report):
    """모듈별로 '무엇이든 실제로 돌았는가' 를 센다."""
    if report.when == "call" or (report.when == "setup" and report.skipped):
        d = _seen.setdefault(Path(str(report.fspath)).name,
                             {"ran": 0, "skipped": 0})
        if report.skipped:
            d["skipped"] += 1
        else:
            d["ran"] += 1


def _bad_modules() -> list[str]:
    bad = []
    for mod in sorted(CRITICAL_MODULES):
        d = _seen.get(mod)
        if d is None or (d["ran"] == 0 and d["skipped"] == 0):
            bad.append(f"{mod}: 수집되지 않았다 "
                       f"(모듈 최상단 importorskip 이 걸렸거나 파일이 없다)")
        elif d["ran"] == 0:
            bad.append(f"{mod}: {d['skipped']}개가 전부 스킵됐다 — 실제로 돈 것이 0개")
    return bad


class _SkipGuardItem(pytest.Item):
    """맨 마지막에 도는 합성 항목.

    `pytest_sessionfinish` 에서 `session.exitstatus` 를 바꾸는 방법은
    pytest 버전에 따라 **전파되지 않는다** (실제로 메시지만 나오고 exit 0
    이었다). 감시가 종료 코드로 이어지지 않으면 CI 에서 무의미하므로,
    진짜 테스트 항목으로 만들어 정상적인 실패 경로를 탄다.
    """

    def runtest(self):
        bad = _bad_modules()
        if not bad:
            return
        msg = ("중요 테스트 모듈이 실제로 돌지 않았다\n"
               + "\n".join("  - " + b for b in bad) + "\n\n"
               "이 모듈들은 **정답 누출 방지**를 검증한다. 스킵된 채 초록불이\n"
               "뜨면 '누출 방지 검증됨' 을 거짓으로 믿게 된다.\n\n"
               "  고치는 법:            pip install -e '.[test]'\n"
               f"  의도적으로 넘기려면:  {ALLOW_SKIP_ENV}=1 pytest")
        if os.environ.get(ALLOW_SKIP_ENV) == "1":
            print("\n[경고] " + msg.replace("\n", "\n[경고] "))
            pytest.skip(f"{ALLOW_SKIP_ENV}=1 로 우회 — 이 실행 결과로 "
                        "누출 방지를 보증하지 마라")
        raise AssertionError(msg)

    def repr_failure(self, excinfo, style=None):
        return str(excinfo.value)

    def reportinfo(self):
        return self.path, 0, "스킵 감시 (R-1)"


def pytest_collection_modifyitems(session, config, items):
    """중요 모듈이 실제로 돌았는지 확인하는 항목을 **맨 뒤에** 붙인다."""
    for it in items:
        _seen.setdefault(Path(str(it.fspath)).name, {"ran": 0, "skipped": 0})
    if config_filtered(config):
        return          # -k / 특정 파일 지정 실행에는 적용하지 않는다
    items.append(_SkipGuardItem.from_parent(
        session, name="test_critical_modules_actually_ran"))


def config_filtered(config) -> bool:
    """-k / -m / 특정 파일 지정으로 **일부만** 고른 실행인가.

    invocation_params.args 를 직접 훑으면 안 된다 — `--deselect X` 의 X 나
    `-p no:cacheprovider` 의 값처럼 대시로 시작하지 않는 **옵션 값**까지
    위치 인자로 오해한다 (그 버그로 감시가 통째로 무력화됐었다).
    pytest 가 이미 파싱해 둔 것을 쓴다.
    """
    opt = config.option
    if getattr(opt, "keyword", "") or getattr(opt, "markexpr", ""):
        return True
    targets = [str(x) for x in getattr(opt, "file_or_dir", []) or []]
    return any(t.rstrip("/") not in ("", ".", "tests") for t in targets)


@pytest.fixture(scope="session")
def backend():
    return get_backend("sm_86")


@pytest.fixture
def hw_a6000():
    """A6000 을 1350 MHz / 7601 MHz 로 고정했을 때의 실효 스펙."""
    return Hardware(
        name="NVIDIA RTX A6000", arch="sm_86", sm_count=84,
        smem_per_block=101376, max_threads_per_sm=1536, regs_per_sm=65536,
        peak_tflops_f16=116.1, bandwidth_gbps=729.7, l2_bytes=6291456,
    )


@pytest.fixture
def hw_other():
    """SM 개수만 다른 가상 GPU. 하드웨어 상수 하드코딩 검출용."""
    return Hardware(
        name="FAKE", arch="sm_86", sm_count=128,
        smem_per_block=65536, max_threads_per_sm=2048, regs_per_sm=65536,
        peak_tflops_f16=200.0, bandwidth_gbps=1000.0, l2_bytes=4194304,
    )


@pytest.fixture
def mk_cfg(backend):
    """KernelConfig 를 간단히 만드는 헬퍼."""
    def _mk(tile=(128, 128, 32), warp=(64, 64, 32), stages=4,
            swizzle=("identity", 8), align=(8, 8, 8), arch="sm_86"):
        ext = backend.ext_from_dict({
            "warp_m": warp[0], "warp_n": warp[1], "warp_k": warp[2],
            "stages": stages, "swizzle_type": swizzle[0],
            "swizzle_n": swizzle[1]})
        return KernelConfig(tile[0], tile[1], tile[2],
                            align[0], align[1], align[2], arch, ext)
    return _mk
