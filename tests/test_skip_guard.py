"""R-1 메타 테스트 — 스킵 감시가 **실제로 실패를 잡는가**.

이 테스트가 없으면 감시 자체가 "조용히 아무것도 안 하는 안전장치" 가 된다.
그건 감시가 잡으려던 바로 그 병이다.

pyarrow 를 가린 하위 프로세스에서 pytest 를 돌려, 중요 모듈이 통째로
스킵될 때 종료 코드가 0 이 아닌지 확인한다.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

#: 하위 프로세스에서는 이 파일을 통째로 건너뛴다. --deselect 를 쓰면
#: pytest 가 "일부만 고른 실행" 으로 보게 되어 감시가 스스로 비활성화된다.
if os.environ.get("KERNELTAB_SKIPGUARD_CHILD") == "1":
    pytest.skip("하위 프로세스에서는 자기 자신을 돌리지 않는다",
                allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: pyarrow / pandas 가 **설치되어 있지 않은** 상황을 흉내낸다.
#: sys.modules 에 None 을 넣으면 import 시 깔끔한 ImportError 가 나고,
#: `pytest.importorskip` 이 그걸 잡아 스킵한다 — 실제 미설치와 같은 경로다.
#: meta_path 에서 예외를 던지면 스킵이 아니라 **수집 에러**가 되어
#: 재현하려는 상황(조용한 스킵)과 달라진다.
BLOCKER = textwrap.dedent("""
    import sys
    for _m in ('pyarrow', 'pandas'):
        sys.modules[_m] = None
""")


def _run_without_pyarrow(tmp_path, extra_env=None):
    site = tmp_path / "sitecustomize.py"
    site.write_text(BLOCKER)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("KERNELTAB_ALLOW_SKIP", None)
    env["KERNELTAB_SKIPGUARD_CHILD"] = "1"   # 자기 자신 재귀 방지
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600)


def test_guard_fails_when_critical_modules_skip(tmp_path):
    """pyarrow 가 없으면 **실패해야 한다.** 초록불이면 감시가 죽은 것이다."""
    r = _run_without_pyarrow(tmp_path)
    out = r.stdout + r.stderr
    assert "pyarrow" in out.lower() or "skip" in out.lower(), \
        f"pyarrow 를 실제로 가리지 못했다:\n{out[-2000:]}"
    assert r.returncode != 0, (
        "pyarrow 를 가렸는데도 pytest 가 통과했다. 스킵 감시가 동작하지 않는다.\n"
        + out[-3000:])
    assert "중요 테스트 모듈이 실제로 돌지 않았다" in out, \
        f"감시 메시지가 안 나왔다:\n{out[-3000:]}"
    assert "test_critical_modules_actually_ran" in out, \
        f"감시가 테스트 항목으로 실패하지 않았다:\n{out[-3000:]}"


def test_allow_skip_env_bypasses(tmp_path):
    """의도적 우회는 통과하되 **경고를 크게** 낸다."""
    r = _run_without_pyarrow(tmp_path, {"KERNELTAB_ALLOW_SKIP": "1"})
    out = r.stdout + r.stderr
    assert r.returncode == 0, f"우회가 동작하지 않는다:\n{out[-3000:]}"
    # 우회하면 감시 항목이 실패가 아니라 **스킵**으로 남아야 한다.
    # 조용히 통과하면 우회했다는 사실 자체가 안 보인다.
    assert "skipped" in out, f"우회 흔적이 없다:\n{out[-2000:]}"
    r2 = _run_without_pyarrow(tmp_path)
    assert r2.returncode != 0, "우회 없이는 실패해야 한다"


def test_critical_modules_actually_exist():
    """감시 목록이 실재하는 파일을 가리키는가. 오타 나면 감시가 무력해진다."""
    from tests.conftest import CRITICAL_MODULES
    for m in CRITICAL_MODULES:
        assert (REPO_ROOT / "tests" / m).exists(), f"{m} 이 없다"
