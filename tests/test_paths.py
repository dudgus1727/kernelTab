"""P-1 — 커널 `.so` 경로를 `kernel_id` 에서 조립한다.

`kernels.jsonl` 의 `so_path` 는 **빌드한 기계의 절대 경로**다. 컨테이너
안에서는 그 경로가 없고, 저장소를 옮기거나 볼륨 마운트 지점이 다르면
그대로 깨진다. 컨테이너화의 전제조건이다.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from kerneltab.build import paths

REPO = Path(__file__).resolve().parent.parent


class TestAssembly:
    def test_shape_of_path(self):
        p = paths.kernel_so("sm86_tb32x128x32_w16x64x32_st3_swid8_a448")
        assert p.name == "sm86_tb32x128x32_w16x64x32_st3_swid8_a448.so"
        assert p.parent == paths.ARTIFACT_DIR / "lib"

    def test_is_absolute_and_under_repo(self):
        p = paths.kernel_so("k")
        assert p.is_absolute()
        assert paths.ARTIFACT_DIR in p.parents

    def test_matches_recorded_so_path_when_present(self):
        """옛 줄의 `so_path` 와 **파일명 부분**이 일치해야 한다.

        ⚠️ 절대 경로 전체를 비교하지 않는다. 패키지 이전에서 산출물을
        `build/artifacts/` -> `artifacts/` 로 **의도적으로 옮겼기 때문**이다.
        옛 줄에 박힌 절대 경로는 그 이동으로 전부 무효가 됐고, 그것이 바로
        P-1 이 `so_path` 기록을 없앤 이유다.

        불변인 것은 `lib/<kernel_id>.so` 라는 **상대 구조**다. 이것이
        깨지면 조립 규칙이 바뀐 것이므로 옛 산출물을 못 찾는다.
        """
        f = paths.RESULTS_DIR / "kernels.jsonl"
        if not f.exists():
            pytest.skip("kernels.jsonl 이 없다")
        import json
        bad = n = 0
        with f.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                old = r.get("so_path")
                if not old:
                    continue
                n += 1
                new = paths.kernel_so(r["kernel_id"])
                if Path(old).parts[-2:] != new.parts[-2:]:
                    bad += 1
        if n == 0:
            pytest.skip("so_path 가 기록된 줄이 없다 (이미 제거됨)")
        assert bad == 0, f"{n}줄 중 {bad}줄 불일치"


class TestNoAbsolutePathInRecords:
    def test_compile_does_not_record_so_path(self):
        """새로 빌드하는 줄에는 `so_path` 가 없어야 한다."""
        src = (REPO / "kerneltab" / "build" / "compile.py").read_text()
        assert '"so_path": str(so)' not in src
        assert '"kernel_id": kid}' in src or '"kernel_id": kid,' in src

    def test_no_caller_reads_so_path(self):
        """읽는 쪽이 하나라도 남아 있으면 컨테이너에서 깨진다."""
        offenders = []
        for f in sorted((REPO / "scripts").glob("*.py")):
            src = f.read_text()
            for node in ast.walk(ast.parse(src)):
                # r["so_path"] 같은 첨자 접근을 찾는다
                if (isinstance(node, ast.Subscript)
                        and isinstance(node.slice, ast.Constant)
                        and node.slice.value == "so_path"):
                    offenders.append(f"{f.name}:{node.lineno}")
        assert not offenders, f"so_path 를 읽는 곳이 남아 있다: {offenders}"
