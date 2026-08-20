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


class TestNoStalePackagePaths:
    """패키지 이전 뒤 남은 `<repo>/build`, `<repo>/core` 같은 경로를 잡는다.

    `phase0_env.py` 가 `REPO_ROOT / "build" / "launch_probe.cu"` 를 그대로
    들고 있었다. 이전 후 그 경로는 없다. **호스트에서는 아무도 안 밟았고
    컨테이너에서 처음 터졌다** — `detect` 가 "빈 커널 런치 오버헤드" 단계에서
    죽었다.

    소스 트리를 옮길 때마다 이런 것이 남는다. grep 한 번이 아니라 **검사로
    고정한다** (`docs/decisions.md` 13 의 여덟 번째).
    """

    MOVED = ("build", "core", "measure", "backends")

    def test_no_repo_root_reference_to_moved_dirs(self):
        import re
        pat = re.compile(
            r'(?:paths\.)?REPO_ROOT\s*/\s*["\'](' + "|".join(self.MOVED) + r')["\']')
        offenders = []
        for d in ("scripts", "kerneltab", "tests"):
            for f in sorted((REPO / d).rglob("*.py")):
                if f.name == "test_paths.py":
                    continue          # 이 파일의 설명문이 패턴에 걸린다
                for i, line in enumerate(f.read_text().splitlines(), 1):
                    if pat.search(line):
                        offenders.append(f"{f.relative_to(REPO)}:{i}: {line.strip()}")
        assert not offenders, (
            "패키지 이전 뒤 남은 경로다. 이 디렉토리들은 kerneltab/ 아래로 "
            "옮겼으니 paths.PKG_ROOT 를 써라:\n  " + "\n  ".join(offenders))

    def test_pkg_root_files_exist(self):
        """`PKG_ROOT` 로 조립하는 실제 파일들이 있는지."""
        for rel in ("build/launch_probe.cu", "measure/kt_ctx.cu",
                    "measure/kt_abi.h"):
            assert (paths.PKG_ROOT / rel).exists(), f"{rel} 이 없다"


class TestKernelIncludes:
    """생성된 커널 `.cu` 는 CUTLASS 만으로 컴파일되지 않는다.

    `emit_cpp()` 결과가 `kt_swizzle.h` / `kt_abi.h` 를 include 한다.
    `check_smem.py` 가 CUTLASS 경로만 주고 있어서 40개 전부
    "BUILD FAIL / compilation terminated." 이 났다 — 그런데 그 메시지는
    `stderr[-800:]` 로 잘려서 **진짜 원인이 안 보였다.**
    """

    def test_includes_measure_dir(self):
        inc = paths.kernel_includes(paths.cutlass_dir(None)) \
            if _cutlass_available() else paths.kernel_includes(Path("/x"))
        assert any(str(paths.PKG_ROOT / "measure") in i for i in inc), (
            "kernel_includes 에 measure/ 가 없다 — kt_swizzle.h 를 못 찾는다")

    def test_emitted_source_needs_those_headers(self):
        """실제로 생성된 소스가 그 헤더를 요구하는지 확인한다."""
        from kerneltab.backends import get_backend
        from kerneltab.core.types import KernelConfig
        be = get_backend("sm_86")
        cfg = KernelConfig(128, 128, 32, 8, 8, 8, "sm_86",
                           be.ext_from_dict({"warp_m": 64, "warp_n": 64,
                                             "warp_k": 32, "stages": 3,
                                             "swizzle_type": "identity",
                                             "swizzle_n": 1}))
        src = be.emit_cpp(cfg)
        needed = [h for h in ("kt_swizzle.h", "kt_abi.h") if f'"{h}"' in src]
        assert needed, "emit_cpp 가 로컬 헤더를 안 쓴다면 이 검사는 낡았다"
        for h in needed:
            assert (paths.PKG_ROOT / "measure" / h).exists(), \
                f"{h} 가 kerneltab/measure/ 에 없다"

    def test_no_bare_cutlass_includes_for_kernel_builds(self):
        """커널을 컴파일하는 곳이 `cutlass_includes` 만 쓰면 안 된다."""
        import re
        offenders = []
        for f in sorted((REPO / "scripts").glob("*.py")) + \
                sorted((REPO / "kerneltab").rglob("*.py")):
            src = f.read_text()
            if "emit_cpp" not in src:
                continue
            for i, line in enumerate(src.splitlines(), 1):
                if re.search(r"paths\.cutlass_includes\(", line):
                    offenders.append(f"{f.relative_to(REPO)}:{i}")
        assert not offenders, (
            "커널을 emit 해서 컴파일하는데 CUTLASS 경로만 준다. "
            f"paths.kernel_includes() 를 써라: {offenders}")


def _cutlass_available() -> bool:
    try:
        paths.cutlass_dir(None)
    except paths.PathError:
        return False
    return True


def test_datasets_는_저장소_루트에서_찾는다():
    """`datasets/` 는 패키지 밖이다 (`hwspec/`, `artifacts/` 와 같다).

    패키지 이전 뒤 `resolve_bundle_path` 가 `PKG_ROOT/datasets` 를 보고 있어서
    릴리즈 번들을 못 찾았다 — `launch_probe.cu` 와 같은 종류의 잔재다.
    """
    from kerneltab.core.bundle import BundleError, resolve_bundle_path
    d = REPO / "datasets"
    if not d.exists():
        pytest.skip("datasets/ 가 없다 (배포 번들 미생성)")
    names = [x.name for x in d.iterdir() if (x / "BUNDLE.json").exists()]
    if not names:
        pytest.skip("번들이 없다")
    try:
        got = resolve_bundle_path(names[0])
    except BundleError as e:
        pytest.fail(f"저장소 루트의 번들을 못 찾는다: {e}")
    assert got == (d / names[0]).resolve()
