"""마이그레이션 나머지 항목 — 컨테이너 이식성의 전제조건들.

각 항목이 **깨지면 컨테이너에서 조용히 틀린다**는 공통점이 있다.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

REPO = Path(__file__).resolve().parent.parent


class TestPathOverride:
    """수정 6 — 컨테이너는 results/ 와 build/artifacts/ 를 볼륨으로 마운트한다."""

    def _paths_in(self, env: dict):
        code = ("import sys; sys.path.insert(0, %r)\n"
                "from build import paths\n"
                "import json; print(json.dumps({'r': str(paths.RESULTS_DIR),"
                "'a': str(paths.ARTIFACT_DIR), 'e': str(paths.ENV_JSON)}))"
                % str(REPO))
        e = dict(os.environ)
        e.pop("KERNELTAB_RESULTS_DIR", None)
        e.pop("KERNELTAB_ARTIFACT_DIR", None)
        e.update(env)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=e, timeout=120)
        assert r.returncode == 0, r.stderr
        return json.loads(r.stdout)

    def test_default_is_repo_relative(self):
        """★ 기본값이 바뀌면 기존 산출물(98만줄, 7.4GB)을 못 찾는다."""
        p = self._paths_in({})
        assert p["r"] == str(REPO / "results")
        assert p["a"] == str(REPO / "build" / "artifacts")

    def test_env_vars_override(self, tmp_path):
        p = self._paths_in({"KERNELTAB_RESULTS_DIR": str(tmp_path / "R"),
                            "KERNELTAB_ARTIFACT_DIR": str(tmp_path / "A")})
        assert p["r"] == str(tmp_path / "R")
        assert p["a"] == str(tmp_path / "A")
        assert p["e"] == str(tmp_path / "R" / "env.json")


class TestDependencies:
    """수정 7 — pyarrow/pandas 는 선택이 아니라 필수다."""

    @pytest.fixture
    def proj(self):
        return tomllib.loads((REPO / "pyproject.toml").read_text())

    @pytest.mark.parametrize("pkg", ["pyarrow", "pandas"])
    def test_is_required_not_optional(self, proj, pkg):
        """없으면 누출 방지 테스트 41개가 통째로 스킵된다 (R-1)."""
        deps = " ".join(proj["project"]["dependencies"])
        assert pkg in deps, f"{pkg} 가 필수 의존성이 아니다"

    def test_lock_includes_pandas(self):
        """계획서의 'pandas 는 미사용' 은 사실이 아니었다."""
        lock = (REPO / "docker" / "requirements.lock").read_text()
        assert "\npandas==" in lock, "lock 에 pandas 가 없다"
        assert "\npyarrow==" in lock

    def test_lock_has_hashes(self):
        """--require-hashes 로 설치하려면 전부 해시가 있어야 한다."""
        lock = (REPO / "docker" / "requirements.lock").read_text()
        pkgs = [l for l in lock.splitlines() if "==" in l and not l.startswith("#")]
        assert pkgs
        assert "--hash=sha256:" in lock


class TestCutlassCommitInjection:
    """수정 11 — .git 이 없어도 커밋을 알아야 한다.

    `cutlass.commit` 은 `env_hash_v2` 의 키다. None 이면 서로 다른 CUTLASS
    버전이 **같은 해시**를 받는다.
    """

    def _info(self, root, override):
        sys.path.insert(0, str(REPO / "scripts"))
        from phase0_env import cutlass_info
        return cutlass_info(str(root), override)

    @pytest.fixture
    def fake_cutlass(self, tmp_path):
        d = tmp_path / "cutlass"
        (d / "include" / "cutlass").mkdir(parents=True)
        (d / "include" / "cutlass" / "cutlass.h").write_text("")
        return d

    def test_injected_commit_is_used(self, fake_cutlass):
        i = self._info(fake_cutlass, "abc123def")
        assert i["commit"] == "abc123def"
        assert i["commit_source"] == "injected"

    def test_unknown_is_marked_not_faked(self, fake_cutlass):
        """★ 주입도 없고 .git 도 없으면 'unknown' 으로 **드러나야** 한다."""
        i = self._info(fake_cutlass, None)
        assert i["commit"] is None
        assert i["commit_source"] == "unknown"

    def test_real_repo_reports_git(self):
        from build import paths
        try:
            root = paths.cutlass_dir(None)
        except Exception:
            pytest.skip("CUTLASS 저장소가 없다")
        if not (root / ".git").exists():
            pytest.skip(".git 이 없는 체크아웃")
        i = self._info(root, None)
        assert i["commit_source"] == "git"
        assert i["commit"]


class TestManifestInEnv:
    """수정 8 — manifest 를 env.json 에 기록하되 해시는 바꾸지 않는다."""

    def test_manifest_does_not_change_hash(self):
        from core.env_hash import env_hash_v2
        env = json.loads((REPO / "results" / "env.json").read_text())
        env.pop("manifest", None)
        before = env_hash_v2(env)
        env["manifest"] = {"kerneltab_tree_hash": "x" * 40,
                           "manifest_hash": "y" * 40}
        assert env_hash_v2(env) == before, (
            "manifest 가 해시를 바꾸면 코드 한 글자 수정에도 재측정해야 한다")
