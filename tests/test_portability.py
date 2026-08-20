"""마이그레이션 나머지 항목 — 컨테이너 이식성의 전제조건들.

각 항목이 **깨지면 컨테이너에서 조용히 틀린다**는 공통점이 있다.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import pytest

try:                                    # Python 3.11+
    import tomllib
except ModuleNotFoundError:             # Python 3.10 — 컨테이너가 여기다
    # ⚠️ `tomllib` 은 3.11 stdlib 이다. 이 파일이 그걸 무조건 import 하는
    #    바람에 **3.10 에서 모듈이 통째로 수집 실패**했다 — 이식성을 검사하는
    #    테스트가 이식성이 없었다. 컨테이너에서 실제로 돌려 보고서야 나왔다
    #    (vermin 은 잡았지만 tests/ 를 대상에 넣고 돌린 적이 없었다).
    import tomli as tomllib

REPO = Path(__file__).resolve().parent.parent


class TestPathOverride:
    """수정 6 — 컨테이너는 results/ 와 artifacts/ 를 볼륨으로 마운트한다."""

    def _paths_in(self, env: dict):
        code = (f"import sys; sys.path.insert(0, {str(REPO)!r})\n"
                "from kerneltab.build import paths\n"
                "import json; print(json.dumps({'r': str(paths.RESULTS_DIR),"
                "'a': str(paths.ARTIFACT_DIR), 'e': str(paths.ENV_JSON)}))")
        e = dict(os.environ)
        e.pop("KERNELTAB_RESULTS_DIR", None)
        e.pop("KERNELTAB_ARTIFACT_DIR", None)
        e.update(env)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=e, timeout=120)
        assert r.returncode == 0, r.stderr
        return json.loads(r.stdout)

    def test_default_is_repo_relative(self):
        """★ 기본값이 바뀌면 기존 산출물(98만줄, 7.4GB)을 못 찾는다.

        패키지 이전에서 `build/artifacts` -> `artifacts` 로 **한 번** 옮겼다.
        그때 디렉토리 이동·기본값·이 단언을 같은 커밋에서 함께 바꿨다.
        앞으로 이 단언만 따로 바뀌면 그것이 사고다.
        """
        p = self._paths_in({})
        assert p["r"] == str(REPO / "results")
        assert p["a"] == str(REPO / "artifacts")

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

    def _info(self, root, override, env_commit=None):
        sys.path.insert(0, str(REPO / "scripts"))
        from phase0_env import cutlass_info
        old = os.environ.get("CUTLASS_COMMIT")
        # ⚠️ 환경변수를 **명시적으로** 지운다. 이미지에는 CUTLASS_COMMIT 이
        #    설정돼 있어서(Dockerfile), 지우지 않으면 "아무것도 모르는" 경우를
        #    검사할 수 없다. 컨테이너에서 이 테스트가 실패해 알았다.
        if env_commit is None:
            os.environ.pop("CUTLASS_COMMIT", None)
        else:
            os.environ["CUTLASS_COMMIT"] = env_commit
        try:
            return cutlass_info(str(root), override)
        finally:
            os.environ.pop("CUTLASS_COMMIT", None)
            if old is not None:
                os.environ["CUTLASS_COMMIT"] = old

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
        """★ 주입도 없고 .git 도 환경변수도 없으면 'unknown' 으로 **드러나야** 한다."""
        i = self._info(fake_cutlass, None)
        assert i["commit"] is None
        assert i["commit_source"] == "unknown"

    def test_env_var_fallback(self, fake_cutlass):
        """컨테이너에서 git 이 막히면(dubious ownership) 여기서 건진다.

        출처를 `env` 로 남긴다 — git 에서 읽은 것처럼 위장하면 나중에
        신뢰도를 판단할 수 없다.
        """
        i = self._info(fake_cutlass, None, env_commit="d" * 40)
        assert i["commit"] == "d" * 40
        assert i["commit_source"] == "env"

    def test_주입이_환경변수보다_우선한다(self, fake_cutlass):
        i = self._info(fake_cutlass, "abc123def", env_commit="d" * 40)
        assert i["commit"] == "abc123def"
        assert i["commit_source"] == "injected"

    def test_real_repo_reports_git(self):
        from kerneltab.build import paths
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

    #: 합성 env. **실제 `results/env.json` 을 읽지 않는다** — 컨테이너에는
    #: 그 파일이 없고(results/ 는 볼륨이다), 검사하려는 성질은 실제 데이터가
    #: 필요하지 않다. 파일에 의존하면 컨테이너에서 이 검사가 통째로 죽는다.
    ENV: ClassVar[dict] = {
        "hardware": {"name": "X", "arch": "sm_86", "sm_count": 84},
        "nvcc_arch_flag": "sm_86",
        "protocol": {"min_warmup": 10},
        "soak": {"enabled": True},
        "segments": {"kernels": 500},
        "clock_locked": True, "locked_mhz": 1350,
        "mem_clock_locked": True, "locked_mem_mhz": 7601,
        "peak_tflops_f16_effective": 120.0,
        "bandwidth_gbps_effective": 700.0,
        "shuffle_seed": 1234,
        "cutlass": {"commit": "c" * 40},
        "cuda": {"nvcc_version": "13.3.73"},
    }

    def test_manifest_does_not_change_hash(self):
        from kerneltab.core.env_hash import env_hash_v2
        env = dict(self.ENV)
        before = env_hash_v2(env)
        env["manifest"] = {"kerneltab_tree_hash": "x" * 40,
                           "manifest_hash": "y" * 40}
        assert env_hash_v2(env) == before, (
            "manifest 가 해시를 바꾸면 코드 한 글자 수정에도 재측정해야 한다")

    def test_실제_env_json_에도_성립한다(self):
        """있으면 실제 데이터로도 확인한다 (합성 env 가 현실과 다를 수 있다)."""
        from kerneltab.build import paths
        from kerneltab.core.env_hash import env_hash_v2
        f = paths.RESULTS_DIR / "env.json"
        if not f.exists():
            pytest.skip(f"{f} 가 없다 (컨테이너 정상)")
        env = json.loads(f.read_text())
        env.pop("manifest", None)
        before = env_hash_v2(env)
        env["manifest"] = {"kerneltab_tree_hash": "x" * 40}
        assert env_hash_v2(env) == before

    def test_합성_env_가_실제_키를_전부_덮는다(self):
        """★ 합성 env 가 낡으면 검사가 조용히 약해진다."""
        from kerneltab.core.env_hash import ENV_HASH_KEYS_V2
        missing = []
        for k in ENV_HASH_KEYS_V2:
            cur, ok = self.ENV, True
            for part in k.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    ok = False
                    break
            if not ok:
                missing.append(k)
        assert not missing, (
            f"합성 env 에 없는 해시 키: {missing}. ENV_HASH_KEYS_V2 가 "
            "늘어났으면 위 ENV 도 함께 채워라.")
