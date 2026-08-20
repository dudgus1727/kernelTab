# 패키지화 — 외부 프로젝트(`kernelrule`)에서 import 하기

목표: `pip install -e ../kernelTab` 으로 설치하면
`core.features` / `core.types` / `backends` 를 쓸 수 있어야 한다.

> ## ✅ 적용 완료 (2026-08-20)
>
> 아래는 계획서다. 실제 적용에서 **세 가지가 계획과 달라졌고**, 그 편이
> 맞다고 판단했다.
>
> | | 계획 | 실제 | 왜 |
> |---|---|---|---|
> | `hwspec/` | `kerneltab/hwspec/` 로 이동, 패키지 데이터 | **최상위에 그대로 둠** | 패키지가 아니라 데이터다. `KERNELTAB_HWSPEC_DIR` 로 위치를 바꿀 수 있게 했고, 못 찾으면 경로와 해결법을 찍고 죽는다. 비-editable 설치에서는 이 환경변수가 필요하다 — 그 대가를 알고 택했다. |
> | `kerneltab/cli.py` + 스크립트 shim | 통합 CLI 로 재작성 | **`scripts/` 를 진입점으로 유지** | resume 명령이 바뀌면 안 되고, CLI 통합은 이전과 독립적인 작업이다. 한 커밋에 섞지 않는다. |
> | `pandas` | 의존성에서 삭제 | **필수로 유지** | "코드가 쓰지 않는다" 가 사실이 아니었다. `core/table.py` 와 `core/bundle.py` 가 쓰고, 둘 다 정답 누출을 막는 소비 경로다. |
>
> 계획대로 한 것: `build/artifacts` -> **`artifacts/`** (패키지 밖으로).
> 디렉토리 이동·`paths.ARTIFACT_DIR` 기본값·`tests/test_portability.py` 의
> 단언을 **같은 커밋에서** 함께 바꿨다. 셋 중 하나만 바꾸면 조용히 깨진다.
>
> ### 회귀 검증 결과 (전부 이전과 동일)
>
> | 무엇 | 결과 |
> |---|---|
> | **`env_hash_v2`** | `1b6a0c0c...` **불변** — 코드 위치는 해시 입력이 아니다 |
> | `table.parquet` | 1,216,431 행, 스키마 동일, **모든 수치 컬럼 비트 동일** |
> | `baseline_rule` regret@1 | 1.1921 (동일) |
> | `baseline_gbdt` | 블록 1.0107 / 5-fold 1.0187 (문서값 1.011 / 1.019) |
> | `validate_table --bundle` | 통과 |
> | `check_anchors` | exit 0 |
> | `sweep --dry-run` | 980,915 작업 인식 |
> | 테스트 | 252 통과, ruff 0 |
>
> ### 이전이 **조용히 무력화한 검사**가 하나 있었다
>
> `tests/test_features.py` 의 레이어링 검사가 `module.split(".")[0]` 로
> 최상위 이름을 봤다. import 가 `from backends...` 에서
> `from kerneltab.backends...` 로 바뀌자 그 값이 `"kerneltab"` 이 되어
> **`"backends" not in imported` 가 항상 참**이 된다. 접두사를 벗겨낸 뒤
> 비교하도록 고치고, "import 를 하나라도 읽었는가" 단언을 함께 넣었다.
> 네임스페이스 이전은 최상위 이름에 의존하는 모든 검사를 이렇게 만든다.

## (아래는 원래 계획서) 결론: 지금 적용하지 않는다. 측정 완료 후에 한다.
>
> 이유는 "확신이 없어서" 가 아니라 **구체적인 문제 두 가지** 때문이다
> (아래 1 절). 그 문제들의 해법이 **디렉토리 이동**을 요구하고, 디렉토리를
> 옮기면 측정이 중단됐을 때 resume 이 깨진다.
>
> 이 문서에 최종 형태를 적어 두고, 측정이 끝나면 그대로 적용한다.

---

## 1. 지금 구조로는 왜 안 되는가

### 문제 A — 최상위 이름이 너무 일반적이다

현재는 flat layout 이고 최상위 패키지가 이렇다.

```
core/  backends/  build/  measure/  hwspec/  scripts/
```

`pip install -e .` 하면 소비 프로젝트의 `sys.path` 에 **`core`, `build`,
`measure` 라는 최상위 모듈이 생긴다.** 이건 사고를 부른다.

* `build` 는 PyPI 에 [실제로 존재하는 패키지](https://pypi.org/project/build/)
  이고 `python -m build` 로 널리 쓰인다. 소비 프로젝트가 그것을 설치하면
  **어느 쪽이 이기는지 설치 순서에 달린다.**
* `core` / `measure` 도 흔한 이름이라 소비 프로젝트 자체의 모듈과 충돌하기
  쉽다.
* 충돌하면 `ImportError` 가 아니라 **다른 모듈이 조용히 import 된다.** 가장
  나쁜 실패 방식이다.

→ `kerneltab.core`, `kerneltab.backends` 처럼 **네임스페이스 아래로** 넣어야
한다. 그러려면 디렉토리를 옮겨야 한다.

### 문제 B — 디렉토리를 옮기면 resume 이 깨진다

측정 중인 프로세스는 이미 모듈을 로드했으므로 파일을 옮겨도 **당장은**
멈추지 않는다. 문제는 **재개(resume)** 다.

```python
# 모든 스크립트 상단
sys.path.insert(0, str(REPO_ROOT))
from core.config import ...
```

디렉토리를 옮긴 뒤 측정이 중단되어 같은 명령으로 재개하면 `ImportError` 가
난다. 40 시간짜리 작업 도중에 이런 위험을 만들 이유가 없다.

또 `build/artifacts/` 가 산출물 경로다. `build/` 를 `src/kerneltab/build/`
로 옮기면 **7.4 GB 의 커널 `.so` 경로가 전부 바뀐다** (P-1 과 얽힌다).
이건 P-1(`so_path` 절대경로 제거) 을 먼저 끝낸 뒤에 해야 안전하다.

---

## 2. 최종 형태 (측정 완료 후 적용)

### 2-1. 디렉토리 이동

```
kerneltab/                 # 새 최상위 패키지
    __init__.py
    core/                  <- core/
    backends/              <- backends/
    build/                 <- build/          (코드만. artifacts 는 아래 참조)
    measure/               <- measure/
    hwspec/known.json      <- hwspec/         (패키지 데이터로 포함)
    cli.py                 <- scripts/ 통합 진입점 (docs/entrypoints.md)
scripts/                   # 기존 스크립트는 얇은 shim 으로 남긴다
docker/  docs/
results/                   # 산출물. 패키지 밖
artifacts/                 # 커널 .so. build/artifacts 에서 **밖으로 뺀다**
```

`build/artifacts` 를 최상위 `artifacts/` 로 빼는 이유: 산출물이 패키지
디렉토리 안에 있으면 `pip install` 이 그것까지 가져가려 하고, 컨테이너
볼륨 마운트 지점으로도 부적절하다.

`hwspec/` 은 지금 "데이터 디렉토리이며 Python 패키지가 아니다" 라는 규약이
있다 (`__init__.py` 없음). 패키지 데이터로 넣되 **`__init__.py` 는 계속
두지 않는다** — `importlib.resources` 로 읽는다.

### 2-2. `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "kerneltab"
version = "0.2.0"
description = "CUTLASS GEMM (형상 x config) -> 성능 표 측정 하네스"
requires-python = ">=3.10"          # vermin --eval-annotations 로 확인함

dependencies = [
    "nvidia-ml-py>=12,<14",         # NVML. 측정 루프의 클럭/온도 스냅샷
    "pyarrow>=15",                  # export. optional 이 아니라 필수다 —
                                    # extras 없이 설치하면 export 가 죽는다
]

[project.optional-dependencies]
plots = ["matplotlib>=3.7"]         # report_phase3.py 의 PNG 만
dev    = ["ruff", "vermin", "uv"]

[project.scripts]
kerneltab = "kerneltab.cli:main"

[tool.setuptools.packages.find]
include = ["kerneltab*"]

[tool.setuptools.package-data]
"kerneltab.hwspec" = ["*.json"]
"kerneltab.measure" = ["*.cu", "*.h"]
"kerneltab.build"   = ["*.cu"]
```

현재 `pyproject.toml` 대비 바뀌는 것:

| 항목 | 현재 | 변경 후 | 이유 |
|---|---|---|---|
| 패키지 | `["core","backends","build","measure"]` | `kerneltab*` | 이름 충돌 (문제 A) |
| `pyarrow` | optional (`export` extra) | **필수** | `export.py` 가 없으면 죽는다 |
| `pandas` | optional 선언 | **삭제** | 코드 어디서도 쓰지 않는다 |
| `matplotlib` | 미선언 | optional (`plots`) | `report_phase3.py` 의 그림 전용 |
| 버전 | `>=` 만 | 상한 추가 + `requirements.lock` | 재현성 |
| 콘솔 스크립트 | 없음 | `kerneltab` | 컨테이너 ENTRYPOINT |

### 2-3. `measure/` 안의 C++ 소스

`kt_abi.h` / `kt_kernel_impl.h` / `kt_swizzle.h` / `kt_ctx.cu` 는 런타임에
nvcc 로 컴파일된다. 패키지 데이터로 포함하고, include 경로를
`importlib.resources.files("kerneltab.measure")` 로 얻어야 한다.

지금은 `paths.REPO_ROOT / "measure"` 로 하드코딩되어 있다
(`build/compile.py: BuildEnv.from_env_json`). 이 부분도 함께 고쳐야 한다.

### 2-4. 기존 스크립트 shim

`scripts/*.py` 를 지우지 않고 얇게 남긴다. README 와 문서에 적힌 명령이
계속 동작해야 하고, 무엇보다 **resume 명령이 바뀌면 안 된다.**

```python
#!/usr/bin/env python3
"""(shim) kerneltab.cli 로 위임한다."""
import sys
from kerneltab.cli import main
sys.exit(main(["measure", *sys.argv[1:]]))
```

---

## 3. 적용 순서와 검증

**전제: `docs/migration_plan.md` 의 P-1 (so_path 절대경로 제거) 이 끝나 있을 것.**
아니면 디렉토리 이동이 7,330 개 `.so` 경로를 전부 깬다.

```bash
# 1) 이동 (git mv 로 이력 보존)
mkdir kerneltab && git mv core backends build measure hwspec kerneltab/
git mv kerneltab/build/artifacts artifacts        # 산출물은 패키지 밖으로

# 2) import 경로 일괄 치환 후 pyproject.toml 교체
#    from core.x import  ->  from kerneltab.core.x import   (약 30 곳)

# 3) 설치 및 검증
pip install -e .
python3 -c "from kerneltab.core import features, types; from kerneltab import backends; print('import ok')"
kerneltab --help

# 4) **회귀 검증** — 기존 데이터로 결과가 바뀌지 않는지
python3 scripts/validate_table.py            # 종료 코드 0
python3 scripts/export.py --out /tmp/new.parquet
python3 - <<'EOF'
import pyarrow.parquet as pq
a = pq.read_table('results/table.parquet'); b = pq.read_table('/tmp/new.parquet')
assert a.schema == b.schema, "스키마가 바뀌었다"
assert a.num_rows == b.num_rows
print("parquet 동일")
EOF

# 5) 소비 프로젝트에서
cd ../kernelrule && pip install -e ../kernelTab
python3 -c "from kerneltab.core.features import waves, can_use_cp_async; print('ok')"
```

4 번이 핵심이다. **패키지화는 동작을 바꾸면 안 된다.** parquet 이 바이트
단위로 같을 필요는 없지만 스키마와 행 수는 같아야 한다.

## 4. 롤백

`git revert` 로 되돌린다. 단 `git mv` 를 되돌리면 `artifacts/` 위치가
다시 바뀌므로, 롤백 후 `build/artifacts` 심볼릭 링크를 만들거나 커널을
다시 빌드해야 한다. **이동 전에 커밋 지점을 명확히 남길 것.**

## 5. 지금 당장 해도 안전한 것 (하지만 하지 않았다)

`pyproject.toml` 의 **의존성 선언만** 고치는 것(`pyarrow` 필수화, `pandas`
삭제)은 실행 중인 코드에 아무 영향이 없다. `pyproject.toml` 은 런타임에
읽히지 않기 때문이다.

그래도 하지 않은 이유: 의존성만 고치고 패키지 구조를 그대로 두면
`pip install -e .` 가 **최상위에 `core`/`build`/`measure` 를 노출**해
문제 A 가 그대로 남는다. 절반만 적용된 상태가 오히려 위험하다 —
"설치되니까 됐다" 고 생각하고 소비 프로젝트에서 쓰기 시작할 수 있다.
한 번에 적용한다.
