# 측정 완료 후 처리할 수정 목록

외부 코드 리뷰(R-1~R-4)와 이번 캠페인에서 반복 확인된 패턴(R-5)이다.

---

## ⛔ 실행 게이트

**Phase 3 전수 측정이 완료되고 `docs/post_measurement.md` 의 백업
1~6 단계를 모두 마친 뒤에만** 시작한다.

```
1. validate_table.py --expect full  통과
2. export.py -> table.parquet
3. bundle.py -> 번들 생성
4. 다른 물리 디스크로 복사
5. 오프사이트 1 부
6. 체크섬 대조
```

**측정 중이거나 백업 전이면 아무것도 하지 않는다.** 이 캠페인에서 나온
버그 대부분이 "안전한 줄 알았던 수정" 에서 나왔다.

---

## 순서

```
1. R-1  테스트 스킵 감지        <- 먼저. 이후 수정의 안전망
2. R-5  env_hash 격리 구조화     <- 세 번 밟은 패턴
3. migration_plan.md 의 11 건   <- P-3 -> P-1 -> P-2 -> 나머지
4. R-2, R-3  sweep.py
5. R-4  린트
6. packaging.md 의 kerneltab.* 이전
```

**R-1 이 먼저인 이유**: 이후 모든 수정의 회귀 테스트가 **실제로 도는지**
보장해야 한다. 지금 상태로 마이그레이션하면 "테스트 통과" 가 거짓일 수 있다.

**R-5 가 두 번째인 이유**: 세 번째로 같은 함정을 밟았고, 마이그레이션
(특히 P-3 의 `env_hash` 재정의) 과 직접 얽힌다.

각 단계마다: 테스트 전체 통과 확인 / `git commit` 을 논리 단위로 분리 /
측정 데이터에 영향을 주는 변경이면 **기존 번들로 회귀 검증**
(`export.py` 재실행 후 parquet 스키마·행 수·주요 통계 동일성).

### 하지 말 것

* 백업 1~6 단계 전에 시작하지 않는다
* `results/*.jsonl` 원본을 수정하지 않는다 (append-only)
* 한 커밋에 여러 R 항목을 섞지 않는다
* `ruff --fix` 결과를 테스트 없이 커밋하지 않는다
* `measure/` 의 blind except 를 일괄 치환하지 않는다. 하나씩 검토한다

---

## R-1. 테스트가 조용히 스킵된다 ★ 최우선

### 현상

```
$ python3 -m pytest tests/ -q
86 passed, 2 skipped in 0.16s      <- 초록불

SKIPPED tests/test_table.py:13   could not import 'pyarrow'
SKIPPED tests/test_bundle.py:10  could not import 'pyarrow'
```

`pytest.importorskip("pyarrow")` 가 모듈 최상단에 있어서, pyarrow 가 없으면
`test_table.py` 22 개 + `test_bundle.py` 19 개, **총 41 개가 통째로 안 돈다.**
요약에는 "2 skipped" 로만 나온다.

하필 그 두 모듈이 **정답 누출 방지를 검증하는 것들**이다. `test_table.py` 의
docstring 은 이렇게 시작한다 — "문서는 지켜지지 않으므로 코드로 강제한다.
그렇다면 그 코드를 지키는 것도 문서여서는 안 된다." 그런데 그 파일이 도는지가
**환경에 pyarrow 가 깔려 있느냐**에 달려 있었다. 문서로 강제하는 것과 다를 바
없어졌다.

`kernelrule` 이 pyarrow 없는 환경에서 돌리면 86 passed 를 보고 "누출 방지
검증됨" 이라고 믿게 된다.

### 수정

스킵 자체를 막을 필요는 없다. **스킵됐다는 사실을 실패로 만들면** 된다.

`tests/conftest.py` 에:

* `CRITICAL_MODULES = {"test_table.py", "test_bundle.py"}`
* 세션 종료 시 위 모듈이 수집되지 않았거나 전부 스킵됐으면 **실패**
* 실패 메시지에 `pip install -e '.[test]'` 안내
* `KERNELTAB_ALLOW_SKIP=1` 로 우회 허용 (우회 시 큰 경고 출력)

pytest 훅 선택은 자유(`pytest_collection_modifyitems`,
`pytest_sessionfinish`, `pytest_report_header`).

**동작을 메타 테스트로 고정한다** — pyarrow 를 일시적으로 가리고 pytest 를
서브프로세스로 돌려 실패하는지 확인.

### 일반화 — 이게 이 항목의 핵심

이 저장소가 **세 번째로 만나는 클래스**다.

| 사례 | 증상 |
|---|---|
| `WARMUP_SECONDS` | 정의만 되고 안 쓰임 — 로그는 "워밍업 한다" 고 찍힘 |
| `MEM_CLOCK_MIN_FRAC` | 주석에 기준만 있고 미구현 |
| `test_table.py` | 스킵되는데 초록불 |

이전 AST 검사는 "상수가 안 쓰임" 만 봤다. 이번엔 **"조건이 안 맞으면 조용히
통과하는 안전장치"** 를 전수 검사한다:

* `importorskip` / `skipif` 를 쓰는 다른 곳
* `if not X: return` 으로 빠져나가는 검증 함수
* 예외를 삼키고 진행하는 검사 (`except: pass`, `except: continue`)
* 조건이 거짓이면 검사를 건너뛰는 분기

발견한 것과 **패턴 자체**를 `docs/decisions.md` 에 기록한다.

---

## R-5. `env_hash` 격리를 구조로 강제 ★

### 현상 — 세 번 밟았다

| # | 어디 | 증상 |
|---|---|---|
| 1 | `export.py` 의 `difficulty` | 모든 조건을 섞어 계산해 난이도가 **22.05 배** |
| 2 | `bundle.py` 의 번들 통계 | 형상 68 개(실제 66), 측정 구간이 폐기 구간부터. **공개 릴리즈 노트에 실릴 뻔했다** |
| 3 | `rehearse.py` 의 드리프트 경고 | 클럭 미고정 리허설 + 폐기 구간이 섞여 변동폭 **55.59%**, 경고 상시 발생 |

전부 **"여러 `env_hash` 가 섞인 파일을 필터 없이 집계"** 였다. 교훈은
`docs/decisions.md` 13 번에 이미 적혀 있다 — **`env_hash` 는 조인 키가 아니라
격리 경계다.** 그런데 규율이 문서에만 있어서 또 밟았다. (R-1 과 같은 교훈의
다른 얼굴이다.)

### 수정

`core/table.py` 의 로더는 이미 이것을 강제한다. **같은 규율을 jsonl 읽기에도
적용한다.**

* `results/*.jsonl` 을 읽는 **모든 경로를 전수 조사**하고 목록화한다
* `env_hash` 필터가 없는 곳에서 의도적인 것과 버그를 구분한다
* 읽기 헬퍼를 하나로 통일하고 **`env_hash` 를 필수 인자로** 만든다
  (기본값 없음 — 명시하지 않으면 에러)
* 전체를 보고 싶은 경우는 명시적 옵션으로만 (`env_hash="ALL"`)
* `export.py` 의 `_per_env` 가드도 이 헬퍼 위로 옮긴다

**테스트로 고정한다**: 여러 조건이 섞인 합성 jsonl 에서 필터 없이 집계를
시도하면 실패하는지.

---

## R-2. `sweep.py` 가 사람용 stdout 을 파싱한다

```python
if line.startswith("세그먼트 ") and n_seg is None: ...
if line.startswith("작업 수:"): ...
```

`SEGJOBS` 는 기계용 접두어로 잘 만들었으나 나머지 둘은 **사람이 읽으라고 쓴
문장**을 파싱한다. `rehearse.py` 출력 문구를 다듬는 순간 33 시간 스윕의
진입점이 깨진다. (`SystemExit` 으로 즉시 죽으므로 조용히 틀리지는 않는다.)

**수정**: `rehearse.py --list-segments --json` 을 추가하고 `sweep.py` 가 그것만
쓰도록 한다. 사람용 출력은 그대로 둔다 (진행 확인에 쓴다).

```json
{"n_segments": 13, "n_jobs": 980915, "segment_kernels": 500,
 "jobs_per_segment": {"0": 65748, "1": 86614}, "env_hash": "..."}
```

`env_hash` 를 포함시켜, `sweep.py` 가 자기가 읽은 `env.json` 과 다르면
거부하도록 한다. 조건이 어긋난 채 스윕이 시작되는 것을 막는다.

---

## R-3. 재개 시 라운드 번호가 0 부터 다시 시작한다

```python
done: set[int] = set()
rnd = 0
```

중단 후 재시작하면:

* 이미 끝난 세그먼트도 매 라운드 프로세스를 띄웠다 `RC_DONE` 으로 즉시 종료
  -> 재개 파싱 6 초 x 13 개 = **라운드당 78 초 낭비**
* 셔플 시드가 `seed ^ (rnd+1)` 이라 **라운드 0 의 순서가 반복**된다
* `sweep.jsonl` 의 라운드 번호가 겹쳐 사후 분석이 헷갈린다

**데이터 정확성 문제는 아니다** — 측정된 작업은 건너뛴다.

**수정**: 시작 시 `sweep.jsonl` 에서 상태 복원. 마지막 `round` 번호로 이어서
시작하고, `rc == RC_DONE` 세그먼트를 `done` 에 넣는다. **`env_hash` 가 다른
항목은 무시한다** (R-5 와 같은 원칙). 합성 로그로 테스트.

---

## R-4. 린트 정리

```
107 RUF100 불필요한 noqa / 31 BLE001 blind except / 24 F401 미사용 import
 18 I001 import 정렬 / ...   총 291 건, 196 건 자동 수정 가능
```

**단계별로 한다. 한 번에 다 고치지 않는다.**

**(a) 자동 수정 먼저**

```bash
ruff check . --fix
python3 -m pytest tests/ -q                    # 반드시 통과
python3 -c "import measure.runner, scripts.rehearse"
```

단독 커밋. 자동 수정은 되돌리기 쉬워야 한다.

**(b) `BLE001` 31 개 — 자동 수정 금지**

`measure/gpu_state.py` 에 8 개가 몰려 있다. NVML 조회는 실패해도 진행해야 하니
의도적일 것이다. 다만 `except Exception` 은 **프로그래밍 오류(오타, 타입
에러)까지 삼킨다.** NVML 이 조용히 안 도는 상태를 못 알아챌 수 있다.

각각 검토해서:

* 의도적으로 삼키는 것 -> 예외 타입을 좁히고(`except (OSError, RuntimeError)`)
  왜 삼키는지 주석
* 방어적으로 넣은 것 -> 최소한 `logging` 으로 남기기
* 실수인 것 -> 제거

**`measure/` 를 특히 주의 깊게 본다.** 여기서 예외가 삼켜지면 클럭 조회 실패나
NVML 오작동을 놓친다. (R-1 의 "조용히 아무것도 안 하는 안전장치" 와 같은 축이다.)

R-1 의 AST 전수 검사가 찾아둔 것 (`docs/decisions.md` 14번):

* **예외를 삼키고 진행: 17건** — `build/compile.py:337`, `measure/gpu_state.py:149`,
  `scripts/rehearse.py:290/406/1155`, `scripts/report_phase3.py:259/991/1016`,
  `scripts/phase0_env.py:69/224`, `scripts/check_anchors.py:83/277`,
  `scripts/manifest.py:107`, `scripts/build_kernels.py:68` 외
* **검증 함수가 조건 불충족 시 조용히 return: 1건**
  — `scripts/rehearse.py:1203` `drift_check()`. 프로브 커널 준비가 실패하면
  `return` 이고 호출부가 `if t_drift:` 라 **드리프트 감시가 남은 실행 내내
  조용히 멈춘다.** 33시간 측정에서는 발생하지 않았지만(242회 정상 기록)
  잠재된 같은 병이다. 경고를 찍고 heartbeat 에 플래그를 남기도록 고친다.

**(c)** `scripts/report_phase3.py` 의 `g if g == g else 1e9` -> `math.isnan(g)`.
NaN 관용구인데 후자가 읽기 쉽고 린터 오탐도 사라진다.

**(d)** `EXE001`, `PLW1510` 등 남는 항목은 판단해서 처리하거나
`pyproject.toml` 의 ruff 설정에 무시 규칙을 넣는다. **무시할 거면 이유를 적는다.**

---

## R-6. 리포트의 앵커 판정이 옛 기준을 쓴다

### 현상

`report_phase3.py` 의 `## 2-b. 앵커` 절이 **절대 1 % 기준**으로 판정한다.
실제 판정기 `check_anchors.py` 는 이미 **노이즈 대비**(sB/sW <= 1.5)로
바뀌었는데 리포트 쪽이 따라가지 않았다.

전수 리포트에서 실제로 엇갈렸다:

| | 판정 |
|---|---|
| `report_phase3.py` 2-b | "**1 % 를 넘는다.** 이 표의 config 순위를 그대로 믿으면 안 된다" |
| `check_anchors.py` | **통과** (짧은 앵커 6개 비율 0.00~0.38, 기준 1.5) |

512³ 앵커 하나가 변동폭 1.81 % 인데 그 커널의 측정 노이즈가 1.29 % 라
비율 0.38 이다. 절대 기준으로는 실패, 노이즈 대비로는 통과다.
**노이즈 대비가 옳다** — 12~56 us 커널에서 1 % 는 달성 불가능한 기준이다.

**데이터에는 영향이 없다.** 리포트 문구만 틀렸다.

### 수정

`_anchor_report()` 를 `check_anchors.py` 와 같은 판정으로 통일한다.
판정 로직이 두 곳에 복제되어 있는 것이 근본 원인이므로 **하나로 합쳐**
양쪽이 같은 함수를 부르게 한다.

### 같이 고칠 것 — 라운드 매핑이 깨졌다

`_round_of_segment()` 가 `(segment, when) -> round` 로 매핑하는데, 한
세그먼트가 여러 라운드에 등장하므로 **뒤 라운드가 앞 라운드를 덮어쓴다.**
결과로 리포트의 "라운드별 추이" 에 마지막 라운드 하나만 나오고,
"절대값 추이(첫 라운드 -> 마지막 라운드)" 는 "라운드가 2개 미만" 으로
아예 계산되지 않는다.

`sweep.jsonl` 에 슬라이스 시각이 있으므로 **앵커의 timestamp 로** 라운드를
찾아야 한다 (세그먼트 번호만으로는 유일하지 않다). R-3 의 재개 상태 복원과
같은 데이터를 쓰므로 함께 고치는 편이 낫다.

---

## 완료 후 보고할 것

* R-1 메타 테스트가 실제로 실패를 잡는지 확인한 결과
* "조용히 아무것도 안 하는 안전장치" 전수 검사에서 추가로 나온 것
* R-5 의 jsonl 읽기 경로 목록과 각각의 판정 (의도 / 버그)
* 린트 잔여 건수와 무시하기로 한 규칙 목록
* 기존 번들 회귀 검증 결과 (parquet 재생성 시 동일성)
