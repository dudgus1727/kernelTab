# 측정 완료 후 처리할 수정 목록

> ## ✅ R-1 ~ R-6 전부 완료 (2026-08-19)
>
> 이후 작업은 `docs/migration_plan.md`(P-1~P-3 + 8건, 완료),
> `docs/packaging.md`(`kerneltab.*` 이전, 완료),
> `docs/container.md`(이미지, 완료), `docs/next_campaign.md`(본 캠페인)
> 으로 넘어갔다.

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

> ### ✅ R-2 / R-3 완료 (2026-08-19)
>
> `rehearse.py --list-segments --json` -> `JSON {...}` 한 줄. sweep 은 이것만
> 쓴다. **`env_hash` 와 `env_hash_v2` 를 둘 다 비교하고 어긋나면 거부**한다 —
> v2 는 조건 자체, 구 해시는 재개 키라 어느 쪽이 달라도 데이터가 갈라진다
> (P-3 로 정의가 바뀌었으므로 v2 기준 비교가 핵심이다).
>
> R-3 은 `sweep.jsonl` 에서 (완료 세그먼트, 다음 라운드) 를 복원한다.
> **다른 `env_hash` 의 항목은 무시**한다 (R-5 와 같은 원칙). 실제 로그로
> 확인: 완료 13/13, 다음 라운드 7. 쓰다 만 줄도 견딘다.
>
> `tests/test_sweep.py` 9개. 사람용 출력을 파싱하는 코드가 남아 있으면
> 실패한다.

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

> ### ✅ R-4 완료 (2026-08-19)
>
> `ruff check .` **0 건**. 다만 "전부 고쳤다" 가 아니라 **판정을 끝냈다** 는
> 뜻이다 — 무시하는 규칙은 `pyproject.toml` 에 이유와 함께 적혀 있다.
>
> * (a) 자동 수정 236건을 **단독 커밋**으로 (`18738c8`).
> * (b) `measure/gpu_state.py` 8건을 **먼저** 처리했다. 여기서 삼키면
>   `sm_clock_mhz`/`mem_clock_mhz` 가 조용히 결측이 되고 33시간 뒤에
>   "조건이 유지됐는가" 를 확인할 근거가 사라진다. 실패를 세어 보고하고,
>   UUID 조회 실패 시 **인덱스로 폴백하던 것**도 제거했다(P-2 와 같은 문제).
> * **조용한 `pass`/`continue`(S110/S112) 9건은 전부 없앴다.** 나머지
>   `BLE001` 은 예외를 잡아 **기록하는** 패턴이라(`row["sass_error"]=repr(e)`)
>   무시하되 이유를 적었다. 아무것도 안 남기는 쪽이 진짜 문제였다.
> * `B015` 가 **검증하지 않는 테스트**를 잡았다 —
>   `assert_no_answers(...) is None` 이 `assert` 없이 식만 있었다.
>
> ⚠️ **함정 하나를 기록해 둔다.** `RUF100`(불필요한 noqa)은 "현재 켜진 규칙"
> 기준으로 판정한다. `E402` 가 꺼진 상태에서 `--fix` 를 돌렸더니
> `# noqa: E402` 127개를 전부 지웠다. 나중에 `E402` 를 켜면 98건이 한꺼번에
> 튀어나온다. **규칙을 새로 켜기 전에 그 규칙의 noqa 가 남아 있는지 확인할 것.**

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

> ### ✅ R-6 완료 (2026-08-19)
>
> 판정을 `core/anchors.py` **하나로** 모았다. `check_anchors.py` 와
> `report_phase3.py` 는 이제 표시만 한다 — 둘이 다른 답을 낼 수 있는
> 경로가 없어졌다.
>
> **라운드 매핑 버그가 생각보다 컸다.** `(segment, when) -> round` 는
> 라운드마다 덮어써서 마지막 하나만 남는다. 그래서 8 라운드를 돌고도
> "라운드가 2개 미만이라 비교할 수 없다" 가 찍혔고, **"모든 세그먼트가
> 함께 드리프트하는가" 검사가 33시간 내내 죽어 있었다.** 그 검사가 바로
> 세그먼트 대책이 못 잡는 종류의 오염을 잡는 것이다. 오류는 하나도 안 났다.
>
> 시각으로 복원해 다시 돌린 결과: **R0 → R7 최대 이동 0.63%** (바닥 1.00%).
> 대책이 33시간 전 구간에서 듣는다는 것이 이제 **실제로 검증됐다.**
>
> 같은 뿌리의 문제를 세 개 더 찾았다.
>
> | | 무엇 | 왜 문제인가 |
> |---|---|---|
> | 1 | `sW` 가 라운드를 가로질러 섞였다 | 판정의 **분모**다 |
> | 2 | 절대값 이동의 바닥이 집계 수준과 안 맞았다 | 노이즈를 드리프트로 오판 |
> | 3 | 단조 판정을 **중앙값**으로 했다 | 타이머 눈금(1.024us)에 붙어 8 라운드 전부 `+0.000%` — 추세를 볼 수 없다 |
>
> 전부 **집계 수준을 맞추는** 문제였다. 분자와 분모를 다른 단위로 재면
> 검사는 조용히 느슨해지거나 조용히 과민해진다.
>
> 덤으로 `core/noise.py` 모델과의 **교차 검증**을 표에 넣었다. 모델은
> 다른 데이터에서 유도한 것인데 12개 앵커 중 11개가 0.28~0.95배로 맞는다.
> 어긋난 하나(`a882@4096`, 4.7배)는 절대값 이동도 가장 컸다 — 주의로 찍되
> 실패로는 만들지 않는다 (긴 앵커는 드리프트 판정 대상이 아니다).
>
> **앞으로는 복원할 필요가 없다.** `sweep.py` 가 `--round` 를 넘기고
> `rehearse.py` 가 앵커 줄에 `round` 를 적는다. `slice` 줄에도 `env_hash`
> 를 적는다.

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

---

# C 계열 — 소비 계약의 구멍 (kernelRule 구현 중 발견)

> `kernelRule` 이 `docs/consumer_contract.md` 대로 표를 소비하며 발견한 것들.
> R 계열과 달리 **측정 코드가 아니라 소비 API** 의 문제다.
> 보고자: kernelRule 1단계 (2026-08-20)

## C-1. `answer_set()` 이 번들 계수를 무시한다 ★

### 현상

노이즈 진입점이 **둘**인데 성격이 다르고, 정답 집합을 만드는 쪽이
**틀린 쪽**을 부른다.

```python
kerneltab.core.noise.noise_floor(t)   # 모듈 전역 상수. A6000 값이 박혀 있다
Bundle.noise_floor                    # 번들 계수로 만든 함수 + tick 부재 시 경고
```

`core/table.py` 의 `answer_set()`:

```python
def answer_set(df, tol: float | None = None):
    from kerneltab.core.noise import noise_floor      # <- 모듈 전역
    ...
    tol = 2.0 * noise_floor(float(best))
```

`Bundle.scoring()` 으로 표를 꺼낸 뒤 `answer_set(df)` 를 부르는 것이
문서가 안내하는 경로인데, 그 순간 **번들의 `noise_floor` 계수는 버려진다.**

### 왜 조용한가

`Bundle.tick_ms` 는 `tick_ms` 부재 시 `BundleSchemaWarning` 을 낸다.
그런데 `answer_set()` 은 `Bundle.tick_ms` 를 **거치지 않으므로 그 경고도
안 난다.** 다른 GPU 번들에서 A6000 눈금이 쓰이는데 아무 신호가 없다.

`R-1` / `R-5` 와 같은 클래스다 — "조건이 안 맞으면 조용히 아무것도 안 한다".
`docs/decisions.md` 가 적은 대로, 이 저장소가 이 클래스를 여러 번 만났다.

### 영향 — 정답 집합이 통째로 어긋난다

눈금이 A6000 의 2배인 GPU 를 가정하고 같은 표로 계산한 것이다
(계수만 바꾸고 시간은 그대로 두었으니 순수한 계수 효과다).

| 형상 | 최적 | `answer_set()` | 번들 계수 기준 | 배수 |
|---|---:|---:|---:|---:|
| `512x512x512` | 11.26 us | 270 | 1,083 | **4.0x** |
| `1x4096x4096` | 59.39 us | 222 | 1,042 | **4.7x** |
| `1x12288x4096` | 156.67 us | 132 | 821 | **6.2x** |
| `1024x1024x1024` | 33.79 us | 149 | 259 | 1.7x |
| `4096x4096x4096` | 1282 us | 2 | 3 | 1.5x |

**허용치를 과소평가하는 방향이다.** `docs/baselines.md` 의 경고
("과소평가하면 노이즈를 신호로 배운다") 가 정확히 그 방향이고, 66형상 중
45개가 0.5 ms 미만이라 대부분의 형상이 영향을 받는다.

A6000 번들에서는 값이 우연히 같아 지금은 증상이 없다. **4090/H100 번들이
나오는 순간 조용히 틀린다.**

### 제안

세 가지 중 아무것도 API 를 깨지 않는다.

```python
# (a) 번들이 자기 계수로 정답 집합을 만든다 — 가장 안전
class Bundle:
    def answer_set(self, df, tol=None):
        if tol is None:
            tol = 2.0 * self.noise_floor(float(df["time_ms"].min()))
        return df[df["time_ms"] <= df["time_ms"].min() * (1.0 + tol)]

# (b) answer_set 이 계수를 받게 한다
def answer_set(df, tol=None, *, noise_floor=None):
    nf = noise_floor or _module_noise_floor
    ...

# (c) 최소한, 계수 출처를 강제한다
#     df 에 bundle_id 태그가 이미 붙어 있으므로(_tag) 그것으로 계수를
#     되찾을 수 있고, 못 찾으면 에러를 낸다
```

**(a) + (c) 를 권한다.** (c) 가 있어야 "헬퍼만 만들고 문서에 쓰라고
적는 것" 을 넘어선다 — `docs/decisions.md` 가 말한 강제 수단이다.

`load_for_scoring()` 이 `env_hash` 컬럼을 남기므로, 그것으로 계수를
조회하는 레지스트리를 두는 방법도 있다.

### kernelRule 쪽 대응 (참고)

kernelTab 을 고치지 않고 `kernelrule/core/noise.py` 에서 막았다.

```
NoiseModel.from_bundle(b)  ->  번들 계수만 읽는다
                               모듈 전역 상수와 대조해 다르면 NoiseMismatchError
                               tick_ms 대체 여부를 tick_is_fallback 에 기록
정답 집합은 PerfTable.answer_mask() 로 계산 (answer_set() 을 안 쓴다)
계산 불가 시 1.0(=100%, 아무것도 구분 못 함) 반환. 0 이 아니다
```

즉 **소비 쪽이 계약을 우회해야 안전해지는 상태**다. 그게 계약의 구멍이라는
신호이므로 보고한다.

---

## C-2. `docs/baselines.md` 의 정적 top-k 표가 `status` 필터 인공물이다

### 현상

`docs/baselines.md` 의 정적 top-k 표(덮개 100%, a888 61형상):

```
k        1       3       5       8      10      20
regret  1.3944  1.0604  1.0235  1.0090  1.0048  1.0009
```

이 표는 **두 절차가 섞여 있다.** 절차별로 다시 계산하면:

| 절차 | k=1 | k=3 | k=8 |
|---|---:|---:|---:|
| `ok` 만 + 개별 전덮개 | 1.394 | 1.383 | **1.383** ← 포화 |
| `ok` 만 + 합집합 덮개 | 1.394 | 1.060 | 1.009 |
| 전체 status + 합집합 덮개 | **1.115** | 1.031 | 1.006 |

k=1 행은 첫 번째 절차의 값인데 k>=3 행은 두 번째 절차의 값이다.
첫 번째 절차에서는 k>=3 이 1.383 에 포화한다.

### 원인

`status == "ok"` 로 자르면 **61형상 전부에서 ok 인 config 가 17,325개 중
3개**만 남는다. `high_outlier_frac` 은 그 **측정 한 건**의 속성이지 config
의 성질이 아니므로(반복 수가 많을수록 IQR 밖 하나가 걸릴 확률이 커진다),
`ok` 만 남기면 "모든 형상에서 우연히 깨끗한 측정이 나온 config" 를 요구하게
된다. 그리디가 셋 중에서 고르니 포화한다.

`best_ms` 자체는 두 정책에서 사실상 같다(중앙 0.2591 vs 0.2586 ms).
**차이는 전적으로 후보 집합의 덮개에서 온다.**

`docs/baselines.md` 자신이 경고한 클래스가 반대 방향으로 나타난 것이다 —
"미측정에 벌점을 주면 그 구조적 사실이 regret 으로 둔갑한다".

### 같이 확인할 것 — 나머지 세 베이스라인도 같은 필터를 쓴다

```
scripts/baseline_gbdt.py:62    df[... & (df.status == "ok")]
scripts/baseline_rule.py:47    load_for_ranking(pq, env_hash=eh)   # ok_only=True 기본
scripts/baseline_vendor.py:143 if ... or c["status"][i] != "ok": continue
```

**벤더 1.088 / 손규칙 1.192 / GBDT 1.019 세 값 모두 `ok` 만으로 계산됐다.**
정적 top-k 만큼 극적이지는 않을 것이다(이들은 형상마다 고르므로 "모든
형상에서 ok" 를 요구하지 않는다). 다만:

* 후보 풀에서 10.7% 가 빠지고, 그중에 그 형상의 진짜 최적이 있을 수 있다
* `best` 분모가 미세하게 달라진다
* GBDT 는 **학습 데이터가 10.7% 줄어든 셈**이다
* 벤더는 매핑 실패 시 최근접 대체를 하는데, `ok` 만 남기면 매핑 실패가 늘어
  최근접 대체 비율이 올라간다 (문서의 "엄격 79% / 최근접 100%" 가 바뀐다)

kernelRule 이 세 값을 **정본 절차(전체 status + 합집합 덮개)로 재계산**해
`baselines/` 에 넣을 예정이다. 재계산 결과는 그쪽 `docs/decisions.md` 에
남기고 여기로 회신한다.

`scripts/baseline_vendor.py --extract` 의 산출물(`vendor.json`)이 저장소에
없어서 재추출이 필요하다. **`nvidia-matmul-heuristics` 는 GPU 프리셋만
쓰고 실제 장치를 안 쓰므로 CPU 에서 돌릴 수 있다.** 그 산출물을
`docs/baselines/` 에 커밋해 두면 이후 재채점이 표만으로 가능해진다 —
지금은 재채점할 때마다 별도 venv 와 네트워크가 필요하다.

### 제안

1. `docs/baselines.md` 의 정적 top-k 표를 **절차 세 개 병기**로 바꾸고
   정본을 명시한다
2. `scripts/baseline_*.py` 에 `--status {ok,all}` 플래그를 두고 **기본을
   `all` 로** 한다. 지금은 `ok` 가 암묵적 기본이고 문서에 안 적혀 있다
3. `vendor.json` 을 `docs/baselines/` 에 커밋한다 (재현성)

---

## C-3. 탐색 축에 구멍이 있다 — 벤더가 알려준다

### 현상

kernelRule 이 벤더 휴리스틱(nvMatmulHeuristics, CUTLASS 타깃)을 우리 표에
매핑하다 발견했다. 벤더 추천 468건(a888 61형상 x top-8) 중 **31건(6.6%)** 이
우리 공간에 없다. 원인이 **축 값 두 개**로 좁혀진다.

```
split_k = 5   16건   축이 [1, 2, 3, 4, 6, 8, 12, 16] — 4와 6 사이가 비었다
stages  = 1   13건   축이 [2..8] — 최소 파이프라인이 2단이다
```

나머지 축(tile / warp / swizzle / mode)의 실패건 값은 **전부 우리 축에
존재하는 값**이다. 조합이 아니라 축 값 자체가 없다.

### K 양극단에 몰린다

```
K=16384  11건    K=128  8건    K=256  5건    K=11008  5건
K=4096    2건  <- 지배적 중간 구간은 거의 실패가 없다
짧은 형상 9.1% (28/308) vs 긴 형상 1.9% (3/160)
```

**물리적으로 말이 된다.**

* K 가 아주 크면 벤더가 `split_k=5` 를 원한다 — 우리 4와 6 사이 값이다
* K 가 아주 작으면(128, 256) `stages=1` 을 원한다 —
  `K/tile_k = 4` 회짜리 mainloop 에는 2단 파이프라인도 깊다

`SPLIT_K` 에 3, 6, 12 를 넣은 것은 "SM 84 = 2²x3x7" 가설 때문이었고
그 가설은 전수에서 기각됐다 (§18.3). **5 는 다른 이유로 필요하다.**

### 영향

이 31건 때문에 3형상에서 최근접 대체가 일어나고, 그것이 벤더 베이스라인의
strict/nearest 격차(1.080 vs 1.102) **전부**를 만든다. 벤더 품질 문제가
아니라 우리 축의 덮개 문제다.

### ★ 제안 — 축 덮개 점검을 절차로 만들어라

일반화하면 이렇다.

> **벤더 휴리스틱을 축 설계의 정보원으로 쓸 수 있다.** 벤더가 반복적으로
> 요청하는데 우리 축에 없는 값은, 그 값이 실제로 유용하다는 신호다.
>
> **다음 캠페인 전에 전체 형상 그리드에 대해 벤더 추천을 뽑아 축 덮개를
> 점검하라.** 측정 전에 할 수 있고 **GPU 도 쓰지 않는다**
> (nvMatmulHeuristics 는 GPU 프리셋만 쓰는 CPU 예측 모델이다).
> 30분이면 되고, 33시간 측정의 축이 비어 있는 것을 미리 막는다.

구체적으로:

1. `scripts/baseline_vendor.py --extract` 를 **다음 캠페인의 형상 그리드**에
   대해 먼저 돌린다
2. 추천된 축 값의 집합과 `backends/cutlass_v2.py` 의 축을 대조한다
3. 벤더가 N건 이상 요청하는데 없는 값은 축 추가를 검토한다
4. 추가하면 `env_hash` 를 갱신해야 한다 (`cutlass_v2.py` 상단 경고)

`docs/next_campaign.md` 의 축 설계 절에서 이 점검을 참조하게 하라.

### 이번 캠페인에 반영한다면

```
SPLIT_K:  [1, 2, 3, 4, 5, 6, 8, 12, 16]      + 5
STAGES:   [1, 2, 3, 4, 5, 6, 7, 8]           + 1  (CUTLASS 가 받는지 확인 필요)
```

`stages=1` 이 CUTLASS 2.x `MmaPipelined` 에서 유효한지는 확인이 필요하다 —
유효하지 않다면 "벤더가 표현하는 것을 우리가 못 만든다" 가 결론이고, 그
자체를 한계로 기록하면 된다.

### 부수 — `vendor.json` 을 커밋하라

`--extract` 산출물이 저장소에 없어서 재채점할 때마다 별도 venv 와 네트워크가
필요했다. kernelRule 은 `datasets/baselines/vendor-a6000-c63710df.json` 으로
저장했다. `docs/baselines/` 에 두면 표만으로 재현된다.

---

## D-1. `peak_tflops_f16_at_mhz` 가 없으면 실효 피크 보정이 **조용히 생략된다**

> ## ✅ 해결 (2026-09-01)
>
> **두 겹으로 막았다.**
>
> 1. **실행 시점** — `phase0_env.py` 가 클럭을 고정했는데 기준 클럭이 없으면
>    종료 코드 4 로 **실패한다.** 메시지가 어느 파일의 어느 항목에 무엇을
>    넣어야 하는지, 그리고 안 넣고 진행하면 무엇이 틀리는지 적는다.
> 2. **항목 추가 시점** — `tests/test_hwspec.py` 가 `known.json` 의 모든 GPU
>    항목에 필수 키 6 개가 있는지, 수치가 말이 되는지, `버스폭 x 2 x 기준
>    클럭`이 `bandwidth_gbps_spec` 과 맞는지, `source` 에 근거가 있는지 본다.
>    **항목을 추가하는 시점에 걸리는 것이 실행 시점에 걸리는 것보다 낫다.**
>
> 되돌려서 확인했다 — `peak_tflops_f16_at_mhz` 를 빼면 2 건, 대역폭 spec 을
> 틀리게 하면 1 건이 실패한다.

`scripts/phase0_env.py:589`

```python
peak_ref = peak_reference_mhz(hw.name)
peak_eff = hw.peak_tflops_f16
if lock.locked and lock.mhz and peak_ref:          # <-- 셋 중 하나만 없어도
    peak_eff = round(hw.peak_tflops_f16 * lock.mhz / peak_ref, 3)
```

`known.json` 에 `peak_tflops_f16_at_mhz` 를 빠뜨리면 `peak_ref` 가 `None` 이
되어 **스펙 피크가 그대로 `peak_tflops_f16_effective` 로 들어간다.** 경고도
오류도 없다. ridge point 가 틀리고 `is_memory_bound` 가 전 형상에서 틀린다.

README 가 "없으면 보정이 조용히 생략된다" 고 이미 적어 두었는데 **코드는
그대로다.** 14 번(조용히 아무것도 안 함) 계열이다.

**제안:** 클럭을 고정했는데(`lock.locked`) `peak_ref` 가 없으면 실패하거나
최소한 경고한다. 새 GPU 를 추가할 때 가장 밟기 쉬운 함정이다.

## D-2. `docker/Dockerfile` 의 자체 점검은 **테스트가 덮지 않는다**

> ## 🟡 부분 해결 (2026-09-01) — 완전하지 않다
>
> 근본 대책은 CI 에서 이미지를 실제로 빌드하는 것이지만 비용이 크다
> (베이스 11.9 GB + CUTLASS clone). 그 전에
> `tests/test_portability.py::TestDockerfileSelfCheck` 가 Dockerfile 의
> `from kerneltab... import ...` 줄을 뽑아 **실제 모듈 경로와 대조**한다.
> 이번 사고(`449c3f6` 이후 빌드 불가)를 재현해 잡는 것을 확인했다.
>
> ⚠️ **이 검사가 못 잡는 것을 함께 적는다** — apt 패키지, 빌드 인자(`ARG`),
> `COPY` 대상, `pip` lock 변경, 베이스 이미지 다이제스트. import 경로만 본다.
> **"테스트가 있으니 안전하다" 로 오해하지 마라.** 가장 흔한 종류를 막을 뿐이다.

`449c3f6` 이 `paths.py` 를 `build/` -> `core/` 로 옮기면서 `tests/test_paths.py`
는 같이 고쳤지만 `docker/Dockerfile` 의 자체 점검은 빠뜨렸다:

```
ImportError: cannot import name 'paths' from 'kerneltab.build'
```

**그 커밋 이후 이 이미지는 어느 호스트에서도 빌드되지 않았다.** 5090 캠페인에서
처음 빌드를 시도하며 발견했다. A6000 이미지는 그 이전에 만들어진 것이다.

15 번(문서로 적은 규율은 지켜지지 않는다)과 같은 계열인데, 여기는 **테스트를
붙일 자리가 없는** 곳이다 — Dockerfile 은 import 를 문자열로 들고 있다.

**제안:** 근본 대책은 CI 에서 이미지 빌드를 돌리는 것이지만 비용이 크다
(11.9 GB 베이스 + CUTLASS 클론). 최소한 **"import 경로를 옮길 때 `docker/` 도
grep 한다"** 를 절차로 둔다.

```bash
grep -rn 'from kerneltab' docker/     # 경로를 옮긴 커밋에서 반드시
```


## D-3. **탐색 축이 `env_hash` 에 안 들어간다**
> ## ✅ 해결 (2026-09-01, `env_hash` 정의 4)
>
> `axis_space_hash` / `shape_grid_hash` / `anchor_shape_hash` 를 키에 넣었다.
> 셋 다 `REQUIRED_V2` 라 비면 예외다 — `None` 을 허용하면 넣은 의미가 없다.
 — 지금은 `shuffle_seed` 의 우연에 기대고 있다

5090 캠페인에서 `SPLIT_K` 에 5, 7 을 추가한 뒤 `phase0_env.py` 를 다시 돌렸다.
`env_hash` 가 `d0b680b2` -> `47c31170` 으로 바뀌었다. **그런데 축이 바뀌어서가
아니었다.**

```
해시 입력 15 개 중 달라진 키:  ['shuffle_seed']
  shuffle_seed: 1780292639 -> 340178505      (매 실행 os.urandom)
★ SPLIT_K 는 ENV_HASH_KEYS 에 아예 없다
```

즉 `--seed 1780292639` 로 고정해서 다시 돌렸다면 **축이 8 개에서 10 개로
바뀌었는데도 같은 `env_hash`** 가 나왔을 것이다. 그러면 서로 다른 탐색 공간에서
잰 데이터가 같은 조건 식별자를 공유한다 — 격리 경계가 뚫린다.

`backends/cutlass_v2.py` 의 docstring 이 "이 파일을 고치면 사람이 직접 판단해
재측정하라" 고 경고하고 있지만, **그것은 문서이지 강제가 아니다** (15 번).
지금 우리를 구한 것은 설계가 아니라 시드가 무작위라는 우연이다.

### 제안 — `manifest_hash` 가 아니라 `axis_space()` 를 넣는다

`manifest_hash`(소스 tree_hash)를 해시 키에서 뺀 이유는 명확하다 — 오타
수정에도 해시가 바뀌어 측정 도중 아무것도 못 고치게 된다. 하지만
`Backend.axis_space()` 는 **정확히 맞는 입도**다:

```python
ENV_HASH_KEYS_V2 += ("axis_space_hash",)   # canonical_hash(backend.axis_space())
```

* 주석·docstring·로그 문구를 고쳐도 안 바뀐다
* `TB_TILES` / `STAGES` / `SPLIT_K` / `WARP_TILES` / `SWIZZLE` 가 바뀌면 **반드시** 바뀐다
* 이미 Protocol 에 있는 메서드다 (C-3 에서 추가했다). 새로 만들 것이 없다

⚠️ `ENV_HASH_DEF_VERSION` 을 3 -> 4 로 올려야 하고, 그러면 **A6000 번들의
기록된 해시를 오늘 다시 계산하면 다른 값이 나온다.** 기록된 값 자체는 안
변하지만(파일에 박혀 있다) 재현 절차가 갈라지므로 버전 표에 남겨야 한다.
그래서 이것은 캠페인 사이에 할 일이지 캠페인 도중에 할 일이 아니다.

### 그리고 `emit_cpp` 는 여전히 안 덮인다

`axis_space()` 는 축만 본다. `emit_cpp()` 의 템플릿 인자 순서나 epilogue
설정을 고치면 여전히 조용히 통과한다. 그것까지 덮으려면 생성된 소스의
해시를 넣어야 하는데, 그러면 사실상 `manifest_hash` 로 되돌아간다.
**완전한 해법은 없고, 축은 덮을 수 있다.**


## D-4. ⛔ **`env_hash` 필드는 아직 구 정의다**
> ## ✅ 해결 (2026-09-01, `env_hash` 정의 4)
>
> `env["env_hash"] = env_hash_v2(env)` 로 통일하고 구 정의는
> `env_hash_legacy` 로 기록만 남긴다. 파급을 먼저 확인했다 —
> `load_bundle` 은 `env_hash` 를 **데이터로 읽기만** 하고 재계산·대조하지
> 않으므로 **이미 릴리즈된 번들은 재생성 불필요**다. 세대는
> `env_hash_def_version`(None / 3 / 4)으로 구분한다.
>
> ⚠️ 정정: "구 정의는 재현 불가능" 은 **다시 측정할 때 값이 달라진다**는
> 뜻이지 기록을 못 푼다는 뜻이 아니었다. 번들 안 `env.json` 으로
> 재계산하면 기록값과 일치한다 (A6000 `828baa644b7dc088` 확인).
 — P-3 이 절반만 적용됐다

`scripts/phase0_env.py` 는 해시를 **둘** 쓴다:

```python
env["env_hash"]    = canonical_hash(env)     # 694행 — 구 정의 (env 전체 해싱)
env["env_hash_v2"] = env_hash_v2(env)        # 700행 — 신 정의 (키 15개만)
```

`core/env_hash.py` 의 docstring 은 신 정의를 **그 정의**로 소개하고 구 정의의
문제(실행마다 변하는 값이 섞임)를 상세히 적어 두었다. 그런데 **파이프라인
전체가 쓰는 것은 `env_hash` 이고 그것이 구 정의다.**

```
측정 줄        records.py 가 env_hash 로 필터한다
재개           sweep.py resume_state(env_hash)
표/번들        table.py, export.py, bundle.py 전부 env_hash
env_hash_v2    sweep.py 의 정합성 대조에서만 쓰인다
```

### 실증 (2026-08-28, 5090 준비 중)

조건을 완전히 고정하고(`--seed` 까지 고정) `phase0_env.py` 를 두 번 돌렸다:

| | `env_hash` (쓰이는 것) | `env_hash_v2` (정의된 것) |
|---|---|---|
| 실행 2 | `47c31170` | `0dc1e84c` |
| 실행 3 | **`1ce1b782`** ← 바뀐다 | `0dc1e84c` ← 동일 |

해시 입력 15 개 중 달라진 것은 **하나도 없다.** 구 해시가 본 것 중 변한 것:

```
created_utc                  13:43:15 -> 14:13:32      실행 시각
launch_overhead(async_small) 0.001704 -> 0.001664      ★ 측정값이다
```

**README 의 ⛔ 경고("측정 중 phase0_env 를 실행하지 마라 — 재실행하면 해시가
바뀌어 98 만 건을 다시 잰다")가 아직 살아 있다.** P-3 은 새 함수를 만들었지만
`env_hash` 필드를 그 함수로 바꾸지는 않았다.

### 그리고 `env_hash_v2` 도 형상 그리드를 안 본다

같은 실험에서 **층 B 의 M 과 층 E 의 사다리를 바꿨는데** `env_hash_v2` 가
`0dc1e84c` 로 동일했다. D-3 의 축(`SPLIT_K`)과 같은 구멍이 형상에도 있다.

```
env_hash_v2 가 못 보는 것
  탐색 축      SPLIT_K / STAGES / TB_TILES / WARP_TILES / SWIZZLE   (D-3)
  ★ 형상 그리드  shapes_layer_a..e                                   (여기)
  emit_cpp     템플릿 인자 순서, epilogue 설정                        (덮기 어렵다)
```

### 제안 — 셋을 한 번에, 캠페인 사이에

```python
ENV_HASH_KEYS_V2 += (
    "axis_space_hash",     # canonical_hash(backend.axis_space())
    "shape_grid_hash",     # canonical_hash([(p.M,p.N,p.K) for p in all_shapes(hw)])
    "anchor_shape_hash",   # canonical_hash(rehearse.DRIFT_SHAPES)   <- 아래 참조
)
env["env_hash"] = env_hash_v2(env)       # 구 정의를 버리고 신 정의로 통일
```

**앵커 형상도 같은 구멍이다.** 5090 G-7 에서 `DRIFT_SHAPES` 를 512³ ->
2048³ 로 바꿨는데 `env_hash` 도 `env_hash_v2` 도 안 바뀌었다. 앵커는 재현성
판정의 입력이고 드리프트 감시의 눈이므로 **측정 조건이다.** 옛 앵커
(`anchors.drift512.jsonl`)를 손으로 옮겨 분리해야 했다 — 해시가 잡아 줬다면
그럴 필요가 없다.

⚠️ 셋 다 파급이 크다.

* `env_hash` 를 바꾸면 **A6000 번들의 기록된 해시는 구 정의**이므로 두 세대의
  해시가 한 이름공간에 섞인다. 번들에 `env_hash_def_version` 이 있으니 그것으로
  구분해야 한다.
* `ENV_HASH_DEF_VERSION` 3 -> 4 가 필요하다.
* **측정 도중에는 절대 하지 마라.** 재개 키가 바뀐다.

**그때까지의 운용 규칙은 그대로다 — 측정이 시작되면 `phase0_env.py` 를 다시
돌리지 않는다.** 지금 5090 캠페인의 `env_hash` 는 `1ce1b782` 로 고정한다.


## D-5. `verify_clock_lock` 의 판정이 A6000 전용이다
> ## ✅ 해결 (2026-09-01)
>
> 판정 입력을 **클럭 분포**(목표 미만 비율 + 최대 이탈 폭)로 바꿨다.
> 지원 클럭 한 칸(`CLOCK_DIP_TOL = 1 %`) 이내면 부스트 입도로 본다.
> 전력 여유는 절대 와트가 아니라 **카드 상한 대비 비율**로 본다.
> `sw_power_cap` 은 기록만 하고 판정에 쓰지 않는다.


```python
swpc = thr.get("sw_power_cap", 0) / n
if swpc > 0.10:      verdict = "lower"
elif max(pw) <= 250: verdict = "raise"
else:                verdict = "hold"
```

5090 에서 이 판정은 **클럭이 완벽하게 유지되는데도 "낮춰라"** 를 낸다.
`sw_power_cap` 이 뜬 51 개 샘플의 클럭이 전부 목표값이었고 전력은 171 W
에서도 떴다 (`decisions.md` 22).

`max(pw) <= 250` 도 A6000 의 300 W 캡 기준이다. 5090 은 600 W 캡이라
250 W 는 41 % 에 불과한데 "여유 있으니 올려라" 가 나온다.

**제안:** 판정 입력을 클럭 분포로 바꾼다.

```
목표 미만 비율      clk < expect 인 샘플 비율
★ 최대 이탈 폭      (expect - min(clk)) / expect
   지원 클럭 한 스텝 이내면 부스트 입도이므로 정상으로 본다
```

스로틀 플래그는 **기록은 하되 판정에 쓰지 않는다.** 전력 캡 관련 비트의
의미가 SKU 마다 다르다.


## D-6. ⛔ 번들이 **A6000 노이즈 계수를 5090 번들에 싣는다** — 세 겹으로 막혀 있다

`answer_set()` 의 동점 판정이 이 계수에 직접 달려 있다. 틀리면 **정답 집합의
크기가 통째로 틀리고**, 그것은 순위 정보가 사라진다는 뜻이다.

### 겹 1 — `scripts/bundle.py:168`

```python
def _noise_coefficients() -> dict:
    """앵커에서 잰 노이즈 바닥 계수. 없으면 core.noise 의 기본값."""
    from kerneltab.core import noise
    return noise.coefficients()        # <- 모듈 상수. A6000 값이다
```

**docstring 은 "앵커에서 잰" 이라고 하는데 앵커를 읽지 않는다.**
`decisions.md` 23 과 같은 부류다 — docstring 이 옳고 코드가 틀렸다.

### 겹 2 — `core/noise.coef_from_observed()` 의 통계항

```python
base = A6000_MEASURED
...
return NoiseCoef(base.sigma_abs_ms, base.sigma_rel, est, ...)
```

눈금만 관측값으로 바꾸고 `sigma_abs_ms` / `sigma_rel` 은 **항상 A6000** 이다.
docstring 이 그 이유를 적어 두긴 했다 ("조건 간 차이가 작고 짧은 형상에서
지배하는 것은 눈금 항") — **A6000 에서는 맞는 말이었다.** 5090 에서는 눈금이
32 배 작아져 그 전제가 뒤집혔다.

### 겹 3 — `TICK_PLAUSIBLE` 이 A6000 눈금 기준이다

```python
TICK_PLAUSIBLE = (0.2, 5.0)      # A6000 눈금(1024ns) 대비 배수
# 허용 범위 204.8 ~ 5120 ns
```

**5090 의 실측 눈금 16 ns 는 이 범위 밖이라 버려지고 A6000 의 1024 ns 로
대체된다.** 즉 "관측 눈금을 쓴다" 는 경로조차 32 배 더 좋은 타이머를 조용히
거부한다.

### 결과

| t | A6000 계수로 계산한 바닥 | 5090 실측 계수 | 배수 |
|---:|---:|---:|---:|
| 0.083 ms (앵커) | **1.234 %** | 0.084 % | 14.7 |
| 0.5 ms | 0.205 % | 0.068 % | 3.0 |
| 2.9 ms | 0.057 % | 0.065 % | 0.9 |

`answer_set()` 의 기본 허용치가 `2 x noise_floor` 이므로, 짧은 형상에서
**정답 집합이 15 배 넓어진다.** 그 형상들이 순위 학습의 어려운 쪽이다.

### 제안

```
1. bundle.py 가 그 캠페인의 anchors.jsonl 에서 계수를 직접 뽑는다
   (짧은 앵커의 절대 sigma 중앙, 긴 앵커의 상대 sigma 중앙, 관측 눈금)
2. TICK_PLAUSIBLE 을 A6000 배수가 아니라 **절대 범위**로 (예: 1 ns ~ 10 us)
3. 계수를 못 뽑으면 A6000 으로 폴백하되 **경고가 아니라 실패**로.
   조용한 폴백이 정확히 이 문제를 만들었다 (decisions 14)
4. BUNDLE.json 의 noise_floor 에 `source`(어느 앵커에서, 몇 개로) 를 남긴다
```

⚠️ **측정 경로가 아니므로 전수 시작을 막지 않는다.** 다만 **번들을 만들기
전에는 반드시 고쳐야 한다** — 안 고치면 공개하는 표의 동점 판정이 틀린다.

## R-6. `OUTLIER_TOL` 이 **분모(`n_reps`)를 안 본다**

```
OUTLIER_TOL = 0.20  고정
n_reps = 5   -> IQR 밖 표본 하나가 1/5 = 0.20  ★ 자동으로 걸린다
n_reps = 100 -> 하나가 0.01
```

같은 측정 품질이 반복 수에 따라 다른 판정을 받는다. 긴 커널은
`target_ms = 20` 아래에서 반복이 바닥(`min_reps_floor = 5`)으로 깎이므로
**표본 하나가 빠지는 것만으로** `high_outlier_frac` 이 된다.

실측 (H100 전수): 이상치 행의 중앙 `n_reps` 가 **5**, `ok` 행은 50.
형상 수준 상관 `r(n_reps) = -0.527`, `r(log 시간) = +0.879`.

### 제안

문턱을 `n_reps` 에 따라 조정한다 — 예: `outlier_frac > max(0.20, 2/n_reps)`.
표본 둘이 빠져야 걸리게 하면 반복 수에 대한 대칭성이 생긴다.

### ⛔ 언제

**지금 고치지 마라.** 네 표의 `status` 가 전부 바뀐다. 다음 캠페인 **전**에
고치고, 그때 네 표를 같은 기준으로 다시 판정할지도 함께 정한다.

### 주의 — 경로는 표마다 다르다

이 결함은 넷이 공유하지만 발현 경로는 다르다 (`decisions` 33,
`consumer_contract` 18). A6000 은 **반대 방향**(반복이 많아서)이고 5090 은
미설명이다. 문턱을 고칠 때 H100 기준만 보고 정하지 마라.
