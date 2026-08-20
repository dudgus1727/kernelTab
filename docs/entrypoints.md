# 진입점 정리와 단일 CLI 설계안

컨테이너는 "이미지 하나 + 인자로 동작 선택" 이어야 한다. 현재는 `scripts/` 에
14 개 스크립트가 흩어져 있고 실행 순서가 README 에만 있다.

**이 문서는 설계안이다. Python CLI 는 구현하지 않았다.**

> ### 대신 컨테이너에 셸 shim 을 두었다 (2026-08-20)
>
> `docker/entrypoint.sh` 가 동사를 `scripts/*.py` 로 넘긴다.
>
> ```
> kerneltab detect | build | drift | probe | rehearse | sweep | anchors
>           export | bundle | validate | verify {smem,splitk,clock}
>           manifest | watch | test
> ```
>
> 아래의 통합 CLI(`kerneltab/cli.py`)를 만들지 **않은** 이유:
>
> * **resume 명령이 바뀌면 안 된다.** 30~40 시간짜리 측정이 중단됐을 때
>   같은 명령으로 재개되어야 하고, `scripts/*.py` 경로가 그 계약이다.
> * CLI 통합은 패키지 이전과 **독립적인 작업**이다. 한 커밋에 섞으면
>   무엇이 무엇을 깼는지 알 수 없다.
> * shim 은 패키지 코드를 전혀 건드리지 않는다. 컨테이너 밖 사용법
>   (`python3 scripts/xxx.py`)이 그대로 남는다.
>
> 실행 절차는 `docs/container.md`.

---

## 1. 현재 스크립트 목록

의존 관계 순. `E` = `results/env.json`, `K` = `results/kernels.jsonl`,
`R` = `results/results.jsonl`.

### 파이프라인 본선

| # | 스크립트 | 하는 일 | 입력 | 출력 | GPU |
|---|---|---|---|---|---|
| 1 | `phase0_env.py` | 환경 점검, 하드웨어 감지, CUTLASS 예제 검증, 클럭 상태 확정, 런치 오버헤드 측정 | (없음) + CLI 인자 | **E**, `clock_lock_check.json` 참조 | 필요 |
| 2 | `count_space.py` | config 열거, 제약 funnel, 형상/커널/런타임 개수 | E | (표준출력만) | 불필요 |
| 3 | `build_kernels.py` | `emit_cpp` → nvcc → ptxas/SASS 분석 → `kt_info` | E | **K**, `artifacts/{src,lib}` | 필요(introspect) |
| 4 | `rehearse.py` | 리허설 측정 (6 형상 × 표본 20 커널) | E, K | **R**, `drift.jsonl`, `telemetry.csv`, `repro.jsonl`, `guard_probe.json` | 필요 |
| 5 | `rehearse.py --all` | **Phase 3 전수 측정** | E, K | **R**, `drift.jsonl`, `telemetry.csv` | 필요 |
| 6 | `export.py` | R ⨝ K 조인 + 파생 지표 계산 | E, K, R | `table.parquet` | 불필요 |
| 7 | `report_rehearsal.py` | 리허설 분석 리포트 | E, K, R, `drift.jsonl`, `telemetry.csv` | (표준출력만) | 불필요 |

### 검증 스크립트 (파이프라인 밖, 있어야 하는 것)

| 스크립트 | 하는 일 | 언제 | GPU |
|---|---|---|---|
| `verify_clock_lock.py` | 5 분 고부하로 클럭 고정이 실제 유지되는지 | phase0 직후 | 필요(독점) |
| `check_smem.py` | `smem_bytes()` 공식 vs 실제 `sizeof(SharedStorage)` | 대량 빌드 전 | 불필요(컴파일만) |
| `validate_constraints.py` | 제약 예측 모델 vs 실제 빌드 결과, smem/HMMA/레지스터/occupancy 교차검증 | 빌드 후 | 불필요 |
| `check_correctness.py` | 빌드된 커널 전수 정확성 (cuBLAS 대조) | 빌드 후 | 필요 |
| `verify_swizzle.py` | 직접 구현한 horizontal 스위즐이 맞는지 | 스위즐 코드 변경 시 | 필요 |
| `smoke_splitk.py` | split-K serial/parallel 정확도, `grid.z` vs 예측 | 측정 전 | 필요 |
| `recheck_stability.py` | 재현성 / 드리프트 재측정 | 조건 변경 시 | 필요(독점) |
| `manifest.py` | 재현성 매니페스트 + 이미지 태그 | 언제든 | 불필요 |

**GPU 독점이 필요한 것**과 아닌 것을 구분해야 한다. `verify_clock_lock.py` 와
`recheck_stability.py` 는 다른 부하가 있으면 결과가 무의미하다 (실제로
정확성 검사를 병행했다가 재현성 통계가 오염된 적이 있다).

### 산출물 의존 그래프

```
phase0_env ──> env.json ─┬─> count_space
                         ├─> build_kernels ──> kernels.jsonl ─┬─> check_correctness
                         │                                    ├─> validate_constraints
                         │                                    ├─> smoke_splitk
                         │                                    └─> rehearse [--all]
                         │                                            │
                         │                          results.jsonl ────┼─> export ──> table.parquet
                         │                          drift.jsonl       └─> report_rehearsal
                         │                          telemetry.csv
                         └─> verify_clock_lock ──> clock_lock_check.json
                                                        │
                                            (phase0_env 가 다시 읽어 env.json 에 포함)
```

`verify_clock_lock` → `clock_lock_check.json` → `phase0_env` 의 **순환**에 주의.
실제 순서는 `phase0_env`(1 차) → `verify_clock_lock` → `phase0_env`(재생성) 이다.
현재 README 에만 적혀 있고 도구가 강제하지 않는다.

---

## 2. 단일 CLI 설계안

```
kerneltab <command> [options]
```

`python -m kerneltab` 또는 콘솔 스크립트. 컨테이너 `ENTRYPOINT` 가 된다.

### 명령

| 명령 | 대체하는 스크립트 | 비고 |
|---|---|---|
| `kerneltab detect` | `phase0_env.py` | 하드웨어 감지 + `env.json` |
| `kerneltab enumerate` | `count_space.py` | config 열거 + 개수 보고 |
| `kerneltab build` | `build_kernels.py` | 커널 빌드 |
| `kerneltab rehearse` | `rehearse.py` | 리허설 측정 |
| `kerneltab measure` | `rehearse.py --all` | 전수 측정 |
| `kerneltab export` | `export.py` | parquet 생성 |
| `kerneltab report` | `report_rehearsal.py` | 리포트 생성 |
| `kerneltab verify <what>` | 검증 스크립트 7 종 | `clocks\|smem\|constraints\|correctness\|swizzle\|splitk\|stability` |
| `kerneltab manifest` | `manifest.py` | 태그/매니페스트 |

검증을 `verify` 하위로 묶는 이유: 개수가 많고, 파이프라인 본선이 아니며,
"무엇을 검증하는가" 가 인자로 표현되는 편이 자연스럽다.

### 공통 옵션 (모든 명령)

```
--gpu <uuid|index>     기본: CUDA_VISIBLE_DEVICES 존중, 없으면 0
                       컨테이너에서는 지정하지 않는 것이 정상
--results-dir <path>   기본: $KERNELTAB_RESULTS_DIR 또는 <repo>/results
--artifact-dir <path>  기본: $KERNELTAB_ARTIFACT_DIR 또는 <repo>/artifacts
--cutlass <path>       기본: $KERNELTAB_CUTLASS_DIR 또는 /opt/cutlass
--jobs <n>             빌드 병렬도. 기본 min(48, cpu_count)
-v / --json            사람이 읽는 출력 / 기계가 읽는 출력
```

`--gpu` 가 **UUID 를 받는 것**이 핵심이다 (감사 P-2 참조). 인덱스는 컨테이너
경계에서 의미가 바뀌지만 UUID 는 바뀌지 않는다.

### 명령별 주요 옵션

```
detect     --externally-locked-mhz <n>       호스트에서 SM 클럭을 고정해 둔 경우
           --externally-locked-mem-mhz <n>   메모리 클럭 (부하 중 실제 관측값)
           --seed <n>                        측정 순서 셔플 시드 고정
           --skip-example                    CUTLASS 예제 검증 생략(빠른 재생성)

build      --align a888,a448|all             빌드할 alignment 조합
           --pilot <n>                       n 개만 빌드하고 멈춘다 (파이프라인 점검)
           --force                           캐시 무시

measure    --dry-run                         규모만 계산
           --limit <n>                       n 개만 (스모크)
           --resume/--no-resume              기본 resume

verify clocks       --minutes 5
verify correctness  --align a888  --gpu <다른 GPU 가능>
verify stability    --n 30 --passes 4 --shapes MxNxK,...
```

### 종료 코드 규약

파이프라인을 스크립트로 엮으려면 필요하다. 현재는 일부만 지킨다.

```
0  성공
1  일반 실패
2  전제 조건 미충족 (env.json 없음, 커널 미빌드, pynvml 없음 …)
3  임계 초과로 자발적 중단 (build_fail > 10%, 드리프트 5% 3 연속 …)
4  검증 실패 (예측 모델 불일치, 정확성 오류 …)
```

### 상태 조회

```
kerneltab status
```
`env.json` / `kernels.jsonl` / `results.jsonl` 을 읽어 "지금 어디까지 왔는가" 를
한 화면에 보여준다. 40 시간짜리 작업에서는 이게 있어야 한다. 지금은 매번
`wc -l` 과 로그 grep 으로 확인하고 있다.

---

## 3. 컨테이너 실행 시나리오

```bash
# 호스트에서 (컨테이너 안에서는 불가능)
sudo nvidia-smi -i 3 -pm 1
sudo nvidia-smi -i 3 -lgc 1350,1350
sudo nvidia-smi -i 3 -lmc 8001

IMG=kerneltab:cu124-9b70b430
VOL="-v $PWD/results:/work/results -v $PWD/artifacts:/work/artifacts"
GPU='--gpus "device=3"'          # 컨테이너 안에서는 index 0 으로 보인다

docker run --rm $GPU $VOL $IMG detect \
    --externally-locked-mhz 1350 --externally-locked-mem-mhz 7601
docker run --rm $GPU $VOL $IMG verify clocks --minutes 5
docker run --rm $GPU $VOL $IMG detect \
    --externally-locked-mhz 1350 --externally-locked-mem-mhz 7601   # 재생성
docker run --rm $GPU $VOL $IMG enumerate
docker run --rm $GPU $VOL $IMG build --align a888,a448 --jobs 40
docker run --rm $GPU $VOL $IMG verify constraints
docker run --rm $GPU $VOL $IMG verify correctness
docker run --rm $GPU $VOL $IMG rehearse
docker run --rm $GPU $VOL $IMG measure          # 수십 시간
docker run --rm      $VOL $IMG export           # GPU 불필요

# 호스트에서 정리
sudo nvidia-smi -i 3 -rgc && sudo nvidia-smi -i 3 -rmc
```

`detect` 를 두 번 부르는 것은 `verify clocks` 결과를 `env.json` 에 넣기
위해서다. 이 순환을 없애려면 `detect` 가 클럭 검증을 직접 수행하는 편이 낫다
(`detect --verify-clocks 5`).

---

## 4. R. `verify clocks` ↔ `detect` 순환 해소안

### 지금의 문제

```
detect ──> env.json         (clock_lock_check.json 이 있으면 읽어서 포함)
   ↑                              │
   └──────────────────────────────┘
verify clocks ──> clock_lock_check.json
```

`detect` 가 `clock_lock_check.json` 을 읽어 `env.json` 에 넣기 때문에, 순서가
`detect` → `verify clocks` → **`detect` 다시** 가 된다. CLI 로 정리할 때
"같은 명령을 두 번 부르는데 두 번째가 중요하다" 는 구조는 설명하기 어렵고,
두 번째 호출을 잊으면 **클럭 검증 증거가 빠진 채로 측정이 시작된다.**

게다가 `detect` 재실행은 `env_hash` 를 바꾼다 (P-3). 즉 이 순환은 P-3 와
맞물려 "두 번째 detect 를 언제 부르느냐" 가 데이터 정합성을 좌우한다.

### 안 A — `detect` 가 클럭 검증을 직접 수행한다 (권장)

```
kerneltab detect --verify-clocks 5     # 5 분 부하 테스트를 내부에서 실행
```

`detect` 한 번으로 끝난다. 순환이 사라지고 "검증 없이 `--externally-locked-*`
를 믿는" 구멍(감사 4 절)도 같이 막힌다 — 인자로 받은 값과 부하 중 실측값이
다르면 그 자리에서 실패시킬 수 있다.

* 장점: 순환 제거, 검증 강제, `env_hash` 를 한 번만 만든다.
* 단점: `detect` 가 5 분 걸리고 GPU 독점을 요구한다. 커널을 아직 안 빌드한
  상태이므로 **검증용 커널을 하나 즉석 빌드**해야 한다 (~20 초). 지금
  `verify_clock_lock.py` 는 이미 빌드된 커널을 골라 쓰므로 그대로는 못 쓴다.
* 완화: `--verify-clocks 0` 으로 생략 가능하게 하되 기본값을 켜 둔다.
  생략하면 `env.json` 에 `clock_verified: false` 를 남긴다.

### 안 B — 병합하지 않고 참조만 한다

`env.json` 이 `clock_lock_check.json` 의 **내용**을 담지 않고 파일명과 그
파일의 해시만 담는다.

```json
"clock_check_ref": {"path": "clock_lock_check.json", "sha256": "..."}
```

`detect` 는 한 번만 부르고, `verify clocks` 는 나중에 아무 때나 부른다.
분석 시점에 두 파일을 조인한다.

* 장점: `detect` 가 가벼워지고 GPU 독점이 필요 없다.
* 단점: **`env_hash` 가 클럭 검증 결과를 담지 않는다.** "이 측정이 검증된
  클럭 조건에서 이루어졌는가" 가 데이터만으로 결정되지 않고, 파일이 옆에
  있어야 한다. 볼륨을 잘못 마운트하면 증거가 사라진다.

### 안 C — `detect --stage1` / `detect --finalize`

명시적으로 두 단계로 나눈다. 지금 하는 일과 같지만 이름이 정직해진다.

* 장점: 각 단계가 하는 일이 분명하다.
* 단점: 여전히 두 번 불러야 하고, `--finalize` 를 잊으면 같은 문제가 난다.
  **문제를 이름으로 바꿔 부를 뿐 해소하지는 않는다.**

### 결론

**안 A 를 권장한다.** 순환을 없애는 유일한 안이고, "검증 없이 인자를 믿는"
별개의 구멍까지 같이 막는다. 5 분 비용은 40 시간짜리 측정 앞에서 무시할 수
있고, 애초에 지금도 사람이 그 5 분을 쓰고 있다.

전제: 검증용 커널을 즉석 빌드하는 경로가 필요하다 (`detect` 가 대표 config
하나를 `emit_cpp` → nvcc 로 만들고 버린다). `check_smem.py` 가 이미 같은
일을 하므로 재사용할 수 있다.

---

## 5. 설계상 결정해야 할 것

1. **`measure` 를 컨테이너 하나로 40 시간 돌릴 것인가.** 중간에 컨테이너가
   죽으면 resume 이 받아주지만, 매번 20 초 워밍업이 다시 들어간다. 문제는
   아니지만 재개 지점마다 미세한 조건 차이가 생긴다.
2. **`build` 산출물 볼륨의 수명.** GPU 아키텍처가 같으면 재사용 가능하지만
   `kernel_id` 에 arch 가 들어 있으므로 섞여도 안전하다. 다만 7.4 GB × GPU
   종류만큼 늘어난다.
3. **`verify` 를 파이프라인이 강제할 것인가.** 지금은 사람이 순서를 지켜야
   한다. `measure` 가 시작 전에 `validate_constraints` 결과를 확인하도록
   만들면 실수가 줄지만, 결합이 생긴다.
4. **한 이미지에서 여러 GPU 를 동시에 측정할 것인가.** 같은 서버의 PCIe/
   메모리 대역폭을 공유하므로 측정끼리 간섭한다. 하지 않는 것이 맞다.
