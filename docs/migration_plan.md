# 마이그레이션 계획 — 측정 완료 직후 적용

`docs/portability_audit.md` 의 수정 11 건을 **전수 측정이 끝난 뒤** 적용하는
절차서다. 그때 이 문서를 그대로 따라 실행할 수 있어야 한다.

> ## ⛔ 착수 조건
>
> 1. `python3 scripts/watch.py` 가 종료 코드 3(완료)일 것.
>    **`pgrep` 로 확인하지 마라** — 감시 셸 자신에 매칭된다 (D-4)
> 2. `python3 scripts/validate_table.py` 통과 (종료 코드 0)
> 3. `git status --porcelain` 가 비어 있을 것 (되돌릴 지점이 명확해야 한다)
> 4. **백업**: `results/` 를 통째로 복사해 둔다. 40 시간짜리 데이터다.
>    ```bash
>    tar -C .. -czf ~/kerneltab-results-$(date +%Y%m%d).tar.gz kernelTab/results
>    ```

---

## 순서 — P-3 → P-1 → P-2 → 나머지

제안하신 순서에 동의한다. 근거:

* **P-3 을 먼저** 하는 이유는 이것만이 **기존 데이터의 해석을 바꾸기**
  때문이다. 나머지는 코드 동작만 바꾼다. 데이터 계층을 먼저 고정해야
  이후 수정의 회귀 테스트를 "기존 데이터로" 할 수 있다.
* **P-1(`so_path`) 을 P-2(GPU 선택) 보다 먼저** 하는 이유는 P-1 이
  `kernels.jsonl` 을 읽는 방식을 바꾸기 때문이다. P-2 는 순수 런타임
  동작이라 데이터와 무관하다. 데이터에 가까운 것부터 처리한다.
* 나머지 8 건은 서로 독립이므로 순서가 자유롭다. 단 **7 번(pyarrow 필수화)
  을 6 번(경로 환경변수) 보다 먼저** 하는 편이 낫다 — 6 번을 검증하려면
  `export.py` 가 돌아야 하고, 그러려면 pyarrow 의존이 정리되어 있어야 한다.

각 단계 사이에 커밋한다. 한 단계가 잘못되면 그 커밋만 되돌린다.

---

> ## ✅ P-3 완료 (2026-08-19)
>
> `core/env_hash.py` + `scripts/migrate_env_registry.py` + `tests/test_env_hash.py`.
>
> **핵심 검증 통과** — 같은 조건으로 `phase0_env.py` 를 두 번 돌렸을 때:
> ```
> 1회차   구 65e39f70...   v2 1b6a0c0c...
> 2회차   구 40d1df47...   v2 1b6a0c0c...
>         ^^^ 매번 다름     ^^^ 동일
> ```
> 구 해시가 조건이 같은데도 매번 다르다는 것이 실증됐다. v2 는 안정적이고
> 캠페인 원본(`1b6a0c0c`)과도 일치한다.
>
> **계획서와 달라진 점 두 가지:**
>
> 1. `segments`, `soak` 을 해시 키에 **추가**했다. 계획서 작성 뒤에 생긴
>    필드인데 둘 다 측정 조건이다 (드리프트 대책의 세그먼트 크기, 워밍업 초).
>    빠뜨리면 대책이 다른 데이터가 같은 해시로 섞인다.
> 2. orphan 이 `de6de3c1` 이 아니라 **`ff1f3049`** 였다. `env.minreps30.json`
>    이 `de6de3c1` 을 보존하고 있었다. 진짜 orphan 은 메모리 클럭 기록을
>    빠뜨려 재실행하면서 덮어쓴 `ff1f3049`(7,538줄)다.
>
> 레지스트리 백필 결과 (`results/env_registry.jsonl`):
> ```
> 368a84f1 -> ffeff01e   226,211줄  env-368a84f1.json
> c63710df -> 1b6a0c0c   991,210줄  env.json
> de6de3c1 -> 990f406d       834줄  env.minreps30.json
> b42df475 -> d58da5cf     1,526줄  env.pre-clocklock.json
> dda3431a -> f763c610         0줄  env.smlock-only.json
> ff1f3049 -> (없음)       7,538줄  orphan (--allow-orphan 으로 명시 허용)
> ```
> 중단 조건도 확인했다 — `--allow-orphan` 없이 돌리면 exit 3 으로 멈춘다.
> `results.jsonl` 은 건드리지 않았고 재개는 그대로 동작한다(980,981 조합 인식).

## 1. P-3 — `env_hash` 재정의 + `env_registry`

**가장 위험한 수정이다.** 잘못하면 98 만 건이 조회 불가능해진다.

### 1-1. 무엇을 바꾸는가

현재 `phase0_env.py: canonical_hash(env)` 는 `env` 딕셔너리 **전체**를
해싱한다. 여기에 실행마다 변하는 값이 들어 있다 (`created_utc`,
`host.ram_available_gb`, `host.hostname`, 경로들).

신 정의: **측정 조건에 실제로 영향을 주는 필드만** 해싱한다.

```python
ENV_HASH_KEYS_V2 = (
    "hardware",                    # 감지된 GPU 스펙 (실효 피크/대역폭 포함)
    "nvcc_arch_flag",
    "protocol",                    # target_ms, min_total_ms, floor, cap, max_reps ...
    "clock_locked", "locked_mhz",
    "mem_clock_locked", "locked_mem_mhz",
    "peak_tflops_f16_effective",
    "bandwidth_gbps_effective",
    "shuffle_seed",
    "cutlass_commit",              # cutlass.commit
    "nvcc_version",                # cuda.nvcc_version
)
```

제외: `created_utc`, `host.*`, `cutlass.dir`, `cuda.nvcc_path`, `device_index`,
`gpu_smi`, `launch_overhead`(측정값이라 실행마다 미세하게 다르다),
`cutlass_example_check`.

### `manifest_hash` 는 해시 키에서 **제외한다** (결정됨)

`manifest_hash` 는 소스 `tree_hash` 를 포함하므로 **코드 한 글자만 바뀌어도
해시가 바뀐다.** 측정 도중 오타 수정조차 불가능해진다 — 규율의 문제가 아니라
실용성의 문제다. 대신 `cutlass_commit` + `nvcc_version` 을 넣는다. 측정
결과에 실제로 영향을 주는 것은 커널 생성 로직과 측정 프로토콜인데, 후자는
`protocol` 필드가, 전자의 대부분은 `cutlass_commit` 이 커버한다.

`manifest_hash` 는 `env.json` 에 **기록은 한다** (`env["manifest"]`). 사후
추적은 여전히 가능하다. 해시 키에서만 뺀다.

> #### ⚠️ 이 선택의 대가 — 반드시 문서·코드 양쪽에 남길 것
>
> **`backends/sm80.py`(커널 생성 코드)를 수정하면 `env_hash` 가 바뀌지
> 않는다.** `cutlass_commit` 도 `protocol` 도 그 변화를 잡지 못하므로,
> 같은 `env_hash` 아래에 서로 다른 커널로 잰 데이터가 섞일 수 있다.
>
> → 그 파일을 고칠 때는 **조건이 달라졌는지 사람이 직접 판단**하고,
> 달라졌다면 `phase0_env.py` 를 다시 돌려 `env_hash` 를 갱신한 뒤
> 재측정해야 한다. 이 경고를 `backends/sm80.py` 모듈 docstring 상단에
> 넣어 두었다 (어떤 변경이 조건을 바꾸는지 목록 포함).
>
> 같은 이유로 `measure/kt_kernel_impl.h`, `measure/kt_swizzle.h`,
> `measure/kt_ctx.cu` 도 해당한다. 이들은 커널 생성/측정 루프의 실체다.

### 1-2. `env_registry.jsonl` 백필

```
results/env_registry.jsonl   (append-only)
{"env_hash": "<구>", "env_hash_v2": "<신>", "recorded_utc": "...", "env": {...전체...}}
```

백필 대상 — 지금 남아 있는 것 전부:

| 파일 | 구 `env_hash` | 조건 |
|---|---|---|
| `results/env.pre-clocklock.json` | `b42df475…` | 클럭 미고정. 리허설 데이터 |
| `results/env.minreps30.json` | `1f0b6924…` | SM 클럭만 고정, min_reps=30 |
| `results/env.smlock-only.json` | `dda3431a…` | SM 클럭 고정 + 새 프로토콜 |
| `results/env.json` (현재) | `368a84f1…` | SM+메모리 클럭 고정. **⛔ 드리프트로 폐기** — `docs/measurement_drift.md` |

주의: `results.jsonl` 에는 `de6de3c1…` 도 있다 (834 줄). 메모리 클럭 고정
직전의 resume 테스트 잔여물이며 **해당 env.json 파일이 남아 있지 않다.**
→ 백필 시 이 해시는 `env` 를 복원할 수 없다. 두 가지 중 택일:
1. `{"env_hash": "de6de3c1…", "env_hash_v2": null, "env": null,
   "note": "orphan — env.json 미보존, 메모리클럭 고정 전 resume 테스트"}`
   로 명시적으로 기록하고 분석에서 제외한다. **(권장)**
2. 해당 834 줄을 폐기 데이터로 간주하고 무시한다.

권장안 1 을 택하는 이유: 폐기하더라도 "왜 이 줄들이 있는가" 가 기록에
남아야 한다. 조용히 무시하면 나중에 누군가 다시 발견하고 혼란스러워한다.

### 1-3. 백필 스크립트 (측정 완료 후 작성)

```
scripts/migrate_env_registry.py
  1) results/env*.json 을 전부 읽는다
  2) 각각에 대해 구 해시(파일에 기록된 env_hash)와 신 해시(신 정의로 계산)를
     구해 env_registry.jsonl 에 append
  3) results.jsonl 의 모든 줄을 훑어 등장하는 env_hash 집합을 만든다
  4) 그 집합이 레지스트리에 **전부** 있는지 확인
  5) 하나라도 없으면 **비영(non-zero) 종료**하고 목록을 출력. 진행 금지.
```

**중단 조건**: `results.jsonl` 에 등장하는 `env_hash` 중 레지스트리에 없는
것이 하나라도 있으면 중단한다. 매핑 불가능한 데이터를 남긴 채로 진행하면
"어떤 조건에서 잰 것인지 모르는 줄" 이 표에 섞인다.

단, `de6de3c1…` 처럼 **의도적으로 orphan 으로 기록한 것**은 예외로 허용하되
`--allow-orphan de6de3c1…` 처럼 **명시적으로 지정**해야 통과시킨다. 기본은
중단이다.

### 1-4. 검증

```bash
# (a) 모든 줄이 매핑되는가
python3 scripts/migrate_env_registry.py --check-only
#   -> "results.jsonl 1,005,xxx 줄, env_hash 4 종, 매핑 실패 0" 이어야 한다

# (b) 신 해시가 조건이 같은 실행에 대해 안정적인가 — 회귀 테스트
#     (측정 완료 후이므로 phase0_env.py 를 두 번 돌려도 안전하다)
python3 scripts/phase0_env.py --device 3 --externally-locked-mhz 1350 \
        --externally-locked-mem-mhz 7601 --seed 3053988298 --skip-example
python3 -c "import json;print(json.load(open('results/env.json'))['env_hash_v2'])"
# 두 번 실행해 **같은 값**이 나와야 한다. 다르면 아직 변동 필드가 남아 있다.

# (c) resume 이 여전히 동작하는가
python3 scripts/rehearse.py --all --dry-run
#   -> "이미 측정 <98만>" 이 나와야 한다. 0 이면 매핑이 안 된 것이다.
```

(b) 가 이 수정의 핵심 검증이다. **같은 조건 두 번 실행 → 같은 해시**가
성립하지 않으면 수정이 실패한 것이다.

### 1-5. 롤백

`env_registry.jsonl` 은 새 파일이므로 지우면 된다. `phase0_env.py` /
`rehearse.py` 변경은 `git revert`. `results.jsonl` 은 **건드리지 않으므로**
롤백이 필요 없다 — 이것이 안 A(레지스트리)를 택한 가장 큰 이유다.

---

## 2. P-1 — `so_path` 절대 경로 제거

### 무엇을 바꾸는가

`kernels.jsonl` 에 `so_path` 를 쓰지 않고, 읽는 쪽이
`ARTIFACT_DIR/lib/{kernel_id}.so` 로 조립한다.

* `build/compile.py: build_kernel` — `row["so_path"]` 기록 제거 (또는 유지하되
  읽는 쪽이 무시)
* `Kernel(r["so_path"])` 호출 3 곳: `rehearse.py`, `smoke_splitk.py`,
  `check_correctness.py`, `verify_clock_lock.py`

### 깨질 수 있는 것

기존 `kernels.jsonl` 7,330 줄에는 `so_path` 가 남아 있다. 읽는 쪽이 무시하고
조립하면 문제없다. **단 `.so` 파일이 실제로 그 위치에 있어야 한다** —
`build/artifacts/lib/` 를 지우면 안 된다.

### 검증

```bash
# 조립 경로가 기존 so_path 와 100% 일치하는지 (파일 접근 없이)
python3 - <<'EOF'
import json, pathlib
from build import paths
bad = 0
for l in open('results/kernels.jsonl'):
    r = json.loads(l)
    want = paths.ARTIFACT_DIR / "lib" / f"{r['kernel_id']}.so"
    if r.get("so_path") and pathlib.Path(r["so_path"]) != want:
        bad += 1; print("불일치:", r["kernel_id"])
print("불일치", bad)
EOF
# -> 0 이어야 한다

# 실제 로드되는지 (GPU 컨텍스트 없이 dlopen 만)
python3 -c "
import ctypes, json
from build import paths
r = json.loads(open('results/kernels.jsonl').readline())
ctypes.CDLL(str(paths.ARTIFACT_DIR/'lib'/f\"{r['kernel_id']}.so\"))
print('dlopen ok')"
```

### 롤백

`git revert`. 데이터 변경 없음.

---

## 3. P-2 — GPU 선택을 UUID 기반으로

### 무엇을 바꾸는가

1. `CUDA_VISIBLE_DEVICES` 가 **이미 설정되어 있으면 덮어쓰지 않는다** (7 곳).
2. `env.json` 의 권위 값을 `device_index` → `hardware_extra.uuid` 로.
   실행 시 UUID → 현재 인덱스 역조회. 못 찾으면 명확히 실패.
3. `nvidia-smi -i` 인자를 UUID 로 준다 (`nvidia-smi -i GPU-93284c84-...`).

### 깨질 수 있는 것

* **텔레메트리가 조용히 죽는 것**이 가장 위험하다. 지금도 `Popen` 이 실패해도
  예외가 안 난다 (수정 9 번과 함께 처리할 것).
* 멀티 GPU 서버에서 UUID 역조회가 잘못되면 **다른 GPU 를 측정**하게 된다.
  이건 조용히 틀린 데이터를 만든다.

### 검증

```bash
# (a) UUID 로 nvidia-smi 가 동작하는지
nvidia-smi -i $(python3 -c "import json;print(json.load(open('results/env.json'))['hardware_extra']['uuid'])") \
           --query-gpu=index,uuid,name --format=csv

# (b) 역조회가 맞는지 — 감지한 UUID 가 env.json 과 같은지
python3 -c "
import json
from core.hardware import device_uuid
env=json.load(open('results/env.json'))
got, want = device_uuid(0), env['hardware_extra']['uuid']
print('일치' if got==want else f'불일치! {got} vs {want}')"

# (c) CUDA_VISIBLE_DEVICES 존중 테스트
CUDA_VISIBLE_DEVICES=0 python3 scripts/count_space.py >/dev/null && echo "존중 ok"

# (d) 텔레메트리가 실제로 기록되는지 (수정 9 와 함께)
#     -> telemetry.csv 가 비어 있지 않고 1 초에 1 줄씩 늘어야 한다
```

(b) 는 **다른 GPU 를 측정하지 않는다**는 것을 확인하는 유일한 검증이다.
반드시 할 것.

### 롤백

`git revert`. 단 `env.json` 을 다시 생성했다면 P-3 의 레지스트리에 새 항목이
추가되어 있을 것이다 (append-only 라 문제없음).

---

## 4. 나머지 8 건

| # | 항목 | 깨질 수 있는 것 | 검증 |
|---|---|---|---|
| 7 | `pyarrow` 필수화, `pandas` 제거 | 없음 (pandas 는 미사용) | `pip install -e .` 후 `python3 scripts/export.py` |
| 4 | `requirements.lock` 사용 | 개발 환경(conda 3.13)과 lock(3.10) 불일치 가능 | `uv pip compile --python-version 3.10` 재실행해 동일한지 |
| 6 | `KERNELTAB_RESULTS_DIR` / `_ARTIFACT_DIR` | **경로가 바뀌면 기존 산출물을 못 찾는다.** 기본값을 반드시 현재 경로로 | 환경변수 없이 실행 → 기존 파일 그대로 읽히는지 |
| 5 | `nvidia-smi -i` UUID (P-2 와 함께) | 위 3 절 참조 | 위 3 절 (a)(d) |
| 9 | 텔레메트리 `Popen` 실패 감지 | 없음 (경고만 추가) | `PATH= python3 ...` 로 nvidia-smi 를 숨기고 경고가 뜨는지 |
| 8 | `env["manifest"]` 추가 | **`env_hash` 가 바뀐다** → 반드시 P-3 이후에, 그리고 `manifest_hash` 를 해시 키에 넣기로 했다면 동시에 | P-3 검증 (b) 재실행 |
| 10 | Python 3.10 실검증 | ctypes 구조체 정렬 등 런타임 차이 | 3.10 컨테이너에서 `verify smem` + `verify splitk` |
| 11 | CUTLASS `.git` 없을 때 커밋 주입 | 없음 | `--cutlass-commit` 인자로 주고 env.json 에 기록되는지 |

**8 번 주의**: `env["manifest"]` 추가는 `env_hash` 를 바꾼다. P-3 에서
`manifest_hash` 를 해시 키에 포함하기로 했으므로 **P-3 과 같은 커밋에서
함께** 처리해야 한다. 따로 하면 해시가 두 번 바뀐다.

---

## 5. 전체 완료 후 최종 확인

```bash
python3 scripts/validate_table.py            # 표 무결성 (종료 코드 0)
python3 scripts/report_phase3.py             # 리포트 재생성
python3 scripts/export.py                    # parquet 재생성
git log --oneline -12                        # 단계별 커밋이 남아 있는지
```

그리고 **컨테이너 검증**으로 넘어간다 (`docker/Dockerfile.draft` 의
"빌드하기 전에 반드시 할 일" 1~3 번이 이 시점에 전부 충족된다).

## 6. 하지 말 것 (재확인)

`docs/portability_audit.md` 의 "T. 고치면 안 되는 것" 8 항목은 이 계획에
포함되지 않는다. 마이그레이션 중에 "이것도 김에 고치자" 는 유혹이 생기므로
그 문서를 먼저 다시 읽을 것.
