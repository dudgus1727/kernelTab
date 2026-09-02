# 소비 인터페이스 계약 — `kernelrule` 이 이 표를 쓰는 방법

`kerneltab` 은 (형상 × config) → 성능 표를 만드는 **측정 하네스**다. 그 표로
config 선택 규칙을 학습·평가하는 것은 별도 프로젝트(`kernelrule`)의 일이다.

이 문서는 그 경계에서 지켜야 할 것을 정한다. 그리고 **문서가 아니라 코드로**
강제한다 — 문서는 지켜지지 않는다.

---

## 1. 가장 중요한 것: 정답을 규칙에 보여주지 마라

이 표에는 "어떤 config 가 빠른가" 의 답이 들어 있다. 규칙 함수가 그것을
입력으로 받으면 규칙은 아무것도 배우지 않고 정답을 베낀다. 그러면
* 학습된 규칙이 실제 성능이 좋아 보이지만
* 측정하지 않은 새 형상에서는 아무 쓸모가 없다.

측정 하네스 쪽은 이 문제를 **자료구조 수준에서** 막아 두었다 —
`Problem` / `KernelConfig` / `Hardware` / `RuntimeConfig` 어디에도 측정된
시간을 넣지 않는다. 하지만 `table.parquet` 은 평면 테이블이라 그 보호가
없다.

## 2. 설치

```bash
cd ../kernelrule
pip install -e ../kernelTab        # editable 로 설치한다
```

**editable 이어야 한다.** `hwspec/` (GPU 스펙 데이터)와 `artifacts/`
(커널 `.so`)는 **패키지 밖**에 있다 — 7.4 GB 산출물이 패키지 안에 있으면
`pip` 이 그것까지 가져가려 하고, 컨테이너 볼륨 마운트 지점으로도
부적절하기 때문이다. 비-editable 로 설치하려면 `KERNELTAB_HWSPEC_DIR` 로
위치를 알려야 한다 (없으면 `HardwareDetectionError` 가 경로를 찍고 죽는다).

> ⚠️ **최상위 이름은 `kerneltab` 하나다.** 예전에는 `core` / `build` /
> `measure` 가 최상위였다. `build` 는 PyPI 에 실제로 존재하는
> 패키지(`python -m build`)이고 나머지도 흔한 이름이라, 충돌하면
> `ImportError` 가 아니라 **다른 모듈이 조용히 import 된다.**

## 3. 두 개의 로더만 쓴다

```python
from kerneltab.core.table import load_for_ranking, load_for_scoring

X = load_for_ranking("results/table.parquet", env_hash="368a84f1")
y = load_for_scoring("results/table.parquet", env_hash="368a84f1")
```

| 함수 | 무엇을 주는가 | 누가 보는가 |
|---|---|---|
| `load_for_ranking()` | **정답이 제거된** 피처 (72 열) | 규칙 함수 |
| `load_for_scoring()` | 조인 키 + 정답 (18 열) | 채점기만 |

`load_for_scoring()` 이 피처를 함께 돌려주지 않는 것은 의도다. 반환값을
그대로 규칙에 넘기는 사고를 막는다. 두 결과는
`(kernel_id, M, N, K, split_k, split_k_mode)` 로 조인한다.

### 제거되는 컬럼

**`ANSWER_COLS` — 정답.** 측정 결과이거나 거기서 직접 유도된 값.
```
time_ms  time_std_ms  time_min_ms  time_max_ms  n_reps  outlier_frac
cublas_ms  tflops  frac_of_peak  vs_cublas
```

**`OUTCOME_COLS` — 측정을 해봐야 아는 값.** 정답은 아니지만 규칙이 쓰면
"돌려보고 고른다" 가 된다.
```
status  error  max_rel_error  actual_split_k
sm_clock_mhz  mem_clock_mhz  gpu_temp_c  power_w  timestamp
```

`actual_split_k` 를 여기 넣은 이유: CUTLASS 가 런타임에 정하는 실제 슬라이스
수라 **`Params::init_grid_tiled_shape()` 를 재현하지 않으면 미리 알 수 없다.**
그 재현은 `backend.effective_split_k()` 로 가능하므로, 필요하면 규칙이
그것을 **직접 계산**해서 쓰면 된다. 측정 결과를 그대로 받는 것과는 다르다.

### 남는 것 — 써도 되는 것

* 형상: `M`, `N`, `K`, `dtype`, `layout_*`
* config: `tile_*`, `align_*`, `ext_*`, `split_k`, `split_k_mode`
* **빌드 시점에 알 수 있는 커널 속성**: `regs_per_thread`, `threads`,
  `smem_dynamic`, `spill_*`, `hmma_count`, `max_blocks_per_sm`,
  `pipeline_kind`, `theoretical_occupancy`, `launchable`,
  `smem_matches`, `hmma_matches`
* 파생 물리 피처: `waves`, `waves_occ`, `tail_waste`, `mainloop_iters`,
  `arith_intensity`, `ridge_point`, `is_memory_bound`, `tail_*_frac`

정적 분석 값(`regs_per_thread`, `spill_*`, `hmma_count` …)을 허용하는 이유는
**커널을 빌드하기만 하면 알 수 있고 실행할 필요가 없기 때문이다.** 실무에서
규칙을 쓸 때도 같은 정보를 얻을 수 있다.

## 4. cuBLAS 는 **행이 아니라 컬럼**이다

`results.jsonl` 에는 두 종류가 섞여 있다.

| 종류 | 무엇 | 개수 (A6000/13.3) |
|---|---|---|
| 측정 | (형상, 커널, 런타임) 조합 | 980,915 |
| **참조** | 형상별 cuBLAS 시간 | 11,536 |

**`table.parquet` 에는 참조 줄이 없다.** `export.py` 가 그것을 각 행의
`cublas_ms` **컬럼**으로 옮긴다. 확인:

```python
(df.kernel_id == "cublas").sum()   # 0 이어야 한다
```

그러니 **후보 열거에서 따로 걸러낼 필요가 없다.** 다만 `results.jsonl` 을
직접 읽는다면 반드시 걸러라 — 합쳐 세면 진행률이 **101.2 %** 가 된다
(실제로 그랬다).

```python
from kerneltab.core import records
for r in records.iter_records(path, env_hash):
    if records.is_reference(r):     # kernel_id 문자열을 직접 비교하지 마라
        continue
```

`is_reference()` 가 유일한 판정 지점이다. 2026-08-21 이후 기록되는 줄에는
`record_kind: "reference"` 가 있고, 그 이전 줄은 `kernel_id` 로 판정한다 —
그 두 경로를 이 함수 하나만 안다.

## 5. `schema_version` — 번들이 무엇을 들고 있는가

| 버전 | 무엇이 들어 있는가 |
|---|---|
| 1 | `noise_floor` 가 통계 모델만 (`sigma_abs_ms`, `sigma_rel`) |
| **2** | `noise_floor.tick_ms` (타이머 분해능) + 표에 `distinct_time_frac` |
| **3** | `aggregate_status` — 형상별 집계가 `ok` 만이 아니라 **유효 측정 전부** |

```python
b = load_bundle(...)
b.schema_version      # 2
b.tick_ms             # 0.001024 — 없으면 경고하고 A6000 관측치로 대체
b.coef                # NoiseCoef — answer_set 에 넘길 계수 (§5.1)
b.noise_floor(t)      # max(통계, 분해능)
b.aggregate_status    # "all" | "ok" — 없으면 "ok" (schema <= 2)
b.info["table_columns"]   # 표에 **실제로** 있는 컬럼
```

> 공개된 `c63710df` / `828baa64` 번들은 **`aggregate_status = "ok"`** 다.
> `difficulty` 와 `distinct_time_frac` 을 그 두 표와 **다음 캠페인 표
> 사이에서 비교하려면** 이 값을 먼저 봐라. 실측 차이는 `difficulty` 중앙
> +0.02 %, `distinct_time_frac` 중앙 −1.49 pp 다.

> 버전은 "무엇이 들어갈 수 있는가" 이고 `table_columns` 는 "실제로 무엇이
> 들어 있는가" 다. 같은 버전이라도 `export` 시점이 다를 수 있으므로
> **후자를 봐라.**

### 새 컬럼 — `distinct_time_frac`

형상별 **서로 다른 시간값 / 후보 수**. 난이도와 **다른 축**이다.

| | 무엇을 말하는가 |
|---|---|
| 난이도 낮음 | 실제로 성능이 비슷하다 (**물리**) |
| `distinct_time_frac` 낮음 | 측정이 구분을 못 한다 (**계측**) |

`ANSWER_COLS` 이므로 `load_for_ranking()` 에는 안 나온다 —
`load_for_scoring()` 에서 채점을 층화·가중할 때 쓴다. `difficulty` 와 같다.

### 5.1 `answer_set()` 은 계수를 **주입받는다**

```python
from kerneltab.core.table import answer_set

b = load_bundle("rtx-a6000-sm_86-828baa64")
ans = answer_set(shape_rows, noise=b.coef)   # ✅
ans = b.answer_set(shape_rows)               # ✅ 같은 것, 잊을 수 없는 쪽
answer_set(shape_rows)                       # ⛔ NoiseCoefRequired
```

**왜 기본값을 없앴나.** 예전에는 `core/noise.py` 의 모듈 전역
`noise_floor(t)` 를 불렀다. 그 계수는 **A6000 에서 잰 값**이라, 4090/H100
번들을 채점할 때도 A6000 눈금을 썼다 — 게다가 `Bundle.tick_ms` 를 안
거치니 **그 경고조차 안 났다.** 눈금이 2 배인 GPU 에서 정답 집합이
1.5~6.2 배 어긋난다.

기본값이 조용히 틀리는 것보다 **명시를 요구하는 쪽**이 안전하다
(`decisions.md` 15 의 원칙). 노이즈와 무관한 고정 허용치를 쓰려면
`answer_set(rows, tol=0.01)` 로 근거를 남기고 쓴다.

## 6. 안전장치

로더를 우회해 직접 DataFrame 을 만들었다면, 규칙에 넘기기 전에 확인한다.

```python
from kerneltab.core.table import assert_no_answers, AnswerLeakError

assert_no_answers(X, where="rank_configs() 입력")   # 섞였으면 AnswerLeakError
```

`kernelrule` 의 단위 테스트에 이것을 넣어 둘 것을 권한다.

```python
def test_no_answer_leak():
    X = load_for_ranking(TABLE, env_hash=ENV)
    assert_no_answers(X)
    # 규칙이 실제로 쓰는 컬럼만 남았는지도 함께 고정해 둔다
    assert "time_ms" not in X.columns
```

## 7. `env_hash` 는 반드시 지정한다

```python
X = load_for_ranking(path, env_hash="368a84f1")
```

지정하지 않았는데 표에 여러 조건이 섞여 있으면 **로더가 예외를 던진다.**
조용히 섞이는 것보다 낫다.

조건이 다르면 절대 시간을 비교할 수 없다. 이 저장소의 데이터만 해도
클럭 미고정(`b42df475`) / SM 클럭만 고정(`1f0b6924`) / 프로토콜 변경
(`dda3431a`) / SM+메모리 고정(`368a84f1`) 이 섞여 있다.

> ⛔ `368a84f1` 의 226,211 행은 **측정 드리프트로 폐기**됐다
> (`docs/measurement_drift.md`). 이 문서의 예시에 그 값이 남아 있는 것은
> 형식을 보이기 위해서다. 실제로는 재측정본의 `env_hash` 를 쓴다.

## 8. `ext_*` 와 아키텍처 전이

`ext_*` 는 아키텍처 전용 필드다 (SM80: `ext_warp_m/n/k`, `ext_stages`,
`ext_swizzle_*`). SM90 데이터를 같은 표에 합치면 `ext_cluster_m` 같은 컬럼이
생기고 **SM80 행에서는 null** 이 된다.

따라서:
* **아키텍처 전이를 노리는 규칙은 `ext_*` 를 쓰지 않는다.** 공통 필드
  (`tile_m/n/k`) 와 파생 물리 피처만으로도 `waves`, `tail_waste`,
  `mainloop_iters` 가 계산되도록 설계되어 있다.
* 아키텍처 특화 규칙이면 `ext_*` 를 쓰되, 그 규칙은 다른 아키텍처에
  적용할 수 없다는 것을 명시할 것.

## 9. `status != "ok"` 는 결측이 아니다

로더는 기본으로 `status == "ok"` 만 남긴다 (`ok_only=True`). 하지만 실패
줄은 **버려진 것이 아니라 명시적으로 기록된 것**이다.

> ⚠️ **이 계약을 우리 코드가 어기고 있었다.** `baseline_*.py` 가
> `ok` 만 남기는 바람에 "61 형상 전부에서 측정된 config" 가 3,465 개가
> 아니라 **3 개**로 줄었고, 정적 top-1 이 1.115 인데 **1.394** 로 공개
> 문서에 실렸다 (`baselines.md` 정정 이력 참고).
>
> **문서가 옳고 코드가 틀렸는데, 문서를 믿었으니 아무도 안 봤다.**
> 지금은 `--status {ok,all}` (기본 `all`) 이고,
> `tests/test_contracts.py` 가 (a) 기본값이 `all` 인지, (b) 새로 생기는
> `status` 비교마다 `# status-filter: <이유>` 표시가 붙는지를 AST 로
> 검사한다. 계약을 적었으면 **그 계약을 검사하는 테스트**를 함께 쓴다.

| status | 의미 | 규칙 관점 |
|---|---|---|
| `launch_infeasible` | `regs × threads > regs_per_sm` — 런치 불가 | **빌드 시점에 알 수 있다.** 규칙이 이런 config 를 고르면 안 된다 |
| `numerical_fail` | 계산 결과가 틀림 | 열거기가 이미 제외한다 |
| `below_launch_overhead` | 측정값이 런치 오버헤드 수준 | 그 형상에서는 시간 비교가 무의미 |
| `high_outlier_frac` | IQR 밖 표본 20% 초과 | 시간값을 믿기 어렵다 |
| `runtime_fail` / `oom` | 실행 실패 | |

`launch_infeasible` 은 **규칙이 피해야 할 것을 배우는 데 쓸 수 있다** —
`launchable` 컬럼이 그 정보를 정답 없이 제공한다.

## 10. 이 계약을 어기면 생기는 일

가장 흔한 사고는 이것이다.

```python
df = pd.read_parquet("table.parquet")          # ← 로더를 안 씀
best = df.loc[df.groupby(["M","N","K"]).time_ms.idxmin()]   # ← 정답으로 라벨 생성
model.fit(df.drop(columns=["time_ms"]), ...)   # ← time_ms 만 뺐다고 안심
```

`time_ms` 만 빼도 `tflops` / `frac_of_peak` / `vs_cublas` 가 남아 있다. 셋 다
`time_ms` 에서 유도된 값이라 **완전한 정답**이다. `load_for_ranking()` 은
이것들을 함께 제거한다.

---

## 부록 — 최소 예제

```python
import pandas as pd
from kerneltab.core.table import load_for_ranking, load_for_scoring

ENV = "368a84f1"
TABLE = "results/table.parquet"

X = load_for_ranking(TABLE, env_hash=ENV)   # 규칙 입력
y = load_for_scoring(TABLE, env_hash=ENV)   # 채점용

KEYS = ["kernel_id", "M", "N", "K", "split_k", "split_k_mode"]

def my_rule(group: pd.DataFrame) -> pd.Series:
    """한 형상의 후보들 중 하나를 고른다. group 에는 정답이 없다."""
    return group.sort_values(["waves", "tail_waste"]).iloc[0]

picked = X.groupby(["M", "N", "K"], group_keys=False).apply(my_rule)

# 채점은 조인해서 한다
scored = picked[KEYS].merge(y, on=KEYS, how="left")
oracle = y.groupby(["M", "N", "K"]).time_ms.min().rename("oracle_ms")
regret = scored.set_index(["M", "N", "K"]).time_ms / oracle
print(regret.describe())
```


---

## 11. RTX 5090 표에서 새로 알아야 할 것 (2026-09-01)

### `status == "ok"` 로 거르면 **A6000 보다 두 배 더 버린다**

| | A6000 | **RTX 5090** |
|---|---:|---:|
| `high_outlier_frac` | 10.65 % | **22.21 %** |

`outlier_frac` 중앙값이 0.273 으로 `OUTLIER_TOL = 0.20` 을 넘는다. 원인은
**짧은 커널에 반복이 많다는 것**이다 — 시간 예산(`target_ms = 20`)을 채우려면
0.2 ms 커널에 100 회 넘게 돌게 되고, 그만큼 IQR 밖으로 밀리는 표본이 늘어난다
(`ok` 행 중앙 69 회 vs `high_outlier_frac` 행 중앙 103 회).

**9 절 그대로다 — 이것은 결측이 아니라 품질 표시이고 `time_ms` 는 유효하다.**
다만 두 표를 나란히 놓고 `status == "ok"` 로 거르면 **5090 쪽 덮개가 얇아
보인다.** 그것은 GPU 차이가 아니라 필터가 만든 인공물이다.

```python
# 권장 — 유효한 시간이 있는 행을 전부 쓴다
df = df[df.time_ms.notna() & (df.status != "numerical_fail")]

# 굳이 걸러야 하면 두 표에 **같은 기준**을 쓰고 그 사실을 적어라
```

### 깊은 클럭 딥 41 행 — `sm_clock_mhz` 로 걸러낼 수 있다

22.5 시간 중 네 번, 1~2 초씩 SM 클럭이 고정값(2392 MHz) 아래로 크게 내려갔다
(2370 / 2235 / 2212 / 2107 / 2055 MHz). **1,267,313 행 중 41 행 = 0.0032 %** 다.

```python
df = df[df.sm_clock_mhz >= 2377]      # 고정값의 -0.63 % 이내
```

전력(316~392 W, 상한 600 W)도 온도(58~62 ℃)도 원인이 아니었다. 지속 없이
튀었다 돌아오는 순간 이벤트다. **측정 줄마다 `sm_clock_mhz` 가 기록되므로
사후에 걸러낼 수 있다** — 이 사실을 모르면 그 행들이 조용히 섞인다.

> 나머지 편차는 지원 클럭 **한 칸**(2385 MHz, −0.29 %)이고 전체의 20.8 % 다.
> 부스트 알고리즘의 입도라 제거할 수 없고, 그 구간의 전력이 오히려 낮다.

### 노이즈 계수는 **번들에서 읽어라**

```python
b = load_bundle("rtx-5090-sm_120-5bb6f403")
X = b.ranking();  y = b.scoring()
ans = b.answer_set(df_one_shape)      # b.coef 를 알아서 넘긴다
```

`core.noise` 의 모듈 상수(`A6000_MEASURED`)를 쓰면 **틀린다.** 두 조건의
차이가 짧은 형상에서 결정적이다:

```
      t        A6000       5090      허용치 배수
  0.083ms     2.467%      0.172%      14.4x
  0.110ms     1.862%      0.167%      11.2x
  2.900ms     0.114%      0.153%       0.7x
```

실제 표에서 A6000 계수로 채점하면 `8x4096x4096` 의 정답 집합이 **1 개 대신
1,477 개**가 된다. 순위 정보가 통째로 사라진다.


---

## 12. 이 표는 **CUTLASS 2.x 공간 안의** 최적이다

### 설계 선택

세 캠페인(A6000 x2, RTX 5090)을 **CUTLASS 2.x API 로 통일**했다. 세대 간
전이를 재려면 config 공간이 같아야 하는데, 3.x 는 SM90 이상 전용이라
Ampere 와 **공통 공간이 없다**. 2.x 는 sm_80 부터 sm_120 까지 같은
파라미터(threadblock/warp tile, stages, swizzle, split-K)로 표현된다.

### 대가

**Hopper/Blackwell 의 3.x 전용 기능이 이 공간에 없다:**

```
TMA (Tensor Memory Accelerator)
cluster_shape (thread block cluster)
warp specialization / 스케줄 종류 (cooperative, pingpong)
tile scheduler (persistent, stream-K)
```

따라서 **"이 config 가 최적" 은 2.x 안에서의 최적이다.** 실무에서 그 GPU 를
쓸 때 3.x 커널이 더 빠를 수 있다 — 특히 Hopper 이상에서.

### 소비하는 쪽에 대한 함의

**이 표로 학습한 규칙은 2.x 필드로만 계산된다.** TMA 나 cluster 같은 물리는
표에 **없으므로 표현되지 않는다.** 규칙이 그것을 배우지 못하는 것이 아니라
**애초에 입력에 없다.**

> ⛔ "이 표로 GPU 물리를 다 안다" 로 읽지 마라. 이 표가 답하는 질문은
> **"2.x config 공간 안에서 무엇을 고를 것인가"** 이지 "이 GPU 에서 가장
> 빠른 GEMM 은 무엇인가" 가 아니다.

### 벤더 비교도 같은 공간이다

`nvidia-matmul-heuristics` 는 `target=CUTLASS` 에서 **2.x 파라미터만 낸다**
(전 형상 확인: 5090 612 건 / A6000 594 건 모두 `cluster(1,1)`,
`instr(16,8,16)`). 라이브러리에 `CUTLASS3` 타겟이 따로 있고 그쪽만 cluster 와
wgmma 를 낸다. 자세한 것은 `docs/baselines.md`.

**따라서 벤더 regret 은 우리와 같은 공간 안에서 계산된다** — 3.x 커널로
오염되지 않는다. `scripts/check_axis_coverage.py` 가 `cluster != (1,1)` 또는
`instr != (16,8,16)` 인 추천을 만나면 경고한다.

> 3.x 로 재측정하는 것은 **별도 연구의 자리**다. 이번 캠페인 계열이 아니다 —
> 그렇게 하면 A6000·4090 과의 비교가 깨진다.

### ★ H100 부터는 "2.x 공간의 최적" 이 **GPU 최고 성능이 아니다** (2026-09-02)

A6000 / 4090 / 5090 에서는 `mma.sync` 가 곧 하드웨어 피크였다 — 실측으로
확인했고, 그래서 이 구분이 필요 없었다. **Hopper 에서 처음 갈린다.**

```
mma.sync (= 이 표의 2.x 공간)   623.9 TFLOP/s @1785 MHz   ★ 이 표의 피크
cuBLAS COMPUTE_32F (wgmma)     ~806                      (4096^3 실측 717 @~1590)
데이터시트 dense               ~835
```

Hopper 의 최고 처리량은 **wgmma(warpgroup async, 3.x)** 로만 나온다.
`mma.sync` 는 서브파티션당 HMMA.16816 을 6 사이클에 하나가 상한이다
(4 x 2048 / 6 = 1365.3 MAC/clk/SM — 실측 1365.1 과 일치).

**즉 이 표의 최적은 그 GPU 최고 성능의 약 75 % 다.**

`known.json` 의 `peak_tflops_f16` 에는 **mma.sync 값(623.9)** 을 넣었다.
`frac_of_peak` 의 분모, SOL 하한, ridge point 셋 다 "우리 공간에서 도달
가능한가" 를 묻는 값이기 때문이다. 데이터시트(835)를 분모로 쓰면
`frac_of_peak` 가 구조적으로 75 % 를 못 넘고, 그것이 "커널이 나쁘다" 로
읽힌다 — 실제로는 API 계열이 못 닿는 것이다.

```
ridge point   623.9 / 4022.8 = 155.1 FLOP/byte     ★ 이 표가 쓰는 값
              (데이터시트로 쓰면 207.6 이 되어 연산 바운드 형상을
               메모리 바운드로 분류한다)
```

> ⛔ **`frac_of_peak` 를 GPU 사양 대비 효율로 인용하지 마라.** 이 표의
> `frac_of_peak` 는 **2.x 공간 안에서의 효율**이다. H100 의 데이터시트
> 대비 값을 원하면 0.747 을 곱하라 (623.9 / 835).
