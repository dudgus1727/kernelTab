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

## 4. 안전장치

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

## 5. `env_hash` 는 반드시 지정한다

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

## 6. `ext_*` 와 아키텍처 전이

`ext_*` 는 아키텍처 전용 필드다 (SM80: `ext_warp_m/n/k`, `ext_stages`,
`ext_swizzle_*`). SM90 데이터를 같은 표에 합치면 `ext_cluster_m` 같은 컬럼이
생기고 **SM80 행에서는 null** 이 된다.

따라서:
* **아키텍처 전이를 노리는 규칙은 `ext_*` 를 쓰지 않는다.** 공통 필드
  (`tile_m/n/k`) 와 파생 물리 피처만으로도 `waves`, `tail_waste`,
  `mainloop_iters` 가 계산되도록 설계되어 있다.
* 아키텍처 특화 규칙이면 `ext_*` 를 쓰되, 그 규칙은 다른 아키텍처에
  적용할 수 없다는 것을 명시할 것.

## 7. `status != "ok"` 는 결측이 아니다

로더는 기본으로 `status == "ok"` 만 남긴다 (`ok_only=True`). 하지만 실패
줄은 **버려진 것이 아니라 명시적으로 기록된 것**이다.

| status | 의미 | 규칙 관점 |
|---|---|---|
| `launch_infeasible` | `regs × threads > regs_per_sm` — 런치 불가 | **빌드 시점에 알 수 있다.** 규칙이 이런 config 를 고르면 안 된다 |
| `numerical_fail` | 계산 결과가 틀림 | 열거기가 이미 제외한다 |
| `below_launch_overhead` | 측정값이 런치 오버헤드 수준 | 그 형상에서는 시간 비교가 무의미 |
| `high_outlier_frac` | IQR 밖 표본 20% 초과 | 시간값을 믿기 어렵다 |
| `runtime_fail` / `oom` | 실행 실패 | |

`launch_infeasible` 은 **규칙이 피해야 할 것을 배우는 데 쓸 수 있다** —
`launchable` 컬럼이 그 정보를 정답 없이 제공한다.

## 8. 이 계약을 어기면 생기는 일

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
