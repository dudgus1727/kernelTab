# 스펙 대비 변경한 것

원 스펙에서 벗어난 결정과 **그 근거**. 각 항목은 "무엇을 바꿨나" 가 아니라
"바꾸지 않았으면 무엇이 틀렸을 것인가" 를 적는다. 근거가 없으면 그것은
결정이 아니라 취향이다.

되돌리기 전에 반드시 읽을 것 — 대부분은 CUTLASS 소스나 실측으로 확인한
것이며, 되돌리면 조용히 틀린 숫자가 나온다.

설계 원칙 자체는 README 의 "설계 결정과 그 이유" 에 있다.

---


원래 지시와 다르게 구현한 부분과 근거. 되돌릴 수 있게 남긴다.

## 1. `EpilogueOutputOp` 의 ScaleType: `NoBetaScaling` → `Default` (+ 런타임 beta=0)

지시는 `ScaleType::NoBetaScaling` 이었고 이유는 "beta=0 이므로 C 를 읽지 않아
더 빠르다" 였다. 그런데 CUTLASS 소스에서 그 enum 은 정반대로 동작한다
(`epilogue/thread/scale_type.h`, `linear_combination.h`):

| ScaleType | 계산식 | `is_source_needed()` |
|---|---|---|
| `Default` | `D = α·Acc + β·C` | `β != 0` |
| `NoBetaScaling` | `D = α·Acc + C` | **항상 true** (C 를 읽는다) |
| `OnlyAlphaScaling` | `D = α·Acc` | **항상 false** |

* `NoBetaScaling` 은 partition 0 에서도 C 를 읽으므로 원래 목표에 반한다.
* `OnlyAlphaScaling` 은 source 를 영원히 읽지 않는다. serial split-K 는
  partition > 0 에서 `set_k_partition()` 이 β=1 로 바꿔 이전 부분합을 다시
  읽어 누적하는데, 이 경로가 죽어서 **결과가 조용히 틀린다.**
* `Default` + 런타임 β=0 이면 partition 0 에서 `is_source_needed()==false` 라
  C 를 읽지 않고(목표 달성), partition > 0 에서는 β=1 이 되어 serial split-K
  가 정상 동작한다.

즉 지시의 *의도*(β=0, C 안 읽기)는 그대로 지키고 enum 만 바꿨다.

`LinearCombination` 의 `Count` 인자는 `align_c` 로 준다. 이것이 C/D 의 벡터
접근 폭을 결정하므로 alignment 가 8 미만인 형상에서 8 로 두면 잘못된 메모리
접근이 된다.

## 2. `is_valid_runtime` 규칙 1: `K % (split_k · tile_k) == 0` → 실효 split_k 일치

원래 규칙을 그대로 쓰면 지시의 다른 두 요구가 **동시에 불가능해진다.**

* 그리드의 어떤 K 도 3 으로 나누어떨어지지 않는다 (4096, 11008, 2의 거듭제곱,
  512, 4100, 4098, 4097). 따라서 `split_k ∈ {3, 6, 12}` 는 **단 한 번도 측정될
  수 없다** — 명시적으로 검증하라고 한 가설이 검증 불가능해진다.
* 층 D 와 리허설 형상 `(1024, 4096, 4100)` 은 `4100 % 32 ≠ 0` 이라 split_k=1
  조차 통과하지 못해 **유효 조합이 0 개**가 된다. alignment=4 경로를 검증할 수
  없다.

CUTLASS 는 애초에 K 가 나누어떨어지지 않아도 알아서 자른다
(`kernel/params_universal_base.h : init_grid_tiled_shape()`):

```
kAlignK   = max(8, (A row-major or B col-major) and K % 64 == 0 ? 64 : 1)
gemm_k_size          = round_up(ceil_div(K, split_k), kAlignK)
grid_tiled_shape.k() = ceil_div(K, gemm_k_size)      <- 실제 슬라이스 수
```

진짜 위험은 나누어떨어지지 않는 것이 아니라 **요청한 split_k 와 실제 슬라이스
수가 다른 것**이다. 예를 들어 K=512 에 split_k=12 를 요청하면 실제로는 8 개가
만들어져, 8 과 12 라는 다른 이름으로 같은 것을 두 번 재게 된다. 그래서 규칙을
이렇게 바꿨다.

```python
rc.split_k * cfg.tile_k <= p.K            # 슬라이스가 최소 K 타일 하나는 담아야
effective_split_k(p, rc) == rc.split_k    # 요청 == CUTLASS 가 실제로 만드는 수
```

지시가 든 예시 `(1024, 4096, 512)` 에서 이 조건이 실제로 거르는 것은 그대로다
(split_k=6/12/16 탈락). `(1024, 4096, 4100)` 은 15 개 런타임 config 를 얻는다.

## 3. 탐색 축 확대: `warp_count ∈ {4,8}` → `{4,8,16}`, 누산기 상한 200 → 256

원래 값은 추정치였다. 레지스터 실사용량과 스필은 빌드 시 `-Xptxas -v` 로
**실측**해 `kernels.jsonl` 에 남기므로, 추정치로 미리 자르는 것보다 빌드해서
기록하는 편이 데이터로서 낫다. "성능 기준으로 거르지 마라" 원칙과도 일관된다.

부수적으로 `threads_per_block <= min(1024, hw.max_threads_per_sm)` 가드를
추가했다. warp_k 분할(PartitionsK)까지 곱하면 블록 스레드 수가 CUDA 한도를
넘을 수 있는데, 이건 런치 자체가 불가능한 조합이다.

결과: alignment 조합당 1,155 → 1,575 개.

## 4. `is_valid_kernel` 에 CUTLASS 컴파일 제약 2 개 추가

빌드해 보면 특정 조합이 `static_assert` 로 거부된다. **성능이 아니라 컴파일
가능성** 문제이므로 `is_valid_kernel` 의 본래 역할에 해당한다. 두 제약 모두
CUTLASS 소스에서 유도한 뒤 실측 1,263 개 빌드와 대조해 **오탐 0 / 미탐 0**
을 확인하고 넣었다 (`scripts/validate_constraints.py`).

**`mainloop_smem_thread_map`** — `transform/pitch_linear_thread_map.h`
```
tile_m · tile_k >= 8 · threads   and   tile_n · tile_k >= 8 · threads
```
A/B 모두 smem 에서 crosswise 레이아웃이고 smem 접근 폭은 전역 alignment 와
무관하게 128 비트 고정이라, 조건이 이렇게 단순해진다.

**`epilogue_thread_map`** — `output_tile_thread_map.h` / `predicated_tile_iterator.h`
`RowArrangement` 가 `8 > warps_n · warps_k` 로 갈리고 각 분기에서
`Iterations::kRow / kColumn` 이 0 이 되면 거부된다.

결과: alignment 조합당 1,575 → 1,365 개 (13.3% 제거). 통째로 죽는 축은 없다.

## 5. horizontal 스위즐을 직접 구현 (`measure/kt_swizzle.h`)

CUTLASS 의 `GemmHorizontalThreadblockSwizzle` 은 `get_tile_offset(GemmCoord)`
시그니처라 `get_tile_offset(int log_tile)` 로 호출하는 `kernel::GemmUniversal`
과 **컴파일되지 않는다** (구버전 `device::Gemm` 경로 전용). 그냥 빼면 커널의
20% 가 사라지고, 무엇보다 **래스터 방향 축 자체가 사라진다.** 이 축은 SM90
3.x API 의 `raster_order = along_m | along_n` 과 직접 대응하므로 여기서 잃으면
나중에 SM80 ↔ SM90 전이 실험이 성립하지 않는다.

직접 짠 코드이므로 별도로 검증했다 (`scripts/verify_swizzle.py`):

| 형상 | 타일 격자 | identity(1) | horizontal |
|---|---|---|---|
| (8192, 2048, 4096) | m=64 ≫ n=16 | 1.621 ms | **1.124 ms** |
| (2048, 8192, 4096) | m=16 ≪ n=64 | **1.119 ms** | 1.590 ms |

* grid dim 이 `(64,16) ↔ (16,64)` 로 실제로 뒤바뀌는 것을 `get_grid_shape()` 로 실측
* 네 형상 모두 `max_rel_error = 0.0` (cuBLAS 와 비트 단위 일치)
* 성능이 **대칭적으로 역전**된다 — 스위즐이 실제로 적용된다는 증거. 단순히
  "값이 다르다" 가 아니라 물리적으로 기대되는 방향으로 다르다.

## 6. split-K 부분합 정밀도 — serial 과 parallel 모두 fp16 (비대칭 아님)

**결론부터: 두 모드의 저장 정밀도는 같다. 비교 조건은 공정하다.**

CUTLASS 2.x `kernel::GemmUniversal` 에서 부분합이 어디에 어떤 타입으로
놓이는지는 이렇다.

| 모드 | 부분합 위치 | 부분합 타입 | 누산 |
|---|---|---|---|
| serial (`kGemm`, k>1) | `D` 자체 | **`ElementC` = fp16** | 파티션마다 fp32 로 더하고 fp16 으로 되씀 |
| parallel (`kGemmSplitKParallel`) | workspace | **`ElementC` = fp16** | 리덕션 커널이 fp32 로 합산 후 1 회 반올림 |

serial 이 fp32 누산기를 이어받는 것이 아니다. `kernel/gemm_universal.h` 에서

```cpp
// For subsequent threadblocks, the source matrix is held in the 'D' tensor.
if (threadblock_tile_offset.k()) {
  iterator_C = iterator_D;      // <- 이전 부분합을 D(=fp16)에서 다시 읽는다
}
```

즉 파티션 사이의 중간값이 **매번 fp16 으로 전역 메모리를 왕복**한다. 반올림
횟수도 두 모드가 비슷하며(각 슬라이스 1 회), 오히려 parallel 쪽이 마지막
합산을 fp32 로 한 번에 하므로 미세하게 유리하다.

**fp32 부분합은 왜 안 쓰는가.** 부분합 타입은 곧 커널의 `ElementC` 다
(`get_workspace_size()` 가 `sizeof(ElementC) * ...` 로 정의되고 epilogue 가
`iterator_D` 로 쓴다). fp32 부분합을 얻으려면 `ElementC = float` 인 커널을
따로 인스턴스화해야 하는데 — CUTLASS 라이브러리가
`find_gemm_operation_for_parallel_reduction()` 으로 하는 일이 정확히 그것이다 —
그러면

* `split_k_mode` 가 런타임 인자가 아니라 **빌드 축**이 되어 커널 수가 2 배가
  되고 KernelConfig / RuntimeConfig 분리의 전제가 깨진다,
* serial 쪽은 부분합이 곧 최종 출력이라 **출력 dtype 자체가 fp32 로 바뀐다.**
  그러면 epilogue 쓰기 트래픽이 2 배가 되어 split-K 를 안 쓰는 측정까지
  전부 오염된다.

그래서 fp16 으로 둔다. 대신 해석에 필요한 정보를 결과에 남긴다:
`results.jsonl` 의 `partials_dtype` (실효 split_k > 1 이면 `"f16"`) 과
`workspace_dtype` (parallel 이면 `"f16"` 부분합 버퍼, serial 이면 `"i32"`
세마포어). 이 두 필드가 없으면 나중에 정확도 분석에서 "방식의 차이" 와
"저장 정밀도의 차이" 를 구분할 수 없다.

측정 시간에는 parallel 의 리덕션 커널을 포함한다 (빼면 serial 과 비교 자체가
성립하지 않는다).

## 7. `launch_infeasible` status 추가

`cutlass::Kernel2` 에는 `__launch_bounds__` 가 없어서 ptxas 가 스레드 수에 맞춰
레지스터를 제한하지 않는다. 그 결과 `regs_per_thread × threads > regs_per_sm`
인 커널이 만들어지는데, 이건 빌드는 되지만 **런치가 불가능**하다
(`cudaOccupancyMaxActiveBlocksPerMultiprocessor` 가 0 을 돌려준다).
빌드 전에는 예측할 수 없고(레지스터는 ptxas 의 결과) 빌드 후에는 정확히
판정되므로, 측정 직전에 확인해 실행하지 않고 이 status 로 기록한다.

## 8. `pipeline_kind` 필드 추가

`stages == 2` 는 `MmaPipelined`(동기 LDG + STS), `stages >= 3` 은
`MmaMultistage`(cp.async / LDGSTS) 로 **구현이 다른 별개 커널**이다.
SASS 에서도 정확히 갈린다 (stages=2 는 LDGSTS 0 개, stages≥3 은 795/795 사용).
`stages` 를 하나의 연속 축으로 보면 잘못된 결론이 나오므로 명시적으로 남긴다.

## 9. warp tile (64,128) 제외 — 계산이 틀린다

`is_valid_kernel` 은 원칙적으로 컴파일/실행 가능성만 본다. 하지만 warp tile
`(64,128)` 은 컴파일도 되고 런치도 되는데 **결과가 틀린다.**

`scripts/check_correctness.py` 로 a888 런치 가능 커널 1,215 개를 전수 검사한
결과:

```
이상 60 / 정상 1,155
이상 커널의 warp tile: {(64,128): 60}   <- 100%, 예외 없음
같은 누산기 수(256 regs/thread)인 mirror (128,64): 60/60 정상
max_rel_error 0.77 ~ 1.13  (완전한 오답)
```

레지스터 수가 원인이 아니라 `warp_n` 이 원인이다. CUTLASS generator 가 만드는
712 개 조합 중 `warp_n >= 128` 은 1 건뿐이고 `(64,128)` 은 0 건 — 검증되지 않은
영역이다. 근본 원인은 더 추적하지 않았다.

성능 필터가 아니라 **정확성** 문제이므로 제외한다. 오답 config 가 성능표에
섞이면 그 표를 쓰는 모든 분석이 오염된다. `(128,64)` 는 정상 동작하므로
(스필로 느리긴 하지만) 남겨둔다 — 느린 것은 데이터로 남길 가치가 있다.

## 10. 측정 반복 수를 고정 하한이 아니라 시간 예산으로 정한다

원래 스펙은 "총 20ms 또는 **최소 30회**" 였다. 그 값은 노이즈가 큰 상황
(스로틀 51%, 재현성 max 8.91%)을 가정한 것이다. 클럭 고정 후에는 2,155 회
연속 측정 변동이 0.11% 라 전제가 달라졌다.

고정 하한을 쓰면 느린 커널에서 최소 반복 수가 시간을 지배한다 —
20ms 커널 × 30회 = 작업당 0.6초. 전수 측정 926,235 건에서 이것만으로
20 시간 이상이 더 든다. 그래서 규칙으로 바꿨다.

```
min_reps = clamp(ceil(min_total_ms / t), min_reps_floor, min_reps_cap)
n_reps   = clamp(target_ms / t,          min_reps,        max_reps)
     target_ms=20, min_total_ms=3, min_reps_floor=5, min_reps_cap=30
```

빠른 커널(≤0.1ms)은 예전과 같은 30회 이상, 느린 커널은 최소 3ms 만 재고
멈춘다. 하한 5 는 IQR 사분위 계산에 필요한 최소 표본이다.

**프로토콜은 `env.json` 의 `protocol` 에 기록되어 `env_hash` 에 반영된다.**
프로토콜이 바뀌면 측정 조건이 바뀐 것이므로 resume 이 예전 줄을 건너뛰면
안 되기 때문이다.

## 11. cp.async 최소 접근 폭 — alignment 1 은 2단만 가능

multistage(stages ≥ 3)는 `cp.async`(LDGSTS)로 전역→smem 복사를 하는데,
cp.async 는 **4/8/16 바이트 접근만** 지원한다. fp16 × alignment 1 = 2 바이트가
정확히 여기 걸려 `static_assert: Size is not supported` 가 난다.

실측: `a118`(K=4097) 에서 stages=2 는 31/31 성공, stages≥3 은 0/140 성공.
`is_valid_kernel` 에 넣었고 실측 2,901 건과 대조해 오탐 0 / 미탐 0 이다.

결과적으로 층 D 의 `K=4097` 형상은 유효 커널이 445 개뿐이다 (다른 alignment 는
1,305~1,435). **alignment 1 은 2단 파이프라인만 쓸 수 있다** 는 것 자체가
이 형상에 대한 결론이다.

## 12. Backend Protocol 에 메서드 추가

명세된 목록 외에 `enumerate_runtime`, `explain_kernel`, `ext_from_dict`,
`pipeline_kind`, `effective_split_k` / `workspace_bytes` 를 두었다. 각각
split-K 축 값 집합의 아키텍처 종속성, 제약 funnel 집계, JSONL 역직렬화,
파이프라인 계열 구분, CUTLASS split-K 의미론 때문이며 전부 백엔드 뒤에 있다.

---

---

---

## 13. 형상/조건 단위 파생 지표는 `env_hash` 별로만 계산한다

`core/table.py` 의 `load_for_ranking` 은 **소비 시점**에 `env_hash` 혼재를
막는다. 그러나 **`export.py` 에는 그 보호가 없다** — `rows` 에는 모든 측정
조건의 줄이 섞여 있고, 파생 지표를 계산하는 곳이 바로 거기다.

조건이 다르면 절대 시간이 비교 불가능하다. 그런데 형상 단위로 중앙값이나
최소값을 내는 순간 그 사실이 사라지고 **아무 오류도 나지 않은 채** 값만
틀린다.

실제로 겪었다. `difficulty = 중앙값 시간 / 최적 시간` 을 `env_hash` 없이
계산했더니 폐기된 드리프트 구간(`368a84f1`, 최대 +1380 % 오염)의 느린 시간이
섞여 **난이도가 22.05 배**까지 나왔다. 조건별로 나눈 뒤에는 1.11~2.60,
중앙 1.58 이다. 20 배 틀린 값이 예외 하나 없이 나왔다.

### 다섯 번 밟고 나서 구조로 옮겼다 (R-5)

문서에 적어 둔 다음에도 두 번 더 밟았다. 전수 조사해 보니 다섯 곳이었다.

| # | 어디 | 증상 |
|---|---|---|
| 1 | `export.py` 의 `difficulty` | 난이도 **22.05 배** |
| 2 | `bundle.py` 의 번들 통계 | 형상 68개(실제 66), 측정 구간이 폐기 구간부터. **공개 릴리즈 노트에 실릴 뻔했다** |
| 3 | `rehearse.py` 의 드리프트 경고 | 변동폭 **55.59%**, 경고 상시 발생 |
| 4 | `rehearse.py` 의 `reproducibility()` | 재현성 기준값을 **다른 조건의 측정치**에서 가져옴 |
| 5 | `recheck_stability.py` | 같은 이유로 재측정 대조가 어긋남 |

그래서 `core/records.py` 하나로 모았다.

```python
def iter_records(path, env_hash): ...       # env_hash 에 기본값이 없다
def aggregate_per_env(rows, key_fn, ...):   # 집계 키에 env_hash 가 강제로 들어간다
ALL = "ALL"                                  # 전체를 보려면 명시해야 한다
```

* `env_hash` 를 안 쓰면 `TypeError`, 빈 문자열이면 `EnvHashError` 다.
  **우회로를 남기지 않는다.**
* 집계는 키에 `env_hash` 가 강제로 들어가므로 섞는 것이 불가능하다.
* `tests/test_records.py` 가 이것을 고정하고, **실제 호출부가 필터를
  쓰는지도** 검사한다 (새로 추가한 사람이 빠뜨리면 거기서 걸린다).

예외는 `validate_table.py` 하나다 — "파일에 조건이 몇 종 섞여 있는가" 를
보고하는 것이 그 검사기의 일이라 의도적으로 전체를 읽는다. 주석에 명시했다.

### 여섯 번째는 성격이 달랐다 — **배포 경계**

앞의 다섯은 전부 **코드 안에서 집계가 오염**된 것이라 `core/records.py` 로
막을 수 있었다. 여섯 번째는 아니었다.

`bundle.py` 가 `results/table.parquet` 을 **파일째 복사**해서, 폐기한 드리프트
데이터 226,145행이 공개 GitHub Release 에 실렸다. 원본은 작업 파일이라 여러
조건을 담는 것이 정상이고(조건 간 비교/드리프트 분석에 쓴다), 문제는 그것이
**걸러지지 않고 경계를 넘어간 것**이다.

`BUNDLE.json` 의 `n_rows` 는 필터한 값이라 980,915 로 **맞았다.** 그래서
아무도 못 알아챘다 — 통계는 맞는데 파일이 틀렸다.

```
읽기 층   iter_records(path, env_hash)     <- 집계 오염을 막는다
배포 층   validate_bundle(dir)             <- 파일이 경계를 넘는 것을 막는다
```

배포 층 검사의 핵심은 **`n_rows` 와 실제 행 수의 일치**다. 그 불일치가
"통계만 필터하고 파일은 안 걸렀다" 의 서명이고, 이 검사 하나면 바로 잡혔다.
`bundle.py` 는 이 검사를 통과하지 못하면 번들을 만들지 않는다.

교훈: **강제 지점은 데이터가 층을 넘는 곳마다 필요하다.** 읽기를 막았다고
쓰기·복사·배포가 안전해지지 않는다.

교훈은 더 넓다 — **`env_hash` 는 조인 키가 아니라 격리 경계다.** 두 조건의
줄이 같은 집계에 들어가는 경로가 생기면, 그 지표는 조용히 틀린다.

---

## 14. 조건이 안 맞으면 조용히 통과하는 안전장치를 금지한다

이 저장소가 **같은 클래스의 버그를 네 번** 만났다.

| # | 사례 | 증상 |
|---|---|---|
| 1 | `WARMUP_SECONDS` | 정의만 되고 안 쓰임. **로그는 "워밍업 한다" 고 찍혔다** |
| 2 | `MEM_CLOCK_MIN_FRAC` | 주석에 판정 기준만 있고 미구현 |
| 3 | `test_table.py` / `test_bundle.py` | pyarrow 가 없으면 41개가 통째로 스킵되는데 **"86 passed" 초록불** |
| 4 | `drift_check()` | 프로브 커널 준비가 실패하면 `return` (None). 호출부는 `if t_drift:` 라 **드리프트 감시가 조용히 멈춘다** |

공통점은 하나다 — **실패했다는 사실이 아무 데도 남지 않는다.** 결과만 보면
정상과 구분이 안 된다. 1번은 33시간 측정에 냉시작 104회를 섞을 뻔했고,
3번은 정답 누출 방지가 검증됐다고 거짓으로 믿게 했다.

### 규칙

**"할 수 없으면 아무것도 안 한다" 는 금지한다.** 셋 중 하나를 골라야 한다.

1. **실패시킨다** — 검증/테스트 경로의 기본값. 조용한 통과보다 시끄러운
   실패가 낫다.
2. **크게 남긴다** — 측정 루프처럼 멈추면 더 곤란한 경우. 경고를 찍고
   **기록에 플래그를 남겨** 사후에 걸러낼 수 있게 한다
   (`soak_info.warmup_mem_clock_ok` 가 그 예다).
3. **명시적으로 우회시킨다** — 환경변수 등으로 의도를 드러내게 하고,
   우회했다는 사실이 출력에 남게 한다 (`KERNELTAB_ALLOW_SKIP=1`).

### 강제 방법

문서로 적어두는 것으로는 부족하다 — 1번을 고치고 나서 2번을, 그 다음
3번을 밟았다. **구조로 막는다.**

* `tests/conftest.py` 가 중요 모듈이 실제로 돌았는지 **합성 테스트 항목**으로
  검사한다. `pytest_sessionfinish` 에서 `session.exitstatus` 를 바꾸는 방법은
  pytest 버전에 따라 전파되지 않는다(실제로 메시지만 나오고 exit 0 이었다).
  진짜 테스트 항목이어야 종료 코드로 이어진다.
* `tests/test_skip_guard.py` 가 **그 감시가 실제로 실패를 잡는지** 확인한다.
  감시에 감시가 없으면 감시 자체가 이 병에 걸린다.
* AST 전수 검사로 남은 사례를 찾는다 (`except: pass`, 검증 함수의 조용한
  `return`). 현재 17건 + 1건이 남아 있고 `docs/pending_fixes.md` R-4(b) 에서
  하나씩 판정한다.

---

## 15. 메타 — 문서로 적은 규율은 지켜지지 않는다

이 저장소에서 가장 여러 번 확인된 사실이다.

| 규율 | 어디에 적었나 | 그 뒤 몇 번 더 밟았나 |
|---|---|---|
| "조건이 안 맞으면 조용히 통과하지 마라" | 코드 주석 | **3번** (`MEM_CLOCK_MIN_FRAC`, `test_table.py`, `drift_check`) |
| "`env_hash` 는 격리 경계다" | `decisions.md` 13번 | **2번** (`bundle.py` 통계, `reproducibility()`) |

두 경우 모두 **교훈을 명시적으로 적어 둔 다음에** 같은 실수를 반복했다.
적는 것이 무용하다는 뜻이 아니라, **적는 것만으로는 부족하다**는 뜻이다.
사람이든 모델이든 코드를 쓸 때 문서를 다시 읽지 않는다.

### 규칙

**규율을 정할 때마다 "이것을 코드로 강제할 방법이 있는가" 를 먼저 묻는다.**
없으면 그 규율은 지켜지지 않을 것이라고 전제하고, 다음 중 하나를 찾는다.

| 강제 수단 | 이 저장소의 예 |
|---|---|
| 타입/시그니처로 불가능하게 | `iter_records(path, env_hash)` — 기본값이 없어 안 쓰면 `TypeError` |
| 키에 강제로 포함 | `aggregate_per_env()` — 집계 키에 `env_hash` 가 들어가 섞을 수 없다 |
| 테스트가 **호출부까지** 검사 | `test_records.py::TestCallersFilter` — 새로 추가한 사람이 빠뜨리면 걸린다 |
| 감시에 감시를 붙임 | `test_skip_guard.py` — 스킵 감시가 실제로 실패를 잡는지 확인 |
| 기본값을 안전한 쪽으로 | `answer_set(tol=None)` -> 형상별 노이즈 바닥 |

`answer_set` 이 이 원칙의 좋은 예다. 고정 `tol=0.01` 을 기본값으로 두면
**아무도 안 바꾼다** — 편해서가 아니라, 바꿔야 한다는 사실을 모르기
때문이다. `tol=None` 이 기본이고 형상별 노이즈 바닥을 쓰면, 고정값을
쓰려는 사람이 **명시적으로 근거를 대야** 한다. 그리고 "15 us 에서 1 % 는
구분 불가" 를 테스트(`test_noise.py`)로 고정해 두면 편의상 되돌리는 것도
막힌다.

헬퍼만 만들고 "이걸 쓰세요" 를 문서에 적는 것은 **강제가 아니다.**
`core/records.py` 를 만들 때 호출부 검사 테스트를 같이 넣지 않았다면
여섯 번째를 밟았을 것이다.

### 남는 한계

전부를 코드로 강제할 수는 없다. 강제할 수 없는 규율은 **적어도 그 사실을
명시**하고(`docs/next_campaign.md` 의 "새 GPU 에서 다시 재야 하는 값"),
체크리스트로 만들어 실행 시점에 읽히게 한다 — 설계 시점에 읽는 문서보다
낫다.

