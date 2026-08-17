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

그래서 `export.py` 에 집계 경로를 하나로 묶었다.

```python
def _per_env(rows, key_fn, val_fn, agg, min_n=5):
    """**env_hash 별로** 집계한다. 형상/조건 단위 파생 지표의 유일한 경로다."""
```

형상별·조건별 집계를 새로 추가할 때는 반드시 이것을 쓴다. 직접
`defaultdict` 로 모으면 같은 함정을 다시 밟는다.

교훈은 더 넓다 — **`env_hash` 는 조인 키가 아니라 격리 경계다.** 두 조건의
줄이 같은 집계에 들어가는 경로가 생기면, 그 지표는 조용히 틀린다.

