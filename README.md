# kerneltab

CUTLASS GEMM 의 (형상 × config) → 성능 표를 만드는 측정 하네스.

config 선택 휴리스틱 연구를 위한 **데이터 수집 도구**이지, 예측기나
오토튜너가 아니다. 그래서 이 저장소의 어떤 자료구조에도 측정된 시간이
들어가지 않는다 (아래 "정답 누출 차단" 참조).

동일한 컨테이너 이미지가 RTX A6000 / RTX 4090 / H100 등 여러 GPU 에서
그대로 돌아가는 것을 전제로 설계했다.

---

## 디렉토리

```
core/
  types.py       데이터클래스 (아키텍처 무관)
  hardware.py    런타임 하드웨어 감지 + NVCC_ARCH
  config.py      alignment 계산, 공통 열거 로직
  shapes.py      형상 그리드 (층별)
  features.py    파생 지표 (waves, tail_waste, ...)
backends/
  __init__.py    get_backend() 레지스트리
  base.py        Backend Protocol
  sm80.py        CUTLASS 2.x API 구현 (sm_80 / sm_86 / sm_89)
  # sm90.py      3.x API 용. 나중에 추가.
hwspec/
  known.json     GPU 이름 -> (peak_tflops_f16, bandwidth_gbps)
                 ※ 데이터 디렉토리다. __init__.py 를 두지 않는다.
build/           커널 생성 / 컴파일 / 정적 분석
measure/         측정 프로토콜, GPU 상태
scripts/         진입점
results/         측정 산출물 (.gitignore 대상)
```

## 실행 순서

```bash
python3 scripts/phase0_env.py --device 3      # 환경 점검 -> results/env.json
python3 scripts/count_space.py                # 탐색 공간 크기 / 제약 funnel
python3 scripts/check_smem.py                 # smem 공식 vs 실제 CUTLASS 타입
```

`results/env.json` 은 Phase 0 이 한 번 쓰고 이후 단계는 읽기만 한다.
모든 측정 줄이 `env_hash` 로 이 파일을 참조하므로 나중에 고치면 안 된다.

---

## 설계 결정과 그 이유

나중에 이 결정들이 왜 그렇게 되었는지 알 수 없게 되면 잘못 수정될 위험이
있으므로 여기 남긴다.

### CUTLASS 2.x API 를 쓴다

`cutlass::gemm::device::GemmUniversal` (ThreadblockShape / WarpShape 기반).

3.x API (CollectiveBuilder, GemmUniversalAdapter, CuTe)는 **SM90 이상 전용**
이라 sm_86 에서 동작하지 않는다. 이것은 CUTLASS **버전**이 아니라 **API 계열**
문제다. 최신 저장소에 두 API 가 함께 있으므로 최신 CUTLASS 를 쓰면서 2.x API
를 쓰는 것이 정상이다.

`ArchTag = cutlass::arch::Sm80` 은 sm_86 에서도 정상이다. ArchTag 는 "이
기능을 지원하는 최소 SM" 을 뜻하고, `Sm86` 태그는 2.x GEMM 경로에 존재하지
않는다. 실제 타겟 SM 은 `nvcc -arch` 가 결정한다.

### KernelConfig / RuntimeConfig 분리

`split_k` 와 `split_k_mode` 는 CUTLASS 에서 **런타임 Arguments** 이지 템플릿
인자가 아니다. 이것을 `KernelConfig` 에 넣으면 빌드해야 할 커널 수가
`len(SPLIT_K) × len(SPLIT_MODE)` = 15 배로 늘어난다. 실제로는 커널 하나를
빌드해 놓고 런타임 config 만 바꿔가며 재면 된다.

### KernelConfig = 공통 필드 + `ext`

`tile_m/tile_n/tile_k` 를 공통 필드로 둔 이유는 `waves`, `tail_waste`,
`mainloop_iters` 같은 물리 피처가 tile 크기만으로 계산되기 때문이다. 이
피처들이 아키텍처 무관하게 동작해야 나중에 "SM80 에서 배운 것이 SM90 에서도
성립하는가" 전이 실험이 성립한다.

아키텍처 전용 축(SM80 의 warp tile / stages / swizzle, SM90 의 cluster /
schedule / tile scheduler)은 `ext` 뒤로 숨긴다. `core/`, `build/`, `measure/`
는 `Sm80Ext` 를 직접 참조하지 않고 Backend Protocol 로만 통신한다.

### split-K 에 3, 6, 12 를 포함한다

A6000 의 SM 개수 84 = 2² × 3 × 7 이라, 3의 배수 split-K 가 wave 정렬에
유리할 수 있다는 가설이 있다. **이 값들을 {1,2,4,8} 로 줄이지 말 것.**
줄이면 가설 자체를 검증할 수 없다.

### alignment 는 탐색 축이 아니다

형상과 레이아웃에서 유도되는 값이다. 낮은 alignment 는 트레이드오프가 아니라
단순 열등(더 좁은 벡터 접근)이므로 형상이 허용하는 최댓값을 쓴다. 다만 CUTLASS
템플릿 인자라 커널 빌드에는 영향을 주므로 `KernelConfig` 에는 포함된다.

벡터화는 항상 연속(contiguous) 차원에 걸리므로 A/B/C 각각 다른 차원을 본다
(`core/config.py: alignments_for`).

빌드 범위는 "형상 그리드에 실제로 등장하는 alignment 조합"으로 제한한다.
가능한 4³ = 64 조합 중 이 그리드에서는 6 개만 등장한다.

### 파생 지표는 JSONL 에 저장하지 않는다

`waves`, `tail_waste`, `mainloop_iters`, `arith_intensity` 등은 원본에서 언제든
재계산할 수 있다. 계산이 공짜이고, 계산식에 버그가 발견됐을 때 수십 시간짜리
측정을 다시 하지 않아도 되기 때문이다. `scripts/export.py` 가 분석 시점에
계산해 parquet 에 넣는다.

예외는 `smem_computed` 와 `expected_hmma` 둘뿐이다. 이 둘은 실측값(ptxas /
SASS)과 대조해 "커널이 의도대로 생성됐는가" 를 검증하는 용도이고, 커널당
1 줄이라 중복 부담이 없다.

### 커널 속성을 results.jsonl 에 복제하지 않는다

`kernels.jsonl` (커널당 1 줄) 과 `results.jsonl` (측정당 1 줄) 을 나누고
`kernel_id` 로 조인한다. 커널 속성을 측정 줄마다 복제하면 수백만 번 중복되고,
재빌드했을 때 두 파일이 조용히 어긋난다.

### 측정 순서를 반드시 셔플한다

순차 측정하면 GPU 온도 드리프트가 config 순서와 상관되어 "뒤에 잰 config 가
느리다" 는 체계적 편향이 생긴다. (커널, 런타임, 형상) 조합을 셔플하고 시드를
`env.json` 에 기록한다.

### 하드웨어 값은 런타임 감지한다

GPU 스펙을 파이썬 코드에 하드코딩하지 않는다. 같은 이미지가 다른 GPU 에서도
정확해야 하기 때문이다.

* API 로 얻는 값(arch, sm_count, smem_per_block, max_threads_per_sm,
  regs_per_sm, l2_bytes, 이름): CUDA Driver API 를 ctypes 로 호출한다.
  `cudaGetDeviceProperties` 는 struct 레이아웃이 CUDA 버전마다 달라 ctypes
  재현이 취약해서 쓰지 않는다.
* API 로 못 얻는 값(`peak_tflops_f16`, `bandwidth_gbps`): `hwspec/known.json`.
  **매핑에 없는 GPU 면 추정하지 않고 `UnknownGpuError` 로 중단한다.**
  roofline 과 ridge point 가 여기 직접 의존하므로 틀린 값은 결과 전체를
  오염시킨다.

`peak_tflops_f16` 은 **dense(비희소) FP16 입력 + FP32 누산** 기준이다.
데이터시트 헤드라인 숫자는 희소(2:4) 기준인 경우가 많으니 그대로 쓰면 안 된다.

### 빌드 플래그도 감지값에서 만든다

`-arch=sm_86` 을 빌드 명령에 하드코딩하지 않고 `NVCC_ARCH[hw.arch]` 에서
생성한다. Hopper 이상은 `a` 접미어(`sm_90a`)가 없으면 TMA/WGMMA 등 3.x API
핵심 기능이 컴파일 단계에서 비활성화된다.

### 정답 누출 차단

`Problem` / `KernelConfig` / `Hardware` / `RuntimeConfig` 에는 측정된 시간을
넣지 않는다. 나중에 이 객체를 입력으로 받는 예측 함수가 정답을 훔쳐보는 것을
구조적으로 막기 위함이다.

### 실패를 결측치로 두지 않는다

`status` 를 `ok | build_fail | runtime_fail | oom | numerical_fail |
below_launch_overhead | high_outlier_frac` 로 명시 기록한다. 빈 줄로 남기면
"측정을 안 한 것" 과 "측정했더니 실패한 것" 을 구분할 수 없다.

---

## 스펙 대비 변경한 것

원래 지시와 다르게 구현한 부분과 근거. 되돌릴 수 있게 남긴다.

### 1. `EpilogueOutputOp` 의 ScaleType: `NoBetaScaling` → `Default` (+ 런타임 beta=0)

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

### 2. `is_valid_runtime` 규칙 1: `K % (split_k · tile_k) == 0` → 실효 split_k 일치

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

### 3. Backend Protocol 에 메서드 3 개 추가

명세된 목록 외에 `enumerate_runtime`, `explain_kernel`, `effective_split_k` /
`workspace_bytes` 를 두었다. 각각 split-K 축 값 집합의 아키텍처 종속성, 제약
funnel 집계, CUTLASS split-K 의미론 때문이며 전부 백엔드 뒤에 있다.

---

## 컨테이너 태그 규칙

이미지 태그에 CUDA 버전과 CUTLASS 커밋을 포함한다.

```
kerneltab:cu129-cutlass-a1b2c3d
```

`latest` 태그는 쓰지 않는다. 재현성을 위해 조합을 태그로 식별한다.
같은 이유로 `results/env.json` 이 CUDA/드라이버/CUTLASS 커밋/GPU UUID 를
전부 기록하고, 모든 측정 줄이 그 해시를 참조한다.
