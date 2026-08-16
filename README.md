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

> # ⛔ 전수 측정 중에는 `scripts/phase0_env.py` 를 실행하지 마라
>
> `env_hash` 에 실행 시각·호스트명·가용 RAM·경로가 섞여 있어 **같은 조건에서
> 재실행해도 해시가 달라진다.** `env_hash` 는 resume 키의 일부이므로, 측정
> 도중에 실행하면 지금까지 잰 것이 전부 무효가 되어 처음부터 다시 잰다.
> 자세한 내용과 수정안은 `docs/portability_audit.md` 의 **P-3** 참조.

## 실행 순서

```bash
# 0) 관리자가 클럭 고정 (권한 필요). 이 하네스는 권한이 없어도 동작한다.
sudo nvidia-smi -i 3 -pm 1
sudo nvidia-smi -i 3 -lgc 1350,1350     # SM 클럭
sudo nvidia-smi -i 3 -lmc 8001          # 메모리 클럭 (아래 주의 참조)

python3 scripts/phase0_env.py --device 3 --externally-locked-mhz 1350
python3 scripts/verify_clock_lock.py --minutes 5   # 부하 상태에서 정말 유지되는가
python3 scripts/count_space.py                     # 탐색 공간 / 제약 funnel
python3 scripts/check_smem.py                      # smem 공식 vs 실제 CUTLASS 타입
python3 scripts/build_kernels.py --align a888,a448 --jobs 40
python3 scripts/validate_constraints.py            # 예측 모델 vs 실측 대조
python3 scripts/check_correctness.py --device 0    # 계산이 틀린 커널 전수 검출
python3 scripts/smoke_splitk.py                    # split-K 경로 스모크
python3 scripts/rehearse.py                        # 리허설 측정
python3 scripts/recheck_stability.py               # 재현성 / 드리프트
python3 scripts/report_rehearsal.py
python3 scripts/export.py                          # -> results/table.parquet
```

> ### ⚠️ 측정이 끝나면 클럭 고정을 반드시 해제할 것
> ```bash
> sudo nvidia-smi -i 3 -rgc && sudo nvidia-smi -i 3 -rmc
> ```
> 고정된 채로 두면 그 GPU 를 쓰는 다른 사용자가 원인 모를 성능 저하를 겪는다.
> 부팅해도 persistence mode 가 켜져 있으면 유지될 수 있다.

### ⚠️ `-lgc` 는 메모리 클럭을 고정하지 않는다

`nvidia-smi -lgc` 는 **SM/graphics 클럭만** 고정한다. 메모리 클럭은 그대로
전력 상태를 따라가며, A6000 기준 **유휴 시 810 MHz(P5) ↔ 부하 시 7601 MHz**
로 9.4 배 차이가 난다.

실측한 결과, 3 분간 유휴 상태였다가 재개하면 첫 측정이 메모리 바운드 config
에서 **최대 66% 느리게** 나온다 (23.9 ms vs 실제 14.4 ms). 램프업이 끝나면
0.05% 수준으로 안정된다. 연속 측정 중에는 문제가 없지만 **시작 직후와 중단 후
재개 직후가 위험**하다.

대책:
* `scripts/rehearse.py` 는 측정 루프 시작 전에 `WARMUP_SECONDS`(기본 20초)
  동안 부하를 걸어 메모리 클럭을 램프업시킨다. 램프업 전후 값을 로그에 남긴다.
* 측정 줄마다 `mem_clock_mhz` 를 기록한다. 나중에 이상치를 만나면 이 값으로
  램프업 문제인지 판별할 수 있다.
* 관리자가 메모리 클럭도 고정하면 근본 해결이다:
  `sudo nvidia-smi -i N -lmc 8001`. 워밍업은 그대로 유지한다 — 클럭을 고정해도
  캐시/드라이버 상태 워밍업 효과는 남으므로 이중 안전장치다.

> **주의: 요청한 값과 실제 값이 다르다.** A6000 에 `-lmc 8001` 을 걸어도 컴퓨트
> 워크로드는 P2 상태로 동작해 실제 메모리 클럭은 **7601 MHz** 다 (P0 최대치가
> 8001). 고정의 효과는 "유휴 810 ↔ 부하 7601" 이 "항상 7601" 로 바뀌는 것이다.
> `env.json` 의 `locked_mem_mhz` 에는 **부하 중 실제 관측값**을 넣어야 한다.

### 대역폭도 클럭에 맞춰 보정한다

`peak_tflops_f16` 을 SM 클럭으로 보정하는 것과 같은 이유로, `bandwidth_gbps`
도 메모리 클럭으로 보정해야 한다. `known.json` 의 `bandwidth_gbps_at_mem_mhz`
가 기준 클럭이다.

```
A6000 스펙 : 768.0 GB/s @ 8001 MHz (P0)
P2 실측     : 729.7 GB/s @ 7601 MHz     <- 컴퓨트 워크로드의 실제 값
```

이 보정을 하지 않으면 ridge point 의 **분모**가 5% 틀린다. SM 클럭 보정과
합쳐서 최종 ridge point 가 결정된다. `phase0_env.py` 는 버스 폭 x 2(DDR) x
클럭 으로 독립 교차 검증도 한다 (384-bit x 2 x 7601 MHz / 8 = 729.7 GB/s).

`results/env.json` 은 Phase 0 이 한 번 쓰고 이후 단계는 읽기만 한다.
모든 측정 줄이 `env_hash` 로 이 파일을 참조하므로 나중에 고치면 안 된다.
조건이 바뀌면(예: 클럭 고정) 기존 파일을 `env.<조건>.json` 으로 보관하고
새로 생성한다.

### 클럭 고정과 roofline 보정 (분자)

클럭을 고정하면 스펙 피크(부스트 클럭 기준)를 그대로 쓸 수 없다. ridge point
가 실제보다 높게 나와 "이 형상이 메모리 바운드인가" 판정이 틀린다.
`hwspec/known.json` 의 `peak_tflops_f16_at_mhz` 로 보정한다.

```
A6000 스펙 : 154.8 TFLOP/s @ 1800 MHz
1350 MHz 고정: 116.1 TFLOP/s
```

분모(대역폭)도 같은 방식으로 보정한다 — 바로 위 "대역폭도 클럭에 맞춰
보정한다" 참조. **둘을 모두 보정해야** 최종 ridge point 가 맞는다.

`env.json` 에 `peak_tflops_f16_effective` 로 기록한다. **모든 호출부는
`core.hardware.hardware_from_env(env)` 로만 `Hardware` 를 만든다** —
`Hardware(**env["hardware"])` 를 직접 쓰면 스펙 피크가 들어와 roofline 이
틀린다. 스펙 기준값도 `ridge_point_spec` 으로 함께 남긴다.

`env.json` 은 `sm_clock_mhz` 와 `mem_clock_mhz` 를 **둘 다** 기록한다.
`-lgc` 는 SM 클럭만 건드리므로, 나중에 "이 측정이 어떤 클럭 조건이었나" 를
재구성하려면 두 값이 모두 필요하다.

> **다른 GPU 로 확장할 때 이것이 결정적이다.** GPU 마다 다른 클럭으로
> 고정하게 되는데, 유효 피크를 쓰지 않으면 "메모리 바운드" 의 의미가 GPU
> 마다 달라져 전이 실험이 오염된다. A6000 을 1350 MHz 로 고정하면 ridge
> point 가 201.6 → 151.2 로 25% 이동하고, 그 사이에 있는 형상들의 bound
> 분류가 통째로 뒤집힌다. 새 GPU 를 추가할 때는 `known.json` 에
> `peak_tflops_f16_at_mhz` 와 `bandwidth_gbps_at_mem_mhz` 를 **반드시** 같이
> 넣어야 하며, 없으면 보정이 조용히 생략된다.

### 측정 조건이 다른 데이터는 섞지 말 것

리허설 데이터(`results.jsonl` 의 `env_hash = b42df475...`)는 **클럭 고정 전**
조건이다. `sw_power_cap` 스로틀이 측정 시간의 51% 동안 걸렸고 SM 클럭이
1485~1935 MHz 를 오갔다. 클럭 고정 후(`env_hash = 1f0b6924...`)의 전수 데이터와
**절대 시간을 직접 비교할 수 없다.** 조인/필터할 때 반드시 `env_hash` 로 나눠라.

| | 미고정 | 1350 MHz 고정 |
|---|---|---|
| 재현성 변동폭 중앙값 | 0.28% | 0.04% |
| p90 | 3.42% | 0.86% |
| max | 8.91% | 4.91% |
| 드리프트 (9분) | 2.59% | 0.05% |
| `sw_power_cap` | 51% | 0% |

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
below_launch_overhead | high_outlier_frac | launch_infeasible` 로 명시
기록한다. 빈 줄로 남기면 "측정을 안 한 것" 과 "측정했더니 실패한 것" 을
구분할 수 없다.

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

### 3. 탐색 축 확대: `warp_count ∈ {4,8}` → `{4,8,16}`, 누산기 상한 200 → 256

원래 값은 추정치였다. 레지스터 실사용량과 스필은 빌드 시 `-Xptxas -v` 로
**실측**해 `kernels.jsonl` 에 남기므로, 추정치로 미리 자르는 것보다 빌드해서
기록하는 편이 데이터로서 낫다. "성능 기준으로 거르지 마라" 원칙과도 일관된다.

부수적으로 `threads_per_block <= min(1024, hw.max_threads_per_sm)` 가드를
추가했다. warp_k 분할(PartitionsK)까지 곱하면 블록 스레드 수가 CUDA 한도를
넘을 수 있는데, 이건 런치 자체가 불가능한 조합이다.

결과: alignment 조합당 1,155 → 1,575 개.

### 4. `is_valid_kernel` 에 CUTLASS 컴파일 제약 2 개 추가

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

### 5. horizontal 스위즐을 직접 구현 (`measure/kt_swizzle.h`)

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

### 6. split-K 부분합 정밀도 — serial 과 parallel 모두 fp16 (비대칭 아님)

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

### 7. `launch_infeasible` status 추가

`cutlass::Kernel2` 에는 `__launch_bounds__` 가 없어서 ptxas 가 스레드 수에 맞춰
레지스터를 제한하지 않는다. 그 결과 `regs_per_thread × threads > regs_per_sm`
인 커널이 만들어지는데, 이건 빌드는 되지만 **런치가 불가능**하다
(`cudaOccupancyMaxActiveBlocksPerMultiprocessor` 가 0 을 돌려준다).
빌드 전에는 예측할 수 없고(레지스터는 ptxas 의 결과) 빌드 후에는 정확히
판정되므로, 측정 직전에 확인해 실행하지 않고 이 status 로 기록한다.

### 8. `pipeline_kind` 필드 추가

`stages == 2` 는 `MmaPipelined`(동기 LDG + STS), `stages >= 3` 은
`MmaMultistage`(cp.async / LDGSTS) 로 **구현이 다른 별개 커널**이다.
SASS 에서도 정확히 갈린다 (stages=2 는 LDGSTS 0 개, stages≥3 은 795/795 사용).
`stages` 를 하나의 연속 축으로 보면 잘못된 결론이 나오므로 명시적으로 남긴다.

### 9. warp tile (64,128) 제외 — 계산이 틀린다

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

### 10. 측정 반복 수를 고정 하한이 아니라 시간 예산으로 정한다

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

### 11. cp.async 최소 접근 폭 — alignment 1 은 2단만 가능

multistage(stages ≥ 3)는 `cp.async`(LDGSTS)로 전역→smem 복사를 하는데,
cp.async 는 **4/8/16 바이트 접근만** 지원한다. fp16 × alignment 1 = 2 바이트가
정확히 여기 걸려 `static_assert: Size is not supported` 가 난다.

실측: `a118`(K=4097) 에서 stages=2 는 31/31 성공, stages≥3 은 0/140 성공.
`is_valid_kernel` 에 넣었고 실측 2,901 건과 대조해 오탐 0 / 미탐 0 이다.

결과적으로 층 D 의 `K=4097` 형상은 유효 커널이 445 개뿐이다 (다른 alignment 는
1,305~1,435). **alignment 1 은 2단 파이프라인만 쓸 수 있다** 는 것 자체가
이 형상에 대한 결론이다.

### 12. Backend Protocol 에 메서드 추가

명세된 목록 외에 `enumerate_runtime`, `explain_kernel`, `ext_from_dict`,
`pipeline_kind`, `effective_split_k` / `workspace_bytes` 를 두었다. 각각
split-K 축 값 집합의 아키텍처 종속성, 제약 funnel 집계, JSONL 역직렬화,
파이프라인 계열 구분, CUTLASS split-K 의미론 때문이며 전부 백엔드 뒤에 있다.

---

---

## `results/table.parquet` — 스키마와 소비 규약

이 표가 이 저장소의 최종 산출물이다. 별도 프로젝트(`kernelrule`)가 소비한다.
`scripts/export.py` 가 `results.jsonl` ⨝ `kernels.jsonl` 을 `kernel_id` 로 조인하고
파생 지표를 계산해 만든다. **원본 JSONL 에서 언제든 재생성 가능하다.**

### 컬럼 — 출처별

`(P)` = 형상, `(R)` = 런타임 config, `(K)` = 커널 config, `(S)` = 정적 분석,
`(M)` = 측정값, `(C)` = 측정 조건, `(D)` = 파생(export 시 계산), `(V)` = 검증 플래그

| 컬럼 | 타입 | 출처 | 의미 |
|---|---|---|---|
| `M`, `N`, `K` | int64 | P | GEMM 형상 |
| `dtype`, `acc_dtype` | string | P | `f16` / `f32` |
| `layout_a/b/c` | string | P | `row` / `col` |
| `split_k` | int64 | R | 요청한 split-K 슬라이스 수 |
| `split_k_mode` | string | R | `serial` \| `parallel` |
| `actual_split_k` | int64 | M | CUTLASS 가 **실제로** 만든 슬라이스 수 (`grid.z`) |
| `kernel_id` | string | K | 커널 식별자. `kernels.jsonl` 조인 키 |
| `arch` | string | K | `sm_86` |
| `tile_m/n/k` | int64 | K | threadblock tile. **아키텍처 공통 필드** |
| `align_a/b/c` | int64 | K | 형상에서 유도된 alignment (탐색 축이 아니다) |
| `ext_warp_m/n/k` | int64 | K | SM80 전용 — warp tile |
| `ext_stages` | int64 | K | SM80 전용 — 파이프라인 단수 |
| `ext_swizzle_type`, `ext_swizzle_n` | string,int | K | SM80 전용 — 래스터 스위즐 |
| `pipeline_kind` | string | K | `pipelined`(stages=2) \| `multistage`(≥3). **구현이 다른 별개 커널이다** |
| `regs_per_thread`, `threads` | int64 | S | ptxas 실측 / `GemmKernel::kThreadCount` |
| `smem_dynamic`, `smem_static_bytes` | int64 | S | CUTLASS 는 dynamic 을 쓰므로 static 은 보통 0 |
| `spill_stores`, `spill_loads`, `local_bytes` | int64 | S | 레지스터 스필 |
| `hmma_count`, `lds_count`, `sts_count`, `ldsm_count`, `ldg_count`, `cpasync_count`, `inst_total` | int64 | S | `nvdisasm -c` SASS 정적 카운트 |
| `max_blocks_per_sm`, `cutlass_max_blocks` | int64 | S | occupancy (CUDA API / CUTLASS 계산) |
| `res_regs`, `res_local` | int64 | S | `cuobjdump -res-usage` 교차검증값 |
| `build_seconds`, `cutlass_commit`, `nvcc_arch` | – | S | 빌드 정보 |
| **`time_ms`** | double | M | **정답.** IQR 제거 후 중앙값 |
| `time_std_ms`, `time_min_ms`, `time_max_ms` | double | M | 측정 분포 |
| `n_reps`, `outlier_frac` | int64,double | M | 실제 반복 수, IQR 제외 비율 |
| **`cublas_ms`** | double | M | **정답.** 같은 형상의 cuBLAS 참조 시간 |
| `max_rel_error` | double | M | `max|D−D_ref| / max|D_ref|` |
| `workspace_bytes`, `workspace_dtype`, `partials_dtype` | – | M | split-K 작업 공간. 부분합은 serial/parallel 모두 fp16 |
| `status` | string | M | `ok` \| `runtime_fail` \| `oom` \| `numerical_fail` \| `below_launch_overhead` \| `high_outlier_frac` \| `launch_infeasible` |
| `error` | string | M | 실패 사유 (성공 줄은 null) |
| `sm_clock_mhz`, `mem_clock_mhz`, `gpu_temp_c`, `power_w` | – | C | 측정 시점 NVML 스냅샷 |
| `clock_locked`, `locked_mhz` | – | C | 클럭 고정 여부/값 |
| `env_hash` | string | C | **측정 조건 식별자. 조인/필터 시 반드시 나눌 것** |
| `timestamp` | string | C | 측정 시각(UTC) |
| `waves`, `waves_occ` | double | D | `features.waves()` — occ 는 `max_blocks_per_sm` 반영 |
| `tail_waste`, `tail_waste_occ` | double | D | `features.tail_waste()` |
| `grid_tiles` | int64 | D | `features.grid_tiles()` |
| `mainloop_iters` | int64 | D | `features.mainloop_iters()` |
| `tail_m_frac`, `tail_n_frac` | double | D | `features.tail_m_frac()` / `tail_n_frac()` |
| `arith_intensity` | double | D | `features.arith_intensity()` |
| `ridge_point`, `ridge_point_spec` | double | D | `features.ridge_point()` — **실효/스펙 두 가지** |
| `is_memory_bound` | bool | D | `features.is_memory_bound()` (실효 기준) |
| `flops`, `bytes_moved` | double | D | `features.flops()` / `bytes_moved()` |
| `theoretical_occupancy`, `regs_total_per_block`, `launchable` | – | D | 자원 파생 |
| **`tflops`, `frac_of_peak`, `vs_cublas`** | double | D | **정답에서 유도된 값** |
| `has_spill`, `smem_matches`, `hmma_matches` | bool | V | 검증 플래그 |
| `smem_computed`, `expected_hmma` | int64 | V | 계산값. 실측(`smem_dynamic`/`hmma_count`)과 대조용 |

### `ext_` 접두어 규칙

아키텍처 전용 필드는 `KernelConfig.ext` 에 있고 export 시 `ext_` 접두어로
평탄화된다. SM90 백엔드를 추가하면 `ext_cluster_m`, `ext_tile_scheduler` 등이
생기고, **SM80 데이터에서는 그 컬럼이 null 이 된다** (반대도 마찬가지).
Parquet 은 null 컬럼을 효율적으로 저장하므로 여러 아키텍처를 한 테이블로
합쳐도 문제없다.

공통 필드(`tile_m/n/k`)만으로 물리 피처가 계산되도록 설계했으므로,
`ext_*` 를 쓰지 않는 규칙은 아키텍처 간 전이가 가능하다.

### ⛔ 소비 규약 — 정답 컬럼을 규칙에 노출하지 말 것

다음은 **측정 결과(정답)** 이며, config 를 선택하는 규칙 함수의 입력이 되면
안 된다.

```
time_ms  time_std_ms  time_min_ms  time_max_ms  n_reps  outlier_frac
cublas_ms  tflops  frac_of_peak  vs_cublas
```

이 저장소는 `Problem` / `KernelConfig` / `Hardware` / `RuntimeConfig` 어디에도
측정 시간을 넣지 않아 **자료구조 수준에서** 누출을 막는다. parquet 은 평면
테이블이라 그 보호가 없으므로, **소비하는 쪽이 구조적으로 보장해야 한다.**

권장 형태 — 규칙 함수에 넘기기 전에 정답 컬럼을 물리적으로 분리한다:

```python
ANSWER_COLS = {"time_ms", "time_std_ms", "time_min_ms", "time_max_ms",
               "n_reps", "outlier_frac", "cublas_ms", "tflops",
               "frac_of_peak", "vs_cublas"}

X = df.drop(columns=ANSWER_COLS)       # 규칙이 보는 것
y = df[["time_ms"]]                    # 채점만 하는 쪽이 보는 것
```

`status` 와 `max_rel_error` 는 정답은 아니지만 **측정을 해봐야 아는 값**이다.
규칙이 이것을 입력으로 쓰면 "돌려보고 고른다" 가 되므로 같이 제외하는 편이
안전하다. `launchable` / `smem_matches` / `hmma_matches` 는 빌드 시점에
알 수 있으므로 써도 된다.

### 필터링 규약

```python
df = df[df.env_hash == "<하나의 해시>"]   # 조건이 다른 데이터를 섞지 말 것
df = df[df.status == "ok"]                # 실패는 결측이 아니라 명시 기록이다
```

`env_hash` 로 나누지 않으면 클럭 고정 전후 데이터가 섞인다. 절대 시간이
비교 불가능해진다.

## 컨테이너 태그 규칙

이미지 태그에 CUDA 버전과 CUTLASS 커밋을 포함한다.

```
kerneltab:cu129-cutlass-a1b2c3d
```

`latest` 태그는 쓰지 않는다. 재현성을 위해 조합을 태그로 식별한다.
같은 이유로 `results/env.json` 이 CUDA/드라이버/CUTLASS 커밋/GPU UUID 를
전부 기록하고, 모든 측정 줄이 그 해시를 참조한다.
