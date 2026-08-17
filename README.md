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

원 스펙에서 벗어난 결정 12 건은 **[`docs/decisions.md`](docs/decisions.md)**
로 옮겼다. 각 항목에 CUTLASS 소스 근거나 실측 근거가 붙어 있다.

| # | 변경 | 되돌리면 |
|---|---|---|
| 1 | `ScaleType`: `NoBetaScaling` → `Default` + 런타임 β=0 | C 를 계속 읽어 대역폭을 낭비 |
| 2 | split-K 유효 규칙 → 실효 `split_k` 일치 | 3·6·12 가 통째로 측정 불가 |
| 3 | `warp_count ∈ {4,8}` → `{4,8,16}`, 누산기 상한 256 | 큰 타일 영역이 비어 버림 |
| 4 | CUTLASS 컴파일 제약 2 개 추가 | 빌드 실패가 결측치로 남음 |
| 5 | horizontal 스위즐 직접 구현 | CUTLASS 것은 `GemmUniversal` 과 시그니처 불일치 |
| 6 | split-K 부분합 fp16 유지 (serial·parallel 대칭) | 없는 비대칭을 가정하게 됨 |
| 7 | `launch_infeasible` status | 레지스터 초과를 빌드 실패와 혼동 |
| 8 | `pipeline_kind` 필드 | cp.async 여부를 사후에 알 수 없음 |
| 9 | warp tile (64,128) 제외 | **계산 결과가 틀린다** (60/60 확인) |
| 10 | 반복 수를 시간 예산으로 결정 | 빠른 커널의 타이머 분해능 한계 |
| 11 | cp.async 최소 접근 폭 4 B | alignment 1 에서 3 단 빌드 실패 |
| 12 | `Backend` Protocol 메서드 추가 | 백엔드별 분기가 core 로 새어 나옴 |

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

### 데이터 배포 — 번들

`results/` 는 gitignore 대상이라 저장소로 표를 넘길 수 없다. 그리고
`table.parquet` 만 넘기면 **해석이 불가능하다** — `env.json` 이 없으면 유효
ridge point 를 모르고 `is_memory_bound` 가 전부 틀린다.

그래서 배포 단위는 파일이 아니라 **번들**이다.

```
datasets/{gpu_slug}-{arch}-{env_hash8}/
    table.parquet       측정 표 (파생 지표 포함)
    env.json            측정 조건 (클럭, 실효 피크/대역폭, 프로토콜)
    kernels.jsonl       커널당 1줄 (정적 분석)
    manifest.json       코드/CUTLASS/패키지 버전
    BUNDLE.json         위 전부의 요약 + 각 파일 sha256
    validate_report.md  무결성 검사 결과
```

디렉토리명에 `env_hash` 가 들어가는 것이 핵심이다. **같은 A6000 이라도 클럭
조건이 다르면 다른 번들이며 섞으면 안 된다.**

```bash
python3 scripts/export.py                       # table.parquet
python3 scripts/bundle.py --archive --archive-raw
```

`bundle.py` 는 `validate_table.py --expect full` 을 먼저 돌리고 **통과하지
못하면 번들을 만들지 않는다.** 검증 안 된 데이터가 배포되면 안 된다.

#### 소비 (kernelrule)

```python
from core.bundle import load_bundle, load_bundles

b = load_bundle("rtx-a6000-sm_86-368a84f1")   # sha256 대조 후 로드
X = b.ranking()      # 규칙 입력 (정답 제거)
y = b.scoring()      # 채점용

df = load_bundles([a6000, rtx4090], common_shapes_only=True)   # 전이 실험
```

`common_shapes_only` 는 **모든 번들에 존재하는 (M,N,K) 만** 남긴다. 형상
그리드의 층 C 는 `waves` 를 고정하고 `sm_count` 에서 M 을 역산하므로
A6000(84 SM)과 4090(128 SM)의 형상이 다르다. 이 구분 없이 전이 실험을 하면
**규칙이 나빠서인지 형상이 달라서인지 구분할 수 없다.**

#### 배포 방법

| 용도 | 방법 |
|---|---|
| 로컬 개발 | `KERNELTAB_DATASETS=/path/to/datasets` 환경변수 |
| 공유 / 백업 | GitHub Release 에셋 (파일당 2 GB 한도) |
| 논문 아티팩트 | Zenodo (DOI, 영구 보존) — 제출 시점에 |

`results.jsonl` 원본(98 만 줄)은 번들에 넣지 않고 `--archive-raw` 로 따로
보관한다. `table.parquet` 은 파생물이라 파생 지표 계산식이 바뀌면 원본에서
다시 만들어야 한다.

### 라이선스 (A-2)

**Apache-2.0** (`LICENSE`), 서드파티 고지는 `NOTICE`.

MIT 가 아니라 Apache-2.0 인 이유:

* **특허 조항.** GEMM config 선택 휴리스틱은 특허가 걸린 영역이다.
  Apache-2.0 의 명시적 특허 실시권 + 특허 소송 시 자동 종료 조항이,
  기업이 이 표를 연구에 쓸 때 법무 검토를 통과시키는 실질적 차이를 만든다.
  MIT 에는 특허 이야기가 아예 없다.
* **CUTLASS 와 호환.** CUTLASS 는 BSD-3-Clause 이고, 이 저장소는 CUTLASS
  소스를 **포함하지 않는다** — `build/paths.py` 가 외부 체크아웃을 찾아
  헤더로 include 할 뿐이다. BSD-3 은 Apache-2.0 과 호환되므로 결합
  저작물을 Apache-2.0 으로 배포할 수 있다. 단 `.so` 를 배포한다면 CUTLASS
  고지를 실어야 한다 (현재 배포 대상은 `.so` 가 아니라 번들이다).
* **NOTICE 관례.** 서드파티 출처를 어디에 적을지가 규약으로 정해져 있다.

측정 표(`datasets/`)는 코드의 파생물이 아니라 측정값이므로 **CC BY 4.0**
으로 따로 배포한다. 인용 시 `env_hash` 와 측정 조건을 함께 밝혀야 한다 —
그것 없이는 재현이 불가능하다.

## 컨테이너 태그 규칙

이미지 태그에 CUDA 버전과 CUTLASS 커밋을 포함한다.

```
kerneltab:cu129-cutlass-a1b2c3d
```

`latest` 태그는 쓰지 않는다. 재현성을 위해 조합을 태그로 식별한다.
같은 이유로 `results/env.json` 이 CUDA/드라이버/CUTLASS 커밋/GPU UUID 를
전부 기록하고, 모든 측정 줄이 그 해시를 참조한다.
