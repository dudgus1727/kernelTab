# 컨테이너 이식성 감사

> # ⛔ 측정 중 금지 — `scripts/phase0_env.py` 를 실행하지 마라
>
> `env_hash` 는 resume 키의 일부인데 `created_utc` / `host.ram_available_gb` /
> `hostname` / 경로가 해시에 들어간다 (아래 **P-3**). 따라서 조건이 완전히
> 동일해도 `phase0_env.py` 를 다시 돌리면 **`env_hash` 가 반드시 바뀌고,
> 지금까지의 측정 전체가 "다른 조건" 으로 취급되어 98 만 건을 처음부터 다시
> 잰다.**
>
> 전수 측정이 끝날 때까지 실행 금지. 이 문서의 수정 목록은 전부 **측정 완료
> 후** 일괄 적용한다.

**상태: 조사만 함. 아무것도 고치지 않았다.** 전수 측정이 진행 중이라
코드 변경이 위험하기 때문이다. 여기 목록화한 것은 측정 완료 후 일괄 수정한다.

감사 시점: Phase 3 측정 중 (`env_hash = 368a84f1...`, 이후 드리프트로 폐기)
매니페스트: `kerneltab:cu124-9b70b430` (`python3 scripts/manifest.py`)

위험도 기준
- **높음** — 컨테이너에서 그대로 두면 **반드시 깨지거나 조용히 틀린 데이터**를 만든다
- **중간** — 실행은 되지만 설정/마운트를 정확히 해야 한다
- **낮음** — 문서화만 하면 된다

---

## 요약: 먼저 고쳐야 할 3 가지

| # | 문제 | 위험도 |
|---|---|---|
| P-1 | `kernels.jsonl` 의 `so_path` 가 **절대 경로** | 높음 |
| P-2 | GPU 인덱스를 `env.json` 에 박고 `CUDA_VISIBLE_DEVICES` 로 강제 | 높음 |
| P-3 | `env_hash` 가 **매 실행마다 달라진다** (시각/가용RAM/호스트명 포함) | 높음 |

이 셋은 서로 독립이고, 셋 다 "실행은 되는데 결과가 이상해지는" 부류다.

---

## 1. 경로 의존성

### P-1. `so_path` 절대 경로 — **높음**

`results/kernels.jsonl` 의 각 줄이 커널 `.so` 를 **절대 경로**로 참조한다.

```json
"so_path": "/home/piai/workspace/kernelTab/build/artifacts/lib/sm86_....so"
```

`measure/runner.py: Kernel.__init__` 이 이 경로로 `ctypes.CDLL` 을 연다.
컨테이너 안 경로가 `/opt/kerneltab/...` 이면 **기존 `kernels.jsonl` 을 그대로
쓸 수 없다.** 같은 이미지로 커널을 다시 빌드하면 새 경로로 다시 기록되므로
새 실행은 문제없지만, 호스트에서 만든 산출물을 컨테이너로 넘길 수 없다.

> 수정안: `so_path` 대신 `kernel_id` 로 `ARTIFACT_DIR/lib/{kernel_id}.so` 를
> 조립한다. 경로를 데이터에 넣지 않는다. (`build/compile.py:build_kernel` 이
> `row["so_path"]` 를 쓰는 곳과 `Kernel(r["so_path"])` 호출 3 곳)

### 산출물 디렉토리가 저장소 안에 고정 — **중간**

`build/paths.py`
```python
REPO_ROOT   = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
ARTIFACT_DIR= REPO_ROOT / "build" / "artifacts"
```

환경변수로 바꿀 수 없다. 컨테이너에서는 두 디렉토리 **모두 쓰기 가능해야
한다** — 커널을 런타임에 빌드하기 때문이다. 현재 규모:

```
build/artifacts/lib   7.4 GB   (커널 .so 7,330 개, 개당 ~1 MB)
build/artifacts/src    30 MB
results                26 MB   (측정이 끝나면 수백 MB)
```

이미지 레이어에 7.4 GB 를 쓰면 컨테이너 삭제 시 사라지고 재빌드에 30 분이
든다. **두 경로 모두 볼륨으로 마운트**해야 한다.

> 수정안: `KERNELTAB_RESULTS_DIR` / `KERNELTAB_ARTIFACT_DIR` 환경변수 지원.

### CUTLASS 탐색 — **낮음**

`build/paths.py: _cutlass_candidates()` 순서:
1. `KERNELTAB_CUTLASS_DIR` / `CUTLASS_DIR` / `CUTLASS_PATH` 환경변수
2. `/opt/cutlass`, `/usr/local/cutlass`
3. `REPO_ROOT.parent/related_work/cutlass`, `REPO_ROOT.parent/cutlass`

컨테이너는 `/opt/cutlass` 에 두면 2 번에서 잡힌다. 3 번의 개인 경로 fallback 은
컨테이너에서 절대 매치되지 않으므로 무해하다. 검증은
`include/cutlass/cutlass.h` 존재로 한다 — 견고하다.

**단, `.git` 이 필요하다.** `cutlass_info()` 가 `git rev-parse HEAD` 로 커밋을
읽는다. `--depth 1` clone 은 `.git` 을 남기므로 괜찮지만, 소스 tarball 을
풀어 넣으면 `cutlass_commit = null` 이 되어 **재현성 추적이 끊긴다.**

### 그 외 경로 — **낮음**

| 위치 | 내용 |
|---|---|
| `scripts/phase0_env.py:65` | `/proc/meminfo` — 컨테이너에서는 **호스트** 메모리가 보인다. 기록용이라 무해하지만 cgroup 한도가 아님을 알아야 한다 |
| `build/paths.py:63` | `/usr/local/cuda/bin/nvcc`, `/opt/cuda/bin/nvcc` fallback — nvidia/cuda 이미지와 일치 |
| `build/compile.py: analyze_sass` | `tempfile.TemporaryDirectory()` — `TMPDIR` 을 따른다. 컨테이너 tmpfs 가 작으면 실패할 수 있으나 cubin 은 수 MB 라 문제없다 |

---

## 2. 실행 파일 의존성

| 명령 | 호출 위치 | 찾는 방식 | 위험도 |
|---|---|---|---|
| `nvcc` | `build/compile.py`, `check_smem.py`, `phase0_env.py` | `shutil.which` → `/usr/local/cuda/bin` → `/opt/cuda/bin` | 낮음 |
| `cuobjdump` | `build/compile.py` (`-res-usage`, `-xelf all`) | `shutil.which` → `$CUDA_HOME/bin` | 낮음 |
| `nvdisasm` | `build/compile.py` (`-c`) | 〃 | 낮음 |
| `nvidia-smi` | `measure/gpu_state.py`, `rehearse.py`, `verify_clock_lock.py`, `phase0_env.py` | **PATH 만** (fallback 없음) | 중간 |
| `git` | `phase0_env.py: cutlass_info`, `manifest.py` | **PATH 만** | 중간 |

* `nvidia-smi` 는 nvidia-container-toolkit 이 호스트에서 주입한다. `--gpus` 가
  없으면 존재하지 않고, 텔레메트리/클럭 조회가 전부 실패한다. 코드는
  `try_lock_clocks` 에서만 실패를 흡수하고 **텔레메트리는 `Popen` 이 조용히
  죽어 빈 CSV 가 남는다** (예외는 안 난다).
* `git` 은 base 이미지에 없을 수 있다. 없으면 `cutlass_commit`/
  `kerneltab_commit` 이 `null` 이 되어 재현성 추적만 끊긴다 (실행은 됨).

**버전 요구사항**
* nvcc: CUTLASS 4.6 + sm_86 2.x API 에는 12.4 로 충분 (현재 사용 중). SM90 이상
  백엔드를 추가하면 12.8+ 가 필요해진다.
* `cuobjdump -res-usage`, `nvdisasm -c` 는 오래된 옵션이라 버전 제약 없음.
* 드라이버: 호스트에서 온다. 현재 580.173.02 (CUDA 13.0 지원). nvcc 12.4 로
  빌드한 바이너리는 상위 드라이버에서 동작한다 (전방 호환).

---

## 3. Python 의존성

### 선언 vs 실제 — **중간**

`pyproject.toml`
```toml
dependencies = ["nvidia-ml-py>=12"]
[project.optional-dependencies]
export = ["pyarrow>=15", "pandas>=2"]
```

| 패키지 | 선언 | 설치됨 | 실사용 | 비고 |
|---|---|---|---|---|
| `nvidia-ml-py` | 필수 | 13.610.43 | ✔ `measure/gpu_state.py` | |
| `pyarrow` | optional | 25.0.1 | ✔ `scripts/export.py` | **extras 없이 설치하면 export 가 죽는다** |
| `pandas` | optional | **미설치** | ✘ **어디서도 안 씀** | 선언에서 빼야 함 |
| `numpy` | 미선언 | 미설치 | ✘ | 문제없음 |

### 버전 고정이 없다 — **높음(재현성)**

전부 `>=` 다. 같은 Dockerfile 을 6 개월 뒤 빌드하면 다른 버전이 들어온다.
`manifest_hash` 는 설치된 버전을 해싱하므로 **차이를 감지는 하지만 막지는
못한다.**

> 수정안: `requirements.lock` (pip-compile / `pip freeze`) 을 만들고
> Dockerfile 이 그것으로 설치한다. `pyproject.toml` 은 개발용으로 남긴다.

### 시스템 Python 의존 — **중간**

현재 `pynvml` 이 `/home/piai/miniconda3/lib/python3.13/site-packages/pynvml.py`
에 있다. **conda 환경에 설치된 것**이고 컨테이너에는 없다. 컨테이너에서는
`pip install nvidia-ml-py` 로 넣으면 되므로 어렵지 않으나, 현재 개발 환경이
conda 라는 사실이 `pyproject.toml` 만 봐서는 드러나지 않는다.

### Python 3.10 호환성 — **정적으로 확인함**

개발은 3.13, 컨테이너 base(ubuntu22.04) 시스템 Python 은 3.10 이다. 정적
검사 결과 **3.10 을 타겟해도 된다.**

```
$ vermin -t=3.10 --eval-annotations --backport argparse --backport dataclasses          --backport statistics --backport typing  core backends build measure scripts
Minimum required versions: 3.10
```

`--eval-annotations`(어노테이션까지 런타임 평가된다고 가정하는 엄격 모드)
에서도 3.10 이다. `ruff check --target-version py310` 도 버전 관련 위반을
보고하지 않았다 (스타일 지적 171 건은 전부 버전과 무관하다).

`X | None` 을 쓰는 파일은 **전부 `from __future__ import annotations` 가 있어**
어노테이션이 문자열로 남는다. 없는 파일은 빈 `__init__.py` 3 개뿐이다.

> 한계: 정적 분석이다. **3.10 에서 실제로 실행해 본 적은 없다.** 런타임
> 동작 차이(stdlib 세부, ctypes 구조체 정렬 등)는 잡지 못한다. 이미지를 처음
> 빌드할 때 `verify smem` / `verify splitk` 정도는 반드시 돌려봐야 한다.

따라서 base 이미지에 별도 Python 을 설치할 필요가 없다. `docker/requirements.lock`
도 `--python-version 3.10` 으로 해석해 생성했다.

---

## 4. 권한 의존성

**코드 안에 `sudo` 는 없다.** 좋다.

| 동작 | 코드 위치 | 권한 실패 시 |
|---|---|---|
| SM 클럭 고정 시도 | `measure/gpu_state.py: try_lock_clocks` | `ClockLockResult(locked=False, error=...)` 반환. 호출부가 `clock_locked=false` 로 기록하고 드리프트 주기를 600s→180s 로 단축한 뒤 **계속 진행** ✔ |
| 클럭 해제 | `reset_clocks` | bool 반환, 호출부 없음 |
| 외부 고정 인정 | `phase0_env.py --externally-locked-mhz / --externally-locked-mem-mhz` | 관리자가 호스트에서 고정한 값을 인자로 받아 `clock_locked=true` 로 기록 ✔ |

**컨테이너 안에서는 `nvidia-smi -lgc` 가 원칙적으로 불가능하다** (`CAP_SYS_ADMIN`
+ 드라이버 접근이 필요하고 컨테이너 툴킷은 이를 주지 않는다). 즉 컨테이너
경로는 항상 `--externally-locked-*` 를 쓴다. 이미 그 경로가 있다는 것이
다행이다.

주의: `--externally-locked-*` 는 **검증 없이 인자를 믿는다.**
`scripts/verify_clock_lock.py` 로 부하 상태 유지를 먼저 확인해야 하며, 이는
문서 규약이지 코드가 강제하지 않는다.

---

## 5. GPU 선택 — **높음**

### P-2. 인덱스를 데이터에 박는다

```python
# scripts/phase0_env.py
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)   # --device 3
env["device_index"] = args.device                        # 3 이 env.json 에 저장됨

# 이후 모든 스크립트 (rehearse/build_kernels/export/... 7 곳)
os.environ["CUDA_VISIBLE_DEVICES"] = str(env["device_index"])   # 다시 3
```

컨테이너에 `--gpus '"device=3"'` 로 3 번을 주입하면 **컨테이너 안에서는 그것이
device 0** 이다. `CUDA_VISIBLE_DEVICES=3` 을 설정하면 보이는 디바이스가 없어
`cuDeviceGetCount()==0` → `HardwareDetectionError` 로 죽는다. (죽는 편이 조용히
다른 GPU 를 쓰는 것보다는 낫다.)

같은 인덱스가 `nvidia-smi -i {device_index}` 로도 쓰인다
(`gpu_state._smi`, `rehearse.start_telemetry`, `verify_clock_lock`). 컨테이너
안 `nvidia-smi` 는 주입된 GPU 만 보므로 `-i 3` 은 실패하고, 텔레메트리 CSV 가
**조용히 비게 된다** — 이쪽이 더 위험하다.

### 이미 있는 안전장치

`env.json` 에 **GPU UUID** 를 기록한다 (`hardware_extra.uuid =
GPU-93284c84-...`). `measure/gpu_state.NvmlProbe` 는 UUID 로 NVML 핸들을 찾고
실패하면 인덱스로 폴백한다. 즉 **NVML 경로는 이미 이식 가능**하고, 문제는
CUDA 인덱스와 `nvidia-smi -i` 두 곳뿐이다.

> 수정안:
> 1. `CUDA_VISIBLE_DEVICES` 가 **이미 설정되어 있으면 덮어쓰지 않는다.**
>    컨테이너/스케줄러가 주입한 것을 존중한다.
> 2. `env.json` 에 `device_index` 대신 `device_uuid` 를 권위 값으로 두고,
>    실행 시 UUID → 현재 인덱스를 역조회한다. UUID 가 안 보이면 명확히 실패.
> 3. `nvidia-smi -i` 인자를 UUID 로 준다 (`nvidia-smi -i GPU-93284c84-...`
>    형식을 지원한다).

---

## 6. 그 외 발견

### P-3. `env_hash` 가 매 실행마다 달라진다 — **높음**

`phase0_env.py: canonical_hash(env)` 가 `env` 딕셔너리 **전체**를 해싱하는데,
거기에 실행마다 변하는 값이 들어 있다.

```
created_utc      2026-08-16T10:15:53.562242Z   <- 매번 다름
host.ram_available_gb  119.9                   <- 매번 다름
host.hostname    server17                      <- 호스트/컨테이너마다 다름
cutlass.dir      /home/piai/workspace/...      <- 경로 의존
cuda.nvcc_path   /usr/local/cuda/bin/nvcc      <- 경로 의존
```

`env_hash` 는 **resume 키의 일부**다 (`rehearse.load_done`). 따라서 조건이
완전히 동일해도 `phase0_env.py` 를 다시 돌리면 이전 측정 전체가 "다른 조건"
으로 취급되어 처음부터 다시 잰다. 지금까지는 조건을 바꿀 때만 재실행해서
드러나지 않았을 뿐이다.

컨테이너에서는 경로와 호스트명이 반드시 달라지므로, **호스트에서 만든
데이터와 컨테이너에서 만든 데이터를 절대 이어붙일 수 없다.**

> 수정안: `manifest.py: HASHED_KEYS` 와 같은 방식으로 **측정 조건에 실제로
> 영향을 주는 필드만** 해싱한다. 후보:
> `hardware`, `nvcc_arch_flag`, `protocol`, `clock_locked`, `locked_mhz`,
> `locked_mem_mhz`, `peak_tflops_f16_effective`, `bandwidth_gbps_effective`,
> `cutlass.commit`, `cuda.nvcc_version`, `cuda.driver_version`,
> `manifest.manifest_hash`.
> 나머지(시각, 호스트, 경로, 가용 RAM)는 기록은 하되 해시에서 뺀다.

### 매니페스트가 `env.json` 에 없다 — **중간**

`scripts/manifest.py` 를 새로 만들었지만 `phase0_env.py` 는 아직 이것을
호출하지 않는다 (측정 중 코드 변경 금지). 완료 후 `env["manifest"]` 로 넣고
`env_hash` 대상에 포함시켜야 "이 데이터가 어떤 코드로 만들어졌는가" 가
데이터 자체에 남는다.

### 드라이버 라이브러리 출처 — **낮음(문서화 필요)**

| 라이브러리 | 어디서 오는가 |
|---|---|
| `libcuda.so.1` | **호스트 드라이버** (`/lib/x86_64-linux-gnu/libcuda.so.1`). 컨테이너 툴킷이 주입 |
| `libcublas.so.12` | **CUDA 툴킷** (`/usr/local/cuda-12.4/targets/.../libcublas.so.12`). 이미지에 포함 |
| `libnvidia-ml.so` | 호스트 드라이버 (pynvml 이 dlopen) |

`core/hardware.py` 가 `ctypes.CDLL("libcuda.so.1")` 로 여는 것은 호스트 것이다.
따라서 **드라이버 버전은 컨테이너로 통일할 수 없다.** GPU 간 비교 시 드라이버
차이가 남는 축이라는 뜻이므로 `env.json` 에 `driver_version` 을 남기는 것이
중요하다 (이미 남기고 있다).

### 커널 `.so` 는 이미지에 넣으면 안 된다 — **확인됨**

* 7.4 GB (7,330 개) 이고 **아키텍처마다 다르다** (`-arch=sm_86` 이 박혀 있다).
* 빌드는 `hw.arch` 에서 `NVCC_ARCH[...]` 로 생성하므로 런타임 빌드가 정상 동작한다.
* 즉 이미지는 "소스 + 툴체인" 만 담고, 커널은 첫 실행 때 30~40 분 빌드한다.
  같은 GPU 에 재실행할 때를 위해 `build/artifacts` 를 볼륨으로 유지해야 한다.

### 측정 산출물의 append-only 특성 — **낮음**

`results/*.jsonl` 은 append-only 이고 resume 이 그 위에서 동작한다. 볼륨을
마운트하지 않으면 컨테이너를 지울 때마다 전부 날아가고 40 시간을 다시 쓴다.
**볼륨 마운트는 선택이 아니라 필수다.**

---

## T. 고치면 안 되는 것 — **의도적으로 그렇게 둔 것들**

위 수정 목록과 헷갈리지 않도록 분리한다. 아래는 **문제처럼 보이지만 문제가
아니다.** 나중에 누군가(또는 미래의 내가) "이것도 이식성 문제네" 하고 고치면
설계 의도가 깨진다.

### 드라이버 버전을 컨테이너로 통일하지 않는다

`libcuda.so.1` / `libnvidia-ml.so` 는 호스트 드라이버에서 오고 컨테이너
툴킷이 주입한다. **구조적으로 통제 불가능**하다. 이미지에 드라이버를 넣는
것은 지원되지 않는 방식이고, 넣어도 커널 모듈과 어긋난다.

→ 통일하려 하지 말고 `env.json` 의 `cuda.driver_version` 으로 **기록**한다.
GPU 간 비교에서 남는 축이라는 것을 인정하고 분석에서 다룬다.

### 커널 `.so` 를 이미지에 넣지 않는다

* 커널에는 `-arch=sm_86` 이 박힌다. 이미지에 넣으면 **그 GPU 전용 이미지**가
  되어 "동일 이미지를 여러 GPU 에" 라는 전제가 무너진다.
* sm_86 기준 7,330 개 × ~1 MB = **7.4 GB**. 아키텍처마다 늘어난다.

→ 런타임 빌드(30~40 분)가 정답이다. 느리다고 이미지에 굽지 말 것.

### 클럭 고정을 코드가 시도하지 않는(못하는) 것

`measure/gpu_state.try_lock_clocks` 가 `nvidia-smi -lgc` 를 시도는 하지만,
**컨테이너 안에서는 원칙적으로 불가능**하다 (`CAP_SYS_ADMIN` + 드라이버 쓰기
접근이 필요하고 컨테이너 툴킷은 주지 않는다).

→ `--privileged` 로 뚫으려 하지 말 것. 호스트에서 고정하고
`--externally-locked-mhz/-mem-mhz` 로 인정받는 지금 구조가 맞다.
코드가 실패를 흡수하고 `clock_locked=false` 로 계속 진행하는 것도 의도된
동작이다 (클럭을 못 잡는 환경에서도 데이터는 남아야 하고, 대신 드리프트
점검 주기가 짧아진다).

### 워밍업 20 초를 클럭 고정 후에도 유지한다

메모리 클럭을 고정하면 램프업 문제는 사라진다. 그래도 `WARMUP_SECONDS` 를
0 으로 만들지 말 것 — 클럭 외에 L2/TLB/드라이버 상태 워밍업 효과가 남고,
**클럭 고정이 풀린 환경에서 실행될 가능성**이 항상 있다 (권한이 없는 서버,
`-lmc` 를 지원하지 않는 GPU). 20 초는 40 시간 대비 무시할 수 있는 비용의
이중 안전장치다.

### `results/*.jsonl` 이 append-only 인 것

수정이 불가능해서 불편해 보이지만 의도다. 측정 중 프로세스가 죽어도 데이터가
남고, resume 이 그 위에서 동작하며, "원본은 절대 고치지 않는다" 는 규약이
파생 지표 재계산의 전제다.

→ 스키마가 바뀌면 파일을 고치지 말고 **읽는 쪽에서 재계산**한다
(`smem_computed` 를 `export.py` 가 다시 계산하는 것이 그 예다).

### 파생 지표를 JSONL 에 저장하지 않는 것

`waves` / `tail_waste` / `arith_intensity` 등이 결과 파일에 없어서 조인이
번거로워 보이지만, 계산식에 버그가 발견됐을 때 40 시간짜리 측정을 다시 하지
않기 위한 것이다. 예외 두 개(`smem_computed`, `expected_hmma`)는 커널 생성
검증용이며 커널당 1 줄이라 중복이 없다.

### `/proc/meminfo` 가 컨테이너에서 호스트 메모리를 보여주는 것

cgroup 한도가 아니라 호스트 값이 보인다. **기록용일 뿐** 어떤 판단에도 쓰지
않으므로 고칠 필요가 없다. 단, `env_hash` 에서는 빼야 한다 (P-3).

### `ArchTag = arch::Sm80` 을 sm_86 에서 쓰는 것

`Sm86` 태그로 "고치면" 컴파일이 깨진다. ArchTag 는 "이 기능을 지원하는 최소
SM" 이고 2.x GEMM 경로에 `Sm86` 은 존재하지 않는다. 실제 타겟은 `nvcc -arch`
가 결정한다.

---

## 수정 목록 (측정 완료 후 일괄 적용)

| # | 항목 | 위험도 | 손대는 파일 |
|---|---|---|---|
| 1 | `so_path` 절대 경로 → `kernel_id` 로 조립 | 높음 | `build/compile.py`, `rehearse.py`, `smoke_splitk.py`, `check_correctness.py` |
| 2 | `CUDA_VISIBLE_DEVICES` 가 이미 있으면 존중, UUID 우선 선택 | 높음 | `phase0_env.py`, 스크립트 7 곳, `gpu_state.py` |
| 3 | `env_hash` 를 측정 조건 필드만으로 계산 (+ 마이그레이션, 아래 참조) | 높음 | `phase0_env.py`, `rehearse.py` |
| 4 | `requirements.lock` 생성, 버전 고정 | 높음 | 신규 + `pyproject.toml` |
| 5 | `nvidia-smi -i` 를 UUID 로 | 중간 | `gpu_state.py`, `rehearse.py`, `verify_clock_lock.py` |
| 6 | `KERNELTAB_RESULTS_DIR` / `_ARTIFACT_DIR` 환경변수 | 중간 | `build/paths.py` |
| 7 | `pyarrow` 를 필수 의존성으로, `pandas` 제거 | 중간 | `pyproject.toml` |
| 8 | `env["manifest"]` 추가 | 중간 | `phase0_env.py` |
| 9 | 텔레메트리 `Popen` 실패를 감지해 경고 | 중간 | `rehearse.py` |
| 10 | Python 3.10 에서 실제로 돌려보기 | 중간 | (검증) |
| 11 | CUTLASS 에 `.git` 없을 때 커밋을 인자로 받는 경로 | 낮음 | `phase0_env.py` |

1~4 를 먼저 하면 컨테이너에서 "실행은 되는데 데이터가 이상한" 상황은 사라진다.

### P-3 마이그레이션: 기존 Phase 3 데이터를 어떻게 살릴 것인가

`env_hash` 정의를 바꾸면 이미 기록된 98 만 줄의 `env_hash` 가 신 정의와
달라진다. 무효화하면 안 된다. 두 안을 검토했다.

**안 B — `env_hash_v2` 필드 병기: 불가능하다.**
`results.jsonl` 은 append-only 라 기존 줄에 필드를 **추가할 수 없다.**
추가하려면 파일을 다시 쓰는 수밖에 없는데, 그것 자체가 "원본은 고치지
않는다" 규약을 깬다.

**안 A — env 레지스트리 (권장).**

`phase0_env.py` 가 `env.json` 을 덮어쓸 때 전체 env 딕셔너리를
`results/env_registry.jsonl` 에 **append** 한다 (append-only, 구 해시를 키로).

```json
{"env_hash": "368a84f1...", "env_hash_v2": "...", "env": { ...전체... }}
```

읽는 쪽(`rehearse.load_done`, `export.py`)은 레지스트리로 구 해시 → 신 해시
매핑을 만든다. **기존 줄을 한 글자도 건드리지 않고** 신 정의로 조인·resume 이
가능해진다.

안 A 를 택하는 이유:
1. append-only 규약을 지킨다. 안 B 는 규약을 깨야만 성립한다.
2. `env_hash_v2` 는 저장된 env 딕셔너리에서 **계산**되므로 손으로 관리할
   매핑 테이블이 없다 — 어긋날 여지가 없다.
3. 지금 수동으로 하고 있는 일(`cp env.json env.pre-clocklock.json`)을 대체한다.
   `env.json` 은 덮어써도 모든 과거 조건이 레지스트리에 남는다.
4. 나중에 정의를 또 바꾸면(v3) 같은 방식으로 재계산하면 된다. 안 B 는
   버전마다 필드가 늘어난다.

주의: 레지스트리를 만들기 전에 존재했던 env 는 아카이브 파일
(`env.pre-clocklock.json`, `env.minreps30.json`, `env.smlock-only.json`) 에서
한 번 백필해야 한다. 지금 남아 있으므로 가능하다.
