# 컨테이너로 측정하기

이미지 하나로 A6000 → 4090 → H100 을 돈다. **커널은 굽지 않는다** —
아키텍처가 박히고 7.4 GB 다. 런타임에 빌드해 볼륨에 캐싱한다.

---

## 0. 호스트 준비 (컨테이너 안에서 못 하는 것)

### 클럭 고정

컨테이너 안에서는 불가능하다. `nvidia-smi -lgc/-lmc` 는 `CAP_SYS_ADMIN` +
드라이버 쓰기 접근이 필요하고 컨테이너 툴킷은 주지 않는다.
⛔ `--privileged` 로 뚫으려 하지 마라.

```bash
G=GPU-93284c84-335c-baab-2005-69c58e5cef15     # nvidia-smi --query-gpu=uuid --format=csv,noheader
I=3                                            # 그 GPU 의 인덱스
sudo nvidia-smi -i $I -pm 1
sudo nvidia-smi -i $I -lgc 1350,1350
sudo nvidia-smi -i $I -lmc 8001
```

**부하 중 실측값을 확인하라.** 요청값이 그대로 걸리지 않는다 — 컴퓨트
워크로드는 P2 로 떨어지고, A6000 은 8001 요청 → **7601** 로 캡됐다.

```bash
nvidia-smi -i $I --query-gpu=pstate,clocks.sm,clocks.mem --format=csv -l 2
```

### 볼륨 디렉토리

```bash
mkdir -p $PWD/data/results $PWD/data/artifacts
```

### ⛔ 항상 `--user $(id -u):$(id -g)` 로 돌린다

편의상 root 로 돌리면 **버그 두 개가 숨는다.** 둘 다 실제로 겪었다.

| root 로 돌리면 | 무슨 일이 일어나나 |
|---|---|
| 볼륨 권한 문제가 안 보인다 | 익명 볼륨/소유권 문제가 프로덕션에서만 터진다 |
| git 의 "dubious ownership" 이 안 난다 | `--user` 일 때만 `cutlass.commit` 이 `null` 이 된다. 그러면 **CUTLASS 버전을 바꿔도 같은 `env_hash`** 가 나온다 |

두 번째가 특히 나쁘다 — 오류가 나지 않고, 값이 있으니 정상으로 보인다.
지금은 `env_hash_v2()` 가 `EnvHashIncomplete` 로 막지만, **막힌 것을
확인하려면 그 조건에서 돌려봐야 한다.**

호스트 파일이 root 소유로 만들어지는 것도 막는다.

> ⛔ **`/data` 를 통째로 마운트하고, 하위 경로를 따로 마운트하지 마라.**
> `-v .../results:/data/results` 처럼 나눠 붙이면 익명 볼륨이 끼어들 여지가
> 생긴다. 그러면 root 소유가 되어 `--user` 로 쓰기가 막히고, 권한이 있어도
> `--rm` 과 함께 결과가 사라진다. 이미지에서 `VOLUME` 선언을 제거한 이유가
> 이것이다.

---

## 1. 빌드

```bash
./docker/build.sh              # 태그는 이미지 안에서 계산한다
```

호스트에서 `manifest.py --tag` 을 돌리면 **호스트의 nvcc** 가 들어가
`kerneltab:cu124-...` 같은 거짓 태그가 나온다. `build.sh` 는 임시 태그로
빌드한 뒤 이미지 안에서 매니페스트를 뽑아 다시 태그한다.

```
kerneltab:cu133-<manifest_hash 앞 8자리>
```

소스가 한 글자라도 바뀌면 `tree_hash` 가 바뀌고 태그가 바뀐다. 즉 **태그만
보면 이 데이터가 어떤 코드로 만들어졌는지 결정된다.** `latest` 는 쓰지 않는다.

---

## 2. 실행

```bash
TAG=$(docker images kerneltab --format '{{.Repository}}:{{.Tag}}' | head -1)
RUN="docker run --rm --gpus \"device=$G\" -v $PWD/data:/data --user $(id -u):$(id -g) $TAG"
```

`--gpus` 로 준 GPU 는 컨테이너 안에서 **인덱스 0** 이 된다. 그래서 인덱스는
뜻이 없고 **UUID 로 지정한다.** 찾지 못하면 0 번으로 떨어지지 않고 실패한다.

```bash
# 1) 환경 감지 — 실측 클럭을 넣는다 (요청값 아님)
$RUN detect --gpu $G --externally-locked-mhz 1350 --externally-locked-mem-mhz 7601

# 2) ⛔ 드리프트 3값 — 새 GPU/새 CUDA 면 반드시 (docs/new_environment_checklist.md G-4)
$RUN drift --gpu $G --max-touch 3000 --step 100

# 3) 커널 빌드 (30~40분, 볼륨에 캐싱된다)
$RUN build --align a888,a448 --jobs 24

# 4) 정합성
$RUN verify smem
$RUN verify splitk
$RUN probe --sample 200 --jobs 16      # 툴체인 회귀 점검

# 5) ⛔ 재현성 검증 (3시간) — 통과해야 전수를 켠다
$RUN sweep --max-rounds 4
$RUN anchors

# 6) 전수
$RUN sweep
$RUN watch                              # 0=진행중 3=끝남 5=죽음

# 7) 산출물 (GPU 불필요)
docker run --rm -v $PWD/data:/data --user $(id -u):$(id -g) $TAG export
docker run --rm -v $PWD/data:/data --user $(id -u):$(id -g) $TAG bundle --archive --archive-raw
```

`3` 번의 `--jobs` 는 CPU 코어 수에 맞춘다. 병렬 nvcc 는 **job 당 수 GB** 를
쓰므로 RAM 을 보고 정하라.

---

## 3. 하지 말 것

| | 왜 |
|---|---|
| `--memory` 제한 | 병렬 nvcc 가 수십 GB 를 쓴다 |
| `--cpus` 제한 | 측정 루프의 **1/3 이 호스트 측 비용**이다 |
| `--privileged` 로 클럭 고정 | `docs/decisions.md` 의 "고치면 안 되는 것" |
| `/data` 하위를 따로 마운트 | 익명 볼륨이 끼면 결과가 사라진다 |
| 12.4 데이터와 시간 비교 | `env_hash` 가 다르다. 조건이 다르면 절대 시간은 비교 불가 |
| CUDA 버전 올리기 | `env_hash_v2` 의 입력이다. **재측정이지 이전이 아니다** |

---

## 4. 드라이버 — 무엇이 호스트에서 오고 무엇이 이미지에서 오는가

```
커널 모드 드라이버   호스트     nvidia-smi 가 보고 (검증 환경: 580.173.02)
유저 모드 libcuda    이미지     CUDA forward-compat (610.43.02)
```

`nvidia-container-toolkit` 이 컨테이너 시작 시 `libcuda.so.1` 을 이미지의
compat 라이브러리로 링크한다. 그래서 **호스트에서 `cuDriverGetVersion()` 이
13000(CUDA 13.0)인데 컨테이너 안에서는 13030(13.3)** 이 나온다. CUDA 13.3
툴킷이 r580 호스트에서 도는 이유가 이것이다.

### compat 은 필수인가 — 아니다

같은 컨테이너 안에서 `libcuda.so.1` 만 호스트의 580 으로 바꿔 돌려봤다.

```bash
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.580.173.02 /tmp/native/libcuda.so.1
LD_LIBRARY_PATH=/tmp/native ./a          # -> OK, driver=13000
```

**CUDA 13.3 은 호스트 드라이버 580 에서 compat 없이도 돈다.** compat 은
툴킷이 기본으로 끼워 넣는 것이지 이 호스트에서 필수는 아니다.

### ★ compat 은 런치 오버헤드를 **+3 % 늘린다**

같은 컨테이너·같은 CUDA 13.3 에서 `libcuda` 만 바꿔 **교대로** 25 회씩
쟀다 (한쪽을 몰아 돌리면 발열/램프업 추세와 섞인다).

| | compat 610.43.02 | native 580.173.02 | 차이 |
|---|---:|---:|---:|
| `launch_async_small` | 1.8303 us | 1.7767 us | **+53.5 ns (+3.01 %)** |
| `launch_async_grid` | 1.8229 us | 1.7749 us | **+48.0 ns (+2.71 %)** |

우연이 아니다 — 순위합 U = 579/625, 564/625 (겹치면 313 근처).

**측정에는 영향이 없다.** 가장 짧은 커널이 12~14 us 이므로 50 ns 는
**0.36 %** 이고, 그 형상의 노이즈 바닥(`sigma_rel(0.014) ≈ 3.1 %`)의
1/9 이다. 드리프트 기울기로 환산하면 **모듈 1 개분**이다 (50.6 us/1,000).

그래도 기록해야 하는 이유:

* 이것은 **런치당 상수**다. 드리프트가 사는 바로 그 자리다.
* 툴킷이 호스트 드라이버를 보고 compat 을 **자동으로** 끼우거나 만다.
  같은 이미지가 호스트에 따라 다르게 동작한다.

### 그래서 compat 을 끄지 않는다

끄면 유저 모드 드라이버가 **다시 호스트마다 달라진다.** compat 을 두면
이미지가 고정하므로 오히려 변수가 줄어든다. 그리고 `driver_user_mode` 가
`env_hash` 에 들어가므로, compat 이 끼는 호스트와 안 끼는 호스트는
**자동으로 다른 `env_hash`** 를 받는다 — 섞일 수 없다.

결과:

* GPU 간 비교에서 **유저 모드 드라이버는 이미지가 고정**한다. 다만
  compat 개입 여부는 호스트에 달렸고, 그 차이는 `env_hash` 가 격리한다.
* `env.json` 은 커널 모드 / 유저 모드 / `cuDriverGetVersion` / `forward_compat`
  을 **전부** 기록한다. 둘이 달라 보이는 것이 정상이다.
* forward-compat 는 공식적으로 **데이터센터 GPU** 대상이다. A6000
  (워크스테이션)에서 동작을 실측했지만 보장은 아니다. **새 호스트에서는
  `verify smem` / `verify splitk` 를 먼저 돌려 확인하라.**

---

## 5. 문제가 생기면

| 증상 | 원인 |
|---|---|
| `⛔ /data/... 에 쓸 수 없다` | 호스트 디렉토리를 안 만들었거나 `--user` uid 불일치 |
| `ld: cannot open output file` | 위와 같다. 익명 볼륨이 끼었을 수 있다 |
| `EnvHashIncomplete: cutlass.commit` | git 이 `/opt/cutlass` 를 못 읽는다. `CUTLASS_COMMIT` 을 설정하거나 `detect --cutlass-commit` |
| `⛔ 검사한 형상이 0개다` | 커널을 덜 빌드했다. `build` 를 `--limit` 없이 |
| `DeviceNotFoundError` | `--gpus` 로 준 GPU 와 `--gpu` UUID 가 다르다 |


---

## 6. 검증 기록 (2026-08-20, A6000 / `kerneltab:cu133-*`)

이미지를 실제로 돌려 확인한 것. **시간 값을 12.4 데이터와 비교하지 않았다** —
조건이 다르다. 목적은 "파이프라인이 온전히 도는가" 다.

### (a) 환경

| | |
|---|---|
| UUID 해석 | 정확. 인덱스 0 폴백 안 함 |
| CUTLASS 예제 | `5 passed / 0 failed`, 98.94 TFLOP/s |
| 외부 클럭 고정 | 인정됨 (1350 MHz / 실측 mem 7601) |
| 볼륨 쓰기 | `results/` `artifacts/` 가 호스트에 기록됨, 소유권 정상 |
| `cutlass.commit` | `dcf215af...`, `commit_source: git` |
| 매니페스트 | `unknown` 없음. 커밋은 `injected` 로 표시 |
| 텔레메트리 | 139 샘플, SM 1350 고정 유지, **throttle 0** |
| NVML | 스냅샷 1,548회, **조회 실패 0회** |

### (b) Python 3.10 실검증

`Python 3.10.12` — 테스트 **267 passed / 2 skipped**, `verify smem` **40/40 일치**,
`verify splitk` **형상 3/3 검사, 이상 없음**.

### (c) SM86 경로 건강성 — 커널 200개, CUDA 12.4 기준선과 대조

| | 기준선 (12.4 / CUTLASS 4.6-dev) | **13.3 / CUTLASS 4.7.0** |
|---|---:|---:|
| 빌드 실패 | 0 | **0 / 200** |
| 스필 (같은 200개) | 7.0 % | **7.0 %** — 변화 **0 건** |
| HMMA 불일치 | 0 | **0** |
| smem 불일치 | 0 | **0** |
| 레지스터 중앙 / 평균 | 144 / 149.6 | **144 / 150.0** |
| 레지스터 >2 변화 | — | 11 건 (5.5 %) |
| **런치 가능성이 뒤집힌 커널** | — | **0 건** |

회귀가 아니다. 후보 집합이 안 바뀌므로 열거 설계도 그대로다. 다만 레지스터가
조금 움직이므로 **표는 재측정 대상**이다 — 그것이 `env_hash` 가 강제하는 바다.

전수 빌드(a888+a448, 2,210개)도 **실패 0**. 스필 7.7 % 인데 warp tile 별로
보면 `(128,64)` 가 **120/120 = 100 %** 다. `warp_m=128` 이 66 형상 중 최적을
한 번도 못 낸 것과 같은 이야기다.

### (d) 리허설 — 형상 6개 × 커널 40개 = 1,520 작업

| | |
|---|---|
| status | ok 1,356 / high_outlier_frac 170 |
| `max_rel_error` (a888) | 중앙 0, 최대 2.1e-03 |
| `max_rel_error` (**a448**) | 중앙 6.9e-04, 최대 1.4e-03 — alignment=4 경로 정상 |
| **재현성** | 20개 재측정, **5 % 초과 0개, 최대 0.91 %** |
| 가드 | a888 커널을 `4100` 형상에 강제로 물리면 `Misaligned Operand` 로 거부 |

cuBLAS 대비 (형상별 최고 커널):

| 형상 | 최고(ms) | cuBLAS(ms) | 비율 |
|---|---:|---:|---:|
| 64x4096x4096 | 0.0625 | 0.0666 | 1.066 |
| 1024x1024x4096 | 0.1065 | 0.1065 | 1.000 |
| 1024x4096x512 | 0.0686 | 0.0686 | 1.000 |
| **1024x4096x4100** | 0.4352 | 0.5007 | **1.151** |
| 4096x4096x4096 | 1.3855 | 1.4193 | 1.024 |
| 8192x4096x4096 | 2.6593 | 2.6593 | 1.000 |

정렬이 깨진 `4100` 에서 격차가 가장 크다 — cuBLAS 가 그 경우에 약하다는
기존 관찰과 일관된다.

### (e) 드리프트 — **CUDA 13.3 에서 다시 쟀다**

| | CUDA 12.4 | **CUDA 13.3** |
|---|---:|---:|
| 문턱 | 1,154 모듈 | **962** |
| 기울기 | 77.4 us/1,000 | **50.6** |
| `b` 변화 | −3.42 % | **−0.31 %** |
| 권장 `segments.kernels` | 461 | **384** |

**CUDA 13 이 드리프트를 고치지 않았다.** 대책이 여전히 필요하고 세그먼트는
더 작아야 한다. 자세한 것은 `docs/next_campaign.md` 6 절.
