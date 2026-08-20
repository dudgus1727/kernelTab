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

# 2) ⛔ 드리프트 3값 — 새 GPU/새 CUDA 면 반드시 (docs/new_gpu_checklist.md G-4)
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

결과:

* GPU 간 비교에서 **유저 모드 드라이버는 변수가 아니다** (이미지가 고정).
* `env.json` 은 `nvidia-smi` 의 커널 모드 버전과 `cuda_driver_api_version`
  을 **둘 다** 기록한다. 둘이 달라 보이는 것이 정상이다.
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
