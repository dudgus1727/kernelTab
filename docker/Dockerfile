# syntax=docker/dockerfile:1.7
#
# kerneltab 측정 하네스 컨테이너.
#
#   docker build -f docker/Dockerfile \
#       --build-arg KERNELTAB_COMMIT=$(git rev-parse HEAD) \
#       -t $(python3 scripts/manifest.py --tag) .
#
# ---------------------------------------------------------------------------
# 이 이미지가 담지 않는 것과 그 이유
# ---------------------------------------------------------------------------
# * 커널 `.so` — 담지 않는다. `-arch=sm_86` 처럼 감지된 arch 로 빌드되므로
#   이미지에 넣으면 그 GPU 전용 이미지가 되고, "같은 이미지를 여러 GPU 에"
#   라는 전제가 무너진다. 크기도 sm_86 기준 7.4 GB 다.
#   런타임에 빌드하고(30~40분) /data/artifacts 볼륨에 캐싱한다.
#
# * results/ — 볼륨 마운트 지점이다. 마운트하지 않으면 컨테이너를 지울 때
#   측정 결과가 사라진다. 선택이 아니라 필수다.
#
# * 클럭 고정 — 컨테이너 안에서 불가능하다. `nvidia-smi -lgc/-lmc` 는
#   CAP_SYS_ADMIN + 드라이버 쓰기 접근이 필요하고 컨테이너 툴킷은 주지 않는다.
#   ⛔ `--privileged` 로 뚫으려 하지 마라. 호스트에서 고정하고
#   `detect --externally-locked-mhz/-mem-mhz` 로 그 값을 인정받는다.
#   (미고정 max 8.91% vs 고정 max 0.35%)
#
# ---------------------------------------------------------------------------
# 드라이버 — 커널 모드는 호스트, **유저 모드는 이미지**
# ---------------------------------------------------------------------------
# 이 문서에 예전에 "드라이버 버전은 컨테이너로 통일할 수 없다" 고 적혀
# 있었는데, 실측해 보니 **절반만 맞다.**
#
#   커널 모드 드라이버   호스트          (nvidia-smi 가 보고. 여기선 580.173.02)
#   유저 모드 libcuda    **이미지**      (CUDA forward-compat, 여기선 610.43.02)
#
# nvidia-container-toolkit 이 컨테이너 시작 시
# `/usr/lib/x86_64-linux-gnu/libcuda.so.1` 을 이미지의
# `/usr/local/cuda/compat/libcuda.so.610.43.02` 로 링크한다. 그래서
# 호스트에서 `cuDriverGetVersion()` 이 **13000**(CUDA 13.0)인데 컨테이너
# 안에서는 **13030**(13.3)이 나온다. CUDA 13.3 툴킷이 r580 호스트에서 도는
# 이유가 이것이다.
#
# ⚠️ 두 가지 결과가 따라온다.
#   1) GPU 간 비교에서 유저 모드 드라이버는 **변수가 아니다** (이미지가 고정).
#      env.json 은 nvidia-smi 의 커널 모드 버전과 cuDriverGetVersion 을
#      **둘 다** 기록한다. 둘이 다른 것이 정상이다.
#   2) forward-compat 는 NVIDIA 가 공식적으로는 **데이터센터 GPU** 대상으로
#      지원한다. A6000(워크스테이션)에서 동작을 실측 확인했지만, 새 호스트
#      에서는 `verify smem` / `verify splitk` 가 도는지 먼저 확인할 것.
#      호스트 드라이버가 너무 낮으면 compat 로도 안 된다.

# ---------------------------------------------------------------------------
# BASE — nvidia/cuda:13.3.1-devel-ubuntu22.04
# ---------------------------------------------------------------------------
# 태그는 같은 이름으로 내용이 바뀌므로 재현성이 없다. 다이제스트로 고정한다.
#   docker buildx imagetools inspect nvidia/cuda:13.3.1-devel-ubuntu22.04
#
# 왜 13.3 인가:
#   * CUTLASS 4.7.0 CHANGELOG 가 "Optimal code generation with CUDA toolkit
#     versions 13.3" 을 명시한다. **벤더가 지목한 조합**이 가장 안전하다.
#     (13.2 는 CUTLASS 4.5.0 이 지목한 것이고 그 사이 두 릴리스가 지났다.)
#   * 13.4 는 developer preview 라 쓰지 않는다.
#   * ubuntu22.04 = Python 3.10 -> docker/requirements.lock 을 그대로 쓴다.
#     24.04(3.12)로 가면 lock 을 다시 만들어야 한다.
#   * CUDA 13.x 는 pre-Turing 만 드롭했다. sm_86/89/90a/100 전부 지원.
#
# devel 이어야 한다. runtime 에는 nvcc / cuobjdump / nvdisasm 이 없다.
#
# ⚠️ nvcc 는 **13.3.73** 이다. `env_hash_v2` 의 입력이므로(core/env_hash.py)
#    버전을 올리면 다른 env_hash 가 되고 기존 표와 절대 시간을 비교할 수
#    없다. **CUDA 를 올리는 것은 재측정이지 이전이 아니다.**
#
# 다중 아키텍처 매니페스트 리스트라 amd64/arm64 둘 다 이 다이제스트로 간다.
ARG BASE_IMAGE=nvidia/cuda@sha256:c13e01184a7fb2a90814153fdff2250f08312377887d2673cf5bc769876a37f3
FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive

# git       : CUTLASS/kerneltab 커밋 기록. 없으면 재현성 추적이 끊긴다.
# python3   : ubuntu22.04 는 3.10. vermin 으로 정적 확인했고 여기서 실측한다.
# zstd      : 번들 압축 (bundle.py --archive)
# ca-certificates : git clone https
# hadolint DL3008(apt 버전 고정)은 의도적으로 따르지 않는다. Ubuntu 아카이브는
# 패치 버전을 회전시키므로 고정하면 몇 달 뒤 빌드가 그냥 실패한다. 그리고
# 이 넷은 **커널 코드 생성에 영향을 주지 않는다** — 재현성이 걸린 것(nvcc,
# CUTLASS, Python 패키지)은 전부 다이제스트/커밋/해시로 고정돼 있다.
# 실제로 무엇으로 빌드됐는지는 manifest.py 가 이미지마다 기록한다.
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates python3 python3-pip zstd \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# CUTLASS — 태그로 고정하고 커밋을 **주입**한다
# ---------------------------------------------------------------------------
# 헤더 온리라 빌드하지 않는다. include/ 와 tools/util/include/ 만 쓴다.
#
# `--depth 1` 이라 .git 이 얕다. `git rev-parse HEAD` 는 동작하지만 얕은
# clone 에 기대는 대신 커밋을 환경변수로 **주입**한다 —
# `cutlass.commit` 은 `env_hash_v2` 의 키이므로 비면 서로 다른 버전이 같은
# 해시를 받는다. `phase0_env.py --cutlass-commit` 이 이것을 받아
# `commit_source: injected` 로 기록한다 (git 에서 읽은 것처럼 위장하지 않는다).
ARG CUTLASS_TAG=v4.7.0
ARG CUTLASS_COMMIT=dcf215af68a2d08d305076c152a06f201728cd53
RUN git clone --depth 1 --branch ${CUTLASS_TAG} \
        https://github.com/NVIDIA/cutlass.git /opt/cutlass \
 && test -f /opt/cutlass/include/cutlass/gemm/device/gemm_universal.h \
 && got=$(git -C /opt/cutlass rev-parse HEAD) \
 && if [ "$got" != "${CUTLASS_COMMIT}" ]; then \
        echo "CUTLASS 커밋 불일치: 태그 ${CUTLASS_TAG} -> $got, 기대 ${CUTLASS_COMMIT}" >&2; \
        exit 1; \
    fi \
 && git config --system --add safe.directory /opt/cutlass \
 && git config --system --add safe.directory /work
# ↑ `--user $(id -u)` 로 돌리면 git 이 root 소유 저장소를 "dubious ownership"
#   으로 거부한다. 그러면 `git rev-parse HEAD` 가 실패하고 `cutlass.commit`
#   이 **null 인 env.json** 이 만들어진다. cutlass.commit 은 env_hash_v2 의
#   키이므로, null 이면 서로 다른 CUTLASS 버전이 **같은 해시**를 받는다.
#   실제로 이 순서로 밟았다. 이제 세 겹으로 막는다:
#     1) safe.directory (여기)
#     2) phase0_env 가 $CUTLASS_COMMIT 로 폴백
#     3) env_hash_v2 가 필수 키 누락 시 예외
# ↑ 태그는 옮겨질 수 있다. 커밋과 대조해 **다르면 빌드를 실패시킨다.**
#   조용히 다른 CUTLASS 로 이미지를 만들면 표가 달라진다.
#
# ⚠️ sm_86 백엔드는 CUTLASS **2.x API**(device::GemmUniversal,
#    ThreadblockShape/WarpShape)에 의존한다. 4.7.0 에 살아 있고
#    deprecated 표시도 없음을 확인했다. 헤더 존재만으로는 부족하므로
#    `probe toolchain` 으로 실제 빌드까지 확인한다 (scripts/probe_toolchain.py).
ENV KERNELTAB_CUTLASS_DIR=/opt/cutlass \
    CUTLASS_COMMIT=${CUTLASS_COMMIT}

# ---------------------------------------------------------------------------
# Python 의존성 — lock 으로 고정, 해시 검증
# ---------------------------------------------------------------------------
# pyproject 의 `>=` 는 재현성이 없다. 6개월 뒤 같은 Dockerfile 이 다른 버전을
# 가져온다. `uv pip compile --python-version 3.10 --generate-hashes` 로 만든
# lock 을 `--require-hashes` 로 설치해 공급망 변조까지 막는다.
COPY docker/requirements.lock /tmp/requirements.lock
RUN python3 -m pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock \
 && rm /tmp/requirements.lock

# ---------------------------------------------------------------------------
# kerneltab
# ---------------------------------------------------------------------------
WORKDIR /work
COPY kerneltab/    /work/kerneltab/
COPY scripts/      /work/scripts/
COPY hwspec/       /work/hwspec/
COPY tests/        /work/tests/
# docker/ 도 넣는다. requirements.lock 이 이미지 안에 있어야
# **이 이미지가 무엇으로 만들어졌는지**를 안에서 확인할 수 있고,
# tests/test_portability.py 의 lock 검사가 컨테이너에서도 돈다.
COPY docker/       /work/docker/
COPY docker/entrypoint.sh /usr/local/bin/kerneltab
COPY pyproject.toml README.md LICENSE NOTICE /work/

# ⚠️ `pip install` 하지 않는다. **PYTHONPATH 로 붙인다.**
#
# 이유 셋:
#   1) site-packages 에 설치하면 코드가 **두 벌**이 된다. scripts/*.py 는
#      `sys.path.insert(0, REPO_ROOT)` 로 /work 를 먼저 보므로, 설치본과
#      /work 가 어긋나면 "고쳤는데 안 고쳐진" 상태가 된다.
#   2) `hwspec/` 과 `artifacts/` 는 패키지 밖이라 어차피 환경변수로
#      경로를 알려줘야 한다 (docs/packaging.md).
#   3) ubuntu22.04 의 pip/setuptools 조합이 PEP 660(editable)을 못 해
#      `-e` 가 실패한다. 그걸 뚫으려고 빌드 시점에 pip 을 업그레이드하면
#      **--require-hashes 로 잠근 의미가 사라진다.**
#
# 소비 프로젝트(kernelrule)의 `pip install -e ../kernelTab` 경로는 호스트
# 에서 검증한다. 이 이미지는 **측정 런타임**이지 배포 대상이 아니다.
ENV PYTHONPATH=/work
RUN chmod +x /usr/local/bin/kerneltab

# 커밋은 빌드 인자로 주입한다. .git 을 이미지에 넣지 않는다 — 크고, 컨텍스트
# 에 따라 dirty 상태가 달라져 레이어 캐시가 무의미해진다.
ARG KERNELTAB_COMMIT=unknown
ENV KERNELTAB_COMMIT=${KERNELTAB_COMMIT}

# 산출물 경로. 둘 다 **쓰기 가능해야 하고 볼륨이어야 한다.**
#   results   : 측정 결과 (append-only jsonl). 날리면 30~40시간을 다시 쓴다.
#   artifacts : 커널 .so 7.4 GB + 생성된 .cu. 날리면 30~40분 재빌드.
ENV KERNELTAB_RESULTS_DIR=/data/results \
    KERNELTAB_ARTIFACT_DIR=/data/artifacts \
    KERNELTAB_HWSPEC_DIR=/work/hwspec
RUN mkdir -p /data/results /data/artifacts
# ⛔ **`VOLUME` 을 선언하지 않는다.** 예전 초안에는
#    `VOLUME ["/data/results", "/data/artifacts"]` 가 있었는데, 그러면
#    `-v $PWD/data:/data` 로 바인드 마운트해도 그 **두 하위 경로만 익명
#    볼륨이 덮어쓴다.** 익명 볼륨은 이미지 내용(root 소유 빈 디렉토리)으로
#    초기화되므로:
#      * `--user $(id -u)` 로 돌리면 쓰기 권한이 없어 실패하고,
#      * 권한이 있어도 `--rm` 과 함께 **측정 결과가 사라진다.**
#    실제로 밟았다. detect 가 "예제 검증 실패" 로 죽었고 원인은
#    `/usr/bin/ld: cannot open output file ...: Permission denied` 였다.
#    바인드 마운트를 강제하는 방법은 VOLUME 이 아니라 **시작 시 쓰기 검사**다
#    (docker/entrypoint.sh).

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 빌드 시점 자체 점검 — GPU 없이 되는 것만.
# 여기서 깨지면 이미지를 만들지 않는다. "실행은 되는데 결과가 틀린" 상태로
# 배포되는 것보다 낫다.
RUN python3 -c "import kerneltab, pyarrow, pandas, pynvml; \
from kerneltab.backends import get_backend; \
from kerneltab.build import paths; \
from kerneltab.core.hardware import known_spec; \
assert get_backend('sm_86'); \
assert (paths.PKG_ROOT/'measure'/'kt_ctx.cu').exists(), 'measure/*.cu 누락'; \
assert known_spec('NVIDIA RTX A6000'), 'hwspec 못 읽음'; \
print('self-check ok', kerneltab.__version__)" \
 && python3 -m pytest /work/tests -q

ENTRYPOINT ["kerneltab"]
CMD ["help"]

# ---------------------------------------------------------------------------
# 사용 예 (호스트에서) — 전체 절차는 docs/container.md
# ---------------------------------------------------------------------------
# G=GPU-93284c84-335c-baab-2005-69c58e5cef15
# sudo nvidia-smi -i 3 -pm 1 && sudo nvidia-smi -i 3 -lgc 1350,1350 \
#   && sudo nvidia-smi -i 3 -lmc 8001            # 클럭은 호스트에서
#
# TAG=$(python3 scripts/manifest.py --tag)
# docker build -f docker/Dockerfile \
#     --build-arg KERNELTAB_COMMIT=$(git rev-parse HEAD) -t $TAG .
#
# V="-v $PWD/data:/data --user $(id -u):$(id -g)"
# docker run --rm --gpus "\"device=$G\"" $V $TAG \
#     detect --gpu $G --externally-locked-mhz 1350 --externally-locked-mem-mhz 7601
