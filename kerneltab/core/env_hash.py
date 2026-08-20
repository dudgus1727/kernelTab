"""`env_hash` 정의 — **측정 조건에 실제로 영향을 주는 필드만** 해싱한다 (P-3).

## 문제

구 정의는 `env` 딕셔너리 **전체**를 해싱했다. 거기엔 실행마다 변하는 값이
섞여 있다 — `created_utc`, `host.hostname`, `host.ram_available_gb`, 경로,
그리고 `launch_overhead`(측정값이라 매번 미세하게 다르다).

그 결과 **조건이 완전히 같아도 `phase0_env.py` 를 다시 돌리면 해시가 바뀐다.**
재개가 끊기고, 같은 조건의 데이터가 다른 해시로 갈라진다. 실제로 이 캠페인에서
메모리 클럭 기록을 빠뜨려 다시 돌렸을 때 해시가 바뀌었다.

## 신 정의

아래 키만 본다. 나머지는 기록은 하되 해시에서 뺀다.

### `manifest_hash` 를 제외한 이유

소스 `tree_hash` 를 포함하므로 **코드 한 글자만 바뀌어도 해시가 바뀐다.**
측정 도중 오타 수정조차 불가능해진다 — 규율이 아니라 실용성의 문제다.

> ⚠️ **대가가 있다.** `backends/sm80.py`, `measure/kt_kernel_impl.h`,
> `measure/kt_swizzle.h`, `measure/kt_ctx.cu` 를 고쳐도 `env_hash` 가
> 바뀌지 않는다. 그 파일들은 커널 생성/측정 루프의 실체이므로, 고칠 때는
> **조건이 달라졌는지 사람이 직접 판단**하고 달라졌으면 `phase0_env.py` 를
> 다시 돌려야 한다. 각 파일 docstring 상단에 이 경고가 있다.
"""

from __future__ import annotations

import hashlib
import json

__all__ = ["ENV_HASH_DEF_VERSION", "ENV_HASH_KEYS", "ENV_HASH_KEYS_V2",
           "EnvHashIncomplete", "canonical_hash", "env_hash_v2", "hash_inputs"]


class EnvHashIncomplete(ValueError):
    """해시에 반드시 들어가야 하는 값이 비어 있다."""


#: **비어 있으면 안 되는** 키. 이 값들이 `null` 이면 서로 다른 조건이 같은
#: 해시를 받는다 — 그리고 아무 오류도 나지 않는다.
#:
#: 실제로 밟았다. 컨테이너에서 `--user` 로 돌렸더니 git 이 `/opt/cutlass`
#: 를 "dubious ownership" 으로 거부했고, `cutlass.commit` 이 `null` 인
#: `env.json` 이 만들어졌다. 그 상태로 측정했으면 **CUTLASS 버전을 바꿔도
#: 같은 env_hash** 가 나온다. 격리 경계가 뚫리는 것이다.
REQUIRED_V2: tuple[str, ...] = (
    "hardware",
    "nvcc_arch_flag",
    "cutlass.commit",
    "cuda.nvcc_version",
)

#: 해시 **정의**의 버전. 키 목록이 바뀌면 반드시 올린다.
#:
#: ⚠️ 왜 필요한가: `env_hash_v2` 값은 `env.json` 과 `env_registry.jsonl` 에
#: **기록되어 남는다.** 정의를 바꾸면 그 기록들이 조용히 재현 불가능해진다 —
#: 같은 함수가 같은 입력에 다른 값을 준다. 버전을 함께 적어야 옛 값이
#: 무엇이었는지 알 수 있다.
#:
#: `tests/test_env_hash.py` 가 키 목록의 해시를 고정한다. 키를 고치고
#: 버전을 안 올리면 **테스트가 실패한다.** 문서가 아니라 코드로 강제한다.
#:
#: | 버전 | 바뀐 것 |
#: |---|---|
#: | 2 | 최초 (P-3). `env` 전체 해싱을 대체 |
#: | 3 | `cuda.driver_user_mode` 추가 (아래 참조) |
ENV_HASH_DEF_VERSION = 3

#: 해시에 들어가는 것. `(env 키, 하위 경로)` — 하위 경로는 점으로 구분한다.
ENV_HASH_KEYS_V2: tuple[str, ...] = (
    "hardware",                  # 감지된 GPU 스펙
    "nvcc_arch_flag",
    "protocol",                  # target_ms, min_reps_floor/cap, max_reps, warmup...
    "soak",                      # 소킹 on/off 와 파라미터
    "segments",                  # 세그먼트 크기 / 워밍업 초 (드리프트 대책)
    "clock_locked", "locked_mhz",
    "mem_clock_locked", "locked_mem_mhz",
    "peak_tflops_f16_effective",
    "bandwidth_gbps_effective",
    "shuffle_seed",
    "cutlass.commit",            # 커널 생성 로직의 대부분을 커버한다
    "cuda.nvcc_version",
    # --- 정의 3 에서 추가 ---------------------------------------------------
    # 유저 모드 libcuda. 컨테이너에서는 **이미지**가 이것을 고정한다
    # (CUDA forward-compat). 커널 모드 드라이버는 호스트마다 다르고 통제할 수
    # 없으므로 해시에 넣지 않는다 — 기록만 한다.
    #
    # 왜 넣는가: compat 계층이 **런치 오버헤드**에 영향을 줄 가능성이
    # 확인되지 않았다. 짧은 커널에서 런치 경로가 지배한다는 것은 이미 안다
    # (docs/measurement_drift.md). 영향이 있다면 이것은 측정 조건이다.
    "cuda.driver_user_mode",
)

#: 별칭. 코드에서는 이쪽을 쓴다 — `_V2` 라는 이름이 정의 버전과 어긋난다.
ENV_HASH_KEYS = ENV_HASH_KEYS_V2

#: 명시적으로 **제외**하는 것. 왜 뺐는지 남긴다 — 나중에 누가 다시 넣으려
#: 할 때 근거가 필요하다.
EXCLUDED_WITH_REASON = {
    "created_utc": "실행 시각. 조건이 아니다",
    "host": "호스트명/여유 RAM 등 실행마다 변한다",
    "device_index": "같은 GPU 를 다른 인덱스로 잡을 수 있다 (P-2 참조)",
    "launch_overhead": "**측정값**이라 실행마다 미세하게 다르다",
    "launch_overhead_ms": "**측정값**. 같은 조건에서도 실행마다 미세하게 다르다",
    "gpu_smi": "nvidia-smi 원문. 드라이버 문자열 등 변동",
    "cutlass_example_check": "빌드 확인 결과. 조건이 아니다",
    "clock_lock_check": "부하 검증 **측정값**",
    "manifest_hash": "소스 tree_hash 포함 -> 한 글자만 고쳐도 바뀐다",
    "phase": "진행 단계 표시",
    "schema_version": "파일 형식 버전",
}


def _dig(env: dict, path: str):
    cur = env
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def hash_inputs(env: dict) -> dict:
    """해시에 들어가는 값만 뽑아 돌려준다. 진단/설명용."""
    return {k: _dig(env, k) for k in ENV_HASH_KEYS}


def canonical_hash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def env_hash_v2(env: dict) -> str:
    """조건이 같으면 **몇 번을 다시 계산해도 같은 값**이 나와야 한다.

    필수 키가 비어 있으면 `EnvHashIncomplete` 다. 조용히 계산해 주면 서로
    다른 조건이 같은 해시를 받는다 — `env_hash` 는 조인 키가 아니라 **격리
    경계**이므로 그 순간 모든 보호가 무의미해진다.
    """
    inputs = hash_inputs(env)
    missing = [k for k in REQUIRED_V2 if not inputs.get(k)]
    if missing:
        raise EnvHashIncomplete(
            f"env_hash 에 필요한 값이 비어 있다: {missing}\n"
            "  이 상태로 해시를 만들면 다른 조건이 같은 해시를 받는다.\n"
            "  cutlass.commit 이면: .git 을 읽을 수 없는 환경이다 "
            "(컨테이너에서 흔하다). phase0_env.py --cutlass-commit 으로 "
            "주입하거나 CUTLASS_COMMIT 환경변수를 설정하라.")
    return canonical_hash(inputs)
