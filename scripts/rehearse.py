#!/usr/bin/env python3
"""Phase 2-2 리허설 / Phase 3 전수 측정.

    python3 scripts/rehearse.py                    # 리허설 (6형상 x 표본 20커널)
    python3 scripts/rehearse.py --all              # Phase 3 전수
    python3 scripts/rehearse.py --all --dry-run    # 규모만 세어본다

측정 순서는 셔플한다 (온도 드리프트가 config 순서와 상관되는 것을 막는다).

results.jsonl 은 append-only 이고 resume 키는
`(env_hash, kernel_id, M, N, K, split_k, split_k_mode)` 다.
env_hash 를 키에 넣는 이유: 클럭 고정 전후처럼 측정 조건이 바뀌면 같은 조합도
다시 재야 한다. 넣지 않으면 조건이 다른 예전 줄 때문에 건너뛰어 버린다.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kerneltab.backends import get_backend
from kerneltab.core import (
    device,
    paths,
    records,
)
from kerneltab.core import kernels as kernels_mod
from kerneltab.core.config import alignments_for, enumerate_runtimes
from kerneltab.core.hardware import hardware_from_env
from kerneltab.core.shapes import all_shapes
from kerneltab.core.types import KernelConfig, Problem, RuntimeConfig
from kerneltab.measure.runner import SOAK_DEFAULTS  # noqa: F401

RESULTS = paths.RESULTS_DIR / "results.jsonl"
DRIFT = paths.RESULTS_DIR / "drift.jsonl"
TELEMETRY = paths.RESULTS_DIR / "telemetry.csv"
KERNELS = paths.RESULTS_DIR / "kernels.jsonl"
#: D-4. 진행 상태 하트비트. 감시자는 pgrep 이 아니라 이 파일을 본다.
#: pgrep -f "rehearse.py --all" 은 **감시자 자신의 명령줄에도 그 문자열이
#: 들어 있어 자기 자신을 찾아낸다.** 2026-08-16 에 이것 때문에 측정이 죽은 뒤
#: 13 시간 동안 알아채지 못했다.
HEARTBEAT = paths.RESULTS_DIR / "heartbeat.json"
HEARTBEAT_SECONDS = 600

# 리허설 형상 (지시된 6개)
REHEARSAL_SHAPES = [
    Problem(64, 4096, 4096),     # 메모리 바운드, split-K 극단
    Problem(1024, 1024, 4096),   # split-K=6 가설 검증 대상
    Problem(1024, 4096, 512),    # K 작음. is_valid_runtime / warp_k 검증
    Problem(1024, 4096, 4100),   # alignment=4 경로
    Problem(4096, 4096, 4096),   # compute 바운드, cuBLAS 대조 기준
    Problem(8192, 4096, 4096),   # 대형
]

#: 드리프트 감시 프로브.
#:
#: ⚠️ **큰 형상 하나로 감시하면 안 된다.** Phase 3 은 4096³ 하나(2.85 ms)로만
#: 감시해서 드리프트를 +5.06 % 로 봤다. 같은 조건에서 512³ 측정은 **+1380 %**
#: 오염되어 있었고 감시에는 전혀 잡히지 않았다.
#:
#: 원인은 누적 모듈에 비례해 커지는 **런치당 상수 오버헤드**다 (모듈 1,000 개당
#: 약 42 us). 상수이므로 긴 커널에서는 안 보이고 짧은 커널을 통째로 삼킨다.
#: 그래서 감시 프로브에 **반드시 작은 형상**이 있어야 한다.
#: docs/measurement_drift.md
DRIFT_SHAPES = [
    Problem(512, 512, 512),      # ~15 us. 런치 오버헤드에 가장 민감
    Problem(4096, 4096, 4096),   # ~2.8 ms. 계산 처리율 쪽
]
DRIFT_SHAPE = DRIFT_SHAPES[-1]   # 하위 호환

#: max_rel_error 가 이 값을 넘으면 numerical_fail.
#: 입력이 1/4 배수라 fp32 누산이 정확하므로 split-K 가 없는 경우 0 에 가깝다.
#: split-K 는 부분합을 fp16 으로 저장하므로 슬라이스 수에 비례해 커진다.
NUMERICAL_TOL = 5e-2

#: outlier_frac 이 이 값을 넘으면 status 에 표시
OUTLIER_TOL = 0.20

# --- 장시간 실행 감시 임계 (요청 O) ---------------------------------------
# --- D-2. 드리프트 판정 (이동 기준) --------------------------------------
#
# "첫 측정 대비" 는 완만한 예열과 급격한 조건 변화를 구분하지 못한다.
# 최근 구간의 이동 중앙값을 기준으로 삼으면 예열은 통과하고 클럭 풀림 /
# 다른 프로세스 난입 / 하드웨어 문제만 잡힌다.
#: 이동 기준 = 최근 이 횟수의 드리프트 측정 중앙값 (10분 주기면 약 1시간)
DRIFT_WINDOW = 6
#: 이동 기준 대비 이만큼 벗어나면 위반 1회
DRIFT_TOL = 0.05
#: 위반이 연속 이 횟수면 중단
DRIFT_STRIKES = 3
#: 이동 기준과 별개로, 소킹 직후 기준값 대비 이 값을 넘으면 경고.
#: 아주 느린 드리프트가 누적되는 것을 놓치지 않기 위함이다.
DRIFT_ABS_WARN = 0.08
#: 최근 구간에서 sw_power_cap 이 이 비율을 넘으면 즉시 경고
#: (클럭 고정이 풀렸거나 다른 프로세스가 GPU 를 쓰고 있다)
POWER_CAP_TOL = 0.05
#: status 분포를 이 주기로 로그에 남긴다
STATUS_LOG_SECONDS = 3600

# --- D-1. 열평형 소킹 ---------------------------------------------------
#
# 2026-08-16 전수 측정이 7.5 시간 만에 "드리프트 5% 3 연속" 으로 중단됐다.
# 원인은 조건 변화가 아니라 **열 포화**였다. SM 클럭 1350 고정, 메모리 7601,
# 스로틀 0%, 온도 56~64°C 안정인데도 기준 config 가 서서히 느려졌다.
#
#     0.0h +0.00%   0.5h +0.90%   1.0h +2.05%   2.0h +3.34%
#     3.0h +4.15%   5.0h +4.91%   7.5h +5.06%   <- 5h 부근에서 포화
#
# 13 시간 유휴(46°C) 후 재측정하면 +0.29% 로 완전히 돌아온다 — 가역적이다.
# `temperature.gpu` 는 5 분이면 69°C 로 평형에 도달하지만 **성능은 몇 시간에
# 걸쳐 계속 느려진다.** 즉 코어 온도로는 평형을 판정할 수 없고, 기준 config 를
# 직접 재서 판정해야 한다 (메모리/기판 쪽의 느린 열용량으로 추정).
#
# 그래서 본 측정 전에 실부하로 소킹하고, **드리프트 측정 자체로** 평형을
# 판정한다.
#
# ── 후속 (2026-08-17): **열 가설은 기각됐다.** ──────────────────────────────
# 위 서술은 관찰이 맞고 해석이 틀렸다. 대조 실험에서 69°C / 230 W (Phase 3 보다
# 뜨겁다) 로 30 분 소킹한 뒤 재측정하니 드리프트가 **-0.07%** 였다. 열이
# 원인이면 소킹 후에도 같은 상승이 나와야 했다. 나오지 않았다.
# (내가 세운 가설이고 D-1 까지 구현한 뒤 스스로 기각했다. 자세한 것은
#  docs/measurement_drift.md.)
#
# 그래서 SOAK_ENABLED 의 기본값은 False 다. 45 분을 쓰고 얻는 것이 -0.07%
# 라면 쓸 이유가 없다. 코드를 지우지 않는 이유는 **다른 GPU 에서는 다를 수
# 있기 때문**이다 — 클라우드 인스턴스는 냉각도 전력 한계도 다르다. 새 GPU
# 에서는 docs/post_measurement.md 10 절의 드리프트 프로파일을 먼저 돌리고,
# 열이 실제 원인으로 나오면 `--soak` 로 켠다.
#: ── 드리프트 대책: 시간 분할 세그먼트 ──────────────────────────────────
#: 드리프트의 원인은 **그 프로세스가 지금까지 실행한 서로 다른 커널의 수**다.
#: 프로세스를 다시 띄우면 완전히 리셋된다. 그래서 커널을 세그먼트로 나누고
#: 세그먼트마다 새 프로세스로 돈다.
#:
#: 순차로 돌면 안 된다 — 세그먼트 0 의 커널은 전부 초반에, 마지막 세그먼트는
#: 전부 후반에 측정되어 **세그먼트 인덱스가 시각과 완전히 상관**된다. 그건
#: 전역 셔플이 막으려던 바로 그 편향이다. 그래서 라운드 로빈으로 돈다:
#: 세그먼트마다 SEGMENT_SECONDS 만큼만 돌고 다음 세그먼트로 넘어간다.
#:
#: scripts/sweep.py 가 이 순회를 관리한다.
SEGMENT_KERNELS = 500          # 프로세스당 로드할 서로 다른 커널 수 상한
SEGMENT_SECONDS = 2700         # 한 슬라이스의 시간 상한 (초)
ANCHOR_KERNELS = 6             # 모든 세그먼트에서 재는 고정 커널
ANCHOR_REPS = 3                # 앵커는 3회 중앙값 (1% 판정에 노이즈가 크다)
#: 앵커를 재는 형상은 DRIFT_SHAPES 를 그대로 쓴다 (짧은 것 + 긴 것).

SOAK_ENABLED = False           # --soak / --no-soak, env.json 의 soak.enabled
SOAK_PROBE_INTERVAL = 300      # 5분마다 기준 config 재측정
SOAK_STABLE_SPAN = 0.003       # 최근 3회(10분 구간)의 (max-min)/median
SOAK_STABLE_RUNS = 3
SOAK_MIN_SECONDS = 45 * 60
SOAK_MAX_SECONDS = 180 * 60    # --soak-max-min 으로 조정

#: 측정 시작 전 GPU 를 달구는 시간(초).
#: `nvidia-smi -lgc` 는 SM 클럭만 고정한다. **메모리 클럭은 고정되지 않으며**
#: 유휴 시 810 MHz(P5)까지 떨어졌다가 부하가 걸려야 7601 MHz 로 올라간다
#: (9.4배 차이). 램프업 전에 측정하면 메모리 바운드 config 가 최대 66% 느리게
#: 측정된다 — 실측으로 확인했다. 연속 측정 중에는 문제가 없지만 시작 직후와
#: 중단 후 재개 직후가 위험하다.
#: 메모리 클럭까지 고정하려면 `sudo nvidia-smi -i N -lmc <mhz>` 가 필요하다.
WARMUP_SECONDS = 20

#: NVML 로 읽은 메모리 클럭이 최대치의 이 비율 미만이면 램프업 중으로 본다
MEM_CLOCK_MIN_FRAC = 0.9


# launchable 은 core/kernels.py 에 있다. 여섯 곳에 흩어져 있었고 결측 처리가
# 서로 달랐다 (docs/decisions.md 13).
launchable = kernels_mod.launchable


THROTTLE_BITS = {
    0x0004: "sw_power_cap",
    0x0008: "hw_slowdown",
    0x0020: "sw_thermal",
    0x0040: "hw_thermal",
    0x0080: "hw_power_brake",
}


# ---------------------------------------------------------------------------
# 커널 표본 — 층화 표집 (완전 무작위는 개수 많은 구성에 쏠린다)
# ---------------------------------------------------------------------------
def stratified_sample(rows: list[dict], n: int, seed: int,
                      regs_per_sm: int) -> list[dict]:
    """검증하려는 코드 경로가 표본에서 빠지지 않도록 층을 먼저 채운다."""
    rng = random.Random(seed)
    rows = [r for r in rows if launchable(r, regs_per_sm)]  # 런치 불가는 제외
    pool = sorted(rows, key=lambda r: r["kernel_id"])
    rng.shuffle(pool)

    picked: list[dict] = []
    taken: set[str] = set()

    def take(pred, k, label):
        got = 0
        for r in pool:
            if got >= k:
                break
            if r["kernel_id"] in taken or not pred(r):
                continue
            taken.add(r["kernel_id"])
            r = dict(r)
            r["stratum"] = label
            picked.append(r)
            got += 1
        return got

    for t in (128, 256, 512, 1024):
        take(lambda r, t=t: r.get("threads") == t, 3, f"threads{t}")
    take(lambda r: r["ext"]["warp_k"] < r["tile"]["k"], 3, "warp_k_split")
    take(lambda r: (r["tile"]["m"], r["tile"]["n"]) == (256, 64), 1, "asym_256x64")
    take(lambda r: (r["tile"]["m"], r["tile"]["n"]) == (64, 256), 1, "asym_64x256")

    for r in pool:  # 나머지는 무작위
        if len(picked) >= n:
            break
        if r["kernel_id"] in taken:
            continue
        taken.add(r["kernel_id"])
        r = dict(r)
        r["stratum"] = "random"
        picked.append(r)
    return picked[:n] if len(picked) > n else picked


def cfg_from_row(row: dict, backend) -> KernelConfig:
    """kernels.jsonl 의 한 줄 -> KernelConfig. ext 복원은 백엔드에 위임한다."""
    a, t = row["align"], row["tile"]
    return KernelConfig(
        tile_m=t["m"], tile_n=t["n"], tile_k=t["k"],
        align_a=a["a"], align_b=a["b"], align_c=a["c"],
        arch=row["arch"], ext=backend.ext_from_dict(row["ext"]),
    )


def config_key(row: dict) -> tuple:
    """alignment 를 뺀 커널 구성. a888 <-> a448 대응짝을 찾는 데 쓴다."""
    t, e = row["tile"], row["ext"]
    return (t["m"], t["n"], t["k"], e["warp_m"], e["warp_n"], e["warp_k"],
            e["stages"], e["swizzle_type"], e["swizzle_n"])


# ---------------------------------------------------------------------------
# 텔레메트리
# ---------------------------------------------------------------------------
def start_telemetry(target: str):
    """`nvidia-smi` 를 1 초 간격으로 띄운다. `target` 은 **UUID** 다 (P-2).

    ⚠️ 예전에는 `Popen` 이 실패해도 아무 일도 일어나지 않았다 — 빈 CSV 만
    남고 측정은 그대로 진행됐다. 33 시간을 돌고 나서야 "텔레메트리가
    비어 있네" 를 알게 된다. `docs/decisions.md` 14번의 "조용히 아무것도
    안 하는 안전장치" 패턴이라 **실제로 줄이 쌓이는지 확인**한다.

    인덱스가 아니라 UUID 로 지정한다. 컨테이너나 `CUDA_VISIBLE_DEVICES`
    재배치 상황에서 인덱스는 **다른 GPU** 를 가리킨다.
    """
    TELEMETRY.parent.mkdir(parents=True, exist_ok=True)
    f = TELEMETRY.open("w")
    cmd = ["nvidia-smi", "-i", str(target),
           "--query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,"
           "power.draw,clocks_throttle_reasons.active",
           "--format=csv", "-l", "1"]
    try:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.PIPE)
    except OSError as e:
        f.close()
        raise RuntimeError(
            f"텔레메트리를 띄우지 못했다: {e}\n  {' '.join(cmd)}") from e

    # 즉시 죽었는지 / 실제로 기록되는지 확인한다. 헤더 + 첫 줄이면 충분하다.
    for _ in range(30):
        time.sleep(0.2)
        if p.poll() is not None:
            err = (p.stderr.read() or b"").decode("utf-8", "replace").strip()
            f.close()
            raise RuntimeError(
                f"텔레메트리가 즉시 종료됐다 (rc={p.returncode})\n"
                f"  {' '.join(cmd)}\n  {err}\n"
                "  -i 에 넘긴 GPU 를 이 기계에서 볼 수 없을 수 있다.")
        if TELEMETRY.exists() and TELEMETRY.stat().st_size > 0:
            return p, f
    p.terminate()
    f.close()
    raise RuntimeError(
        f"텔레메트리가 6초 동안 한 줄도 쓰지 않았다.\n  {' '.join(cmd)}\n"
        "  빈 CSV 를 남긴 채 33시간을 돌면 조건 검증이 불가능해진다.")


def analyze_telemetry() -> dict:
    if not TELEMETRY.exists():
        return {}
    clocks, temps, powers = [], [], []
    throttle = Counter()
    n = 0
    for line in TELEMETRY.read_text().splitlines()[1:]:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 6:
            continue
        n += 1
        try:
            clocks.append(int(parts[1].split()[0]))
            temps.append(int(parts[3]))
            powers.append(float(parts[4].split()[0]))
            bits = int(parts[5], 16)
        except (ValueError, IndexError):
            continue
        for mask, name in THROTTLE_BITS.items():
            if bits & mask:
                throttle[name] += 1
    if not clocks:
        return {"samples": n}
    return {
        "samples": n,
        "sm_clock_min": min(clocks), "sm_clock_max": max(clocks),
        "sm_clock_mean": round(sum(clocks) / len(clocks), 1),
        "temp_min": min(temps), "temp_max": max(temps),
        "power_min": round(min(powers), 1), "power_max": round(max(powers), 1),
        "throttle_seconds": dict(throttle),
    }


# ---------------------------------------------------------------------------
def split_segments(kernel_ids, seg_size: int, seed: int):
    """커널을 세그먼트로 나눈다. -> {kernel_id: 세그먼트 번호}, 세그먼트 수

    **무작위 분할**이다. 커널 id 순으로 자르면 세그먼트가 tile 크기 같은
    config 축과 상관되고, 세그먼트 간 오차가 그대로 config 편향이 된다.
    씨앗을 고정하므로 재개해도 같은 분할이 나온다 — 이게 깨지면 이미 측정한
    커널이 다른 세그먼트로 가서 재개가 어긋난다.
    """
    ids = sorted(kernel_ids)                    # 입력 순서에 의존하지 않도록
    random.Random(seed ^ 0x53454704).shuffle(ids)   # "SEG"
    seg = {kid: i // seg_size for i, kid in enumerate(ids)}
    n_seg = (len(ids) + seg_size - 1) // seg_size
    return seg, n_seg


def pick_anchors(rows, n: int, seed: int):
    """모든 세그먼트에서 재는 고정 커널.

    세그먼트마다 프로세스가 다르므로 세그먼트 사이에 계통 오차가 있을 수
    있다. 앵커를 매 세그먼트에서 재면 그 오차를 사후에 잴 수 있다.

    ⚠️ **반드시 짧은 커널을 넣어야 한다.** 런치 오버헤드 드리프트는 상수라
    긴 커널에서는 안 보인다. 큰 커널만 앵커로 쓰면 이번에 감시가 놓친 것과
    같은 실수를 반복한다.
    """
    by_time_proxy = sorted(rows, key=lambda r: (r["tile"]["m"] * r["tile"]["n"]))
    if not by_time_proxy:
        return []
    picks = []
    # 작은 타일 / 중간 / 큰 타일에서 골고루
    k = len(by_time_proxy)
    rng = random.Random(seed ^ 0x414E4348)
    for lo, hi in ((0, k // 3), (k // 3, 2 * k // 3), (2 * k // 3, k)):
        band = by_time_proxy[lo:hi]
        if band:
            picks += rng.sample(band, min(max(n // 3, 1), len(band)))
    return picks[:n]


ANCHORS = paths.RESULTS_DIR / "anchors.jsonl"


def measure_anchors(ctx, kernels, anchor_rows, probe, env, segment, when,
                    rnd=None):
    """세그먼트마다 같은 커널을 재서 세그먼트 간 오차를 사후에 잴 수 있게 한다.

    ⚠️ `round` 를 반드시 남긴다 (R-6). `sweep.py` 의 실행 단위는 세그먼트가
    아니라 **슬라이스 = (라운드, 세그먼트)** 이고, 같은 세그먼트가 라운드마다
    다시 돈다. 예전에는 `segment` 와 `when` 만 적어서, 사후에 라운드를
    복원하려면 `(segment, when) -> round` 로 매핑할 수밖에 없었고 **라운드마다
    덮어써서 마지막 하나만 남았다.** 그 결과 "라운드에 따라 전체가 함께
    드리프트하는가" 라는 검사가 통째로 죽어 있었다 — 8 라운드를 돌고도
    "라운드가 2개 미만이라 비교할 수 없다" 가 찍혔다.

    `results.jsonl` 이 아니라 별도 파일에 쓴다 — 같은 (커널, 형상) 을 여러 번
    재므로 무결성 검사의 중복 판정에 걸린다. 앵커는 측정 표의 일부가 아니라
    **측정 조건의 기록**이다.
    """
    if not anchor_rows:
        return
    snap = probe.snapshot()
    with ANCHORS.open("a") as f:
        for r in anchor_rows:
            kern = kernels.get(r["kernel_id"])
            if kern is None:
                continue
            for p in DRIFT_SHAPES:
                # 3회 중앙값. 앵커의 판정 기준이 1% 인데 512³ 짜리는 한 번
                # 재면 노이즈가 그보다 크다 — 그러면 "대책이 실패했다" 와
                # "짧은 커널이라 원래 시끄럽다" 를 구분할 수 없다.
                ms = [_probe_shape(ctx, kern, p) for _ in range(ANCHOR_REPS)]
                ms = [x for x in ms if x is not None]
                if not ms:
                    continue
                m = sorted(ms, key=lambda x: x.time_ms)[len(ms) // 2]
                f.write(json.dumps({
                    "kernel_id": r["kernel_id"],
                    "problem": {"M": p.M, "N": p.N, "K": p.K},
                    "time_ms": m.time_ms, "time_std_ms": m.time_std_ms,
                    "n_reps": m.n_reps, "anchor_reps": len(ms),
                    "time_spread_pct": round(
                        100 * (ms[-1].time_ms - ms[0].time_ms)
                        / max(m.time_ms, 1e-9), 3) if len(ms) > 1 else 0.0,
                    "segment": segment, "when": when, "round": rnd,
                    "sm_clock_mhz": snap["sm_clock_mhz"],
                    "gpu_temp_c": snap["gpu_temp_c"],
                    "env_hash": env["env_hash"],
                    "timestamp": datetime.now(timezone.utc)
                    .isoformat().replace("+00:00", "Z"),
                }, ensure_ascii=False) + "\n")


def load_done(env_hash: str) -> set[tuple]:
    """이미 측정된 조합. **같은 측정 조건(env_hash)의 줄만** 완료로 본다."""
    if not RESULTS.exists():
        return set()
    done = set()
    for d in records.iter_records(RESULTS, env_hash):
        try:
            p, r = d["problem"], d["runtime"]
        except KeyError:
            continue
        done.add((d["kernel_id"], p["M"], p["N"], p["K"],
                  r["split_k"], r["split_k_mode"]))
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-kernels", type=int, default=20)
    ap.add_argument("--all", action="store_true",
                    help="Phase 3: 전체 형상 그리드 x 빌드된 커널 전부")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--soak-max-min", type=int, default=0,
                    help="열평형 소킹 최대 시간(분). 평형 판정이 먼저 되면 그때 끝난다")
    ap.add_argument("--soak", dest="soak", action="store_true", default=None,
                    help="열평형 소킹을 켠다. A6000 에서는 이득이 없었다"
                         "(-0.07%%) — 새 GPU 에서 열이 원인으로 확인됐을 때만")
    ap.add_argument("--no-soak", dest="soak", action="store_false",
                    help="소킹을 끈다 (기본)")
    ap.add_argument("--round", type=int, default=None, dest="round_",
                    help="sweep.py 가 넘기는 라운드 번호. 앵커 줄에 기록해 "
                         "사후에 라운드별 추이를 볼 수 있게 한다 (R-6)")
    ap.add_argument("--segment", type=int, default=None,
                    help="이 세그먼트의 커널만 측정한다. scripts/sweep.py 가 "
                         "라운드 로빈으로 넘겨준다")
    ap.add_argument("--segment-kernels", type=int, default=0,
                    help=f"세그먼트당 커널 수 (기본 env 또는 {SEGMENT_KERNELS})")
    ap.add_argument("--time-budget", type=float, default=SEGMENT_SECONDS,
                    help="이 시간(초)이 지나면 깔끔하게 멈춘다. 종료 코드 7. "
                         "병리적으로 느린 세그먼트가 라운드를 막지 않도록 하는 "
                         "안전장치이며, 진행 배분은 --max-jobs 로 한다")
    ap.add_argument("--max-jobs", type=int, default=0,
                    help="이 개수만 측정하고 멈춘다. 종료 코드 7. "
                         "세그먼트마다 커널 실행 시간 분포가 달라 시간 고정은 "
                         "진행률이 어긋난다 — 작업 수로 배분하는 쪽이 맞다")
    ap.add_argument("--list-segments", action="store_true",
                    help="세그먼트 분할만 출력하고 끝낸다")
    ap.add_argument("--json", action="store_true",
                    help="--list-segments 결과를 기계용 JSON 으로도 낸다 "
                         "(`JSON {...}` 한 줄). sweep.py 가 이것만 쓴다")
    args = ap.parse_args()

    env = json.loads(paths.ENV_JSON.read_text())
    # P-2: UUID 가 권위다. 저장된 인덱스를 그대로 쓰면 컨테이너나
    #      CUDA_VISIBLE_DEVICES 가 설정된 환경에서 **다른 GPU 를 측정**한다.
    #      이미 설정돼 있으면 존중하고, 없는 UUID 면 명확히 실패한다.
    device.resolve_device(env)
    hw = hardware_from_env(env)
    backend = get_backend(hw.arch)
    seed = env["shuffle_seed"]
    launch_overhead_ms = env["launch_overhead_ms"]
    drift_period = env["drift_check_seconds"]
    clock_locked = env["clock_locked"]

    from kerneltab.measure.gpu_state import NvmlProbe
    from kerneltab.measure.runner import Ctx, Kernel

    # --- 빌드된 커널 로드 ---------------------------------------------------
    rows = [json.loads(l) for l in KERNELS.read_text().splitlines() if l.strip()]
    ok_rows = [r for r in rows if r.get("build_status") == "ok"]
    by_align = defaultdict(list)
    for r in ok_rows:
        a = r["align"]
        by_align[(a["a"], a["b"], a["c"])].append(r)

    if args.all:
        # Phase 3: 층화 표집을 하지 않고 전부 쓴다. 단 두 가지는 뺀다.
        #   1) 런치 불가 (regs * threads > regs_per_sm)
        #   2) 현재 열거기 기준으로 더 이상 유효하지 않은 커널.
        #      kernels.jsonl 은 append-only 라 과거 열거 공간에서 빌드된 것이
        #      남아 있다 (예: 계산이 틀리는 warp tile (64,128)). 측정 대상은
        #      항상 "지금의 is_valid_kernel 이 인정하는 집합" 이어야 한다.
        from kerneltab.core.config import alignment_combos, enumerate_kernels

        valid_ids = {backend.kernel_id(c) for c in
                     enumerate_kernels(hw, backend,
                                       alignment_combos(all_shapes(hw)))}
        launch_ok = [r for r in ok_rows if launchable(r, hw.regs_per_sm)]
        sample = [r for r in launch_ok if r["kernel_id"] in valid_ids]
        picked = sample
        partners = []
        shapes = all_shapes(hw)
        print(f"[Phase 3 전수] 빌드 {len(ok_rows)}개 -> 런치 가능 {len(launch_ok)}개 "
              f"-> 현재 유효 {len(sample)}개, 형상 {len(shapes)}개")
        dropped = len(launch_ok) - len(sample)
        if dropped:
            print(f"  (열거기에서 제외된 커널 {dropped}개는 측정하지 않는다)")
    else:
        shapes = REHEARSAL_SHAPES
        base = by_align.get((8, 8, 8), [])
        picked = stratified_sample(base, args.n_kernels, seed, hw.regs_per_sm)
        # a448 대응짝: 동일 구성의 alignment 변형. 이러면 (1024,4096,4100)
        # 형상에서 alignment 효과만 분리해서 볼 수 있다.
        a448_by_key = {config_key(r): r for r in by_align.get((4, 4, 8), [])}
        partners = []
        for r in picked:
            q = a448_by_key.get(config_key(r))
            if q:
                q = dict(q)
                q["stratum"] = r["stratum"] + "/a448"
                partners.append(q)
        sample = picked + partners
        print(f"커널 표본: a888 {len(picked)}개 + a448 대응짝 {len(partners)}개")
        for r in picked:
            print(f"  [{r['stratum']:14s}] {r['kernel_id']}  "
                  f"thr={r['threads']} regs={r['regs_per_thread']} "
                  f"blk/sm={r['max_blocks_per_sm']}")

    # --- 세그먼트 분할 (드리프트 대책) --------------------------------------
    # 한 프로세스가 로드하는 서로 다른 커널 수를 SEGMENT_KERNELS 로 묶는다.
    # 이게 드리프트의 유일한 설명 변수다 (docs/measurement_drift.md).
    seg_map = None
    n_seg = 1
    anchors = []
    if args.all:
        seg_size = (args.segment_kernels
                    or (env.get("segments") or {}).get("kernels")
                    or SEGMENT_KERNELS)
        seg_map, n_seg = split_segments(
            [r["kernel_id"] for r in sample], seg_size, seed)
        anchors = pick_anchors(sample,
                               (env.get("segments") or {}).get(
                                   "anchor_kernels", ANCHOR_KERNELS), seed)
        anchor_ids = {r["kernel_id"] for r in anchors}
        if args.list_segments:
            print(f"세그먼트 {n_seg}개 x 커널 {seg_size}개 "
                  f"(전체 {len(sample)}개)")
            # 세그먼트별 작업 수. 무작위 분할이라 커널 수는 같지만 형상
            # alignment 와 런타임 조합 수가 달라 작업 수는 1.8 배까지 갈린다.
            # 균등 슬라이스로 돌면 가벼운 세그먼트가 먼저 끝나고 후반
            # 라운드에 무거운 것만 남아 시각 상관이 되살아난다. sweep.py 가
            # 이 수로 세그먼트마다 슬라이스를 비례 배분한다.
            _cnt = Counter()
            for r in sample:
                _a = (r["align"]["a"], r["align"]["b"], r["align"]["c"])
                _cfg = cfg_from_row(r, backend)
                for _p in shapes:
                    if alignments_for(_p) != _a:
                        continue
                    _cnt[seg_map[r["kernel_id"]]] += len(
                        enumerate_runtimes(backend, _p, _cfg))
            for _s in sorted(_cnt):
                print(f"SEGJOBS {_s} {_cnt[_s]}")
            if args.json:
                # R-2: 기계용 인터페이스. sweep.py 는 **이것만** 쓴다.
                # 사람용 출력을 파싱하면 문구를 다듬는 순간 33시간 스윕의
                # 진입점이 깨진다.
                import json as _json
                _payload = {
                    "n_segments": n_seg,
                    "n_jobs": sum(_cnt.values()),
                    "segment_kernels": seg_size,
                    "jobs_per_segment": {str(k): v for k, v in sorted(_cnt.items())},
                    "anchors": [r["kernel_id"] for r in anchors],
                    # 조건이 어긋난 채 스윕이 시작되는 것을 막는다.
                    # v2 는 조건 자체, 구 해시는 **재개 키**다. 어느 쪽이
                    # 달라도 데이터가 갈라진다 (P-3).
                    "env_hash": env["env_hash"],
                    "env_hash_v2": env.get("env_hash_v2"),
                }
                print("JSON " + _json.dumps(_payload, ensure_ascii=False))
            print(f"앵커 {len(anchors)}개: "
                  + ", ".join(r["kernel_id"] for r in anchors))
            # 여기서 끝내지 않는다 — sweep.py 는 세그먼트 수와 **전체 작업 수**
            # 를 같이 읽어 슬라이스 크기를 정한다. 작업 수는 아래에서 출력된다.
        if args.segment is not None:
            if not 0 <= args.segment < n_seg:
                print(f"세그먼트 번호는 0~{n_seg - 1} 이다")
                return 2
            keep = {k for k, v in seg_map.items() if v == args.segment}
            before = len(sample)
            sample = [r for r in sample if r["kernel_id"] in keep]
            print(f"[세그먼트 {args.segment}/{n_seg - 1}] "
                  f"커널 {before} -> {len(sample)}개 "
                  f"(앵커 {len(anchor_ids - keep)}개 별도)")

    # --- 작업 목록 ---------------------------------------------------------
    jobs = []
    for r in sample:
        a = (r["align"]["a"], r["align"]["b"], r["align"]["c"])
        for p in shapes:
            if alignments_for(p) != a:
                continue
            cfg = cfg_from_row(r, backend)
            for rc in enumerate_runtimes(backend, p, cfg):
                jobs.append((r, p, rc))

    print(f"\n작업 수: {len(jobs):,} "
          f"(형상 {len(shapes)}개, 커널 {len(sample)}개)")
    by_shape = Counter((p.M, p.N, p.K) for _, p, _ in jobs)
    for k, v in sorted(by_shape.items()):
        print(f"  {k}: {v}")
    if args.dry_run or args.list_segments:
        return 0

    rng = random.Random(seed)
    rng.shuffle(jobs)  # ★ 온도 드리프트가 config 순서와 상관되지 않도록
    if args.limit:
        jobs = jobs[: args.limit]

    done = load_done(env["env_hash"])
    n_before = len(jobs)
    jobs = [j for j in jobs
            if (j[0]["kernel_id"], j[1].M, j[1].N, j[1].K,
                j[2].split_k, j[2].split_k_mode) not in done]
    # len(done) 에는 cuBLAS 참조 줄도 섞여 있으므로 실제로 건너뛴 수를 센다.
    print(f"남은 작업: {len(jobs):,} "
          f"(이미 측정 {n_before - len(jobs):,}, 기록된 줄 {len(done):,})")
    if not jobs:
        return 0

    # --- 초기화 ------------------------------------------------------------
    ctx = Ctx(paths.ARTIFACT_DIR / "libkt_ctx.so", 0)
    ctx.set_protocol(env)
    kernels: dict[str, Kernel] = {}
    for r in sample:
        kernels[r["kernel_id"]] = Kernel(paths.kernel_so(r["kernel_id"]))
    # 앵커는 이 세그먼트에 속하지 않아도 열어야 한다. 개수가 작아(6개)
    # 모듈 압력에 실질적 영향이 없다.
    for r in anchors:
        if r["kernel_id"] not in kernels:
            kernels[r["kernel_id"]] = Kernel(paths.kernel_so(r["kernel_id"]))
    probe = NvmlProbe(uuid=env["hardware_extra"]["uuid"], index=0)

    # 드리프트 감시 커널은 **모든 세그먼트에서 같아야** 한다. 세그먼트마다
    # 다른 커널을 쓰면 drift.jsonl 을 세그먼트 사이에서 비교할 수 없다.
    # 그리고 picked 는 세그먼트 필터 **전**의 목록이라 그대로 쓰면 이 세그먼트에
    # 없는 커널을 가리켜 KeyError 가 난다 (3시간 검증에서 잡혔다).
    # 앵커는 모든 세그먼트에서 로드되므로 그 중 하나를 쓴다.
    if anchors:
        drift_kernel_id = anchors[0]["kernel_id"]
    else:
        drift_kernel_id = picked[0]["kernel_id"]
    if drift_kernel_id not in kernels:
        drift_kernel_id = sample[0]["kernel_id"]
    tele_proc, tele_file = start_telemetry(device.smi_target(env))
    print(f"[telemetry] {TELEMETRY} (1초 간격)")
    # --- D-1 열평형 소킹. 여기부터 시간을 잰다 (D-3 의 soak_elapsed_s) --------
    _soak_cfg = env.get("soak") or {}
    _soak_on = (args.soak if args.soak is not None
                else _soak_cfg.get("enabled", SOAK_ENABLED))
    if _soak_on:
        soak_info = thermal_soak(
            ctx, kernels, sample, probe, drift_kernel_id,
            max_seconds=(args.soak_max_min * 60 if args.soak_max_min
                         else _soak_cfg.get("max_seconds", SOAK_MAX_SECONDS)),
            min_seconds=_soak_cfg.get("min_seconds", SOAK_MIN_SECONDS),
            interval=_soak_cfg.get("probe_interval_s", SOAK_PROBE_INTERVAL),
            span_tol=_soak_cfg.get("stable_span", SOAK_STABLE_SPAN),
            runs=_soak_cfg.get("stable_runs", SOAK_STABLE_RUNS))
    else:
        # 소킹을 건너뛰어도 워밍업은 반드시 한다 — 메모리 클럭이 유휴 시
        # 810 MHz 까지 떨어지고, 램프업 전에 재면 최대 66% 느리게 나온다.
        # WARMUP_SECONDS 는 정의만 되어 있고 쓰이지 않았다 — 워밍업이
        # thermal_soak 안에만 있어서 소킹을 끄면 **워밍업이 통째로 사라졌다.**
        # 세그먼트 방식은 프로세스를 104 번 새로 띄우므로 매번 냉시작 측정이
        # 섞인다. 3시간 검증에서 첫 슬라이스 앵커가 9.75% 튄 것이 이것이다.
        _warm_s = (env.get("segments") or {}).get(
            "warmup_seconds", WARMUP_SECONDS)
        print(f"[soak] 비활성 (A6000 에서 이득 없음). "
              f"워밍업 {_warm_s}초. --soak 로 켤 수 있다", flush=True)
        _t0 = time.time()
        _wk = kernels[drift_kernel_id]
        _max_mem = (env.get("clocks") or {}).get("max_memory_mhz")
        _floor = _max_mem * MEM_CLOCK_MIN_FRAC if _max_mem else None
        _n = 0
        _snap = None
        # 시간이 아니라 **메모리 클럭이 올라온 것**으로 끝낸다. 유휴 상태에서
        # 재개하면 메모리 클럭이 810 MHz 까지 떨어져 있고, 램프업 전에 재면
        # 메모리 바운드 config 가 최대 66% 느리게 측정된다 (실측).
        # 최대 3배까지 기다린다.
        while time.time() - _t0 < _warm_s * 3:
            if _probe_ref(ctx, _wk) is None:
                break
            _n += 1
            if time.time() - _t0 < _warm_s:
                continue
            _snap = probe.snapshot()
            if _floor is None or (_snap.get("mem_clock_mhz") or 0) >= _floor:
                break
        _snap = _snap or probe.snapshot()
        _mem_ok = _floor is None or (_snap.get("mem_clock_mhz") or 0) >= _floor
        print(f"  워밍업 {_n}회 {time.time() - _t0:.0f}초, "
              f"SM {_snap['sm_clock_mhz']} MHz "
              f"mem {_snap.get('mem_clock_mhz')} MHz "
              f"{_snap['gpu_temp_c']}C", flush=True)
        if not _mem_ok:
            # 여기서 멈추지는 않는다 — 스윕 전체가 서면 더 곤란하다. 대신
            # 크게 경고하고 기록에 남겨 리포트에서 걸러낼 수 있게 한다.
            print(f"  !! 메모리 클럭 {_snap.get('mem_clock_mhz')} MHz 가 "
                  f"최대치의 {100 * MEM_CLOCK_MIN_FRAC:.0f}%({_floor:.0f} MHz) "
                  f"미만이다. 램프업이 끝나지 않았다 — 이 슬라이스의 "
                  f"메모리 바운드 측정은 느리게 나올 수 있다.", flush=True)
        soak_info = {"soak_enabled": False, "soak_seconds": round(time.time() - _t0, 1),
                     "warmup_reps": _n, "soak_ref_last_ms": None,
                     "warmup_mem_clock_mhz": _snap.get("mem_clock_mhz"),
                     "warmup_mem_clock_ok": bool(_mem_ok)}
    soak_started = time.time()
    thermal = {"soak_elapsed_s": 0.0, "drift_ratio": 1.0}
    # 절대 기준. 소킹을 했으면 소킹 직후 값, 안 했으면 **첫 드리프트 측정값**
    # 으로 래치한다. None 으로 두면 drift_ratio 가 영원히 1.0 이라 D-3 이
    # 무의미해진다.
    drift_base = soak_info.get("soak_ref_last_ms")

    # --- alignment 가드가 실제로 필요한지 확인 (측정 전에) ---------------
    print("\n[가드 검증] (1024,4096,4100) 에 a888 커널을 강제로 물려본다")
    try:
        if args.all:
            raise RuntimeError("전수 모드에서는 생략")
        guard = alignment_guard_probe(ctx, sample, kernels, hw)
        for g in guard:
            print(f"  {g['kernel_id'][:52]:52s} can_implement={g['can_implement']}"
                  + (f"  max_rel_error={g['max_rel_error']:.3e}"
                     if g.get("max_rel_error") is not None else ""))
        (paths.RESULTS_DIR / "guard_probe.json").write_text(
            json.dumps(guard, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"  가드 검증 중 예외: {e!r}")

    cublas_cache: dict[tuple, dict] = {}
    last_drift = 0.0
    last_status_log = time.time()
    drift_hist: list[float] = []          # D-2 이동 기준용
    drift_strikes = 0
    aborted = None
    t_start = time.time()

    stats = Counter()
    n = 0
    time_up = False
    last_heartbeat = 0.0
    write_heartbeat(state="starting", done=0, total=len(jobs),
                    env_hash=env["env_hash"], soak=soak_info)
    # 세그먼트 시작 앵커. 끝 앵커와 짝지어 이 세그먼트 안의 이동량을,
    # 세그먼트끼리 비교해 세그먼트 간 오프셋을 잰다.
    if args.all and anchors:
        measure_anchors(ctx, kernels, anchors, probe, env,
                        args.segment if args.segment is not None else -1,
                        "start", args.round_)
    try:
        with RESULTS.open("a") as out:
            for (krow, p, rc) in jobs:
                # 시간 예산 — 다음 세그먼트로 넘어가기 위해 깔끔하게 멈춘다.
                # 중간에 멈춰도 results.jsonl 은 append-only 라 재개가 된다.
                if args.max_jobs and n >= args.max_jobs:
                    time_up = True
                    print(f"\n[세그먼트] 작업 예산 {args.max_jobs:,}개 소진. "
                          f"{(time.time() - t_start) / 60:.1f}분 걸렸다.",
                          flush=True)
                    break
                if args.time_budget and (time.time() - t_start) > args.time_budget:
                    time_up = True
                    print(f"\n[세그먼트] 시간 상한 {args.time_budget:.0f}초 "
                          f"도달. {n:,}개 측정하고 넘긴다. (작업 예산보다 "
                          f"느리다 — 이 세그먼트가 무거운지 확인하라)", flush=True)
                    break
                thermal["soak_elapsed_s"] = round(time.time() - soak_started, 1)
                row = measure_one(ctx, kernels[krow["kernel_id"]], krow, p, rc,
                                  probe, env, launch_overhead_ms, hw.regs_per_sm,
                                  thermal)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                stats[row["status"]] += 1
                n += 1

                # cuBLAS 참조는 형상당 한 번
                key = (p.M, p.N, p.K)
                if key not in cublas_cache:
                    cublas_cache[key] = measure_cublas(ctx, p, probe, env, out)

                # --- 요청 O: 장시간 실행 감시 ---------------------------
                if time.time() - last_drift > drift_period:
                    last_drift = time.time()
                    t_drift = drift_check(ctx, kernels[drift_kernel_id],
                                          drift_kernel_id, probe, env,
                                          thermal)
                    if t_drift:
                        # D-3: 이 시점의 열 상태를 이후 측정 줄에 실어 보낸다
                        thermal["soak_elapsed_s"] = round(
                            time.time() - soak_started, 1)
                        if not drift_base:
                            drift_base = t_drift     # 소킹 없이 시작한 경우
                        if drift_base:
                            thermal["drift_ratio"] = round(t_drift / drift_base, 5)
                        # D-2: 이동 중앙값 기준. 완만한 예열은 통과시키고
                        #      급격한 변화만 잡는다.
                        if drift_hist:
                            ref = statistics.median(drift_hist[-DRIFT_WINDOW:])
                            dev = abs(t_drift - ref) / ref
                            if dev > DRIFT_TOL:
                                drift_strikes += 1
                                print(f"  !! 드리프트 {100 * dev:.2f}% "
                                      f"({t_drift:.4f} vs 이동기준 {ref:.4f} ms) "
                                      f"연속 {drift_strikes}회", flush=True)
                            else:
                                drift_strikes = 0
                        drift_hist.append(t_drift)
                        # 절대 상한: 소킹 직후 대비 누적 드리프트 경고
                        if drift_base and abs(t_drift - drift_base) / drift_base \
                                > DRIFT_ABS_WARN:
                            print(f"  !! 소킹 후 누적 드리프트 "
                                  f"{100 * (t_drift - drift_base) / drift_base:+.2f}% "
                                  f"> {100 * DRIFT_ABS_WARN:.0f}% "
                                  f"(이동기준으로는 정상이지만 느린 누적이다)",
                                  flush=True)
                        if drift_strikes >= DRIFT_STRIKES:
                            aborted = (f"이동 기준 대비 {DRIFT_TOL:.0%} 초과가 "
                                       f"{DRIFT_STRIKES}회 연속 — 조건이 변했다")
                            break
                    tel = telemetry_tail_stats(int(drift_period) + 60)
                    if tel and tel["sw_power_cap_frac"] > POWER_CAP_TOL:
                        print(f"  !! sw_power_cap {100 * tel['sw_power_cap_frac']:.1f}% "
                              f"(최근 {tel['n']}초, 클럭 {tel['clk_min']}~"
                              f"{tel['clk_max']} MHz) — 클럭 고정이 풀렸거나 "
                              f"다른 프로세스가 GPU 를 쓰고 있다", flush=True)

                if time.time() - last_status_log > STATUS_LOG_SECONDS:
                    last_status_log = time.time()
                    el = (time.time() - t_start) / 3600
                    print(f"  [{el:.1f}h] status {dict(stats)}  "
                          f"진행 {n}/{len(jobs)} "
                          f"({100 * n / len(jobs):.1f}%)  "
                          f"telemetry {telemetry_tail_stats(3600)}", flush=True)

                if n % 50 == 0 or n == len(jobs):
                    el = time.time() - t_start
                    print(f"  {n}/{len(jobs)}  {el / 60:.1f}분  "
                          f"{dict(stats)}", flush=True)
                if time.time() - last_heartbeat > HEARTBEAT_SECONDS or n == len(jobs):
                    last_heartbeat = time.time()
                    el = time.time() - t_start
                    rate = n / max(el, 1e-9)
                    write_heartbeat(
                        state="running", done=n, total=len(jobs),
                        pct=round(100 * n / max(len(jobs), 1), 3),
                        elapsed_h=round(el / 3600, 3),
                        eta_h=round((len(jobs) - n) / max(rate, 1e-9) / 3600, 2),
                        rate_per_s=round(rate, 2), status=dict(stats),
                        env_hash=env["env_hash"], soak=soak_info,
                        thermal=dict(thermal))
                    print(f"  [hb] {n}/{len(jobs)} "
                          f"({100 * n / max(len(jobs), 1):.2f}%) "
                          f"ETA {(len(jobs) - n) / max(rate, 1e-9) / 3600:.1f}h",
                          flush=True)

            # --- 재현성 검증 ---------------------------------------------
            print("\n[재현성] 무작위 20개 조합 재측정")
            repro = ([] if args.all else
                     reproducibility(ctx, kernels, jobs, probe, env,
                                     launch_overhead_ms, seed, hw.regs_per_sm))
            if args.all and anchors:
                measure_anchors(ctx, kernels, anchors, probe, env,
                                args.segment if args.segment is not None else -1,
                                "end", args.round_)
        write_heartbeat(state="finishing", done=n, total=len(jobs),
                        status=dict(stats), env_hash=env["env_hash"],
                        soak=soak_info, thermal=dict(thermal))
    finally:
        tele_proc.terminate()
        try:
            tele_proc.wait(timeout=5)
        except Exception:
            tele_proc.kill()
        tele_file.close()
        probe.close()
        ctx.close()

    # R-4b: NVML 조회 실패를 세어 보고한다. 조용히 None 이 들어가면
    #       sm_clock/mem_clock 이 결측으로 기록되고, 33시간 뒤에
    #       "조건이 유지됐는가" 를 확인할 근거가 사라진다.
    nvml_fail = probe.failures()
    write_heartbeat(state="aborted" if aborted else "done", done=n,
                    total=len(jobs), status=dict(stats),
                    env_hash=env["env_hash"], soak=soak_info,
                    thermal=dict(thermal), abort_reason=aborted,
                    nvml=nvml_fail)
    print("\n" + probe.report())
    report(stats, repro, clock_locked)
    if aborted:
        print(f"\n!! 중단: {aborted}")
        print("   results.jsonl 은 그대로 남아 있으므로 원인을 고친 뒤 "
              "같은 명령으로 이어서 진행하면 된다.")
        return 3
    if time_up:
        return 7      # 정상. sweep.py 가 다음 세그먼트로 넘어간다
    return 0


def measure_one(ctx, kern, krow, p: Problem, rc: RuntimeConfig, probe, env,
                launch_overhead_ms: float, regs_per_sm: int,
                thermal: dict | None = None) -> dict:
    from kerneltab.measure.runner import KtProblemC

    parallel = rc.split_k_mode == "parallel"
    kp = KtProblemC(p.M, p.N, p.K, rc.split_k, 1 if parallel else 0)

    row = {
        "kernel_id": krow["kernel_id"],
        "problem": {"M": p.M, "N": p.N, "K": p.K, "dtype": p.dtype},
        "runtime": {"split_k": rc.split_k, "split_k_mode": rc.split_k_mode},
        "env_hash": env["env_hash"],
        "clock_locked": env["clock_locked"],
        # D-3: 열 상태. 사후에 "열이 랭킹에 영향을 줬는가" 를 데이터로 검증할 수
        # 있어야 한다. 이 정보가 없어서 226k 행을 폐기해야 했다.
        "soak_elapsed_s": (thermal or {}).get("soak_elapsed_s"),
        "drift_ratio": (thermal or {}).get("drift_ratio"),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    if not launchable(krow, regs_per_sm):
        row.update(status="launch_infeasible",
                   error=f"regs_per_thread({krow.get('regs_per_thread')}) * "
                         f"threads({krow.get('threads')}) > regs_per_sm({regs_per_sm})")
        return row

    try:
        ctx.prepare_problem(p.M, p.N, p.K)
    except Exception as e:
        row.update(status="runtime_fail", error=f"prepare_problem: {e}")
        return row

    ws = kern.workspace_bytes(kp)
    actual_k = kern.grid_k(kp)
    can = kern.can_implement(kp)
    row["workspace_bytes"] = ws
    row["actual_split_k"] = actual_k          # ★ effective_split_k() 실측 대조용
    row["predicted_split_k"] = rc.split_k
    # 부분합 저장 정밀도. serial 과 parallel 이 같은 조건인지 나중에 판별하려면
    # 이 정보가 있어야 한다 (README "split-K 부분합 정밀도" 참조).
    #   parallel : 부분합을 workspace(ElementC=fp16)에 슬라이스별로 저장
    #   serial   : 부분합을 D(ElementC=fp16)에 두고 다음 파티션이 읽어 누적
    # 둘 다 fp16 이라 비교 조건은 동일하다.
    row["workspace_dtype"] = ("f16" if parallel
                              else ("i32" if actual_k > 1 else None))
    row["partials_dtype"] = "f16" if actual_k > 1 else None

    if can != 0:
        row.update(status="runtime_fail",
                   error=f"can_implement: {kern.status_string(can)}")
        return row
    try:
        bufs = ctx.buffers(ws, parallel)
    except MemoryError as e:
        row.update(status="oom", error=str(e))
        return row

    st, handle = kern.prepare(kp, bufs)
    if st != 0 or not handle:
        row.update(status="runtime_fail",
                   error=f"kt_prepare: {kern.status_string(st)}")
        return row

    try:
        reduce_slices = actual_k if parallel else 0
        # 1) 정확도: 한 번 실행하고 cuBLAS 참조와 비교
        st = ctx.run_once(kern.launch_addr, handle, reduce_slices)
        if st != 0:
            row.update(status="runtime_fail", error=f"run_once: {ctx.last_error()}")
            return row
        err = ctx.max_rel_error()
        row["max_rel_error"] = err

        # 2) 시간
        st, m = ctx.measure(kern.launch_addr, handle, reduce_slices)
        if st != 0:
            row.update(status="runtime_fail", error=f"measure: {ctx.last_error()}")
            return row
        row.update(
            time_ms=m.time_ms, time_std_ms=m.time_std_ms,
            time_min_ms=m.time_min_ms, time_max_ms=m.time_max_ms,
            n_reps=m.n_reps, outlier_frac=round(m.outlier_frac, 4),
            # 예비/워밍업 회계. 시간 예산으로 바꿨으므로 **실제로 몇 번
            # 돌았는지** 남긴다 — 없으면 "워밍업이 조용히 0 이 됐다" 를
            # 사후에 확인할 수 없다.
            n_probe=m.n_probe, n_warmup=m.n_warmup,
            overhead_ms=round(m.overhead_ms, 4),
        )
    finally:
        kern.release(handle)

    snap = probe.snapshot()
    row["sm_clock_mhz"] = snap["sm_clock_mhz"]
    row["mem_clock_mhz"] = snap["mem_clock_mhz"]
    row["gpu_temp_c"] = snap["gpu_temp_c"]
    row["power_w"] = snap["power_w"]

    if err > NUMERICAL_TOL or err < 0:
        row["status"] = "numerical_fail"
    elif row["time_ms"] < 3 * launch_overhead_ms:
        row["status"] = "below_launch_overhead"
    elif row["outlier_frac"] > OUTLIER_TOL:
        row["status"] = "high_outlier_frac"
    else:
        row["status"] = "ok"
    return row


def alignment_guard_probe(ctx, sample, kernels, hw) -> list[dict]:
    """is_valid_runtime 의 alignment 조건이 실제로 필요한지 직접 확인한다.

    (1024, 4096, 4100) 은 K % 8 != 0 이라 align_a/b = 4 다. 여기에 a888 커널을
    쓰면 8원소 벡터 로드가 경계를 넘는다. 우리 열거기는 이 조합을 만들지
    않지만, "만들면 정말로 틀리는가" 를 확인해 두지 않으면 그 가드가 왜
    있는지 알 수 없게 된다.
    """
    from kerneltab.measure.runner import KtProblemC

    p = Problem(1024, 4096, 4100)
    out = []
    a888 = [r for r in sample
            if (r["align"]["a"], r["align"]["b"], r["align"]["c"]) == (8, 8, 8)][:3]
    ctx.prepare_problem(p.M, p.N, p.K)
    for r in a888:
        k = kernels[r["kernel_id"]]
        kp = KtProblemC(p.M, p.N, p.K, 1, 0)
        can = k.can_implement(kp)
        rec = {"kernel_id": r["kernel_id"],
               "problem": {"M": p.M, "N": p.N, "K": p.K},
               "can_implement": k.status_string(can)}
        if can == 0:
            bufs = ctx.buffers(k.workspace_bytes(kp), False)
            st, h = k.prepare(kp, bufs)
            if st == 0 and h:
                try:
                    rs = ctx.run_once(k.launch_addr, h, 0)
                    rec["run_status"] = rs
                    rec["max_rel_error"] = ctx.max_rel_error() if rs == 0 else None
                finally:
                    k.release(h)
            else:
                rec["prepare_status"] = k.status_string(st)
        out.append(rec)
    return out


def measure_cublas(ctx, p: Problem, probe, env, out) -> dict:
    ctx.prepare_problem(p.M, p.N, p.K)
    st, m = ctx.measure_cublas()
    snap = probe.snapshot()
    row = {
        "kernel_id": "cublas",
        "problem": {"M": p.M, "N": p.N, "K": p.K, "dtype": p.dtype},
        "runtime": {"split_k": 1, "split_k_mode": "serial"},
        "time_ms": m.time_ms, "time_std_ms": m.time_std_ms,
        "n_reps": m.n_reps, "outlier_frac": round(m.outlier_frac, 4),
        "max_rel_error": 0.0,
        "status": "ok" if st == 0 else "runtime_fail",
        "sm_clock_mhz": snap["sm_clock_mhz"], "gpu_temp_c": snap["gpu_temp_c"],
        "clock_locked": env["clock_locked"], "env_hash": env["env_hash"],
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    out.flush()
    return row


def _probe_ref(ctx, kern_obj, shape=DRIFT_SHAPE) -> float | None:
    """기준 config 를 한 번 재서 시간을 돌려준다 (소킹 판정 / 드리프트 공용)."""
    from kerneltab.measure.runner import KtProblemC

    ctx.prepare_problem(shape.M, shape.N, shape.K)
    kp = KtProblemC(shape.M, shape.N, shape.K, 1, 0)
    bufs = ctx.buffers(kern_obj.workspace_bytes(kp), False)
    st, h = kern_obj.prepare(kp, bufs)
    if st != 0 or not h:
        return None
    try:
        _, m = ctx.measure(kern_obj.launch_addr, h, 0)
    finally:
        kern_obj.release(h)
    return m.time_ms


def thermal_soak(ctx, kernels, sample, probe, ref_kid: str,
                 max_seconds: int = SOAK_MAX_SECONDS,
                 min_seconds: int = SOAK_MIN_SECONDS,
                 interval: int = SOAK_PROBE_INTERVAL,
                 span_tol: float = SOAK_STABLE_SPAN,
                 runs: int = SOAK_STABLE_RUNS) -> dict:
    """본 측정 전 실부하로 열평형에 도달시킨다 (D-1).

    코어 온도는 5 분이면 평형이지만 성능은 몇 시간에 걸쳐 계속 느려진다.
    그래서 온도가 아니라 **기준 config 의 측정값 자체**로 판정한다.
    최근 3 회(10 분 구간)의 (max-min)/median 이 SOAK_STABLE_SPAN 이내면 평형.

    돌려주는 값은 results 줄에 기록된다 (D-3).
    """
    from kerneltab.measure.runner import KtProblemC

    ref = kernels[ref_kid]
    # 소킹 부하는 실제 측정과 비슷해야 한다. 큰 형상을 반복한다.
    load_kid = ref_kid
    lk = kernels[load_kid]
    load_shape = Problem(8192, 4096, 4096)
    ctx.prepare_problem(load_shape.M, load_shape.N, load_shape.K)
    lkp = KtProblemC(load_shape.M, load_shape.N, load_shape.K, 1, 0)
    lbufs = ctx.buffers(lk.workspace_bytes(lkp), False)
    lst, lh = lk.prepare(lkp, lbufs)
    if lst != 0 or not lh:
        print("[soak] 부하 커널 prepare 실패 — 소킹을 건너뛴다")
        return {"soak_seconds": 0, "soak_reason": "prepare_fail"}

    t0 = time.time()
    probes: list[tuple[float, float]] = []
    s0 = probe.snapshot()
    print(f"[soak] 시작  temp={s0['gpu_temp_c']}°C mem={s0['mem_clock_mhz']}MHz  "
          f"최소 {min_seconds // 60}분 / 최대 {max_seconds // 60}분", flush=True)
    reason = "max_seconds"
    try:
        while True:
            # 5분 부하
            t_chunk = time.time()
            while time.time() - t_chunk < interval:
                if ctx.run_once(lk.launch_addr, lh, 0) != 0:
                    break
            el = time.time() - t0
            t_ref = _probe_ref(ctx, ref)
            # 부하 형상으로 되돌린다 (다음 청크를 위해)
            ctx.prepare_problem(load_shape.M, load_shape.N, load_shape.K)
            if t_ref is None:
                break
            probes.append((el, t_ref))
            sn = probe.snapshot()
            rel = 100 * (t_ref - probes[0][1]) / probes[0][1]
            span = None
            if len(probes) >= runs:
                w = [v for _, v in probes[-runs:]]
                span = (max(w) - min(w)) / statistics.median(w)
            print(f"[soak] {el / 60:5.1f}분  ref={t_ref:.4f} ms ({rel:+.2f}%)  "
                  f"span={'-' if span is None else f'{100 * span:.3f}%'}  "
                  f"temp={sn['gpu_temp_c']}°C power={sn['power_w']:.0f}W", flush=True)
            if el >= max_seconds:
                break
            if (el >= min_seconds and span is not None
                    and span <= span_tol):
                reason = "stable"
                break
    finally:
        lk.release(lh)

    base = probes[0][1] if probes else None
    last = probes[-1][1] if probes else None
    # 최근 두 점의 기울기로 잔여 드리프트를 대략 추정한다 (로그 목적)
    slope = None
    if len(probes) >= 2 and probes[-1][0] > probes[-2][0]:
        slope = ((probes[-1][1] - probes[-2][1]) / probes[-2][1]
                 / (probes[-1][0] - probes[-2][0]) * 3600)
    info = {
        "soak_seconds": round(time.time() - t0, 1),
        "soak_reason": reason,
        "soak_ref_first_ms": base,
        "soak_ref_last_ms": last,
        "soak_total_drift": (last - base) / base if base else None,
        "soak_slope_per_hour": slope,
        "soak_probes": [[round(a, 1), b] for a, b in probes],
    }
    print(f"[soak] 종료 ({reason})  {info['soak_seconds'] / 60:.1f}분, "
          f"누적 드리프트 {100 * (info['soak_total_drift'] or 0):+.2f}%, "
          f"잔여 기울기 {'?' if slope is None else f'{100 * slope:+.2f}%/h'}",
          flush=True)
    return info


def write_heartbeat(**kw) -> None:
    """감시자가 읽을 진행 상태. 원자적으로 교체한다."""
    kw["pid"] = os.getpid()
    kw["utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    tmp = HEARTBEAT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(kw, ensure_ascii=False, indent=2))
    tmp.replace(HEARTBEAT)


def telemetry_tail_stats(seconds: int) -> dict:
    """최근 N초 구간의 텔레메트리 요약. 감시용이므로 파일 끝만 본다."""
    if not TELEMETRY.exists():
        return {}
    lines = TELEMETRY.read_text().splitlines()[1:][-seconds:]
    n = 0
    thr = Counter()
    clks = []
    for line in lines:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            clks.append(int(parts[1].split()[0]))
            bits = int(parts[5], 16)
        except (ValueError, IndexError):
            continue
        n += 1
        for mask, name in THROTTLE_BITS.items():
            if bits & mask:
                thr[name] += 1
    if not n:
        return {}
    return {"n": n, "sw_power_cap_frac": thr.get("sw_power_cap", 0) / n,
            "throttle": dict(thr),
            "clk_min": min(clks), "clk_max": max(clks)}


def _probe_shape(ctx, kern, p: Problem):
    from kerneltab.measure.runner import KtProblemC

    ctx.prepare_problem(p.M, p.N, p.K)
    kp = KtProblemC(p.M, p.N, p.K, 1, 0)
    bufs = ctx.buffers(kern.workspace_bytes(kp), False)
    st, h = kern.prepare(kp, bufs)
    if st != 0 or not h:
        return None
    try:
        st, m = ctx.measure(kern.launch_addr, h, 0)
    finally:
        kern.release(h)
    return m


def drift_check(ctx, kern, kernel_id, probe, env,
                thermal: dict | None = None) -> float | None:
    """기준 커널을 **여러 형상**에서 재고 전부 기록한다.

    ⚠️ 큰 형상 하나만 재면 안 된다. Phase 3 이 4096³ 하나로 감시해서
    드리프트를 +5.06 % 로 봤는데, 같은 조건의 512³ 측정은 +1380 % 오염되어
    있었다. 런치당 상수 오버헤드는 긴 커널에서 안 보인다.

    돌려주는 값은 **작은 형상** 쪽이다 — 감시의 민감도가 거기서 나온다.
    """
    extra = {}
    for q in DRIFT_SHAPES[:-1]:
        mq = _probe_shape(ctx, kern, q)
        if mq is not None:
            extra[f"time_ms_{q.M}"] = mq.time_ms
            extra[f"n_reps_{q.M}"] = mq.n_reps

    p = DRIFT_SHAPE
    m = _probe_shape(ctx, kern, p)
    if m is None:
        return
    snap = probe.snapshot()
    with DRIFT.open("a") as f:
        f.write(json.dumps({
            "kernel_id": kernel_id,
            "problem": {"M": p.M, "N": p.N, "K": p.K},
            "time_ms": m.time_ms, "time_std_ms": m.time_std_ms,
            "n_reps": m.n_reps,
            **extra,
            "sm_clock_mhz": snap["sm_clock_mhz"],
            "gpu_temp_c": snap["gpu_temp_c"], "power_w": snap["power_w"],
            "clock_locked": env["clock_locked"],
            # drift.jsonl 에 env_hash 가 없어 조건별로 나눌 수 없었다 (감사 지적).
            "env_hash": env["env_hash"],
            "soak_elapsed_s": (thermal or {}).get("soak_elapsed_s"),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }) + "\n")
    return m.time_ms


def reproducibility(ctx, kernels, jobs, probe, env, launch_overhead_ms,
                    seed, regs_per_sm):
    rng = random.Random(seed + 1)
    pick = rng.sample(jobs, min(20, len(jobs)))
    # ⚠️ **이 측정 조건의 줄만** 기준값으로 쓴다. 필터 없이 읽으면 클럭
    #    미고정 리허설이나 폐기된 구간의 시간이 기준이 되어 재현성 숫자가
    #    통째로 무의미해진다 (R-5, docs/decisions.md 13).
    first = {}
    for d in records.iter_records(RESULTS, env["env_hash"]):
        if d["kernel_id"] == "cublas":
            continue
        p, r = d["problem"], d["runtime"]
        first[(d["kernel_id"], p["M"], p["N"], p["K"],
               r["split_k"], r["split_k_mode"])] = d.get("time_ms")

    out = []
    with (paths.RESULTS_DIR / "repro.jsonl").open("a") as f:
        for (krow, p, rc) in pick:
            row = measure_one(ctx, kernels[krow["kernel_id"]], krow, p, rc, probe,
                              env, launch_overhead_ms, regs_per_sm)
            row["repro_pass"] = 2
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            key = (krow["kernel_id"], p.M, p.N, p.K, rc.split_k, rc.split_k_mode)
            t0 = first.get(key)
            t1 = row.get("time_ms")
            if t0 and t1:
                out.append({"key": key, "first_ms": t0, "second_ms": t1,
                            "rel_diff": abs(t1 - t0) / t0})
    return out


def report(stats, repro, clock_locked):
    print("\n" + "=" * 72)
    print("리허설 완료")
    print("=" * 72)
    print(f"status: {dict(stats)}")

    if repro:
        bad = [r for r in repro if r["rel_diff"] > 0.05]
        worst = max(repro, key=lambda r: r["rel_diff"])
        print(f"\n재현성: {len(repro)}개 재측정, 5% 초과 {len(bad)}개, "
              f"최대 차이 {100 * worst['rel_diff']:.2f}%")
        for r in sorted(repro, key=lambda r: -r["rel_diff"])[:5]:
            print(f"  {100 * r['rel_diff']:6.2f}%  {r['first_ms']:.4f} -> "
                  f"{r['second_ms']:.4f} ms  {r['key'][0][:50]}")

    if DRIFT.exists():
        # 이 측정 조건의 줄만 본다 (R-5). drift.jsonl 은 append-only 라
        # 클럭 미고정 리허설과 폐기된 구간이 같이 들어 있고, 필터하지 않으면
        # 변동폭이 55% 로 나와 경고가 상시 울린다.
        _eh = json.loads(paths.ENV_JSON.read_text())["env_hash"]
        ds = records.load_records(DRIFT, _eh)
        if ds:
            ts = [d["time_ms"] for d in ds]
            mean = sum(ts) / len(ts)
            span = (max(ts) - min(ts)) / mean if mean else 0
            print(f"\n드리프트: {len(ds)}회 점검  "
                  f"min={min(ts):.4f} max={max(ts):.4f} mean={mean:.4f} ms  "
                  f"변동폭={100 * span:.2f}%")
            if span > 0.05:
                print("  !! 5% 이상 변동. clock_locked="
                      f"{clock_locked} 상태에서의 드리프트다.")

    tel = analyze_telemetry()
    if tel:
        print(f"\n텔레메트리: {tel}")


if __name__ == "__main__":
    raise SystemExit(main())
