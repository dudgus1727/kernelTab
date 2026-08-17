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
import subprocess
import sys
import time
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backends import get_backend  # noqa: E402
from build import paths  # noqa: E402
from core.hardware import hardware_from_env  # noqa: E402
from core.config import alignments_for, enumerate_runtimes  # noqa: E402
from core.shapes import all_shapes  # noqa: E402
from measure.runner import SOAK_DEFAULTS  # noqa: E402,F401
from core.types import Hardware, KernelConfig, Problem, RuntimeConfig  # noqa: E402

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

DRIFT_SHAPE = Problem(4096, 4096, 4096)

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


def launchable(krow: dict, regs_per_sm: int) -> bool:
    """런치 가능한가.

    cutlass::Kernel2 에는 __launch_bounds__ 가 없어서 ptxas 가 스레드 수에 맞춰
    레지스터를 제한하지 않는다. 그 결과 regs_per_thread * threads 가 SM 당
    레지스터 수를 넘는 커널이 만들어지고, 이런 커널은 런치 자체가 실패한다
    (cudaOccupancyMaxActiveBlocksPerMultiprocessor 도 0 을 돌려준다).
    실제로 런치를 시도해 에러를 받기보다 미리 판정하는 편이 명확하고 빠르다.
    """
    r, t = krow.get("regs_per_thread"), krow.get("threads")
    if not r or not t:
        return True
    return r * t <= regs_per_sm

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
def start_telemetry(device: int):
    TELEMETRY.parent.mkdir(parents=True, exist_ok=True)
    f = TELEMETRY.open("w")
    p = subprocess.Popen(
        ["nvidia-smi", "-i", str(device),
         "--query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,"
         "power.draw,clocks_throttle_reasons.active",
         "--format=csv", "-l", "1"],
        stdout=f, stderr=subprocess.DEVNULL)
    return p, f


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
def load_done(env_hash: str) -> set[tuple]:
    """이미 측정된 조합. **같은 측정 조건(env_hash)의 줄만** 완료로 본다."""
    if not RESULTS.exists():
        return set()
    done = set()
    for line in RESULTS.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            if d.get("env_hash") != env_hash:
                continue
            p, r = d["problem"], d["runtime"]
            done.add((d["kernel_id"], p["M"], p["N"], p["K"],
                      r["split_k"], r["split_k_mode"]))
        except Exception:
            continue
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
    args = ap.parse_args()

    env = json.loads(paths.ENV_JSON.read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(env["device_index"])
    hw = hardware_from_env(env)
    backend = get_backend(hw.arch)
    seed = env["shuffle_seed"]
    launch_overhead_ms = env["launch_overhead_ms"]
    drift_period = env["drift_check_seconds"]
    clock_locked = env["clock_locked"]

    from measure.gpu_state import NvmlProbe  # noqa: E402
    from measure.runner import Ctx, Kernel, KtProblemC  # noqa: E402

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
        from core.config import alignment_combos, enumerate_kernels  # noqa: E402

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
    if args.dry_run:
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
        kernels[r["kernel_id"]] = Kernel(r["so_path"])
    probe = NvmlProbe(uuid=env["hardware_extra"]["uuid"], index=0)

    drift_kernel_id = picked[0]["kernel_id"]
    tele_proc, tele_file = start_telemetry(env["device_index"])
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
        print("[soak] 비활성 (A6000 에서 이득 없음). 워밍업만 한다. "
              "--soak 로 켤 수 있다")
        soak_info = {"soak_enabled": False, "soak_seconds": 0.0,
                     "soak_ref_last_ms": None}
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
    last_heartbeat = 0.0
    write_heartbeat(state="starting", done=0, total=len(jobs),
                    env_hash=env["env_hash"], soak=soak_info)
    try:
        with RESULTS.open("a") as out:
            for (krow, p, rc) in jobs:
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

    write_heartbeat(state="aborted" if aborted else "done", done=n,
                    total=len(jobs), status=dict(stats),
                    env_hash=env["env_hash"], soak=soak_info,
                    thermal=dict(thermal), abort_reason=aborted)
    report(stats, repro, clock_locked)
    if aborted:
        print(f"\n!! 중단: {aborted}")
        print("   results.jsonl 은 그대로 남아 있으므로 원인을 고친 뒤 "
              "같은 명령으로 이어서 진행하면 된다.")
        return 3
    return 0


def measure_one(ctx, kern, krow, p: Problem, rc: RuntimeConfig, probe, env,
                launch_overhead_ms: float, regs_per_sm: int,
                thermal: dict | None = None) -> dict:
    from measure.runner import KtProblemC

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
    from measure.runner import KtProblemC

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
    from measure.runner import KtProblemC

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
    from measure.runner import KtProblemC

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


def drift_check(ctx, kern, kernel_id, probe, env,
                thermal: dict | None = None) -> float | None:
    from measure.runner import KtProblemC

    p = DRIFT_SHAPE
    ctx.prepare_problem(p.M, p.N, p.K)
    kp = KtProblemC(p.M, p.N, p.K, 1, 0)
    bufs = ctx.buffers(kern.workspace_bytes(kp), False)
    st, h = kern.prepare(kp, bufs)
    if st != 0 or not h:
        return
    try:
        st, m = ctx.measure(kern.launch_addr, h, 0)
    finally:
        kern.release(h)
    snap = probe.snapshot()
    with DRIFT.open("a") as f:
        f.write(json.dumps({
            "kernel_id": kernel_id,
            "problem": {"M": p.M, "N": p.N, "K": p.K},
            "time_ms": m.time_ms, "time_std_ms": m.time_std_ms,
            "n_reps": m.n_reps,
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
    first = {}
    for line in RESULTS.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
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
        ds = [json.loads(l) for l in DRIFT.read_text().splitlines() if l.strip()]
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
