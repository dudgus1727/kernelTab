#!/usr/bin/env python3
"""Phase 2-2: 리허설 측정.

전수가 아니라 6개 형상 x 층화 표집한 20개 커널 config x 유효 런타임 전부.
측정 순서는 셔플한다 (온도 드리프트가 config 순서와 상관되는 것을 막는다).

    python3 scripts/rehearse.py
    python3 scripts/rehearse.py --dry-run     # 작업 목록만 세어본다

results.jsonl 은 append-only. 이미 있는 (kernel_id, 형상, 런타임) 은 건너뛴다.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backends import get_backend  # noqa: E402
from build import paths  # noqa: E402
from core.config import alignments_for, enumerate_runtimes  # noqa: E402
from core.types import Hardware, KernelConfig, Problem, RuntimeConfig  # noqa: E402

RESULTS = paths.RESULTS_DIR / "results.jsonl"
DRIFT = paths.RESULTS_DIR / "drift.jsonl"
TELEMETRY = paths.RESULTS_DIR / "telemetry.csv"
KERNELS = paths.RESULTS_DIR / "kernels.jsonl"

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
def load_done() -> set[tuple]:
    if not RESULTS.exists():
        return set()
    done = set()
    for line in RESULTS.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            p, r = d["problem"], d["runtime"]
            done.add((d["kernel_id"], p["M"], p["N"], p["K"],
                      r["split_k"], r["split_k_mode"]))
        except Exception:
            continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-kernels", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    env = json.loads(paths.ENV_JSON.read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(env["device_index"])
    hw = Hardware(**env["hardware"])
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

    base = by_align.get((8, 8, 8), [])
    picked = stratified_sample(base, args.n_kernels, seed, hw.regs_per_sm)
    # a448 대응짝: 동일 구성의 alignment 변형. 이러면 (1024,4096,4100) 형상에서
    # alignment 효과만 분리해서 볼 수 있다.
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
        for p in REHEARSAL_SHAPES:
            if alignments_for(p) != a:
                continue
            cfg = cfg_from_row(r, backend)
            for rc in enumerate_runtimes(backend, p, cfg):
                jobs.append((r, p, rc))

    print(f"\n작업 수: {len(jobs)} "
          f"(형상 {len(REHEARSAL_SHAPES)}개, 커널 {len(sample)}개)")
    by_shape = Counter((p.M, p.N, p.K) for _, p, _ in jobs)
    for k, v in sorted(by_shape.items()):
        print(f"  {k}: {v}")
    if args.dry_run:
        return 0

    rng = random.Random(seed)
    rng.shuffle(jobs)  # ★ 온도 드리프트가 config 순서와 상관되지 않도록
    if args.limit:
        jobs = jobs[: args.limit]

    done = load_done()
    jobs = [j for j in jobs
            if (j[0]["kernel_id"], j[1].M, j[1].N, j[1].K,
                j[2].split_k, j[2].split_k_mode) not in done]
    print(f"남은 작업: {len(jobs)} (완료 {len(done)})")
    if not jobs:
        return 0

    # --- 초기화 ------------------------------------------------------------
    ctx = Ctx(paths.ARTIFACT_DIR / "libkt_ctx.so", 0)
    kernels: dict[str, Kernel] = {}
    for r in sample:
        kernels[r["kernel_id"]] = Kernel(r["so_path"])
    probe = NvmlProbe(uuid=env["hardware_extra"]["uuid"], index=0)

    tele_proc, tele_file = start_telemetry(env["device_index"])
    print(f"[telemetry] {TELEMETRY} (1초 간격)")

    # --- alignment 가드가 실제로 필요한지 확인 (측정 전에) ---------------
    print("\n[가드 검증] (1024,4096,4100) 에 a888 커널을 강제로 물려본다")
    try:
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
    drift_kernel_id = picked[0]["kernel_id"]
    last_drift = 0.0
    t_start = time.time()

    stats = Counter()
    n = 0
    try:
        with RESULTS.open("a") as out:
            for (krow, p, rc) in jobs:
                row = measure_one(ctx, kernels[krow["kernel_id"]], krow, p, rc,
                                  probe, env, launch_overhead_ms, hw.regs_per_sm)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                stats[row["status"]] += 1
                n += 1

                # cuBLAS 참조는 형상당 한 번
                key = (p.M, p.N, p.K)
                if key not in cublas_cache:
                    cublas_cache[key] = measure_cublas(ctx, p, probe, env, out)

                if time.time() - last_drift > drift_period:
                    last_drift = time.time()
                    drift_check(ctx, kernels[drift_kernel_id], drift_kernel_id,
                                probe, env)

                if n % 50 == 0 or n == len(jobs):
                    el = time.time() - t_start
                    print(f"  {n}/{len(jobs)}  {el / 60:.1f}분  "
                          f"{dict(stats)}", flush=True)

            # --- 재현성 검증 ---------------------------------------------
            print("\n[재현성] 무작위 20개 조합 재측정")
            repro = reproducibility(ctx, kernels, jobs, probe, env,
                                    launch_overhead_ms, seed, hw.regs_per_sm)
    finally:
        tele_proc.terminate()
        try:
            tele_proc.wait(timeout=5)
        except Exception:
            tele_proc.kill()
        tele_file.close()
        probe.close()
        ctx.close()

    report(stats, repro, clock_locked)
    return 0


def measure_one(ctx, kern, krow, p: Problem, rc: RuntimeConfig, probe, env,
                launch_overhead_ms: float, regs_per_sm: int) -> dict:
    from measure.runner import KtProblemC

    parallel = rc.split_k_mode == "parallel"
    kp = KtProblemC(p.M, p.N, p.K, rc.split_k, 1 if parallel else 0)

    row = {
        "kernel_id": krow["kernel_id"],
        "problem": {"M": p.M, "N": p.N, "K": p.K, "dtype": p.dtype},
        "runtime": {"split_k": rc.split_k, "split_k_mode": rc.split_k_mode},
        "env_hash": env["env_hash"],
        "clock_locked": env["clock_locked"],
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


def drift_check(ctx, kern, kernel_id, probe, env) -> None:
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
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }) + "\n")


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
