#!/usr/bin/env python3
"""재현성 / 드리프트 재측정 — 간섭 없는 상태에서.

리허설 중 다른 GPU 에서 정확성 검사를 병행했고 측정 자체가 3.9분으로 짧아
드리프트 점검이 2회밖에 못 돌았다. 클럭이 고정되지 않은 환경에서 측정 노이즈가
실제로 얼마인지 알아야 Phase 3 의 반복 수와 status 임계를 정할 수 있다.

    python3 scripts/recheck_stability.py --passes 3 --minutes 10
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kerneltab.core import (
    device,
    paths,
    records,
)
from kerneltab.core import kernels as kernels_mod
from kerneltab.core.hardware import hardware_from_env
from kerneltab.core.types import KernelConfig, Problem

RESULTS = paths.RESULTS_DIR / "results.jsonl"
KERNELS = paths.RESULTS_DIR / "kernels.jsonl"
DRIFT = paths.RESULTS_DIR / "drift.jsonl"
OUT = paths.RESULTS_DIR / "stability.json"

DRIFT_SHAPE = Problem(4096, 4096, 4096)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="재측정할 조합 수")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--shapes", default=None,
                    help="느린 형상 검증용. 'MxNxK,MxNxK' 형식. 주면 "
                         "results.jsonl 표본 대신 이 형상들에서 조합을 뽑는다.")
    args = ap.parse_args()

    env = json.loads(paths.ENV_JSON.read_text())
    # P-2: UUID 가 권위다. 저장된 인덱스를 그대로 쓰면 컨테이너나
    #      CUDA_VISIBLE_DEVICES 가 설정된 환경에서 **다른 GPU 를 측정**한다.
    #      이미 설정돼 있으면 존중하고, 없는 UUID 면 명확히 실패한다.
    device.resolve_device(env)
    hw = hardware_from_env(env)

    from kerneltab.measure.gpu_state import NvmlProbe
    from kerneltab.measure.runner import Ctx, Kernel, KtProblemC

    kern = {r["kernel_id"]: r for r in
            (json.loads(l) for l in KERNELS.read_text().splitlines() if l.strip())}
    # ⚠️ 이 측정 조건의 줄만. 조건이 섞이면 재측정 대조가 어긋난다 (R-5).
    res = records.load_records(RESULTS, env["env_hash"])
    okrows = [r for r in res if r.get("status") == "ok" and r["kernel_id"] != "cublas"]

    rng = random.Random(env["shuffle_seed"] ^ 0xC0FFEE)
    if args.shapes:
        # min_reps 를 낮춘 뒤 느린 형상에서도 재현성이 유지되는지 확인한다.
        from kerneltab.backends import get_backend
        from kerneltab.core.config import (
            alignment_combos,
            alignments_for,
            enumerate_kernels,
            enumerate_runtimes,
        )
        from kerneltab.core.shapes import all_shapes
        from kerneltab.core.types import RuntimeConfig  # noqa: F401

        backend = get_backend(hw.arch)
        valid = {backend.kernel_id(c) for c in
                 enumerate_kernels(hw, backend, alignment_combos(all_shapes(hw)))}
        usable = [r for r in kern.values()
                  if r.get("build_status") == "ok"
                  and r["kernel_id"] in valid
                  and kernels_mod.launchable(r, hw.regs_per_sm)]
        shapes = []
        for tok in args.shapes.split(","):
            M, N, K = (int(x) for x in tok.strip().split("x"))
            shapes.append(Problem(M, N, K))
        cand = []
        for sp in shapes:
            a = alignments_for(sp)
            ks = [r for r in usable
                  if (r["align"]["a"], r["align"]["b"], r["align"]["c"]) == a]
            for r in rng.sample(ks, min(40, len(ks))):
                cfg = KernelConfig(
                    tile_m=r["tile"]["m"], tile_n=r["tile"]["n"],
                    tile_k=r["tile"]["k"], align_a=a[0], align_b=a[1],
                    align_c=a[2], arch=r["arch"],
                    ext=backend.ext_from_dict(r["ext"]))
                for rc2 in enumerate_runtimes(backend, sp, cfg):
                    cand.append({"kernel_id": r["kernel_id"],
                                 "problem": {"M": sp.M, "N": sp.N, "K": sp.K},
                                 "runtime": {"split_k": rc2.split_k,
                                             "split_k_mode": rc2.split_k_mode}})
        pick = rng.sample(cand, min(args.n, len(cand)))
        drift_kid = pick[0]["kernel_id"]
        print(f"느린 형상 모드: {[f'{s.M}x{s.N}x{s.K}' for s in shapes]} "
              f"후보 {len(cand)} 중 {len(pick)}개")
    else:
        pick = rng.sample(okrows, min(args.n, len(okrows)))
        drift_kid = min({r["kernel_id"] for r in okrows})

    ctx = Ctx(paths.ARTIFACT_DIR / "libkt_ctx.so", 0)
    ctx.set_protocol(env)
    probe = NvmlProbe(uuid=env["hardware_extra"]["uuid"], index=0)
    libs: dict[str, Kernel] = {}

    def get(kid):
        if kid not in libs:
            libs[kid] = Kernel(paths.kernel_so(kid))
        return libs[kid]

    def measure(kid, M, N, K, sk, mode):
        k = get(kid)
        par = mode == "parallel"
        kp = KtProblemC(M, N, K, sk, 1 if par else 0)
        ctx.prepare_problem(M, N, K)
        gz = k.grid_k(kp)
        bufs = ctx.buffers(k.workspace_bytes(kp), par)
        st, h = k.prepare(kp, bufs)
        if st != 0 or not h:
            return None
        try:
            # 실제 측정 경로(measure_one)와 동일하게 정확도 검사용 1회 실행을
            # 먼저 한다. 이게 빠지면 첫 측정만 cold 상태가 되어 재현성 통계가
            # 실제 경로보다 나쁘게 나온다.
            ctx.run_once(k.launch_addr, h, gz if par else 0)
            _, m = ctx.measure(k.launch_addr, h, gz if par else 0)
        finally:
            k.release(h)
        return m.time_ms

    samples: dict[tuple, list[float]] = defaultdict(list)
    drift_rows = []
    t0 = time.time()
    deadline = t0 + args.minutes * 60

    print(f"재현성 {len(pick)}개 조합 x {args.passes} 회, 드리프트 기준 커널 {drift_kid}")
    try:
        for p in range(args.passes):
            for r in pick:
                pr, rt = r["problem"], r["runtime"]
                key = (r["kernel_id"], pr["M"], pr["N"], pr["K"],
                       rt["split_k"], rt["split_k_mode"])
                t = measure(*key)
                if t:
                    samples[key].append(t)
                    if args.verbose:
                        prev = samples[key][0]
                        d = 100 * (t - prev) / prev if prev else 0
                        mark = "  <-- 이상" if abs(d) > 5 else ""
                        print(f"    p{p+1} {t:10.4f} ms ({d:+6.2f}%) "
                              f"({key[1]},{key[2]},{key[3]}) sk{key[4]}"
                              f"{key[5][:4]} {key[0][:44]}{mark}", flush=True)
            # 드리프트 기준: 매 pass 마다
            t = measure(drift_kid, DRIFT_SHAPE.M, DRIFT_SHAPE.N, DRIFT_SHAPE.K,
                        1, "serial")
            snap = probe.snapshot()
            row = {"kernel_id": drift_kid, "time_ms": t,
                   "sm_clock_mhz": snap["sm_clock_mhz"],
                   "gpu_temp_c": snap["gpu_temp_c"], "power_w": snap["power_w"],
                   "clock_locked": env["clock_locked"], "pass": p,
                   "problem": {"M": DRIFT_SHAPE.M, "N": DRIFT_SHAPE.N,
                               "K": DRIFT_SHAPE.K},
                   "timestamp": datetime.now(timezone.utc).isoformat()
                   .replace("+00:00", "Z")}
            drift_rows.append(row)
            with DRIFT.open("a") as f:
                f.write(json.dumps(row) + "\n")
            el = time.time() - t0
            print(f"  pass {p + 1}/{args.passes}  {el / 60:.1f}분  "
                  f"drift={t:.4f} ms  clk={snap['sm_clock_mhz']}MHz "
                  f"temp={snap['gpu_temp_c']}C", flush=True)
            # 남은 시간을 균등하게 쓴다 (온도가 오를 시간을 준다)
            if p < args.passes - 1:
                gap = (deadline - time.time()) / max(1, args.passes - 1 - p)
                if gap > 0:
                    time.sleep(min(gap, 300))
    finally:
        probe.close()
        ctx.close()

    spreads = []
    for _key, v in samples.items():
        if len(v) < 2:
            continue
        med = statistics.median(v)
        spread = (max(v) - min(v)) / med if med else 0
        spreads.append((spread, key, v))
    spreads.sort(reverse=True)

    # pass 별 편차 — 첫 회만 튀는지(cold) 전반적으로 흔들리는지 구분
    per_pass = {}
    for v in samples.values():
        for i, x in enumerate(v):
            per_pass.setdefault(i, []).append(
                x / statistics.median(v) if statistics.median(v) else 1.0)
    print("\n  pass 별 상대값 중앙값 (1.0 = 그 조합의 중앙값):")
    for i in sorted(per_pass):
        v = sorted(per_pass[i])
        print(f"    pass {i + 1}: median={v[len(v) // 2]:.4f}  "
              f"max={v[-1]:.4f}")

    print("\n" + "=" * 74)
    print(f"재현성: {len(spreads)}개 조합, {args.passes}회 측정")
    print("=" * 74)
    over5 = [s for s in spreads if s[0] > 0.05]
    allv = [s[0] for s in spreads]
    print(f"  변동폭 (max-min)/median:  median={statistics.median(allv) * 100:.2f}%  "
          f"p90={sorted(allv)[int(len(allv) * 0.9)] * 100:.2f}%  "
          f"max={max(allv) * 100:.2f}%")
    print(f"  5% 초과: {len(over5)}/{len(spreads)}")
    for sp, key, v in spreads[:8]:
        print(f"    {sp * 100:6.2f}%  {[round(x, 4) for x in v]}  "
              f"({key[1]},{key[2]},{key[3]}) sk{key[4]}{key[5][:4]} {key[0][:40]}")

    ts = [d["time_ms"] for d in drift_rows if d["time_ms"]]
    if len(ts) > 1:
        mean = statistics.mean(ts)
        print(f"\n드리프트 기준 config: {len(ts)}회  min={min(ts):.4f} "
              f"max={max(ts):.4f} mean={mean:.4f} "
              f"std={statistics.pstdev(ts):.5f} ms")
        print(f"  변동폭 = {100 * (max(ts) - min(ts)) / mean:.2f}%")

    OUT.write_text(json.dumps({
        "n_combos": len(spreads), "passes": args.passes,
        "spread_median": statistics.median(allv) if allv else None,
        "spread_max": max(allv) if allv else None,
        "over_5pct": len(over5),
        "drift_times_ms": ts,
        "clock_locked": env["clock_locked"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
