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


def spread_rows(samples: dict) -> list:
    """`{key: [측정값...]}` -> `[(변동폭, key, 값들)]`, 큰 순.

    ⛔ 예전에는 이 계산이 `main()` 안에 인라인으로 있었고 루프 변수가
       `_key` 인데 `spreads.append((spread, key, v))` 로 **바깥 스코프의
       `key`** 를 담았다. 그 `key` 는 측정 루프가 남긴 **마지막 조합**이라
       보고서의 여덟 줄이 전부 같은 이름을 달고 나왔다 — 시간이
       0.1485 ms 와 4.1667 ms 로 다른데 이름은 같았다.

       판정(`over_5pct` / `spread_max`)은 값에서 나오므로 멀쩡했다.
       **틀린 것은 "무엇이 흔들렸는가" 뿐이다** — 그런데 그것이 이 보고서를
       읽는 이유다. 5 % 를 넘는 조합이 나오면 엉뚱한 커널을 쫓게 된다.
       (`docs/decisions.md` 23 — 돌기는 도는데 틀린 종류)

       인라인이라 테스트를 붙일 자리가 없었다. 그래서 함수로 뺀다.
    """
    out = []
    for key, v in samples.items():
        if len(v) < 2:
            continue
        med = statistics.median(v)
        out.append(((max(v) - min(v)) / med if med else 0.0, key, v))
    out.sort(key=lambda r: r[0], reverse=True)
    return out


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
    # status-filter: 재측정 대상을 고르는 자리. 이미 산포가 넓다고 표시된
    # 줄을 다시 재면 안정성 점검의 귀무분포가 오염된다.
    okrows = [r for r in res
              if r.get("status") == "ok" and records.is_measurement(r)]

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

    # ⛔ **스윕과 같은 버퍼 상태에서 재야 한다.**
    #
    #    `rehearse.py` 는 세그먼트 시작에 모든 버퍼를 최대 크기로 선할당한다
    #    (`MEASURE_PATH_REVISION` 2/3). 여기서 안 하면 이 도구만 옛 지연
    #    경로로 재고, **재현성 통계가 실제 측정 경로의 것이 아니게 된다.**
    #
    #    실측으로 드러났다: pass1 에서 workspace 가 가장 큰 parallel split-K
    #    조합 두 개에만 **상수 0.58 ms** 가 붙었다 (10.67 ms 기준 5.5 %,
    #    14.64 ms 기준 3.9 % — 비율이 아니라 절대량이 같다). 별도 프로세스
    #    두 회차에서 pass1 값이 0.03 % 이내로 재현됐다.
    #
    #        회차1  5.62%  [11.2556, 10.6791, 10.6549]  (4096,4096,16384) sk12para
    #        회차2  5.55%  [11.2523, 10.6590, 10.6845]  같은 조합
    #
    #    ★ 이것은 기준을 푸는 것이 아니라 **조건을 맞추는 것**이다
    #      (`decisions.md` 29 의 구분). 바로 위 `measure()` 의 "정확도 검사용
    #      1회 실행" 주석과 같은 종류의 수정이다.
    from kerneltab.backends import get_backend as _get_backend
    from kerneltab.core.shapes import all_shapes as _all_shapes
    _sh = list(_all_shapes(hw))
    _bm = max(q.M for q in _sh)
    _bn = max(q.N for q in _sh)
    _bk = max(q.K for q in _sh)
    ctx.prepare_problem(_bm, _bn, _bk)
    _ws = (2 * max(q.M * q.N for q in _sh)
           * max(_get_backend(hw.arch).axis_space()["split_k"]))
    ctx.buffers(_ws, parallel=False)
    print(f"[버퍼] 최대 {_bm}x{_bn}x{_bk} + workspace {_ws / 2**30:.2f} GiB "
          f"선할당 (스윕과 같은 상태로 맞춘다)", flush=True)

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

    # ⛔ **버리는 사전 pass 하나.** 스윕과 같은 장치 상태에서 재기 위해서다.
    #
    #    실측 (H100, 2026-09-03): `sm90_tb256x256x64_w128x64x64_st3_swid8_a888`
    #    을 한 번 돌리면 `sm90_tb256x256x32_w128x64x32_st6_swid4_a888`
    #    (4096,4096,16384 sk12 parallel)이 그 프로세스 안에서 **그 뒤로 계속
    #    4.7 % 빨라진다** (11.22 -> 10.67 ms). 짝을 여섯 번 바꿔 확인했고
    #    (P=0/2/10/20/28 은 -0.03~+0.05 %), 형상이 같은 조합(P=0)은 아무 일도
    #    하지 않으므로 형상이 아니라 **커널**이 일으킨다.
    #
    #    ⚠️ 지속 부하도 온도도 아니다 — 같은 조합만 6 분간 연속으로 재면
    #       11.20~11.22 로 평평하다 (표류 -0.15 %, 클럭 1200/2619 고정).
    #
    #    스윕은 슬라이스마다 618 개 커널을 돌리므로 **항상 그 상태에서 잰다**
    #    (이 조합의 스윕 값 10.7277 / 10.7857 = 빠른 쪽). 실제로 스윕 데이터는
    #    슬라이스 앞 10 행이 중앙 -0.025 % 로 전이가 보이지 않는다.
    #    반면 recheck 는 차가운 프로세스에서 시작해 **pass1 만 다른 상태**에서
    #    잰다. G-7 5 번이 그것 때문에 실패했다.
    #
    #    ★ 기준을 푸는 것이 아니라 **조건을 맞추는 것**이다 (decisions 29).
    #      `results.jsonl` 에는 아무것도 안 쓰므로 캠페인 데이터는 그대로다.
    print(f"[예열] {len(pick)}개 조합 1회 (버림) — 스윕과 같은 장치 상태로 맞춘다",
          flush=True)
    for r in pick:
        pr, rt = r["problem"], r["runtime"]
        measure(r["kernel_id"], pr["M"], pr["N"], pr["K"],
                rt["split_k"], rt["split_k_mode"])

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

    spreads = spread_rows(samples)

    # pass 별 편차 — 첫 회만 튀는지(cold) 전반적으로 흔들리는지 구분
    per_pass = {}
    for v in samples.values():
        for i, x in enumerate(v):
            per_pass.setdefault(i, []).append(
                x / statistics.median(v) if statistics.median(v) else 1.0)
    print("\n  pass 별 상대값 (1.0 = 그 조합의 중앙값):")
    for i in sorted(per_pass):
        v = sorted(per_pass[i])
        print(f"    pass {i + 1}: median={v[len(v) // 2]:.4f}  "
              f"mean={sum(v) / len(v):.4f}  min={v[0]:.4f}  max={v[-1]:.4f}")
    # 원자료 — 판정에는 안 쓴다. 실패했을 때 "표집인가 계통인가" 를 사후에
    # 다시 물을 수 있어야 한다.
    raw = paths.RESULTS_DIR / "stability_rows.jsonl"
    with raw.open("w") as f:
        for key, v in samples.items():
            f.write(json.dumps({"key": [str(x) for x in key],
                                "times_ms": v}, ensure_ascii=False) + "\n")
    print(f"    원자료 {raw}")

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
        # ⛔ 어느 조합이 흔들렸는지 **파일에 남긴다.** 예전에는 stdout 에만
        #    있었고 그마저 이름이 틀렸다. 게이트가 "1건 초과" 라고만 하면
        #    무엇을 쫓아야 할지 알 수 없다.
        "worst": [{"spread": round(sp, 6), "kernel_id": k[0],
                   "M": k[1], "N": k[2], "K": k[3],
                   "split_k": k[4], "split_k_mode": k[5],
                   "times_ms": [round(x, 6) for x in v]}
                  for sp, k, v in spreads[:8]],
        "drift_times_ms": ts,
        "clock_locked": env["clock_locked"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
