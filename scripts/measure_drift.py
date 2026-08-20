#!/usr/bin/env python3
"""드리프트 3 값 측정 — `docs/new_environment_checklist.md` 의 ⛔ G-4 게이트.

    python3 scripts/measure_drift.py                       # env.json 의 GPU
    python3 scripts/measure_drift.py --gpu GPU-93284c84-...
    python3 scripts/measure_drift.py --max-touch 3000 --step 100

## 무엇을 재는가

한 프로세스가 서로 다른 커널을 많이 실행하면 **런치당 상수 오버헤드**가
커진다. CUDA 의 LAZY 모듈 로딩 때문이고, 디바이스 모듈은 각 커널의 **첫
런치** 때 하나씩 올라간다. 상수이므로 긴 커널에서는 안 보이고 짧은 커널을
통째로 삼킨다.

`t = a + b · work` 로 분해한다. 모듈 압력이 런치 경로에 작용하면 `a` 만
커지고 `b`(계산 속도)는 그대로다. 그것이 "발열이나 클럭이 아니라 모듈
누적" 이라는 판별이기도 하다.

| 값 | 뜻 | A6000 / CUDA 12.4 |
|---|---|---|
| **문턱** | 몇 개까지는 `a` 가 전혀 안 늘어나는가 | ~1,200 모듈 |
| **기울기** | 문턱 위에서 모듈 1,000 개당 `a` 증가량 | ~75 us |
| **왜곡 배율** | 짧은 커널 대 긴 커널의 상대 드리프트 비 | ~100 배 |

⚠️ **A6000/12.4 값을 그대로 가정하지 마라.** GPU·드라이버·CUDA 버전마다
다르다. 특히 LAZY 로딩 동작은 CUDA 메이저 버전에서 바뀔 수 있다.

## 왜 짧은 커널이 반드시 필요한가

A6000 에서 4096³ 프로브 하나로 감시하며 "+5.06 % 니까 견딜 만하다" 고
판단했는데, 같은 시각 512³ 측정은 **+1380 % 오염**돼 있었다. 감시 지표를
잘못 골랐다는 사실 자체를 몰랐다.

그래서 이 스크립트는 **가장 짧은 프로브가 `--min-short-us` 보다 길면
시작하지 않는다.** 조용히 덜 민감한 상태로 도는 것을 막는다.

## 출력

`results/drift_profile.json`. `rehearse.py` 의 `segments.kernels` 를 여기서
정한다 (권장값 = 문턱의 40 %).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kerneltab.build import paths
from kerneltab.build.compile import build_ctx_so
from kerneltab.core import device, noise, records
from kerneltab.measure.gpu_state import NvmlProbe
from kerneltab.measure.runner import Ctx, Kernel, KtProblemC

#: `a` 를 뽑기 위한 형상. **네 자릿수의 work 범위**를 덮어야 절편이
#: 안정적으로 나온다. 가장 작은 것이 "짧은 커널" 역할을 겸한다.
PROBE_SHAPES = [256, 512, 1024, 2048, 4096]

#: 모듈을 올리기 위한 "터치" 문제. 작을수록 좋다 — 우리는 시간을 재는 것이
#: 아니라 **모듈 로드만** 시키려는 것이다.
#:
#: 여러 크기를 두는 이유: 큰 타일이나 split-k 커널은 아주 작은 문제에서
#: `can_implement` 가 거절한다. 스모크 실행에서 200개 중 16개(8%)가 그렇게
#: 빠졌다. 빠지면 **압력 축이 실제보다 짧아지고 문턱이 틀린다.**
#: 문제 크기는 모듈 로드와 무관하므로 키워서 다시 시도한다.
TOUCH_SIZES = (256, 512, 1024)

#: 프로브 커널 수. 여러 개를 쓰는 이유는 `a` 증가가 커널마다 1.5 배쯤
#: 갈리기 때문이다 (A6000). 하나만 보면 그 커널의 특성을 문턱으로 착각한다.
N_PROBES = 5

#: 라운드마다 프로브를 몇 번 재고 중앙값을 쓸지.
REPS = 3

#: 가장 짧은 프로브가 이보다 길면 시작하지 않는다 (us).
MIN_SHORT_US = 50.0

#: 문턱 판정: `a` 가 기준선 + k·sigma 를 넘고 **그 뒤로 계속** 넘어 있으면
#: 그 지점이 문턱이다. 한 점만 튀는 것은 노이즈다.
KNEE_SIGMA_K = 3.0

#: 권장 세그먼트 크기 = 문턱 × 이 비율.
SEGMENT_FRACTION = 0.40

OUT = paths.RESULTS_DIR / "drift_profile.json"


@dataclass
class Fit:
    a_us: float          # 절편 = 런치당 상수 (us)
    b_us_per_gflop: float
    r2: float
    n: int


def _wls(work: list[float], t_ms: list[float]) -> Fit:
    """`t = a + b·work` 가중 최소제곱.

    가중치는 `core/noise.py` 의 노이즈 바닥에서 온다. 균등 가중을 쓰면 가장
    큰 형상이 회귀를 지배해 **절편이 그 하나의 잡음에 끌려간다** — 우리가
    알고 싶은 것이 바로 그 절편이다.
    """
    w = [1.0 / max(noise.noise_floor_ms(t), 1e-9) ** 2 for t in t_ms]
    sw = sum(w)
    mx = sum(wi * x for wi, x in zip(w, work, strict=True)) / sw
    my = sum(wi * y for wi, y in zip(w, t_ms, strict=True)) / sw
    sxx = sum(wi * (x - mx) ** 2 for wi, x in zip(w, work, strict=True))
    sxy = sum(wi * (x - mx) * (y - my)
              for wi, x, y in zip(w, work, t_ms, strict=True))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    ss_res = sum(wi * (y - (a + b * x)) ** 2
                 for wi, x, y in zip(w, work, t_ms, strict=True))
    ss_tot = sum(wi * (y - my) ** 2 for wi, y in zip(w, t_ms, strict=True))
    return Fit(a_us=a * 1000.0, b_us_per_gflop=b * 1000.0,
               r2=1 - ss_res / ss_tot if ss_tot else float("nan"),
               n=len(work))


def probe_once(ctx, kern, M: int) -> float | None:
    ctx.prepare_problem(M, M, M)
    kp = KtProblemC(M, M, M, 1, 0)
    bufs = ctx.buffers(kern.workspace_bytes(kp), False)
    st, h = kern.prepare(kp, bufs)
    if st != 0 or not h:
        return None
    try:
        st, m = ctx.measure(kern.launch_addr, h, 0)
    finally:
        kern.release(h)
    return m.time_ms if st == 0 else None


def measure_probes(ctx, probes: dict) -> dict:
    """{kernel_id: {M: time_ms}} — 각 3회 중앙값."""
    out = {}
    for kid, kern in probes.items():
        per = {}
        for M in PROBE_SHAPES:
            ts = [probe_once(ctx, kern, M) for _ in range(REPS)]
            ts = [t for t in ts if t]
            if ts:
                per[M] = statistics.median(ts)
        if len(per) >= 3:
            out[kid] = per
    return out


def touch(ctx, kern) -> str:
    """커널을 **한 번만** 런치해 디바이스 모듈을 올린다. `"ok"` 또는 실패 사유.

    시간은 재지 않는다. 측정 프로토콜을 쓰면 워밍업 반복까지 돌아 훨씬
    느리고, 우리가 원하는 것은 모듈 로드뿐이다.

    ⚠️ 실패를 **삼키지 않는다.** 압력 축이 곧 x 축이므로, 터치가 조용히
    실패하면 "모듈 3,000개" 라고 적어 놓고 실제로는 1,200개인 상태로
    판정하게 된다. 문턱이 통째로 틀린다.
    """
    why = "none"
    for M in TOUCH_SIZES:
        ctx.prepare_problem(M, M, M)
        kp = KtProblemC(M, M, M, 1, 0)
        try:
            bufs = ctx.buffers(kern.workspace_bytes(kp), False)
            st, h = kern.prepare(kp, bufs)
        except (MemoryError, RuntimeError, OSError) as e:
            why = type(e).__name__
            continue
        if st != 0 or not h:
            why = f"prepare={st}"
            continue
        try:
            if ctx.run_once(kern.launch_addr, h, 0) == 0:
                return "ok"
            why = "launch"
        finally:
            kern.release(h)
    return why


def find_knee(ns: list[int], a_us: list[float],
              sigma: float) -> tuple[int | None, float]:
    """`a` 가 평탄한 마지막 지점. `(문턱, 기준선)`.

    한 점만 튀는 것은 노이즈이므로 **그 뒤로 계속 넘어 있어야** 문턱으로
    본다. A6000 에서 무릎은 선형 증가의 시작이 아니라 **문턱**이었다 —
    1,100 까지 증가가 정확히 0이고 1,200~1,300 에서 다섯 커널이 동시에
    꺾였다.
    """
    if len(ns) < 4:
        return None, (a_us[0] if a_us else 0.0)
    base = statistics.median(a_us[: max(len(a_us) // 4, 2)])
    lim = base + KNEE_SIGMA_K * sigma
    for i in range(len(ns)):
        if all(a > lim for a in a_us[i:]):
            return ns[i], base
    return None, base


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpu", default=None, help="GPU UUID. 생략하면 env.json")
    ap.add_argument("--max-touch", type=int, default=3000)
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--probes", type=int, default=N_PROBES)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--time-budget", type=float, default=3.0, help="시간 (h)")
    ap.add_argument("--min-short-us", type=float, default=MIN_SHORT_US)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    if not paths.ENV_JSON.exists():
        print("results/env.json 이 없다. scripts/phase0_env.py 를 먼저 돌려라.")
        return 2
    env = json.loads(paths.ENV_JSON.read_text())
    if args.gpu:
        env.setdefault("hardware_extra", {})["uuid"] = args.gpu
    idx, uuid = device.resolve_device(env)

    rows = [r for r in records.iter_records(paths.RESULTS_DIR / "kernels.jsonl",
                                            records.ALL)
            if r.get("build_status") == "ok"
            and paths.kernel_so(r["kernel_id"]).exists()]
    if len(rows) < args.probes + args.step * 4:
        print(f"빌드된 커널이 {len(rows)}개뿐이다. "
              f"scripts/build_kernels.py 를 먼저 돌려라.")
        return 2

    # 프로브는 **결정론적으로** 고르고 터치 목록에서 뺀다. 프로브 자신의
    # 모듈은 처음부터 올라가 있으므로 압력 축에 포함되면 안 된다.
    rng = random.Random(args.seed)
    order = sorted(rows, key=lambda r: r["kernel_id"])
    rng.shuffle(order)
    probe_rows, touch_rows = order[: args.probes], order[args.probes:]
    touch_rows = touch_rows[: args.max_touch]

    print(f"GPU        idx={idx} uuid={uuid[:20]}...")
    print(f"프로브     {len(probe_rows)}개 x 형상 {PROBE_SHAPES}")
    print(f"터치 대상  {len(touch_rows):,}개  (step={args.step})")
    print(f"시간 예산  {args.time_budget}h\n")

    # 없거나 낡았으면 여기서 빌드한다. 손으로 복사하게 두면 잊는다 —
    # 실제로 캠페인 디렉토리를 새로 만들면서 .so 만 빠져 죽었다.
    # (낡은 .so 는 Ctx 의 ABI 핸드셰이크가 따로 거부한다.)
    ctx = Ctx(build_ctx_so(env), 0)
    ctx.set_protocol(env)
    nv = NvmlProbe(uuid=uuid, index=0)
    probes = {r["kernel_id"]: Kernel(paths.kernel_so(r["kernel_id"]))
              for r in probe_rows}

    # --- 워밍업 --------------------------------------------------------
    # 유휴 상태에서 재면 메모리 클럭이 램프업 전이라 최대 66% 느리다.
    print("워밍업 ...", flush=True)
    for _ in range(3):
        measure_probes(ctx, probes)

    first = measure_probes(ctx, probes)
    if not first:
        print("프로브 측정에 전부 실패했다.")
        return 2
    shortest = min(min(v.values()) for v in first.values()) * 1000
    print(f"가장 짧은 프로브 {shortest:.1f} us")
    if shortest > args.min_short_us:
        print(f"\n⛔ 가장 짧은 프로브가 {args.min_short_us} us 보다 길다.")
        print("   드리프트는 런치당 상수라 긴 커널에서는 보이지 않는다.")
        print("   A6000 에서 4096³ 하나로 감시하다 512³ 가 +1380% 오염된 것을")
        print("   놓쳤다. PROBE_SHAPES 에 더 작은 형상을 넣고 다시 돌려라.")
        return 3

    # --- 압력을 올리며 반복 ---------------------------------------------
    t0 = time.time()
    steps: list[dict] = []
    touch_fail: Counter = Counter()
    touched = 0
    i = 0
    while True:
        snap = nv.snapshot()
        meas = measure_probes(ctx, probes)
        fits = {}
        for kid, per in meas.items():
            Ms = sorted(per)
            work = [2.0 * M ** 3 / 1e9 for M in Ms]      # GFLOP
            fits[kid] = _wls(work, [per[M] for M in Ms])
        steps.append({
            "n_modules": touched,
            "elapsed_s": round(time.time() - t0, 1),
            "sm_clock_mhz": snap.get("sm_clock_mhz"),
            "mem_clock_mhz": snap.get("mem_clock_mhz"),
            "gpu_temp_c": snap.get("gpu_temp_c"),
            "times_ms": {k: {str(m): v for m, v in per.items()}
                         for k, per in meas.items()},
            "fit": {k: {"a_us": round(f.a_us, 4),
                        "b_us_per_gflop": round(f.b_us_per_gflop, 6),
                        "r2": round(f.r2, 6)} for k, f in fits.items()},
        })
        a_med = statistics.median(f.a_us for f in fits.values())
        print(f"  모듈 {touched:5,d}  a={a_med:8.2f}us  "
              f"{snap.get('gpu_temp_c')}C {snap.get('sm_clock_mhz')}MHz  "
              f"{(time.time() - t0) / 60:.1f}분", flush=True)

        if i * args.step >= len(touch_rows):
            break
        if (time.time() - t0) / 3600 > args.time_budget:
            print("시간 예산 소진. 여기까지로 판정한다.")
            break
        batch = touch_rows[i * args.step:(i + 1) * args.step]
        for r in batch:
            try:
                k = Kernel(paths.kernel_so(r["kernel_id"]))
            except OSError as e:
                touch_fail[f"dlopen:{type(e).__name__}"] += 1
                continue
            why = touch(ctx, k)
            if why == "ok":
                touched += 1
            else:
                touch_fail[why] += 1
        i += 1

    nv.report()

    # --- 판정 -----------------------------------------------------------
    ns = [s["n_modules"] for s in steps]
    a_med = [statistics.median(f["a_us"] for f in s["fit"].values())
             for s in steps]
    b_med = [statistics.median(f["b_us_per_gflop"] for f in s["fit"].values())
             for s in steps]
    # 평탄 구간의 흔들림에서 sigma 를 잡는다 (전체로 잡으면 드리프트가
    # 분모를 부풀려 문턱을 못 찾는다 — R-6 에서 같은 실수를 했다).
    head = a_med[: max(len(a_med) // 4, 2)]
    sigma = statistics.pstdev(head) if len(head) >= 2 else 0.0
    knee, base_a = find_knee(ns, a_med, sigma)

    slope = None
    if knee is not None:
        xs = [n for n in ns if n >= knee]
        ys = [a for n, a in zip(ns, a_med, strict=True) if n >= knee]
        if len(xs) >= 3:
            mx, my = statistics.fmean(xs), statistics.fmean(ys)
            sxx = sum((x - mx) ** 2 for x in xs)
            if sxx:
                slope = 1000.0 * sum((x - mx) * (y - my)
                                     for x, y in zip(xs, ys, strict=True)) / sxx

    # 왜곡 배율 — 짧은 형상과 긴 형상의 **상대** 드리프트 비
    def rel(M):
        f = [s["times_ms"][k][str(M)] for k in first
             for s in (steps[0],) if str(M) in s["times_ms"].get(k, {})]
        l_ = [s["times_ms"][k][str(M)] for k in first
              for s in (steps[-1],) if str(M) in s["times_ms"].get(k, {})]
        if not f or not l_:
            return None
        return statistics.median(l_) / statistics.median(f) - 1

    r_short, r_long = rel(PROBE_SHAPES[0]), rel(PROBE_SHAPES[-1])
    distortion = (abs(r_short / r_long)
                  if r_short is not None and r_long not in (None, 0) else None)

    rec = int(knee * SEGMENT_FRACTION) if knee else None
    out = {
        "gpu": env["hardware"]["name"], "uuid": uuid,
        "arch": env["hardware"]["arch"],
        "cuda": env["cuda"]["nvcc_version"],
        "driver": env["cuda"].get("driver_version"),
        "cutlass_commit": (env.get("cutlass") or {}).get("commit"),
        # ⚠️ 어느 조건에서 잰 값인지 함께 남긴다. 드리프트 3값은
        #    GPU 만이 아니라 **툴체인마다** 달라진다 — 같은 A6000 을
        #    CUDA 12.4 와 13.3 에서 재니 문턱이 1,154 -> 962 였다.
        "env_hash": env.get("env_hash"),
        "env_hash_v2": env.get("env_hash_v2"),
        "env_hash_def_version": env.get("env_hash_def_version"),
        "cutlass_version": (env.get("cutlass") or {}).get("version"),
        "driver_kernel_mode": (env.get("cuda") or {}).get("driver_kernel_mode"),
        "driver_user_mode": (env.get("cuda") or {}).get("driver_user_mode"),
        "forward_compat": (env.get("cuda") or {}).get("forward_compat"),
        "timestamp": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"),
        "threshold_kernels": knee,
        "us_per_1k_modules": round(slope, 2) if slope is not None else None,
        "distortion_ratio": round(distortion, 1) if distortion else None,
        "recommended_segment_kernels": rec,
        "baseline_a_us": round(base_a, 3),
        "a_sigma_us": round(sigma, 3),
        "b_drift_pct": (round(100 * (b_med[-1] / b_med[0] - 1), 2)
                        if b_med and b_med[0] else None),
        "rel_drift_short_pct": (round(100 * r_short, 2)
                                if r_short is not None else None),
        "rel_drift_long_pct": (round(100 * r_long, 2)
                               if r_long is not None else None),
        "shapes": PROBE_SHAPES, "step": args.step, "reps": REPS,
        "n_probes": len(probes), "max_modules": ns[-1] if ns else 0,
        "nvml_failures": nv.failures(),
        # 터치 실패를 남긴다. x 축이 실제보다 짧으면 문턱이 통째로 틀린다.
        "touch_attempted": min(len(touch_rows), i * args.step),
        "touch_failed": dict(touch_fail),
        "steps": steps,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))

    print("\n" + "=" * 68)
    print(f"문턱        {knee or '검출 안 됨'} 모듈"
          f"   (기준선 a={base_a:.1f}us, sigma={sigma:.1f}us)")
    print(f"기울기      {out['us_per_1k_modules']} us / 모듈 1,000개")
    print(f"왜곡 배율   {out['distortion_ratio']}배  "
          f"(짧은 {out['rel_drift_short_pct']}% vs 긴 {out['rel_drift_long_pct']}%)")
    print(f"b 변화      {out['b_drift_pct']}%  "
          "<- 0 에 가까워야 한다. 크면 모듈이 아니라 클럭/발열이다.")
    if touch_fail:
        n_f = sum(touch_fail.values())
        att = min(len(touch_rows), i * args.step)
        print(f"\n!! 터치 실패 {n_f:,}/{att:,}회 ({100 * n_f / max(att, 1):.1f}%) "
              f"— 압력 축이 그만큼 짧다.")
        for why, c in touch_fail.most_common(5):
            print(f"     {why:24s} {c:,}회")
    print(f"\n권장 segments.kernels = {rec}  (문턱의 {SEGMENT_FRACTION:.0%})")
    if knee is None:
        print("\n⚠️ 문턱을 찾지 못했다. 둘 중 하나다:")
        print("   (a) 이 GPU/CUDA 조합에는 드리프트가 없다 — 좋은 소식이지만")
        print("       --max-touch 를 늘려 더 밀어 보고 확인하라.")
        print("   (b) 압력을 충분히 올리지 못했다 (최대 "
              f"{ns[-1] if ns else 0} 모듈).")
        print("   **드리프트가 없다고 단정하지 마라.**")
    print(f"-> {args.out}")
    ctx.close()
    return 0 if knee else 4


if __name__ == "__main__":
    raise SystemExit(main())
