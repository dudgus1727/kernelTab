#!/usr/bin/env python3
"""**측정 전에** `below_launch_overhead` 로 죽을 형상을 찾는다.

speed-of-light 하한 — 어떤 config 도 이보다 빠를 수 없는 시간 — 을 내고
`below_launch_overhead` 문턱(`3 x launch_bracketed_grid_ms`)과 비교한다.

    sol = max(2MNK / peak_eff,  (MK + KN + MN) x 2 / bw_eff)

## ★ 심각도가 균일하지 않다

    문턱의 100 % 아래   어떤 config 도 문턱 위로 못 올라간다 -> 통째로 결측
    ★ 70~90 %          **느린 config 는 살고 빠른 config 만 잘린다**
                       정답 쪽만 검열되므로 가장 나쁘다 — 남은 값으로 순위를
                       매기면 틀린 답이 나오는데 그것이 결측으로 보이지 않는다

100 % 아래보다 70~90 % 가 더 위험하다. 5090 에서 `1024³`(75 %)을 남기는 것이
`512³`(9 %)을 남기는 것보다 나쁘다고 판단해 층 E 사다리를 통째로 밀었다.

## ★ 형상 그리드와 앵커를 **둘 다** 본다

5090 에서 그리드만 고치고 `DRIFT_SHAPES` 를 놓쳐 G-7 이 세 항목 실패했다.
앵커는 표의 일부가 아니라 **측정 조건의 기록**이라 눈에 덜 띈다.

    python3 tools/sol_screen.py
    python3 tools/sol_screen.py --warn-frac 0.5   # 여유 배수를 더 크게 본다
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kerneltab.core import paths
from kerneltab.core.hardware import hardware_from_env
from kerneltab.core.shapes import all_layers
from kerneltab.core.types import Problem


def sol_ms(p: Problem, peak_tflops: float, bw_gbps: float) -> tuple[float, str]:
    """(SOL 하한 ms, 무엇이 지배하는가)."""
    flop = 2.0 * p.M * p.N * p.K
    byts = (p.M * p.K + p.K * p.N + p.M * p.N) * 2.0
    t_c = flop / (peak_tflops * 1e12) * 1e3
    t_m = byts / (bw_gbps * 1e9) * 1e3
    return (max(t_c, t_m), "compute" if t_c >= t_m else "memory")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warn-frac", type=float, default=1.5,
                    help="문턱 대비 이 배수 미만이면 경고 (기본 1.5배)")
    args = ap.parse_args()

    env = json.loads(paths.ENV_JSON.read_text())
    hw = hardware_from_env(env)
    peak = env.get("peak_tflops_f16_effective") or hw.peak_tflops_f16
    bw = env.get("bandwidth_gbps_effective") or hw.bandwidth_gbps
    thr = env.get("below_launch_overhead_ms")
    if not thr:
        thr = 3.0 * env["launch_overhead"]["launch_bracketed_grid_ms"]

    print(f"{hw.name} ({hw.arch})  env_hash {env['env_hash'][:8]}  "
          f"clock_locked={env.get('clock_locked')}")
    print(f"실효 피크 {peak:.1f} TFLOP/s  실효 대역폭 {bw:.1f} GB/s")
    print(f"below_launch_overhead 문턱 {thr * 1e3:.3f} us "
          f"(= 3 x {thr / 3 * 1e3:.3f} us)\n")

    # 앵커는 표가 아니라 조건의 기록이라 눈에 덜 띈다 — 그리드와 같은 표에 둔다.
    from importlib import import_module
    sys.path.insert(0, str(paths.REPO_ROOT / "scripts"))
    drift = import_module("rehearse").DRIFT_SHAPES

    groups = dict(all_layers(hw))
    groups["★ anchors (DRIFT_SHAPES)"] = list(drift)

    n_dead = n_warn = 0
    for name, probs in groups.items():
        rows = []
        for p in probs:
            s, why = sol_ms(p, peak, bw)
            frac = s / thr
            rows.append((frac, p, s, why))
        rows.sort(key=lambda r: r[0])
        # ⛔ 걸린 것을 **전부** 찍는다. 상위 몇 개만 보이면 개수와 목록이
        #    어긋나고, 그것이 바로 이 도구가 막으려는 실패다.
        bad = [r for r in rows if r[0] < args.warn_frac]
        print(f"{name}  ({len(probs)} 형상, 문턱 {args.warn_frac}배 미만 "
              f"{len(bad)}개)")
        for frac, p, s, why in bad:
            if frac < 1.0:
                # ⛔ SOL 이 문턱 아래 = **빠른 config 부터 잘린다.**
                #    70~90 % 대가 최악이다 (느린 것만 살아남아 정답이 검열된다).
                flag = ("  ⛔ 빠른 config 부터 잘린다"
                        + (" ★ 최악 구간" if frac >= 0.5 else " (거의 전부 결측)"))
                n_dead += 1
            else:
                # 문턱 **위**면 어떤 config 도 잘리지 않는다. 다만 런치
                # 오버헤드가 측정 시간의 큰 몫이라 분해능이 나빠진다.
                flag = (f"  ⚠️ 여유 {frac:.2f}배 — 잘리지는 않지만 런치가 "
                        f"시간의 {100 / frac / 3:.0f}% 다")
                n_warn += 1
            print(f"    {p.M:6d}x{p.N:6d}x{p.K:6d}  SOL {s * 1e3:9.2f} us "
                  f"= 문턱의 {frac * 100:7.1f} %  ({why}){flag}")
        rest = rows[len(bad):]
        if rest:
            print(f"    나머지 {len(rest)}개는 문턱의 "
                  f"{rest[0][0] * 100:.0f} % ~ {rest[-1][0] * 100:.0f} % 다")
    print()
    if n_dead:
        print(f"⛔ 문턱 아래 {n_dead}개 — 형상 그리드(core/shapes.py)나 "
              "앵커(rehearse.DRIFT_SHAPES)를 **측정 전에** 고쳐라.")
        return 1
    if n_warn:
        print(f"문턱 아래 0개. 여유가 {args.warn_frac}배 미만인 것 "
              f"{n_warn}개는 **잘리지 않는다** — 분해능만 주의하면 된다.")
    else:
        print("모든 형상과 앵커가 문턱의 "
              f"{args.warn_frac}배 위다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
