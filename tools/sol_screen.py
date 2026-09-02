#!/usr/bin/env python3
"""SOL(speed-of-light) 하한으로 `below_launch_overhead` 위험 형상을 **측정 전에** 찾는다.

    python3 tools/sol_screen.py                    # results/env.json 을 읽는다
    python3 tools/sol_screen.py --json

## 왜 측정 전인가

측정한 뒤에 알면 그 형상은 다음 캠페인까지 비어 있다. 5090 에서 6 형상이
문턱 아래였고 **측정 전에 고쳐 실패 0 건**으로 끝냈다.

## 심각도가 균일하지 않다

    문턱의 100 % 아래   측정해도 순위가 없다 — 결측과 같다
    ★ 70~90 %          느린 config 는 살고 빠른 config 만 잘린다.
                       **정답 쪽만 검열**되므로 가장 나쁘다 — 남은 값으로
                       순위를 매기면 틀린 답이 나오는데 결측으로 안 보인다

## ⛔ 형상 그리드와 앵커를 **둘 다** 본다

5090 에서 그리드만 고치고 `DRIFT_SHAPES` 를 놓쳐 G-7 이 세 항목 실패했다.
앵커가 런치 오버헤드에 지배되면 값이 이산적으로 튄다 (12.288 / 14.336 us 를
오갔는데 각각 브래킷 오버헤드의 3.0 배와 3.5 배였다).

## ⚠️ SOL 은 하한이다

`SOL >= 문턱` 이면 안전이 **보장**되고, 아래면 그 형상이 문턱 밑으로 갈
**가능성**이 있다는 뜻이다. 실제 시간은 항상 SOL 보다 크다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kerneltab.core import paths                       # noqa: E402
from kerneltab.core.hardware import hardware_from_env  # noqa: E402
from kerneltab.core.shapes import all_layers           # noqa: E402


def sol_us(M: int, N: int, K: int, peak_tflops: float, bw_gbps: float) -> float:
    """계산 하한과 대역폭 하한 중 큰 쪽 (us). 실제 시간은 항상 이보다 크다."""
    flops = 2.0 * M * N * K
    nbytes = (M * K + K * N + M * N) * 2          # fp16
    return max(flops / (peak_tflops * 1e12), nbytes / (bw_gbps * 1e9)) * 1e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None, help="env.json 경로")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    env = json.loads(Path(a.env or paths.ENV_JSON).read_text())
    hw = hardware_from_env(env)
    # ★ 실효값을 쓴다. 스펙 피크로 재면 SOL 이 작게 나와 위험을 놓친다.
    peak = env.get("peak_tflops_f16_effective") or hw.peak_tflops_f16
    bw = env.get("bandwidth_gbps_effective") or hw.bandwidth_gbps
    thr_us = env["below_launch_overhead_ms"] * 1000.0

    # 앵커도 함께 본다. rehearse 를 import 하면 GPU 초기화가 걸리므로
    # 형상 정의만 소스에서 읽는다.
    import re
    src = (REPO_ROOT / "scripts" / "rehearse.py").read_text(encoding="utf-8")
    blk = src[src.index("DRIFT_SHAPES = ["):]
    blk = blk[:blk.index("]", blk.index("["))]
    anchors = [tuple(int(x) for x in m)
               for m in re.findall(r"Problem\((\d+),\s*(\d+),\s*(\d+)\)", blk)]

    rows = []
    for layer, ps in all_layers(hw).items():
        for p in ps:
            rows.append((layer, p.M, p.N, p.K))
    for (M, N, K) in anchors:
        rows.append(("anchor(DRIFT_SHAPES)", M, N, K))

    out = []
    seen = set()
    for layer, M, N, K in rows:
        key = (layer, M, N, K)
        if key in seen:
            continue
        seen.add(key)
        s = sol_us(M, N, K, peak, bw)
        out.append({"layer": layer, "shape": f"{M}x{N}x{K}", "sol_us": round(s, 2),
                    "frac_of_threshold": round(s / thr_us, 3)})

    bad = sorted([r for r in out if r["frac_of_threshold"] < 1.0],
                 key=lambda r: r["sol_us"])
    danger = [r for r in out if 0.7 <= r["frac_of_threshold"] < 1.0]

    if a.json:
        print(json.dumps({"threshold_us": thr_us, "peak_tflops_effective": peak,
                          "bandwidth_gbps_effective": bw,
                          "below": bad, "all": out}, ensure_ascii=False, indent=1))
    else:
        print(f"실효 피크   {peak} TFLOP/s")
        print(f"실효 대역폭 {bw} GB/s")
        print(f"문턱        {thr_us:.3f} us  (= 3 x launch_bracketed_grid)")
        print(f"형상        {len(out)}개 (층 항목 + 앵커)")
        if not bad:
            print("\n문턱 아래 형상 없음.")
        else:
            print(f"\n⛔ 문턱 아래 {len(bad)}개"
                  + (f" (그중 70~90% 구간 {len(danger)}개 — 정답 쪽만 검열된다)"
                     if danger else ""))
            print(f"\n{'층':22s} {'형상':22s} {'SOL(us)':>9s} {'문턱대비':>8s}")
            for r in bad:
                print(f"{r['layer']:22s} {r['shape']:22s} "
                      f"{r['sol_us']:9.2f} {100 * r['frac_of_threshold']:7.0f}%")
    return 4 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
