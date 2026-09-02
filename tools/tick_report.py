#!/usr/bin/env python3
"""`tick_probe` 의 원시 표본에서 이벤트 눈금을 판정한다.

⛔ **절대값이 격자 위에 있는가로 검증하지 마라.** 0.72 ms 짜리 값은 16 ns
눈금의 45,000 배라 부동소수 오차가 눈금을 넘어 쌓인다 — 격자가 맞는데도
설명력이 3 % 로 나온다 (5090 에서 32 ns 로 잘못 결론냈던 자리).
**간격으로 재고 정수 ns 로 스냅한 뒤** 검증한다. 판정 코어는
`kerneltab.core.noise.tick_grid()` 를 그대로 쓴다 — 같은 판정이 두 곳에
있으면 하나는 어긋난다 (decisions 12).

## 과대 추정을 직접 검사한다

후보는 관측된 **최소 간격의 배수**로만 생성되므로, 표본이 성기면 진짜
눈금의 2 배·3 배가 뽑힌다. 그래서 양쪽에서 조인다:

  아래에서  후보의 절반이 설명력을 유지하는가 (유지되면 더 세밀할 수 있다)
  위에서    ★ 후보의 **홀수 배수**에 값이 있는가
            홀수 배수가 관측되면 그 후보는 진짜 눈금의 짝수배일 수 없다

    python3 tools/tick_report.py /tmp/ticks.csv
"""
from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kerneltab.core.noise import (
    TICK_COVER_MIN, TICK_ON_GRID_TOL, odd_multiple_frac, tick_grid,
    tick_ms_observed,
)


def cover(values, tick_ms: float) -> float:
    """값들이 이 격자 위에 있는 비율. `tick_grid` 과 같은 허용오차를 쓴다."""
    if tick_ms <= 0:
        return 0.0
    return sum(1 for x in values
               if abs(x / tick_ms - round(x / tick_ms)) < TICK_ON_GRID_TOL) / len(values)


# ⛔ 홀수 배수 판정은 `core.noise.odd_multiple_frac()` 하나뿐이다.
#    여기 다시 구현하지 마라 — `tools/tick_probe.py`(실측 경로 수집 +
#    부분표본 재추정)도 같은 함수를 쓴다. 두 도구가 서로 다른 허용오차로
#    판정하면 한쪽이 격자로 인정한 값을 다른 쪽이 세지 않는다.


def analyze(label: str, values: list[float]) -> float | None:
    uniq = sorted(set(round(v, 9) for v in values))
    if len(uniq) < 2:
        print(f"\n[{label}] 고유값 {len(uniq)}개 — 눈금을 볼 수 없다")
        return None
    base = tick_ms_observed(uniq)
    tick, frac = tick_grid(uniq)
    print(f"\n[{label}] 표본 {len(values)}  고유값 {len(uniq)}  "
          f"중앙 {statistics.median(values) * 1e3:.3f} us  "
          f"최소간격 {base * 1e6:.4g} ns")
    print(f"  {'후보(ns)':>10s} {'설명력':>7s} {'홀수배수':>8s}   판정")
    picked = None
    for mult in (0.5, 1, 2, 3, 4, 6, 8):
        ns = round(base * mult * 1e6)
        if ns <= 0:
            continue
        cand = ns * 1e-6
        c, odd = cover(uniq, cand), odd_multiple_frac(uniq, cand)
        mark = ""
        if tick and abs(cand - tick) < 1e-12:
            mark, picked = "  <- tick_grid 채택", cand
        elif c < TICK_COVER_MIN:
            mark = f"  설명력 {TICK_COVER_MIN:.0%} 미만"
        elif odd < 0.05:
            mark = "  ★ 홀수 배수가 없다 — 진짜 눈금의 짝수배일 수 있다"
        print(f"  {ns:10d} {c * 100:6.1f}% {odd * 100:7.1f}%{mark}")
    if picked:
        print(f"  -> 이 길이에서의 눈금 {picked * 1e6:.0f} ns "
              f"(설명력 {frac * 100:.0f}%)")
    return picked


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rows = [ln.split(",") for ln in Path(sys.argv[1]).read_text().splitlines()
            if ln and not ln.startswith("#") and not ln.startswith("target")]
    by: dict[str, list[float]] = defaultdict(list)
    for tgt, ms in rows:
        by[tgt].append(float(ms))

    picks = {}
    for tgt in sorted(by, key=float):
        p = analyze(f"목표 {tgt} us", by[tgt])
        if p:
            picks[tgt] = p
    if not picks:
        print("\n눈금을 정하지 못했다. 표본을 늘려라 (--reps).")
        return 1

    vals = sorted(set(picks.values()))
    print("\n" + "=" * 60)
    print("길이별 눈금:", ", ".join(f"{k}us -> {v * 1e6:.0f}ns"
                                   for k, v in picks.items()))
    if len(vals) > 1:
        print("⚠️ 길이마다 다르게 나왔다. **짧은 쪽을 믿어라** — 긴 커널은 "
              "값 하나가 격자를 벗어나기만 해도 설명력이 무너진다.")
    print(f"★ 결론: tick_ms = {min(vals):.9f}  ({min(vals) * 1e6:.0f} ns)")
    print("  core/noise.py 의 NoiseCoef.tick_ms 와 대조하라. 다르면 "
          "앵커에서 뽑은 값이 이 실측과 왜 다른지 규명한 뒤 진행할 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
