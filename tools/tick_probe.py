#!/usr/bin/env python3
"""이벤트 타이머의 **눈금(양자)**을 직접 잰다.

    python3 tools/tick_probe.py --kernel-id <id> --shape 2048x2048x2048 --reps 3000

## 왜 직접 재야 하나

눈금은 환경마다 다르다 — A6000 1024 ns / 5090 16 ns. 이 값은 노이즈 바닥의
분해능 항 `tick/t` 에 그대로 들어가고, 그것이 `answer_set()` 의 동점 판정을
정한다. 5090 에서 계수를 잘못 잡았을 때 정답 집합이 1 개 대 1,418 개로 갈렸다.

## ⛔ 이 추정은 **항상 상한이다** — 그것이 이 도구의 핵심 출력이다

값들은 진짜 양자 q 의 정수배 `n_i·q` 다. 관측된 최소 간격은 `m·q` (m>=1) 이고
우리는 그것을 눈금으로 삼는다. **m>1 이면 과대추정인데, 값만 보고는 구분할
수 없다** — q 격자 위의 값은 전부 2q 격자로도, 그 어떤 q/k 격자로도 설명된다.
5090 이 정확히 이렇게 32 ns 로 결론냈다가 나중에 16 ns 로 정정했다
(`decisions.md` 26).

그래서 이 도구는 **두 가지를 한다:**

1. 표본을 크게 잡고 격자 적합을 `core.noise.tick_grid()` 로 한다
   (정수 ns 스냅 + 설명력이 같으면 가장 거친 후보).
2. **추정의 취약성을 정량화한다** — 부분표본으로 다시 추정해서, 최소 간격을
   만든 그 한 쌍이 빠지면 값이 2 배가 되는지 본다. `1x` 간격이 몇 번밖에
   없으면 그 추정은 표본 하나에 매달려 있다는 뜻이다.

**"안정" 이 "맞다" 는 뜻은 아니다.** 여전히 상한이며, 결론에 그렇게 적어라.

## 중앙값을 쓰지 않는다

`KtMeasure.time_ms` 는 IQR 제거 후 **중앙값**이라 표본 수가 짝수면 두 값의
평균 — 격자 밖으로 떨어진다. 격자에 정확히 놓이는 것은 `time_min_ms` /
`time_max_ms` 뿐이므로 그 둘만 모은다.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kerneltab.core import paths                      # noqa: E402
from kerneltab.core.noise import (                     # noqa: E402
    odd_multiple_frac, tick_grid, tick_ms_observed,
)


def collect(kernel_id: str, M: int, N: int, K: int, reps: int) -> list[float]:
    from kerneltab.measure.runner import Ctx, Kernel, KtProblemC

    env = json.loads(paths.ENV_JSON.read_text())
    ctx = Ctx(paths.ARTIFACT_DIR / "libkt_ctx.so", 0)
    ctx.set_protocol(env)
    kern = Kernel(paths.kernel_so(kernel_id))
    ctx.prepare_problem(M, N, K)
    kp = KtProblemC(M, N, K, 1, 0)
    bufs = ctx.buffers(kern.workspace_bytes(kp), False)
    st, h = kern.prepare(kp, bufs)
    if st != 0 or not h:
        raise SystemExit(f"prepare 실패 (status={st}) — 이 커널/형상 조합이 안 된다")
    vals: list[float] = []
    try:
        for _ in range(reps):
            st, m = ctx.measure(kern.launch_addr, h, 0)
            if st != 0:
                continue
            # ★ 격자에 정확히 놓이는 것만. 중앙값은 보간될 수 있다.
            vals.append(m.time_min_ms)
            vals.append(m.time_max_ms)
    finally:
        kern.release(h)
        ctx.close()
    return vals


def fragility(values, trials: int = 200, frac: float = 0.8, seed: int = 0) -> dict:
    """부분표본으로 다시 추정한다. 값이 흔들리면 그 추정은 표본 하나에 매달렸다."""
    xs = sorted({round(float(v), 9) for v in values})
    rng = random.Random(seed)
    got = Counter()
    for _ in range(trials):
        sub = rng.sample(xs, max(2, int(len(xs) * frac)))
        t, _c = tick_grid(sub)
        got[None if t is None else round(t * 1e6)] += 1     # ns
    return {str(k): v for k, v in sorted(got.items(), key=lambda kv: -kv[1])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-id", required=True)
    ap.add_argument("--shape", default="2048x2048x2048")
    ap.add_argument("--reps", type=int, default=3000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    M, N, K = (int(x) for x in a.shape.split("x"))

    vals = collect(a.kernel_id, M, N, K, a.reps)
    xs = sorted({round(float(v), 9) for v in vals})
    tick, cover = tick_grid(xs)
    base = tick_ms_observed(xs)

    gaps = [round((b - a_) * 1e6) for a_, b in zip(xs, xs[1:])]   # ns
    mult = Counter()
    if tick:
        for g in gaps:
            mult[round(g / (tick * 1e6))] += 1
    ones = mult.get(1, 0)

    frag = fragility(xs)
    out = {
        "shape": a.shape, "kernel_id": a.kernel_id,
        "samples": len(vals), "distinct": len(xs),
        "min_ms": xs[0], "max_ms": xs[-1],
        "median_ms": statistics.median(xs),
        "min_gap_ns": round(base * 1e6, 4) if base else None,
        "tick_ms": tick, "tick_ns": round(tick * 1e6) if tick else None,
        "cover": round(cover, 4),
        "gap_multiples": {str(k): v for k, v in sorted(mult.items())},
        "one_x_gaps": ones,
        "subsample_ticks_ns": frag,
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))

    print("\n" + "=" * 68)
    print(f"표본 {len(vals):,}  고유값 {len(xs):,}  "
          f"범위 {xs[0] * 1e3:.3f}~{xs[-1] * 1e3:.3f} us")
    print(f"눈금 추정 {out['tick_ns']} ns  (설명력 {100 * cover:.1f}%, "
          f"최소 간격 {out['min_gap_ns']} ns)")
    print(f"간격 배수 분포 {dict(sorted(mult.items())[:8])}")
    print(f"1x 간격 {ones}회  <- 이 값이 작으면 추정이 표본 하나에 매달려 있다")
    print(f"부분표본 재추정 {frag}")
    if len(frag) > 1:
        print("  ⚠️ 부분표본에서 값이 갈린다. 표본을 늘려라 (--reps).")

    # ★ 위에서 조인다 — 부분표본(아래에서 조이기)과 상보적이다.
    #
    #   후보가 진짜 눈금의 짝수배라면 그 후보의 **홀수 배수**인 값이 존재할
    #   수 없다. 홀수 배수가 충분히 관측되면 "상한" 을 값으로 좁힐 수 있다.
    #   H100 에서 16 ns 가 설명력 100 % 인데 홀수 배수 0 % 라 기각됐고
    #   32 ns 가 51.7 % 로 채택됐다 (2026-09-02).
    tick = out["tick_ns"] * 1e-6
    odd_here = odd_multiple_frac(xs, tick)
    odd_half = odd_multiple_frac(xs, tick / 2)
    print(f"\n홀수 배수 검사  이 후보 {100 * odd_here:.1f}%"
          f"   절반({out['tick_ns'] / 2:g} ns) {100 * odd_half:.1f}%")
    if odd_here >= 0.05:
        print("  ★ 이 후보의 홀수 배수가 관측된다 — 진짜 눈금의 짝수배가"
              " 아니다. 부분표본도 안정이면 상한이 아니라 **값**이다.")
    else:
        print("  ⛔ 이 후보의 홀수 배수가 없다 — **진짜 눈금이 절반일 수"
              " 있다.** 절반 후보의 설명력을 확인하라.")
        print("\n⛔ 이 값은 **상한**이다. 5090 이 고유값 92 개에서 32 ns 로"
              " 결론냈다가 16 ns 로 정정했다 (decisions 26).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
