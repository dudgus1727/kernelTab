#!/usr/bin/env python3
"""측정 오버헤드 실측 — 워밍업 시간 예산화가 실제로 얼마나 줄였는가.

    python3 scripts/report_overhead.py --env-hash <8자리>

`results.jsonl` 의 `n_probe` / `n_warmup` / `overhead_ms` 를 읽어
**옛 규칙이었다면 얼마였을지**와 대조한다. 벽시계로 재면 호스트 부하와
섞이므로, 기록된 GPU 시간으로 직접 센다.

옛 규칙:
    probe = max(3, min_warmup)                      = 10
    warm  = max(min_warmup, warmup_frac x n_reps)
    오버헤드 = (probe + warm) x 1회 소요시간
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kerneltab.build import paths
from kerneltab.core import records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--env-hash", default=None)
    ap.add_argument("--results", default=None)
    a = ap.parse_args()

    env = json.loads(paths.ENV_JSON.read_text()) if paths.ENV_JSON.exists() else {}
    eh = a.env_hash or (env.get("env_hash") or records.ALL)
    proto = {**{"warmup_frac": 0.2, "min_warmup": 10}, **(env.get("protocol") or {})}
    f = Path(a.results) if a.results else paths.RESULTS_DIR / "results.jsonl"

    new_ms = old_ms = meas_ms = 0.0
    n = 0
    warm_zero = 0
    per_job = []
    for r in records.iter_records(f, eh):
        if r.get("status") not in ("ok", "high_outlier_frac"):
            continue
        t, nr = r.get("time_ms"), r.get("n_reps")
        ov = r.get("overhead_ms")
        if not t or not nr or ov is None:
            continue
        n += 1
        if not r.get("n_warmup"):
            warm_zero += 1
        old_probe = max(3, int(proto["min_warmup"]))
        old_warm = max(int(proto["min_warmup"]),
                       int(proto["warmup_frac"] * nr))
        o = (old_probe + old_warm) * t
        new_ms += ov
        old_ms += o
        meas_ms += nr * t
        per_job.append((ov, o))

    if not n:
        print("overhead_ms 가 기록된 줄이 없다. 옛 빌드로 측정한 데이터다.")
        print("  (n_probe/n_warmup/overhead_ms 는 2026-08-20 이후 기록된다)")
        return 2

    print(f"작업 {n:,}  (env_hash={str(eh)[:8]})")
    print(f"{'':16} {'옛(s)':>10} {'새(s)':>10} {'절감':>8}")
    print(f"{'오버헤드':16} {old_ms / 1000:10.1f} {new_ms / 1000:10.1f} "
          f"{100 * (1 - new_ms / max(old_ms, 1e-9)):7.1f}%")
    print(f"{'측정':16} {meas_ms / 1000:10.1f} {meas_ms / 1000:10.1f}")
    tot_o, tot_n = old_ms + meas_ms, new_ms + meas_ms
    print(f"{'합계(GPU)':16} {tot_o / 1000:10.1f} {tot_n / 1000:10.1f} "
          f"{100 * (1 - tot_n / max(tot_o, 1e-9)):7.1f}%")
    print(f"\n오버헤드 비중  옛 {100 * old_ms / tot_o:.1f}%  "
          f"-> 새 {100 * new_ms / tot_n:.1f}%")
    r = [o / max(p, 1e-9) for p, o in per_job]
    print(f"작업당 절감 배수  중앙 {statistics.median(r):.2f}x  "
          f"최대 {max(r):.1f}x")
    if warm_zero:
        print(f"\n⛔ 워밍업이 0 회인 작업 {warm_zero:,}개 — warmup_reps_floor 확인")
        return 4

    print("\n⚠️ GPU 시간만 센 것이다. 작업당 벽시계의 1/3 은 호스트 측")
    print("   고정 비용(동기화, ctypes, JSON 기록)이라 전체 단축은 이보다 작다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
