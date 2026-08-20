#!/usr/bin/env python3
"""G-7 재현성 검증 게이트 — 다섯 검사를 **한 번에, 한 종료 코드로**.

    python3 scripts/sweep.py --max-rounds 4     # 3시간
    python3 scripts/gate_g7.py                  # 판정

`docs/new_environment_checklist.md` 의 G-7 이다. 검사를 다섯 개 따로 돌리면
하나를 빼먹고, 빼먹은 줄 모른 채 전수를 켠다. 그래서 하나로 묶는다.

| # | 무엇 | 기준 |
|---|---|---|
| 1 | 짧은 앵커의 세그먼트 간 편차 | `sB/sW` <= 1.5 |
| 2 | 라운드 간 절대값 추이 | 노이즈 바닥 이내 |
| 3 | 슬라이스 `start -> end` | 기준선 대비 악화 없음 |
| 4 | **워밍업 하한** | `n_warmup` >= floor, 0 회 **0 건** |
| 5 | 재현성 | 5 % 초과 **0 건** |

## 3 번에 절대 1 % 를 쓰지 않는 이유

체크리스트에는 "1 % 이내" 로 적혀 있었지만, **12.4 캠페인에서도 그 기준을
넘었다** (짧은 앵커 하나가 -3.02 %). 워밍업과 무관하게 원래 그런 값이다.
절대 기준을 그대로 쓰면 워밍업 변경과 상관없는 이유로 G-7 이 실패한다.

그래서 **기준선과 비교**한다. 기준선이 없으면 절대 기준으로 떨어지되,
그 사실을 찍는다 — 조용히 통과시키지 않는다.

## 4 번과 5 번이 이번에 새로 중요하다

워밍업을 시간 예산으로 바꿨다. 세그먼트마다 프로세스를 새로 띄우므로
슬라이스 시작은 매번 냉시작이고, 워밍업을 줄였으니 그 자리가 무너지지
않았는지 확인해야 한다.
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
from kerneltab.core import anchors, records

#: 슬라이스 내 이동이 기준선의 몇 배를 넘으면 악화로 볼지.
WITHIN_SLICE_WORSE = 2.0

#: 기준선이 없을 때의 절대 상한 (%).
WITHIN_SLICE_ABS = 4.0

#: 재현성 허용 (상대 차이).
REPRO_TOL = 0.05


def _fmt(ok: bool) -> str:
    return "통과" if ok else "**실패**"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--env-hash", default=None)
    ap.add_argument("--baseline", default=None,
                    help="슬라이스 내 이동의 기준선 JSON "
                         "(이전 캠페인의 gate_g7 결과)")
    ap.add_argument("--out", default=str(paths.RESULTS_DIR / "gate_g7.json"))
    a = ap.parse_args()

    if not paths.ENV_JSON.exists():
        print("results/env.json 이 없다.")
        return 2
    env = json.loads(paths.ENV_JSON.read_text())
    eh = (a.env_hash or env["env_hash"])[:8]
    proto = env.get("protocol") or {}
    warm_floor = int(proto.get("warmup_reps_floor") or 1)

    rows, rnd, src, n_sl = anchors.load(eh)
    if not rows:
        print(f"⛔ env_hash={eh} 의 앵커 기록이 없다. sweep 을 먼저 돌려라.")
        return 2
    rep = anchors.analyze(rows, rnd, eh, round_source=src, n_slices=n_sl)

    checks, verdicts = [], []

    # --- 1, 2: core/anchors 가 판정한다 ------------------------------------
    seg_fail = [f for f in rep.failures if "세그먼트 간 편차" in f]
    abs_fail = [f for f in rep.failures if "절대값" in f or "단조" in f]
    checks.append(("1. 세그먼트 간 편차 (sB/sW)", not seg_fail,
                   f"짧은 앵커 최대 변동폭 {rep.worst_short:.2f}%"))
    checks.append(("2. 라운드 간 절대값 추이", not abs_fail,
                   (f"R{rep.abs_first}->R{rep.abs_last} 최대 "
                    f"{rep.abs_worst:.2f}% (바닥 {rep.abs_floor:.2f}%)")
                   if rep.abs_moves else "라운드가 2개 미만 — **검사 못 함**"))
    if not rep.abs_moves:
        checks[-1] = (checks[-1][0], False, checks[-1][2])

    # --- 3: 슬라이스 내 이동 -----------------------------------------------
    base = {}
    if a.baseline and Path(a.baseline).exists():
        b = json.loads(Path(a.baseline).read_text())
        base = {f"{r[0]}@{r[1]}": abs(r[2]) for r in b.get("within_slice", [])}
    moves = [(kid, M, d) for kid, M, d in rep.within_slice]
    worst_k, worst_v, worst_lim = None, 0.0, WITHIN_SLICE_ABS
    for kid, M, d in moves:
        lim = (max(WITHIN_SLICE_WORSE * base[f"{kid}@{M}"], 1.0)
               if f"{kid}@{M}" in base else WITHIN_SLICE_ABS)
        if abs(d) - lim > worst_v - worst_lim:
            worst_k, worst_v, worst_lim = f"{kid[-30:]}@{M}", abs(d), lim
    ok3 = worst_v <= worst_lim
    checks.append(("3. 슬라이스 start->end", ok3,
                   (f"최대 {worst_v:.2f}% ({worst_k}), 허용 {worst_lim:.2f}% "
                    + ("[기준선 대비]" if base else "[기준선 없음 — 절대값]"))))

    # --- 4: 워밍업 하한 -----------------------------------------------------
    n = zero = below = 0
    warms = []
    for r in records.iter_records(paths.RESULTS_DIR / "results.jsonl", eh):
        w = r.get("n_warmup")
        if w is None:
            continue
        n += 1
        warms.append(w)
        if w == 0:
            zero += 1
        elif w < warm_floor:
            below += 1
    if not n:
        checks.append(("4. 워밍업 하한", False,
                       "n_warmup 이 기록된 줄이 없다 — **검사 못 함**"))
    else:
        checks.append(("4. 워밍업 하한", zero == 0 and below == 0,
                       f"작업 {n:,}, 0회 {zero}건, 하한({warm_floor}) 미만 "
                       f"{below}건, 최소 {min(warms)} 중앙 "
                       f"{int(statistics.median(warms))}"))

    # --- 5: 재현성 ----------------------------------------------------------
    f = paths.RESULTS_DIR / "repro.jsonl"
    first = {}
    for d in records.iter_records(paths.RESULTS_DIR / "results.jsonl", eh):
        if d.get("kernel_id") == "cublas" or not d.get("time_ms"):
            continue
        p, rt = d["problem"], d["runtime"]
        first.setdefault((d["kernel_id"], p["M"], p["N"], p["K"],
                          rt["split_k"], rt["split_k_mode"]), d["time_ms"])
    diffs = []
    for d in records.iter_records(f, eh):
        p, rt = d.get("problem"), d.get("runtime")
        if not p or not rt or not d.get("time_ms"):
            continue
        t0 = first.get((d["kernel_id"], p["M"], p["N"], p["K"],
                        rt["split_k"], rt["split_k_mode"]))
        if t0:
            diffs.append(abs(d["time_ms"] - t0) / t0)
    if not diffs:
        checks.append(("5. 재현성", False, "재측정 기록이 없다 — **검사 못 함**"))
    else:
        over = [d for d in diffs if d > REPRO_TOL]
        checks.append(("5. 재현성", not over,
                       f"{len(diffs)}개 재측정, {REPRO_TOL:.0%} 초과 "
                       f"{len(over)}건, 최대 {100 * max(diffs):.2f}%"))

    # --- 출력 ---------------------------------------------------------------
    print(f"G-7 재현성 검증  env_hash={eh}  "
          f"(앵커 {rep.n_rows:,}줄, 슬라이스 {n_sl}, 라운드 {len(rep.rounds)})\n")
    for name, ok, detail in checks:
        verdicts.append(ok)
        print(f"  {_fmt(ok):>8}  {name:28s}  {detail}")
    if rep.notes:
        print("\n주의 (실패는 아니다)")
        for x in rep.notes:
            print(f"  - {x}")

    out = {"env_hash": eh,
           "checks": [{"name": nm, "ok": ok, "detail": d}
                      for nm, ok, d in checks],
           "within_slice": [[kid, M, d] for kid, M, d in moves],
           "worst_short_pct": rep.worst_short,
           "abs_worst_pct": rep.abs_worst, "abs_floor_pct": rep.abs_floor,
           "warmup_min": min(warms) if warms else None,
           "repro_max": max(diffs) if diffs else None,
           "ok": all(verdicts)}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n-> {a.out}")

    if all(verdicts):
        print("\n통과. 전수를 켜도 된다.")
        return 0
    print("\n⛔ 통과하지 못했다. **전수를 켜지 마라.**")
    print("   docs/new_environment_checklist.md G-7")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
