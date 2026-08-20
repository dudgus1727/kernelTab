#!/usr/bin/env python3
"""스윕 진행 상황 — 슬라이스/라운드 단위 요약 (읽기 전용).

    python3 scripts/sweep_status.py

`watch.py` 는 **슬라이스 하나**(rehearse 프로세스)의 하트비트를 본다.
슬라이스가 끝나면 "완료" 라고 찍히므로 30~40 시간짜리 스윕의 진행률로는
쓸 수 없다. 이 스크립트는 `sweep.jsonl` 을 읽어 **캠페인 전체**를 본다.

⚠️ **읽기만 한다.** 측정 중에 안전하게 돌릴 수 있어야 하므로 어떤 파일도
쓰지 않고 GPU 를 건드리지 않는다.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kerneltab.build import paths
from kerneltab.core import anchors, records


def _ts(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--env-hash", default=None)
    ap.add_argument("--anchors", action="store_true",
                    help="앵커 판정도 함께 (느리다)")
    a = ap.parse_args()

    env = json.loads(paths.ENV_JSON.read_text()) if paths.ENV_JSON.exists() else {}
    eh = (a.env_hash or env.get("env_hash") or records.ALL)[:8]

    # --- sweep.jsonl ---------------------------------------------------------
    sw = paths.RESULTS_DIR / "sweep.jsonl"
    start, slices, done_ev = None, [], None
    cur = ""
    for r in records.iter_records(sw, records.ALL):
        ev = r.get("event")
        if ev == "sweep_start":
            cur = str(r.get("env_hash") or "")
            if cur.startswith(eh):
                start = r
        elif ev == "slice" and (str(r.get("env_hash") or "") or cur).startswith(eh):
            slices.append(r)
        elif ev == "sweep_done" and str(r.get("env_hash") or cur).startswith(eh):
            done_ev = r
    if not start:
        print(f"env_hash={eh} 의 sweep_start 가 없다. 아직 시작 안 했다.")
        return 2

    n_seg = start.get("n_segments")
    n_jobs = start.get("n_jobs") or 0
    t0 = _ts(start.get("timestamp"))
    last = _ts(slices[-1]["timestamp"]) if slices else t0
    now = datetime.now(timezone.utc)
    elapsed = (now - t0).total_seconds() / 3600 if t0 else 0.0

    rounds = sorted({int(s.get("round", 0)) for s in slices})
    per_round = Counter(int(s.get("round", 0)) for s in slices)
    rc = Counter(s.get("rc") for s in slices)

    # --- results.jsonl 로 실제 측정 줄 수 -----------------------------------
    n_rows = 0
    st = Counter()
    warm = []
    for r in records.iter_records(paths.RESULTS_DIR / "results.jsonl", eh):
        n_rows += 1
        st[r.get("status")] += 1
        if r.get("n_warmup") is not None:
            warm.append(r["n_warmup"])

    rate = n_rows / max((now - t0).total_seconds(), 1) if t0 else 0
    remain = max(n_jobs - n_rows, 0)
    eta = remain / rate / 3600 if rate > 0 else float("nan")

    print(f"스윕  env_hash={eh}  세그먼트 {n_seg}  전체 작업 {n_jobs:,}")
    print(f"  시작 {start.get('timestamp')}  경과 {elapsed:.1f}h  "
          f"(마지막 슬라이스 {(now - last).total_seconds() / 60:.0f}분 전)"
          if last else "")
    print(f"  측정 {n_rows:,} / {n_jobs:,} ({100 * n_rows / max(n_jobs, 1):.1f}%)"
          f"   {rate:.1f}/s   ETA {eta:.1f}h")
    print(f"  라운드 {rounds}  슬라이스 {len(slices)}  "
          f"(라운드별 {dict(sorted(per_round.items()))})")
    print(f"  슬라이스 종료코드 {dict(rc)}   "
          "(0=완료 7=예산소진 그외=이상)")
    print(f"  status {dict(st)}")
    if warm:
        z = sum(1 for w in warm if w == 0)
        print(f"  워밍업 최소 {min(warm)} 중앙 {int(statistics.median(warm))} "
              f"0회 {z}건" + ("  ⛔" if z else ""))
    if done_ev:
        print(f"  ** sweep_done: 라운드 {done_ev.get('rounds')}, "
              f"{done_ev.get('hours')}h **")

    # --- 텔레메트리 ---------------------------------------------------------
    tf = paths.RESULTS_DIR / "telemetry.csv"
    if tf.exists():
        rows = tf.read_text().splitlines()[1:]
        sm, temp = [], []
        for ln in rows[-4000:]:
            p = ln.split(",")
            try:
                sm.append(float(p[1]))
                temp.append(float(p[3]))
            except (IndexError, ValueError):
                continue
        if sm:
            print(f"  클럭 {min(sm):.0f}~{max(sm):.0f} MHz  "
                  f"온도 {min(temp):.0f}~{max(temp):.0f}C  "
                  f"(최근 {len(sm)} 샘플)")

    # --- 앵커 ---------------------------------------------------------------
    rows_a, rnd, src, n_sl = anchors.load(eh)
    if not rows_a:
        print("\n앵커 기록이 아직 없다.")
        return 0
    rep = anchors.analyze(rows_a, rnd, eh, round_source=src, n_slices=n_sl)
    print(f"\n앵커 {rep.n_rows:,}줄, 조합 {len(rep.stats)}개, "
          f"라운드 {len(rep.rounds)} ({src})")
    print(f"  짧은 앵커 최대 변동폭 {rep.worst_short:.2f}%")
    if rep.abs_moves:
        print(f"  절대값 R{rep.abs_first}->R{rep.abs_last} 최대 "
              f"{rep.abs_worst:.2f}% (바닥 {rep.abs_floor:.2f}%)")
    else:
        print("  절대값 추이: 라운드 2개 미만 — 아직 판정 불가")
    if a.anchors:
        for s in rep.stats:
            if not s.judged:
                continue
            print(f"    {s.kernel_id[-34:]:>34} @{s.M:<5d} {s.median_ms:8.4f}ms "
                  f"sB={s.s_between:.2f}% sW="
                  f"{'n/a' if s.s_within is None else format(s.s_within, '.2f') + '%'}"
                  f" {'OK' if s.ok else '실패'}")
    # ⚠️ **진행 중 판정을 최종 판정처럼 읽으면 안 된다.**
    #    노이즈 바닥(sW)은 같은 슬라이스의 start/end 쌍 3개 이상이 있어야
    #    추정된다. 스윕 초반에는 그것이 없어서 절대 기준으로 떨어지고,
    #    짧은 앵커는 노이즈 자체가 몇 %라 쉽게 "실패" 로 찍힌다.
    #    최종 판정은 **G-7 이 끝난 뒤 gate_g7.py** 가 한다.
    provisional = sum(1 for f in rep.failures if "sW=n/a" in f)
    n_judged = sum(1 for x in rep.stats if x.judged)
    if provisional:
        print(f"  -> **잠정** — 실패 {len(rep.failures)}건 중 {provisional}건이 "
              f"노이즈 바닥을 아직 추정 못 한 것이다 (슬라이스 {n_sl}개).")
        print("     같은 슬라이스의 start/end 쌍이 3개 이상 쌓여야 판정된다.")
        print("     최종 판정은 gate_g7.py 로 한다.")
    else:
        print(f"  -> {rep.verdict}")
    for f in rep.failures:
        tag = "(표본 부족, 잠정)" if "sW=n/a" in f else "!!"
        print(f"     {tag} {f}")
    if n_judged and not rep.failures:
        print(f"     (판정 대상 앵커 {n_judged}개 전부 통과)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
