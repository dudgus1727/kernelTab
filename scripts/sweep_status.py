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

from kerneltab.core import anchors, paths, records


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
    # ⚠️ **한 env_hash 안에 여러 실행이 있을 수 있다.** G-7(검증)과 G-8(전수)은
    #    같은 조건이므로 같은 env_hash 를 쓰고, 그래야 재개가 된다. 그런데
    #    진행률·속도를 통째로 합치면 틀린다 — 실제로 첫 30분 보고가
    #    "150,868 측정, 86.3/s" 라고 나왔다. 그 중 120,000 은 G-7 것이고
    #    86.3/s 는 물리적으로 불가능한 값이다 (실측 14~17/s).
    #
    #    **누적**(캠페인 완료율)과 **이번 실행**(속도/ETA)을 나눠 센다.
    sw = paths.RESULTS_DIR / "sweep.jsonl"
    runs, cur = [], ""
    for r in records.iter_records(sw, records.ALL):
        ev = r.get("event")
        if ev == "sweep_start":
            cur = str(r.get("env_hash") or "")
            if cur.startswith(eh):
                runs.append({"start": r, "slices": [], "done": None})
        elif not runs:
            continue
        elif ev == "slice" and (str(r.get("env_hash") or "") or cur).startswith(eh):
            runs[-1]["slices"].append(r)
        elif ev == "sweep_done" and str(r.get("env_hash") or cur).startswith(eh):
            runs[-1]["done"] = r
    if not runs:
        print(f"env_hash={eh} 의 sweep_start 가 없다. 아직 시작 안 했다.")
        return 2
    start = runs[-1]["start"]
    slices = runs[-1]["slices"]
    done_ev = runs[-1]["done"]
    prev_slices = sum(len(x["slices"]) for x in runs[:-1])
    n_seg = start.get("n_segments")
    n_jobs = start.get("n_jobs") or 0
    t0 = _ts(start.get("timestamp"))
    last = _ts(slices[-1]["timestamp"]) if slices else t0
    now = datetime.now(timezone.utc)
    elapsed = (now - t0).total_seconds() / 3600 if t0 else 0.0

    rounds = sorted({int(s.get("round", 0)) for s in slices})
    per_round = Counter(int(s.get("round", 0)) for s in slices)
    rc = Counter(s.get("rc") for s in slices)

    # --- results.jsonl -------------------------------------------------------
    # 누적은 캠페인 완료율, 이번 실행분은 속도/ETA 의 분자다.
    # ⚠️ **참조 줄(cuBLAS)을 측정과 함께 세면 안 된다.** 진행률이 100 % 를
    #    넘는다 — 실제로 101.2 % 를 냈다. 참조는 형상당 하나씩 쌓이는
    #    별개 기록이고 config 후보가 아니다.
    n_rows = n_run = n_ref = 0
    st = Counter()
    warm = []
    for r in records.iter_records(paths.RESULTS_DIR / "results.jsonl", eh):
        if records.is_reference(r):
            n_ref += 1
            continue
        n_rows += 1
        st[r.get("status")] += 1
        if r.get("n_warmup") is not None:
            warm.append(r["n_warmup"])
        ts = _ts(r.get("timestamp"))
        if t0 and ts and ts >= t0:
            n_run += 1

    rate = n_run / max((now - t0).total_seconds(), 1) if t0 else 0
    remain = max(n_jobs - n_rows, 0)
    eta = remain / rate / 3600 if rate > 0 else float("nan")

    print(f"스윕  env_hash={eh}  세그먼트 {n_seg}  전체 작업 {n_jobs:,}")
    print(f"  시작 {start.get('timestamp')}  경과 {elapsed:.1f}h  "
          f"(마지막 슬라이스 {(now - last).total_seconds() / 60:.0f}분 전)"
          if last else "")
    print(f"  누적 {n_rows:,} / {n_jobs:,} "
          f"({100 * n_rows / max(n_jobs, 1):.1f}%)"
          f"   [+ cuBLAS 참조 {n_ref:,}]"
          + (f"   [이전 실행 {len(runs) - 1}회, 슬라이스 {prev_slices}]"
             if len(runs) > 1 else ""))
    print(f"  이번 실행 {n_run:,}줄   {rate:.1f}/s   ETA {eta:.1f}h")
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
        print(f"  ** 이번 실행 종료: 라운드 {done_ev.get('rounds')}, "
              f"{done_ev.get('hours')}h **")
    for i, x in enumerate(runs[:-1]):
        d = x["done"]
        print(f"  (이전 실행 {i + 1}: 슬라이스 {len(x['slices'])}"
              + (f", 라운드 {d.get('rounds')}, {d.get('hours')}h" if d else ", 미완")
              + ")")

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
