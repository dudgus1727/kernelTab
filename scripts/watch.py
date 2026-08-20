#!/usr/bin/env python3
"""측정 진행 감시 (D-4).

**`pgrep -f "rehearse.py --all"` 를 쓰지 마라.** 감시자 자신의 명령줄에도 그
문자열이 들어 있어 자기 자신을 찾아낸다. 2026-08-16 에 이것 때문에 측정이
죽은 뒤 13 시간 동안 알아채지 못했다.

대신 측정 프로세스가 10 분마다 쓰는 `results/heartbeat.json` 을 본다.
* 파일의 `pid` 가 실제로 살아 있는가 (`/proc/<pid>` 존재 확인)
* 하트비트가 얼마나 오래됐는가 (오래되면 죽은 것이다)

    python3 scripts/watch.py                 # 한 번 보고
    python3 scripts/watch.py --wait          # 끝날 때까지 (감시용, 배경 실행)
    python3 scripts/watch.py --json          # 기계용

종료 코드
    0  정상 종료(done) 또는 --wait 없이 조회 성공
    3  중단(aborted)
    5  하트비트가 오래됐다 = 죽었을 가능성
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kerneltab.core import paths

HEARTBEAT = paths.RESULTS_DIR / "heartbeat.json"

#: 이 시간(초) 넘게 하트비트가 갱신되지 않으면 죽은 것으로 본다.
#: 측정 주기가 10 분이므로 넉넉히 3 배.
STALE_SECONDS = 1800


def pid_alive(pid: int) -> bool:
    return pid > 0 and Path(f"/proc/{pid}").exists()


def read() -> dict | None:
    if not HEARTBEAT.exists():
        return None
    try:
        return json.loads(HEARTBEAT.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def snapshot() -> dict:
    hb = read()
    if hb is None:
        return {"verdict": "no_heartbeat",
                "msg": f"{HEARTBEAT} 이 없다. 측정을 시작한 적이 없거나 "
                       "다른 results 디렉토리를 보고 있다."}
    try:
        age = (datetime.now(timezone.utc)
               - datetime.strptime(hb["utc"], "%Y-%m-%dT%H:%M:%S.%f%z")).total_seconds()
    except (KeyError, ValueError):
        age = float("inf")
    alive = pid_alive(hb.get("pid", -1))
    state = hb.get("state")

    if state in ("done", "aborted", "finishing"):
        verdict = state
    elif not alive:
        verdict = "dead"
    elif age > STALE_SECONDS:
        verdict = "stale"
    else:
        verdict = "running"
    return {"verdict": verdict, "age_s": round(age, 1), "pid_alive": alive, **hb}


def render(s: dict) -> str:
    if s["verdict"] == "no_heartbeat":
        return s["msg"]
    icon = {"running": "정상", "done": "완료", "aborted": "!! 중단",
            "dead": "!! 프로세스 없음", "stale": "!! 하트비트 정체",
            "finishing": "마무리 중"}.get(s["verdict"], s["verdict"])
    lines = [
        f"[{icon}]  pid={s.get('pid')} (살아있음={s.get('pid_alive')})  "
        f"하트비트 {s.get('age_s')}초 전",
        f"  진행   {s.get('done', 0):,} / {s.get('total', 0):,}  "
        f"({s.get('pct', 0)}%)   경과 {s.get('elapsed_h', 0)}h  "
        f"ETA {s.get('eta_h', '?')}h  ({s.get('rate_per_s', '?')}/s)",
        f"  status {s.get('status')}",
        f"  env_hash {str(s.get('env_hash'))[:16]}",
    ]
    soak = s.get("soak") or {}
    if soak:
        lines.append(
            f"  소킹   {soak.get('soak_seconds', 0) / 60:.1f}분 "
            f"({soak.get('soak_reason')}), 누적 드리프트 "
            f"{100 * (soak.get('soak_total_drift') or 0):+.2f}%")
    th = s.get("thermal") or {}
    if th:
        lines.append(f"  열상태 소킹후 {th.get('soak_elapsed_s', 0) / 3600:.2f}h, "
                     f"drift_ratio {th.get('drift_ratio')}")
    if s.get("abort_reason"):
        lines.append(f"  중단 사유: {s['abort_reason']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", action="store_true",
                    help="종료(done/aborted/dead)까지 기다린다")
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    while True:
        s = snapshot()
        print(json.dumps(s, ensure_ascii=False) if args.json else render(s),
              flush=True)
        if not args.wait:
            break
        if s["verdict"] in ("done", "aborted", "dead", "stale", "no_heartbeat"):
            break
        time.sleep(args.interval)

    return {"aborted": 3, "dead": 5, "stale": 5, "no_heartbeat": 5}.get(
        s["verdict"], 0)


if __name__ == "__main__":
    raise SystemExit(main())
