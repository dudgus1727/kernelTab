#!/usr/bin/env python3
"""Phase 3 전체 스윕 — 시간 분할 세그먼트로 돈다 (드리프트 대책).

측정 드리프트의 원인은 **그 프로세스가 지금까지 실행한 서로 다른 커널의
누적 수**다. 경과 시간도, 온도도, 작업량도 아니다. 프로세스를 다시 띄우면
완전히 리셋된다. (근거와 실험은 `docs/measurement_drift.md`)

그래서 커널을 세그먼트로 나누고, **세그먼트마다 새 프로세스**로 돈다.

순차로 돌면 안 된다:

    세그먼트 0 (커널 0~499)     -> 전부 초반 3 시간에 측정
    세그먼트 12 (커널 6000~)    -> 전부 마지막 3 시간에 측정

이러면 세그먼트 인덱스가 시각과 완전히 상관되고, 세그먼트는 커널의 집합이므로
**커널 정체성이 시각과 상관**된다. 그건 전역 셔플이 막으려던 바로 그 편향이다.

그래서 라운드 로빈으로 돈다. 각 라운드에서 모든 세그먼트를 조금씩 진행시키고,
세그먼트 순서도 라운드마다 섞는다. 그러면 어느 커널의 측정도 33 시간 전체에
흩어진다.

    python3 scripts/sweep.py                 # 끝까지
    python3 scripts/sweep.py --dry-run       # 계획만
    python3 scripts/sweep.py --max-rounds 2  # 짧은 검증용

중단해도 `results.jsonl` 이 append-only 라 그대로 다시 돌리면 이어진다.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build import paths  # noqa: E402

REHEARSE = REPO_ROOT / "scripts" / "rehearse.py"
LOG = paths.RESULTS_DIR / "sweep.jsonl"

RC_DONE = 0        # 이 세그먼트에 남은 작업이 없다
RC_ABORT = 3       # 이상 감지로 중단
RC_TIME_UP = 7     # 시간 예산 소진 — 정상, 다음 세그먼트로


def log(rec: dict) -> None:
    rec["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def n_segments(seg_kernels: int) -> tuple[int, int]:
    out = subprocess.run(
        [sys.executable, str(REHEARSE), "--all", "--list-segments",
         "--segment-kernels", str(seg_kernels)],
        capture_output=True, text=True, cwd=REPO_ROOT)
    if out.returncode != 0:
        print(out.stdout + out.stderr)
        raise SystemExit("세그먼트 목록을 얻지 못했다")
    n_seg = n_jobs = None
    for line in out.stdout.splitlines():
        if line.startswith("세그먼트 ") and n_seg is None:
            n_seg = int(line.split()[1].rstrip("개"))
        if line.startswith("작업 수:"):
            n_jobs = int(line.split(":")[1].split()[0].replace(",", ""))
    if n_seg is None or n_jobs is None:
        raise SystemExit("세그먼트/작업 수를 파싱하지 못했다:\n" + out.stdout)
    return n_seg, n_jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment-kernels", type=int, default=0,
                    help="세그먼트당 커널 수. 0 이면 rehearse.py 기본값")
    ap.add_argument("--slice-jobs", type=int, default=0,
                    help="세그먼트 한 번에 측정할 작업 수. 진행 배분의 기준이다. "
                         "0 이면 라운드 수 목표(--target-rounds)에서 계산한다")
    ap.add_argument("--target-rounds", type=int, default=8,
                    help="전체를 몇 라운드에 나눠 돌지. 라운드가 많을수록 각 "
                         "커널의 측정이 시간축에 고르게 흩어진다")
    ap.add_argument("--slice-seconds", type=float, default=0,
                    help="한 세그먼트의 시간 상한(초). 작업 예산보다 느린 "
                         "세그먼트가 라운드를 막지 않게 하는 안전장치")
    ap.add_argument("--max-rounds", type=int, default=0,
                    help="이 라운드 수만큼만 돈다 (짧은 검증용)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    env = json.loads(paths.ENV_JSON.read_text())
    seg_cfg = env.get("segments") or {}
    seg_kernels = args.segment_kernels or seg_cfg.get("kernels", 500)
    slice_s = args.slice_seconds or seg_cfg.get("seconds", 1800)
    seed = env["shuffle_seed"]

    n_seg, n_jobs = n_segments(seg_kernels)
    # 진행 배분은 **작업 수**로 한다. 세그먼트마다 커널 실행 시간 분포가
    # 달라서 시간 고정은 진행률이 어긋난다. 시간은 상한으로만 쓴다.
    slice_jobs = args.slice_jobs or max(
        1, -(-n_jobs // (n_seg * max(args.target_rounds, 1))))
    print(f"세그먼트 {n_seg}개 x 커널 {seg_kernels}개, 전체 작업 {n_jobs:,}개")
    print(f"한 세그먼트당 {slice_jobs:,}개씩 라운드 로빈 "
          f"(목표 {args.target_rounds}라운드, 시간 상한 {slice_s / 60:.0f}분)")
    print(f"예상 재시작 {n_seg * args.target_rounds}회")
    print(f"env_hash={env['env_hash'][:16]}  로그: {LOG}")
    if args.dry_run:
        return 0

    log({"event": "sweep_start", "n_segments": n_seg, "n_jobs": n_jobs,
         "segment_kernels": seg_kernels, "slice_jobs": slice_jobs,
         "slice_seconds": slice_s, "target_rounds": args.target_rounds,
         "env_hash": env["env_hash"], "pid": os.getpid()})

    done: set[int] = set()
    rnd = 0
    t0 = time.time()
    while len(done) < n_seg:
        if args.max_rounds and rnd >= args.max_rounds:
            print(f"\n--max-rounds {args.max_rounds} 도달. 멈춘다.")
            break
        # 라운드마다 세그먼트 순서를 바꾼다. 고정 순서면 "항상 먼저 도는
        # 세그먼트" 가 생겨 라운드 안에서 다시 시각 상관이 만들어진다.
        order = [i for i in range(n_seg) if i not in done]
        random.Random(seed ^ (rnd + 1)).shuffle(order)
        print(f"\n{'=' * 60}\n라운드 {rnd}: 세그먼트 {len(order)}개 "
              f"(완료 {len(done)}/{n_seg})  경과 {(time.time()-t0)/3600:.1f}h")
        for seg in order:
            cmd = [sys.executable, str(REHEARSE), "--all",
                   "--segment", str(seg),
                   "--segment-kernels", str(seg_kernels),
                   "--max-jobs", str(slice_jobs),
                   "--time-budget", str(slice_s)]
            print(f"\n--- 라운드 {rnd} 세그먼트 {seg} ---", flush=True)
            t = time.time()
            rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
            el = time.time() - t
            log({"event": "slice", "round": rnd, "segment": seg,
                 "rc": rc, "seconds": round(el, 1)})
            if rc == RC_DONE:
                done.add(seg)
                print(f"[세그먼트 {seg}] 완료 ({el / 60:.1f}분)")
            elif rc == RC_TIME_UP:
                print(f"[세그먼트 {seg}] 시간 소진, 다음으로 ({el / 60:.1f}분)")
            elif rc == RC_ABORT:
                print(f"\n!! 세그먼트 {seg} 가 이상을 감지하고 중단했다.")
                print("   원인을 확인하기 전에는 계속 돌리지 않는다.")
                log({"event": "sweep_abort", "round": rnd, "segment": seg})
                return 3
            else:
                print(f"\n!! 세그먼트 {seg} 가 예상 못 한 코드 {rc} 로 끝났다.")
                log({"event": "sweep_error", "round": rnd,
                     "segment": seg, "rc": rc})
                return 4
        rnd += 1

    el = time.time() - t0
    print(f"\n{'=' * 60}\n전체 완료: 세그먼트 {len(done)}/{n_seg}, "
          f"라운드 {rnd}, {el / 3600:.1f}시간")
    log({"event": "sweep_done", "rounds": rnd, "segments_done": len(done),
         "hours": round(el / 3600, 2)})
    print("\n다음: docs/post_measurement.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
