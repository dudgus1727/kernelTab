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

from build import paths

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


def segment_plan(seg_kernels: int, env: dict) -> dict:
    """`rehearse.py --list-segments --json` 만 쓴다 (R-2).

    예전에는 사람용 출력("세그먼트 13개 x ...", "작업 수: 980,915")을
    파싱했다. 문구를 다듬는 순간 33시간 스윕의 진입점이 깨진다.
    `SystemExit` 으로 죽으니 조용히 틀리지는 않지만 고칠 이유는 충분하다.

    그리고 **`env_hash` 가 어긋나면 거부한다.** sweep 과 rehearse 가 서로
    다른 `env.json` 을 보고 있으면 조건이 다른 데이터가 섞인다.
    """
    out = subprocess.run(
        [sys.executable, str(REHEARSE), "--all", "--list-segments", "--json",
         "--segment-kernels", str(seg_kernels)],
        capture_output=True, text=True, cwd=REPO_ROOT)
    if out.returncode != 0:
        print(out.stdout + out.stderr)
        raise SystemExit("세그먼트 목록을 얻지 못했다")
    payload = None
    for line in out.stdout.splitlines():
        if line.startswith("JSON "):
            payload = json.loads(line[5:])
    if payload is None:
        raise SystemExit(
            "rehearse.py 가 JSON 을 내지 않았다 (--json 을 지원하는지 확인).\n"
            + out.stdout[-2000:])

    # 조건 일치 확인. v2 는 조건 자체, 구 해시는 재개 키다 (P-3).
    for key, mine in (("env_hash", env.get("env_hash")),
                      ("env_hash_v2", env.get("env_hash_v2"))):
        theirs = payload.get(key)
        if mine and theirs and mine != theirs:
            raise SystemExit(
                f"!! {key} 가 어긋난다. 스윕을 시작하지 않는다.\n"
                f"     sweep.py 가 읽은 env.json: {str(mine)[:16]}\n"
                f"     rehearse.py 가 읽은 것    : {str(theirs)[:16]}\n"
                "   측정 도중 env.json 이 바뀌었을 수 있다. 조건이 다른\n"
                "   데이터가 같은 파일에 섞이면 되돌릴 수 없다.")
    payload["jobs_per_segment"] = {int(k): v
                                   for k, v in payload["jobs_per_segment"].items()}
    return payload


def resume_state(env_hash: str | None, n_seg: int) -> tuple[set[int], int]:
    """`sweep.jsonl` 에서 (완료 세그먼트, 다음 라운드) 를 복원한다 (R-3).

    **다른 `env_hash` 의 항목은 무시한다** — 이전 캠페인의 로그가 같은
    파일에 남아 있다 (R-5 와 같은 원칙).
    """
    if not SWEEP_LOG_OK():
        return set(), 0
    done: set[int] = set()
    last_round = -1
    cur_env = None
    for line in LOG.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("event") == "sweep_start":
            cur_env = r.get("env_hash")
            # 새 스윕이 시작됐으면 그 앞의 상태는 무시한다
            if env_hash and cur_env and not str(cur_env).startswith(env_hash[:8]):
                continue
            done, last_round = set(), -1
            continue
        if r.get("event") != "slice":
            continue
        if env_hash and cur_env and not str(cur_env).startswith(env_hash[:8]):
            continue
        last_round = max(last_round, int(r.get("round", 0)))
        if r.get("rc") == RC_DONE:
            done.add(int(r["segment"]))
    return done, max(last_round, 0)


def SWEEP_LOG_OK() -> bool:
    return LOG.exists()


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
    slice_s = args.slice_seconds or seg_cfg.get("seconds", 2700)
    seed = env["shuffle_seed"]

    plan = segment_plan(seg_kernels, env)
    n_seg, n_jobs = plan["n_segments"], plan["n_jobs"]
    per_seg = plan["jobs_per_segment"]
    # 진행 배분은 **작업 수**로 한다. 프로토콜이 시간 예산으로 반복 수를
    # 정하므로 커널 속도가 상쇄되어 작업당 벽시계 편차가 6.6 % 뿐이다.
    #
    # 다만 세그먼트마다 작업 수가 1.8 배까지 갈린다 (커널 수는 같아도
    # alignment/런타임 조합 수가 다르다). 모두에게 같은 슬라이스를 주면
    # 가벼운 세그먼트가 먼저 끝나고 **후반 라운드에 무거운 것만 남아**
    # 시각 상관이 되살아난다. 그래서 세그먼트마다 자기 작업 수에 비례해
    # 나눠 준다 — 그러면 전부 같은 라운드에 끝난다.
    R = max(args.target_rounds, 1)
    slice_of = {s: (args.slice_jobs or max(1, -(-per_seg[s] // R)))
                for s in per_seg}
    print(f"세그먼트 {n_seg}개 x 커널 {seg_kernels}개, 전체 작업 {n_jobs:,}개")
    print(f"세그먼트별 작업 {min(per_seg.values()):,}~{max(per_seg.values()):,}개 "
          f"({max(per_seg.values()) / min(per_seg.values()):.2f}배)")
    print(f"슬라이스 {min(slice_of.values()):,}~{max(slice_of.values()):,}개 "
          f"(작업 수 비례, 목표 {R}라운드, 시간 상한 {slice_s / 60:.0f}분)")
    print(f"예상 재시작 {n_seg * R}회")
    print(f"env_hash={env['env_hash'][:16]}  로그: {LOG}")
    if args.dry_run:
        return 0

    log({"event": "sweep_start", "n_segments": n_seg, "n_jobs": n_jobs,
         "segment_kernels": seg_kernels, "slice_of": slice_of,
         "jobs_per_segment": per_seg,
         "slice_seconds": slice_s, "target_rounds": args.target_rounds,
         "env_hash": env["env_hash"], "pid": os.getpid()})

    # R-3: 중단 후 재시작하면 상태를 복원한다. 안 하면
    #   * 이미 끝난 세그먼트도 매 라운드 프로세스를 띄웠다 즉시 종료
    #     (재개 파싱 6초 x 13개 = 라운드당 78초 낭비)
    #   * 셔플 시드가 seed^(rnd+1) 이라 **라운드 0 의 순서가 반복**된다
    #   * sweep.jsonl 의 라운드 번호가 겹쳐 사후 분석이 헷갈린다
    # 데이터 정확성 문제는 아니다 — 측정된 작업은 건너뛴다.
    done, rnd = resume_state(env.get("env_hash"), n_seg)
    if done or rnd:
        print(f"재개: 라운드 {rnd} 부터, 완료 세그먼트 {len(done)}/{n_seg}")
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
                   "--max-jobs", str(slice_of[seg]),
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
