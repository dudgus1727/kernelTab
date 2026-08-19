#!/usr/bin/env python3
"""앵커로 세그먼트 간 편차를 검사한다 — 드리프트 대책이 듣는지 판정.

`sweep.py` 는 커널을 세그먼트로 나눠 세그먼트마다 새 프로세스로 돈다.
프로세스가 다르면 계통 오차가 생길 수 있으므로, **모든 세그먼트에서 같은
앵커 커널을 재서** 그 오차를 직접 잰다 (`results/anchors.jsonl`).

⚠️ 판정은 **짧은 앵커**로 한다. 드리프트의 정체는 런치당 상수 오버헤드라
긴 커널에서는 안 보인다. Phase 3 이 4096³ 하나로 감시해서 +5.06 % 로 봤을 때
512³ 측정은 +1380 % 오염되어 있었다. 같은 실수를 반복하지 않는다.

판정은 **절대 기준이 아니라 노이즈 대비**로 한다.

512³ 앵커는 12~23 us 라 측정 노이즈 자체가 몇 %다. 거기에 1 % 절대 기준을
들이대면 달성이 불가능하고, 달성 못 했다고 해서 대책이 실패한 것도 아니다.
물어야 할 것은 **"노이즈 대비 계통 성분이 있는가"** 다.

    세그먼트 간 편차 <= 노이즈 바닥   -> 노이즈에 묻힘. 통과
    세그먼트 간 편차 >  노이즈 바닥   -> 계통 오차. 원인 조사

노이즈 바닥은 **같은 프로세스 안의 start/end 쌍**에서 추정한다. 그 둘은
세그먼트도 프로세스도 같으므로 차이는 시간에 따른 측정 노이즈뿐이다.

두 값을 **같은 통계량**으로 비교해야 한다. 세그먼트 간은 max-min(극단값),
노이즈는 p75(전형값) 로 재면 표본 수가 많은 쪽이 무조건 커 보인다.
그래서 둘 다 **표준편차**로 잰다.

    sigma_between = 세그먼트별 중앙값들의 표준편차
    sigma_within  = 슬라이스 안 start/end 차이의 표준편차 / sqrt(2)
                    (독립 두 측정의 차이는 분산이 2배이므로)

    비율 = sigma_between / sigma_within
      ~1 이면 세그먼트 간 차이가 측정 노이즈로 전부 설명된다 -> 통과
      >1.5 면 노이즈로 설명 안 되는 계통 성분이 있다 -> 조사

통과 기준:
  1. 짧은 앵커의 비율 <= 1.5 (또는 sigma_between <= --tol)
  2. 라운드에 따른 단조 증가 없음   (있으면 세그먼트 밖에 다른 누적이 있다)
  3. 라운드 간 절대값 이동이 노이즈 바닥 이내

    python3 scripts/check_anchors.py
    python3 scripts/check_anchors.py --tol 1.5
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build import paths

ANCHORS = paths.RESULTS_DIR / "anchors.jsonl"
SWEEP = paths.RESULTS_DIR / "sweep.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1.0,
                    help="허용 변동폭 하한 (%%). 실제 기준은 "
                         "max(--tol, 노이즈 바닥) 이다")
    ap.add_argument("--env-hash", default=None)
    args = ap.parse_args()

    if not ANCHORS.exists():
        print(f"{ANCHORS} 가 없다. sweep.py 를 먼저 돌려라.")
        return 2

    env = json.loads(paths.ENV_JSON.read_text())
    eh = args.env_hash or env["env_hash"]

    rows = []
    for line in ANCHORS.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue          # 쓰다 만 줄
        if not str(r.get("env_hash", "")).startswith(eh[:8]):
            continue
        rows.append(r)

    if not rows:
        print(f"env_hash={eh[:8]} 인 앵커 기록이 없다.")
        return 2

    # (kernel_id, M) -> {segment: [time_ms...]}
    by_key: dict[tuple, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for r in rows:
        key = (r["kernel_id"], r["problem"]["M"])
        by_key[key][r["segment"]].append(r["time_ms"])

    # 짧은 앵커부터 (기준 시간 오름차순)
    order = sorted(by_key, key=lambda k: statistics.median(
        [t for v in by_key[k].values() for t in v]))

    # --- 노이즈 바닥 추정 --------------------------------------------------
    # 같은 슬라이스의 start/end 쌍. 세그먼트도 프로세스도 같으므로 그 차이는
    # 순수하게 시간에 따른 측정 노이즈다. 이것이 판정의 분모가 된다.
    noise = {}
    for key in by_key:
        kid, M = key
        per_slice = defaultdict(dict)
        for r in rows:
            if (r["kernel_id"], r["problem"]["M"]) != key:
                continue
            per_slice[r["segment"]].setdefault(r["when"], []).append(r["time_ms"])
        d = []
        for _seg, w in per_slice.items():
            if "start" in w and "end" in w:
                a = statistics.median(w["start"])
                b = statistics.median(w["end"])
                d.append((a / b - 1) * 100)
        # 차이의 표준편차 / sqrt(2) = 단일 측정의 노이즈 표준편차
        noise[key] = (statistics.pstdev(d) / (2 ** 0.5)
                      if len(d) >= 3 else None)

    print(f"앵커 기록 {len(rows):,}줄, 조합 {len(by_key)}개, "
          f"env_hash={eh[:8]}")
    print("판정은 절대 기준이 아니라 **노이즈 대비**로 한다 "
          "(같은 프로세스의 start/end 쌍에서 추정).\n")
    print(f"{'앵커':>40} {'형상':>6} {'중앙(ms)':>10} {'세그':>4} "
          f"{'폭':>6} {'sB':>6} {'sW':>6} {'비율':>6} {'판정':>6}")

    RATIO_MAX = 1.5
    fails = []
    for i, key in enumerate(order):
        kid, M = key
        med = {s: statistics.median(v) for s, v in by_key[key].items()}
        overall = statistics.median(med.values())
        vals = [100 * (v / overall - 1) for v in med.values()]
        spread = max(vals) - min(vals)
        s_between = statistics.pstdev(vals) if len(vals) >= 2 else 0.0
        s_within = noise[key]
        judged = i < max(len(order) // 2, 1)      # 짧은 절반으로만 판정
        if s_within is None:
            ratio = None
            ok = s_between <= args.tol
            rs = "  n/a"
        else:
            ratio = s_between / max(s_within, 1e-9)
            ok = ratio <= RATIO_MAX or s_between <= args.tol
            rs = f"{ratio:5.2f}"
        mark = ("OK" if ok else "실패") if judged else "참고"
        if judged and not ok:
            fails.append((kid, M, s_between))
        print(f"{kid[-40:]:>40} {M:6d} {overall:10.4f} {len(med):4d} "
              f"{spread:5.2f}% {s_between:5.2f}% "
              f"{(s_within if s_within is not None else float('nan')):5.2f}% "
              f"{rs:>6} {mark:>6}")
    print("\n  폭=세그먼트 간 max-min, sB=세그먼트 간 표준편차, "
          "sW=측정 노이즈 표준편차")
    print(f"  판정: 비율(sB/sW) <= {RATIO_MAX} 또는 sB <= {args.tol}%  "
          f"(짧은 앵커 절반만)")

    # --- 라운드 추이 -------------------------------------------------------
    print("\n라운드별 추이 (짧은 앵커 3개 중앙값 기준)")
    rounds = _round_of_segment()
    if rounds:
        short = order[: max(len(order) // 3, 1)]
        by_round: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            key = (r["kernel_id"], r["problem"]["M"])
            if key not in short:
                continue
            rd = rounds.get((r["segment"], r["when"]))
            if rd is not None:
                base = statistics.median(
                    [t for v in by_key[key].values() for t in v])
                by_round[rd].append(r["time_ms"] / base)
        prev = None
        mono = 0
        for rd in sorted(by_round):
            m = statistics.median(by_round[rd])
            arrow = ""
            if prev is not None:
                arrow = "up" if m > prev else "down"
                mono += 1 if m > prev else -1
            print(f"  라운드 {rd:3d}  {100 * (m - 1):+7.2f}%  {arrow}")
            prev = m
        if len(by_round) >= 4 and mono >= len(by_round) - 1:
            fails.append(("라운드 단조 증가", 0, 0.0))
            print("  !! 라운드마다 단조 증가한다 — 세그먼트 밖에 다른 누적이 "
                  "있다. 프로세스 재시작이 완전히 리셋하지 못하는 것이므로 "
                  "드라이버 수준 상태를 의심해야 한다.")
    else:
        print("  (sweep.jsonl 이 없어 라운드를 알 수 없다)")

    # --- 절대값 추이 -------------------------------------------------------
    # 비율(세그먼트 간 편차)만 보면 **모든 세그먼트가 함께 나빠지는** 경우를
    # 놓친다. 편차는 0인데 전체가 드리프트하는 상황이다. 그래서 첫 라운드와
    # 마지막 라운드의 절대값을 직접 비교한다.
    print("\n절대값 추이 (첫 라운드 -> 마지막 라운드)")
    if rounds:
        r_of = {}
        for r in rows:
            rd = rounds.get((r["segment"], r["when"]))
            if rd is not None:
                r_of.setdefault(rd, []).append(r)
        if len(r_of) >= 2:
            first, last = min(r_of), max(r_of)
            print(f"{'앵커':>46} {'형상':>6} {f'R{first}(ms)':>10} "
                  f"{f'R{last}(ms)':>10} {'변화':>8}")
            worst = 0.0
            for key in order:
                kid, M = key
                a = [x["time_ms"] for x in r_of[first]
                     if (x["kernel_id"], x["problem"]["M"]) == key]
                b = [x["time_ms"] for x in r_of[last]
                     if (x["kernel_id"], x["problem"]["M"]) == key]
                if not a or not b:
                    continue
                ma, mb = statistics.median(a), statistics.median(b)
                d = (mb / ma - 1) * 100
                short = order.index(key) < max(len(order) // 2, 1)
                if short:
                    worst = max(worst, abs(d))
                print(f"{kid[-46:]:>46} {M:6d} {ma:10.4f} {mb:10.4f} "
                      f"{d:+7.2f}%")
            nf = max([noise[k] for k in order[: max(len(order) // 2, 1)]
                      if noise[k] is not None] + [args.tol])
            if worst > nf:
                fails.append(("라운드 간 절대값 이동", 0, worst))
                print(f"  !! 짧은 앵커의 절대값이 {worst:.2f}% 움직였다 "
                      f"(노이즈 바닥 {nf:.2f}%). "
                      f"세그먼트 간 편차가 작아도 전체가 함께 드리프트하고 "
                      f"있다는 뜻이다 — 세그먼트 밖의 원인을 찾아야 한다.")
        else:
            print("  (라운드가 2개 미만이라 비교할 수 없다)")
    else:
        print("  (sweep.jsonl 이 없어 라운드를 알 수 없다)")

    # --- 세그먼트 시작/끝 이동 --------------------------------------------
    print("\n세그먼트 안에서의 이동 (start -> end, 짧은 앵커)")
    moved = []
    for key in order[: max(len(order) // 3, 1)]:
        kid, M = key
        st = [r["time_ms"] for r in rows
              if (r["kernel_id"], r["problem"]["M"]) == key and r["when"] == "start"]
        en = [r["time_ms"] for r in rows
              if (r["kernel_id"], r["problem"]["M"]) == key and r["when"] == "end"]
        if not st or not en:
            continue
        d = (statistics.median(en) / statistics.median(st) - 1) * 100
        moved.append(d)
        print(f"  {kid[-40:]:>40} @{M:<5d} {d:+7.2f}%")
    # 세그먼트 내 start->end 이동은 노이즈 바닥의 정의 그 자체이므로
    # 실패 조건으로 쓰지 않는다. 참고로만 출력한다.

    print()
    if fails:
        print(f"!! 실패 {len(fails)}건 — 대책이 충분하지 않다.")
        print("   짧은 앵커만 흔들리면 segments.kernels 를 더 줄여라.")
        print("   모든 앵커가 흔들리면 세그먼트 밖의 원인이다 — 조사할 것.")
        print("   docs/measurement_drift.md")
        return 1
    print("통과: 짧은 앵커의 세그먼트 간 변동이 측정 노이즈로 설명된다.")
    print("      즉 세그먼트마다 프로세스를 새로 띄워도 계통 오차가 없다.")
    return 0


def _round_of_segment() -> dict:
    """sweep.jsonl 에서 (세그먼트, when) -> 라운드. 근사다."""
    if not SWEEP.exists():
        return {}
    out = {}
    for line in SWEEP.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue          # 중단 시 쓰다 만 줄
        if r.get("event") == "slice":
            out[(r["segment"], "start")] = r["round"]
            out[(r["segment"], "end")] = r["round"]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
