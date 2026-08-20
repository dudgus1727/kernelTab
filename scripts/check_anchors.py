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

⚠️ **판정 자체는 이 파일에 없다.** `core/anchors.py` 하나에만 있고 여기서는
표시만 한다. 예전에는 `report_phase3.py` 가 같은 데이터를 절대 1% 기준으로
따로 해석해서 **두 곳이 서로 다른 답**을 냈다 (R-6). 사람이 읽는 것은
리포트 쪽이라 틀린 답이 더 널리 읽혔다.

    python3 scripts/check_anchors.py
    python3 scripts/check_anchors.py --tol 1.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kerneltab.build import paths
from kerneltab.core import anchors

ANCHORS = paths.RESULTS_DIR / "anchors.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=anchors.DEFAULT_TOL_PCT,
                    help="허용 변동폭 하한 (%%). 실제 기준은 "
                         "max(--tol, 노이즈 바닥) 이다")
    ap.add_argument("--env-hash", default=None)
    args = ap.parse_args()

    if not ANCHORS.exists():
        print(f"{ANCHORS} 가 없다. sweep.py 를 먼저 돌려라.")
        return 2

    env = json.loads(paths.ENV_JSON.read_text())
    eh = (args.env_hash or env["env_hash"])[:8]

    rows, rnd, src, n_sl = anchors.load(eh)
    if not rows:
        print(f"env_hash={eh} 인 앵커 기록이 없다.")
        return 2
    rep = anchors.analyze(rows, rnd, eh, tol_pct=args.tol,
                          round_source=src, n_slices=n_sl)

    print(f"앵커 기록 {rep.n_rows:,}줄, 조합 {len(rep.stats)}개, "
          f"env_hash={rep.env_hash}")
    print("판정은 절대 기준이 아니라 **노이즈 대비**로 한다 "
          "(같은 조건의 start/end 쌍에서 추정).\n")
    print(f"{'앵커':>40} {'형상':>6} {'중앙(ms)':>10} {'세그':>4} "
          f"{'폭':>6} {'sB':>6} {'sW':>6} {'sW1':>6} {'모델':>6} "
          f"{'비율':>6} {'판정':>6}")
    for st in rep.stats:
        rs = "  n/a" if st.ratio is None else f"{st.ratio:5.2f}"
        mark = ("OK" if st.ok else "실패") if st.judged else "참고"
        print(f"{st.kernel_id[-40:]:>40} {st.M:6d} {st.median_ms:10.4f} "
              f"{st.n_segments:4d} {st.spread_pct:5.2f}% {st.s_between:5.2f}% "
              f"{_pct(st.s_within)} {_pct(st.s_within_slice)} "
              f"{st.model_pct:5.2f}% {rs:>6} {mark:>6}")
    print("\n  폭=세그먼트 간 max-min, sB=세그먼트 간 표준편차")
    print("  sW =세그먼트 중앙값의 표준오차  <- 판정 분모 (sB 와 집계 수준이 같다)")
    print("  sW1=슬라이스 1회 측정의 노이즈  <- 참고. sW 보다 크면 평균화가 듣는 것")
    print("  모델=core/noise.py 의 sigma_rel(t)=0.000374/t+0.00044 예측값.")
    print("       **다른 데이터에서 나온 모델**이므로 sW1 과 맞으면 교차 검증이다.")
    print(f"  판정: 비율(sB/sW) <= {anchors.RATIO_MAX} 또는 sB <= {args.tol}%  "
          f"(짧은 앵커 절반만)")

    # --- 라운드 -----------------------------------------------------------
    src_txt = {"recorded": "앵커 줄에 기록된 round",
               "timestamp": "sweep.jsonl 슬라이스 구간에서 시각으로 복원",
               "none": "알 수 없음"}[rep.round_source]
    print(f"\n라운드별 추이 (짧은 앵커 {max(len(rep.stats) // 3, 1)}개 중앙값 기준)")
    print(f"  라운드 출처: {src_txt}  (슬라이스 {rep.n_slices}개)")
    if rep.rounds:
        prev = None
        for pt in rep.rounds:
            if prev is None:
                arrow = ""
            elif abs(pt.rel_pct - prev) < 1e-9:
                arrow = "="        # 타이머 눈금에 걸려 중앙값이 같다
            else:
                arrow = "up" if pt.rel_pct > prev else "down"
            print(f"  라운드 {pt.round:3d}  중앙 {pt.rel_pct:+7.3f}%  "
                  f"평균 {pt.mean_pct:+7.3f}%  n={pt.n:<5d} {arrow}")
            prev = pt.rel_pct
        print("  중앙값이 0.000% 로 고정돼 보이면 타이머 눈금(1.024us) 때문이다 —")
        print("  14us 앵커에서 한 눈금이 7.3% 다. 평균 쪽을 함께 볼 것.")
    else:
        print("  (라운드를 알 수 없다)")

    # --- 절대값 -----------------------------------------------------------
    print("\n절대값 추이 (첫 라운드 -> 마지막 라운드)")
    if rep.abs_moves:
        print(f"{'앵커':>46} {'형상':>6} {f'R{rep.abs_first}(ms)':>10} "
              f"{f'R{rep.abs_last}(ms)':>10} {'변화':>8}")
        for m in rep.abs_moves:
            print(f"{m.kernel_id[-46:]:>46} {m.M:6d} {m.first_ms:10.4f} "
                  f"{m.last_ms:10.4f} {m.delta_pct:+7.2f}%")
        print(f"  짧은 앵커 최대 이동 {rep.abs_worst:.2f}% "
              f"(노이즈 바닥 {rep.abs_floor:.2f}%)")
    else:
        print("  (라운드가 2개 미만이라 비교할 수 없다)")

    # --- 세그먼트 안 이동 (참고) ------------------------------------------
    print("\n세그먼트 안에서의 이동 (start -> end, 짧은 앵커)")
    for kid, M, d in rep.within_slice:
        print(f"  {kid[-40:]:>40} @{M:<5d} {d:+7.2f}%")
    print("  (노이즈 바닥의 정의 그 자체이므로 실패 조건으로 쓰지 않는다)")

    if rep.notes:
        print("\n주의 (실패는 아니다)")
        for n in rep.notes:
            print(f"  - {n}")

    print()
    if not rep.ok:
        print(f"!! 실패 {len(rep.failures)}건 — 대책이 충분하지 않다.")
        for f in rep.failures:
            print(f"   - {f}")
        print("   짧은 앵커만 흔들리면 segments.kernels 를 더 줄여라.")
        print("   모든 앵커가 흔들리면 세그먼트 밖의 원인이다 — 조사할 것.")
        print("   docs/measurement_drift.md")
        return 1
    print("통과: 짧은 앵커의 세그먼트 간 변동이 측정 노이즈로 설명된다.")
    print("      즉 세그먼트마다 프로세스를 새로 띄워도 계통 오차가 없다.")
    return 0


def _pct(v: float | None) -> str:
    return "   n/a" if v is None else f"{v:5.2f}%"


if __name__ == "__main__":
    raise SystemExit(main())
