"""앵커 분석 — 세그먼트 간 계통 오차 판정. **판정 로직의 유일한 구현이다.**

## 왜 이 모듈이 있는가 (R-6)

같은 데이터를 두 곳이 따로 해석했고, 서로 다른 답을 냈다.

| | 판정 | 결과 |
|---|---|---|
| `scripts/check_anchors.py` | 노이즈 대비 (`sB/sW <= 1.5`) | **통과** |
| `scripts/report_phase3.py` | 절대 기준 (`변동폭 <= 1%`) | "1% 를 넘는다" |

512³ 앵커는 12~23 us 라 **측정 노이즈 자체가 몇 %다.** 거기에 1% 절대
기준을 들이대면 달성이 불가능하고, 달성 못 했다고 해서 대책이 실패한 것도
아니다. 물어야 할 것은 "노이즈 대비 계통 성분이 있는가" 다.

리포트 쪽이 틀렸다. 그런데 **리포트가 사람이 읽는 문서**이므로, 틀린 쪽이
더 널리 읽힌다. 그래서 판정을 여기 하나로 모으고 두 스크립트는 **표시만**
한다. `core/records.py` 가 읽기를, `validate_bundle()` 이 배포를 강제하는
것과 같은 구조다 (`docs/decisions.md` 13).

## 라운드 매핑 — 앵커는 슬라이스 단위로 봐야 한다

`sweep.py` 의 실행 단위는 **슬라이스 = (라운드, 세그먼트)** 다. 세그먼트
하나가 라운드마다 다시 돈다. 그런데 `anchors.jsonl` 에는 `segment` 와
`when` 만 있어서, 옛 코드는 이렇게 매핑했다:

    out[(segment, "start")] = round     # 라운드마다 덮어쓴다

라운드 8개를 돌아도 **마지막 라운드 하나만 남는다.** 실제로 검증 실행에서
"라운드별 추이" 에 라운드 3 한 줄만 찍혔고, "절대값 추이" 는 "라운드가 2개
미만이라 비교할 수 없다" 로 통째로 죽어 있었다. 그 비교가 바로 **모든
세그먼트가 함께 드리프트하는 경우**를 잡는 검사인데, 그것이 안 돌고 있었다.

같은 버그가 노이즈 바닥에도 있었다. `sW` 는 "같은 슬라이스의 start/end 쌍"
이어야 하는데 세그먼트로만 묶어서 **라운드를 가로질러 섞고 있었다.** 그
값이 판정의 분모다.

두 가지로 고친다.

1. **앞으로**: `rehearse.py` 가 `round` 를 앵커 줄에 기록한다
   (`sweep.py` 가 `--round` 로 넘긴다). 재구성이 필요 없다.
2. **이미 측정한 데이터**: `sweep.jsonl` 의 `slice` 이벤트는 슬라이스가
   **끝날 때** 기록되고 `seconds` 를 담는다. 즉 구간 `[ts - seconds, ts]`
   가 그 슬라이스다. 앵커의 `timestamp` 를 그 구간에 넣어 복원한다.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from kerneltab.core import noise as noise_model
from kerneltab.core import records

__all__ = [
    "DEFAULT_TOL_PCT",
    "RATIO_MAX",
    "AnchorReport",
    "AnchorStat",
    "analyze",
    "load",
    "slice_intervals",
]

#: sB/sW 가 이 값을 넘으면 노이즈로 설명되지 않는 계통 성분이 있다.
RATIO_MAX = 1.5

#: sB 가 이 값(%) 이하면 비율과 무관하게 통과. 노이즈가 아주 작은 앵커에서
#: 비율이 과민해지는 것을 막는 하한이다.
DEFAULT_TOL_PCT = 1.0

#: 관측 노이즈가 `core/noise.py` 모델의 몇 배를 넘으면 주의로 표시할지.
#: **실패로 만들지 않는다** — 긴 앵커는 드리프트 판정 대상이 아니고
#: (드리프트는 런치당 상수라 긴 커널에서는 안 보인다), 이 신호 하나로
#: 캠페인 전체를 실패로 뒤집는 것은 근거가 약하다. 대신 크게 찍는다.
MODEL_RATIO_WARN = 3.0

#: 라운드 간 절대값 이동의 허용 배수. 라운드 중앙값의 흔들림(연속 차분
#: 표준편차)의 몇 배까지를 노이즈로 볼지. 2 는 약 2-sigma 다.
ABS_FLOOR_K = 2.0

#: 슬라이스 구간 판정 여유(초). `seconds` 는 소수점 1자리로 반올림되고
#: 로그 기록과 측정 사이에 약간의 지연이 있다.
SLICE_SLACK_S = 15.0


def _ts(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


@dataclass
class Slice:
    """`sweep.py` 의 실행 단위. 세그먼트가 아니라 (라운드, 세그먼트) 다."""
    round: int
    segment: int
    t0: float
    t1: float


@dataclass
class AnchorStat:
    kernel_id: str
    M: int
    median_ms: float
    n_segments: int
    spread_pct: float          # 세그먼트 중앙값들의 max-min
    s_between: float           # 세그먼트 간 표준편차 (%)
    s_within: float | None     # 세그먼트 중앙값의 표준오차 (%) — 판정 분모
    s_within_slice: float | None   # 슬라이스 1회 측정의 노이즈 (%) — 참고
    model_pct: float           # core/noise.py 가 예측하는 노이즈 바닥 (%)
    ratio: float | None
    judged: bool               # 짧은 절반만 판정한다
    ok: bool


@dataclass
class RoundPoint:
    round: int
    rel_pct: float             # 앵커 기준선 대비, 중앙값 (%)
    mean_pct: float            # 같은 것을 평균으로. 아래 주석 참고
    n: int


@dataclass
class AbsMove:
    kernel_id: str
    M: int
    first_ms: float
    last_ms: float
    delta_pct: float
    judged: bool


@dataclass
class AnchorReport:
    env_hash: str
    n_rows: int
    stats: list[AnchorStat] = field(default_factory=list)
    rounds: list[RoundPoint] = field(default_factory=list)
    abs_moves: list[AbsMove] = field(default_factory=list)
    abs_first: int | None = None
    abs_last: int | None = None
    abs_worst: float | None = None
    abs_floor: float | None = None
    #: (kernel_id, M, 중앙값 이동 %, 평균 이동 %, 눈금 하나 %)
    within_slice: list[tuple[str, int, float, float, float]] = field(
        default_factory=list)
    notes: list[str] = field(default_factory=list)
    n_slices: int = 0
    round_source: str = "none"       # "recorded" | "timestamp" | "none"
    failures: list[str] = field(default_factory=list)
    tol_pct: float = DEFAULT_TOL_PCT

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def worst_short(self) -> float:
        """짧은(판정 대상) 앵커의 최대 세그먼트 간 변동폭 (%)."""
        v = [s.spread_pct for s in self.stats if s.judged]
        return max(v) if v else 0.0

    @property
    def verdict(self) -> str:
        if not self.stats:
            return "앵커 기록이 없다 — 세그먼트 방식으로 측정하지 않았다."
        if self.ok:
            return ("**통과.** 짧은 앵커의 세그먼트 간 변동이 측정 노이즈로 "
                    "설명된다. 세그먼트마다 프로세스를 새로 띄워도 계통 "
                    "오차가 없다.")
        return "**실패 " + str(len(self.failures)) + "건.** " + " / ".join(
            self.failures)


# ---------------------------------------------------------------------------
# 읽기
# ---------------------------------------------------------------------------
def slice_intervals(sweep_path: str | Path, env_hash: str) -> list[Slice]:
    """`sweep.jsonl` 에서 슬라이스 구간을 복원한다.

    `slice` 이벤트는 슬라이스가 **끝날 때** 기록되므로 `timestamp` 는 종료
    시각이고, 시작은 `timestamp - seconds` 다.
    """
    # ⚠️ `records.ALL` 을 쓰는 이유. `sweep.py` 는 `env_hash` 를 `sweep_start`
    #    /`sweep_done` 에만 적고 `slice` 줄에는 안 적는다 (지금은 적지만 이미
    #    측정한 파일에는 없다). 그래서 **직전 `sweep_start` 의 조건을
    #    물려받아** 판정한다. 필터를 건너뛰는 것이 아니라, 파일에 없는 필드를
    #    타임라인에서 복원하는 것이다 — 아래에서 반드시 걸러낸다.
    out: list[Slice] = []
    cur = ""
    for r in records.iter_records(sweep_path, records.ALL):
        ev = r.get("event")
        if ev == "sweep_start":
            cur = str(r.get("env_hash") or "")
            continue
        if ev != "slice":
            continue
        eh = str(r.get("env_hash") or "") or cur
        if env_hash != records.ALL and not eh.startswith(env_hash):
            continue
        t1 = _ts(r.get("timestamp"))
        if t1 is None:
            continue
        sec = float(r.get("seconds") or 0.0)
        out.append(Slice(int(r.get("round", 0)), int(r["segment"]),
                         t1 - sec - SLICE_SLACK_S, t1 + SLICE_SLACK_S))
    out.sort(key=lambda s: s.t0)
    return out


def assign_rounds(rows: list[dict],
                  slices: list[Slice]) -> tuple[list[int | None], str]:
    """앵커 줄마다 라운드를 붙인다. 기록된 값이 있으면 그것을 쓴다.

    돌려주는 두 번째 값은 출처다 — 재구성한 것인지 기록된 것인지를 리포트에
    밝혀야 한다. 재구성은 근사이므로 근거를 숨기면 안 된다.
    """
    if rows and all(r.get("round") is not None for r in rows):
        return [int(r["round"]) for r in rows], "recorded"
    if not slices:
        return [None] * len(rows), "none"
    by_seg: dict[int, list[Slice]] = defaultdict(list)
    for s in slices:
        by_seg[s.segment].append(s)
    out: list[int | None] = []
    for r in rows:
        rec = r.get("round")
        if rec is not None:
            out.append(int(rec))
            continue
        t = _ts(r.get("timestamp"))
        seg = r.get("segment")
        cand = [s for s in by_seg.get(seg, []) if t is not None and s.t0 <= t <= s.t1]
        # 구간이 겹치면 늦게 시작한 쪽. sweep 은 순차 실행이라 정상적으로는
        # 겹치지 않지만, 여유(SLICE_SLACK_S) 때문에 경계에서 겹칠 수 있다.
        out.append(max(cand, key=lambda s: s.t0).round if cand else None)
    return out, "timestamp"


def load(env_hash: str, results_dir: str | Path | None = None
         ) -> tuple[list[dict], list[int | None], str, int]:
    """`(앵커 줄, 라운드, 라운드 출처, 슬라이스 수)`."""
    from kerneltab.core import paths
    d = Path(results_dir) if results_dir else paths.RESULTS_DIR
    rows = records.load_records(d / "anchors.jsonl", env_hash)
    sl = slice_intervals(d / "sweep.jsonl", env_hash)
    rnd, src = assign_rounds(rows, sl)
    return rows, rnd, src, len(sl)


# ---------------------------------------------------------------------------
# 판정
# ---------------------------------------------------------------------------
def analyze(rows: list[dict], rnd: list[int | None], env_hash: str,
            tol_pct: float = DEFAULT_TOL_PCT, round_source: str = "none",
            n_slices: int = 0) -> AnchorReport:
    """앵커 판정. `check_anchors.py` 와 `report_phase3.py` 가 같이 쓴다."""
    rep = AnchorReport(env_hash=env_hash[:8], n_rows=len(rows),
                       tol_pct=tol_pct, round_source=round_source,
                       n_slices=n_slices)
    if not rows:
        return rep

    def key_of(r):
        return (r["kernel_id"], r["problem"]["M"])

    by_key: dict[tuple, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for r in rows:
        by_key[key_of(r)][r["segment"]].append(r["time_ms"])

    # 짧은 앵커부터. 드리프트는 런치당 상수라 긴 커널에서는 안 보인다.
    order = sorted(by_key, key=lambda k: statistics.median(
        [t for v in by_key[k].values() for t in v]))
    n_judged = max(len(order) // 2, 1)
    base = {k: statistics.median([t for v in by_key[k].values() for t in v])
            for k in order}

    # --- 노이즈 바닥 ------------------------------------------------------
    # start/end 쌍은 같은 프로세스·같은 조건이므로 그 차이는 순수하게 시간에
    # 따른 측정 노이즈다. 이것이 판정의 분모가 된다.
    #
    # ⚠️ **집계 수준을 sB 와 맞춰야 한다.** sB 는 *세그먼트 중앙값*들의
    #    표준편차다. 따라서 분모도 "세그먼트 중앙값이 얼마나 흔들리는가"
    #    (=표준오차)여야 한다. 슬라이스 1회 측정의 노이즈를 분모로 쓰면
    #    분모만 커져 비율이 낮게 나오고, 판정이 **조용히 느슨해진다.**
    #    그래서 두 개를 따로 계산한다.
    #
    #      s_within        세그먼트로 묶은 start/end -> 세그먼트 중앙값의 SE
    #                      (sB 와 같은 수준. 판정에 쓴다)
    #      s_within_slice  슬라이스(라운드,세그먼트)로 묶은 start/end
    #                      -> 1회 측정의 노이즈 (참고. 평균화 효과를 본다)
    def _pairs(level):
        buf: dict[tuple, dict[object, dict[str, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))
        for r, q in zip(rows, rnd, strict=True):
            buf[key_of(r)][level(r, q)][r["when"]].append(r["time_ms"])
        out: dict[tuple, float | None] = {}
        for k in order:
            d = [(statistics.median(w["start"]) / statistics.median(w["end"]) - 1)
                 * 100 for w in buf[k].values()
                 if w.get("start") and w.get("end")]
            # 독립 두 측정의 차이는 분산이 2배이므로 sqrt(2) 로 나눈다.
            out[k] = statistics.pstdev(d) / (2 ** 0.5) if len(d) >= 3 else None
        return out

    noise = _pairs(lambda r, q: r["segment"])
    noise1 = _pairs(lambda r, q: (q, r["segment"]))

    # --- 세그먼트 간 편차 -------------------------------------------------
    for i, k in enumerate(order):
        med = {s: statistics.median(v) for s, v in by_key[k].items()}
        overall = statistics.median(med.values())
        vals = [100 * (v / overall - 1) for v in med.values()]
        s_b = statistics.pstdev(vals) if len(vals) >= 2 else 0.0
        s_w = noise[k]
        ratio = None if s_w is None else s_b / max(s_w, 1e-9)
        ok = (s_b <= tol_pct) if ratio is None else (
            ratio <= RATIO_MAX or s_b <= tol_pct)
        judged = i < n_judged
        rep.stats.append(AnchorStat(k[0], k[1], overall, len(med),
                                    max(vals) - min(vals), s_b, s_w,
                                    noise1[k], 100 * noise_model.sigma_rel(overall),
                                    ratio, judged, ok))
        if judged and not ok:
            rep.failures.append(
                f"`{k[0][-24:]}`@{k[1]} 의 세그먼트 간 편차가 노이즈로 "
                f"설명되지 않는다 (sB={s_b:.2f}%, sW="
                f"{'n/a' if s_w is None else format(s_w, '.2f') + '%'})")

    # --- 측정 노이즈가 모델과 맞는가 (교차 검증) --------------------------
    # `core/noise.py` 의 sigma_rel(t) = 0.000374/t + 0.00044 는 **다른**
    # 데이터(캠페인 앵커의 절대 시간 분포)에서 나온 것이다. 여기서 관측한
    # 슬라이스 내 노이즈와 맞으면 모델이 독립적으로 검증된 것이고, 크게
    # 벗어나면 그 앵커에 노이즈가 아닌 무언가가 있다.
    for st in rep.stats:
        if st.s_within_slice is None or st.model_pct <= 0:
            continue
        rt = st.s_within_slice / st.model_pct
        if rt > MODEL_RATIO_WARN:
            rep.notes.append(
                f"`{st.kernel_id[-24:]}`@{st.M} 의 측정 노이즈가 모델의 "
                f"{rt:.1f}배다 ({st.s_within_slice:.2f}% vs {st.model_pct:.2f}%). "
                f"노이즈로 보기 어려운 성분이 있다 — 절대값 추이를 함께 볼 것.")

    if all(q is None for q in rnd):
        return rep

    # --- 라운드별 추이 ----------------------------------------------------
    short3 = set(order[: max(len(order) // 3, 1)])
    by_round: dict[int, list[float]] = defaultdict(list)
    for r, q in zip(rows, rnd, strict=True):
        if q is not None and key_of(r) in short3:
            by_round[q].append(r["time_ms"] / base[key_of(r)])
    # 단조 증가 판정은 **평균**으로 한다. 중앙값은 타이머 눈금에 걸려
    # 라운드가 달라도 같은 값이 나오므로 추세를 감지하지 못한다.
    prev, mono = None, 0
    for q in sorted(by_round):
        m = statistics.median(by_round[q])
        mu = statistics.fmean(by_round[q])
        if prev is not None:
            mono += 1 if mu > prev else -1
        # ⚠️ 중앙값과 평균을 같이 낸다. 짧은 앵커의 시간은 CUDA 이벤트
        #    타이머 눈금(1.024 us)에 걸려 **양자화**되어 있다. 14 us 앵커에서
        #    한 눈금은 7.3% 이므로 중앙값은 대부분 같은 눈금에 떨어져 정확히
        #    0.000% 로 찍힌다. 그것을 "완벽하게 안정적" 으로 읽으면 안 된다 —
        #    중앙값이 둔한 것이다. 평균은 눈금 사이 비율을 반영한다.
        rep.rounds.append(RoundPoint(q, 100 * (m - 1), 100 * (mu - 1),
                                     len(by_round[q])))
        prev = mu
    if len(by_round) >= 4 and mono >= len(by_round) - 1:
        rep.failures.append(
            "라운드마다 단조 증가한다 — 세그먼트 밖에 다른 누적이 있다. "
            "프로세스 재시작이 완전히 리셋하지 못한다는 뜻이므로 드라이버 "
            "수준 상태를 의심해야 한다.")

    # --- 절대값 추이 ------------------------------------------------------
    # 비율만 보면 **모든 세그먼트가 함께 나빠지는** 경우를 놓친다. 편차는
    # 0인데 전체가 드리프트하는 상황이다.
    per: dict[int, dict[tuple, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for r, q in zip(rows, rnd, strict=True):
        if q is not None:
            per[q][key_of(r)].append(r["time_ms"])
    if len(per) >= 2:
        f, la = min(per), max(per)
        rep.abs_first, rep.abs_last = f, la
        worst = 0.0
        for i, k in enumerate(order):
            a, b = per[f].get(k), per[la].get(k)
            if not a or not b:
                continue
            ma, mb = statistics.median(a), statistics.median(b)
            judged = i < n_judged
            d = (mb / ma - 1) * 100
            if judged:
                worst = max(worst, abs(d))
            rep.abs_moves.append(AbsMove(k[0], k[1], ma, mb, d, judged))
        # 바닥도 **집계 수준을 맞춰야 한다.** 비교 대상은 *라운드 중앙값*
        # 두 개의 차이이므로, 바닥은 "라운드 중앙값이 얼마나 흔들리는가"
        # 여야 한다. 세그먼트 중앙값의 표준오차를 쓰면 바닥이 낮게 잡혀
        # **노이즈를 드리프트로 오판한다.**
        #
        # 흔들림은 라운드 간 **연속 차분**으로 잰다. 단순 표준편차를 쓰면
        # 드리프트 자체가 분모를 부풀려 스스로를 감춘다 — 선형 드리프트의
        # 연속 차분은 작고 일정하지만 표준편차는 크다.
        succ = []
        for k in order[:n_judged]:
            ser = [statistics.median(per[q][k]) for q in sorted(per)
                   if per[q].get(k)]
            if len(ser) >= 3:
                d = [100 * (ser[i + 1] / ser[i] - 1) for i in range(len(ser) - 1)]
                succ.append(statistics.pstdev(d))
        floor = max([ABS_FLOOR_K * x for x in succ] + [tol_pct])
        rep.abs_worst, rep.abs_floor = worst, floor
        if worst > floor:
            rep.failures.append(
                f"짧은 앵커의 절대값이 라운드 {f}->{la} 사이에 {worst:.2f}% "
                f"움직였다 (노이즈 바닥 {floor:.2f}%). 세그먼트 간 편차가 "
                f"작아도 전체가 함께 드리프트한다는 뜻이다.")

    # --- 세그먼트 안 이동 (참고) ------------------------------------------
    # 이것은 노이즈 바닥의 정의 그 자체이므로 실패 조건으로 쓰지 않는다.
    #
    # ⚠️ **중앙값과 평균을 함께 낸다.** 짧은 앵커의 시간은 타이머 눈금
    #    (1.024 us)에 양자화돼 있고, 14 us 커널에서 한 눈금은 7.3 % 다.
    #    중앙값만 보면 눈금 하나가 "-6.49 %" 라는 큰 이상 신호처럼 보인다.
    #    실제로 그런 값이 나왔고 확인해 보니 **0.84 눈금**이었다.
    #    `tick_pct` 를 함께 실어 "분해 가능한 차이인가" 를 판정할 수 있게 한다.
    for k in order[: max(len(order) // 3, 1)]:
        st = [r["time_ms"] for r in rows
              if key_of(r) == k and r["when"] == "start"]
        en = [r["time_ms"] for r in rows
              if key_of(r) == k and r["when"] == "end"]
        if st and en:
            ms, me = statistics.median(st), statistics.median(en)
            rep.within_slice.append((
                k[0], k[1],
                (me / ms - 1) * 100,
                (statistics.fmean(en) / statistics.fmean(st) - 1) * 100,
                100 * noise_model.tick_pct(ms)))
    return rep
