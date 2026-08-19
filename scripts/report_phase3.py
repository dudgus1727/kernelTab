#!/usr/bin/env python3
"""Phase 3 전수 측정 리포트 — 표 품질과 물리 현상.

규칙 평가(정적 top-k, regret 등)는 여기 넣지 않는다. 그것은 이 표를 소비하는
별도 프로젝트(kernelrule)의 일이고, 측정 하네스가 자기 데이터로 규칙을
평가하기 시작하면 "표를 만드는 도구" 와 "표를 쓰는 도구" 의 경계가 무너진다.

    python3 scripts/report_phase3.py                     # 현재 env.json 의 해시
    python3 scripts/report_phase3.py --env-hash b42df475 # 특정 조건만
    python3 scripts/report_phase3.py --out results/report

메모리 절약을 위해 results.jsonl 을 **두 번 스트리밍**한다 (1 회차: 형상별
최고 성능과 status, 2 회차: 축별 상대 성능 집계). 백만 줄을 통째로 메모리에
올리지 않는다.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backends import get_backend
from build import paths
from core import anchors as A
from core import features as F
from core import records
from core.hardware import hardware_from_env
from core.types import KernelConfig, Problem, RuntimeConfig

RESULTS = paths.RESULTS_DIR / "results.jsonl"
KERNELS = paths.RESULTS_DIR / "kernels.jsonl"
DRIFT = paths.RESULTS_DIR / "drift.jsonl"
TELEMETRY = paths.RESULTS_DIR / "telemetry.csv"

THROTTLE_BITS = {
    0x0004: "sw_power_cap", 0x0008: "hw_slowdown", 0x0020: "sw_thermal",
    0x0040: "hw_thermal", 0x0080: "hw_power_brake",
}


# ---------------------------------------------------------------------------
def iter_rows(env_hash: str):
    """results.jsonl 을 스트리밍한다. 지정한 측정 조건의 줄만 넘긴다.

    구현이 `core.records` 하나로 모여 있다 — 같은 필터를 여러 곳에 복제하면
    한 곳만 빠뜨려도 조용히 틀린다 (R-5).
    """
    return records.iter_records(RESULTS, env_hash)


def load_kernels() -> dict:
    return {r["kernel_id"]: r
            for r in (json.loads(l) for l in KERNELS.read_text().splitlines() if l.strip())}


def cfg_of(k: dict, backend) -> KernelConfig:
    t, a = k["tile"], k["align"]
    return KernelConfig(t["m"], t["n"], t["k"], a["a"], a["b"], a["c"],
                        k["arch"], backend.ext_from_dict(k["ext"]))


def q(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(len(s) - 1, max(0, round(p * (len(s) - 1))))
    return s[i]


def dist(xs: list[float]) -> str:
    if not xs:
        return "n=0"
    return (f"n={len(xs)} min={min(xs):.3f} p25={q(xs, .25):.3f} "
            f"med={q(xs, .5):.3f} p75={q(xs, .75):.3f} max={max(xs):.3f}")


# ---------------------------------------------------------------------------
def _med(v):
    return statistics.median(v) if v else float("nan")


def _difficulty(shape_cfg):
    """[(형상, 난이도, 후보수)] — 난이도 내림차순.

    난이도 = 중앙값 시간 / 최적 시간. 무작위 config 가 최적 대비 몇 배 느린가.
    """
    out = []
    for k, d in shape_cfg.items():
        ts = sorted(d.values())
        if len(ts) < 5:
            continue
        out.append((k, statistics.median(ts) / ts[0], len(ts)))
    out.sort(key=lambda r: -r[1])
    return out


def _is_a888(key) -> bool:
    """이 형상이 alignment (8,8,8) 인가. 층 D(정렬 검증용)를 걸러낸다."""
    from core.config import alignments_for as _af
    from core.types import Problem as _P
    M, N, K = key
    return _af(_P(M, N, K)) == (8, 8, 8)


def _layer_difficulty(diff, hw):
    """[(층 이름, 형상 수, 난이도 중앙/최소/최대, 난이도>2 개수)]"""
    from core.shapes import all_layers
    dmap = {k: d for k, d, _ in diff}
    out = []
    for name, probs in all_layers(hw).items():
        ds = [dmap[(p.M, p.N, p.K)] for p in probs
              if (p.M, p.N, p.K) in dmap]
        if not ds:
            continue
        out.append((name, len(ds), statistics.median(ds), min(ds), max(ds),
                    sum(1 for d in ds if d > 2.0)))
    out.sort(key=lambda r: -r[2])
    return out


def _difficulty_corr(diff, hw):
    """난이도가 어떤 형상 속성과 상관되는가."""
    import math as _m

    def pearson(a, b):
        n = len(a)
        if n < 3:
            return float("nan")
        ma, mb = sum(a) / n, sum(b) / n
        ca = [x - ma for x in a]
        cb = [x - mb for x in b]
        da = sum(x * x for x in ca) ** 0.5
        db = sum(x * x for x in cb) ** 0.5
        return sum(x * y for x, y in zip(ca, cb)) / (da * db) if da and db else float("nan")

    y = [d for _, d, _ in diff]
    M = [k[0] for k, _, _ in diff]
    N = [k[1] for k, _, _ in diff]
    K = [k[2] for k, _, _ in diff]
    flop = [2 * a * b * c for a, b, c in zip(M, N, K)]
    byte = [2 * (a * c + c * b + a * b) for a, b, c in zip(M, N, K)]
    ai = [f / b for f, b in zip(flop, byte)]
    return [
        ("log M", pearson(y, [_m.log(max(x, 1)) for x in M])),
        ("log N", pearson(y, [_m.log(max(x, 1)) for x in N])),
        ("log K", pearson(y, [_m.log(max(x, 1)) for x in K])),
        ("log 연산강도 (FLOP/byte)", pearson(y, [_m.log(max(x, 1e-9)) for x in ai])),
        ("log 총 FLOP", pearson(y, [_m.log(max(x, 1)) for x in flop])),
        ("후보 수", pearson(y, [n for _, _, n in diff])),
    ]


def _static_topk(shape_cfg, diff, kmax=8):
    """형상 무관 고정 k 개의 geomean regret. 탐욕적으로 고른다.

    전체 / 어려운 절반 / 쉬운 절반으로 나눠 돌려준다. 정적 top-k 의 regret 이
    낮게 나올 때 그게 "선택이 쉽다" 인지 "쉬운 형상이 많다" 인지 구분하려면
    이 분해가 필요하다.
    """
    import math as _m

    shapes = [k for k, _, _ in diff]
    best = {k: min(shape_cfg[k].values()) for k in shapes}
    half = len(diff) // 2
    hard = {k for k, _, _ in diff[:half]}
    # config -> {형상: regret}
    cfgs = {}
    for k in shapes:
        for i, t in shape_cfg[k].items():
            cfgs.setdefault(i, {})[k] = t / best[k]
    # 측정되지 않은 (형상, config) 에 벌점을 주면 안 된다. alignment 가
    # 형상을 나누므로 **어떤 config 도 모든 형상에서 측정되지 않는다** —
    # a888 커널은 alignment 4 형상에 애초에 쓰이지 않는다. 벌점을 주면
    # 그 구조적 사실이 regret 으로 둔갑한다.
    # 그래서 **덮은 형상에서만** regret 을 재고 덮개율을 따로 보고한다.
    def geo(sel, group):
        tot = 0.0
        n = 0
        for k in group:
            rs = [cfgs[i][k] for i in sel if k in cfgs[i]]
            if not rs:
                continue
            tot += _m.log(min(rs))
            n += 1
        if not n:
            return float("nan"), 0.0
        return _m.exp(tot / n), n / len(group)

    hard_s = [k for k in shapes if k in hard]
    easy_s = [k for k in shapes if k not in hard]
    chosen = []
    out = []
    pool = list(cfgs)
    for _ in range(kmax):
        bestc, bestkey = None, None
        for c in pool:
            g, cov = geo([*chosen, c], shapes)
            # 덮개를 먼저 늘리고, 같은 덮개면 regret 을 줄인다
            key = (-cov, 1e9 if math.isnan(g) else g)
            if bestkey is None or key < bestkey:
                bestc, bestkey = c, key
        if bestc is None:
            break
        chosen.append(bestc)
        pool.remove(bestc)
        g, cov = geo(chosen, shapes)
        gh, _ = geo(chosen, hard_s)
        ge, _ = geo(chosen, easy_s)
        out.append((len(chosen), g, gh, ge, cov))
    return out


def _rank_stability(by):
    """형상마다 최적 대비 1%/5% 이내 config 수. 순위가 의미 있는지 본다.

    작은 형상은 커널 시간이 런치 오버헤드와 같은 자릿수라 config 를 바꿔도
    성능이 거의 같다. 그런 형상에서는 1 등과 2 등의 차이가 측정 노이즈보다
    작아 **순위 자체가 의미 없다.** 규칙이 그걸 알아야 틀린 선택에 부당한
    벌점을 주지 않는다.
    """
    out = []
    for (M, N, K), ts in by.items():
        if len(ts) < 5:
            continue
        best = min(ts)
        out.append((M, N, K, len(ts), best,
                    sum(1 for t in ts if t <= best * 1.01),
                    sum(1 for t in ts if t <= best * 1.05)))
    # 1% 이내 비율이 높은 순
    out.sort(key=lambda r: -r[5] / r[3])
    return out


def _anchor_report(env_hash: str):
    """앵커 판정. **계산은 `core/anchors.py` 가 한다** (R-6).

    예전에는 이 함수가 판정을 따로 했다 — 절대 1% 기준으로. `check_anchors.py`
    는 노이즈 대비로 판정해서 **같은 데이터에 두 답**이 나왔다. 512³ 앵커는
    12~23 us 라 노이즈 자체가 2% 대이므로 1% 절대 기준은 달성 불가능하고,
    못 달성했다고 대책이 실패한 것도 아니다. 리포트 쪽이 틀렸는데, 사람이
    읽는 것은 리포트라 틀린 답이 더 널리 읽혔다.
    """
    from core import anchors
    rows, rnd, src, n_sl = anchors.load(env_hash)
    if not rows:
        return None
    return anchors.analyze(rows, rnd, env_hash, round_source=src,
                           n_slices=n_sl)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-hash", default=None,
                    help="분석할 측정 조건. 기본은 현재 env.json 의 해시")
    ap.add_argument("--out", default=str(paths.RESULTS_DIR / "report"))
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    env = json.loads(paths.ENV_JSON.read_text())
    hw = hardware_from_env(env)
    backend = get_backend(hw.arch)
    env_hash = args.env_hash or env["env_hash"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    kern = load_kernels()
    cfg_cache: dict[str, KernelConfig] = {}

    def cfg(kid: str) -> KernelConfig | None:
        if kid not in cfg_cache:
            k = kern.get(kid)
            if k is None:
                return None
            cfg_cache[kid] = cfg_of(k, backend)
        return cfg_cache[kid]

    L: list[str] = []          # 마크다운 줄
    def w(s: str = "") -> None:
        L.append(s)
        print(s)

    w(f"# Phase 3 리포트 — `env_hash = {env_hash[:16]}`")
    w()
    w(f"- GPU: {hw.name} ({hw.arch}), SM {hw.sm_count}")
    w(f"- 실효 피크 {hw.peak_tflops_f16} TFLOP/s @ {env.get('locked_mhz')} MHz, "
      f"실효 대역폭 {hw.bandwidth_gbps} GB/s @ {env.get('locked_mem_mhz')} MHz")
    w(f"- ridge point {F.ridge_point(hw):.1f} FLOP/byte")
    w(f"- 프로토콜 {env.get('protocol')}")
    w()

    # ---------------- 1 회차: status, 형상별 최고, cuBLAS -----------------
    status = Counter()
    err_reason = Counter()
    best: dict[tuple, dict] = {}
    cublas: dict[tuple, float] = {}
    # 순위 안정성용. 형상당 시간 목록만 모은다 (형상 66개라 메모리는 작다).
    shape_times: dict[tuple, list] = defaultdict(list)
    # 난이도 / 정적 top-k 용: (형상) -> {config 인덱스: 시간}
    # config 를 문자열 대신 정수로 인터닝해서 메모리를 줄인다.
    cfg_ix: dict[tuple, int] = {}
    shape_cfg: dict[tuple, dict] = defaultdict(dict)
    n_rows = 0
    tmin = tmax = None
    for d in iter_rows(env_hash):
        n_rows += 1
        _ts = parse_ts(d.get("timestamp", ""))
        if _ts is not None:
            tmin = _ts if tmin is None else min(tmin, _ts)
            tmax = _ts if tmax is None else max(tmax, _ts)
        p = d["problem"]
        key = (p["M"], p["N"], p["K"])
        if d["kernel_id"] == "cublas":
            if d.get("time_ms"):
                cublas[key] = min(cublas.get(key, math.inf), d["time_ms"])
            continue
        if d.get("status") == "ok" and d.get("time_ms"):
            shape_times[key].append(d["time_ms"])
            rt = d.get("runtime") or {}
            ck = (d["kernel_id"], rt.get("split_k"), rt.get("split_k_mode"))
            i = cfg_ix.get(ck)
            if i is None:
                i = cfg_ix[ck] = len(cfg_ix)
            cur = shape_cfg[key].get(i)
            if cur is None or d["time_ms"] < cur:
                shape_cfg[key][i] = d["time_ms"]
        st = d.get("status")
        status[st] += 1
        if st != "ok":
            why = (d.get("error") or "")[:60]
            if not why and st == "numerical_fail":
                why = f"max_rel_error={d.get('max_rel_error')}"
            elif not why and st == "high_outlier_frac":
                why = f"outlier_frac={d.get('outlier_frac')}"
            err_reason[(st, why)] += 1
            continue
        t = d.get("time_ms")
        if not t:
            continue
        if key not in best or t < best[key]["time_ms"]:
            best[key] = d

    w("## 1. status 분포")
    w()
    w("| status | 건수 | 비율 |")
    w("|---|---:|---:|")
    tot = sum(status.values())
    for k, v in status.most_common():
        w(f"| `{k}` | {v:,} | {100 * v / max(tot, 1):.2f}% |")
    w(f"\n측정 줄 {tot:,} (+ cuBLAS 참조 {len(cublas)} 형상), 파일에서 읽은 줄 {n_rows:,}")
    if err_reason:
        w("\n실패 원인별 (상위 15):\n")
        w("| status | 사유 | 건수 |")
        w("|---|---|---:|")
        for (st, why), v in err_reason.most_common(15):
            w(f"| `{st}` | `{why or '-'}` | {v:,} |")
    w()

    # ---------------- 2. 드리프트 / 텔레메트리 ----------------------------
    w("## 2. 드리프트 / 텔레메트리")
    w()
    ds = [json.loads(l) for l in DRIFT.read_text().splitlines() if l.strip()] \
        if DRIFT.exists() else []
    ds = [d for d in ds if d.get("time_ms")]
    # drift.jsonl / telemetry.csv 에는 env_hash 가 없다 (기록 누락 — 마이그레이션
    # 항목). 그래서 측정 줄의 타임스탬프 창으로 걸러 조건을 맞춘다.
    if tmin is not None:
        ds = [d for d in ds
              if (x := parse_ts(d.get("timestamp", ""))) is not None
              and tmin - 120 <= x <= tmax + 120]
    w(f"_drift.jsonl / telemetry.csv 에 `env_hash` 가 없어 측정 타임스탬프 "
      f"창({fmt_ts(tmin)} ~ {fmt_ts(tmax)}) 으로 걸렀다._")
    w()
    if len(ds) >= 2:
        ts = [d["time_ms"] for d in ds]
        mean = statistics.mean(ts)
        span = (max(ts) - min(ts)) / mean
        w(f"- 기준 config 재측정 {len(ds)}회: min={min(ts):.4f} max={max(ts):.4f} "
          f"mean={mean:.4f} std={statistics.pstdev(ts):.5f} ms")
        w(f"- **변동 폭 {100 * span:.2f}%** "
          + ("(5% 이하 — 조건 유지됨)" if span <= 0.05 else "**(5% 초과 — 경고)**"))
        cl = [d.get("sm_clock_mhz") for d in ds if d.get("sm_clock_mhz")]
        tp = [d.get("gpu_temp_c") for d in ds if d.get("gpu_temp_c")]
        if cl:
            w(f"- 클럭 {min(cl)}~{max(cl)} MHz, 온도 {min(tp)}~{max(tp)} °C")
    else:
        w("- 드리프트 기록이 2 회 미만이다.")
    w()
    tel = telemetry_summary(tmin, tmax)
    if tel:
        w(f"- 텔레메트리 {tel['n']:,}초")
        w(f"  - SM 클럭 {tel['clk_min']}~{tel['clk_max']} MHz (중앙값 {tel['clk_med']})")
        w(f"  - 메모리 클럭 {tel['mem_min']}~{tel['mem_max']} MHz (중앙값 {tel['mem_med']})")
        w(f"  - 온도 {tel['t_min']}~{tel['t_max']} °C, 전력 {tel['p_min']}~{tel['p_max']} W")
        if tel["throttle"]:
            w("  - **스로틀 발생**: "
              + ", ".join(f"{k} {v}초 ({100 * v / tel['n']:.1f}%)"
                          for k, v in tel["throttle"].items()))
        else:
            w("  - 스로틀 발생 구간 없음")
    w()

    # ---------------- 2-b. 앵커 (세그먼트 간 계통 오차) --------------------
    w("## 2-b. 앵커 — 세그먼트 간 계통 오차")
    w()
    w("측정은 커널을 세그먼트로 나눠 **세그먼트마다 새 프로세스**로 돈다")
    w("(드리프트 대책, `docs/measurement_drift.md`). 프로세스가 다르면 계통")
    w("오차가 생길 수 있으므로 모든 세그먼트에서 같은 앵커 커널을 잰다.")
    w()
    w("⚠️ 판정은 **짧은 앵커**로 한다. 드리프트의 정체는 런치당 상수라 긴")
    w("커널에서는 안 보인다.")
    w()
    anc = _anchor_report(env_hash)
    if anc is None:
        w("`results/anchors.jsonl` 이 없다 — 세그먼트 방식으로 측정하지 않았다.")
    else:
        w("판정은 **절대 기준이 아니라 노이즈 대비**다. 512³ 앵커는 14~56 us 라")
        w("측정 노이즈 자체가 몇 %다. 물어야 할 것은 \"노이즈 대비 계통 성분이")
        w("있는가\" 이지 \"변동폭이 1% 미만인가\" 가 아니다.")
        w()
        w("| 앵커 | 형상 | 중앙(ms) | 세그 | 폭 | sB | sW | 모델 | 비율 | 판정 |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|")
        for st in anc.stats:
            rs = "n/a" if st.ratio is None else f"{st.ratio:.2f}"
            mark = ("OK" if st.ok else "**실패**") if st.judged else "참고"
            sw = "n/a" if st.s_within is None else f"{st.s_within:.2f}%"
            w(f"| `{st.kernel_id[-38:]}` | {st.M} | {st.median_ms:.4f} "
              f"| {st.n_segments} | {st.spread_pct:.2f}% | {st.s_between:.2f}% "
              f"| {sw} | {st.model_pct:.2f}% | {rs} | {mark} |")
        w()
        w("- `sB` = 세그먼트 중앙값들의 표준편차 (계통 성분)")
        w("- `sW` = 세그먼트 중앙값의 표준오차 (노이즈). **`sB` 와 집계 수준을**")
        w("  **맞춘 값이다** — 1회 측정 노이즈를 분모로 쓰면 판정이 조용히")
        w("  느슨해진다.")
        w("- `모델` = `core/noise.py` 의 `sigma_rel(t)=0.000374/t+0.00044`.")
        w("  **다른 데이터에서 유도한 모델**이므로 `sW` 와 맞으면 교차 검증이다.")
        w(f"- 판정: 비율 <= {A.RATIO_MAX} 또는 `sB` <= {anc.tol_pct}% "
          f"(짧은 앵커 절반만. 드리프트는 런치당 상수라 긴 커널에서는 안 보인다)")
        w()

        # --- 라운드 ------------------------------------------------------
        w("**라운드 추이** — 세그먼트 밖에 누적이 있는가")
        w()
        src_txt = {"recorded": "앵커 줄에 기록된 `round`",
                   "timestamp": "`sweep.jsonl` 슬라이스 구간에서 시각으로 복원",
                   "none": "알 수 없음"}[anc.round_source]
        w(f"라운드 출처: {src_txt} (슬라이스 {anc.n_slices}개).")
        w()
        if anc.rounds:
            w("| 라운드 | 기준선 대비 (중앙) | (평균) | n |")
            w("|---:|---:|---:|---:|")
            for pt in anc.rounds:
                w(f"| {pt.round} | {pt.rel_pct:+.3f}% | {pt.mean_pct:+.3f}% "
                  f"| {pt.n} |")
            w()
            w("중앙값이 0.000% 로 고정돼 보이는 것은 **타이머 눈금** 때문이다.")
            w("CUDA 이벤트 타이머의 눈금은 1.024 us 이고, 14 us 앵커에서 한")
            w("눈금은 7.3% 다. 중앙값은 대부분 같은 눈금에 떨어진다 — 완벽히")
            w("안정적인 것이 아니라 **중앙값이 둔한 것**이다. 평균을 함께 본다.")
            w()
        else:
            w("라운드를 알 수 없어 이 검사가 돌지 않았다.")
            w()
        if anc.abs_moves:
            w(f"**절대값 이동** (라운드 {anc.abs_first} → {anc.abs_last}). ")
            w("비율만 보면 **모든 세그먼트가 함께** 나빠지는 경우를 놓친다 —")
            w("편차는 0인데 전체가 드리프트하는 상황이다.")
            w()
            w(f"| 앵커 | 형상 | R{anc.abs_first}(ms) | R{anc.abs_last}(ms) | 변화 |")
            w("|---|---:|---:|---:|---:|")
            for m in anc.abs_moves:
                w(f"| `{m.kernel_id[-38:]}` | {m.M} | {m.first_ms:.4f} "
                  f"| {m.last_ms:.4f} | {m.delta_pct:+.2f}% |")
            w()
            w(f"짧은 앵커 최대 이동 **{anc.abs_worst:.2f}%**, "
              f"노이즈 바닥 {anc.abs_floor:.2f}%.")
            w()
        if anc.notes:
            w("**주의** (실패는 아니다)")
            w()
            for n in anc.notes:
                w(f"- {n}")
            w()
        w(f"→ {anc.verdict}")
    w()
    w("자세한 판정은 `python3 scripts/check_anchors.py`.")
    w()

    # ---------------- 2-c. 순위 안정성 -------------------------------------
    w("## 2-c. 순위 안정성 — 순위가 의미 있는 형상인가")
    w()
    w("작은 형상은 커널 실행 시간이 런치 오버헤드와 같은 자릿수라, config 를")
    w("바꿔도 성능이 거의 같다. 그런 형상에서는 **1 등과 2 등의 차이가 측정")
    w("노이즈보다 작아 순위 자체가 의미 없다.**")
    w()
    w("각 형상에서 최적 대비 1 % 이내에 들어오는 config 가 몇 개인지 센다.")
    w("이 수가 크면 그 형상에서는 어떤 config 를 골라도 같다는 뜻이고,")
    w("규칙이 그걸 알아야 한다 (틀린 선택에 벌점을 주면 안 된다).")
    w()
    st = _rank_stability(shape_times)
    if not st:
        w("(측정 데이터가 부족하다)")
    else:
        w("| 형상 | 후보 | 최적(ms) | 1% 이내 | 5% 이내 | 순위 |")
        w("|---|---:|---:|---:|---:|---|")
        for _M, _N, _K, _n, _best, _w1, _w5 in st[:25]:
            tag = ("**의미 없음**" if _w1 >= max(3, 0.10 * _n)
                   else ("주의" if _w1 >= 3 else "유효"))
            w(f"| {_M}x{_N}x{_K} | {_n} | {_best:.4f} | {_w1} | {_w5} | {tag} |")
        n_bad = sum(1 for r in st if r[5] >= max(3, 0.10 * r[3]))
        w()
        w(f"- 순위가 의미 없는 형상 **{n_bad}/{len(st)}** "
          f"(최적 대비 1% 이내 config 가 후보의 10% 이상)")
        w("- 이 형상들은 `launch_overhead_frac` 이 큰 쪽에 몰려 있어야 한다 "
          "— 그렇지 않으면 다른 원인이다")
    w()

    # ---------------- 2-d. 형상 난이도 / 정적 top-k -----------------------
    w("## 2-d. 형상 난이도 — 평가를 층화해야 하는가")
    w()
    w("`난이도 = 중앙값 시간 / 최적 시간`. **무작위로 고른 config 가 최적")
    w("대비 몇 배 느린가**를 뜻한다. 1.05 면 아무거나 골라도 되는 형상,")
    w("3.0 이면 선택이 결정적인 형상이다.")
    w()
    w("이게 중요한 이유: regret 을 형상별로 **균등 평균** 내면 쉬운 형상이")
    w("분모를 부풀린다. 어려운 형상에서 20 % 손해를 봐도 쉬운 형상 10 개가")
    w("1.00 이면 전체 geomean 이 좋아 보인다. 쉬운 형상이 많으면 균등")
    w("geomean 은 의미가 없고, 난이도로 가중하거나 어려운 형상만 따로")
    w("보고해야 한다.")
    w()
    diff = _difficulty(shape_cfg)
    if len(diff) < 4:
        w("(형상이 부족해 계산할 수 없다)")
    else:
        ds = sorted(d for _, d, _ in diff)
        w(f"- 형상 {len(diff)}개, 난이도 중앙값 **{_med(ds):.2f}배**, "
          f"범위 {ds[0]:.2f} ~ {ds[-1]:.2f}")
        w(f"- 난이도 < 1.2 (선택이 거의 무의미) **"
          f"{sum(1 for d in ds if d < 1.2)}개**")
        w(f"- 난이도 > 2.0 (선택이 결정적) **"
          f"{sum(1 for d in ds if d > 2.0)}개**")
        w()
        w("| | 형상 | 후보 | 최적(ms) | 난이도 |")
        w("|---|---|---:|---:|---:|")
        for tag, sel in (("가장 어려움", diff[:8]), ("가장 쉬움", diff[-8:])):
            for k, d, n in sel:
                w(f"| {tag} | {k[0]}x{k[1]}x{k[2]} | {n} | "
                  f"{min(shape_cfg[k].values()):.4f} | **{d:.2f}** |")
        w()
        # 층별 기여 — 어느 층이 어려운 형상을 공급하는가
        w("**층별 난이도** — 어느 층이 어려운 형상을 공급하는가")
        w()
        lay = _layer_difficulty(diff, hw)
        if lay:
            w("| 층 | 형상 | 난이도 중앙 | 최소 | 최대 | 난이도>2.0 |")
            w("|---|---:|---:|---:|---:|---:|")
            for name, n, med, lo, hi, nhard in lay:
                w(f"| {name} | {n} | **{med:.2f}** | {lo:.2f} | {hi:.2f} "
                  f"| {nhard} |")
            w()
            w("층을 하나 빼면 난이도 분포가 통째로 달라 보인다. 다른 GPU 로")
            w("갈 때 어느 층을 촘촘히 할지의 근거가 여기 있다.")
        w()
        # 상관
        w("**난이도와 무엇이 상관되는가**")
        w()
        cor = _difficulty_corr(diff, hw)
        for name, r in cor:
            w(f"- {name}: r = {r:+.3f}")
        w()

    w("### 정적 top-k — 형상 무관 고정 config")
    w()
    w("형상을 보지 않고 **고정된 k 개**만 시도했을 때의 regret 이다.")
    w("규칙이 이겨야 하는 하한선이고, 동시에 **문제가 얼마나 쉬운지**의")
    w("척도다. 전체 / 어려운 절반 / 쉬운 절반으로 나눠 본다 — 세 숫자가")
    w("크게 다르면 층화가 필수라는 증거다.")
    w()
    if len(diff) >= 4:
        # a888 형상만: alignment 가 형상을 나누므로 전체로 재면 덮개가
        # 100% 가 되지 않는다. a888 로 좁히면 덮개가 깨끗해지고 해석이
        # 분명해진다. 층 D 5개는 alignment 검증용이지 실무 워크로드가
        # 아니므로 **주 지표에서 빼는 편이 오히려 맞다.**
        a8 = [d for d in diff if _is_a888(d[0])]
        if len(a8) >= 4:
            tk8 = _static_topk(shape_cfg, a8, kmax=20)
            w(f"**주 지표 — a888 형상 {len(a8)}개만** "
              f"(alignment 가 갈리지 않아 덮개가 깨끗하다)")
            w()
            w("| k | 전체 | 어려운 절반 | 쉬운 절반 | 덮개 |")
            w("|---:|---:|---:|---:|---:|")
            for k, a, b, c, cov in tk8:
                w(f"| {k} | {a:.4f} | **{b:.4f}** | {c:.4f} | {100 * cov:.0f}% |")
            w()
        w(f"**참고 — 전체 형상 {len(diff)}개**")
        w()
        tk = _static_topk(shape_cfg, diff, kmax=20)
        w("| k | 전체 | 어려운 절반 | 쉬운 절반 | 덮개 |")
        w("|---:|---:|---:|---:|---:|")
        for k, a, b, c, cov in tk:
            w(f"| {k} | {a:.4f} | **{b:.4f}** | {c:.4f} | {100 * cov:.0f}% |")
        w()
        w("(geomean regret = 고른 것 / 최적, 1.0 이 완벽. **덮은 형상에서만**")
        w("잰다 — alignment 가 형상을 나누므로 어떤 config 도 모든 형상에서")
        w("측정되지는 않는다. 덮개가 낮으면 regret 도 신뢰할 수 없다.)")
        w()
        if tk and tk[-1][4] < 0.9:
            w(f"- ⚠️ 덮개 {100 * tk[-1][4]:.0f}% — 측정이 아직 부족하다. "
              f"이 표는 전수 완료 후에 다시 봐야 한다.")
        if tk:
            _, a1, b1, c1, _cov = tk[-1]
            if b1 - c1 > 0.02:
                w(f"- 어려운 절반과 쉬운 절반의 차이가 **{b1 - c1:.3f}** 이다. "
                  f"**층화가 필수다** — 균등 geomean 은 쉬운 형상에 가려진다.")
            else:
                w("- 두 절반의 차이가 작다. 균등 geomean 을 써도 크게 왜곡되지 "
                  "않는다.")
    w()

    # ---------------- 3 / 6. cuBLAS 대비 ----------------------------------
    w("## 3. cuBLAS 대비 — 개선 여지의 상한선")
    w()
    w("`C/A` = cuBLAS 시간 / 전수 최적 시간. **1보다 크면 우리 최적이 더 빠르다.**")
    w("이 값이 이 프로젝트가 노릴 수 있는 개선 여지의 상한선이다.")
    w()
    ca: dict[tuple, float] = {}
    for _key, b in best.items():
        c = cublas.get(key)
        if c and b.get("time_ms"):
            ca[key] = c / b["time_ms"]
    if ca:
        v = list(ca.values())
        w(f"- 분포: {dist(v)}")
        w(f"- cuBLAS 를 이긴 형상: {sum(1 for x in v if x > 1.0)}/{len(v)} "
          f"({100 * sum(1 for x in v if x > 1.0) / len(v):.1f}%)")
        w(f"- 1.10 배 이상: {sum(1 for x in v if x >= 1.1)}  /  "
          f"0.95 배 미만(뒤진 형상): {sum(1 for x in v if x < 0.95)}")
        w()
        w("형상 특성별 분해 (C/A 중앙값):")
        w()
        w("| 구간 | 형상 수 | C/A 중앙값 | 최소 | 최대 |")
        w("|---|---:|---:|---:|---:|")
        for label, sel in shape_buckets(ca, best, hw, cfg):
            vv = [ca[k] for k in sel]
            if vv:
                w(f"| {label} | {len(vv)} | {q(vv, .5):.3f} | {min(vv):.3f} | {max(vv):.3f} |")
        w()
        w("형상별 상위/하위 5:")
        w()
        w("| 형상 | C/A | 최적 ms | cuBLAS ms | 최적 config |")
        w("|---|---:|---:|---:|---|")
        srt = sorted(ca.items(), key=lambda kv: -kv[1])
        picks, seen = [], set()
        for k, _ in list(srt[:5]) + list(srt[-5:]):
            if k not in seen:
                seen.add(k)
                picks.append(k)
        for key in picks:
            b = best[key]
            w(f"| {key} | {ca[key]:.3f} | {b['time_ms']:.4f} | {cublas[key]:.4f} "
              f"| `{describe(b, kern)}` |")
    else:
        w("- cuBLAS 참조가 없다.")
    w()

    # ---------------- 2 회차: 축별 상대 성능 -----------------------------
    #   rel = (형상 최적 시간) / (이 측정 시간)  -> 1.0 이 그 형상의 최적
    axes: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    sk_best: dict[tuple, dict] = defaultdict(dict)   # 형상 -> split_k -> 최소 시간
    skmode_best: dict[tuple, dict] = defaultdict(dict)
    for d in iter_rows(env_hash):
        if d["kernel_id"] == "cublas" or d.get("status") != "ok":
            continue
        t = d.get("time_ms")
        p = d["problem"]
        key = (p["M"], p["N"], p["K"])
        b = best.get(key)
        if not (t and b):
            continue
        rel = b["time_ms"] / t
        k = kern.get(d["kernel_id"])
        if k is None:
            continue
        rt = d["runtime"]
        e = k["ext"]
        sk_best[key][rt["split_k"]] = min(
            sk_best[key].get(rt["split_k"], math.inf), t)
        skmode_best[key][(rt["split_k"], rt["split_k_mode"])] = min(
            skmode_best[key].get((rt["split_k"], rt["split_k_mode"]), math.inf), t)
        spill = (k.get("spill_stores") or 0) + (k.get("spill_loads") or 0) > 0
        axes["spill"]["스필 있음" if spill else "스필 없음"].append(rel)
        axes["pipeline"][k.get("pipeline_kind") or "?"].append(rel)
        axes["align"][f"a{k['align']['a']}{k['align']['b']}{k['align']['c']}"].append(rel)
        axes["swizzle"][f"{e['swizzle_type']}{e['swizzle_n']}"].append(rel)
        axes["warp_k"]["분할(warp_k<tile_k)" if e["warp_k"] < k["tile"]["k"]
                       else "미분할"].append(rel)
        axes["split_mode"][rt["split_k_mode"] if rt["split_k"] > 1
                           else "split_k=1"].append(rel)
        axes["threads"][str(k.get("threads"))].append(rel)
        axes["stages"][str(e["stages"])].append(rel)
        axes["tile_mn"][f"{k['tile']['m']}x{k['tile']['n']}"].append(rel)

    # ---------------- 4. split-K 가설 -------------------------------------
    w("## 4. split-K 가설 — waves < 1 형상에서 3, 6 이 1보다 빠른가")
    w()
    w("A6000 의 SM 84 = 2² × 3 × 7 이므로 3 의 배수 split-K 가 wave 정렬에")
    w("유리할 수 있다는 가설. 아래 값은 **해당 split_k 의 최적 시간 / 형상 전체")
    w("최적 시간** 이며 1.000 이 그 형상의 최적이다.")
    w()
    lo = waves_bucket(best, hw, cfg, lambda x: x < 1.0)
    w(f"waves < 1 형상 {len(lo)}개 (tile 128x128, split_k=1 기준):")
    w()
    split_table(w, lo, sk_best)
    w()
    hi3 = waves_bucket(best, hw, cfg, lambda x: 1.0 <= x < 4.0)
    w(f"참고 — 1 ≤ waves < 4 형상 {len(hi3)}개:")
    w()
    split_table(w, hi3, sk_best)
    w()
    verdict = split_verdict(lo, sk_best)
    w(f"판정 — {verdict}")
    w()

    # ---------------- 5. waves > 40 에서 split_k=16 -----------------------
    w("## 5. waves > 40 형상에서 split_k=16 현상")
    w()
    big = waves_bucket(best, hw, cfg, lambda x: x > 40.0)
    if not big:
        w("해당 형상이 없다.")
    else:
        w(f"대상 형상 {len(big)}개.")
        w()
        split_table(w, big, sk_best)
        w()
        w("각 형상의 최적 config 분해:")
        w()
        w("| 형상 | waves | 최적 split_k / mode | tile | pipeline | 최적 ms |")
        w("|---|---:|---|---|---|---:|")
        for key in sorted(big):
            b = best[key]
            c = cfg(b["kernel_id"])
            p = Problem(*key)
            rc = RuntimeConfig(b["runtime"]["split_k"], b["runtime"]["split_k_mode"])
            k = kern[b["kernel_id"]]
            wv = F.waves(p, hw, c, RuntimeConfig(1, "serial")) if c else float("nan")
            w(f"| {key} | {wv:.1f} | {rc.split_k} / {rc.split_k_mode} "
              f"| {k['tile']['m']}x{k['tile']['n']}x{k['tile']['k']} "
              f"| {k.get('pipeline_kind')} | {b['time_ms']:.4f} |")
        w()
        w(split_k16_analysis(big, sk_best, skmode_best))
    w()

    # ---------------- 7 / 8 / 9. 축별 분포 --------------------------------
    w("## 6. 축별 성능 분포")
    w()
    w("`rel` = 형상 최적 시간 / 이 측정 시간. **1.0 = 그 형상의 최적**, 작을수록 느리다.")
    w()
    titles = {
        "spill": "7. 스필 유무", "pipeline": "8. pipeline_kind",
        "align": "9-a. alignment", "swizzle": "9-b. swizzle",
        "warp_k": "9-c. warp_k 분할", "split_mode": "9-d. split_k_mode",
        "threads": "9-e. threads/block", "stages": "9-f. stages",
        "tile_mn": "9-g. threadblock tile (M,N)",
    }
    for name in ("spill", "pipeline", "align", "swizzle", "warp_k",
                 "split_mode", "threads", "stages", "tile_mn"):
        g = axes.get(name)
        if not g:
            continue
        w(f"### {titles[name]}")
        w()
        w("| 값 | n | 중앙값 | p90 | 최대 | **최적을 낸 횟수** |")
        w("|---|---:|---:|---:|---:|---:|")
        wins = Counter()
        for _key, b in best.items():
            k = kern.get(b["kernel_id"])
            if k is None:
                continue
            wins[bucket_of(name, k, b)] += 1
        for val in sorted(g, key=lambda x: -q(g[x], .5)):
            v = g[val]
            w(f"| `{val}` | {len(v):,} | {q(v, .5):.3f} | {q(v, .9):.3f} "
              f"| {max(v):.3f} | {wins.get(val, 0)} |")
        w()
        w(axis_verdict(name, g, wins))
        w()

    # ---------------- 그림 ------------------------------------------------
    if not args.no_plots and ca:
        try:
            make_plots(out, ca, axes, sk_best, best, hw, cfg)
            w("## 7. 그림")
            w()
            for png in sorted(out.glob("*.png")):
                w(f"- `{png.name}`")
            w()
        except Exception as e:  # pragma: no cover
            w(f"(그림 생성 실패: {e!r})")

    md = out / f"report_{env_hash[:8]}.md"
    md.write_text("\n".join(L) + "\n")
    print(f"\n[리포트] {md}")
    return 0


# ---------------------------------------------------------------------------
def bucket_of(name: str, k: dict, d: dict) -> str:
    e, rt = k["ext"], d["runtime"]
    if name == "spill":
        return ("스필 있음" if (k.get("spill_stores") or 0)
                + (k.get("spill_loads") or 0) > 0 else "스필 없음")
    if name == "pipeline":
        return k.get("pipeline_kind") or "?"
    if name == "align":
        return f"a{k['align']['a']}{k['align']['b']}{k['align']['c']}"
    if name == "swizzle":
        return f"{e['swizzle_type']}{e['swizzle_n']}"
    if name == "warp_k":
        return "분할(warp_k<tile_k)" if e["warp_k"] < k["tile"]["k"] else "미분할"
    if name == "split_mode":
        return rt["split_k_mode"] if rt["split_k"] > 1 else "split_k=1"
    if name == "threads":
        return str(k.get("threads"))
    if name == "stages":
        return str(e["stages"])
    return f"{k['tile']['m']}x{k['tile']['n']}"


def axis_verdict(name: str, g: dict, wins: Counter) -> str:
    """이 축이 실제로 의미가 있었는가."""
    meds = {k: q(v, .5) for k, v in g.items()}
    if len(meds) < 2:
        return "_값이 하나뿐이라 판정할 수 없다._"
    spread = max(meds.values()) - min(meds.values())
    used = [k for k in g if wins.get(k, 0) > 0]
    lines = [f"- 중앙값 격차 **{spread:.3f}** "
             f"(최고 `{max(meds, key=meds.get)}` / 최저 `{min(meds, key=meds.get)}`)"]
    lines.append(f"- 최적을 한 번이라도 낸 값: {len(used)}/{len(g)} — "
                 + (", ".join(f"`{u}`" for u in sorted(used)) if used else "없음"))
    if len(used) <= 1 and spread >= 0.05:
        lines.append(f"- → **선택 축이 아니다.** 어떤 형상에서도 `{used[0] if used else '?'}` "
                     "만 최적이고 격차도 크다. 열거에서 고정해도 손실이 없다 "
                     "(축이 무의미한 것이 아니라 **답이 정해져 있다**).")
    elif len(used) <= 1:
        lines.append(f"- → 항상 `{used[0] if used else '?'}` 가 최적이지만 중앙값 격차가 "
                     "5% 미만이다. 고정해도 손실이 작다.")
    elif spread < 0.02:
        lines.append("- → 중앙값 차이가 2% 미만이지만 최적을 낸 값이 여러 개다. "
                     "**평균으로는 안 보이고 형상에 따라 갈리는 축**이다 — "
                     "규칙이 필요한 종류.")
    else:
        lines.append("- → **의미 있는 축이다.** 값에 따라 성능이 갈리고 "
                     "최적도 형상마다 다르다.")
    return "\n".join(lines)


def describe(d: dict, kern: dict) -> str:
    k = kern.get(d["kernel_id"])
    if not k:
        return d["kernel_id"]
    e, rt = k["ext"], d["runtime"]
    return (f"tb{k['tile']['m']}x{k['tile']['n']}x{k['tile']['k']} "
            f"w{e['warp_m']}x{e['warp_n']}x{e['warp_k']} st{e['stages']} "
            f"{e['swizzle_type'][:2]}{e['swizzle_n']} "
            f"sk{rt['split_k']}{rt['split_k_mode'][:4]}")


def shape_buckets(ca: dict, best: dict, hw, cfg):
    """형상 특성별 구간. (라벨, 형상키 목록) 을 순서대로 낸다."""
    def waves_of(key):
        b = best[key]
        c = cfg(b["kernel_id"])
        return F.waves(Problem(*key), hw, c, RuntimeConfig(1, "serial")) if c else None

    out = []
    for lo, hi, lab in ((0, 1, "waves < 1"), (1, 4, "1 ≤ waves < 4"),
                        (4, 40, "4 ≤ waves < 40"), (40, 1e18, "waves ≥ 40")):
        sel = [k for k in ca if (wv := waves_of(k)) is not None and lo <= wv < hi]
        out.append((lab, sel))
    rp = F.ridge_point(hw)
    out.append(("memory-bound (AI < ridge)",
                [k for k in ca if F.arith_intensity(Problem(*k)) < rp]))
    out.append(("compute-bound (AI ≥ ridge)",
                [k for k in ca if F.arith_intensity(Problem(*k)) >= rp]))
    out.append(("M ≤ 32 (극단 skinny)", [k for k in ca if k[0] <= 32]))
    return out


def waves_bucket(best: dict, hw, cfg, pred) -> list[tuple]:
    out = []
    for key, b in best.items():
        c = cfg(b["kernel_id"])
        if c is None:
            continue
        wv = F.waves(Problem(*key), hw, c, RuntimeConfig(1, "serial"))
        if pred(wv):
            out.append(key)
    return sorted(out)


def split_table(w, shapes: list[tuple], sk_best: dict) -> None:
    if not shapes:
        w("_해당 형상이 없다._")
        return
    all_sk = sorted({s for k in shapes for s in sk_best.get(k, {})})
    w("| 형상 | " + " | ".join(f"sk={s}" for s in all_sk) + " |")
    w("|---|" + "---:|" * len(all_sk))
    for k in shapes[:20]:
        row = sk_best.get(k, {})
        b = min(row.values()) if row else None
        cells = [f"{b / row[s]:.3f}" if s in row and b else "–" for s in all_sk]
        w(f"| {k} | " + " | ".join(cells) + " |")
    if len(shapes) > 20:
        w(f"\n_({len(shapes)}개 중 20개만 표시)_")


def split_verdict(shapes: list[tuple], sk_best: dict) -> str:
    if not shapes:
        return "대상 형상이 없어 판정 불가."
    win3 = win6 = win1 = other = 0
    for k in shapes:
        row = sk_best.get(k, {})
        if not row:
            continue
        bs = min(row, key=row.get)
        if bs == 1:
            win1 += 1
        elif bs == 3:
            win3 += 1
        elif bs == 6:
            win6 += 1
        else:
            other += 1
    n = win1 + win3 + win6 + other
    if n == 0:
        return "판정 불가."
    if win3 + win6 == 0:
        return (f"**가설 미지지.** waves<1 형상 {n}개 중 split_k=3 또는 6 이 최적인 "
                f"경우가 하나도 없다 (split_k=1 최적 {win1}, 기타 {other}). "
                "3 의 배수 split-K 가 wave 정렬에 유리하다는 근거를 찾지 못했다.")
    return (f"split_k=3 최적 {win3}개, split_k=6 최적 {win6}개, "
            f"split_k=1 최적 {win1}개, 기타 {other}개 / 전체 {n}개. "
            "3 의 배수가 최적인 경우가 존재하므로 형상별 분해가 필요하다.")


def split_k16_analysis(shapes, sk_best, skmode_best) -> str:
    """waves 가 매우 큰 형상에서 split_k 를 키우는 것이 유리한지 판정."""
    lines = []
    better16 = 0
    for k in shapes:
        row = sk_best.get(k, {})
        if 1 in row and 16 in row and row[16] < row[1]:
            better16 += 1
    lines.append(f"- split_k=16 이 split_k=1 보다 빠른 형상: "
                 f"{better16}/{len(shapes)}")
    modes = Counter()
    for k in shapes:
        row = skmode_best.get(k, {})
        if row:
            modes[min(row, key=row.get)] += 1
    lines.append("- 최적 (split_k, mode) 분포: "
                 + ", ".join(f"`{s}/{m}`×{c}" for (s, m), c in modes.most_common(6)))
    if better16 == 0:
        lines.append("- → **리허설의 관찰은 표본 편향이었다.** 전수에서는 waves 가 큰 "
                     "형상에서 split_k 를 키우는 것이 유리하지 않다.")
    elif better16 == len(shapes):
        lines.append("- → **실제 현상이다.** 모든 대상 형상에서 split_k=16 이 더 빠르다. "
                     "원인 후보: tail effect 완화, L2 재사용 패턴 변화.")
    else:
        lines.append("- → 형상에 따라 갈린다. 위 표의 tile 크기·pipeline_kind 와 "
                     "함께 봐야 한다.")
    return "\n".join(lines)


def fmt_ts(t: float | None) -> str:
    import datetime as _dt
    return "?" if t is None else _dt.datetime.fromtimestamp(
        t, _dt.timezone.utc).strftime("%m-%d %H:%M")


def parse_ts(x: str) -> float | None:
    """ISO(results/drift) 와 nvidia-smi 형식(telemetry) 을 모두 초로."""
    import datetime as _dt
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y/%m/%d %H:%M:%S.%f"):
        try:
            return _dt.datetime.strptime(x.strip(), fmt).replace(
                tzinfo=_dt.timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def telemetry_summary(t0: float | None = None, t1: float | None = None) -> dict | None:
    if not TELEMETRY.exists():
        return None
    clk, mem, tp, pw = [], [], [], []
    thr = Counter()
    n = 0
    for line in TELEMETRY.read_text().splitlines()[1:]:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 6:
            continue
        if t0 is not None:
            ts = parse_ts(parts[0])
            if ts is None or not (t0 - 120 <= ts <= t1 + 120):
                continue
        try:
            clk.append(int(parts[1].split()[0]))
            mem.append(int(parts[2].split()[0]))
            tp.append(int(parts[3]))
            pw.append(float(parts[4].split()[0]))
            bits = int(parts[5], 16)
        except (ValueError, IndexError):
            continue
        n += 1
        for mask, name in THROTTLE_BITS.items():
            if bits & mask:
                thr[name] += 1
    if not n:
        return None
    return {"n": n, "clk_min": min(clk), "clk_max": max(clk),
            "clk_med": int(statistics.median(clk)),
            "mem_min": min(mem), "mem_max": max(mem),
            "mem_med": int(statistics.median(mem)),
            "t_min": min(tp), "t_max": max(tp),
            "p_min": round(min(pw), 1), "p_max": round(max(pw), 1),
            "throttle": dict(thr)}


def make_plots(out: Path, ca, axes, sk_best, best, hw, cfg) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 그림의 라벨은 영어로 쓴다. 기본 폰트(DejaVu Sans)에 한글 글리프가 없고,
    # 컨테이너 base 이미지에도 한글 폰트가 없다. 마크다운 본문이 한글이므로
    # 그림은 축 이름만 알아보면 된다.

    # (1) C/A 히스토그램
    v = list(ca.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(v, bins=30, color="#4C72B0", edgecolor="white")
    ax.axvline(1.0, color="#C44E52", lw=1.5, ls="--", label="cuBLAS parity")
    ax.set_xlabel("C/A = cuBLAS time / best time   (>1 = ours is faster)")
    ax.set_ylabel("number of shapes")
    ax.set_title("Headroom over cuBLAS (upper bound)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "ca_hist.png", dpi=130)
    plt.close(fig)

    # (2) 축별 상대 성능 박스플롯
    for name in ("spill", "pipeline", "swizzle", "warp_k", "split_mode", "stages"):
        g = axes.get(name)
        if not g or len(g) < 2:
            continue
        keys = sorted(g, key=lambda x: -q(g[x], .5))
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(keys)), 4))
        ax.boxplot([g[k] for k in keys], tick_labels=keys, showfliers=False)
        ax.set_ylabel("rel  (1.0 = best for that shape)")
        ax.set_title(f"relative performance by {name}")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out / f"axis_{name}.png", dpi=130)
        plt.close(fig)

    # (3) waves vs C/A 산점도
    xs, ys = [], []
    for key, r in ca.items():
        b = best[key]
        c = cfg(b["kernel_id"])
        if c is None:
            continue
        xs.append(F.waves(Problem(*key), hw, c, RuntimeConfig(1, "serial")))
        ys.append(r)
    if xs:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(xs, ys, s=18, alpha=.7, color="#55A868")
        ax.axhline(1.0, color="#C44E52", lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("waves (split_k=1, tile of best config)")
        ax.set_ylabel("C/A")
        ax.set_title("C/A vs waves")
        fig.tight_layout()
        fig.savefig(out / "waves_vs_ca.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
