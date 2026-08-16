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

from backends import get_backend  # noqa: E402
from build import paths  # noqa: E402
from core import features as F  # noqa: E402
from core.hardware import hardware_from_env  # noqa: E402
from core.types import KernelConfig, Problem, RuntimeConfig  # noqa: E402

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
    """results.jsonl 을 스트리밍한다. 지정한 측정 조건의 줄만 넘긴다."""
    with RESULTS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("env_hash", "").startswith(env_hash):
                yield d


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
    i = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[i]


def dist(xs: list[float]) -> str:
    if not xs:
        return "n=0"
    return (f"n={len(xs)} min={min(xs):.3f} p25={q(xs, .25):.3f} "
            f"med={q(xs, .5):.3f} p75={q(xs, .75):.3f} max={max(xs):.3f}")


# ---------------------------------------------------------------------------
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
            w(f"  - **스로틀 발생**: "
              + ", ".join(f"{k} {v}초 ({100 * v / tel['n']:.1f}%)"
                          for k, v in tel["throttle"].items()))
        else:
            w("  - 스로틀 발생 구간 없음")
    w()

    # ---------------- 3 / 6. cuBLAS 대비 ----------------------------------
    w("## 3. cuBLAS 대비 — 개선 여지의 상한선")
    w()
    w("`C/A` = cuBLAS 시간 / 전수 최적 시간. **1보다 크면 우리 최적이 더 빠르다.**")
    w("이 값이 이 프로젝트가 노릴 수 있는 개선 여지의 상한선이다.")
    w()
    ca: dict[tuple, float] = {}
    for key, b in best.items():
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
        for key, b in best.items():
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
