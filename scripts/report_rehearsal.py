#!/usr/bin/env python3
"""리허설 결과 분석. 측정과 분리되어 있어 원본만 있으면 몇 번이든 다시 돌린다.

    python3 scripts/report_rehearsal.py

다루는 것:
  A. actual_split_k (실측 grid.z) vs effective_split_k() 예측
  E. 1024스레드 커널의 성능 분포
  J. 스필 커널 vs 비스필 커널의 성능 분포
  + status 분포, cuBLAS 대비, split-K 가설, 드리프트, 텔레메트리
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backends import get_backend  # noqa: E402
from build import paths  # noqa: E402
from core.types import Hardware, Problem, RuntimeConfig  # noqa: E402

RESULTS = paths.RESULTS_DIR / "results.jsonl"
KERNELS = paths.RESULTS_DIR / "kernels.jsonl"
DRIFT = paths.RESULTS_DIR / "drift.jsonl"


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def hr(t: str) -> None:
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def quart(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    s = sorted(xs)
    n = len(s)
    return (f"min={s[0]:.4f} p25={s[n // 4]:.4f} med={s[n // 2]:.4f} "
            f"p75={s[3 * n // 4]:.4f} max={s[-1]:.4f}")


def main() -> int:
    env = json.loads(paths.ENV_JSON.read_text())
    hw = Hardware(**env["hardware"])
    backend = get_backend(hw.arch)

    res = load(RESULTS)
    kern = {r["kernel_id"]: r for r in load(KERNELS)}
    meas = [r for r in res if r["kernel_id"] != "cublas"]
    cublas = {(r["problem"]["M"], r["problem"]["N"], r["problem"]["K"]): r
              for r in res if r["kernel_id"] == "cublas"}
    ok = [r for r in meas if r.get("status") == "ok"]

    print(f"측정 {len(meas)}줄 (+ cuBLAS {len(cublas)}줄), 커널 {len(kern)}개")
    hr("status 분포")
    for k, v in Counter(r.get("status") for r in meas).most_common():
        print(f"  {str(k):24s} {v:6d}  ({100 * v / len(meas):5.1f}%)")

    # ---- A. actual_split_k 검증 -----------------------------------------
    hr("A. actual_split_k (실측 grid.z) vs effective_split_k() 예측")
    bad = []
    for r in meas:
        a = r.get("actual_split_k")
        if a is None:
            continue
        p = Problem(r["problem"]["M"], r["problem"]["N"], r["problem"]["K"])
        rc = RuntimeConfig(r["runtime"]["split_k"], r["runtime"]["split_k_mode"])
        pred = backend.effective_split_k(p, rc)
        if a != pred or a != rc.split_k:
            bad.append((r["kernel_id"], (p.M, p.N, p.K), rc.split_k, pred, a))
    n_have = sum(1 for r in meas if r.get("actual_split_k") is not None)
    print(f"  대조 가능한 줄: {n_have}")
    print(f"  불일치: {len(bad)}")
    for b in bad[:10]:
        print(f"    {b[1]} 요청={b[2]} 예측={b[3]} 실측={b[4]}  {b[0][:44]}")
    if not bad and n_have:
        print("  -> 요청 split_k == effective_split_k() == 실제 grid.z, 전부 일치")

    # split_k 값별 등장 (3,6,12 가 실제로 측정됐는지)
    print("\n  split_k 값별 측정 수:")
    for k, v in sorted(Counter(
            (r["runtime"]["split_k"], r["runtime"]["split_k_mode"])
            for r in meas).items()):
        print(f"    split_k={k[0]:2d} {k[1]:8s} {v:5d}")

    # ---- 정확도 ----------------------------------------------------------
    hr("정확도 (max_rel_error), split-K 모드별")
    grp = defaultdict(list)
    for r in ok:
        e = r.get("max_rel_error")
        if e is None:
            continue
        sk, mode = r["runtime"]["split_k"], r["runtime"]["split_k_mode"]
        grp[(mode if sk > 1 else "none", sk)].append(e)
    for k in sorted(grp, key=lambda x: (x[0], x[1])):
        v = sorted(grp[k])
        print(f"  {k[0]:8s} split_k={k[1]:2d}  n={len(v):4d}  "
              f"median={v[len(v) // 2]:.3e}  max={v[-1]:.3e}")

    # ---- E. 1024 스레드 그룹 ---------------------------------------------
    hr("E. threads_per_block 별 성능 분포 (형상별 최고 성능 대비 비율)")
    perf_by_shape = defaultdict(float)
    for r in ok:
        key = (r["problem"]["M"], r["problem"]["N"], r["problem"]["K"])
        perf_by_shape[key] = max(perf_by_shape[key], 1.0 / r["time_ms"])

    def rel_perf(r):
        key = (r["problem"]["M"], r["problem"]["N"], r["problem"]["K"])
        best = perf_by_shape[key]
        return (1.0 / r["time_ms"]) / best if best else 0.0

    by_thr = defaultdict(list)
    for r in ok:
        k = kern.get(r["kernel_id"])
        if k:
            by_thr[k.get("threads")].append(rel_perf(r))
    print(f"  {'threads':>8s} {'n':>6s} {'median':>8s} {'p90':>8s} "
          f"{'max':>8s}  (1.0 = 해당 형상 최고)")
    for t in sorted(by_thr, key=lambda x: (x is None, x)):
        v = sorted(by_thr[t])
        n = len(v)
        print(f"  {str(t):>8s} {n:6d} {v[n // 2]:8.3f} {v[int(n * 0.9)]:8.3f} "
              f"{v[-1]:8.3f}")

    # ---- J. 스필 vs 비스필 -----------------------------------------------
    hr("J. 스필 커널 vs 비스필 커널 성능 분포")
    spill, nospill = [], []
    for r in ok:
        k = kern.get(r["kernel_id"])
        if not k:
            continue
        (spill if (k.get("spill_stores") or 0) + (k.get("spill_loads") or 0) > 0
         else nospill).append(rel_perf(r))
    for label, v in (("스필 있음", spill), ("스필 없음", nospill)):
        if not v:
            print(f"  {label}: 표본 없음")
            continue
        s = sorted(v)
        n = len(s)
        print(f"  {label}: n={n:5d}  median={s[n // 2]:.3f}  "
              f"p90={s[int(n * 0.9)]:.3f}  max={s[-1]:.3f}")
    # warp tile 별
    print("\n  warp tile 별 상대 성능 (중앙값):")
    by_warp = defaultdict(list)
    for r in ok:
        k = kern.get(r["kernel_id"])
        if k:
            by_warp[(k["ext"]["warp_m"], k["ext"]["warp_n"])].append(rel_perf(r))
    for w in sorted(by_warp, key=str):
        v = sorted(by_warp[w])
        sp = any((kern[r["kernel_id"]].get("spill_stores") or 0) > 0
                 for r in ok
                 if kern.get(r["kernel_id"])
                 and (kern[r["kernel_id"]]["ext"]["warp_m"],
                      kern[r["kernel_id"]]["ext"]["warp_n"]) == w)
        print(f"    {str(w):12s} n={len(v):5d} median={v[len(v) // 2]:.3f} "
              f"max={v[-1]:.3f}" + ("   <- 스필" if sp else ""))

    # ---- K. pipeline_kind ------------------------------------------------
    hr("K. pipeline_kind (pipelined=2단 vs multistage=3단 이상)")
    by_pk = defaultdict(list)
    for r in ok:
        k = kern.get(r["kernel_id"])
        if k:
            by_pk[k.get("pipeline_kind")].append(rel_perf(r))
    for pk in sorted(by_pk, key=str):
        v = sorted(by_pk[pk])
        n = len(v)
        print(f"  {str(pk):12s} n={n:5d} median={v[n // 2]:.3f} "
              f"p90={v[int(n * 0.9)]:.3f} max={v[-1]:.3f}")

    # ---- 형상별 최고 성능 vs cuBLAS --------------------------------------
    hr("형상별 최고 커널 vs cuBLAS")
    best_of = {}
    for r in ok:
        key = (r["problem"]["M"], r["problem"]["N"], r["problem"]["K"])
        if key not in best_of or r["time_ms"] < best_of[key]["time_ms"]:
            best_of[key] = r
    print(f"  {'shape':24s} {'best ms':>9s} {'cuBLAS ms':>10s} {'ratio':>7s}  config")
    for key in sorted(best_of):
        b = best_of[key]
        cb = cublas.get(key, {}).get("time_ms")
        k = kern.get(b["kernel_id"], {})
        e = k.get("ext", {})
        cfg = (f"tb{k.get('tile', {}).get('m')}x{k.get('tile', {}).get('n')}"
               f"x{k.get('tile', {}).get('k')} st{e.get('stages')} "
               f"sw{e.get('swizzle_type', '')[:2]}{e.get('swizzle_n')} "
               f"sk{b['runtime']['split_k']}{b['runtime']['split_k_mode'][:4]}")
        ratio = f"{cb / b['time_ms']:.3f}" if cb else "n/a"
        print(f"  {str(key):24s} {b['time_ms']:9.4f} "
              f"{cb if cb else 0:10.4f} {ratio:>7s}  {cfg}")

    # ---- split-K 가설 ----------------------------------------------------
    hr("split-K 값별 최고 성능 (84 = 2^2 x 3 x 7 가설)")
    for key in sorted(best_of):
        rows = [r for r in ok
                if (r["problem"]["M"], r["problem"]["N"], r["problem"]["K"]) == key]
        by_sk = defaultdict(list)
        for r in rows:
            by_sk[r["runtime"]["split_k"]].append(r["time_ms"])
        if len(by_sk) < 2:
            continue
        best = min(min(v) for v in by_sk.values())
        parts = " ".join(
            f"{sk}:{best / min(v):.3f}" for sk, v in sorted(by_sk.items()))
        print(f"  {str(key):24s} {parts}")
    print("  (값 = 해당 split_k 의 최고 성능 / 전체 최고 성능, 1.0 이 최고)")

    # ---- 드리프트 --------------------------------------------------------
    hr("드리프트")
    ds = load(DRIFT)
    if ds:
        ts = [d["time_ms"] for d in ds]
        mean = statistics.mean(ts)
        print(f"  {len(ds)}회  min={min(ts):.4f} max={max(ts):.4f} "
              f"mean={mean:.4f} std={statistics.pstdev(ts):.5f} ms")
        print(f"  변동폭 (max-min)/mean = {100 * (max(ts) - min(ts)) / mean:.2f}%"
              + ("   !! 5% 초과" if (max(ts) - min(ts)) / mean > 0.05 else ""))
        cl = [d.get("sm_clock_mhz") for d in ds if d.get("sm_clock_mhz")]
        tp = [d.get("gpu_temp_c") for d in ds if d.get("gpu_temp_c")]
        if cl:
            print(f"  클럭 {min(cl)}~{max(cl)} MHz, 온도 {min(tp)}~{max(tp)}°C")
    else:
        print("  기록 없음")

    # ---- 텔레메트리 ------------------------------------------------------
    hr("텔레메트리 / 스로틀링")
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from rehearse import analyze_telemetry  # noqa: E402

    tel = analyze_telemetry()
    if tel:
        for k, v in tel.items():
            print(f"  {k}: {v}")
        if not tel.get("throttle_seconds"):
            print("  -> 스로틀링 구간 없음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
