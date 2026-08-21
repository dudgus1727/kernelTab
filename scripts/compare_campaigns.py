#!/usr/bin/env python3
"""두 캠페인 표를 대조한다 — 툴체인이 바뀌면 무엇이 달라지는가.

    python3 scripts/compare_campaigns.py \\
        --a results/table.parquet --a-env c63710df \\
        --b <다른 표> --b-env 828baa64

## 무엇을 묻는가

같은 GPU, 같은 형상, 같은 커널을 **다른 툴체인**에서 다시 쟀다. 표가
얼마나 달라지는가 — 즉 **표에 유통기한이 있는가.**

| 절 | 무엇 | 왜 중요한가 |
|---|---|---|
| 1 | 절대 성능 변화 | 맥락. 빨라졌나 느려졌나 |
| 2 | **순위 안정성** | 절대 시간이 변해도 순위가 남으면 규칙이 살아남는다 |
| 3 | 축별 결론 | "이 축은 답이 정해져 있다" 가 유지되는가 |

⚠️ **읽기 전용이다.** 어떤 파일도 쓰지 않는다 (`--out` 을 준 경우 제외).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kerneltab.core import noise

#: 순위 비교에 쓸 상위 개수.
TOP_JACCARD = 20
TOP_TAU = 50

#: 축별 결론을 확인할 컬럼과, 12.4 에서 "최적 0회" 였던 값.
AXES = [
    ("ext_warp_m", 128),
    ("split_k_mode", "parallel"),
    ("ext_stages", 8),
]

KEY = ["kernel_id", "M", "N", "K", "split_k", "split_k_mode"]


def _load(path, env):
    import pyarrow.parquet as pq
    cols = ["env_hash", "status", "time_ms", "difficulty", *KEY,
            "ext_warp_m", "ext_warp_n", "ext_stages", "distinct_time_frac"]
    sc = set(pq.read_schema(path).names)
    df = pq.read_table(path, columns=[c for c in cols if c in sc]).to_pandas()
    if env:
        df = df[df.env_hash.astype(str).str.startswith(env)]
    df = df[(df.status == "ok") & df.time_ms.notna() & (df.time_ms > 0)].copy()
    df["_shape"] = list(zip(df.M, df.N, df.K))
    return df


def kendall_tau_b(xs, ys) -> float:
    """순위 상관. `scipy` 없이 O(n^2) — 상위 50개라 충분하다."""
    n = len(xs)
    if n < 2:
        return float("nan")
    con = dis = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = xs[i] - xs[j], ys[i] - ys[j]
            if a == 0 and b == 0:
                tx += 1
                ty += 1
            elif a == 0:
                tx += 1
            elif b == 0:
                ty += 1
            elif (a > 0) == (b > 0):
                con += 1
            else:
                dis += 1
    d = math.sqrt((con + dis + tx) * (con + dis + ty))
    return (con - dis) / d if d else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", required=True)
    ap.add_argument("--a-env", default=None)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b", required=True)
    ap.add_argument("--b-env", default=None)
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--out", default=None)
    x = ap.parse_args()

    A, B = _load(x.a, x.a_env), _load(x.b, x.b_env)
    print(f"{x.a_name}: {len(A):,}행 {A._shape.nunique()}형상   "
          f"{x.b_name}: {len(B):,}행 {B._shape.nunique()}형상")

    ka = A.set_index(KEY)
    kb = B.set_index(KEY)
    common = ka.index.intersection(kb.index)
    print(f"공통 (형상,커널,런타임) 조합 {len(common):,}  "
          f"({x.a_name} 전용 {len(ka.index.difference(kb.index)):,}, "
          f"{x.b_name} 전용 {len(kb.index.difference(ka.index)):,})")
    ta = ka.loc[common, "time_ms"]
    tb = kb.loc[common, "time_ms"]

    # --- 1. 절대 성능 -------------------------------------------------------
    rel = (tb / ta - 1) * 100
    print(f"\n## 1. 절대 성능 ({x.a_name} -> {x.b_name})")
    q = rel.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    print(f"  조합별 변화  중앙 {rel.median():+.2f}%   "
          f"p1 {q[0.01]:+.1f}%  p25 {q[0.25]:+.2f}%  "
          f"p75 {q[0.75]:+.2f}%  p99 {q[0.99]:+.1f}%")
    best_a = A.groupby("_shape").time_ms.min()
    best_b = B.groupby("_shape").time_ms.min()
    sh = sorted(set(best_a.index) & set(best_b.index))
    bt = [(best_b[s] / best_a[s] - 1) * 100 for s in sh]
    print(f"  형상별 **최적 시간** 변화  중앙 {st.median(bt):+.2f}%  "
            f"최소 {min(bt):+.2f}%  최대 {max(bt):+.2f}%  "
            f"(개선 {sum(1 for v in bt if v < 0)}/{len(bt)})")

    # --- 2. 순위 안정성 -----------------------------------------------------
    print(f"\n## 2. 순위 안정성 (형상 {len(sh)}개)")
    jac, taus, same_best, res_best = [], [], 0, 0
    for s in sh:
        ga = A[A._shape == s].nsmallest(TOP_TAU, "time_ms")
        gb = B[B._shape == s].nsmallest(TOP_TAU, "time_ms")
        sa = {tuple(r) for r in ga[KEY].head(TOP_JACCARD).to_numpy()}
        sb = {tuple(r) for r in gb[KEY].head(TOP_JACCARD).to_numpy()}
        jac.append(len(sa & sb) / len(sa | sb) if sa | sb else float("nan"))

        # tau 는 **양쪽 상위 50 의 합집합** 중 둘 다에 있는 조합으로 잰다.
        ia = ga.set_index(KEY).time_ms
        ib = gb.set_index(KEY).time_ms
        both = ia.index.intersection(ib.index)
        if len(both) >= 5:
            taus.append(kendall_tau_b(list(ia.loc[both]), list(ib.loc[both])))

        ba = tuple(A.loc[A[A._shape == s].time_ms.idxmin(), KEY])
        bb = tuple(B.loc[B[B._shape == s].time_ms.idxmin(), KEY])
        if ba == bb:
            same_best += 1
        # 최적이 바뀌었어도 **분해 가능한 차이인가**
        tb_of_a = ib.get(ba)
        if tb_of_a is not None:
            if (tb_of_a / best_b[s] - 1) <= 2 * noise.noise_floor(best_b[s]):
                res_best += 1
    print(f"  상위 {TOP_JACCARD} Jaccard   중앙 {st.median(jac):.3f}  "
          f"평균 {st.fmean(jac):.3f}  최소 {min(jac):.3f}")
    print(f"  Kendall tau (상위 {TOP_TAU} 공통)  중앙 {st.median(taus):.3f}  "
          f"평균 {st.fmean(taus):.3f}  최소 {min(taus):.3f}  (n={len(taus)})")
    print(f"  최적 config 가 **그대로**       {same_best}/{len(sh)} "
          f"({100 * same_best / len(sh):.0f}%)")
    print(f"  옛 최적이 새 표에서 **노이즈 이내**  {res_best}/{len(sh)} "
          f"({100 * res_best / len(sh):.0f}%)   <- 실질적으로 같은 선택")

    # --- 3. 축별 결론 -------------------------------------------------------
    print("\n## 3. 축별 결론 — 12.4 에서 '최적 0회' 였던 값이 유지되는가")
    print(f"{'축':>16} {'값':>10} {x.a_name + ' 최적':>10} {x.b_name + ' 최적':>10}  판정")
    axis_rows = []
    for col, val in AXES:
        if col not in A.columns or col not in B.columns:
            continue
        oa = A.loc[A.groupby("_shape").time_ms.idxmin()]
        ob = B.loc[B.groupby("_shape").time_ms.idxmin()]
        ca = int((oa[col] == val).sum())
        cb = int((ob[col] == val).sum())
        ok = (ca == 0) == (cb == 0)
        axis_rows.append({"axis": col, "value": val, "a": ca, "b": cb, "same": ok})
        print(f"{col:>16} {val!s:>10} {ca:10d} {cb:10d}  "
              + ("유지" if ok else "**뒤집힘**"))

    out = {
        "a": {"name": x.a_name, "path": x.a, "env": x.a_env, "rows": len(A)},
        "b": {"name": x.b_name, "path": x.b, "env": x.b_env, "rows": len(B)},
        "n_common_combos": len(common), "n_shapes": len(sh),
        "rel_median_pct": float(rel.median()),
        "best_time_change_median_pct": st.median(bt),
        "jaccard_top20": {"median": st.median(jac), "min": min(jac)},
        "kendall_tau_top50": {"median": st.median(taus), "min": min(taus)},
        "same_best_frac": same_best / len(sh),
        "best_within_noise_frac": res_best / len(sh),
        "axes": axis_rows,
    }
    if x.out:
        Path(x.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\n-> {x.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
