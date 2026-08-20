#!/usr/bin/env python3
"""베이스라인 3 — GBDT 랭커의 regret@k. 학습 모델의 상한.

lightgbm/scikit-learn 은 별도 의존성이므로 격리된 venv 에서 돌린다.

    python3 -m venv /tmp/gbdt
    /tmp/gbdt/bin/pip install lightgbm scikit-learn pandas pyarrow
    /tmp/gbdt/bin/python scripts/baseline_gbdt.py --split block
    /tmp/gbdt/bin/python scripts/baseline_gbdt.py --split kfold

**측정 시간과 그로부터 유도된 값은 피처에 넣지 않는다.** 목표값은
형상 안의 상대 시간 `log(t / t_best)` 이므로 형상 사이 절대 시간은 안 쓴다.

분할 두 가지:
  block  M 구간 블록 분할. **주 지표.** 형상 일반화를 실제로 시험한다.
  kfold  형상 단위 5-fold. **낙관적 상한.** 형상을 무작위로 나누면
         M=1024 가 학습에, M=1000 이 검증에 들어가는 사실상 보간이 된다.
두 값의 격차가 "형상 일반화가 얼마나 어려운가" 를 보여준다.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

#: 정답과 그로부터 유도된 값. core/table.py 의 ANSWER_COLS 와 같아야 한다.
ANSWER = {"time_ms", "time_std_ms", "time_min_ms", "time_max_ms", "n_reps",
          "outlier_frac", "cublas_ms", "tflops", "frac_of_peak", "vs_cublas",
          "difficulty"}
#: 측정을 해봐야 아는 값.
OUTCOME = {"status", "error", "max_rel_error", "actual_split_k", "sm_clock_mhz",
           "mem_clock_mhz", "gpu_temp_c", "power_w", "soak_elapsed_s",
           "drift_ratio", "timestamp", "env_hash", "clock_locked", "kernel_id"}


def geo(v):
    import numpy as np
    return math.exp(np.mean(np.log(v))) if len(v) else float("nan")


def main() -> int:
    import lightgbm as lgb
    import numpy as np
    import pyarrow.parquet as pq

    from kerneltab.build import paths

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("block", "kfold"), default="block")
    ap.add_argument("--m-threshold", type=int, default=2048)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    env = json.loads(paths.ENV_JSON.read_text())
    eh = env["env_hash"][:16]
    df = pq.read_table(paths.RESULTS_DIR / "table.parquet").to_pandas()
    df = df[df.env_hash.astype(str).str.startswith(eh[:8]) & (df.status == "ok")]
    df = df[df.time_ms > 0].copy()
    df["_shape"] = list(zip(df.M, df.N, df.K))
    df["_y"] = np.log(df.time_ms / df.groupby("_shape").time_ms.transform("min"))

    feat = [c for c in df.columns
            if c not in ANSWER and c not in OUTCOME and not c.startswith("_")]
    X = df[feat].copy()
    drop = []
    for c in list(X.columns):
        d = str(X[c].dtype)
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
        elif d.startswith(("object", "string", "str")):
            if X[c].nunique(dropna=False) <= 1:
                drop.append(c)          # 상수 컬럼 (dtype, arch 등)
            else:
                X[c] = X[c].astype("category")
        elif not d.startswith(("int", "float", "uint")):
            drop.append(c)
    X = X.drop(columns=drop)
    feat = [f for f in feat if f not in drop]
    print(f"행 {len(df):,}  피처 {len(feat)}개 (정답/결과 제외, 버림 {len(drop)})")

    def fit_predict(train_mask):
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.06,
                              num_leaves=127, min_child_samples=40,
                              subsample=0.8, colsample_bytree=0.8, verbose=-1)
        m.fit(X[train_mask], df._y[train_mask])
        return m, m.predict(X[~train_mask])

    per_shape = {}
    if a.split == "block":
        te = df.M > a.m_threshold
        print(f"블록 분할 M>{a.m_threshold}: 학습 {(~te).sum():,} / "
              f"검증 {te.sum():,} (검증 형상 {df[te]._shape.nunique()}개)")
        m, p = fit_predict(~te)
        d = df[te].copy()
        d["_p"] = p
        for sh, g in d.groupby("_shape"):
            per_shape[sh] = {k: g.nsmallest(k, "_p").time_ms.min() / g.time_ms.min()
                             for k in (1, 3, 5)}
        imp = sorted(zip(feat, m.feature_importances_), key=lambda x: -x[1])[:12]
    else:
        shapes = sorted(df._shape.unique())
        order = np.random.RandomState(0).permutation(len(shapes))
        fold = {shapes[order[i]]: i % a.folds for i in range(len(shapes))}
        df["_fold"] = df._shape.map(fold)
        imp = None
        for f_ in range(a.folds):
            te = df._fold == f_
            m, p = fit_predict(~te)
            d = df[te].copy()
            d["_p"] = p
            for sh, g in d.groupby("_shape"):
                per_shape[sh] = {k: g.nsmallest(k, "_p").time_ms.min() / g.time_ms.min()
                                 for k in (1, 3, 5)}
            if imp is None:
                imp = sorted(zip(feat, m.feature_importances_), key=lambda x: -x[1])[:12]
            print(f"  fold {f_}: 검증 형상 {d._shape.nunique()}개")

    dm = pq.read_table(paths.RESULTS_DIR / "table.parquet",
                       columns=["env_hash", "M", "N", "K", "difficulty"]).to_pandas()
    dm = dm[dm.env_hash.astype(str).str.startswith(eh[:8])].dropna(subset=["difficulty"])
    dmap = dm.groupby(["M", "N", "K"]).difficulty.first().to_dict()
    med = np.median(list(dmap.values()))
    hard = {s for s in per_shape if dmap.get(s, 0) >= med}

    print(f"\n[{a.split}]  홀드아웃 형상 {len(per_shape)}개")
    print(f"{'k':>3} {'전체':>9} {'어려운절반':>11} {'쉬운절반':>10}")
    for k in (1, 3, 5):
        print(f"{k:3d} {geo([v[k] for v in per_shape.values()]):9.4f} "
              f"{geo([v[k] for s, v in per_shape.items() if s in hard]):11.4f} "
              f"{geo([v[k] for s, v in per_shape.items() if s not in hard]):10.4f}")

    print("\n피처 중요도 상위 12 (LLM 에게 줄 힌트 — 어떤 물리량이 설명력을 갖는가):")
    for f_, v in imp:
        print(f"   {f_:26s} {v}")
    print("\n최악 형상 (k=1):")
    for s, v in sorted(per_shape.items(), key=lambda x: -x[1][1])[:8]:
        print(f"   {s[0]}x{s[1]}x{s[2]}  regret {v[1]:.3f}  난이도 {dmap.get(s, 0):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
