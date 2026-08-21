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
    import pandas as pd
    import pyarrow.parquet as pq

    from kerneltab.core import paths

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("block", "kfold", "transfer"),
                    default="block")
    ap.add_argument("--m-threshold", type=int, default=2048)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--eval-table", default=None,
                    help="`--split transfer` 용. **다른 조건의 표**로 채점한다. "
                         "학습은 기본 표(KERNELTAB_RESULTS_DIR)로 한다")
    ap.add_argument("--eval-env-hash", default=None)
    ap.add_argument("--status", choices=("ok", "all"), default="all",
                    help="측정 줄 필터. **기본 all** — `high_outlier_frac` 은 "
                         "결측이 아니라 품질 표시다 (consumer_contract 9절). "
                         "`ok` 만 남기면 전 형상 덮개가 필요한 지표(정적 "
                         "top-k)가 통째로 왜곡된다: a888 61형상 전부에서 ok 인 "
                         "config 가 3,465개 -> **3개** 로 줄어든다")
    a = ap.parse_args()

    env = json.loads(paths.ENV_JSON.read_text())
    eh = env["env_hash"][:16]
    df = pq.read_table(paths.RESULTS_DIR / "table.parquet").to_pandas()
    df = df[df.env_hash.astype(str).str.startswith(eh[:8])]
    # status-filter: --status 플래그를 해석하는 자리. 기본은 all.
    if a.status == "ok":
        df = df[df.status == "ok"]
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
    if a.split == "transfer":
        # ★ **툴체인 전이** — 한 조건으로 학습하고 **다른 조건의 표**로 채점한다.
        #
        #    kfold/block 은 같은 조건 안에서 형상을 나눈다. 그것은 "안 본
        #    형상에 일반화하는가" 를 잰다. 여기서 묻는 것은 다른 질문이다:
        #    **"표의 유통기한이 있는가."** 툴체인이 바뀌면 12.4 로 만든
        #    모델과 규칙을 다시 봐야 하는가.
        #
        #    학습에 **전 형상**을 쓴다 — 형상 일반화가 아니라 조건 전이를
        #    격리해서 재려는 것이다. 같은 형상·같은 커널을 다른 툴체인에서
        #    다시 잰 표로 채점한다.
        if not a.eval_table:
            print("--split transfer 에는 --eval-table 이 필요하다.")
            return 2
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.06,
                              num_leaves=127, min_child_samples=40,
                              subsample=0.8, colsample_bytree=0.8, verbose=-1)
        m.fit(X, df._y)
        imp = sorted(zip(feat, m.feature_importances_), key=lambda x: -x[1])[:12]

        ev = pq.read_table(Path(a.eval_table)).to_pandas()
        eeh = (a.eval_env_hash or "")[:8]
        if eeh:
            ev = ev[ev.env_hash.astype(str).str.startswith(eeh)]
        ev = ev[ev.time_ms > 0]
        # status-filter: --status 플래그 (평가 표). 기본은 all.
        if a.status == "ok":
            ev = ev[ev.status == "ok"]
        ev = ev.copy()
        ev["_shape"] = list(zip(ev.M, ev.N, ev.K))
        # ⚠️ 피처를 **학습 때와 같은 순서·같은 처리**로 만든다. 컬럼이
        #    하나라도 어긋나면 LightGBM 이 조용히 다른 것을 읽는다.
        miss = [c for c in feat if c not in ev.columns]
        if miss:
            print(f"평가 표에 없는 피처: {miss}")
            return 2
        # ⚠️ **학습 때와 같은 범주 매핑**을 써야 한다. 평가 표에서 새로
        #    `astype("category")` 를 하면 범주 순서가 달라지고, LightGBM 은
        #    같은 코드를 **다른 값**으로 읽는다. 조용히 틀리는 종류다 —
        #    다행히 여기서는 categorical_feature 불일치로 죽는다.
        Xe = ev[feat].copy()
        for c in Xe.columns:
            if str(X[c].dtype) == "category":
                Xe[c] = pd.Categorical(Xe[c], categories=X[c].cat.categories)
                unseen = Xe[c].isna().sum() - ev[c].isna().sum()
                if unseen > 0:
                    print(f"  [주의] {c}: 학습에 없던 값 {unseen:,}행 -> 결측")
            elif Xe[c].dtype == bool:
                Xe[c] = Xe[c].astype(int)
        ev["_p"] = m.predict(Xe)
        print(f"전이: 학습 {len(df):,}행 ({eh[:8]}) -> "
              f"평가 {len(ev):,}행 ({eeh or '전체'}), 형상 "
              f"{ev._shape.nunique()}개")
        for sh, g in ev.groupby("_shape"):
            per_shape[sh] = {k: g.nsmallest(k, "_p").time_ms.min() / g.time_ms.min()
                             for k in (1, 3, 5)}
    elif a.split == "block":
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

    dpath = Path(a.eval_table) if a.eval_table else paths.RESULTS_DIR / "table.parquet"
    deh = (a.eval_env_hash or eh)[:8] if a.eval_table else eh[:8]
    dm = pq.read_table(dpath,
                       columns=["env_hash", "M", "N", "K", "difficulty"]).to_pandas()
    dm = dm[dm.env_hash.astype(str).str.startswith(deh)].dropna(subset=["difficulty"])
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
