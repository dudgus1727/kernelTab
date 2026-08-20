#!/usr/bin/env python3
"""베이스라인 2 — 손으로 쓴 규칙의 regret@k.

    python3 scripts/baseline_rule.py                    # rules.handwritten
    python3 scripts/baseline_rule.py rules.other

⛔ **동점 처리에 시간이 들어가면 안 된다.**
   `sorted([(score, time)])` 는 동점일 때 시간으로 정렬해 **정답을 훔쳐본다.**
   실제로 이 하네스의 첫 판에서 그 버그가 있었고, regret 이 낙관적으로
   나왔다. 동점은 반드시 **시간과 무관한 결정론적 키**로 깬다.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kerneltab.core import paths
from kerneltab.core.table import (
    assert_no_answers,
    load_for_ranking,
    load_for_scoring,
)


def geo(v):
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("module", nargs="?", default="rules.handwritten")
    ap.add_argument("--env-hash", default=None)
    a = ap.parse_args()

    env = json.loads(paths.ENV_JSON.read_text())
    eh = (a.env_hash or env["env_hash"])[:16]
    pq = str(paths.RESULTS_DIR / "table.parquet")

    X = load_for_ranking(pq, env_hash=eh).reset_index(drop=True)   # 정답 없음
    y = load_for_scoring(pq, env_hash=eh).reset_index(drop=True)   # 채점용
    assert_no_answers(X)
    assert len(X) == len(y)
    print(f"규칙 입력 {X.shape} (정답 제거), 채점 {y.shape}")

    mod = importlib.import_module(a.module)
    sc = [mod.score(r) for r in X.to_dict("records")]

    t = y["time_ms"].values
    shape = list(zip(X["M"], X["N"], X["K"]))
    best, diff, by = {}, {}, {}
    for i, sh in enumerate(shape):
        if not (t[i] and t[i] > 0):
            continue
        if sh not in best or t[i] < best[sh]:
            best[sh] = t[i]
        d = y["difficulty"].values[i]
        if not math.isnan(d):
            diff[sh] = d
        # ⛔ 동점을 시간으로 깨면 정답 누출이다. 행 인덱스로 깬다.
        by.setdefault(sh, []).append((sc[i], i, t[i]))

    med = statistics.median(diff.values())
    hard = {s for s, d in diff.items() if d >= med}
    print(f"\n{'k':>3} {'전체':>9} {'어려운절반':>11} {'쉬운절반':>10}")
    out = {}
    for k in (1, 3, 5):
        rs = []
        for sh, lst in by.items():
            top = sorted(lst, key=lambda z: (z[0], z[1]))[:k]
            rs.append((sh, min(tt for _, _, tt in top) / best[sh]))
        out[k] = rs
        print(f"{k:3d} {geo([r for _, r in rs]):9.4f} "
              f"{geo([r for s, r in rs if s in hard]):11.4f} "
              f"{geo([r for s, r in rs if s not in hard]):10.4f}")

    print("\n최악 형상 (k=1):")
    for sh, r in sorted(out[1], key=lambda x: -x[1])[:8]:
        print(f"   {sh[0]}x{sh[1]}x{sh[2]}  regret {r:.3f}  난이도 {diff.get(sh, 0):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
