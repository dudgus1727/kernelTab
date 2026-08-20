#!/usr/bin/env python3
"""표 무결성 검사 — 규칙 학습에 들어가기 전의 관문.

전수 측정이 끝난 뒤 "이 표를 믿어도 되는가" 를 판정한다. 하나라도 실패하면
비영(non-zero) 종료한다. 통과하지 못한 표로 규칙을 학습하면 학습이 무엇을
배웠는지 알 수 없게 된다.

    python3 scripts/validate_table.py                     # 현재 env.json 조건
    python3 scripts/validate_table.py --env-hash b42df475
    python3 scripts/validate_table.py --list-missing 50   # 빠진 조합 목록

종료 코드
    0  통과
    2  전제 조건 미충족 (파일 없음 등)
    4  검증 실패
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import itertools

from backends import get_backend
from build import paths
from core import kernels as kernels_mod
from core.config import (
    alignment_combos,
    alignments_for,
    enumerate_kernels,
    enumerate_runtimes,
)
from core.hardware import hardware_from_env
from core.shapes import all_shapes
from core.types import KernelConfig

RESULTS = paths.RESULTS_DIR / "results.jsonl"
KERNELS = paths.RESULTS_DIR / "kernels.jsonl"

#: SOL(speed of light) 여유. 측정 오차와 실효 피크 추정 오차를 감안해
#: 이 비율만큼은 SOL 보다 빨라도 넘어간다. 그 이상이면 물리적으로 불가능하다.
SOL_TOLERANCE = 0.97

#: max_rel_error 상한 (rehearse.NUMERICAL_TOL 과 같아야 한다)
NUMERICAL_TOL = 5e-2


class Check:
    def __init__(self) -> None:
        self.fails: list[str] = []
        self.warns: list[str] = []

    def fail(self, msg: str) -> None:
        self.fails.append(msg)
        print(f"  [FAIL] {msg}")

    def warn(self, msg: str) -> None:
        self.warns.append(msg)
        print(f"  [warn] {msg}")

    def ok(self, msg: str) -> None:
        print(f"  [ok]   {msg}")


def hr(t: str) -> None:
    print()
    print("=" * 74)
    print(t)
    print("=" * 74)


def validate_bundle(bundle_dir: Path) -> int:
    """번들 무결성 — **배포 경계 검사** (`--bundle`).

    `validate_table.py` 본체는 `results/` 를 본다. 이건 **배포되는 것**을 본다.
    둘은 다르다. 실제로 표는 통과했는데 번들에 폐기 데이터가 실렸다.

    검사:
      1. `table.parquet` 의 `env_hash` 가 단일인가
      2. `BUNDLE.json` 의 `n_rows` 와 **실제 행 수**가 일치하는가  <- 핵심
      3. 각 파일의 sha256 이 `BUNDLE.json` 과 맞는가
      4. 해석에 필요한 파일이 다 있는가

    2번이 이번 버그를 잡는 검사다. `n_rows` 는 필터한 값이었고 파일은
    안 걸렀기 때문에 어긋났는데, 그 불일치를 봤으면 바로 잡혔다.
    """
    import hashlib

    import pyarrow.parquet as pq

    d = Path(bundle_dir)
    print(f"번들 검사 — {d}")
    bj = d / "BUNDLE.json"
    if not bj.exists():
        print("  BUNDLE.json 이 없다")
        return 2
    info = json.loads(bj.read_text())
    eh = str(info.get("env_hash") or "")
    fails = []

    for req in ("table.parquet", "env.json", "kernels.jsonl", "manifest.json"):
        if not (d / req).exists():
            fails.append(f"{req} 이 없다 (이 파일 없이는 표를 해석할 수 없다)")

    tp = d / "table.parquet"
    if tp.exists():
        t = pq.read_table(tp, columns=["env_hash"])
        seen: dict[str, int] = {}
        for x in t.column("env_hash").to_pylist():
            k = str(x)[:8]
            seen[k] = seen.get(k, 0) + 1
        print(f"  table.parquet {t.num_rows:,}행, env_hash {len(seen)}종")
        for k, v in sorted(seen.items(), key=lambda kv: -kv[1]):
            mark = "" if eh.startswith(k) else "   <- 다른 조건"
            print(f"    {k}  {v:>9,}행{mark}")
        if len(seen) != 1 or not any(eh.startswith(k) for k in seen):
            fails.append(f"env_hash 가 단일이 아니다 ({len(seen)}종). "
                         "배포되는 표는 단일 조건이어야 한다")
        # ★ 이번 버그를 잡는 검사
        want = info.get("n_rows")
        if want is not None and want != t.num_rows:
            fails.append(f"BUNDLE.json 의 n_rows({want:,}) 와 실제 행 수"
                         f"({t.num_rows:,}) 가 다르다 — 통계만 필터하고 "
                         "파일은 안 걸렀다는 뜻이다")

    for name, meta in (info.get("files") or {}).items():
        f = d / name
        if not f.exists():
            fails.append(f"{name}: 파일 없음")
            continue
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        if h != meta.get("sha256"):
            fails.append(f"{name}: sha256 불일치")

    print()
    if fails:
        print(f"!! 실패 {len(fails)}건")
        for f_ in fails:
            print(f"   - {f_}")
        return 5
    print("통과: 단일 조건, 행 수 일치, 체크섬 일치")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", metavar="DIR",
                    help="번들 디렉토리를 검사한다 (배포 경계 검사)")
    ap.add_argument("--env-hash", default=None)
    ap.add_argument("--list-missing", type=int, default=20)
    ap.add_argument("--list-bad", type=int, default=20)
    ap.add_argument("--expect", choices=("full", "subset"), default="full",
                    help="full: 전체 열거 대비 누락/커버리지를 본다 (Phase 3 관문). "
                         "subset: 실제로 측정된 형상 안에서만 본다 (리허설 등 "
                         "부분 측정 데이터 점검용)")
    args = ap.parse_args()

    if args.bundle:
        return validate_bundle(Path(args.bundle))

    if not RESULTS.exists() or not KERNELS.exists():
        print("results.jsonl 또는 kernels.jsonl 이 없다.")
        return 2

    env = json.loads(paths.ENV_JSON.read_text())
    hw = hardware_from_env(env)
    backend = get_backend(hw.arch)
    want_hash = args.env_hash or env["env_hash"]
    c = Check()

    print(f"표 무결성 검사  env_hash = {want_hash[:16]}")
    print(f"  GPU {hw.name}, 실효 피크 {hw.peak_tflops_f16} TFLOP/s, "
          f"실효 대역폭 {hw.bandwidth_gbps} GB/s")

    kern = {r["kernel_id"]: r
            for r in (json.loads(l) for l in KERNELS.read_text().splitlines() if l.strip())}

    # ---------------- 스트리밍 1 회차 -------------------------------------
    seen: Counter = Counter()               # (kid, M,N,K, sk, mode) -> 횟수
    hashes: Counter = Counter()
    status: Counter = Counter()
    unknown_kid: Counter = Counter()
    sol_bad: list[tuple] = []
    err_by_sk: dict[tuple, list[float]] = defaultdict(list)
    ok_per_shape: Counter = Counter()
    n_lines = n_sel = n_cublas = 0

    peak_flops = hw.peak_tflops_f16 * 1e12
    # ⚠️ 여기는 **의도적으로** 전체를 읽는다. 이 검사기의 일 중 하나가
    #    "파일에 조건이 몇 종 섞여 있는가" 를 보고하는 것이라, 필터하면
    #    그 검사가 불가능해진다. 선택된 줄만 세는 것은 아래 h/env_hash 비교로
    #    한다. (R-5 의 예외 — core.records.ALL 과 같은 취지)
    with RESULTS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                c.fail(f"results.jsonl {n_lines} 번째 줄 파싱 실패")
                continue
            h = d.get("env_hash", "")
            hashes[h[:16]] += 1
            if not h.startswith(want_hash[:16]):
                continue
            n_sel += 1
            p, rt = d["problem"], d["runtime"]
            key = (p["M"], p["N"], p["K"])
            if d["kernel_id"] == "cublas":
                n_cublas += 1
                continue
            seen[(d["kernel_id"], *key, rt["split_k"], rt["split_k_mode"])] += 1
            st = d.get("status")
            status[st] += 1
            if d["kernel_id"] not in kern:
                unknown_kid[d["kernel_id"]] += 1
            if st != "ok":
                continue
            ok_per_shape[key] += 1
            t = d.get("time_ms")
            if t:
                sol_ms = 2.0 * p["M"] * p["N"] * p["K"] / peak_flops * 1e3
                if t < sol_ms * SOL_TOLERANCE:
                    sol_bad.append((key, d["kernel_id"], t, sol_ms))
            e = d.get("max_rel_error")
            if e is not None:
                mode = rt["split_k_mode"] if rt["split_k"] > 1 else "none"
                err_by_sk[(mode, rt["split_k"])].append(e)

    # ---------------- 1. env_hash 일관성 ---------------------------------
    hr("1. env_hash 일관성")
    print(f"  파일 전체 {n_lines:,} 줄, 이 조건 {n_sel:,} 줄 "
          f"(측정 {sum(status.values()):,} + cuBLAS {n_cublas:,})")
    print("  파일에 존재하는 env_hash:")
    for h, v in hashes.most_common():
        mark = "  <- 검사 대상" if want_hash.startswith(h) else ""
        print(f"    {h}  {v:,} 줄{mark}")
    if n_sel == 0:
        c.fail(f"env_hash {want_hash[:16]} 인 줄이 하나도 없다")
        return finish(c)
    if len(hashes) > 1:
        c.warn(f"파일에 측정 조건이 {len(hashes)} 종 섞여 있다. "
               "조인/분석 시 반드시 env_hash 로 나눌 것 "
               "(설계상 정상이지만 소비하는 쪽이 실수하기 쉽다)")
    else:
        c.ok("측정 조건이 하나뿐이다")

    # ---------------- 2. 중복 ---------------------------------------------
    hr("2. 중복 측정")
    dup = [(k, v) for k, v in seen.items() if v > 1]
    if dup:
        c.fail(f"같은 키가 두 번 이상 나온 조합 {len(dup)}개")
        for k, v in dup[: args.list_bad]:
            print(f"      {v}회  {k}")
    else:
        c.ok(f"고유 조합 {len(seen):,}개, 중복 0")

    # ---------------- 3. kernels.jsonl 조인 -------------------------------
    hr("3. kernels.jsonl 조인")
    if unknown_kid:
        c.fail(f"kernels.jsonl 에 없는 kernel_id {len(unknown_kid)} 종 "
               f"({sum(unknown_kid.values()):,} 줄)")
        for kid, v in unknown_kid.most_common(args.list_bad):
            print(f"      {v:,}줄  {kid}")
    else:
        c.ok(f"모든 kernel_id 가 kernels.jsonl 에 존재한다 "
             f"(참조된 커널 {len({k[0] for k in seen}):,}종)")

    # ---------------- 4. 누락 ---------------------------------------------
    hr("4. 누락 — 기대 작업 수 vs 실제")
    if args.expect == "subset":
        print("  --expect subset: 실제로 측정된 형상으로 기대 목록을 좁힌다")
    expected, missing = expected_jobs(
        hw, backend, kern, seen,
        only_shapes={(k[1], k[2], k[3]) for k in seen} if args.expect == "subset"
        else None,
        only_kernels={k[0] for k in seen} if args.expect == "subset" else None)
    print(f"  기대 조합 {len(expected):,}  실제 측정 {len(seen):,}  "
          f"누락 {len(missing):,}")
    if missing:
        frac = len(missing) / max(len(expected), 1)
        (c.fail if frac > 0.001 else c.warn)(
            f"누락 {len(missing):,}건 ({100 * frac:.3f}%)")
        by_shape = Counter((m[1], m[2], m[3]) for m in missing)
        print("    형상별 누락 상위:")
        for k, v in by_shape.most_common(10):
            print(f"      {k}: {v:,}")
        print(f"    예시 {min(args.list_missing, len(missing))}건:")
        for m in list(missing)[: args.list_missing]:
            print(f"      {m}")
    else:
        c.ok("누락 없음")
    extra = [k for k in seen if k not in expected]
    if extra:
        by_shape = Counter((k[1], k[2], k[3]) for k in extra)
        grid = {(p.M, p.N, p.K) for p in all_shapes(hw)}
        off_grid = {s for s in by_shape if s not in grid}
        msg = f"기대 목록에 없는 측정 {len(extra):,}건"
        if off_grid:
            msg += (f" — 그중 {sum(by_shape[s] for s in off_grid):,}건은 "
                    f"형상 그리드 밖이다 ({sorted(off_grid)[:3]}...). "
                    "리허설 전용 형상이면 정상이다")
        else:
            msg += " — 열거기가 바뀌었을 수 있다"
        c.warn(msg)
        for k in extra[:5]:
            print(f"      {k}")

    # ---------------- 5. 커버리지 -----------------------------------------
    hr("5. 형상 커버리지")
    shapes = all_shapes(hw)
    if args.expect == "subset":
        meas = {(k[1], k[2], k[3]) for k in seen}
        shapes = [s for s in shapes if (s.M, s.N, s.K) in meas] or shapes
    have = {s for s in shapes if ok_per_shape[(s.M, s.N, s.K)] > 0}
    zero = [s for s in shapes if (s.M, s.N, s.K) not in
            {(x.M, x.N, x.K) for x in have}]
    print(f"  형상 {len(shapes)}개 중 ok 측정이 있는 형상 {len(have)}개")
    if zero:
        c.fail(f"ok 측정이 0 인 형상 {len(zero)}개 — 이 형상은 표에서 쓸 수 없다")
        for s in zero[:20]:
            print(f"      ({s.M},{s.N},{s.K})  ok={ok_per_shape[(s.M, s.N, s.K)]}")
    else:
        c.ok("모든 형상에 ok 측정이 1개 이상 있다")
    thin = [(k, v) for k, v in ok_per_shape.items() if v < 10]
    if thin:
        c.warn(f"ok 측정이 10 개 미만인 형상 {len(thin)}개 — 최적 선택이 불안정하다")
        for k, v in sorted(thin, key=lambda x: x[1])[:10]:
            print(f"      {k}: {v}")

    # ---------------- 6. 물리적으로 불가능한 값 ---------------------------
    hr("6. SOL(speed of light) 위반")
    print(f"  기준: t < 2·M·N·K / {hw.peak_tflops_f16} TFLOP/s × {SOL_TOLERANCE}")
    if sol_bad:
        c.fail(f"SOL 보다 빠른 측정 {len(sol_bad)}건 — 측정 오류다")
        for key, kid, t, sol in sol_bad[: args.list_bad]:
            print(f"      {key} {t:.4f} ms < SOL {sol:.4f} ms  ({kid})")
    else:
        c.ok("SOL 위반 없음")

    # ---------------- 7. max_rel_error 분포 -------------------------------
    hr("7. max_rel_error 분포 (split_k / mode 별)")
    print(f"  {'mode':>9s} {'split_k':>8s} {'n':>8s} {'median':>10s} "
          f"{'max':>10s}   판정")
    bad_err = 0
    ser_med: dict[int, float] = {}
    for (mode, sk) in sorted(err_by_sk, key=lambda x: (x[0], x[1])):
        v = sorted(err_by_sk[(mode, sk)])
        med, mx = v[len(v) // 2], v[-1]
        verdict = "ok"
        if mx > NUMERICAL_TOL:
            verdict = "**상한 초과**"
            bad_err += 1
        if mode == "parallel" and med > 1e-2:
            verdict = "parallel 인데 중앙값이 크다"
            bad_err += 1
        if mode == "serial":
            ser_med[sk] = med
        print(f"  {mode:>9s} {sk:8d} {len(v):8,d} {med:10.3e} {mx:10.3e}   {verdict}")
    if bad_err:
        c.fail(f"max_rel_error 이상 그룹 {bad_err}개")
    else:
        c.ok("모든 그룹이 상한 이내")
    if len(ser_med) >= 3:
        ks = sorted(ser_med)
        mono = sum(1 for a, b in itertools.pairwise(ks) if ser_med[b] >= ser_med[a] - 1e-9)
        if mono >= len(ks) - 2:
            c.ok("serial 오차가 split_k 에 대해 대체로 증가한다 (예상대로)")
        else:
            c.warn("serial 오차가 split_k 에 대해 단조 증가하지 않는다 — "
                   "부분합 누적 경로를 확인할 것")

    # ---------------- 8. status 요약 --------------------------------------
    hr("8. status 요약")
    tot = sum(status.values())
    for k, v in status.most_common():
        print(f"  {k!s:24s} {v:9,d}  {100 * v / max(tot, 1):6.2f}%")
    ok_frac = status.get("ok", 0) / max(tot, 1)
    if ok_frac < 0.80:
        c.fail(f"ok 비율이 {100 * ok_frac:.1f}% 로 낮다")
    elif ok_frac < 0.90:
        c.warn(f"ok 비율 {100 * ok_frac:.1f}%")
    else:
        c.ok(f"ok 비율 {100 * ok_frac:.1f}%")

    return finish(c)


def expected_jobs(hw, backend, kern: dict, seen: Counter,
                  only_shapes: set | None = None,
                  only_kernels: set | None = None):
    """지금의 열거기 기준으로 기대되는 (커널, 형상, 런타임) 조합.

    측정 대상은 rehearse.py --all 과 같은 규칙으로 정한다:
    빌드 성공 ∧ 런치 가능 ∧ 현재 is_valid_kernel 통과.
    """
    valid = {backend.kernel_id(c) for c in
             enumerate_kernels(hw, backend, alignment_combos(all_shapes(hw)))}
    usable = []
    for r in kern.values():
        if r.get("build_status") != "ok" or r["kernel_id"] not in valid:
            continue
        if only_kernels is not None and r["kernel_id"] not in only_kernels:
            continue
        if not kernels_mod.launchable(r, hw.regs_per_sm):
            continue
        usable.append(r)

    expected = set()
    for r in usable:
        a = (r["align"]["a"], r["align"]["b"], r["align"]["c"])
        cfg = KernelConfig(r["tile"]["m"], r["tile"]["n"], r["tile"]["k"],
                           a[0], a[1], a[2], r["arch"],
                           backend.ext_from_dict(r["ext"]))
        for p in all_shapes(hw):
            if alignments_for(p) != a:
                continue
            if only_shapes is not None and (p.M, p.N, p.K) not in only_shapes:
                continue
            for rc in enumerate_runtimes(backend, p, cfg):
                expected.add((r["kernel_id"], p.M, p.N, p.K,
                              rc.split_k, rc.split_k_mode))
    missing = expected - set(seen)
    return expected, missing


def finish(c: Check) -> int:
    hr("결과")
    if c.fails:
        print(f"  실패 {len(c.fails)}건, 경고 {len(c.warns)}건")
        for f in c.fails:
            print(f"    - {f}")
        print("\n  이 표로 규칙 학습에 들어가지 마라.")
        return 4
    print(f"  통과 (경고 {len(c.warns)}건)")
    for wmsg in c.warns:
        print(f"    - {wmsg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
