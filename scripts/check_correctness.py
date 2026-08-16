#!/usr/bin/env python3
"""빌드된 커널 전수 정확성 검사 (측정 없이 1회 실행 + cuBLAS 대조).

리허설에서 특정 config 가 100% 오답을 내는 것이 발견되어 만들었다.
Phase 3 전수 측정에 들어가기 전에 "계산이 틀린 커널" 을 미리 전부 찾아두면,
수십만 줄의 측정 결과 중 어느 것이 쓰레기인지 사후에 뒤지지 않아도 된다.

시간 측정을 하지 않으므로 다른 GPU 에서 병행 실행해도 된다.

    python3 scripts/check_correctness.py --device 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build import paths  # noqa: E402
from core.hardware import hardware_from_env  # noqa: E402
from core.config import alignments_for  # noqa: E402
from core.types import Hardware, Problem  # noqa: E402

OUT = paths.RESULTS_DIR / "correctness.jsonl"
TOL = 5e-2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None,
                    help="검사에 쓸 GPU (기본: env.json 의 device_index)")
    ap.add_argument("--align", default="a888")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    env = json.loads(paths.ENV_JSON.read_text())
    dev = args.device if args.device is not None else env["device_index"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(dev)
    hw = hardware_from_env(env)

    from measure.runner import Ctx, Kernel, KtProblemC  # noqa: E402

    rows = [json.loads(l) for l in
            (paths.RESULTS_DIR / "kernels.jsonl").read_text().splitlines()
            if l.strip()]
    want = set(args.align.split(","))
    cand = [r for r in rows
            if r.get("build_status") == "ok"
            and f"a{r['align']['a']}{r['align']['b']}{r['align']['c']}" in want
            and r["regs_per_thread"] * r["threads"] <= hw.regs_per_sm]
    cand.sort(key=lambda r: r["kernel_id"])
    if args.limit:
        cand = cand[: args.limit]

    # 타일이 큰 config 도 최소 한 번은 온전한 타일을 돌도록 넉넉한 형상을 쓴다.
    shape = Problem(1024, 1024, 1024)
    assert alignments_for(shape) == (8, 8, 8)

    print(f"GPU {dev} 에서 {len(cand)}개 커널 정확성 검사  형상={shape.M}x{shape.N}x{shape.K}")
    ctx = Ctx(paths.ARTIFACT_DIR / "libkt_ctx.so", 0)
    ctx.set_protocol(env)
    ctx.prepare_problem(shape.M, shape.N, shape.K)
    kp = KtProblemC(shape.M, shape.N, shape.K, 1, 0)

    bad: list[dict] = []
    n_ok = 0
    with OUT.open("w") as f:
        for i, r in enumerate(cand, 1):
            rec = {"kernel_id": r["kernel_id"]}
            try:
                k = Kernel(r["so_path"])
                can = k.can_implement(kp)
                if can != 0:
                    rec.update(status="not_implementable",
                               detail=k.status_string(can))
                else:
                    bufs = ctx.buffers(k.workspace_bytes(kp), False)
                    st, h = k.prepare(kp, bufs)
                    if st != 0 or not h:
                        rec.update(status="prepare_fail",
                                   detail=k.status_string(st))
                    else:
                        try:
                            rs = ctx.run_once(k.launch_addr, h, 0)
                            if rs != 0:
                                rec.update(status="run_fail",
                                           detail=ctx.last_error())
                            else:
                                e = ctx.max_rel_error()
                                rec.update(
                                    status="ok" if e <= TOL else "numerical_fail",
                                    max_rel_error=e)
                        finally:
                            k.release(h)
            except Exception as exc:  # pragma: no cover
                rec.update(status="exception", detail=repr(exc))
            f.write(json.dumps(rec) + "\n")
            if rec["status"] == "ok":
                n_ok += 1
            else:
                bad.append({**rec, **{"tile": r["tile"], "ext": r["ext"],
                                      "threads": r["threads"],
                                      "regs": r["regs_per_thread"],
                                      "spill": r["spill_stores"]}})
            if i % 250 == 0:
                print(f"  {i}/{len(cand)}  정상 {n_ok}  이상 {len(bad)}", flush=True)
    ctx.close()

    print(f"\n정상 {n_ok} / 이상 {len(bad)} / 전체 {len(cand)}")
    if not bad:
        return 0

    print("\nstatus 별:")
    for k, v in Counter(b["status"] for b in bad).most_common():
        print(f"  {k:20s} {v}")

    print("\n이상 커널의 공통 속성:")
    for label, key in (
        ("warp tile (M,N)", lambda b: (b["ext"]["warp_m"], b["ext"]["warp_n"])),
        ("warp_k", lambda b: b["ext"]["warp_k"]),
        ("tile (M,N,K)", lambda b: (b["tile"]["m"], b["tile"]["n"], b["tile"]["k"])),
        ("stages", lambda b: b["ext"]["stages"]),
        ("threads", lambda b: b["threads"]),
        ("accum regs/thread",
         lambda b: b["ext"]["warp_m"] * b["ext"]["warp_n"] // 32),
        ("spill?", lambda b: b["spill"] > 0),
    ):
        print(f"  {label}: {dict(sorted(Counter(key(b) for b in bad).items(), key=str))}")

    # 같은 속성인데 정상인 것이 있는지 (있으면 그 속성이 원인이 아니다)
    okset = defaultdict(int)
    for r in cand:
        if r["kernel_id"] in {b["kernel_id"] for b in bad}:
            continue
        okset[(r["ext"]["warp_m"], r["ext"]["warp_n"])] += 1
    print("\n  warp tile 별 (정상 개수):", dict(sorted(okset.items(), key=str)))
    print(f"\n상세: {OUT}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
