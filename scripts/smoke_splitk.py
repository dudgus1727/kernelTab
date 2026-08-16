#!/usr/bin/env python3
"""split-K 경로 스모크 테스트 — 리허설 전에 한 번.

parallel 모드는 GEMM 이 부분합을 workspace 에 쓰고 별도 리덕션 커널이 합치는
2단 구성이라, 여기가 틀리면 리허설 결과의 절반이 조용히 쓰레기가 된다.

동시에 요청 A(actual_split_k 실측 대조)도 여기서 먼저 확인한다.

    python3 scripts/smoke_splitk.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backends import get_backend  # noqa: E402
from build import paths  # noqa: E402
from core.hardware import hardware_from_env  # noqa: E402
from core.types import Hardware, Problem, RuntimeConfig  # noqa: E402

SHAPES = [Problem(1024, 1024, 4096), Problem(1024, 4096, 512),
          Problem(1024, 4096, 4100)]


def main() -> int:
    env = json.loads(paths.ENV_JSON.read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(env["device_index"])
    hw = hardware_from_env(env)
    backend = get_backend(hw.arch)

    from measure.runner import Ctx, Kernel, KtProblemC  # noqa: E402

    rows = [json.loads(l) for l in
            (paths.RESULTS_DIR / "kernels.jsonl").read_text().splitlines() if l.strip()]
    ok = [r for r in rows if r.get("build_status") == "ok"
          and r["regs_per_thread"] * r["threads"] <= hw.regs_per_sm]

    def pick(align, tile=(128, 128, 32)):
        for r in sorted(ok, key=lambda r: r["kernel_id"]):
            a = (r["align"]["a"], r["align"]["b"], r["align"]["c"])
            t = (r["tile"]["m"], r["tile"]["n"], r["tile"]["k"])
            if a == align and t == tile and r["ext"]["stages"] == 4:
                return r
        return None

    ctx = Ctx(paths.ARTIFACT_DIR / "libkt_ctx.so", 0)
    bad = 0
    try:
        for p in SHAPES:
            from core.config import alignments_for
            al = alignments_for(p)
            krow = pick(al)
            if krow is None:
                print(f"({p.M},{p.N},{p.K}) align={al}: 해당 커널 없음")
                continue
            k = Kernel(krow["so_path"])
            ctx.prepare_problem(p.M, p.N, p.K)
            _, cub = ctx.measure_cublas()
            print(f"\n({p.M},{p.N},{p.K})  align={al}  kernel={krow['kernel_id']}")
            print(f"  cuBLAS {cub.time_ms:.4f} ms")
            print(f"  {'split_k':>8s} {'mode':>9s} {'grid.z':>7s} {'예측':>5s} "
                  f"{'time_ms':>9s} {'max_rel_err':>12s} {'ws_MB':>8s}")
            tbl = {}
            from core.config import dtype_bytes  # noqa: F401
            cfg_tile_k = krow["tile"]["k"]
            for sk in (1, 2, 3, 4, 6, 8, 12, 16):
                for mode in ("serial", "parallel"):
                    if sk == 1 and mode != "serial":
                        continue
                    rc = RuntimeConfig(sk, mode)
                    if sk * cfg_tile_k > p.K:
                        continue
                    pred = backend.effective_split_k(p, rc)
                    kp = KtProblemC(p.M, p.N, p.K, sk, 1 if mode == "parallel" else 0)
                    gz = k.grid_k(kp)
                    ws = k.workspace_bytes(kp)
                    valid = pred == sk
                    if ws:
                        ctx.buffers(ws, mode == "parallel")
                    bufs = ctx.buffers(ws, mode == "parallel")
                    st, h = k.prepare(kp, bufs)
                    if st != 0 or not h:
                        print(f"  {sk:8d} {mode:>9s} {gz:7d} {pred:5d}  "
                              f"prepare 실패: {k.status_string(st)}")
                        continue
                    try:
                        rs = ctx.run_once(k.launch_addr, h, gz if mode == "parallel" else 0)
                        err = ctx.max_rel_error() if rs == 0 else -1
                        _, m = ctx.measure(k.launch_addr, h,
                                           gz if mode == "parallel" else 0)
                    finally:
                        k.release(h)
                    flag = ""
                    if gz != pred:
                        flag += "  <-- 예측 불일치!"
                        bad += 1
                    if err > 5e-2 or err < 0:
                        flag += "  <-- 정확도 문제!"
                        bad += 1
                    skip = "" if valid else "  (열거에서 제외되는 조합)"
                    tbl[(sk, mode)] = (m.time_ms, err)
                    print(f"  {sk:8d} {mode:>9s} {gz:7d} {pred:5d} "
                          f"{m.time_ms:9.4f} {err:12.3e} {ws / 2**20:8.2f}"
                          f"{flag}{skip}")
            print(f"\n  [split_k 별 serial vs parallel]  "
                  f"부분합 저장은 양쪽 다 fp16 (serial 은 D 를 재읽기)")
            print(f"    {'split_k':>8s} | {'serial ms':>10s} {'serial err':>11s}"
                  f" | {'par ms':>10s} {'par err':>11s}")
            for sk in sorted({key[0] for key in tbl}):
                se = tbl.get((sk, "serial"))
                pe = tbl.get((sk, "parallel"))
                def fmt(x, i, w):
                    return f"{x[i]:{w}}" if x else " " * 10
                print(f"    {sk:8d} | {fmt(se, 0, '10.4f')} {fmt(se, 1, '11.3e')}"
                      f" | {fmt(pe, 0, '10.4f')} {fmt(pe, 1, '11.3e')}")
    finally:
        ctx.close()
    print("\n결론: " + ("이상 없음" if bad == 0 else f"!! {bad}건 문제"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
