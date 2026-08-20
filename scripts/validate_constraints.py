#!/usr/bin/env python3
"""빌드 실측 결과로 예측 모델을 검증한다.

* epilogue_thread_map_ok()  vs  실제 static_assert 빌드 실패
* smem_bytes()              vs  sizeof(SharedStorage)
* expected_hmma()           vs  SASS HMMA 카운트

모델이 실측과 어긋나면 그 모델로 config 를 미리 거르면 안 된다.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kerneltab.backends import get_backend
from kerneltab.backends.sm80 import (
    epilogue_thread_map_ok,
    mainloop_smem_thread_map_ok,
)
from kerneltab.core import paths
from kerneltab.core.types import KernelConfig

KERNELS = paths.RESULTS_DIR / "kernels.jsonl"

EPILOGUE_ASSERTS = (
    "ThreadMap::Iterations::kColumn must be > 0",
    "Iteration Count Column must be > 0",
    "Iteration Count Row must be > 0",
    "Accessing too many elements per access",
)
MAINLOOP_ASSERTS = ("Number of iterations must be non-zero",)
CPASYNC_ASSERTS = ("Size is not supported",)


def predict_ok(r: dict) -> tuple[bool, bool]:
    """(epilogue_ok, mainloop_ok)"""
    t, e, a = r["tile"], r["ext"], r["align"]
    wm = t["m"] // e["warp_m"]
    wn = t["n"] // e["warp_n"]
    wk = t["k"] // e["warp_k"]
    threads = wm * wn * wk * 32
    # cp.async 는 4/8/16 바이트만 지원한다 (fp16 x alignment 1 = 2바이트 불가)
    cpasync_ok = not (e["stages"] > 2 and min(a["a"], a["b"]) * 2 < 4)
    return (
        epilogue_thread_map_ok(t["n"], wm, wn, wk, a["c"]),
        mainloop_smem_thread_map_ok(t["m"], t["n"], t["k"], threads) and cpasync_ok,
    )


def main() -> int:
    rows = [json.loads(l) for l in KERNELS.read_text().splitlines() if l.strip()]
    print(f"kernels.jsonl: {len(rows)} 줄")

    status = Counter(r.get("build_status") for r in rows)
    print(f"build_status: {dict(status)}")
    fails = [r for r in rows if r.get("build_status") == "build_fail"]
    print(f"\n실패 원인별 ({len(fails)}건):")
    for msg, n in Counter(r.get("build_error", "?") for r in fails).most_common():
        print(f"  {n:5d}  {msg}")

    # --- 1) epilogue thread map 예측 vs 실측 --------------------------------
    print("\n" + "=" * 72)
    print("epilogue_thread_map_ok() 예측 vs 실제 빌드 결과")
    print("=" * 72)
    tp = fp = tn = fn = 0
    fn_examples, fp_examples = [], []
    other_fail = Counter()
    for r in rows:
        epi_ok, main_ok = predict_ok(r)
        pred_ok = epi_ok and main_ok
        st = r.get("build_status")
        err = r.get("build_error", "")
        if st == "build_fail":
            known = (any(s in err for s in EPILOGUE_ASSERTS)
                     or any(s in err for s in MAINLOOP_ASSERTS)
                     or any(s in err for s in CPASYNC_ASSERTS))
            if not known:
                other_fail[err] += 1
                continue
            if pred_ok:
                fn += 1  # 실패인데 통과로 예측 (모델이 놓침)
                if len(fn_examples) < 8:
                    fn_examples.append((r["kernel_id"], err))
            else:
                tp += 1
        elif st == "ok":
            if pred_ok:
                tn += 1
            else:
                fp += 1  # 성공인데 실패로 예측 (모델이 과잉 차단 -> 위험)
                if len(fp_examples) < 8:
                    fp_examples.append(
                        (r["kernel_id"], f"epi={epi_ok} main={main_ok}"))

    print(f"  실패를 실패로 맞춤 (TP): {tp}")
    print(f"  성공을 성공으로 맞춤 (TN): {tn}")
    print(f"  놓친 실패        (FN): {fn}   <- 모델이 못 잡은 것")
    print(f"  잘못 막을 성공   (FP): {fp}   <- 있으면 절대 필터로 쓰면 안 됨")
    for kid, err in fn_examples:
        print(f"    FN: {kid}  {err}")
    for kid, why in fp_examples:
        print(f"    FP: {kid}  {why}")
    if other_fail:
        print("\n  모델링하지 않은 실패:")
        for msg, n in other_fail.most_common():
            print(f"    {n:5d}  {msg}")

    # --- 2) smem / hmma 대조 ------------------------------------------------
    ok = [r for r in rows if r.get("build_status") == "ok"]
    # kernels.jsonl 은 append-only 라 smem_computed 는 빌드 시점의 공식 값이다.
    # 공식을 고쳤을 수 있으므로 **지금 공식으로 다시 계산해** 대조한다.
    be = get_backend(ok[0]["arch"]) if ok else None
    def recompute(r):
        cfg = KernelConfig(
            tile_m=r["tile"]["m"], tile_n=r["tile"]["n"], tile_k=r["tile"]["k"],
            align_a=r["align"]["a"], align_b=r["align"]["b"],
            align_c=r["align"]["c"], arch=r["arch"],
            ext=be.ext_from_dict(r["ext"]))
        return be.smem_bytes(cfg, 2)
    for r in ok:
        r["smem_recomputed"] = recompute(r)
    stale = [r for r in ok if r["smem_recomputed"] != r.get("smem_computed")]
    print(f"\n(참고) 저장된 smem_computed 가 현재 공식과 다른 줄: {len(stale)} "
          f"— 공식 수정 이력. export.py 는 항상 재계산한다.")
    smem_bad = [r for r in ok if r.get("smem_dynamic") != r["smem_recomputed"]]
    hmma_bad = [r for r in ok if r.get("hmma_count") != r.get("expected_hmma")]
    print("\n" + "=" * 72)
    print(f"smem_computed vs sizeof(SharedStorage): 불일치 {len(smem_bad)}/{len(ok)}")
    for r in smem_bad[:5]:
        print(f"  {r['kernel_id']}: {r['smem_recomputed']} vs {r['smem_dynamic']}")
    print(f"expected_hmma vs SASS HMMA:             불일치 {len(hmma_bad)}/{len(ok)}")
    ratios = Counter()
    for r in hmma_bad:
        exp, got = r.get("expected_hmma"), r.get("hmma_count")
        ratios[round(got / exp, 3) if exp else None] += 1
    for k, v in ratios.most_common(10):
        print(f"  비율 {k}: {v}건")
    for r in hmma_bad[:5]:
        print(f"  {r['kernel_id']}: expected={r['expected_hmma']} "
              f"sass={r['hmma_count']}")

    # ptxas 와 cuobjdump 교차 검증
    reg_bad = [r for r in ok
               if r.get("res_regs") is not None
               and r.get("regs_per_thread") is not None
               and r["res_regs"] != r["regs_per_thread"]]
    print(f"ptxas regs vs cuobjdump regs:           불일치 {len(reg_bad)}/{len(ok)}")

    # occupancy 교차 검증
    occ_bad = [r for r in ok
               if r.get("max_blocks_per_sm") != r.get("cutlass_max_blocks")]
    print(f"cudaOccupancy vs CUTLASS maximum_active_blocks: "
          f"불일치 {len(occ_bad)}/{len(ok)}")
    for r in occ_bad[:5]:
        print(f"  {r['kernel_id']}: {r['max_blocks_per_sm']} vs "
              f"{r['cutlass_max_blocks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
