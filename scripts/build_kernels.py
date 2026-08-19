#!/usr/bin/env python3
"""Phase 2-1: 커널 생성 + 빌드 + 정적 분석 -> results/kernels.jsonl

    python3 scripts/build_kernels.py --pilot 3            # 파일럿
    python3 scripts/build_kernels.py --align a888,a448    # 본 빌드
    python3 scripts/build_kernels.py                      # 전체 alignment 조합

kernels.jsonl 은 append-only 다. 이미 있는 kernel_id 는 건너뛰므로 중간에
죽어도 그대로 다시 돌리면 이어서 진행한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backends import get_backend  # noqa: E402
from build import paths  # noqa: E402
from core import device  # noqa: E402
from core.hardware import hardware_from_env  # noqa: E402
from build.compile import BuildEnv, build_ctx_so, build_kernel, introspect  # noqa: E402
from core.config import alignment_combos, dtype_bytes, enumerate_kernels  # noqa: E402
from core.shapes import all_shapes  # noqa: E402
from core.types import Hardware  # noqa: E402

KERNELS_JSONL = paths.RESULTS_DIR / "kernels.jsonl"

#: build_fail 비율이 이 값을 넘으면 중단하고 보고한다 (열거 설계 재검토 신호).
#
# 주의: 완료 순서는 편향되어 있다. static_assert 실패는 ~1초, 성공은 ~9초라
# 실패가 앞쪽에 몰려 도착한다. 따라서 표본이 적을 때의 순간 비율은 위로
# 치우친다. 충분한 표본이 쌓인 뒤 일정 간격으로만 판정한다.
FAIL_RATE_ABORT = 0.10
FAIL_RATE_MIN_SAMPLES = 1000
FAIL_RATE_CHECK_EVERY = 100


def load_env() -> dict:
    if not paths.ENV_JSON.exists():
        raise SystemExit("results/env.json 이 없다. scripts/phase0_env.py 를 먼저 돌려라.")
    return json.loads(paths.ENV_JSON.read_text())


def align_tag(cfg) -> str:
    return f"a{cfg.align_a}{cfg.align_b}{cfg.align_c}"


def existing_ids() -> set[str]:
    if not KERNELS_JSONL.exists():
        return set()
    ids = set()
    with KERNELS_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["kernel_id"])
            except Exception:
                continue
    return ids


def stratified_pilot(kernels, backend, n: int):
    """파일럿은 무작위가 아니라 서로 다른 코드 경로를 골라 뽑는다."""
    picks, seen = [], set()

    def take(pred, k=1):
        c = 0
        for cfg in kernels:
            if c >= k:
                break
            kid = backend.kernel_id(cfg)
            if kid in seen or not pred(cfg):
                continue
            seen.add(kid)
            picks.append(cfg)
            c += 1

    take(lambda c: (c.tile_m, c.tile_n, c.tile_k) == (128, 128, 32)
         and c.ext.warp_k == c.tile_k)                    # 가장 흔한 형태
    take(lambda c: c.ext.warp_k < c.tile_k)               # warp_k 분할 (PartitionsK=2)
    take(lambda c: c.ext.swizzle_type == "horizontal")    # 다른 swizzle
    take(lambda c: (c.tile_m, c.tile_n) == (256, 64))     # 비대칭 타일
    return picks[:n] if n < len(picks) else picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=min(48, (os.cpu_count() or 8)))
    ap.add_argument("--pilot", type=int, default=0, help="N개만 빌드하고 멈춘다")
    ap.add_argument("--align", default="a888,a448",
                    help="빌드할 alignment 조합 (콤마 구분, 'all' 가능)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    env = load_env()
    # 어떤 CUDA 호출보다 먼저. Phase 0 이 고른 물리 GPU 에 고정한다.
    # P-2: UUID 가 권위다. 저장된 인덱스를 그대로 쓰면 컨테이너나
    #      CUDA_VISIBLE_DEVICES 가 설정된 환경에서 **다른 GPU 를 측정**한다.
    #      이미 설정돼 있으면 존중하고, 없는 UUID 면 명확히 실패한다.
    device.resolve_device(env)
    hw = hardware_from_env(env)
    backend = get_backend(hw.arch)
    nb = dtype_bytes("f16")
    be = BuildEnv.from_env_json(env)
    paths.ensure_dirs()

    print(f"[ctx] libkt_ctx.so 빌드 중...")
    ctx_so = build_ctx_so(env, force=args.force)
    print(f"[ctx] {ctx_so}")

    shapes = all_shapes(hw)
    kernels = enumerate_kernels(hw, backend, alignment_combos(shapes))
    if args.align != "all":
        want = set(args.align.split(","))
        kernels = [c for c in kernels if align_tag(c) in want]

    if args.pilot:
        kernels = stratified_pilot(kernels, backend, args.pilot)
    elif args.limit:
        kernels = kernels[: args.limit]

    done = set() if args.force else existing_ids()
    todo = [c for c in kernels if backend.kernel_id(c) not in done]
    print(f"[build] 대상 {len(kernels)}개 중 {len(todo)}개 빌드 "
          f"({len(kernels) - len(todo)}개는 이미 완료), jobs={args.jobs}")
    if not todo:
        return 0

    t0 = time.time()
    rows: list[dict] = []
    fail_counter = Counter()
    n_done = n_fail = 0
    aborted = False

    with KERNELS_JSONL.open("a") as out, ThreadPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(build_kernel, c, backend, be, args.force): c for c in todo}
        for fut in as_completed(futs):
            cfg = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # pragma: no cover
                r = {"kernel_id": backend.kernel_id(cfg),
                     "build_status": "build_fail", "build_error": repr(e)}

            row = {
                "kernel_id": r["kernel_id"],
                "arch": cfg.arch,
                "tile": {"m": cfg.tile_m, "n": cfg.tile_n, "k": cfg.tile_k},
                "align": {"a": cfg.align_a, "b": cfg.align_b, "c": cfg.align_c},
                "ext": asdict(cfg.ext),
                "smem_computed": backend.smem_bytes(cfg, nb),
                "expected_hmma": backend.expected_hmma(cfg),
                "pipeline_kind": backend.pipeline_kind(cfg),
                "cutlass_commit": env["cutlass"]["commit"],
                "nvcc_arch": env["nvcc_arch_flag"],
            }
            row.update({k: v for k, v in r.items() if k != "kernel_id"})
            row.pop("build_error_full", None)
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

            n_done += 1
            if row["build_status"] == "build_fail":
                n_fail += 1
                fail_counter[row.get("build_error", "unknown")] += 1

            if n_done % 25 == 0 or n_done == len(todo):
                el = time.time() - t0
                rate = n_done / max(el, 1e-9)
                eta = (len(todo) - n_done) / max(rate, 1e-9)
                print(f"  {n_done}/{len(todo)}  fail={n_fail} "
                      f"({100 * n_fail / n_done:.1f}%)  "
                      f"{rate:.1f}/s  ETA {eta / 60:.1f}분", flush=True)

            if (n_done >= FAIL_RATE_MIN_SAMPLES
                    and n_done % FAIL_RATE_CHECK_EVERY == 0
                    and n_fail / n_done > FAIL_RATE_ABORT and not aborted):
                aborted = True
                print(f"\n!! build_fail 비율 {100 * n_fail / n_done:.1f}% > "
                      f"{100 * FAIL_RATE_ABORT:.0f}% — 중단한다.")
                for f in futs:
                    f.cancel()
                break

    _report(rows, hw, backend, nb, fail_counter, time.time() - t0)
    return 2 if aborted else 0


def _report(rows, hw, backend, nb, fail_counter, elapsed):
    ok = [r for r in rows if r.get("build_status") in ("ok", "cached")]
    fail = [r for r in rows if r.get("build_status") == "build_fail"]
    print("\n" + "=" * 72)
    print(f"빌드 완료: 성공 {len(ok)} / 실패 {len(fail)} / 전체 {len(rows)}  "
          f"({elapsed / 60:.1f}분)")
    print("=" * 72)

    if fail:
        print(f"\n실패 원인별 ({len(fail)}건):")
        for msg, n in fail_counter.most_common():
            print(f"  {n:5d}  {msg}")

    if not ok:
        return

    regs = [r["regs_per_thread"] for r in ok if r.get("regs_per_thread")]
    if regs:
        regs.sort()
        print(f"\n레지스터/스레드: min={regs[0]} p25={regs[len(regs) // 4]} "
              f"median={regs[len(regs) // 2]} p75={regs[3 * len(regs) // 4]} "
              f"max={regs[-1]}")

    spilled = [r for r in ok if (r.get("spill_stores") or 0) > 0
               or (r.get("spill_loads") or 0) > 0]
    print(f"\n스필 커널: {len(spilled)}/{len(ok)} = "
          f"{100 * len(spilled) / len(ok):.1f}%")
    if spilled:
        by_warp = Counter((r["ext"]["warp_m"], r["ext"]["warp_n"]) for r in spilled)
        tot_warp = Counter((r["ext"]["warp_m"], r["ext"]["warp_n"]) for r in ok)
        print("  warp tile 별 스필 비율:")
        for k in sorted(tot_warp, key=lambda x: -by_warp.get(x, 0)):
            print(f"    {str(k):12s} {by_warp.get(k, 0):5d}/{tot_warp[k]:5d} "
                  f"= {100 * by_warp.get(k, 0) / tot_warp[k]:5.1f}%")

    mism_smem = [r for r in ok
                 if r.get("smem_dynamic") is not None
                 and r["smem_dynamic"] != r["smem_computed"]]
    print(f"\nsmem 계산값 vs 실측: 불일치 {len(mism_smem)}건")
    for r in mism_smem[:5]:
        print(f"  {r['kernel_id']}: computed={r['smem_computed']} "
              f"actual={r['smem_dynamic']}")


if __name__ == "__main__":
    raise SystemExit(main())
