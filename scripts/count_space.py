#!/usr/bin/env python3
"""Phase 1 검증: 탐색 공간의 크기를 세고 제약 funnel 을 출력한다.

측정을 시작하기 전에 "몇 개를 빌드하고 몇 번 측정할 것인가"를 확정한다.
예상 규모에서 크게 벗어나면 제약에 버그가 있는 것이다.

사용:
    python3 scripts/count_space.py [--device 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kerneltab.backends import get_backend
from kerneltab.core import paths
from kerneltab.core.config import (
    alignment_combos,
    alignments_for,
    dtype_bytes,
    enumerate_kernels_with_funnel,
    enumerate_runtimes,
)
from kerneltab.core.features import (
    arith_intensity,
    is_memory_bound,
    mainloop_iters,
    ridge_point,
    waves,
)
from kerneltab.core.hardware import (
    detect_hardware,
    hardware_from_env,
)
from kerneltab.core.shapes import all_layers, all_shapes
from kerneltab.core.types import Hardware, Problem


def load_hw(device: int) -> Hardware:
    """env.json 이 있으면 그것을 쓰고, 없으면 직접 감지한다."""
    if paths.ENV_JSON.exists():
        env = json.loads(paths.ENV_JSON.read_text())
        return hardware_from_env(env)
    return detect_hardware(device)


def hr(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    hw = load_hw(args.device)
    backend = get_backend(hw.arch)
    nb = dtype_bytes("f16")

    print(f"GPU: {hw.name} ({hw.arch})  SM={hw.sm_count}  "
          f"smem/block={hw.smem_per_block}  L2={hw.l2_bytes // 2**20}MB")
    print(f"backend: {type(backend).__name__}")

    # ---------------- 형상 ----------------
    hr("형상 그리드")
    layers = all_layers(hw)
    total_with_dups = 0
    for name, probs in layers.items():
        total_with_dups += len(probs)
        print(f"  {name:14s} {len(probs):3d}개")
    shapes = all_shapes(hw)
    print(f"  {'-' * 30}")
    print(f"  {'층 합계':14s} {total_with_dups:3d}개 (층 간 중복 포함)")
    print(f"  {'고유 형상':14s} {len(shapes):3d}개 (중복 제거)")

    print("\n  층 C (waves 역산) M 값:")
    print("   ", [p.M for p in layers["c_waves"]])

    # ---------------- alignment ----------------
    hr("alignment 조합 (형상에서 유도)")
    combos = alignment_combos(shapes)
    counts: Counter = Counter(alignments_for(p) for p in shapes)
    for c in combos:
        exs = [f"({p.M},{p.N},{p.K})" for p in shapes if alignments_for(p) == c][:3]
        print(f"  a{c[0]}{c[1]}{c[2]}  형상 {counts[c]:3d}개   예: {', '.join(exs)}")
    print(f"  -> 빌드 대상 alignment 조합 {len(combos)}개 "
          f"(가능한 4^3=64 조합 중 실제 등장분만)")

    # ---------------- 커널 ----------------
    hr("커널 열거 funnel (alignment 조합 1개 기준)")
    kernels, funnel = enumerate_kernels_with_funnel(hw, backend, combos)
    raw = sum(funnel.values())
    order = [
        "tile_divisible_by_warp",
        "warp_count_mn",
        "warp_shape_vs_instruction",
        "warp_k_divisible",
        "threads_per_block",
        "smem_capacity",
        "accumulator_registers",
        "horizontal_swizzle_n",
        "mainloop_smem_thread_map",
        "epilogue_thread_map",
        "warp_n_incorrect_results",
    ]
    remaining = raw
    print(f"  {'축 곱집합(raw)':30s} {raw:7d}")
    for k in order:
        n = funnel.get(k, 0)
        remaining -= n
        print(f"  - {k:28s} {n:7d}  탈락  -> 남음 {remaining:7d}")
    print(f"  {'VALID':30s} {funnel.get('VALID', 0):7d}")

    per_align = funnel.get("VALID", 0)
    print(f"\n  alignment 조합당 유효 커널: {per_align}")
    print(f"  전체 빌드 대상 커널: {len(kernels)} "
          f"({per_align} x {len(combos)} alignment 조합)")

    # 유효 커널의 분포
    hr("유효 커널 분포 (alignment 조합 1개 기준)")
    one = [k for k in kernels if (k.align_a, k.align_b, k.align_c) == combos[0]]
    for key, label in (
        (lambda c: (c.tile_m, c.tile_n), "threadblock tile (M,N)"),
        (lambda c: c.tile_k, "tile_k"),
        (lambda c: c.ext.stages, "stages"),
        (lambda c: (c.ext.warp_m, c.ext.warp_n), "warp tile (M,N)"),
        (lambda c: (c.ext.swizzle_type, c.ext.swizzle_n), "swizzle"),
    ):
        cnt = Counter(key(c) for c in one)
        print(f"\n  {label}:")
        for k, v in sorted(cnt.items(), key=lambda x: str(x[0])):
            print(f"    {k!s:22s} {v:5d}")

    # smem 분포
    smems = sorted(backend.smem_bytes(c, nb) for c in one)
    print(f"\n  smem_bytes: min={smems[0]}  median={smems[len(smems) // 2]}  "
          f"max={smems[-1]}  (한도 {hw.smem_per_block})")

    # ---------------- 런타임 조합 ----------------
    hr("대표 형상별 (커널 x 런타임) 조합 수")
    reps = [
        Problem(64, 4096, 4096),
        Problem(1024, 1024, 4096),
        Problem(1024, 4096, 512),
        Problem(1024, 4096, 4100),
        Problem(4096, 4096, 4096),
        Problem(8192, 4096, 4096),
    ]
    rp = ridge_point(hw)
    print(f"  roofline ridge point = {rp:.1f} FLOP/byte\n")
    print(f"  {'shape':22s} {'align':7s} {'kern':>6s} {'runtime합':>10s} "
          f"{'AI':>7s} {'bound':>7s}")
    grand = 0
    for p in reps:
        al = alignments_for(p)
        ks = [k for k in kernels if (k.align_a, k.align_b, k.align_c) == al]
        total = sum(len(enumerate_runtimes(backend, p, k)) for k in ks)
        grand += total
        print(f"  ({p.M},{p.N},{p.K})".ljust(24)
              + f"a{al[0]}{al[1]}{al[2]}".ljust(8)
              + f"{len(ks):6d} {total:10d} "
              + f"{arith_intensity(p):7.1f} "
              + f"{'mem' if is_memory_bound(p, hw) else 'comp':>7s}")
    print(f"  {'-' * 62}")
    print(f"  리허설 6개 형상 전수 조합 합계: {grand:,}")

    # split-K 분포 확인 (3의 배수가 살아남는지)
    hr("split-K 값별 유효 조합 수 (대표 형상)")
    for p in (Problem(1024, 1024, 4096), Problem(1024, 4096, 512)):
        al = alignments_for(p)
        ks = [k for k in kernels if (k.align_a, k.align_b, k.align_c) == al]
        cnt: Counter = Counter()
        for k in ks:
            for rc in enumerate_runtimes(backend, p, k):
                cnt[(rc.split_k, rc.split_k_mode)] += 1
        print(f"  ({p.M},{p.N},{p.K}):")
        for key in sorted(cnt):
            print(f"    split_k={key[0]:2d} {key[1]:8s} {cnt[key]:6d}")

    # ---------------- 파생 지표 스모크 테스트 ----------------
    hr("파생 지표 스모크 테스트")
    cfg = next(c for c in one if (c.tile_m, c.tile_n, c.tile_k) == (128, 128, 32))
    from kerneltab.core.types import RuntimeConfig

    for p in (Problem(1024, 1024, 4096), Problem(8192, 4096, 4096)):
        for sk in (1, 6):
            rc = RuntimeConfig(split_k=sk, split_k_mode="serial")
            print(f"  ({p.M},{p.N},{p.K}) split_k={sk}: "
                  f"waves={waves(p, hw, cfg, rc):.3f}  "
                  f"mainloop_iters={mainloop_iters(p, cfg, rc)}  "
                  f"AI={arith_intensity(p):.1f}")

    # ---------------- 전수 측정 규모 (Phase 3 참고용, 실행하지 않음) ------
    hr("Phase 3 전수 측정 규모 (참고용 — 이번에 실행하지 않는다)")
    total_meas = 0
    for p in shapes:
        al = alignments_for(p)
        ks = [k for k in kernels if (k.align_a, k.align_b, k.align_c) == al]
        total_meas += sum(len(enumerate_runtimes(backend, p, k)) for k in ks)
    print(f"  (형상 x 커널 x 런타임) 총 측정 수: {total_meas:,}")
    print("  ※ 실제 소요 시간은 형상마다 크게 다르다. 프로토콜이 '총 20ms 또는")
    print("     최소 30회' 이므로 커널이 0.67ms 보다 느리면 최소 반복 수가")
    print("     지배한다 (8192^3 은 커널 1회가 ~20ms -> 작업당 1초 이상).")
    print("     정확한 견적은 scripts/rehearse.py --all --dry-run 과")
    print("     리허설 실측 효율을 함께 봐야 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
