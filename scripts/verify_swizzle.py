#!/usr/bin/env python3
"""요청 H: 직접 짠 kt::HorizontalThreadblockSwizzle 이 실제로 동작하는지 검증.

우리가 쓴 코드라 CUTLASS 원본과 동작이 같다는 보장이 없다. 여기에 버그가 있으면
**결과는 맞는데 타일 순서만 달라 성능만 틀리게** 나오고, 정확도 검증에는 전혀
걸리지 않는다. 그래서 별도로 세 가지를 본다.

1) 정확도  — 여러 형상에서 cuBLAS 대비 max_rel_error
2) grid dim — identity 는 (m_tiles, n_tiles, k), horizontal 은 (n_tiles, m_tiles, k)
               로 실제로 뒤바뀌는지 실측
3) 성능    — identity 와 horizontal 이 "같은 결과, 다른 성능" 을 내는지.
             완전히 동일하면 스위즐이 적용되지 않은 것이다.

    python3 scripts/verify_swizzle.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backends import get_backend  # noqa: E402
from build import paths  # noqa: E402
from core import device  # noqa: E402
from core.hardware import hardware_from_env  # noqa: E402
from build.compile import BuildEnv, build_ctx_so, build_kernel  # noqa: E402
from core.types import Hardware, KernelConfig, Problem  # noqa: E402

# 래스터 방향 차이가 드러나려면 타일 격자가 정사각이 아니어야 하고
# 타일 수가 SM 수보다 충분히 많아야 한다.
SHAPES = [
    Problem(4096, 4096, 4096),
    Problem(8192, 2048, 4096),   # m_tiles >> n_tiles
    Problem(2048, 8192, 4096),   # n_tiles >> m_tiles
    Problem(1024, 4096, 2048),
]

# 스위즐만 다르고 나머지는 완전히 동일한 커널들
BASE = dict(tile_m=128, tile_n=128, tile_k=32,
            warp_m=64, warp_n=64, warp_k=32, stages=4)
VARIANTS = [("identity", 1), ("identity", 8), ("horizontal", 1)]


def main() -> int:
    env = json.loads(paths.ENV_JSON.read_text())
    # P-2: UUID 가 권위다. 저장된 인덱스를 그대로 쓰면 컨테이너나
    #      CUDA_VISIBLE_DEVICES 가 설정된 환경에서 **다른 GPU 를 측정**한다.
    #      이미 설정돼 있으면 존중하고, 없는 UUID 면 명확히 실패한다.
    device.resolve_device(env)
    hw = hardware_from_env(env)
    backend = get_backend(hw.arch)
    be = BuildEnv.from_env_json(env)
    build_ctx_so(env)

    from measure.runner import Ctx, Kernel, KtProblemC  # noqa: E402

    cfgs = {}
    for swz, n in VARIANTS:
        ext = backend.ext_from_dict({
            "warp_m": BASE["warp_m"], "warp_n": BASE["warp_n"],
            "warp_k": BASE["warp_k"], "stages": BASE["stages"],
            "swizzle_type": swz, "swizzle_n": n})
        cfg = KernelConfig(
            tile_m=BASE["tile_m"], tile_n=BASE["tile_n"], tile_k=BASE["tile_k"],
            align_a=8, align_b=8, align_c=8, arch=hw.arch, ext=ext)
        cfgs[(swz, n)] = cfg

    print("빌드 중...")
    libs = {}
    for key, cfg in cfgs.items():
        r = build_kernel(cfg, backend, be)
        if r["build_status"] != "ok":
            print(f"  !! {key} 빌드 실패: {r.get('build_error')}")
            return 1
        print(f"  {backend.kernel_id(cfg)}  regs={r['regs_per_thread']} "
              f"smem={r['smem_dynamic']}")
        libs[key] = Kernel(paths.kernel_so(r["kernel_id"]))

    ctx = Ctx(paths.ARTIFACT_DIR / "libkt_ctx.so", 0)
    ctx.set_protocol(env)
    ok = True
    try:
        # --- 2) grid dim 실측 -------------------------------------------
        print("\n" + "=" * 72)
        print("grid dim 실측 (논리 타일 격자 vs 실제 런치 grid)")
        print("=" * 72)
        for p in SHAPES:
            kp = KtProblemC(p.M, p.N, p.K, 1, 0)
            tm, tn, tk = libs[("identity", 1)].tiled_shape(kp)
            print(f"\n  ({p.M},{p.N},{p.K})  타일격자 m={tm} n={tn} k={tk}")
            for key in VARIANTS:
                g = libs[key].grid_shape(kp)
                if key[0] == "identity":
                    want = (tm * key[1], (tn + key[1] - 1) // key[1], tk)
                    note = "M 우선 래스터"
                else:
                    want = (tn, tm, tk)
                    note = "N 우선 래스터"
                mark = "OK " if g == want else "!! "
                if g != want:
                    ok = False
                print(f"    {mark}{str(key):18s} grid={g}  기대={want}  ({note})")

        # --- 1) 정확도 + 3) 성능 ----------------------------------------
        print("\n" + "=" * 72)
        print("정확도 + 성능 (동일 커널, 스위즐만 다름)")
        print("=" * 72)
        for p in SHAPES:
            ctx.prepare_problem(p.M, p.N, p.K)
            kp = KtProblemC(p.M, p.N, p.K, 1, 0)
            _, cub = ctx.measure_cublas()
            print(f"\n  ({p.M},{p.N},{p.K})   cuBLAS {cub.time_ms:.4f} ms")
            times = {}
            for key in VARIANTS:
                k = libs[key]
                bufs = ctx.buffers(k.workspace_bytes(kp), False)
                st, h = k.prepare(kp, bufs)
                if st != 0:
                    print(f"    !! {key} prepare 실패")
                    ok = False
                    continue
                try:
                    st = ctx.run_once(k.launch_addr, h, 0)
                    err = ctx.max_rel_error() if st == 0 else -1.0
                    st2, m = ctx.measure(k.launch_addr, h, 0)
                finally:
                    k.release(h)
                times[key] = m.time_ms
                bad = "" if 0 <= err < 1e-3 else "   <-- 정확도 문제!"
                if not (0 <= err < 1e-3):
                    ok = False
                print(f"    {str(key):18s} {m.time_ms:.4f} ms  "
                      f"({100 * cub.time_ms / m.time_ms:5.1f}% of cuBLAS)  "
                      f"max_rel_err={err:.2e}{bad}")
            base = times.get(("identity", 1))
            if base:
                for key in VARIANTS[1:]:
                    if key not in times:
                        continue
                    d = (times[key] - base) / base
                    flag = "  <-- 성능 차이 없음 (스위즐 미적용 의심)" \
                        if abs(d) < 1e-3 else ""
                    print(f"      vs identity1: {100 * d:+6.2f}%{flag}")
    finally:
        ctx.close()

    print("\n" + "=" * 72)
    print("결론: " + ("모든 검증 통과" if ok else "!! 문제 발견 — 재빌드 전에 수정 필요"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
