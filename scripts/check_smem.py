#!/usr/bin/env python3
"""backend.smem_bytes() 가 CUTLASS 의 실제 SharedStorage 크기와 맞는지 검증.

Phase 2 에서 수천 개를 빌드하기 전에 공식이 맞는지 확인한다. 공식이 과대
평가면 유효한 config 를 잃고, 과소평가면 빌드/런치가 깨진다.

emit_cpp() 결과를 -DKT_NO_IMPL 로 컴파일해 타입만 인스턴스화하고
sizeof(GemmKernel::SharedStorage) 를 출력시킨다.

사용:
    python3 scripts/check_smem.py [--n 40] [--jobs 24]
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kerneltab.backends import get_backend
from kerneltab.build import paths
from kerneltab.core.config import alignment_combos, dtype_bytes, enumerate_kernels
from kerneltab.core.hardware import (
    detect_hardware,
    hardware_from_env,
    nvcc_arch_flag,
)
from kerneltab.core.shapes import all_shapes

MAIN = """
#include <cstdio>
int main() {
  using K = KtGemm::GemmKernel;
  printf("{\\"kernel_id\\": \\"%s\\", \\"smem\\": %zu, \\"threads\\": %d, "
         "\\"mainloop\\": %zu, \\"epilogue\\": %zu}\\n",
         KT_KERNEL_ID,
         sizeof(typename K::SharedStorage),
         int(K::kThreadCount),
         sizeof(typename K::Mma::SharedStorage),
         sizeof(typename K::Epilogue::SharedStorage));
  return 0;
}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="검사할 커널 수")
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if paths.ENV_JSON.exists():
        env = json.loads(paths.ENV_JSON.read_text())
        hw = hardware_from_env(env)
        cutlass = Path(env["cutlass"]["dir"])
    else:
        hw = detect_hardware(0)
        cutlass = paths.cutlass_dir()

    backend = get_backend(hw.arch)
    nb = dtype_bytes("f16")
    shapes = all_shapes(hw)
    kernels = enumerate_kernels(hw, backend, alignment_combos(shapes))

    rng = random.Random(args.seed)
    # 극단값(최대/최소 smem)을 반드시 포함시킨다.
    by_smem = sorted(kernels, key=lambda c: backend.smem_bytes(c, nb))
    picks = [by_smem[0], by_smem[-1], by_smem[len(by_smem) // 2]]
    picks += rng.sample(kernels, max(0, args.n - len(picks)))

    workdir = paths.ARTIFACT_DIR / "smemcheck"
    workdir.mkdir(parents=True, exist_ok=True)
    arch = nvcc_arch_flag(hw.arch)

    def build_and_run(cfg):
        kid = backend.kernel_id(cfg)
        src = workdir / f"{kid}.cu"
        src.write_text(backend.emit_cpp(cfg) + MAIN)
        exe = workdir / kid
        cmd = [
            str(paths.nvcc_path()), "-std=c++17", f"-arch={arch}", "-O0",
            "-DKT_NO_IMPL", *paths.kernel_includes(cutlass),
            str(src), "-o", str(exe),
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            return cfg, None, p.stderr[-800:]
        r = subprocess.run([str(exe)], capture_output=True, text=True)
        if r.returncode != 0:
            return cfg, None, r.stderr[-800:]
        return cfg, json.loads(r.stdout.strip()), None

    print(f"{len(picks)}개 커널을 {args.jobs} job 으로 빌드/검증 중...")
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(build_and_run, picks))

    ok = mismatch = fail = 0
    print(f"\n{'kernel_id':62s} {'계산':>7s} {'실측':>7s} {'main':>7s} {'epi':>7s} {'thr':>5s}")
    for cfg, out, err in results:
        kid = backend.kernel_id(cfg)
        if out is None:
            fail += 1
            print(f"{kid:62s}   BUILD FAIL")
            print(f"    {err.splitlines()[-1] if err else ''}")
            continue
        calc = backend.smem_bytes(cfg, nb)
        got = out["smem"]
        flag = "" if calc == got else "  <-- MISMATCH"
        if calc == got:
            ok += 1
        else:
            mismatch += 1
        print(f"{kid:62s} {calc:7d} {got:7d} {out['mainloop']:7d} "
              f"{out['epilogue']:7d} {out['threads']:5d}{flag}")

    print(f"\n일치 {ok} / 불일치 {mismatch} / 빌드실패 {fail}")
    return 0 if (mismatch == 0 and fail == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
