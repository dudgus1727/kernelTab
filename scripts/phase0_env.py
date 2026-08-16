#!/usr/bin/env python3
"""Phase 0: 환경 점검 + 하드웨어 감지 + 런치 오버헤드 측정 -> results/env.json

env.json 은 이 스크립트가 한 번 쓰고 이후 단계는 읽기만 한다.
모든 측정 결과 줄이 env_hash 로 이 파일을 참조하므로 나중에 수정하면 안 된다.

사용:
    python3 scripts/phase0_env.py --device 0
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build import paths  # noqa: E402
from core.hardware import (  # noqa: E402
    detect_hardware,
    device_uuid,
    extra_device_info,
    hardware_to_dict,
    nvcc_arch_flag,
)
from measure.gpu_state import drift_check_seconds, try_lock_clocks  # noqa: E402


def run(cmd: list[str], **kw) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return p.returncode, (p.stdout + p.stderr).strip()


def host_info() -> dict:
    mem = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            mem[k] = int(v.strip().split()[0])  # kB
    except OSError:
        pass
    return {
        "hostname": platform.node(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": round(mem.get("MemTotal", 0) / 1024 / 1024, 1),
        "ram_available_gb": round(mem.get("MemAvailable", 0) / 1024 / 1024, 1),
    }


def cuda_info() -> dict:
    nvcc = paths.nvcc_path()
    _, ver = run([str(nvcc), "--version"])
    m = re.search(r"release (\d+\.\d+), V(\S+)", ver)
    drv_api = ctypes.c_int()
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        lib.cuInit(0)
        lib.cuDriverGetVersion(ctypes.byref(drv_api))
    except OSError:
        drv_api.value = 0
    _, drv = run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    return {
        "nvcc_path": str(nvcc),
        "nvcc_release": m.group(1) if m else None,
        "nvcc_version": m.group(2) if m else None,
        "cuda_home": str(paths.cuda_home()),
        "driver_version": drv.splitlines()[0].strip() if drv else None,
        "cuda_driver_api_version": drv_api.value,
        "cuobjdump": str(paths.cuda_bin("cuobjdump")),
        "nvdisasm": str(paths.cuda_bin("nvdisasm")),
    }


def cutlass_info(explicit: str | None) -> dict:
    root = paths.cutlass_dir(explicit)
    rc, commit = run(["git", "-C", str(root), "rev-parse", "HEAD"])
    rc2, dirty = run(["git", "-C", str(root), "status", "--porcelain"])
    rc3, desc = run(["git", "-C", str(root), "describe", "--tags", "--always"])
    ver = None
    vfile = root / "include" / "cutlass" / "version.h"
    if vfile.exists():
        txt = vfile.read_text()
        parts = {
            k: re.search(rf"#define CUTLASS_{k}\s+(\d+)", txt)
            for k in ("MAJOR", "MINOR", "PATCH")
        }
        if all(parts.values()):
            ver = ".".join(parts[k].group(1) for k in ("MAJOR", "MINOR", "PATCH"))

    # 2.x GEMM API 가 이 커밋에서 deprecated 표시되었는지 확인
    dep_hits = []
    for rel in (
        "include/cutlass/gemm/device/gemm_universal.h",
        "include/cutlass/gemm/device/gemm_universal_base.h",
        "include/cutlass/gemm/kernel/default_gemm_universal.h",
    ):
        f = root / rel
        if not f.exists():
            dep_hits.append(f"{rel}: MISSING")
            continue
        for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            if re.search(r"deprecat", line, re.I):
                dep_hits.append(f"{rel}:{i}: {line.strip()[:120]}")

    return {
        "dir": str(root),
        "commit": commit.strip() if rc == 0 else None,
        "describe": desc.strip() if rc3 == 0 else None,
        "version": ver,
        "worktree_dirty": bool(dirty.strip()) if rc2 == 0 else None,
        "gemm_2x_api_deprecation_hits": dep_hits,
        "gemm_2x_api_deprecated": bool(dep_hits),
    }


def gpu_smi_info(index: int) -> dict:
    fields = [
        "name",
        "uuid",
        "pci.bus_id",
        "ecc.mode.current",
        "ecc.mode.pending",
        "persistence_mode",
        "compute_mode",
        "memory.total",
        "clocks.max.sm",
        "clocks.max.memory",
        "clocks.default_applications.graphics",
        "power.limit",
    ]
    rc, out = run(
        [
            "nvidia-smi",
            "-i",
            str(index),
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader",
        ]
    )
    if rc != 0:
        return {"error": out}
    vals = [v.strip() for v in out.splitlines()[0].split(",")]
    return dict(zip(fields, vals))


def pynvml_info() -> dict:
    try:
        import pynvml
    except ImportError:
        return {"available": False, "error": "pynvml (nvidia-ml-py) 미설치"}
    try:
        pynvml.nvmlInit()
        d = {
            "available": True,
            "nvml_driver_version": pynvml.nvmlSystemGetDriverVersion(),
            "device_count": pynvml.nvmlDeviceGetCount(),
        }
        try:
            import importlib.metadata as md

            d["package_version"] = md.version("nvidia-ml-py")
        except Exception:
            pass
        pynvml.nvmlShutdown()
        return d
    except Exception as e:
        return {"available": False, "error": repr(e)}


def measure_launch_overhead(arch_flag: str) -> dict:
    paths.ensure_dirs()
    src = REPO_ROOT / "build" / "launch_probe.cu"
    exe = paths.ARTIFACT_DIR / "launch_probe"
    cmd = [
        str(paths.nvcc_path()),
        "-std=c++17",
        f"-arch={arch_flag}",
        "-O3",
        str(src),
        "-o",
        str(exe),
    ]
    rc, out = run(cmd)
    if rc != 0:
        raise RuntimeError(f"launch_probe 빌드 실패:\n{out}")
    rc, out = run([str(exe)])
    if rc != 0:
        raise RuntimeError(f"launch_probe 실행 실패:\n{out}")
    return json.loads(out.splitlines()[-1])


def sanity_check_example(cutlass_root: Path, arch_flag: str) -> dict:
    """CUTLASS Ampere fp16 예제(47_ampere_gemm_universal_streamk) 빌드/실행.

    이게 실패하면 이후 모든 작업이 무의미하므로 여기서 중단한다.
    """
    paths.ensure_dirs()
    src = (
        cutlass_root
        / "examples"
        / "47_ampere_gemm_universal_streamk"
        / "ampere_gemm_universal_streamk.cu"
    )
    if not src.exists():
        return {"ok": False, "error": f"예제 소스 없음: {src}"}
    exe = paths.ARTIFACT_DIR / "cutlass_example47"
    cmd = [
        str(paths.nvcc_path()),
        "-std=c++17",
        f"-arch={arch_flag}",
        "-O3",
        *paths.cutlass_includes(cutlass_root),
        f"-I{cutlass_root / 'examples' / 'common'}",
        str(src),
        "-o",
        str(exe),
        "-lcublas",
    ]
    rc, out = run(cmd)
    if rc != 0:
        return {"ok": False, "stage": "compile", "error": out[-4000:]}
    rc, out = run([str(exe)])
    if rc != 0:
        return {"ok": False, "stage": "run", "error": out[-4000:]}
    passed = out.count("Disposition: Passed")
    failed = out.count("Disposition: Failed")
    gflops = [float(x) for x in re.findall(r"GFLOPs:\s*([\d.]+)", out)]
    return {
        "ok": failed == 0 and passed > 0,
        "example": src.name,
        "checks_passed": passed,
        "checks_failed": failed,
        "best_tflops": round(max(gflops) / 1000, 2) if gflops else None,
        "stdout_tail": out[-1500:],
    }


def canonical_hash(obj: dict) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0, help="사용할 GPU 인덱스")
    ap.add_argument("--cutlass", default=None, help="CUTLASS 저장소 경로")
    ap.add_argument("--lock-mhz", type=int, default=None, help="고정할 SM 클럭")
    ap.add_argument("--seed", type=int, default=None, help="측정 순서 셔플 시드")
    ap.add_argument("--skip-example", action="store_true")
    args = ap.parse_args()

    # 물리 GPU 를 이 프로세스와 모든 자식(nvcc 산출물 포함)에 고정한다.
    # 이후 CUDA 관점의 device 0 == 물리 args.device.
    smi = gpu_smi_info(args.device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    print("=" * 72)
    print("Phase 0: 환경 점검 및 하드웨어 감지")
    print("=" * 72)

    host = host_info()
    cuda = cuda_info()
    cutlass = cutlass_info(args.cutlass)
    nvml = pynvml_info()

    print(f"\n[host] {host['hostname']}  {host['os']}")
    print(f"       CPU {host['cpu_count']} cores, "
          f"RAM {host['ram_available_gb']}/{host['ram_total_gb']} GB available")
    print(f"[cuda] nvcc {cuda['nvcc_release']} (V{cuda['nvcc_version']}), "
          f"driver {cuda['driver_version']}, driver API {cuda['cuda_driver_api_version']}")
    print(f"[cutlass] {cutlass['dir']}")
    print(f"          version {cutlass['version']}  commit {cutlass['commit']}")
    print(f"          2.x GEMM API deprecated? "
          f"{'YES' if cutlass['gemm_2x_api_deprecated'] else 'no'}")
    print(f"[gpu {args.device}] {smi.get('name')}  {smi.get('pci.bus_id')}")
    print(f"          ECC={smi.get('ecc.mode.current')}  "
          f"persistence={smi.get('persistence_mode')}  "
          f"compute_mode={smi.get('compute_mode')}")
    print(f"[pynvml] {nvml}")

    if not nvml.get("available"):
        print("\n!! pynvml 이 없다. `pip install nvidia-ml-py` 후 다시 실행하라.")
        return 2

    # --- 하드웨어 감지 -----------------------------------------------------
    hw = detect_hardware(0)
    extra = extra_device_info(0)
    arch_flag = nvcc_arch_flag(hw.arch)

    print("\n--- detect_hardware() ---")
    for k, v in hardware_to_dict(hw).items():
        print(f"  {k:22s} {v}")
    print(f"  {'nvcc -arch':22s} {arch_flag}")
    print(f"  {'uuid':22s} {extra['uuid']}")
    print(f"  {'smem_per_block(default)':22s} {extra['smem_per_block_default']} "
          f"(opt-in 이 아닌 정적 한도. CUTLASS 는 opt-in 을 쓴다)")

    # --- CUTLASS 예제 sanity check -----------------------------------------
    if args.skip_example:
        ex = {"ok": None, "skipped": True}
    else:
        print("\n--- CUTLASS Ampere fp16 예제 빌드/실행 ---")
        ex = sanity_check_example(Path(cutlass["dir"]), arch_flag)
        if not ex["ok"]:
            print("!! 예제 검증 실패. 여기서 중단한다.")
            print(ex.get("error", "")[:3000])
            return 3
        print(f"  {ex['example']}: {ex['checks_passed']} passed / "
              f"{ex['checks_failed']} failed, best {ex['best_tflops']} TFLOP/s")

    # --- 클럭 고정 ---------------------------------------------------------
    print("\n--- 클럭 고정 시도 ---")
    lock = try_lock_clocks(args.device, args.lock_mhz)
    drift_s = drift_check_seconds(lock.locked)
    if lock.locked:
        print(f"  OK: {lock.mhz} MHz 로 고정. 드리프트 점검 주기 {drift_s}s")
    else:
        print(f"  !! 실패 (target {lock.target_mhz} MHz): {lock.error}")
        print(f"  -> clock_locked=false 로 기록. 드리프트 점검 주기를 "
              f"{drift_s}s 로 단축하고 측정은 계속한다.")

    # --- 런치 오버헤드 -----------------------------------------------------
    print("\n--- 빈 커널 런치 오버헤드 ---")
    lo = measure_launch_overhead(arch_flag)
    for k, v in lo.items():
        if k.endswith("_ms"):
            print(f"  {k:28s} {v * 1000:8.3f} us")
    launch_overhead_ms = lo["launch_bracketed_grid_ms"]
    print(f"  -> 기준값(launch_overhead_ms) = {launch_overhead_ms * 1000:.3f} us; "
          f"time_ms < {3 * launch_overhead_ms * 1000:.3f} us 면 below_launch_overhead")

    # --- env.json ----------------------------------------------------------
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "big")
    env = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "phase": "0",
        "host": host,
        "cuda": cuda,
        "cutlass": cutlass,
        "gpu_smi": smi,
        "device_index": args.device,
        "hardware": hardware_to_dict(hw),
        "hardware_extra": extra,
        "nvcc_arch_flag": arch_flag,
        "pynvml": nvml,
        "cutlass_example_check": ex,
        "clock_locked": lock.locked,
        "locked_mhz": lock.mhz,
        "clock_lock_target_mhz": lock.target_mhz,
        "clock_lock_error": lock.error,
        "drift_check_seconds": drift_s,
        "launch_overhead": lo,
        "launch_overhead_ms": launch_overhead_ms,
        "below_launch_overhead_ms": 3 * launch_overhead_ms,
        "shuffle_seed": seed,
    }
    env["env_hash"] = canonical_hash(env)

    paths.ensure_dirs()
    paths.ENV_JSON.write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[env.json] {paths.ENV_JSON}  (env_hash={env['env_hash'][:16]}...)")
    print(f"[shuffle_seed] {seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
