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
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build import paths
from core.env_hash import env_hash_v2
from core.hardware import (
    bandwidth_from_api,
    bandwidth_reference_mhz,
    detect_hardware,
    extra_device_info,
    hardware_to_dict,
    known_spec,
    nvcc_arch_flag,
    peak_reference_mhz,
)
from measure.gpu_state import (
    ClockLockResult,
    drift_check_seconds,
    try_lock_clocks,
)
from measure.runner import (
    PROTOCOL_DEFAULTS,
    SEGMENT_DEFAULTS,
    SOAK_DEFAULTS,
)


def _query_sm_clock(index: int) -> int | None:
    rc, out = run(["nvidia-smi", "-i", str(index),
                   "--query-gpu=clocks.current.sm",
                   "--format=csv,noheader,nounits"])
    try:
        return int(float(out.splitlines()[0].strip()))
    except Exception:
        return None


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


def cutlass_info(explicit: str | None, commit_override: str | None = None) -> dict:
    """CUTLASS 커밋/버전. `.git` 이 없으면 `commit_override` 로 주입한다 (수정 11).

    컨테이너에서는 특정 커밋으로 얕게 clone 하거나 tarball 을 풀어 쓰므로
    `.git` 이 없을 수 있다. 그러면 `commit` 이 `None` 이 되는데,
    **`cutlass.commit` 은 `env_hash_v2` 의 키다**(P-3) — `None` 으로 두면
    서로 다른 CUTLASS 버전이 같은 해시를 받는다.

    `--cutlass-commit` 으로 주입하고, 주입했다는 사실을 기록에 남긴다.
    """
    root = paths.cutlass_dir(explicit)
    rc, commit = run(["git", "-C", str(root), "rev-parse", "HEAD"])
    # 추적 파일의 수정과 미추적 파일을 **구분한다.** 빌드에 영향을 주는
    # 것은 추적 파일 쪽이다. 실측 캠페인의 env.json 에는
    # worktree_dirty=true 가 찍혀 있는데, 실제로는 관계 없는 디렉토리가
    # 하나 놓여 있었을 뿐이었다. 재현성 신호가 상시 켜져 있으면 아무도
    # 안 본다 — R-1 이 잡은 병과 같다.
    rc2, dirty = run(["git", "-C", str(root), "status", "--porcelain",
                      "--untracked-files=no"])
    _rc2u, untracked = run(["git", "-C", str(root), "ls-files",
                            "--others", "--exclude-standard"])
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
            if re.search(r"deprecat", line, re.IGNORECASE):
                dep_hits.append(f"{rel}:{i}: {line.strip()[:120]}")

    detected = commit.strip() if rc == 0 else None
    if not detected and commit_override:
        detected = commit_override.strip()
    return {
        "dir": str(root),
        "commit": detected,
        # 어떻게 알아냈는지 남긴다. 주입한 값을 git 에서 읽은 것처럼
        # 보이게 하면 나중에 신뢰도를 판단할 수 없다.
        "commit_source": ("git" if rc == 0 else
                          ("injected" if commit_override else "unknown")),
        "describe": desc.strip() if rc3 == 0 else None,
        "version": ver,
        # 빌드에 영향을 주는 것 = 추적 파일의 수정. 이것이 true 면 commit
        # 만으로는 재현할 수 없다.
        "worktree_dirty": bool(dirty.strip()) if rc2 == 0 else None,
        "worktree_modified": [x[3:] for x in dirty.splitlines()][:20]
        if rc2 == 0 else None,
        # 미추적 파일은 빌드에 안 들어간다. 참고로만 센다.
        "untracked_files": len(untracked.splitlines()) if _rc2u == 0 else None,
        "gemm_2x_api_deprecation_hits": dep_hits,
        "gemm_2x_api_deprecated": bool(dep_hits),
    }


def clock_state(index: int) -> dict:
    """SM / 메모리 클럭을 둘 다 기록한다.

    -lgc 는 SM 클럭만 고정하고 메모리 클럭은 건드리지 않는다. roofline 의
    분모(대역폭)는 메모리 클럭에 달려 있으므로 두 값이 모두 있어야 나중에
    "이 측정이 어떤 클럭 조건이었나" 를 재구성할 수 있다.
    """
    fields = ["clocks.current.sm", "clocks.current.memory",
              "clocks.max.sm", "clocks.max.memory",
              "clocks.default_applications.graphics",
              "clocks.applications.graphics"]
    rc, out = run(["nvidia-smi", "-i", str(index),
                   f"--query-gpu={','.join(fields)}",
                   "--format=csv,noheader,nounits"])
    if rc != 0:
        return {"error": out}
    vals = [v.strip() for v in out.splitlines()[0].split(",")]
    d = {}
    for f, v in zip(fields, vals):
        try:
            d[f.replace("clocks.", "").replace(".", "_") + "_mhz"] = int(float(v))
        except ValueError:
            d[f.replace("clocks.", "").replace(".", "_") + "_mhz"] = None
    d["sm_clock_mhz"] = d.get("current_sm_mhz")
    d["mem_clock_mhz"] = d.get("current_memory_mhz")
    return d


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
        except md.PackageNotFoundError:
            # 소스에서 직접 쓰는 경우 등. 버전 기록만 빠지고 측정에는
            # 영향이 없다 — manifest 가 별도로 패키지 목록을 남긴다.
            d["package_version"] = None
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


def _safe_manifest() -> dict | None:
    """`scripts/manifest.py` 로 코드/의존성 버전을 모은다.

    실패해도 측정을 막지 않는다 — 기록용이기 때문이다. 다만 **조용히
    비우지 않고** 이유를 남긴다 (`docs/decisions.md` 14번).
    """
    try:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from manifest import build as _build
        return _build()
    except Exception as e:
        print(f"  [경고] manifest 수집 실패: {e!r} — env.json 에 이유만 남긴다")
        return {"error": repr(e)}


def canonical_hash(obj: dict) -> str:
    """구 정의 — `env` 전체를 해싱한다.

    ⚠️ 실행마다 변하는 값이 섞여 있어 **조건이 같아도 값이 달라진다.**
    기존 데이터의 조회 키라 유지할 뿐이고, 새 판정에는 `env_hash_v2` 를
    쓴다 (`core/env_hash.py`).
    """
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0, help="사용할 GPU 인덱스")
    ap.add_argument("--cutlass", default=None, help="CUTLASS 저장소 경로")
    ap.add_argument("--cutlass-commit", default=None,
                    help="CUTLASS 에 .git 이 없을 때 커밋 해시를 주입한다 "
                         "(컨테이너에서 얕은 clone/tarball 을 쓰는 경우). "
                         "cutlass.commit 은 env_hash_v2 의 키라 비워 두면 "
                         "서로 다른 버전이 같은 해시를 받는다")
    ap.add_argument("--lock-mhz", type=int, default=None, help="고정할 SM 클럭")
    ap.add_argument("--externally-locked-mem-mhz", type=int, default=None,
                    help="관리자가 nvidia-smi -lmc 로 메모리 클럭을 고정한 경우. "
                         "요청값이 아니라 **부하 중 실제 관측값**을 넣을 것 "
                         "(컴퓨트 P2 상태에서는 P0 최대치보다 낮다).")
    ap.add_argument("--externally-locked-mhz", type=int, default=None,
                    help="관리자가 이미 nvidia-smi -lgc 로 고정해 둔 경우 그 값. "
                         "부하 테스트(scripts/verify_clock_lock.py)로 유지되는 것을 "
                         "확인한 뒤에만 쓸 것.")
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
    cutlass = cutlass_info(args.cutlass, args.cutlass_commit)
    nvml = pynvml_info()

    print(f"\n[host] {host['hostname']}  {host['os']}")
    print(f"       CPU {host['cpu_count']} cores, "
          f"RAM {host['ram_available_gb']}/{host['ram_total_gb']} GB available")
    print(f"[cuda] nvcc {cuda['nvcc_release']} (V{cuda['nvcc_version']}), "
          f"driver {cuda['driver_version']}, driver API {cuda['cuda_driver_api_version']}")
    print(f"[cutlass] {cutlass['dir']}")
    print(f"          version {cutlass['version']}  commit {cutlass['commit']}"
          f"  ({cutlass['commit_source']})")
    if cutlass["commit_source"] == "unknown":
        print("  !! CUTLASS 커밋을 알 수 없다 (.git 이 없고 주입도 없다).")
        print("     cutlass.commit 은 env_hash_v2 의 키라, 비워 두면 서로")
        print("     다른 CUTLASS 버전이 **같은 해시**를 받는다.")
        print("     --cutlass-commit <해시> 로 주입하라.")
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
    print("\n--- 클럭 고정 ---")
    if args.externally_locked_mhz:
        # 이 프로세스에 권한이 없어도 관리자가 이미 고정해 둔 경우가 있다.
        # 아이들 클럭만 보고 판단하면 안 되므로 부하 검증 결과를 함께 기록한다.
        cur = _query_sm_clock(args.device)
        lock = ClockLockResult(True, args.externally_locked_mhz,
                               args.externally_locked_mhz, None)
        print(f"  외부에서 고정됨: {args.externally_locked_mhz} MHz "
              f"(현재 판독 {cur} MHz)")
    else:
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
    # 클럭을 고정하면 SM 클럭만 내려가고 메모리 클럭은 그대로다. 스펙 피크를
    # 그대로 쓰면 roofline ridge point 가 실제보다 높게 나와 어떤 형상이
    # 메모리 바운드인지 판정이 틀린다. 실효 피크를 함께 기록한다.
    peak_ref = peak_reference_mhz(hw.name)
    peak_eff = hw.peak_tflops_f16
    if lock.locked and lock.mhz and peak_ref:
        peak_eff = round(hw.peak_tflops_f16 * lock.mhz / peak_ref, 3)
        print("\n--- 실효 피크 (클럭 고정 보정) ---")
        print(f"  스펙 {hw.peak_tflops_f16} TFLOP/s @ {peak_ref} MHz")
        print(f"  실효 {peak_eff} TFLOP/s @ {lock.mhz} MHz")
        print("  (ridge point 는 대역폭 보정 후 아래에서 최종 출력)")
    clk_state = clock_state(args.device)
    print("\n--- 클럭 상태 ---")
    print(f"  SM {clk_state.get('sm_clock_mhz')} MHz "
          f"(최대 {clk_state.get('max_sm_mhz')})   "
          f"메모리 {clk_state.get('mem_clock_mhz')} MHz "
          f"(최대 {clk_state.get('max_memory_mhz')})")
    ck = paths.RESULTS_DIR / "clock_lock_check.json"
    clock_check = json.loads(ck.read_text()) if ck.exists() else None

    # --- 실효 대역폭 (메모리 클럭 보정) --------------------------------
    # -lgc 는 SM 클럭만, -lmc 는 메모리 클럭만 고정한다. 게다가 컴퓨트
    # 워크로드는 P2 상태로 내려가 메모리 클럭이 P0 최대치보다 낮다.
    # 스펙 대역폭(P0 기준)을 그대로 쓰면 roofline 의 분모가 틀린다.
    # 실효 대역폭은 **오직 하나의 경로**로 계산한다: 버스 폭 x 2 x 관측 클럭.
    # known.json 의 스펙 값은 교차검증에만 쓴다.
    spec = known_spec(hw.name)
    bus = extra.get("memory_bus_width_bits")
    mem_locked = args.externally_locked_mem_mhz
    mem_obs = (clock_check or {}).get("mem_clk_median") or clk_state.get("mem_clock_mhz")
    mem_used = mem_locked or mem_obs
    bw_eff = round(bandwidth_from_api(bus, mem_used), 2) if (bus and mem_used) else None
    bw_ref = bandwidth_reference_mhz(hw.name)
    bw_spec = spec.get("bandwidth_gbps_spec")
    checks = []
    if spec.get("mem_bus_bits") is not None:
        ok = spec["mem_bus_bits"] == bus
        checks.append(f"버스 폭 API {bus} vs known {spec['mem_bus_bits']} "
                      f"{'일치' if ok else '<-- 불일치!'}")
    if bw_spec and bw_ref:
        at_ref = bandwidth_from_api(bus, bw_ref)
        ok = abs(at_ref - bw_spec) / bw_spec < 0.02
        checks.append(f"기준 클럭 {bw_ref} MHz 에서 계산 {at_ref:.1f} vs "
                      f"스펙 {bw_spec} {'일치' if ok else '<-- 불일치!'}")
    print("\n--- 실효 대역폭 (계산 경로: 버스 폭 x 2(DDR) x 관측 클럭) ---")
    print(f"  {bus}-bit x 2 x {mem_used} MHz / 8 = {bw_eff} GB/s "
          f"({'고정' if mem_locked else '관측'})")
    for c in checks:
        print(f"  교차검증: {c}")
    print(f"  참고: 스펙 {bw_spec} GB/s 는 P0({bw_ref} MHz) 기준이며, 컴퓨트 "
          f"워크로드는 P2 로 동작하므로 도달 불가능한 값이다.")
    bw_from_api = bw_eff

    print("\n--- roofline ---")
    print(f"  스펙 기준 ridge point "
          f"{hw.peak_tflops_f16 * 1e12 / (hw.bandwidth_gbps * 1e9):.1f} FLOP/byte")
    print(f"  실효 기준 ridge point {peak_eff * 1e12 / (bw_eff * 1e9):.1f} FLOP/byte"
          f"   ({peak_eff} TFLOP/s / {bw_eff} GB/s)")

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
        "peak_tflops_f16_spec": hw.peak_tflops_f16,
        "peak_tflops_f16_at_mhz": peak_ref,
        "peak_tflops_f16_effective": peak_eff,
        "bandwidth_gbps_spec": bw_spec,
        "bandwidth_gbps_api_at_p0": hw.bandwidth_gbps,
        "mem_bus_bits": bus,
        "bandwidth_gbps_at_mem_mhz": bw_ref,
        "bandwidth_gbps_effective": bw_eff,
        "bandwidth_gbps_from_bus_width": bw_from_api,
        "mem_clock_locked": bool(mem_locked),
        "locked_mem_mhz": mem_locked,
        "mem_clock_used_mhz": mem_used,
        "clock_lock_check": clock_check,
        "clocks": clk_state,
        "sm_clock_mhz": clk_state.get("sm_clock_mhz"),
        "mem_clock_mhz": clk_state.get("mem_clock_mhz"),
        "launch_overhead": lo,
        "launch_overhead_ms": launch_overhead_ms,
        "below_launch_overhead_ms": 3 * launch_overhead_ms,
        "shuffle_seed": seed,
        "protocol": PROTOCOL_DEFAULTS,
        "soak": SOAK_DEFAULTS,
        # 드리프트 대책. 이 값이 바뀌면 측정 조건이 바뀐 것이므로 env_hash 도
        # 바뀌어야 한다 — 다른 세그먼트 크기로 잰 데이터는 섞으면 안 된다.
        "segments": SEGMENT_DEFAULTS,
        # 수정 8: 코드/CUTLASS/패키지 버전을 env.json 에 기록한다.
        # **해시 키에는 안 들어간다** (P-3) — manifest_hash 는 소스
        # tree_hash 를 포함해서 한 글자만 고쳐도 값이 바뀌고, 그러면
        # 측정 도중 오타 수정조차 못 한다. 사후 추적용으로 기록만 한다.
        "manifest": _safe_manifest(),
    }
    env["env_hash"] = canonical_hash(env)
    # P-3: 측정 조건에만 의존하는 해시. 구 해시는 실행마다 변하는 값
    # (created_utc, host.*, launch_overhead 측정값)을 포함해서, 조건이
    # 같아도 다시 돌리면 값이 바뀐다. 재개가 끊기고 같은 조건의 데이터가
    # 갈라진다 — 이 캠페인에서 실제로 겪었다.
    # 구 해시는 그대로 두어 기존 98만 줄의 조회를 깨뜨리지 않는다.
    env["env_hash_v2"] = env_hash_v2(env)

    paths.ensure_dirs()
    paths.ENV_JSON.write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[env.json] {paths.ENV_JSON}  (env_hash={env['env_hash'][:16]}...)")
    print(f"[shuffle_seed] {seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
