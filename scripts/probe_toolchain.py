#!/usr/bin/env python3
"""툴체인(nvcc + CUTLASS)이 이 백엔드로 커널을 만들 수 있는가 — 표본 빌드.

    python3 scripts/probe_toolchain.py --sample 3                 # P0 게이트
    python3 scripts/probe_toolchain.py --sample 200 --jobs 16 \
        --cutlass /opt/cutlass --baseline results/kernels.jsonl   # 건강성

## 왜 헤더 존재 확인으로는 부족한가

우리 sm_86 백엔드는 CUTLASS **2.x API**(`device::GemmUniversal`,
ThreadblockShape/WarpShape)에 전적으로 의존한다. 헤더가 남아 있어도

* 템플릿 인자 순서나 기본값이 바뀌면 **컴파일이 깨지고**,
* 컴파일이 되어도 `static_assert` 가 조용히 다른 경로를 고르거나
  ptxas 결과(레지스터·스필)가 달라져 **같은 config 가 다른 커널이 된다.**

두 번째가 더 위험하다. 아무 오류 없이 "실행은 되는데 표가 다른" 상태가 된다.
그래서 이 스크립트는 빌드 성공 여부만이 아니라 **레지스터·스필·HMMA·smem 을
기준선과 대조**한다.

## 무엇을 보나

| | 무엇 | 왜 |
|---|---|---|
| `build_status` | 컴파일 성공률 | 2.x API 가 살아 있는가 |
| `spill_*` | 스필 분포 | ptxas 회귀의 가장 민감한 지표 |
| `regs_per_thread` | 레지스터 | 점유율이 바뀌면 성능이 바뀐다 |
| `hmma_count` vs `expected_hmma` | SASS 명령 수 | 다른 mma 경로를 골랐는가 |
| `res_shared` vs `smem_computed` | smem | 레이아웃 가정이 유효한가 |

⚠️ **시간은 재지 않는다.** 이것은 툴체인 검사이지 성능 비교가 아니다.
조건(CUDA 버전)이 다르면 절대 시간을 비교하면 안 된다.

## 산출물

`--out` 에 JSON 하나. `results/kernels.jsonl` 은 **건드리지 않는다** —
표본이고 조건도 다르므로 캠페인 기록에 섞이면 안 된다.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kerneltab.backends import get_backend
from kerneltab.build import paths
from kerneltab.build.compile import BuildEnv, build_kernel
from kerneltab.core.config import alignment_combos, enumerate_kernels
from kerneltab.core.hardware import hardware_from_env
from kerneltab.core.shapes import all_shapes

#: 대조에서 "달라졌다" 로 볼 레지스터 차이. ptxas 는 마이너 버전 사이에서도
#: 1~2개는 흔들리므로 그것까지 회귀로 부르면 신호가 묻힌다.
REG_TOL = 2


def nvcc_version(nvcc: str) -> str:
    r = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "release" in line:
            # "Cuda compilation tools, release 12.4, V12.4.99"
            return line.split(",")[-1].strip().lstrip("V")
    return "unknown"


def cutlass_info(d: Path) -> dict:
    v = {}
    h = d / "include" / "cutlass" / "version.h"
    if h.exists():
        txt = h.read_text()
        for k in ("MAJOR", "MINOR", "PATCH"):
            for line in txt.splitlines():
                if line.startswith(f"#define CUTLASS_{k}"):
                    v[k.lower()] = line.split()[-1]
    # git 이 없을 수 있다 (컨테이너). 커밋을 못 읽는 것은 치명적이지 않지만
    # **조용히 넘기지는 않는다** — 어떤 CUTLASS 였는지가 재현성의 핵심이다.
    try:
        r = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"],
                           capture_output=True, text=True)
        commit = r.stdout.strip() if r.returncode == 0 else None
        commit_source = "git" if r.returncode == 0 else "git-failed"
    except FileNotFoundError:
        commit = os.environ.get("CUTLASS_COMMIT") or None
        commit_source = "env" if commit else "unavailable(git 없음)"
    # 2.x device API 가 실제로 있는가. 없으면 빌드는 어차피 다 실패한다.
    api = d / "include" / "cutlass" / "gemm" / "device" / "gemm_universal.h"
    return {
        "dir": str(d),
        "version": ".".join(v.get(k, "?") for k in ("major", "minor", "patch")),
        "commit": commit,
        "commit_source": commit_source,
        "gemm_universal_h": api.exists(),
    }


def sample_configs(env: dict, n: int, seed: int) -> list:
    """전수 열거에서 **결정론적으로** 표본을 뽑는다.

    무작위로 뽑되 시드를 고정한다. 앞에서부터 n 개를 쓰면 열거 순서상 특정
    alignment·타일에 몰려 스필 분포가 왜곡된다.
    """
    hw = hardware_from_env(env)
    be = get_backend(hw.arch)
    combos = alignment_combos(all_shapes(hw))
    ks = enumerate_kernels(hw, be, combos)
    rng = random.Random(seed)
    return hw, be, (ks if n >= len(ks) else rng.sample(ks, n)), len(ks)


def compare(rows: list[dict], baseline_path: Path) -> dict:
    """같은 kernel_id 를 기준선과 대조한다."""
    if not baseline_path or not baseline_path.exists():
        return {"available": False,
                "reason": f"{baseline_path} 가 없다"}
    base = {}
    for line in baseline_path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("build_status") == "ok":
            base[r["kernel_id"]] = r
    common = [r for r in rows if r["kernel_id"] in base]
    if not common:
        return {"available": False, "reason": "겹치는 kernel_id 가 없다"}

    reg_moved, spill_moved, hmma_moved, fail_now = [], [], [], []
    for r in common:
        b = base[r["kernel_id"]]
        if r.get("build_status") != "ok":
            fail_now.append({"kernel_id": r["kernel_id"],
                             "error": r.get("build_error")})
            continue
        dr = (r.get("regs_per_thread") or 0) - (b.get("regs_per_thread") or 0)
        if abs(dr) > REG_TOL:
            reg_moved.append({"kernel_id": r["kernel_id"],
                              "was": b.get("regs_per_thread"),
                              "now": r.get("regs_per_thread")})
        sb = bool(b.get("spill_stores") or b.get("spill_loads"))
        sn = bool(r.get("spill_stores") or r.get("spill_loads"))
        if sb != sn:
            spill_moved.append({"kernel_id": r["kernel_id"],
                                "was": sb, "now": sn,
                                "bytes_now": r.get("spill_stores")})
        if r.get("hmma_count") != b.get("hmma_count"):
            hmma_moved.append({"kernel_id": r["kernel_id"],
                               "was": b.get("hmma_count"),
                               "now": r.get("hmma_count")})
    return {
        "available": True,
        "baseline": str(baseline_path),
        "n_common": len(common),
        "build_fail_now_ok_before": fail_now,
        "regs_changed": reg_moved,
        "spill_changed": spill_moved,
        "hmma_changed": hmma_moved,
        "reg_tolerance": REG_TOL,
    }


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("build_status") == "ok"]
    fail = [r for r in rows if r.get("build_status") != "ok"]
    regs = sorted(r["regs_per_thread"] for r in ok if r.get("regs_per_thread"))
    spilled = [r for r in ok if (r.get("spill_stores") or 0)
               or (r.get("spill_loads") or 0)]
    hmma_bad = [r["kernel_id"] for r in ok
                if r.get("hmma_count") != r.get("expected_hmma")]
    # ⚠️ `res_shared`(cuobjdump -res-usage)가 아니라 `smem_dynamic` 이다.
    #    CUTLASS 커널은 smem 을 **동적으로** 잡으므로 정적 smem 은 0 이다.
    #    res_shared 로 대조하면 전부 불일치로 나온다.
    #    `smem_dynamic` 은 introspect() 가 cudaFuncGetAttributes 로 얻는다 —
    #    CUDA 컨텍스트가 필요하므로 GPU 가 없으면 결측이다.
    smem_bad = [r["kernel_id"] for r in ok
                if r.get("smem_dynamic") is not None
                and r["smem_dynamic"] != r.get("smem_computed")]
    smem_unknown = sum(1 for r in ok if r.get("smem_dynamic") is None)

    def q(p):
        return regs[min(len(regs) - 1, int(len(regs) * p))] if regs else None

    return {
        "n": len(rows),
        "ok": len(ok),
        "build_fail": len(fail),
        "fail_rate": round(len(fail) / max(len(rows), 1), 4),
        "spill_rate": round(len(spilled) / max(len(ok), 1), 4),
        "regs_p50": q(0.5), "regs_p90": q(0.9),
        "regs_max": regs[-1] if regs else None,
        "hmma_mismatch": len(hmma_bad),
        "hmma_mismatch_ids": hmma_bad[:10],
        "smem_mismatch": len(smem_bad),
        "smem_mismatch_ids": smem_bad[:10],
        # 조용히 0/0 이 되지 않도록 **검사하지 못한 개수**를 함께 낸다.
        "smem_unchecked": smem_unknown,
        "build_errors": Counter(
            (r.get("build_error") or "")[:120] for r in fail).most_common(5),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sample", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--cutlass", default=None,
                    help="CUTLASS 저장소. 생략하면 env.json 의 값")
    ap.add_argument("--nvcc", default=None,
                    help="nvcc 경로. 생략하면 env.json 의 값")
    ap.add_argument("--workdir", default=None,
                    help="생성 .cu / .so 를 둘 곳. 캠페인 산출물과 섞지 "
                         "않도록 기본은 임시 디렉토리다")
    ap.add_argument("--baseline", default=str(paths.RESULTS_DIR / "kernels.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep", action="store_true", help="workdir 를 지우지 않는다")
    args = ap.parse_args()

    if not paths.ENV_JSON.exists():
        print(f"{paths.ENV_JSON} 이 없다. scripts/phase0_env.py 를 먼저 돌려라.")
        return 2
    env = json.loads(paths.ENV_JSON.read_text())

    hw, be_backend, cfgs, n_total = sample_configs(env, args.sample, args.seed)

    cutlass = Path(args.cutlass) if args.cutlass else Path(env["cutlass"]["dir"])
    nvcc = args.nvcc or env["cuda"]["nvcc_path"]
    if not Path(nvcc).exists() and not shutil.which(nvcc):
        print(f"nvcc 를 찾을 수 없다: {nvcc}")
        return 2
    ci = cutlass_info(cutlass)
    if not ci["gemm_universal_h"]:
        print("⛔ include/cutlass/gemm/device/gemm_universal.h 가 없다 — "
              "CUTLASS 2.x device API 가 제거됐다.")
        return 3

    work = Path(args.workdir) if args.workdir else Path(
        os.environ.get("TMPDIR", "/tmp")) / f"kt_probe_{os.getpid()}"
    src, lib = work / "src", work / "lib"
    src.mkdir(parents=True, exist_ok=True)
    lib.mkdir(parents=True, exist_ok=True)
    benv = BuildEnv(
        nvcc=str(nvcc), arch_flag=env["nvcc_arch_flag"], cutlass_dir=cutlass,
        src_dir=src, lib_dir=lib,
        include_args=(*tuple(paths.cutlass_includes(cutlass)),
                      f"-I{paths.REPO_ROOT / 'measure'}"),
    )

    nv = nvcc_version(str(nvcc))
    print(f"nvcc      {nv}  ({nvcc})")
    print(f"CUTLASS   {ci['version']}  {(ci['commit'] or '?')[:12]}"
          f" ({ci['commit_source']})  {cutlass}")
    print(f"arch      {benv.arch_flag}")
    print(f"표본      {len(cfgs)} / 전체 {n_total:,}  (seed={args.seed})")
    print(f"작업 경로 {work}\n")

    rows, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(build_kernel, c, be_backend, benv): c for c in cfgs}
        for i, f in enumerate(as_completed(futs), 1):
            cfg = futs[f]
            try:
                r = f.result()
            except Exception as e:
                # 빌드 워커가 죽어도 표본 전체를 잃지 않는다. 무엇이 죽었는지
                # 남긴다 — 조용히 빠지면 성공률이 부풀려진다.
                r = {"kernel_id": be_backend.kernel_id(cfg),
                     "build_status": "probe_error", "build_error": repr(e)}
            r["ext"] = asdict(cfg.ext) if hasattr(cfg.ext, "__dataclass_fields__") \
                else dict(cfg.ext or {})
            r["tile"] = {"m": cfg.tile_m, "n": cfg.tile_n, "k": cfg.tile_k}
            r["align"] = {"a": cfg.align_a, "b": cfg.align_b, "c": cfg.align_c}
            r["smem_computed"] = be_backend.smem_bytes(cfg, 2)
            r["expected_hmma"] = be_backend.expected_hmma(cfg, 2)
            rows.append(r)
            mark = "." if r.get("build_status") == "ok" else "F"
            print(mark, end="", flush=True)
            if i % 50 == 0:
                print(f" {i}/{len(cfgs)}", flush=True)
    print(f"\n\n{time.time() - t0:.1f}초\n")

    out = {
        "probe": "toolchain",
        "nvcc_version": nv,
        "nvcc_path": str(nvcc),
        "cutlass": ci,
        "arch_flag": benv.arch_flag,
        "gpu_name": hw.name,
        "sample": len(cfgs), "n_total": n_total, "seed": args.seed,
        "summary": summarize(rows),
        "compare": compare(rows, Path(args.baseline) if args.baseline else None),
        "rows": rows,
    }
    s = out["summary"]
    print(f"빌드 성공 {s['ok']}/{s['n']}   실패율 {100 * s['fail_rate']:.1f}%")
    print(f"스필      {100 * s['spill_rate']:.1f}%")
    print(f"레지스터  p50={s['regs_p50']} p90={s['regs_p90']} max={s['regs_max']}")
    print(f"HMMA 불일치 {s['hmma_mismatch']}   smem 불일치 {s['smem_mismatch']}"
          + (f"   (smem 미검사 {s['smem_unchecked']} — GPU 컨텍스트 없음)"
             if s["smem_unchecked"] else ""))
    for err, n in s["build_errors"]:
        print(f"  실패 {n}회: {err}")
    c = out["compare"]
    if c.get("available"):
        print(f"\n기준선 대조 ({c['n_common']}개 공통, {c['baseline']})")
        print(f"  이전 ok -> 지금 실패 : {len(c['build_fail_now_ok_before'])}")
        print(f"  레지스터 변화 (>{REG_TOL}) : {len(c['regs_changed'])}")
        print(f"  스필 유무 변화        : {len(c['spill_changed'])}")
        print(f"  HMMA 수 변화          : {len(c['hmma_changed'])}")
        for k in ("build_fail_now_ok_before", "regs_changed", "spill_changed",
                  "hmma_changed"):
            for x in c[k][:5]:
                print(f"    {k}: {x}")
    else:
        print(f"\n기준선 대조 불가: {c.get('reason')}")

    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\n-> {args.out}")
    if not args.keep and not args.workdir:
        shutil.rmtree(work, ignore_errors=True)

    # 종료 코드: 0 통과 / 4 빌드 실패 있음 / 5 기준선 대비 회귀
    if s["build_fail"]:
        return 4
    if c.get("available") and (c["regs_changed"] or c["spill_changed"]
                               or c["hmma_changed"]):
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
