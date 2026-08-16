#!/usr/bin/env python3
"""results.jsonl + kernels.jsonl -> results/table.parquet

원본에서 언제든 재생성 가능해야 한다. 그래서
  - 파생 지표는 JSONL 에 저장하지 않고 **여기서** 계산한다.
    (계산식에 버그가 나도 수십 시간짜리 측정을 다시 하지 않는다)
  - ext 는 `ext_` 접두어로 평탄화한다. 다른 아키텍처 데이터를 같은 테이블로
    합칠 때 해당 없는 컬럼은 null 이 되고, Parquet 은 null 컬럼을 효율적으로
    저장하므로 문제없다.
  - CSV 는 만들지 않는다 (telemetry 제외).

    python3 scripts/export.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backends import get_backend  # noqa: E402
from build import paths  # noqa: E402
from core import features as F  # noqa: E402
from core.types import Hardware, KernelConfig, Problem, RuntimeConfig  # noqa: E402

RESULTS = paths.RESULTS_DIR / "results.jsonl"
KERNELS = paths.RESULTS_DIR / "kernels.jsonl"
OUT = paths.RESULTS_DIR / "table.parquet"

#: kernels.jsonl 에서 그대로 가져오는 컬럼
KERNEL_COLS = [
    "arch", "regs_per_thread", "smem_static_bytes", "smem_dynamic",
    "smem_computed", "spill_stores", "spill_loads", "local_bytes",
    "hmma_count", "expected_hmma", "lds_count", "sts_count", "ldsm_count",
    "ldg_count", "cpasync_count", "inst_total", "threads",
    "max_blocks_per_sm", "cutlass_max_blocks", "pipeline_kind",
    "build_seconds", "cutlass_commit", "nvcc_arch", "res_regs", "res_local",
]


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    env = json.loads(paths.ENV_JSON.read_text())
    hw_spec = Hardware(**env["hardware"])
    # 클럭을 고정하면 SM 클럭만 내려가고 메모리 클럭은 그대로다. roofline 을
    # 스펙 피크로 계산하면 ridge point 가 실제보다 높게 나와 "이 형상이 메모리
    # 바운드인가" 판정이 틀린다. 분석에는 실효 피크를 쓴다.
    peak_eff = env.get("peak_tflops_f16_effective") or hw_spec.peak_tflops_f16
    hw = replace(hw_spec, peak_tflops_f16=peak_eff)
    backend = get_backend(hw.arch)
    if peak_eff != hw_spec.peak_tflops_f16:
        print(f"[roofline] 실효 피크 {peak_eff} TFLOP/s "
              f"(스펙 {hw_spec.peak_tflops_f16} @ {env.get('peak_tflops_f16_at_mhz')} MHz, "
              f"고정 {env.get('locked_mhz')} MHz)")

    kern = {r["kernel_id"]: r for r in load(KERNELS)}
    res = load(RESULTS)
    cublas = {(r["problem"]["M"], r["problem"]["N"], r["problem"]["K"]):
              r.get("time_ms") for r in res if r["kernel_id"] == "cublas"}

    rows = []
    for r in res:
        if r["kernel_id"] == "cublas":
            continue
        k = kern.get(r["kernel_id"])
        if k is None:
            continue
        pr, rt = r["problem"], r["runtime"]
        p = Problem(pr["M"], pr["N"], pr["K"], pr.get("dtype", "f16"))
        rc = RuntimeConfig(rt["split_k"], rt["split_k_mode"])
        cfg = KernelConfig(
            tile_m=k["tile"]["m"], tile_n=k["tile"]["n"], tile_k=k["tile"]["k"],
            align_a=k["align"]["a"], align_b=k["align"]["b"],
            align_c=k["align"]["c"], arch=k["arch"],
            ext=backend.ext_from_dict(k["ext"]),
        )

        row: dict = {
            "kernel_id": r["kernel_id"],
            "M": p.M, "N": p.N, "K": p.K, "dtype": p.dtype,
            "acc_dtype": p.acc_dtype,
            "layout_a": p.layout_a, "layout_b": p.layout_b, "layout_c": p.layout_c,
            "split_k": rc.split_k, "split_k_mode": rc.split_k_mode,
            "tile_m": cfg.tile_m, "tile_n": cfg.tile_n, "tile_k": cfg.tile_k,
            "align_a": cfg.align_a, "align_b": cfg.align_b, "align_c": cfg.align_c,
        }
        # ★ 아키텍처 전용 필드는 ext_ 접두어로 평탄화
        for kk, vv in k["ext"].items():
            row[f"ext_{kk}"] = vv
        for c in KERNEL_COLS:
            row[c] = k.get(c)

        # 측정값
        for c in ("time_ms", "time_std_ms", "time_min_ms", "time_max_ms",
                  "n_reps", "outlier_frac", "max_rel_error", "workspace_bytes",
                  "actual_split_k", "workspace_dtype", "partials_dtype",
                  "status", "sm_clock_mhz", "gpu_temp_c",
                  "power_w", "clock_locked", "env_hash", "timestamp", "error"):
            row[c] = r.get(c)
        row["cublas_ms"] = cublas.get((p.M, p.N, p.K))

        # --- 파생 지표 (여기서 계산, JSONL 에는 없다) ---------------------
        bps = k.get("max_blocks_per_sm") or 1
        row["grid_tiles"] = F.grid_tiles(p, cfg, rc)
        row["waves"] = F.waves(p, hw, cfg, rc)
        row["waves_occ"] = F.waves(p, hw, cfg, rc, blocks_per_sm=bps)
        row["tail_waste"] = F.tail_waste(p, hw, cfg, rc)
        row["tail_waste_occ"] = F.tail_waste(p, hw, cfg, rc, blocks_per_sm=bps)
        row["mainloop_iters"] = F.mainloop_iters(p, cfg, rc)
        row["tail_m_frac"] = F.tail_m_frac(p, cfg)
        row["tail_n_frac"] = F.tail_n_frac(p, cfg)
        row["arith_intensity"] = F.arith_intensity(p)
        row["ridge_point"] = F.ridge_point(hw)
        row["ridge_point_spec"] = F.ridge_point(hw_spec)
        row["peak_tflops_used"] = peak_eff
        row["locked_mhz"] = env.get("locked_mhz")
        row["is_memory_bound"] = F.is_memory_bound(p, hw)
        row["flops"] = F.flops(p)
        row["bytes_moved"] = F.bytes_moved(p)

        # 자원 파생
        thr = k.get("threads") or 0
        regs = k.get("regs_per_thread") or 0
        row["theoretical_occupancy"] = (
            (k.get("max_blocks_per_sm") or 0) * thr / hw.max_threads_per_sm
            if thr else None)
        row["regs_total_per_block"] = regs * thr
        row["launchable"] = (regs * thr <= hw.regs_per_sm) if regs and thr else None
        row["has_spill"] = ((k.get("spill_stores") or 0)
                            + (k.get("spill_loads") or 0)) > 0
        row["smem_matches"] = k.get("smem_dynamic") == k.get("smem_computed")
        row["hmma_matches"] = k.get("hmma_count") == k.get("expected_hmma")

        # 성능 파생
        t = r.get("time_ms")
        if t:
            row["tflops"] = F.flops(p) / (t * 1e-3) / 1e12
            row["frac_of_peak"] = row["tflops"] / hw.peak_tflops_f16
            cb = row["cublas_ms"]
            row["vs_cublas"] = (cb / t) if cb else None
        else:
            row["tflops"] = row["frac_of_peak"] = row["vs_cublas"] = None

        rows.append(row)

    if not rows:
        print("내보낼 측정 결과가 없다.")
        return 1

    cols = list({c for r in rows for c in r})
    table = pa.table({c: [r.get(c) for r in rows] for c in sorted(cols)})
    pq.write_table(table, args.out, compression="zstd")
    print(f"{args.out}  {len(rows)}행 x {len(cols)}열  "
          f"({Path(args.out).stat().st_size / 1024:.1f} KB)")
    print("\n컬럼:")
    for c in sorted(cols):
        print(f"  {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
