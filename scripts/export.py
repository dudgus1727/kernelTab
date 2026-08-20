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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kerneltab.backends import get_backend
from kerneltab.build import paths
from kerneltab.core import features as F
from kerneltab.core import kernels as kernels_mod
from kerneltab.core import records
from kerneltab.core.hardware import hardware_from_env
from kerneltab.core.types import Hardware, KernelConfig, Problem, RuntimeConfig

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
    hw = hardware_from_env(env)          # 유효 피크 적용본 (분석용)
    peak_eff = hw.peak_tflops_f16
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
                  "status", "sm_clock_mhz", "mem_clock_mhz", "gpu_temp_c",
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
        row["theoretical_occupancy"] = (
            (k.get("max_blocks_per_sm") or 0) * thr / hw.max_threads_per_sm
            if thr else None)
        row["regs_total_per_block"] = kernels_mod.regs_total_per_block(k)
        # 결측이면 None 이다 — launchable() 은 True 를 주지만, 표의 컬럼은
        # "모른다" 를 그대로 남겨야 소비 쪽이 판단할 수 있다.
        row["launchable"] = (None if row["regs_total_per_block"] is None
                             else kernels_mod.launchable(k, hw.regs_per_sm))
        row["has_spill"] = ((k.get("spill_stores") or 0)
                            + (k.get("spill_loads") or 0)) > 0
        # smem_computed 는 kernels.jsonl 에 빌드 시점 공식으로 박혀 있으므로
        # 항상 지금 공식으로 재계산한다 (append-only 파일은 고칠 수 없다).
        row["smem_computed"] = backend.smem_bytes(cfg, 2)
        row["smem_computed_at_build"] = k.get("smem_computed")
        row["smem_matches"] = k.get("smem_dynamic") == row["smem_computed"]
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

    # 조건별 집계는 core.records.aggregate_per_env 하나로 모았다 (R-5).
    _per_env = records.aggregate_per_env

    # --- 형상 난이도 -------------------------------------------------------
    # difficulty = 그 형상에서 무작위로 고른 config 가 최적 대비 몇 배 느린가
    #              (= 중앙값 시간 / 최적 시간)
    # 1.05 면 아무거나 골라도 되는 형상, 3.0 이면 선택이 결정적인 형상이다.
    #
    # ⛔ 이것은 **정답에서 유도된 값**이므로 core/table.py 의 ANSWER_COLS 에
    #    들어 있다. 규칙이 "이 형상은 어려우니 신중하게" 를 알면 안 된다.
    #    load_for_scoring 에만 나온다.
    #    여기서 미리 계산해 두는 것은 kernelrule 이 평가를 층화할 때마다
    #    표를 다시 훑지 않아도 되게 하려는 것이다.
    import statistics as _st
    # ⚠️ **env_hash 별로** 계산한다. 측정 조건이 다른 데이터를 섞으면
    #    난이도가 오염된다 — 폐기된 드리프트 구간(368a84f1)의 시간이 섞이면
    #    중앙값이 부풀어 난이도가 22배까지 나온다 (실제로 그랬다).
    _diff = _per_env(
        rows,
        key_fn=lambda r: (r["M"], r["N"], r["K"]),
        val_fn=lambda r: (r["time_ms"]
                          if r.get("status") == "ok" and r.get("time_ms")
                          else None),
        agg=lambda ts: _st.median(ts) / min(ts),
        min_n=5)          # 표본이 적으면 중앙값이 의미 없다
    # --- 형상 분해능 -------------------------------------------------------
    # distinct_time_frac = 서로 다른 시간값 수 / 후보 수
    #
    # 난이도와 **다른 축**이다.
    #   난이도 낮음          실제로 성능이 비슷하다        (물리)
    #   distinct 비율 낮음   측정이 구분을 못 한다          (계측)
    #
    # CUDA 이벤트 타이머는 양자화돼 있고(core/noise.py 의 EVENT_TICK_MS),
    # 짧은 형상에서는 여러 config 가 **같은 눈금에 떨어져 시간이 문자
    # 그대로 동일**하게 기록된다. 실측: (1,12288,4096) 은 후보 12,213 개에
    # 서로 다른 값이 **1,638 개**뿐이다 (0.134).
    #
    # 이 값이 낮으면 그 형상에서 순위를 다투는 것이 무의미하다 — 규칙을
    # 채점할 때 그 형상의 가중치를 낮추거나 따로 보고해야 한다.
    #
    # ⛔ 정답에서 유도된 값이므로 ANSWER_COLS 다. 규칙 입력에는 안 나온다.
    _distinct = _per_env(
        rows,
        key_fn=lambda r: (r["M"], r["N"], r["K"]),
        val_fn=lambda r: (round(r["time_ms"], 9)
                          if r.get("status") == "ok" and r.get("time_ms")
                          else None),
        agg=lambda ts: len(set(ts)) / len(ts),
        min_n=5)
    _ndist = _per_env(
        rows,
        key_fn=lambda r: (r["M"], r["N"], r["K"]),
        val_fn=lambda r: (round(r["time_ms"], 9)
                          if r.get("status") == "ok" and r.get("time_ms")
                          else None),
        agg=lambda ts: len(set(ts)),
        min_n=5)

    for r in rows:
        k = (str(r.get("env_hash") or ""), r["M"], r["N"], r["K"])
        r["difficulty"] = _diff.get(k)
        r["distinct_time_frac"] = _distinct.get(k)
        r["n_distinct_times"] = _ndist.get(k)
    if _diff:
        _cur = [v for k, v in _diff.items()
                if k[0] and k[0] == env.get("env_hash")]
        print(f"  난이도: (env_hash, 형상) {len(_diff)}쌍, "
              f"현재 조건 형상 {len(_cur)}개, 중앙값 "
              f"{_st.median(_cur):.2f}배" if _cur else
              f"  난이도: {len(_diff)}쌍 (현재 조건 데이터 없음)")
    else:
        print("  난이도: 계산 불가")

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
