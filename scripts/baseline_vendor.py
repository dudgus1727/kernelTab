#!/usr/bin/env python3
"""베이스라인 1 — 벤더 휴리스틱(nvMatmulHeuristics)의 regret@k.

측정 없이 config 하나를 고르는 **런타임 디스패치** 시나리오의 실질적 상대다.
정적 top-1 은 실무에서 아무도 안 쓰므로 베이스라인으로 약하다.

`C/A`(cuBLAS 대비)와 다르다. C/A 는 cuBLAS 의 **다른 커널 계열** 대비라
구현 차이가 섞인다. 여기서는 **같은 CUTLASS 커널 공간 안에서** 휴리스틱의
순위 품질만 본다.

nvMatmulHeuristics 는 별도 의존성이므로 이 저장소 환경을 오염시키지 않도록
격리된 venv 에서 추출 단계만 돌린다.

    python3 -m venv /tmp/nvmmh && /tmp/nvmmh/bin/pip install nvidia-matmul-heuristics
    /tmp/nvmmh/bin/python scripts/baseline_vendor.py --extract vendor.json
    python3 scripts/baseline_vendor.py --score vendor.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: GPU 이름 -> nvMatmulHeuristics 프리셋. 다른 GPU 로 옮길 때 여기만 늘린다.
#: 하드코딩하지 않고 `hw.name` 에서 유도한다.
GPU_PRESETS = {
    "rtx a6000": "RTX_A6000", "rtx 4090": "RTX_4090", "rtx 3090": "RTX_3090",
    "rtx 5090": "RTX_5090", "rtx 6000 ada": "RTX_6000_ADA",
    "a100": "A100_SXM_80GB", "a40": "A40_PCIE", "a30": "A30_PCIE", "a10": "A10_PCIE",
    "h100": "H100_SXM", "h200": "H200_SXM", "l40s": "L40S", "l40": "L40", "l4": "L4",
    "b200": "B200",
}

PAT = re.compile(
    r"stages\((\d+)\)\s+cta\((\d+) (\d+) (\d+)\)\s+warp\((\d+) (\d+) (\d+)\)"
    r"\s+instr\((\d+) (\d+) (\d+)\)\s+splitK\((\d+)\)\s+swizz\((\d+)\)"
    r"\s+ctaOrder\((\d+)\)")


def preset_for(name: str) -> str:
    low = name.lower().replace("nvidia", "").strip()
    for k, v in sorted(GPU_PRESETS.items(), key=lambda kv: -len(kv[0])):
        if k in low:
            return v
    raise SystemExit(
        f"'{name}' 에 대응하는 nvMatmulHeuristics 프리셋을 모른다.\n"
        f"  GPU_PRESETS 에 추가하라. 가능한 값은 "
        f"NvMatmulHeuristicsNvidiaGpu 의 멤버다.")


def extract(out_path: str, count: int) -> int:
    """휴리스틱에서 형상별 top-k 를 뽑는다. 격리된 venv 에서 돌린다."""
    import nvMatmulHeuristics as nv

    sys.path.insert(0, str(REPO_ROOT))
    from kerneltab.core import paths
    from kerneltab.core.hardware import hardware_from_env
    from kerneltab.core.shapes import all_shapes

    env = json.loads(paths.ENV_JSON.read_text())
    hw = hardware_from_env(env)
    preset = preset_for(hw.name)
    print(f"{hw.name} -> nvMatmulHeuristics 프리셋 {preset}")

    h = nv.NvMatmulHeuristicsInterface(nv.NvMatmulHeuristicsTarget.CUTLASS,
                                       precision="HSS")
    hd = h.createHardwareDescriptor()
    h.setHardwarePredefinedGpu(hd, getattr(nv.NvMatmulHeuristicsNvidiaGpu, preset))

    # 형상 레이아웃은 전 형상 공통 (A row / B col / C row = TN_ROW)
    layout = nv.NvMatmulHeuristicsMatmulLayout.TN_ROW_MAJOR
    out = {"_meta": {"gpu": hw.name, "preset": preset, "env_hash": env["env_hash"],
                     "count": count}}
    for p in all_shapes(hw):
        cfgs = h.get_with_mnk(p.M, p.N, p.K, layout, count, hd)
        lst = []
        for c in cfgs:
            kern, rt = c["kernel"], c.get("runtime")
            if not isinstance(kern, str):        # GemmConfig 객체로 오는 경우
                g = kern
                lst.append({"stages": g.stages,
                            "cta": [g.cta_tile_m, g.cta_tile_n, g.cta_tile_k],
                            "warp": [g.warp_tile_m, g.warp_tile_n, g.warp_tile_k],
                            "split_k": g.split_k, "swizzle": g.swizzle_factor,
                            "cta_order": g.cta_order,
                            "pred_ms": (rt or 0) * 1000.0, "raw": str(kern)})
                continue
            mo = PAT.search(kern)
            if not mo:
                lst.append({"raw": kern, "parse_fail": True})
                continue
            g = [int(x) for x in mo.groups()]
            lst.append({"stages": g[0], "cta": g[1:4], "warp": g[4:7],
                        "split_k": g[10], "swizzle": g[11], "cta_order": g[12],
                        "pred_ms": (rt or 0) * 1000.0, "raw": kern})
        out[f"{p.M}x{p.N}x{p.K}"] = lst
    Path(out_path).write_text(json.dumps(out, indent=1))
    print(f"{len(out)-1} 형상 -> {out_path}")
    return 0


#: 최근접 매핑의 축별 가중치. cta tile 을 가장 중시한다.
NEAR_W = (3, 3, 3, 1, 1, 1, 1, 0.5, 2, 0.5)


def _dist(a, b) -> float:
    return sum(w * (math.log2(max(x, 1) + 1) - math.log2(max(y, 1) + 1)) ** 2
               for w, x, y in zip(NEAR_W, a, b))


def score(path: str, ok_only: bool = False) -> int:
    sys.path.insert(0, str(REPO_ROOT))
    import pyarrow.parquet as pq

    from kerneltab.core import paths

    env = json.loads(paths.ENV_JSON.read_text())
    eh = env["env_hash"]
    vend = json.loads(Path(path).read_text())
    meta = vend.pop("_meta", {})
    if meta and not eh.startswith(str(meta.get("env_hash", ""))[:8]):
        print(f"!! 추출 시점 env_hash({str(meta.get('env_hash'))[:8]}) 와 "
              f"현재({eh[:8]}) 가 다르다. 측정 조건이 어긋난다.")
        return 2

    t = pq.read_table(paths.RESULTS_DIR / "table.parquet",
                      columns=["env_hash", "M", "N", "K", "tile_m", "tile_n",
                               "tile_k", "ext_warp_m", "ext_warp_n", "ext_warp_k",
                               "ext_stages", "ext_swizzle_type", "ext_swizzle_n",
                               "split_k", "split_k_mode", "time_ms", "status",
                               "difficulty"])
    c = {k: t.column(k).to_pylist() for k in t.column_names}
    best, diff = {}, {}
    cand = defaultdict(list)
    for i in range(t.num_rows):
        if not str(c["env_hash"][i]).startswith(eh[:8]):
            continue
        # status-filter: --status 플래그. 기본은 all.
        if ok_only and c["status"][i] != "ok":
            continue
        tm = c["time_ms"][i]
        if not tm:
            continue
        sh = (c["M"][i], c["N"][i], c["K"][i])
        if sh not in best or tm < best[sh]:
            best[sh] = tm
        if c["difficulty"][i]:
            diff[sh] = c["difficulty"][i]
        cand[sh].append((
            (c["tile_m"][i], c["tile_n"][i], c["tile_k"][i],
             c["ext_warp_m"][i], c["ext_warp_n"][i], c["ext_warp_k"][i],
             c["ext_stages"][i],
             c["ext_swizzle_n"][i] if c["ext_swizzle_type"][i] == "identity" else 0,
             c["split_k"][i], 0 if c["split_k_mode"][i] == "serial" else 1), tm))

    KS = (1, 3, 5)
    strict = {k: [] for k in KS}
    near = {k: [] for k in KS}
    n_exact = n_tot = 0
    for skey, lst in vend.items():
        sh = tuple(int(x) for x in skey.split("x"))
        if sh not in best:
            continue
        lut = dict(cand[sh])
        ts_s, ts_n = [], []
        for cf in lst:
            if cf.get("parse_fail"):
                continue
            n_tot += 1
            # CUTLASS 2.x 의 split_k 기본은 serial 이다
            key = (cf["cta"][0], cf["cta"][1], cf["cta"][2],
                   cf["warp"][0], cf["warp"][1], cf["warp"][2], cf["stages"],
                   cf["swizzle"] if cf["cta_order"] == 0 else 0, cf["split_k"], 0)
            if key in lut:
                n_exact += 1
                ts_s.append(lut[key])
                ts_n.append(lut[key])
            else:
                ts_s.append(None)
                ts_n.append(min(cand[sh], key=lambda vt: _dist(key, vt[0]))[1])
        for k in KS:
            g = [x for x in ts_s[:k] if x is not None]
            if g:
                strict[k].append((sh, min(g) / best[sh]))
            g = [x for x in ts_n[:k] if x is not None]
            if g:
                near[k].append((sh, min(g) / best[sh]))

    print(f"엄격 매핑 {n_exact}/{n_tot} ({100 * n_exact / n_tot:.1f}%)")
    med = statistics.median(diff.values())
    hard = {s for s, d in diff.items() if d >= med}

    def geo(v):
        return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")

    for name, res in (("엄격(매핑된 형상만)", strict), ("최근접(전 형상)", near)):
        print(f"\n[{name}]")
        print(f"{'k':>3} {'덮개':>7} {'전체':>9} {'어려운절반':>11} {'쉬운절반':>10}")
        for k in KS:
            rs = res[k]
            print(f"{k:3d} {100 * len(rs) / len(best):6.0f}% "
                  f"{geo([r for _, r in rs]):9.4f} "
                  f"{geo([r for s, r in rs if s in hard]):11.4f} "
                  f"{geo([r for s, r in rs if s not in hard]):10.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", metavar="OUT",
                    help="휴리스틱에서 뽑아 JSON 으로 저장 (nvMatmulHeuristics 필요)")
    ap.add_argument("--score", metavar="IN", help="뽑아둔 JSON 을 표에 대조해 채점")
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--status", choices=("ok", "all"), default="all",
                    help="측정 줄 필터. **기본 all** — `high_outlier_frac` 은 "
                         "결측이 아니라 품질 표시다 (consumer_contract 9절)")
    a = ap.parse_args()
    if a.extract:
        return extract(a.extract, a.count)
    if a.score:
        # status-filter: --status 플래그를 score() 로 넘긴다. 기본은 all.
        return score(a.score, a.status == "ok")
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
