#!/usr/bin/env python3
"""요청 M: 클럭 고정이 부하 상태에서 실제로 유지되는지 검증.

`nvidia-smi -lgc` 로 고정해도 전력/온도 캡이 걸리면 클럭이 내려간다.
아이들 상태에서 1350 MHz 로 보인다고 고정된 것이 아니므로, 고부하 형상으로
연속 측정하면서 텔레메트리를 봐야 한다.

    python3 scripts/verify_clock_lock.py --minutes 5 --expect-mhz 1350
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build import paths  # noqa: E402
from core.hardware import hardware_from_env  # noqa: E402
from core.types import Hardware, Problem  # noqa: E402

TELE = paths.RESULTS_DIR / "telemetry_clocklock.csv"
OUT = paths.RESULTS_DIR / "clock_lock_check.json"

LOAD_SHAPE = Problem(8192, 4096, 4096)

THROTTLE_BITS = {
    0x0001: "gpu_idle",
    0x0002: "app_clocks_setting",
    0x0004: "sw_power_cap",
    0x0008: "hw_slowdown",
    0x0010: "sync_boost",
    0x0020: "sw_thermal",
    0x0040: "hw_thermal",
    0x0080: "hw_power_brake",
    0x0100: "display_clock",
}


def start_telemetry(device: int):
    f = TELE.open("w")
    p = subprocess.Popen(
        ["nvidia-smi", "-i", str(device),
         "--query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,"
         "power.draw,clocks_throttle_reasons.active",
         "--format=csv", "-l", "1"],
        stdout=f, stderr=subprocess.DEVNULL)
    return p, f


def parse_telemetry() -> list[dict]:
    out = []
    for line in TELE.read_text().splitlines()[1:]:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            out.append({
                "clk": int(parts[1].split()[0]),
                "mem": int(parts[2].split()[0]),
                "temp": int(parts[3]),
                "power": float(parts[4].split()[0]),
                "bits": int(parts[5], 16),
            })
        except (ValueError, IndexError):
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--expect-mhz", type=int, default=None)
    args = ap.parse_args()

    env = json.loads(paths.ENV_JSON.read_text())
    dev = env["device_index"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(dev)
    hw = hardware_from_env(env)

    from measure.runner import Ctx, Kernel, KtProblemC  # noqa: E402

    rows = [json.loads(l) for l in
            (paths.RESULTS_DIR / "kernels.jsonl").read_text().splitlines() if l.strip()]
    # 리허설에서 이 형상 최적이었던 구성 = 가장 부하가 큰 쪽
    want = ("sm86_tb128x128x64_w64x64x64_st3_swid2_a888",
            "sm86_tb128x128x64_w16x64x64_st3_swid2_a888")
    krow = next((r for r in rows if r["kernel_id"] in want), None)
    if krow is None:
        krow = next(r for r in rows
                    if r["build_status"] == "ok"
                    and (r["tile"]["m"], r["tile"]["n"]) == (128, 128)
                    and r["regs_per_thread"] * r["threads"] <= hw.regs_per_sm)

    expect = args.expect_mhz
    if expect is None:
        rc = subprocess.run(
            ["nvidia-smi", "-i", str(dev), "--query-gpu=clocks.current.sm",
             "--format=csv,noheader,nounits"], capture_output=True, text=True)
        expect = int(float(rc.stdout.strip().splitlines()[0]))

    print(f"GPU {dev}  기대 클럭 {expect} MHz  부하 형상 "
          f"({LOAD_SHAPE.M},{LOAD_SHAPE.N},{LOAD_SHAPE.K})")
    print(f"커널 {krow['kernel_id']}")

    ctx = Ctx(paths.ARTIFACT_DIR / "libkt_ctx.so", 0)
    ctx.set_protocol(env)
    k = Kernel(krow["so_path"])
    ctx.prepare_problem(LOAD_SHAPE.M, LOAD_SHAPE.N, LOAD_SHAPE.K)
    kp = KtProblemC(LOAD_SHAPE.M, LOAD_SHAPE.N, LOAD_SHAPE.K, 1, 0)
    bufs = ctx.buffers(k.workspace_bytes(kp), False)
    st, h = k.prepare(kp, bufs)
    if st != 0 or not h:
        print("prepare 실패")
        return 1

    proc, f = start_telemetry(dev)
    times: list[tuple[float, float]] = []
    t0 = time.time()
    try:
        while time.time() - t0 < args.minutes * 60:
            _, m = ctx.measure(k.launch_addr, h, 0)
            times.append((time.time() - t0, m.time_ms))
            el = time.time() - t0
            if len(times) % 20 == 0:
                print(f"  {el / 60:4.1f}분  {m.time_ms:.4f} ms", flush=True)
    finally:
        k.release(h)
        ctx.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        f.close()

    tel = parse_telemetry()
    if not tel:
        print("텔레메트리 없음")
        return 1

    clks = [t["clk"] for t in tel]
    pw = [t["power"] for t in tel]
    tp = [t["temp"] for t in tel]
    thr = Counter()
    for t in tel:
        for mask, name in THROTTLE_BITS.items():
            if t["bits"] & mask:
                thr[name] += 1
    n = len(tel)
    at_target = sum(1 for c in clks if c >= expect)

    print("\n" + "=" * 74)
    print(f"클럭 고정 검증  ({args.minutes}분, 텔레메트리 {n}초, 측정 {len(times)}회)")
    print("=" * 74)
    print(f"  clocks.sm   min={min(clks)} max={max(clks)} "
          f"mean={statistics.mean(clks):.1f} median={statistics.median(clks)} MHz")
    print(f"              기대치({expect}) 이상 유지: {at_target}/{n} "
          f"= {100 * at_target / n:.1f}%")
    mem = [t["mem"] for t in tel]
    print(f"  clocks.mem  min={min(mem)} max={max(mem)} "
          f"median={statistics.median(mem)} MHz "
          f"(고정 안 하면 유휴 시 810 까지 떨어진다)")
    print(f"  power.draw  min={min(pw):.1f} max={max(pw):.1f} "
          f"mean={statistics.mean(pw):.1f} W  (캡 300W)")
    print(f"  temp        시작={tp[0]} 최고={max(tp)} 마지막={tp[-1]} °C")
    print(f"  throttle    ", end="")
    if thr:
        print({k2: f"{v}s ({100 * v / n:.1f}%)" for k2, v in thr.most_common()})
    else:
        print("없음")

    swpc = thr.get("sw_power_cap", 0) / n
    # 측정 시간의 시작/끝 사분위 비교 = 워밍업 드리프트
    ts = [t for _, t in times]
    q = max(1, len(ts) // 4)
    first_q, last_q = statistics.median(ts[:q]), statistics.median(ts[-q:])
    print(f"  time_ms     첫 1/4 중앙값={first_q:.4f}  마지막 1/4={last_q:.4f}  "
          f"차이={100 * (last_q - first_q) / first_q:+.2f}%")
    print(f"              전체 min={min(ts):.4f} max={max(ts):.4f} "
          f"변동폭={100 * (max(ts) - min(ts)) / statistics.median(ts):.2f}%")

    print("\n판정:")
    verdict = "hold"
    if swpc > 0.10:
        verdict = "lower"
        print(f"  !! sw_power_cap {100 * swpc:.1f}% > 10% — 클럭을 더 내려야 한다.")
    elif max(pw) <= 250:
        verdict = "raise"
        print(f"  sw_power_cap {100 * swpc:.1f}% <= 10% 이고 최대 전력 "
              f"{max(pw):.1f}W <= 250W 로 여유 있음 — 클럭을 올려볼 수 있다.")
    else:
        print(f"  sw_power_cap {100 * swpc:.1f}% <= 10%, 최대 전력 {max(pw):.1f}W "
              f"— 현재 값 유지가 적절하다.")

    OUT.write_text(json.dumps({
        "expect_mhz": expect, "samples": n, "measurements": len(times),
        "clk_min": min(clks), "clk_max": max(clks),
        "clk_mean": statistics.mean(clks), "clk_at_target_frac": at_target / n,
        "power_min": min(pw), "power_max": max(pw), "power_mean": statistics.mean(pw),
        "temp_start": tp[0], "temp_max": max(tp), "temp_end": tp[-1],
        "mem_clk_min": min(mem), "mem_clk_max": max(mem),
        "mem_clk_median": statistics.median(mem),
        "throttle_seconds": dict(thr), "sw_power_cap_frac": swpc,
        "time_first_quartile_ms": first_q, "time_last_quartile_ms": last_q,
        "time_min_ms": min(ts), "time_max_ms": max(ts),
        "verdict": verdict,
    }, indent=2))
    print(f"\n{OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
