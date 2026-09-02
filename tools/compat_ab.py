#!/usr/bin/env python3
"""forward-compat 이 런치 오버헤드에 주는 영향을 A/B 로 잰다.

    # 컨테이너 안에서
    python3 tools/compat_ab.py --n 25

## 왜 재는가

`nvidia-container-toolkit` 은 이미지의 CUDA forward-compat `libcuda` 를
끼워 넣는다. 그래서 **유저 모드 드라이버가 호스트가 아니라 이미지에서
온다** — GPU 간 비교에서 변수가 하나 줄어드는 대신, compat 계층이 런치
경로에 얹힌다.

    A6000   +53.5 ns (+3.01 %)   느려진다
    5090    -167 ns  (-9.1 %)    ★ 빨라진다

**방향이 GPU 마다 다르다.** 측정에 미치는 영향은 작지만(가장 짧은 커널
14 us 기준 0.36 %, 노이즈 바닥의 1/9) 이것은 **런치당 상수**다 — 드리프트가
사는 바로 그 자리다. 그래서 새 환경마다 잰다.

## 교대로 잰다

한쪽을 몰아 돌리면 발열/램프업 추세와 섞인다. A/B/A/B 로 교대한다.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kerneltab.core import paths            # noqa: E402

FIELDS = ("launch_async_small_ms", "launch_async_grid_ms",
          "launch_bracketed_small_ms", "launch_bracketed_grid_ms")


def find_native(hint: str | None) -> Path:
    """호스트(커널 모드와 짝인) libcuda 를 찾는다. compat 이 아닌 쪽."""
    if hint:
        return Path(hint)
    cands = sorted(Path("/usr/lib/x86_64-linux-gnu").glob("libcuda.so.[0-9]*"))
    cands = [c for c in cands if "compat" not in str(c) and c.name.count(".") >= 3]
    if not cands:
        raise SystemExit(
            "호스트 libcuda 를 못 찾았다. 컨테이너 안에서 돌리고 있는가?\n"
            "  ldconfig -p | grep libcuda  로 확인하고 --native 로 지정하라.")
    return cands[-1]


def run(exe: Path, env_extra: dict) -> dict:
    e = dict(os.environ)
    e.update(env_extra)
    r = subprocess.run([str(exe)], capture_output=True, text=True, env=e)
    if r.returncode != 0:
        raise SystemExit(f"launch_probe 실패:\n{r.stdout}\n{r.stderr}")
    return json.loads(r.stdout.splitlines()[-1])


def mannwhitney_u(a: list[float], b: list[float]) -> float:
    """작은 표본용 순위합 U. 두 분포가 겹치면 len(a)*len(b)/2 근처다.

    ⛔ **동점에 중간 순위를 준다.** 예전에는 `sorted((값, 그룹))` 의 순서를
       그대로 순위로 썼는데, 그러면 값이 같을 때 그룹 0 이 항상 앞에 서서
       **한쪽이 전부 이긴 것처럼 보인다.** 실제로 `launch_bracketed_grid` 는
       compat/native 의 중앙값이 정확히 같은데(둘 다 3.0720 us) `U = 0/625`
       가 나왔다 — "완전히 다르다" 는 판정이다.

       이 하네스의 값은 **이벤트 눈금(4090: 32 ns)에 양자화**되어 있어 동점이
       흔하다. 동점 처리는 선택이 아니라 필수다.
    """
    n1, n2 = len(a), len(b)
    if not n1 or not n2:
        return 0.0
    xs = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1][0] == xs[i][0]:
            j += 1
        mid = (i + 1 + j + 1) / 2.0        # 1-기반 중간 순위
        for k in range(i, j + 1):
            ranks[k] = mid
        i = j + 1
    r = sum(rk for rk, (_, g) in zip(ranks, xs) if g == 0)
    return r - n1 * (n1 + 1) / 2.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25, help="각 조건의 반복 수")
    ap.add_argument("--native", default=None, help="호스트 libcuda 경로")
    ap.add_argument("--exe", default=None, help="launch_probe 경로")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    exe = Path(a.exe) if a.exe else (paths.ARTIFACT_DIR / "launch_probe")
    if not exe.exists():
        raise SystemExit(f"{exe} 가 없다. detect(phase0_env)를 먼저 돌려라.")
    native = find_native(a.native)
    d = Path("/tmp/kt_native_libcuda")
    d.mkdir(exist_ok=True)
    link = d / "libcuda.so.1"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(native)
    print(f"compat  : (기본) ldconfig 가 고르는 것")
    print(f"native  : {native}")

    res = {"compat": {f: [] for f in FIELDS}, "native": {f: [] for f in FIELDS}}
    for i in range(a.n):
        for tag, extra in (("compat", {}),
                           ("native", {"LD_LIBRARY_PATH": str(d)})):
            j = run(exe, extra)
            for f in FIELDS:
                if f in j:
                    res[tag][f].append(j[f])
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{a.n}", flush=True)

    out = {"native_libcuda": str(native), "n": a.n, "fields": {}}
    print(f"\n{'항목':28s} {'compat(us)':>11s} {'native(us)':>11s} "
          f"{'차이(ns)':>9s} {'차이(%)':>8s} {'U':>10s}")
    for f in FIELDS:
        c, n = res["compat"][f], res["native"][f]
        if not c or not n:
            continue
        mc, mn = statistics.median(c), statistics.median(n)
        u = mannwhitney_u(c, n)
        rec = {"compat_median_ms": mc, "native_median_ms": mn,
               "delta_ns": round((mc - mn) * 1e6, 2),
               "delta_pct": round(100 * (mc - mn) / mn, 3) if mn else None,
               "u": u, "u_center": len(c) * len(n) / 2.0,
               "compat": c, "native": n}
        out["fields"][f] = rec
        print(f"{f:28s} {mc * 1000:11.4f} {mn * 1000:11.4f} "
              f"{rec['delta_ns']:9.1f} {rec['delta_pct']:7.2f}% "
              f"{u:6.0f}/{len(c) * len(n)}")
    print(f"\nU 가 {len(res['compat'][FIELDS[0]]) ** 2 // 2} 근처면 두 분포가 겹친다"
          " (= 차이 없음).")
    print("⚠️ compat 을 끄지 않는다 — 끄면 유저 모드가 다시 호스트마다 달라진다."
          " 여기서 재는 것은 '얼마나 얹히는가' 다.")
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
