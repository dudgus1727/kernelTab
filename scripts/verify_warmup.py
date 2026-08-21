#!/usr/bin/env python3
"""워밍업 시간 예산화가 **측정값을 바꾸지 않는지** 확인한다.

    python3 scripts/verify_warmup.py --gpu <UUID>
    python3 scripts/verify_warmup.py --pairs 24 --reps 5

## 무엇을 묻는가

예비 실행과 워밍업을 시간 예산으로 바꾸면 느린 커널의 워밍업이 크게 줄어든다
(캠페인 실측: 오버헤드가 GPU 시간의 **68.2 %**). 절약은 확실하지만 위험도
분명하다 — **워밍업이 부족하면 첫 측정이 냉시작이 되어 값이 느려진다.**

A6000 에서 워밍업 없이 재면 4096³ 이 **+26.32 %** 였다. 그 정도 크기의
왜곡이면 표 전체가 못 쓰게 된다.

## 어떻게 재는가

같은 `(커널, 형상)` 을 **옛 프로토콜과 새 프로토콜로 교대**로 잰다.
한쪽을 몰아서 돌리면 발열·클럭 램프업 추세와 섞인다 — compat 계층의
50 ns 를 잡을 때 쓴 것과 같은 설계다.

## 판정 — **대조군을 함께 잰다**

노이즈 모델(`core/noise.py`) 하나로 판정하면 안 된다. 모델은 앵커 커널에서
뽑은 것이고, 커널마다 실제 산포가 다르다. 실제로 프로토콜이 **완전히 같은**
조합(`warm 10->10`, 오버헤드 100.2->100.2 ms)이 +0.30 % 차이로 "실패" 로
찍혔다 — 변경 때문일 수 없는 차이다.

그래서 **옛 프로토콜을 두 번** 잰다. `옛A / 새 / 옛B` 를 교대로 돌리고,
"옛 vs 새" 차이를 "옛A vs 옛B" 차이와 비교한다. 후자가 이 조합의
**귀무분포**다 — 같은 조건에서 두 번 재면 얼마나 다른가.

    새 - 옛  <=  max(k x SE,  노이즈 바닥,  |옛A - 옛B|)     -> 통과

`SE` 는 그룹 내 산포에서 나온다. 분자와 분모의 집계 수준을 맞추는 것은
앵커 판정에서 쓴 것과 같은 원칙이다 (`core/anchors.py`).

**방향도 본다.** 워밍업이 부족하면 새 쪽이 **느려진다**(냉시작). 빨라지는
방향의 이탈은 워밍업으로 설명되지 않으므로 주의로만 찍고 실패로 세지 않는다.

## 무엇을 함께 보나

`n_warmup` 이 **0 이 되지 않는지**. `WARMUP_SECONDS` 가 정의만 되고
쓰이지 않아 워밍업이 통째로 사라진 적이 있고, 그때 로그는 "워밍업만 한다"
를 찍고 있었다. 로그가 아니라 값을 본다.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kerneltab.build.compile import build_ctx_so
from kerneltab.core import device, noise, paths, records
from kerneltab.core import kernels as kernels_mod
from kerneltab.core.hardware import hardware_from_env
from kerneltab.measure.runner import (
    PROTOCOL_DEFAULTS,
    Ctx,
    Kernel,
    KtProblemC,
    KtProtocolC,
    protocol_from_env,
)

#: 시간 분포를 덮도록 고르는 형상. 느린 쪽이 이 변경의 대상이다.
SHAPES = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048),
          (4096, 4096, 4096), (8192, 4096, 4096)]

#: 노이즈 바닥의 몇 배까지 허용할지.
TOL_K = 3.0


def old_protocol(env: dict) -> KtProtocolC:
    """시간 예산을 끈 것 = 예전 동작."""
    d = dict(PROTOCOL_DEFAULTS)
    d.update(env.get("protocol") or {})
    d["probe_budget_ms"] = 0.0
    d["warmup_budget_ms"] = 0.0
    return KtProtocolC(**{k: d[k] for k in PROTOCOL_DEFAULTS})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--pairs", type=int, default=20,
                    help="(커널, 형상) 조합 수")
    ap.add_argument("--reps", type=int, default=5, help="교대 반복 횟수")
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--tol-k", type=float, default=TOL_K)
    ap.add_argument("--out", default=str(paths.RESULTS_DIR / "warmup_check.json"))
    a = ap.parse_args()

    if not paths.ENV_JSON.exists():
        print("results/env.json 이 없다.")
        return 2
    env = json.loads(paths.ENV_JSON.read_text())
    if a.gpu:
        env.setdefault("hardware_extra", {})["uuid"] = a.gpu
    device.resolve_device(env)
    hw = hardware_from_env(env)

    rows = [r for r in records.iter_records(
        paths.RESULTS_DIR / "kernels.jsonl", records.ALL)
        if r.get("build_status") == "ok"
        and kernels_mod.launchable(r, hw.regs_per_sm)
        and paths.kernel_so(r["kernel_id"]).exists()]
    if not rows:
        print("빌드된 커널이 없다. build_kernels.py 를 먼저 돌려라.")
        return 2

    rng = random.Random(a.seed)
    picks = rng.sample(rows, min(a.pairs, len(rows)))
    pairs = [(r, SHAPES[i % len(SHAPES)]) for i, r in enumerate(picks)]

    # 없거나 낡았으면 여기서 빌드한다. 손으로 복사하게 두면 잊는다 —
    # 실제로 캠페인 디렉토리를 새로 만들면서 .so 만 빠져 죽었다.
    # (낡은 .so 는 Ctx 의 ABI 핸드셰이크가 따로 거부한다.)
    ctx = Ctx(build_ctx_so(env), 0)
    p_new = protocol_from_env(env)
    p_old = old_protocol(env)
    print(f"조합 {len(pairs)}개 x 교대 {a.reps}회   허용 = "
          f"{a.tol_k} x 노이즈 바닥\n")
    print(f"{'커널':>30} {'형상':>6} {'옛(ms)':>9} {'새-옛':>8} {'대조':>8} "
          f"{'허용':>7} {'warm':>9} {'오버헤드':>16} {'판정':>5}")

    out, fails, notes, saved_old, saved_new = [], [], [], 0.0, 0.0
    for krow, (M, N, K) in pairs:
        kern = Kernel(paths.kernel_so(krow["kernel_id"]))
        ctx.prepare_problem(M, N, K)
        kp = KtProblemC(M, N, K, 1, 0)
        bufs = ctx.buffers(kern.workspace_bytes(kp), False)
        st, h = kern.prepare(kp, bufs)
        if st != 0 or not h:
            continue
        try:
            o, n, ctrl = [], [], []
            wo = wn = None
            oo = on = 0.0
            for _ in range(a.reps):
                # 옛A / 새 / 옛B 를 교대로. 몰아서 돌리면 발열·램프업 추세와
                # 섞이고, 대조군이 없으면 이 조합의 산포를 알 수 없다.
                s1, m1 = ctx.measure(kern.launch_addr, h, 0, p_old)
                s2, m2 = ctx.measure(kern.launch_addr, h, 0, p_new)
                s3, m3 = ctx.measure(kern.launch_addr, h, 0, p_old)
                if s1 == 0:
                    o.append(m1.time_ms)
                    wo = m1.n_warmup
                    oo += m1.overhead_ms
                if s2 == 0:
                    n.append(m2.time_ms)
                    wn = m2.n_warmup
                    on += m2.overhead_ms
                if s3 == 0:
                    ctrl.append(m3.time_ms)
        finally:
            kern.release(h)
        if len(o) < 3 or len(n) < 3:
            continue
        # 눈금은 **이 장비에서 관측된 값**으로 잡는다. A6000 상수를 그냥
        # 쓰면 다른 GPU 에서 허용치가 틀린다 — 그리고 이 스크립트가 두 번
        # 틀린 곳이 정확히 허용치다.
        coef = noise.coef_from_observed(o + n + ctrl, krow["kernel_id"])
        mo, mn = statistics.median(o), statistics.median(n)
        mc = statistics.median(ctrl) if len(ctrl) >= 3 else mo
        rel = (mn - mo) / mo
        # 대조군: 같은 프로토콜을 두 번 잰 차이. 이 조합의 귀무분포다.
        rel_ctrl = (mc - mo) / mo
        # 그룹 내 산포에서 중앙값 차이의 표준오차를 잡는다.
        pooled = statistics.fmean([statistics.pstdev(o), statistics.pstdev(n)])
        se = (pooled / max(len(o), 1) ** 0.5) / mo if mo else 0.0
        floor = coef.floor(mo)
        # 허용치는 **대조군을 포함**해서 잡는다. 같은 프로토콜을 두 번 재서
        # X 만큼 달랐다면, 프로토콜 간 X 이하의 차이는 증거가 되지 못한다.
        #
        # ⚠️ 선형 추세 보정(rel - rel_ctrl/2)을 먼저 시도했다가 버렸다.
        #    짧은 커널의 중앙값은 타이머 눈금에 양자화돼 있어서, 원래 차이가
        #    **정확히 0** 인데 대조군이 한 눈금 움직이면 보정이 **없던 효과를
        #    만들어낸다.** 실제로 세 조합이 그렇게 "실패" 로 찍혔다
        #    (전부 `새-옛 = +0.00%`). 매끄러운 추세를 가정할 수 없다.
        tol = max(a.tol_k * se, floor, abs(rel_ctrl))
        # **방향이 중요하다.** 워밍업이 부족하면 새 쪽이 **느려진다**(냉시작).
        # 빨라지는 방향의 이탈은 워밍업으로 설명되지 않는다 — 산포다.
        cold = rel > tol
        ok = not cold
        rel_adj = rel
        saved_old += oo
        saved_new += on
        if cold:
            fails.append((krow["kernel_id"], M, rel, rel_ctrl, rel_adj, tol,
                          wo, wn))
        elif abs(rel) > tol:
            notes.append((krow["kernel_id"], M, rel, rel_ctrl, rel_adj, tol,
                          wo, wn))
        out.append({"kernel_id": krow["kernel_id"], "M": M, "N": N, "K": K,
                    "old_ms": mo, "new_ms": mn, "control_ms": mc,
                    "rel": rel, "rel_control": rel_ctrl, "tol": tol,
                    "noise_floor": floor, "n_warmup_old": wo,
                    "n_warmup_new": wn, "overhead_old_ms": oo,
                    "overhead_new_ms": on, "ok": ok})
        print(f"{krow['kernel_id'][-30:]:>30} {M:6d} {mo:9.4f} "
              f"{100 * rel:+7.2f}% {100 * rel_ctrl:+7.2f}% "
              f"{100 * tol:6.2f}% "
              f"{str(wo) + '->' + str(wn):>9} "
              f"{oo:7.1f}->{on:6.1f}ms {'OK' if ok else '실패':>5}")

    ctx.close()
    if not out:
        print("\n⛔ 비교한 조합이 0개다. '통과' 가 아니라 검사를 못 한 것이다.")
        return 2

    zero_warm = [r for r in out if not r["n_warmup_new"]]
    print(f"\n오버헤드 합계  옛 {saved_old / 1000:.1f}s -> 새 "
          f"{saved_new / 1000:.1f}s  "
          f"({100 * (1 - saved_new / max(saved_old, 1e-9)):.1f}% 절감)")
    print(f"워밍업이 0 이 된 조합: {len(zero_warm)}")
    Path(a.out).write_text(json.dumps(
        {"tol_k": a.tol_k, "n_pairs": len(out),
         "overhead_old_ms": saved_old, "overhead_new_ms": saved_new,
         "rows": out}, ensure_ascii=False, indent=1))
    print(f"-> {a.out}")

    if zero_warm:
        print("\n⛔ 워밍업이 0 회가 된 조합이 있다. warmup_reps_floor 를 확인하라.")
        return 4
    if notes:
        # 실패가 아니다. **냉시작으로 설명되지 않는 방향**의 이탈이다.
        print(f"\n주의 (실패 아님) — 산포가 큰 조합 {len(notes)}개")
        print("  새 쪽이 **빨라진** 이탈이므로 워밍업 부족으로 설명되지 않는다.")
        for kid, M, rel, rc, _adj, tol, wo, wn in notes[:8]:
            same = " (워밍업 동일)" if wo == wn else ""
            print(f"     {kid[-38:]:>38} @{M} 차이 {100 * rel:+.2f}% "
                  f"대조 {100 * rc:+.2f}% 허용 {100 * tol:.2f}%{same}")
    if fails:
        print(f"\n⛔ 새 프로토콜이 **느려진** 조합 {len(fails)}개 — 냉시작 의심.")
        for kid, M, rel, rc, _adj, tol, wo, wn in fails[:8]:
            print(f"     {kid[-38:]:>38} @{M} 차이 {100 * rel:+.2f}% "
                  f"(대조 {100 * rc:+.2f}%, 허용 {100 * tol:.2f}%, "
                  f"warm {wo}->{wn})")
        print("   warmup_budget_ms / warmup_reps_floor 를 올려라.")
        return 1
    print("\n통과: 측정값 차이가 전부 노이즈 바닥 이내다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
