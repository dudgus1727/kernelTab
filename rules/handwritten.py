"""손으로 쓴 config 선택 규칙 (kernelrule 설계문서 §9.4).

**측정 없이** 형상 하나에 대해 config 하나를 고른다 (런타임 디스패치).
빌드타임 오토튜닝(k개를 재보고 고르는 것)과는 다른 문제다.

제약 — 이 규칙이 정직하려면 지켜야 한다:
  * `core/features.py` 가 주는 피처와 커널의 **정적 속성**만 쓴다
  * 측정 시간, 그로부터 유도된 값(`difficulty` 등)은 쓰지 않는다
  * M/N/K 로 직접 분기하지 않는다 (형상 그리드에 과적합된다)
  * 리터럴 상수 8개 이하

점수가 낮을수록 좋다.

--- 설계 근거 -------------------------------------------------------------
첫 시도는 `tail_waste` 최소화를 1순위로 뒀다가 regret 1.78 로 정적 top-1
(1.394) 보다 나빴다. 원인은 **지배적인 항이 빠진 것**이었다: 규칙이 tail 이
작은 쪽만 좇아 32x128 같은 작은 타일에 wave 48개짜리 config 를 골랐다.

GEMM 한 번의 DRAM 트래픽은

    ceil(M/tm) x ceil(N/tn) 타일 x 각 타일이 읽는 (tm+tn)K 원소

이므로 **타일 크기가 곧 데이터 재사용**이고, 이것이 다른 무엇보다 크다.
작은 타일은 tail 이 작은 대신 A/B 를 훨씬 여러 번 다시 읽는다.

`1/tm + 1/tn` 로 근사하면 안 된다 — 그건 타일이 항상 꽉 찬다고 가정한다.
M=1 인 decode 형상에서 tile_m=128 을 고르면 타일의 99% 가 낭비인데 근사식은
그걸 모르고 큰 타일을 계속 밀어준다 (실제로 그 형상들에서 regret 2.9 가
나왔다). ceil 을 그대로 쓰면 M=1 에서 gm=1 이라 tm 을 키워도 타일 수가 줄지
않고 (tm+tn) 만 늘어 **자동으로 벌점**이 된다.

두 번째 항은 **SM 활용률**이다. GPU 는 wave 단위로 시간을 쓰므로 유효 연산
처리량은 `peak x (1 - tail_waste)` 이고, 따라서 연산 쪽 시간은
`1/(1 - tail_waste)` 에 비례한다. 이 형태가 중요하다 — 512³ 에 128x128 을
쓰면 타일이 16개뿐이라 84개 SM 중 16개만 돌고(tail_waste 0.81) 계수가
5.3 배가 된다. 64x64 면 타일 64개로 1.3 배다. 선형 항으로는 이 차이가
안 나온다.
"""
from __future__ import annotations

# 리터럴 8개.
W_REUSE = 1.00     # DRAM 트래픽 ∝ ceil(M/tm)·ceil(N/tn)·(tm+tn) (지배 항)
W_UTIL = 0.85      # SM 활용률. 연산 시간 ∝ 1/(1 - tail_waste)
W_OCC = 0.25       # smem/레지스터 압력으로 떨어진 occupancy
W_SPLIT = 0.40     # split-K 리덕션 비용 (이미 꽉 찼으면 손해)
SPILL_PENALTY = 9.0   # 스필은 66형상에서 최적을 낸 적이 0회 (전수 확인)
PIPE_BONUS = 0.05  # multistage(cp.async) 선호
TILE_REF = 256.0   # 재사용 항 정규화 (128x128 꽉 찬 타일이 1.0 이 되도록)


def score(f: dict) -> float:
    """config 하나의 점수. 낮을수록 좋다. f 는 표의 한 행(피처만)."""
    tm, tn = f["tile_m"], f["tile_n"]
    w = f["waves_occ"] or f["waves"] or 1.0

    # 1) 데이터 재사용 — 지배 항.
    #    트래픽 = ceil(M/tm)·ceil(N/tn)·(tm+tn)·K, 유용한 출력은 M·N.
    #    K 와 M·N 은 형상 안에서 상수이므로 순위에 영향이 없다.
    gm = -(-f["M"] // tm)          # ceil
    gn = -(-f["N"] // tn)
    s = W_REUSE * gm * gn * (tm + tn) * TILE_REF / (f["M"] * f["N"])

    # 2) SM 활용률 — GPU 는 wave 단위로 시간을 쓴다. 유효 처리량이
    #    peak x (1 - tail_waste) 이므로 연산 시간은 그 역수에 비례한다.
    tw = f["tail_waste_occ"]
    if tw is None or tw != tw:
        tw = f["tail_waste"]
    s += W_UTIL * (1.0 / max(1.0 - tw, 0.02) - 1.0)

    # 4) occupancy — 낮으면 메모리 지연을 못 숨긴다.
    s += W_OCC * (1.0 - (f["theoretical_occupancy"] or 0.0))

    # 5) split-K 는 리덕션과 workspace 를 더한다. 이미 wave 가 충분하면
    #    쪼갤 이유가 없고, 부족할 때만 SM 을 채우는 값을 한다.
    sk = f["split_k"] or 1
    if sk > 1:
        s += W_SPLIT * (sk - 1) * max(w, 0.25)

    # 6) 스필 배제. 전수에서 스필 커널은 최적을 낸 적이 없다 (축 격차 0.572).
    if f["has_spill"]:
        s += SPILL_PENALTY

    # 7) cp.async multistage 선호.
    if f["pipeline_kind"] == "multistage":
        s -= PIPE_BONUS

    return s
