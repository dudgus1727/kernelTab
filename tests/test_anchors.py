"""`core/anchors.py` — 앵커 판정과 라운드 매핑 (R-6).

여기서 고정하는 것은 두 가지다.

1. **판정은 노이즈 대비다.** 절대 1% 기준으로 돌아가면 짧은 앵커는 영원히
   실패한다 (512³ 은 14 us 라 노이즈 자체가 2% 대다). 예전에
   `report_phase3.py` 가 절대 기준으로 따로 판정해서 `check_anchors.py` 와
   다른 답을 냈다.
2. **라운드는 슬라이스 단위로 복원된다.** `(segment, when) -> round` 로
   매핑하면 라운드마다 덮어써서 마지막 하나만 남는다. 그러면 "전체가 함께
   드리프트하는가" 검사가 통째로 죽는다 — 그런데 **아무 오류도 안 난다.**
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone

import pytest

from core import anchors

EH = "c63710df" + "0" * 56
OTHER = "deadbeef" + "0" * 56
T0 = datetime(2026, 8, 17, 0, 0, 0, tzinfo=timezone.utc)
SLICE_S = 600.0


def _iso(dt):
    return dt.isoformat().replace("+00:00", "Z")


def _write(tmp_path, *, n_rounds=4, n_seg=3, base=None, offset=None,
           round_gain=None, env_hash=EH, record_round=False,
           slice_env=True, noise_sd=0.0, seed=1234, tick_ms=0.0):
    """앵커 + sweep 로그를 만든다.

    `offset[seg]`   세그먼트 고유의 계통 오차 (배수)
    `round_gain[r]` 라운드마다 전체에 곱해지는 값 (드리프트)
    `noise_sd`      측정 노이즈 (상대 표준편차). 시드를 고정해 결정론적이다.
    `tick_ms`       타이머 눈금. 실제 CUDA 이벤트 타이머는 1.024 us 다.
    """
    base = base or {("k_short", 512): 0.02, ("k_long", 4096): 2.0}
    offset = offset or {}
    round_gain = round_gain or {}
    rng = random.Random(seed)
    arows, srows = [], []
    t = T0
    for rnd in range(n_rounds):
        for seg in range(n_seg):
            t0 = t
            for when in ("start", "end"):
                for (kid, M), b in base.items():
                    ms = (b * offset.get(seg, 1.0) * round_gain.get(rnd, 1.0)
                          * (rng.gauss(1.0, noise_sd) if noise_sd else 1.0))
                    if tick_ms:
                        ms = round(ms / tick_ms) * tick_ms
                    row = {
                        "kernel_id": kid,
                        "problem": {"M": M, "N": M, "K": M},
                        "time_ms": ms, "segment": seg, "when": when,
                        "env_hash": env_hash,
                        "timestamp": _iso(t0 + timedelta(
                            seconds=5 if when == "start" else SLICE_S - 5)),
                    }
                    if record_round:
                        row["round"] = rnd
                    arows.append(row)
            t = t0 + timedelta(seconds=SLICE_S)
            ev = {"event": "slice", "round": rnd, "segment": seg, "rc": 0,
                  "seconds": SLICE_S, "timestamp": _iso(t)}
            if slice_env:
                ev["env_hash"] = env_hash
            srows.append(ev)
    (tmp_path / "anchors.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in arows))
    (tmp_path / "sweep.jsonl").write_text(
        json.dumps({"event": "sweep_start", "env_hash": env_hash,
                    "timestamp": _iso(T0)}) + "\n"
        + "".join(json.dumps(r) + "\n" for r in srows))
    return tmp_path


def _load(tmp_path, eh=EH[:8]):
    return anchors.load(eh, results_dir=tmp_path)


# ---------------------------------------------------------------------------
# 라운드 매핑 — R-6 의 본체
# ---------------------------------------------------------------------------
def test_모든_라운드가_복원된다(tmp_path):
    """옛 버그: `(segment, when) -> round` 는 라운드마다 덮어썼다.

    8 라운드를 돌고도 라운드가 1개만 남아 절대값 추이 검사가 죽었다.
    """
    _write(tmp_path, n_rounds=4, n_seg=3)
    rows, rnd, src, n_sl = _load(tmp_path)
    assert src == "timestamp"
    assert n_sl == 12
    assert sorted(set(rnd)) == [0, 1, 2, 3], "라운드가 뭉개졌다"
    assert None not in rnd


def test_기록된_round_를_우선한다(tmp_path):
    _write(tmp_path, n_rounds=3, n_seg=2, record_round=True)
    _rows, rnd, src, _ = _load(tmp_path)
    assert src == "recorded"
    assert sorted(set(rnd)) == [0, 1, 2]


def test_sweep_없이는_라운드를_모른다(tmp_path):
    _write(tmp_path, n_rounds=3, n_seg=2)
    (tmp_path / "sweep.jsonl").unlink()
    _rows, rnd, src, n_sl = _load(tmp_path)
    assert src == "none" and n_sl == 0 and set(rnd) == {None}


def test_slice_에_env_hash_가_없어도_sweep_start_에서_물려받는다(tmp_path):
    """이미 측정한 파일에는 `slice` 줄에 `env_hash` 가 없다."""
    _write(tmp_path, n_rounds=2, n_seg=2, slice_env=False)
    _rows, rnd, src, n_sl = _load(tmp_path)
    assert src == "timestamp" and n_sl == 4
    assert sorted(set(rnd)) == [0, 1]


def test_다른_env_hash_의_슬라이스는_섞이지_않는다(tmp_path):
    _write(tmp_path, n_rounds=2, n_seg=2)
    # 다른 캠페인의 로그를 같은 파일에 이어 붙인다 (append-only 니 정상이다)
    extra = [json.dumps({"event": "sweep_start", "env_hash": OTHER,
                         "timestamp": _iso(T0 + timedelta(days=1))})]
    for i in range(4):
        extra.append(json.dumps({
            "event": "slice", "round": 9, "segment": i % 2, "rc": 0,
            "seconds": SLICE_S,
            "timestamp": _iso(T0 + timedelta(days=1, seconds=SLICE_S * (i + 1)))}))
    with (tmp_path / "sweep.jsonl").open("a") as f:
        f.write("\n".join(extra) + "\n")
    _rows, rnd, _src, n_sl = _load(tmp_path)
    assert n_sl == 4, "다른 조건의 슬라이스가 섞였다"
    assert 9 not in set(rnd)


def test_구간_밖의_앵커는_라운드가_없다(tmp_path):
    _write(tmp_path, n_rounds=2, n_seg=2)
    p = tmp_path / "anchors.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    rows[0]["timestamp"] = _iso(T0 - timedelta(days=5))
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _rows, rnd, _src, _ = _load(tmp_path)
    assert rnd[0] is None


# ---------------------------------------------------------------------------
# 판정 — 노이즈 대비이지 절대 기준이 아니다
# ---------------------------------------------------------------------------
def _analyze(tmp_path):
    rows, rnd, src, n_sl = _load(tmp_path)
    return anchors.analyze(rows, rnd, EH, round_source=src, n_slices=n_sl)


def test_노이즈만_있으면_통과한다(tmp_path):
    """세그먼트 간 편차가 노이즈와 같은 크기면 통과여야 한다.

    변동폭 자체는 1% 를 넘는다 — 절대 기준이면 실패로 뒤집힌다.
    """
    _write(tmp_path, n_rounds=8, n_seg=4, noise_sd=0.03)
    rep = _analyze(tmp_path)
    assert rep.worst_short > 1.0, "이 픽스처는 절대 1% 기준을 넘어야 의미가 있다"
    assert rep.ok, rep.failures


def test_세그먼트_계통_오차는_잡는다(tmp_path):
    """노이즈는 없고 세그먼트마다 고정 오프셋만 있는 경우."""
    _write(tmp_path, n_rounds=4, n_seg=4,
           offset={0: 1.0, 1: 1.05, 2: 0.95, 3: 1.10}, noise_sd=0.0005)
    rep = _analyze(tmp_path)
    assert not rep.ok
    assert any("세그먼트 간 편차" in f for f in rep.failures)


def test_전체가_함께_드리프트하면_잡는다(tmp_path):
    """세그먼트 간 편차는 0인데 라운드마다 전체가 느려지는 경우.

    비율(sB/sW)만 보면 통과한다. 절대값 추이가 없으면 **아무도 못 잡는다** —
    그리고 그 검사가 라운드 매핑 버그로 죽어 있었다.
    """
    _write(tmp_path, n_rounds=6, n_seg=3,
           round_gain={0: 1.0, 1: 1.02, 2: 1.04, 3: 1.06, 4: 1.08, 5: 1.10},
           noise_sd=0.0005)
    rep = _analyze(tmp_path)
    assert rep.abs_first == 0 and rep.abs_last == 5
    assert rep.abs_worst > 5.0
    assert not rep.ok
    assert any("절대값" in f for f in rep.failures)
    assert any("단조 증가" in f for f in rep.failures)


def test_라운드가_뭉개지면_그_검사는_돌지_않는다(tmp_path):
    """옛 매핑을 흉내낸다 — 라운드를 하나로 만들면 드리프트를 못 잡는다.

    회귀 테스트의 핵심이다. 버그가 있어도 **예외가 안 난다.**
    """
    _write(tmp_path, n_rounds=6, n_seg=3,
           round_gain={0: 1.0, 1: 1.02, 2: 1.04, 3: 1.06, 4: 1.08, 5: 1.10},
           noise_sd=0.0005)
    rows, _rnd, _src, n_sl = _load(tmp_path)
    squashed = [5] * len(rows)          # 옛 코드의 결과: 마지막 라운드만 남음
    rep = anchors.analyze(rows, squashed, EH, round_source="timestamp",
                          n_slices=n_sl)
    assert rep.abs_moves == [] and rep.ok, (
        "이 단언이 깨지면 옛 버그가 드리프트를 잡았다는 뜻이다 — "
        "실제로는 못 잡았고, 그래서 이 매핑을 고쳤다")


def test_짧은_앵커만_판정한다(tmp_path):
    _write(tmp_path, n_rounds=3, n_seg=3)
    rep = _analyze(tmp_path)
    judged = [s for s in rep.stats if s.judged]
    assert judged and all(s.median_ms <= 0.5 for s in judged)


def test_sW_는_sB_와_집계_수준이_같다(tmp_path):
    """분모를 1회 측정 노이즈로 쓰면 비율이 낮아져 판정이 조용히 느슨해진다."""
    _write(tmp_path, n_rounds=8, n_seg=3, noise_sd=0.03)
    st = rep_st = _analyze(tmp_path).stats[0]
    assert st.s_within and st.s_within_slice
    assert st.ratio == pytest.approx(st.s_between / st.s_within, rel=1e-9)
    # 1회 측정 노이즈는 세그먼트 중앙값의 표준오차보다 커야 한다. 그것을
    # 분모로 쓰면 비율이 낮아져 판정이 조용히 느슨해진다.
    assert rep_st.s_within_slice > rep_st.s_within


def test_노이즈_모델과_비교한다(tmp_path):
    _write(tmp_path, n_rounds=3, n_seg=3)
    rep = _analyze(tmp_path)
    for st in rep.stats:
        assert st.model_pct > 0
    # 짧은 앵커의 모델 노이즈가 긴 앵커보다 커야 한다 (절대 지터 성분)
    assert rep.stats[0].model_pct > rep.stats[-1].model_pct


def test_다른_env_hash_의_앵커는_읽지_않는다(tmp_path):
    _write(tmp_path, n_rounds=2, n_seg=2)
    with (tmp_path / "anchors.jsonl").open("a") as f:
        f.write(json.dumps({
            "kernel_id": "k_short", "problem": {"M": 512, "N": 512, "K": 512},
            "time_ms": 99.0, "segment": 0, "when": "start",
            "env_hash": OTHER, "timestamp": _iso(T0)}) + "\n")
    rows, _rnd, _src, _ = _load(tmp_path)
    assert all(r["time_ms"] < 10 for r in rows)


def test_앵커가_없으면_None_이_아니라_빈_리포트(tmp_path):
    rep = anchors.analyze([], [], EH)
    assert rep.stats == [] and rep.ok
    assert "앵커 기록이 없다" in rep.verdict


def test_단조_판정은_중앙값이_아니라_평균으로_한다(tmp_path):
    """중앙값이 눈금에 붙어 안 움직이는데 평균은 움직이는 경우.

    실측 캠페인이 정확히 이 상태였다 — 8 라운드 전부 중앙값 `+0.000%`.
    짧은 앵커는 14 us 인데 CUDA 이벤트 타이머 눈금이 1.024 us 라 한 눈금이
    7.3% 다. 대부분의 측정이 같은 눈금에 떨어져 **중앙값이 둔해진다.**
    단조 판정을 중앙값으로 하면 그 상태에서 드리프트를 통째로 놓친다.

    눈금 위 값만 써서 중앙값을 강제로 고정하고, 라운드마다 위쪽 눈금에
    떨어지는 비율만 늘린다.
    """
    tick, lo = 0.001024, 0.014336
    arows, srows = [], []
    t = T0
    for rnd in range(6):
        for seg in range(3):
            t0 = t
            for when in ("start", "end"):
                # 12개 중 위쪽 눈금 개수만 라운드에 따라 늘린다 (0..5).
                # 5개까지는 중앙값(6,7번째)이 아래 눈금에 그대로 머문다.
                up = rnd
                vals = [lo + tick] * up + [lo] * (12 - up)
                for j, v in enumerate(vals):
                    arows.append({
                        "kernel_id": "k_short",
                        "problem": {"M": 512, "N": 512, "K": 512},
                        "time_ms": v, "segment": seg, "when": when,
                        "env_hash": EH,
                        "timestamp": _iso(t0 + timedelta(
                            seconds=5 + j if when == "start" else SLICE_S - 5)),
                    })
            t = t0 + timedelta(seconds=SLICE_S)
            srows.append({"event": "slice", "round": rnd, "segment": seg,
                          "rc": 0, "seconds": SLICE_S, "timestamp": _iso(t),
                          "env_hash": EH})
    (tmp_path / "anchors.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in arows))
    (tmp_path / "sweep.jsonl").write_text(
        json.dumps({"event": "sweep_start", "env_hash": EH,
                    "timestamp": _iso(T0)}) + "\n"
        + "".join(json.dumps(r) + "\n" for r in srows))

    rep = _analyze(tmp_path)
    med = [pt.rel_pct for pt in rep.rounds]
    assert max(med) - min(med) < 1e-9, "이 픽스처는 중앙값이 고정돼야 의미가 있다"
    assert rep.rounds[-1].mean_pct > rep.rounds[0].mean_pct, (
        "평균은 움직여야 한다 — 그렇지 않으면 이 테스트가 아무것도 안 잡는다")
    assert any("단조 증가" in f for f in rep.failures), (
        "중앙값으로 판정하면 여기서 드리프트를 놓친다")
