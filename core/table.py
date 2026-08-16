"""`results/table.parquet` 소비 인터페이스 — 정답 누출을 구조적으로 막는다.

이 저장소는 `Problem` / `KernelConfig` / `Hardware` / `RuntimeConfig` 어디에도
측정 시간을 넣지 않아 **자료구조 수준에서** 누출을 막는다. 하지만 parquet 은
평면 테이블이라 그 보호가 없다. "정답 컬럼을 drop 하세요" 라고 문서에 적는
것으로는 부족하다 — 문서는 지켜지지 않는다.

그래서 두 개의 로더를 제공한다. 소비 프로젝트가 이 둘만 쓰면 **실수로 정답을
노출할 수 없다.**

    from core.table import load_for_ranking, load_for_scoring

    X = load_for_ranking("results/table.parquet")   # 규칙 함수에 넘겨도 되는 것
    y = load_for_scoring("results/table.parquet")   # 채점만 하는 쪽이 보는 것

pyarrow / pandas 는 이 모듈 안에서만 import 한다. 측정 경로는 이 모듈을
쓰지 않으므로 의존성이 늘어나지 않는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = [
    "ANSWER_COLS",
    "OUTCOME_COLS",
    "SAFE_META_COLS",
    "load_for_ranking",
    "load_for_scoring",
    "assert_no_answers",
    "AnswerLeakError",
]


class AnswerLeakError(RuntimeError):
    """규칙에 넘기면 안 되는 컬럼이 섞였다."""


#: **정답.** 측정 결과 그 자체이거나 거기서 직접 유도된 값.
#: 규칙 함수의 입력이 되면 그 규칙은 아무것도 배우지 않고 정답을 베낀다.
ANSWER_COLS = (
    "time_ms", "time_std_ms", "time_min_ms", "time_max_ms",
    "n_reps", "outlier_frac",
    "cublas_ms",
    "tflops", "frac_of_peak", "vs_cublas",
)

#: 정답은 아니지만 **측정을 해봐야 아는 값.** 규칙이 이것을 쓰면
#: "돌려보고 고른다" 가 되므로 기본적으로 함께 제외한다.
#: (빌드 시점에 알 수 있는 `launchable` / `smem_matches` / `hmma_matches` 는
#:  여기 넣지 않는다 — 그것들은 써도 된다.)
OUTCOME_COLS = (
    "status", "error", "max_rel_error",
    "actual_split_k",          # 런타임에 CUTLASS 가 정하는 실제 슬라이스 수
    "sm_clock_mhz", "mem_clock_mhz", "gpu_temp_c", "power_w",  # 측정 시점 상태
    "timestamp",
)

#: 규칙에 넘겨도 되지만 **피처가 아니라 메타데이터**인 것.
#: 조인·필터에는 필요하므로 남기되, 규칙이 이것으로 학습하지 않도록 주의.
SAFE_META_COLS = ("kernel_id", "env_hash", "arch", "cutlass_commit", "nvcc_arch")


def _read(path: str | Path, env_hash: str | None, ok_only: bool):
    import pyarrow.parquet as pq

    df = pq.read_table(path).to_pandas()
    if env_hash:
        # 측정 조건이 다른 데이터를 섞으면 절대 시간이 비교 불가능해진다.
        df = df[df["env_hash"].astype(str).str.startswith(env_hash)]
    elif df["env_hash"].nunique() > 1:
        raise AnswerLeakError(
            f"table 에 측정 조건이 {df['env_hash'].nunique()} 종 섞여 있다. "
            "env_hash 를 지정해 하나만 고르라 — 조건이 다르면 절대 시간을 "
            "비교할 수 없다.\n  " + ", ".join(sorted(df["env_hash"].unique())[:5])
        )
    if ok_only:
        df = df[df["status"] == "ok"]
    return df.reset_index(drop=True)


def load_for_ranking(
    path: str | Path,
    env_hash: str | None = None,
    ok_only: bool = True,
    keep_outcomes: bool = False,
) -> "pd.DataFrame":
    """규칙 함수에 넘길 수 있는 형태. **정답 컬럼이 제거되어 있다.**

    Parameters
    ----------
    env_hash
        측정 조건. 지정하지 않았는데 표에 여러 조건이 섞여 있으면 예외.
    ok_only
        `status == "ok"` 만 남긴다 (기본). 실패는 결측이 아니라 명시 기록이므로
        규칙 입력에서는 빼는 것이 맞다.
    keep_outcomes
        True 면 `OUTCOME_COLS` 를 남긴다. **권장하지 않는다.** 진단용.
    """
    df = _read(path, env_hash, ok_only)
    drop = list(ANSWER_COLS) + ([] if keep_outcomes else list(OUTCOME_COLS))
    return df.drop(columns=[c for c in drop if c in df.columns])


def load_for_scoring(
    path: str | Path,
    env_hash: str | None = None,
    ok_only: bool = True,
) -> "pd.DataFrame":
    """채점용. 정답을 포함한다. **규칙 함수에 넘기면 안 된다.**

    조인 키(`kernel_id`, 형상, 런타임 config)와 정답만 남긴다. 피처를 함께
    돌려주지 않는 것은 의도다 — 이 반환값을 그대로 규칙에 넘기는 사고를
    막는다.
    """
    df = _read(path, env_hash, ok_only)
    keys = [c for c in ("kernel_id", "env_hash", "M", "N", "K",
                        "split_k", "split_k_mode") if c in df.columns]
    ans = [c for c in ANSWER_COLS + ("status",) if c in df.columns]
    return df[keys + ans]


def assert_no_answers(df, where: str = "규칙 입력") -> None:
    """정답 컬럼이 섞이지 않았는지 확인한다. 소비 쪽 단위 테스트용.

    로더를 우회해 직접 만든 DataFrame 을 규칙에 넘기기 전에 부른다.
    """
    bad = [c for c in ANSWER_COLS if c in getattr(df, "columns", ())]
    if bad:
        raise AnswerLeakError(
            f"{where} 에 정답 컬럼이 섞였다: {bad}\n"
            "  core.table.load_for_ranking() 을 쓰거나 이 컬럼들을 drop 하라."
        )
