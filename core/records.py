"""`results/*.jsonl` 읽기 — **`env_hash` 는 필수 인자다** (R-5).

## 왜 이 모듈이 있는가

`core/table.py` 의 로더는 소비 시점에 `env_hash` 혼재를 막는다. 그러나
**JSONL 을 직접 읽는 경로에는 그 보호가 없었고**, 같은 함정을 다섯 번 밟았다.

| # | 어디 | 증상 |
|---|---|---|
| 1 | `export.py` 의 `difficulty` | 모든 조건을 섞어 계산해 난이도가 **22.05 배** |
| 2 | `bundle.py` 의 번들 통계 | 형상 68개(실제 66), 측정 구간이 폐기 구간부터. **공개 릴리즈 노트에 실릴 뻔했다** |
| 3 | `rehearse.py` 의 드리프트 경고 | 클럭 미고정 리허설이 섞여 변동폭 **55.59%**, 경고 상시 발생 |
| 4 | `rehearse.py` 의 `reproducibility()` | 재현성 기준값을 **다른 조건의 측정치**에서 가져옴 |
| 5 | `recheck_stability.py` | 같은 이유로 재측정 대조가 어긋남 |

전부 "여러 `env_hash` 가 섞인 파일을 필터 없이 읽음" 이다.
`docs/decisions.md` 13번에 교훈을 적어 뒀는데도 그 다음 코드에서 또 밟았다.
**규율이 문서에만 있으면 지켜지지 않는다.** 그래서 구조로 강제한다.

## 규칙

* `env_hash` 에 **기본값이 없다.** 안 쓰면 `TypeError` 다.
* 전체를 보려면 `env_hash=ALL` 을 **명시**해야 한다.
* 형상/조건 단위 집계는 `aggregate_per_env()` 만 쓴다.

`env_hash` 는 조인 키가 아니라 **격리 경계**다. 두 조건의 줄이 같은 집계에
들어가는 경로가 생기면 그 지표는 예외 하나 없이 조용히 틀린다.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

__all__ = [
    "ALL",
    "EnvHashError",
    "aggregate_per_env",
    "env_hashes",
    "iter_records",
    "load_records",
]

#: 전체를 읽겠다는 **명시적** 표시. 실수로 전체를 읽는 일이 없도록
#: 빈 문자열이나 None 은 허용하지 않는다.
ALL = "ALL"

#: 파일별 JSON 파싱 실패 줄 수. 쓰다 만 마지막 줄 하나는 정상이지만
#: 여러 줄이면 손상이다 — 조용히 넘기지 않고 `parse_skips()` 로 드러낸다.
_parse_skips: dict[str, int] = {}


def parse_skips() -> dict[str, int]:
    """지금까지 건너뛴 깨진 줄 수. 진단용."""
    return dict(_parse_skips)


class EnvHashError(ValueError):
    """`env_hash` 를 쓰지 않았거나 파일에 그 조건이 없다."""


def _match(row: dict, want: str) -> bool:
    if want == ALL:
        return True
    got = str(row.get("env_hash") or "")
    return bool(got) and got.startswith(want)


def iter_records(path: str | Path, env_hash: str) -> Iterator[dict]:
    """JSONL 을 한 줄씩 넘긴다. **이 조건의 줄만.**

    `env_hash` 는 앞자리 일치로 본다 (8자만 줘도 된다). 전체를 원하면
    `env_hash=records.ALL` 을 명시하라 — 기본값은 없다.
    """
    if not env_hash:
        raise EnvHashError(
            "env_hash 를 지정해야 한다. 전체를 읽으려면 "
            "core.records.ALL 을 명시하라 (실수로 조건을 섞으면 "
            "지표가 조용히 틀린다).")
    p = Path(path)
    if not p.exists():
        return
    with p.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                # append-only 파일이라 **쓰다 만 마지막 줄**은 정상이다.
                # 다만 여러 줄이 깨졌다면 파일이 손상된 것이므로 센다.
                _parse_skips[str(p)] = _parse_skips.get(str(p), 0) + 1
                continue
            if _match(r, env_hash):
                yield r


def load_records(path: str | Path, env_hash: str) -> list[dict]:
    """`iter_records` 를 리스트로. 큰 파일이면 `iter_records` 를 써라."""
    return list(iter_records(path, env_hash))


def env_hashes(path: str | Path) -> dict[str, int]:
    """파일에 어떤 조건이 몇 줄 있는지. 진단용."""
    out: dict[str, int] = defaultdict(int)
    for r in iter_records(path, ALL):
        out[str(r.get("env_hash") or "(없음)")] += 1
    return dict(out)


def aggregate_per_env(
    rows: Iterable[dict],
    key_fn: Callable[[dict], tuple],
    val_fn: Callable[[dict], Any],
    agg: Callable[[list], Any],
    min_n: int = 1,
) -> dict[tuple, Any]:
    """**`env_hash` 별로** 집계한다. 형상/조건 단위 파생 지표의 유일한 경로다.

    돌려주는 키는 `(env_hash, *key_fn(row))` 다. `env_hash` 가 키에 강제로
    들어가므로 조건을 섞을 수 없다.

    `difficulty` 를 이것 없이 계산했다가 폐기된 드리프트 구간의 느린 시간이
    섞여 난이도가 22 배로 나왔다. 새 집계를 추가할 때는 반드시 이것을 쓴다.
    """
    buf: dict[tuple, list] = defaultdict(list)
    for r in rows:
        v = val_fn(r)
        if v is None:
            continue
        buf[(str(r.get("env_hash") or ""), *tuple(key_fn(r)))].append(v)
    return {k: agg(v) for k, v in buf.items() if len(v) >= min_n}
