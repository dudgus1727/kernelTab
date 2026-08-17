"""번들 로더 — `kernelrule` 이 쓰는 진입점 (C-2).

`core/table.py` 의 정답 제거 규약을 **번들 단위로** 감싼다. 파일 하나가 아니라
번들을 다루는 이유는, `table.parquet` 만으로는 해석이 불가능하기 때문이다 —
`env.json` 이 없으면 유효 ridge point 를 모르고 `is_memory_bound` 가 전부
틀린다.

    from core.bundle import load_bundle, load_bundles

    b = load_bundle("rtx-a6000-sm_86-368a84f1")
    X = b.ranking()          # 규칙 입력 (정답 제거됨)
    y = b.scoring()          # 채점용

    df = load_bundles(["rtx-a6000-...", "rtx-4090-..."],
                      common_shapes_only=True)    # 전이 실험

번들 경로는 `KERNELTAB_DATASETS` 환경변수 또는 절대 경로로 준다.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from core.table import AnswerLeakError, load_for_ranking, load_for_scoring

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = ["Bundle", "load_bundle", "load_bundles", "BundleError",
           "resolve_bundle_path"]


class BundleError(RuntimeError):
    """번들이 손상됐거나 규약을 어겼다."""


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def resolve_bundle_path(ref: str | Path) -> Path:
    """번들 이름 또는 경로를 실제 디렉토리로 푼다.

    탐색 순서: 절대/상대 경로 → `$KERNELTAB_DATASETS/<ref>` → `./datasets/<ref>`
    """
    p = Path(ref)
    if (p / "BUNDLE.json").exists():
        return p.resolve()
    roots = []
    envv = os.environ.get("KERNELTAB_DATASETS")
    if envv:
        roots += [Path(x) for x in envv.split(os.pathsep) if x]
    roots.append(Path(__file__).resolve().parent.parent / "datasets")
    for r in roots:
        cand = r / str(ref)
        if (cand / "BUNDLE.json").exists():
            return cand.resolve()
    raise BundleError(
        f"번들 '{ref}' 를 찾지 못했다.\n"
        f"  탐색: {[str(p)] + [str(r / str(ref)) for r in roots]}\n"
        "  KERNELTAB_DATASETS 환경변수로 데이터셋 루트를 지정하라.")


@dataclass(frozen=True)
class Bundle:
    path: Path
    info: dict

    # -- 기본 정보 ---------------------------------------------------------
    @property
    def bundle_id(self) -> str:
        return self.info["bundle_id"]

    @property
    def env_hash(self) -> str:
        return self.info["env_hash"]

    @property
    def table_path(self) -> Path:
        return self.path / "table.parquet"

    def env(self) -> dict:
        return json.loads((self.path / "env.json").read_text())

    def shape_layers(self) -> dict:
        return self.info.get("shape_layers", {})

    def shapes(self) -> set[tuple[int, int, int]]:
        return {tuple(s) for v in self.shape_layers().values() for s in v}

    # -- 로더 --------------------------------------------------------------
    def ranking(self, **kw) -> "pd.DataFrame":
        """규칙 입력. 정답 컬럼이 제거되어 있다."""
        return _tag(load_for_ranking(self.table_path,
                                     env_hash=self.env_hash[:16], **kw), self)

    def scoring(self, **kw) -> "pd.DataFrame":
        """채점용. 정답 포함. **규칙 함수에 넘기면 안 된다.**"""
        return _tag(load_for_scoring(self.table_path,
                                     env_hash=self.env_hash[:16], **kw), self)


def _tag(df, b: Bundle):
    """행마다 어느 번들에서 왔는지 표시한다. 여러 번들을 합칠 때 필요하다."""
    df = df.copy()
    df["bundle_id"] = b.info["bundle_id"]
    df["gpu_name"] = b.info["gpu_name"]
    df["arch"] = b.info["arch"]
    df["sm_count"] = b.info["sm_count"]
    return df


def load_bundle(ref: str | Path, verify: bool = True) -> Bundle:
    """번들 하나를 연다.

    verify=True(기본) 면 `BUNDLE.json` 에 기록된 sha256 과 실제 파일을
    대조한다. 불일치면 예외 — 배포 중 손상되거나 누군가 표만 바꿔치기한
    것을 조용히 넘기면 안 된다.
    """
    path = resolve_bundle_path(ref)
    info = json.loads((path / "BUNDLE.json").read_text())
    if verify:
        bad = []
        for name, meta in (info.get("files") or {}).items():
            f = path / name
            if not f.exists():
                bad.append(f"{name}: 파일 없음")
                continue
            if f.stat().st_size != meta["bytes"]:
                bad.append(f"{name}: 크기 {f.stat().st_size} != {meta['bytes']}")
            elif _sha256(f) != meta["sha256"]:
                bad.append(f"{name}: sha256 불일치")
        if bad:
            raise BundleError(
                f"번들 {path.name} 무결성 검사 실패:\n  " + "\n  ".join(bad))
    return Bundle(path, info)


def load_bundles(
    refs: Iterable[str | Path],
    common_shapes_only: bool = False,
    kind: str = "ranking",
    verify: bool = True,
    **kw,
) -> "pd.DataFrame":
    """여러 번들을 하나의 표로 합친다.

    Parameters
    ----------
    common_shapes_only
        **모든 번들에 존재하는 (M, N, K) 만 남긴다.**

        이 옵션이 필요한 이유: 형상 그리드의 층 C 는 `waves` 를 고정하고
        `sm_count` 에서 M 을 **역산**한다. 그래서 A6000(84 SM)과
        4090(128 SM)의 형상 목록이 다르다. 이 구분 없이 전이 실험을 하면
        규칙이 나빠서 성능이 떨어진 것인지 **애초에 형상이 달라서**인지
        구분할 수 없다.

        전이 실험(한 GPU 에서 배운 규칙을 다른 GPU 에 적용)에서는 반드시
        True 로 둔다. 단일 GPU 분석이면 무관하다.
    kind
        `"ranking"`(정답 제거, 기본) 또는 `"scoring"`(정답 포함).
    verify
        각 번들의 sha256 대조.

    컬럼은 합집합으로 맞춘다. 어느 번들에 없는 컬럼은 null 이 된다
    (SM90 의 `ext_cluster_*` 등). Parquet/pandas 모두 null 을 잘 다룬다.
    """
    import pandas as pd

    bundles = [load_bundle(r, verify=verify) for r in refs]
    if not bundles:
        raise BundleError("번들이 하나도 없다")

    ids = [b.bundle_id for b in bundles]
    if len(set(ids)) != len(ids):
        raise BundleError(f"같은 번들이 중복 지정됐다: {ids}")

    frames = [(b.ranking(**kw) if kind == "ranking" else b.scoring(**kw))
              for b in bundles]

    if common_shapes_only:
        common = None
        for f in frames:
            s = set(map(tuple, f[["M", "N", "K"]].to_numpy().tolist()))
            common = s if common is None else (common & s)
        if not common:
            raise BundleError(
                "번들들 사이에 공통 형상이 하나도 없다.\n"
                "  층 C 는 sm_count 에서 M 을 역산하므로 GPU 마다 다르다. "
                "층 A/B/D/E 는 GPU 무관이므로 보통 공통 형상이 존재한다 — "
                "0 이면 형상 그리드 정의가 서로 다른 버전일 가능성이 크다.")
        frames = [f[[tuple(x) in common for x in
                     f[["M", "N", "K"]].to_numpy().tolist()]] for f in frames]

    out = pd.concat(frames, ignore_index=True, sort=False)
    # 합친 뒤에도 정답이 없어야 한다 (ranking 인 경우)
    if kind == "ranking":
        from core.table import ANSWER_COLS
        leaked = [c for c in ANSWER_COLS if c in out.columns]
        if leaked:
            raise AnswerLeakError(f"번들 결합 후 정답 컬럼이 남았다: {leaked}")
    return out
