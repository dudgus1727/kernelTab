"""축 덮개 점검 — **우리 탐색 공간 밖을 남이 추천하는가.**

캠페인을 돌리기 **전에** 한 번 돌린다. 외부 휴리스틱(nvMatmulHeuristics)이
고르는 config 를 우리 열거 축과 대조해서, 우리가 아예 안 재는 값이 있으면
드러낸다. 33시간을 쓰고 나서 "그 값을 안 쟀네" 를 알면 늦다.

    python3 scripts/check_axis_coverage.py --vendor docs/baselines/vendor_a6000_828baa64.json

**두 가지를 구분하는 것이 요점이다:**

| | 무엇인가 | 어떻게 해야 하나 |
|---|---|---|
| 진짜 구멍 | 우리 백엔드에서 유효한데 축에 없다 | **축에 넣는다** |
| 남의 공간 | 우리 백엔드에서 성립하지 않는다 | `KNOWN` 에 근거와 함께 기록 |

벤더 휴리스틱은 cuBLASLt 커널을 겨냥하므로 같은 이름의 축이 다른 것을
뜻할 수 있다. 그래서 "밖에 있다" 만으로는 결론이 안 나고, **백엔드에서
직접 확인한 근거**를 요구한다.

종료 코드: 0 정상 / 4 근거 없는 공간 밖 값 있음 / 2 입력 문제.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kerneltab.backends import get_backend

#: 공간 밖인데 **근거가 확인된** 값. 근거 없이는 여기 넣지 마라.
#:
#: 형식: (축, 값) -> 근거 한 줄. 근거는 "확인했다" 가 아니라 **무엇을 어떻게
#: 확인했는지**여야 한다.
KNOWN = {
    ("stages", 1): (
        "CUTLASS 2.x OpClassTensorOp 에서 **수치가 틀린다.** sm_86, "
        "tb=128x128x32 / warp=64x64x32, fp16 in / fp32 accum, "
        "M=N=256 K=128 으로 확인: nvcc OK, "
        "can_implement()=kSuccess, 실행 OK 인데 결과가 65,536 원소 중 "
        "62,674 개 불일치 (최대 상대오차 32). stages=2/3 은 정확히 일치. "
        "SASS 상 stages=1 은 LDGSTS 8개(멀티스테이지 경로)로 stages=2 "
        "(LDGSTS 0, MmaPipelined)와 다른 커널이다. "
        "**can_implement 가 통과시키므로 열거기로는 못 거른다** — "
        "축에서 빼는 것이 유일한 방어다. (2026-08-21 확인)"),
}

#: 벤더 JSON 한 항목에서 축 값을 뽑는 방법.
def _axis_values(rec: dict) -> dict:
    cta = rec.get("cta") or [None, None, None]
    warp = rec.get("warp") or [None, None, None]
    return {
        "tb_tile": (cta[0], cta[1]),
        "tile_k": cta[2],
        "warp_tile": (warp[0], warp[1]),
        "warp_k": warp[2],
        "stages": rec.get("stages"),
        "split_k": rec.get("split_k"),
        "swizzle_n": rec.get("swizzle"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vendor", required=True,
                    help="baseline_vendor.py --extract 로 뽑은 JSON")
    ap.add_argument("--arch", default="sm_86")
    ap.add_argument("--top", type=int, default=None,
                    help="형상당 상위 N개만 본다 (기본 전부)")
    a = ap.parse_args()

    path = Path(a.vendor)
    if not path.is_file():
        print(f"입력이 없다: {path}", file=sys.stderr)
        return 2
    data = json.loads(path.read_text())
    meta = data.pop("_meta", {})
    # 축 목록은 **백엔드에서** 읽는다. backends.sm80 을 직접 import 하면
    # Protocol 규약이 깨진다 (decisions.md 12).
    space = get_backend(a.arch).axis_space()
    outside: dict[tuple, Counter] = defaultdict(Counter)   # (축,값) -> 형상별 수
    shapes_hit: dict[tuple, set] = defaultdict(set)
    n_rec = 0
    for shape, recs in data.items():
        for rec in (recs[:a.top] if a.top else recs):
            n_rec += 1
            for axis, val in _axis_values(rec).items():
                if val is None or None in (val if isinstance(val, tuple) else ()):
                    continue
                if axis not in space:
                    continue
                if val not in space[axis]:
                    outside[(axis, val)][shape] += 1
                    shapes_hit[(axis, val)].add(shape)

    print(f"입력   {path.name}  ({meta.get('gpu', '?')}, "
          f"env_hash {str(meta.get('env_hash', '?'))[:8]}, count={meta.get('count')})")
    print(f"추천   {n_rec:,}개 / 형상 {len(data)}개")
    print("축     " + ", ".join(f"{k} {len(v)}" for k, v in sorted(space.items())))

    if not outside:
        print("\n공간 밖 추천 없음. 축 덮개 이상 없다.")
        return 0

    n_out = sum(sum(c.values()) for c in outside.values())
    print(f"\n공간 밖 추천 {n_out:,}/{n_rec:,} ({100 * n_out / n_rec:.1f}%)")
    print(f"\n{'축':>10} {'값':>12} {'추천수':>7} {'형상':>6}  판정")
    unexplained = []
    for (axis, val), cnt in sorted(outside.items(),
                                   key=lambda kv: -sum(kv[1].values())):
        n = sum(cnt.values())
        known = KNOWN.get((axis, val))
        print(f"{axis:>10} {val!s:>12} {n:>7,} {len(shapes_hit[(axis, val)]):>6}  "
              + ("근거 있음" if known else "**근거 없음**"))
        if not known:
            unexplained.append((axis, val, n, sorted(shapes_hit[(axis, val)])[:3]))

    for (axis, val), why in KNOWN.items():
        if (axis, val) in outside:
            print(f"\n[{axis}={val}] {why}")

    if unexplained:
        print("\n" + "=" * 70)
        print("근거 없는 공간 밖 값이 있다. 캠페인 전에 결론을 내라.")
        print("=" * 70)
        for axis, val, n, ex in unexplained:
            print(f"  {axis}={val}  추천 {n:,}회  예: {', '.join(ex)}")
        print("\n두 갈래다:")
        print("  (a) 우리 백엔드에서 **유효하다**  -> 축에 넣고 다시 열거한다")
        print("  (b) 우리 백엔드에서 **성립 안 한다** -> 확인한 근거를 "
              "KNOWN 에 적는다")
        print("\n확인 방법: 그 값으로 커널 하나를 emit_cpp -> nvcc -> "
              "can_implement -> **참조값과 대조**까지 돌린다.")
        print("⚠️ can_implement() 통과를 근거로 삼지 마라 — stages=1 이 "
              "통과하고도 62,674/65,536 원소가 틀렸다.")
        return 4

    print("\n공간 밖 값이 전부 근거가 있다. 축 덮개 이상 없다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
