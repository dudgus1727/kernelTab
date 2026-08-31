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

종료 코드
---------
====  ===================================================================
0     정상
2     입력 문제
4     **축 구멍** — 근거 없는 공간 밖 값이 있다. 우리 공간을 넓히거나
      근거를 적으면 없어진다.
5     ★ **3.x 추천 검출** — 벤더가 다른 API 공간의 커널을 추천한다.
      축을 넓혀도 안 없어진다. 사유가 다르므로 코드도 다르다.
====  ===================================================================

⚠️ 이 스크립트를 호출하는 쪽은 **`!= 0` 으로 보라.** `== 4` 로 보면
5 를 놓친다.
"""
from __future__ import annotations

import argparse
import json
import re
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

#: `raw` 문자열에서 3.x 파라미터를 읽는 대비책.
#:
#: `--extract` 가 `cluster`/`instr` 를 **구조화해서** 기록하기 시작한 것은
#: 2026-09-01 이다. 그 전 JSON 에는 필드가 없다 — 그런데 검출기가 `.get()`
#: 으로만 보면 **옛 파일은 조용히 통과한다.** 그것이 "경고가 안 뜬다" 의
#: 나쁜 쪽이다 (decisions.md 27). `raw` 에는 항상 들어 있으므로 거기서 읽는다.
_RAW_CLUSTER = re.compile(r"cluster\((\d+)\s+(\d+)\)")
_RAW_INSTR = re.compile(r"instr\((\d+)\s+(\d+)\s+(\d+)\)")


def three_x_params(rec: dict) -> tuple | None:
    """`(cluster, instr)` 또는 **읽을 수 없으면 None.**

    None 은 "3.x 가 아니다" 가 아니라 **"모른다"** 다. 둘을 섞지 마라.
    """
    cluster = tuple(rec["cluster"]) if rec.get("cluster") else None
    instr = tuple(rec["instr"]) if rec.get("instr") else None
    raw = rec.get("raw") or ""
    if cluster is None and (m := _RAW_CLUSTER.search(raw)):
        cluster = (int(m[1]), int(m[2]))
    if instr is None and (m := _RAW_INSTR.search(raw)):
        instr = (int(m[1]), int(m[2]), int(m[3]))
    if cluster is None and instr is None:
        return None
    return cluster, instr


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
    # 축 목록은 **백엔드에서** 읽는다. backends.cutlass_v2 을 직접 import 하면
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
    # ★ 3.x 커널을 추천하고 있지 않은가 (docs/baselines.md).
    #    우리 표는 CUTLASS 2.x 공간이다. 벤더가 3.x 를 추천하면
    #    "가장 가까운 config" 로 대체돼 **의도와 다른 커널이 채점된다** —
    #    그리고 그것은 축 구멍이 아니라 **API 차이**라 축을 넓혀도 안 없어진다.
    #    2026-09-01 확인 시점에는 target=CUTLASS 가 전부 (1,1)/(16,8,16) 이었다.
    #
    #    ⚠️ 0건이어도 이 검사를 지우지 마라. "경고가 안 뜬다" 는
    #    **정상**과 **감시가 죽었다**를 다 뜻한다 (decisions.md 27).
    three_x, unknown_3x, seen_3x = [], 0, 0
    for sh, lst in data.items():
        for rec in lst:
            got = three_x_params(rec)
            if got is None:
                unknown_3x += 1
                continue
            seen_3x += 1
            cluster, instr = got
            if (cluster and cluster != (1, 1)) or (instr and instr != (16, 8, 16)):
                three_x.append((sh, cluster, instr))
    if not seen_3x:
        print("\n⛔ 3.x 검출기가 돌 수 없다 — cluster/instr 를 읽을 수 있는 "
              f"추천이 하나도 없다 ({unknown_3x:,}건 전부 불명).")
        print("   `--extract` 를 다시 돌려서 cluster/instr 를 기록하게 하라.")
        print("   ★ '경고가 없다' 를 '이상 없다' 로 읽으면 안 된다 "
              "(decisions.md 27).")
        return 2
    if three_x:
        print(f"\n⛔ 3.x 전용 파라미터를 가진 추천 {len(three_x)}건 "
              f"(cluster != (1,1) 또는 instr != (16,8,16)):")
        for sh, cluster, instr in three_x[:5]:
            print(f"     {sh:22s} cluster={cluster} instr={instr}")
        if len(three_x) > 5:
            print(f"     ... 외 {len(three_x) - 5}건")
        print("\n   target=CUTLASS 를 쓰는데 3.x 가 나온다면 라이브러리 "
              "기본값이 바뀌었거나\n"
              "   버전이 올라간 것이다. ★ 이 표는 2.x 공간이므로 "
              "**벤더 비교가 불공정해진다** —\n"
              "   이 추천들은 '가장 가까운 config' 로 대체되어 채점된다.\n"
              "   축 구멍이 아니라 API 차이다. 축을 넓혀도 안 없어진다.\n"
              "\n   할 일: scripts/baseline_vendor.py 의 target 지정을 "
              "확인하라\n"
              "         (nvMatmulHeuristicsTarget.CUTLASS 여야 한다. "
              "CUTLASS3 는 3.x 다).\n"
              "         근거는 docs/baselines.md 의 "
              "'벤더 휴리스틱은 2.x 파라미터만 낸다'.\n")

    print(f"추천   {n_rec:,}개 / 형상 {len(data)}개")
    print("축     " + ", ".join(f"{k} {len(v)}" for k, v in sorted(space.items())))

    if not outside:
        print("\n공간 밖 추천 없음. 축 덮개 이상 없다.")
        return 5 if three_x else 0

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
        # 3.x 가 섞여 있으면 그쪽이 먼저다 — **공간 자체가 다르면**
        # 축 구멍 판정을 그대로 믿을 수 없다.
        return 5 if three_x else 4

    print("\n공간 밖 값이 전부 근거가 있다. 축 덮개 이상 없다.")
    return 5 if three_x else 0


if __name__ == "__main__":
    raise SystemExit(main())
