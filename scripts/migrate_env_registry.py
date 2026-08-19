#!/usr/bin/env python3
"""P-3 — `env_registry.jsonl` 백필 + `env_hash` 매핑 검증.

`results.jsonl` 은 **건드리지 않는다.** 구 해시를 그대로 두고, 구→신 매핑을
별도 append-only 파일에 남긴다. 그래서 롤백이 파일 하나 삭제로 끝난다.

    python3 scripts/migrate_env_registry.py --check-only
    python3 scripts/migrate_env_registry.py
    python3 scripts/migrate_env_registry.py --allow-orphan ff1f3049

**중단 조건**: `results.jsonl` 에 등장하는 `env_hash` 중 레지스트리에 없는
것이 하나라도 있으면 중단한다. 매핑 불가능한 데이터를 남긴 채 진행하면
"어떤 조건에서 잰 것인지 모르는 줄" 이 표에 섞인다.

orphan 은 **명시적으로 지정**해야 통과한다. 기본은 중단이다.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build import paths  # noqa: E402
from core.env_hash import env_hash_v2, hash_inputs  # noqa: E402
from core.records import ALL, iter_records  # noqa: E402

REGISTRY = paths.RESULTS_DIR / "env_registry.jsonl"


def discover_env_files() -> dict[str, dict]:
    """`results/env*.json` 을 전부 읽어 구 해시 -> env 로."""
    out: dict[str, dict] = {}
    for f in sorted(paths.RESULTS_DIR.glob("env*.json")):
        try:
            e = json.loads(f.read_text())
        except Exception as ex:
            print(f"  !! {f.name} 파싱 실패: {ex}")
            continue
        h = e.get("env_hash")
        if not h:
            continue
        if h in out and out[h]["_file"] != f.name:
            print(f"  !! 같은 env_hash 를 가진 파일이 둘: "
                  f"{out[h]['_file']}, {f.name}")
        e["_file"] = f.name
        out[h] = e
    return out


def hashes_in_results() -> dict[str, int]:
    seen: dict[str, int] = {}
    for r in iter_records(paths.RESULTS_DIR / "results.jsonl", ALL):
        h = str(r.get("env_hash") or "(없음)")
        seen[h] = seen.get(h, 0) + 1
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true",
                    help="레지스트리를 쓰지 않고 매핑만 확인한다")
    ap.add_argument("--allow-orphan", action="append", default=[],
                    metavar="HASH8",
                    help="env.json 이 남아 있지 않은 해시를 명시적으로 허용. "
                         "여러 번 줄 수 있다")
    a = ap.parse_args()

    envs = discover_env_files()
    print(f"env 파일 {len(envs)}종")
    for h, e in sorted(envs.items(), key=lambda kv: kv[1]["_file"]):
        print(f"  {e['_file']:32} {h[:8]} -> v2 {env_hash_v2(e)[:8]}")

    used = hashes_in_results()
    print(f"\nresults.jsonl 에 등장하는 env_hash {len(used)}종")
    allow = {x[:8] for x in a.allow_orphan}
    orphans, mapped = [], []
    for h, n in sorted(used.items(), key=lambda kv: -kv[1]):
        if h in envs:
            mapped.append(h)
            print(f"  {h[:8]}  {n:>9,}줄   OK ({envs[h]['_file']})")
        elif h[:8] in allow:
            orphans.append(h)
            print(f"  {h[:8]}  {n:>9,}줄   orphan (허용됨)")
        else:
            orphans.append(h)
            print(f"  {h[:8]}  {n:>9,}줄   **매핑 실패** — env.json 이 없다")

    unallowed = [h for h in orphans if h[:8] not in allow]
    if unallowed:
        print(f"\n!! 매핑 실패 {len(unallowed)}종. 중단한다.")
        print("   어떤 조건에서 잰 것인지 모르는 줄을 남긴 채 진행하면 안 된다.")
        print("   폐기 데이터라 무시해도 된다면 명시적으로 허용하라:")
        for h in unallowed:
            print(f"     --allow-orphan {h[:8]}")
        return 3

    print(f"\n매핑 성공 {len(mapped)}종, orphan {len(orphans)}종 (전부 명시 허용)")
    if a.check_only:
        return 0

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    lines = []
    for h, e in envs.items():
        env = {k: v for k, v in e.items() if k != "_file"}
        lines.append({
            "env_hash": h, "env_hash_v2": env_hash_v2(e),
            "source_file": e["_file"], "recorded_utc": now,
            "n_rows": used.get(h, 0),
            "hash_inputs": hash_inputs(e),
            "env": env,
        })
    for h in orphans:
        lines.append({
            "env_hash": h, "env_hash_v2": None, "source_file": None,
            "recorded_utc": now, "n_rows": used.get(h, 0),
            "hash_inputs": None, "env": None,
            "note": "orphan — env.json 미보존. 조건을 복원할 수 없으므로 "
                    "분석에서 제외한다",
        })
    with REGISTRY.open("a") as f:
        for ln in lines:
            f.write(json.dumps(ln, ensure_ascii=False, default=str) + "\n")
    print(f"{REGISTRY} 에 {len(lines)}줄 append")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
