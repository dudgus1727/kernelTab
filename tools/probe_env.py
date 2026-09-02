#!/usr/bin/env python3
"""프로브가 **캠페인 조건 안에서** 돌았는지 대조한다.

    python3 tools/probe_env.py --probe-json /tmp/peak.json
    python3 tools/probe_env.py --driver 13030 --nvcc 13.3.73

## 왜 필요한가

측정 도구를 캠페인 조건 밖에서 컴파일·실행하면 그 값은 캠페인 값이 아니다.
**그리고 이것은 조용히 틀린다** — 프로브가 정상적으로 돌고 그럴듯한 값이
나오므로 "쟀다" 와 "캠페인 조건에서 쟀다" 를 결과만 보고 구분할 방법이 없다.

실제로 눈금 32 ns 하나가 그렇게 무효가 됐다 (2026-09-02). 호스트 nvcc 12.8
로 컴파일해 쟀는데 캠페인 이미지는 nvcc 13.3 + compat libcuda 였다.

## ★ 축은 nvcc 가 아니라 libcuda 다

이벤트 눈금·런치 오버헤드·드라이버 경로가 전부 유저 모드 드라이버에 달렸다.
호스트 native 와 이미지 compat 은 **다른 드라이버**다:

    호스트에서 컴파일·실행   native libcuda 580.173.02
    캠페인 이미지 안         compat libcuda 610.43.02

`cudaDriverGetVersion()` 이 그 둘을 가른다 (compat 13030 vs native 13010 등).
nvcc 버전도 함께 보지만, 어긋났을 때 더 무서운 쪽은 드라이버다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kerneltab.core import paths            # noqa: E402


def expected(env_path=None) -> dict:
    env = json.loads(Path(env_path or paths.ENV_JSON).read_text())
    c = env["cuda"]
    return {"driver_api_version": c["cuda_driver_api_version"],
            "nvcc_version": c["nvcc_version"],
            "driver_user_mode": c.get("driver_user_mode"),
            "env_hash": env.get("env_hash", "")[:8]}


def header_stamp(text: str) -> dict:
    """`tick_probe` 의 `# ...` 헤더에서 스탬프를 뽑는다."""
    out = {}
    m = re.search(r"driver_api_version=(\d+)", text)
    if m:
        out["driver_api_version"] = int(m.group(1))
    m = re.search(r"nvcc_version=([\w.]+)", text)
    if m:
        out["nvcc_version"] = m.group(1)
    return out


def check(observed: dict, env_path=None, where: str = "프로브") -> int:
    """어긋나면 4 를 돌려준다. 조용히 통과시키지 않는다."""
    exp = expected(env_path)
    bad = []
    for k in ("driver_api_version", "nvcc_version"):
        if k not in observed:
            bad.append(f"{k}: 프로브가 안 찍었다 — 스탬프 없는 옛 산출물이다")
        elif str(observed[k]) != str(exp[k]):
            bad.append(f"{k}: 프로브 {observed[k]}  vs  env.json {exp[k]}")
    if not bad:
        print(f"[probe-env] {where}: 캠페인 조건 일치 "
              f"(driver {exp['driver_api_version']}, nvcc {exp['nvcc_version']}, "
              f"env {exp['env_hash']})")
        return 0
    print(f"\n⛔ {where} 가 캠페인 조건 밖에서 돌았다.\n  "
          + "\n  ".join(bad)
          + "\n\n  이 값은 캠페인 값이 아니다. **이미지 안에서** 컴파일·실행하라:\n"
            "    docker run --rm -e NVIDIA_DISABLE_REQUIRE=1 --gpus '\"device=<UUID>\"' \\\n"
            "        -v $PWD:/src --entrypoint bash <TAG> -c 'nvcc ... && ./probe'\n"
            "\n  ★ 축은 nvcc 가 아니라 libcuda 다. 호스트 native 와 이미지 compat 은\n"
            f"    다른 드라이버다 (이 캠페인: {exp['driver_user_mode']}).",
          file=sys.stderr)
    return 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-json", help="프로브가 낸 JSON 한 줄 파일")
    ap.add_argument("--probe-header", help="tick_probe CSV (첫 줄의 # 헤더를 본다)")
    ap.add_argument("--driver", type=int, help="직접 지정")
    ap.add_argument("--nvcc")
    ap.add_argument("--env", default=None)
    a = ap.parse_args()

    if a.probe_json:
        obs = json.loads(Path(a.probe_json).read_text().strip().splitlines()[-1])
        where = Path(a.probe_json).name
    elif a.probe_header:
        obs = header_stamp(Path(a.probe_header).read_text()[:2000])
        where = Path(a.probe_header).name
    else:
        obs = {}
        if a.driver is not None:
            obs["driver_api_version"] = a.driver
        if a.nvcc:
            obs["nvcc_version"] = a.nvcc
        where = "인자"
    return check(obs, a.env, where)


if __name__ == "__main__":
    raise SystemExit(main())
