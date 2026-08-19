"""GPU 선택 — **UUID 가 권위이고, 인덱스는 매번 역조회한다** (P-2).

## 왜

`env.json` 의 `device_index` 를 그대로 믿으면 안 된다.

* **컨테이너**: `--gpus '"device=3"'` 로 띄우면 컨테이너 안에서는 그 GPU 가
  인덱스 0 이다. 저장된 3 을 쓰면 존재하지 않는 장치를 잡거나, 더 나쁘게는
  **다른 GPU 를 측정**한다.
* `CUDA_VISIBLE_DEVICES` 가 이미 설정된 환경: 인덱스가 재배치된다.
* 서버에 GPU 를 추가/제거하면 인덱스가 밀린다.

UUID 는 물리 장치에 붙어 있어 이 전부와 무관하다.

## 규칙

1. `CUDA_VISIBLE_DEVICES` 가 **이미 설정되어 있으면 덮어쓰지 않는다.**
   호출자가 의도적으로 좁혀 놓은 것을 스크립트가 되돌리면 안 된다.
2. UUID → 현재 인덱스 역조회. **못 찾으면 명확히 실패한다.**
   조용히 0 으로 폴백하면 다른 GPU 를 측정하고도 모른다
   (`docs/decisions.md` 14번의 "조용히 아무것도 안 하는" 패턴).
3. `nvidia-smi -i` 에도 UUID 를 넘긴다. 인덱스로 넘기면 같은 문제가 난다.
"""

from __future__ import annotations

import os
import subprocess

__all__ = ["DeviceNotFoundError", "list_devices", "index_of_uuid",
           "resolve_device", "smi_target"]


class DeviceNotFoundError(RuntimeError):
    """지정한 UUID 의 GPU 가 이 기계에 없다."""


def list_devices() -> list[tuple[int, str, str]]:
    """`[(index, uuid, name)]`. `nvidia-smi` 로 본 **물리** 목록이다."""
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        raise DeviceNotFoundError(f"nvidia-smi 를 실행할 수 없다: {e}") from e
    if p.returncode != 0:
        raise DeviceNotFoundError(
            f"nvidia-smi 가 실패했다 (rc={p.returncode}): {p.stderr.strip()}")
    out = []
    for line in p.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            out.append((int(parts[0]), parts[1], ",".join(parts[2:])))
    return out


def index_of_uuid(uuid: str) -> int:
    """UUID → 현재 인덱스. **없으면 예외.** 조용히 0 을 돌려주지 않는다."""
    if not uuid:
        raise DeviceNotFoundError(
            "GPU UUID 가 비어 있다. env.json 의 hardware_extra.uuid 를 확인하라.")
    devs = list_devices()
    for idx, u, _ in devs:
        if u == uuid:
            return idx
    raise DeviceNotFoundError(
        f"UUID {uuid} 인 GPU 를 찾을 수 없다.\n"
        f"  이 기계에 보이는 GPU {len(devs)}개:\n"
        + "\n".join(f"    {i}  {u}  {n}" for i, u, n in devs)
        + "\n  컨테이너라면 --gpus 로 그 장치를 넘겼는지 확인하라.\n"
          "  **인덱스로 폴백하지 않는다** — 다른 GPU 를 측정하고도 모르게 된다.")


def resolve_device(env: dict, *, set_visible: bool = True) -> tuple[int, str]:
    """측정에 쓸 GPU 를 정한다. `(현재 인덱스, uuid)`.

    `CUDA_VISIBLE_DEVICES` 가 이미 있으면 **건드리지 않고**, 그 안에서
    0 번이 대상이라고 본다 (호출자가 이미 좁혀 놓은 것이다).
    """
    uuid = (env.get("hardware_extra") or {}).get("uuid") or ""
    preset = os.environ.get("CUDA_VISIBLE_DEVICES")
    if preset is not None and preset != "":
        # 호출자가 정한 것을 존중한다. 다만 그것이 맞는 GPU 인지는 확인한다.
        from core.hardware import device_uuid
        try:
            got = device_uuid(0)
        except Exception:
            got = ""
        if uuid and got and got != uuid:
            raise DeviceNotFoundError(
                f"CUDA_VISIBLE_DEVICES={preset} 로 보이는 GPU 의 UUID 가\n"
                f"  env.json 과 다르다.\n"
                f"    보이는 것: {got}\n"
                f"    기대한 것: {uuid}\n"
                "  다른 GPU 를 측정하려는 것이다. 측정 조건이 어긋난다.")
        return 0, (uuid or got)
    idx = index_of_uuid(uuid)
    if set_visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    return idx, uuid


def smi_target(env: dict) -> str:
    """`nvidia-smi -i` 에 넘길 값. **UUID 를 쓴다.**

    인덱스로 넘기면 컨테이너/재배치 상황에서 다른 GPU 를 가리킨다.
    UUID 가 없으면 인덱스로 떨어지되, 그 사실이 보이도록 문자열로 남긴다.
    """
    uuid = (env.get("hardware_extra") or {}).get("uuid")
    if uuid:
        return uuid
    return str(env.get("device_index", 0))
