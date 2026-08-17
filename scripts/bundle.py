#!/usr/bin/env python3
"""데이터 배포 번들 생성 (C-1).

`results/` 는 gitignore 대상이라 **표를 kernelrule 에 넘길 경로가 없다.**
그리고 `table.parquet` 만 넘기면 해석이 불가능하다 — `env.json` 이 없으면
유효 ridge point 를 모르고, 그러면 `is_memory_bound` 가 전부 틀린다.

그래서 배포 단위를 파일이 아니라 **번들**로 정의한다.

    datasets/{gpu_slug}-{arch}-{env_hash8}/
        table.parquet          측정 표 (파생 지표 포함)
        env.json               측정 조건 (클럭, 실효 피크/대역폭, 프로토콜)
        kernels.jsonl          커널당 1줄 (정적 분석 결과)
        manifest.json          코드/CUTLASS/패키지 버전
        BUNDLE.json            위 전부의 요약 + 체크섬
        validate_report.md     무결성 검사 결과

디렉토리명에 `env_hash` 가 들어가는 것이 핵심이다. 같은 A6000 이라도 클럭
조건이 다르면 다른 번들이며 **섞으면 안 된다.**

    python3 scripts/bundle.py                       # 현재 env.json 조건
    python3 scripts/bundle.py --env-hash b42df475
    python3 scripts/bundle.py --archive             # + tar.zst
    python3 scripts/bundle.py --skip-validate       # (권장하지 않음)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build import paths  # noqa: E402
from core.hardware import hardware_from_env  # noqa: E402
from core.shapes import all_layers  # noqa: E402

DATASETS = REPO_ROOT / "datasets"
RESULTS = paths.RESULTS_DIR / "results.jsonl"
KERNELS = paths.RESULTS_DIR / "kernels.jsonl"
TABLE = paths.RESULTS_DIR / "table.parquet"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def gpu_slug(name: str) -> str:
    """'NVIDIA RTX A6000' -> 'rtx-a6000'."""
    s = name.lower().replace("nvidia", "").replace("geforce", "")
    return "-".join(s.split()) or "gpu"


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return str(n)


def measurement_running() -> bool:
    """측정이 도는 중인가. heartbeat.json 의 pid 를 /proc 로 확인한다.

    pgrep 을 쓰지 않는다 — 감시하는 쪽 명령줄에 패턴이 들어가 자기 자신을
    찾는다 (scripts/waitpid.sh 참조).
    """
    hb = paths.RESULTS_DIR / "heartbeat.json"
    if not hb.exists():
        return False
    try:
        pid = json.loads(hb.read_text()).get("pid")
    except Exception:
        return False
    return bool(pid) and Path(f"/proc/{pid}").exists()


def run_validate(env_hash: str, out: Path) -> tuple[bool, str]:
    """validate_table.py --expect full 을 돌려 결과를 문자열로 받는다."""
    p = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "validate_table.py"),
         "--expect", "full", "--env-hash", env_hash],
        capture_output=True, text=True, cwd=REPO_ROOT)
    return p.returncode == 0, (p.stdout + p.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-hash", default=None)
    ap.add_argument("--out", default=str(DATASETS))
    ap.add_argument("--archive", action="store_true",
                    help="tar.zst 압축본과 .sha256 도 만든다")
    ap.add_argument("--archive-raw", action="store_true",
                    help="results.jsonl 원본을 별도 압축 보관 (C-4). 번들에는 "
                         "넣지 않는다 — table.parquet 은 파생물이라 계산식이 "
                         "바뀌면 원본에서 다시 만들어야 한다")
    ap.add_argument("--skip-validate", action="store_true",
                    help="무결성 검사를 건너뛴다. 검증 안 된 데이터를 배포하게 "
                         "되므로 진단 목적에만 쓸 것")
    args = ap.parse_args()

    if not TABLE.exists():
        print(f"{TABLE} 가 없다. 먼저 scripts/export.py 를 돌려라.")
        return 2

    env = json.loads(paths.ENV_JSON.read_text())
    env_hash = args.env_hash or env["env_hash"]
    if not env["env_hash"].startswith(env_hash):
        print(f"경고: --env-hash {env_hash} 가 현재 env.json "
              f"({env['env_hash'][:16]}) 과 다르다.")
        print("  번들의 env.json 은 현재 파일이 되므로 조건이 어긋난다. 중단한다.")
        return 2

    hw = hardware_from_env(env)
    bundle_id = f"{gpu_slug(hw.name)}-{hw.arch}-{env_hash[:8]}"
    out = Path(args.out) / bundle_id

    # --- C-1: 검증을 통과하지 못하면 번들을 만들지 않는다 -------------------
    report = "(건너뜀)"
    if args.skip_validate:
        print("!! --skip-validate: 무결성 검사를 건너뛴다")
    else:
        print("무결성 검사 중 (validate_table.py --expect full)...")
        ok, report = run_validate(env_hash, out)
        if not ok:
            print(report[-4000:])
            print("\n!! 무결성 검사 실패 — 번들을 만들지 않는다.")
            print("   검증 안 된 데이터가 배포되면 안 된다. "
                  "docs/post_measurement.md 3절의 대응을 따르라.")
            return 4
        print("  통과")

    out.mkdir(parents=True, exist_ok=True)

    # --- 파일 복사 ---------------------------------------------------------
    shutil.copy2(TABLE, out / "table.parquet")
    shutil.copy2(paths.ENV_JSON, out / "env.json")
    shutil.copy2(KERNELS, out / "kernels.jsonl")

    from manifest import build as build_manifest  # noqa: E402
    man = build_manifest()
    (out / "manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False) + "\n")
    (out / "validate_report.md").write_text(
        f"# 무결성 검사 — `{bundle_id}`\n\n```\n{report}\n```\n")

    # 번들은 코드와 분리되어 유통된다. 그러니 라이선스가 **파일 안에** 있어야
    # 한다 — 저장소를 안 본 사람이 tar 하나만 받아도 조건을 알 수 있어야 한다.
    # 데이터는 코드의 파생물이 아니므로 Apache-2.0 이 아니라 CC BY 4.0 이다.
    # (소프트웨어 라이선스를 데이터셋에 붙이면 이용자가 오히려 혼란스럽다.)
    (out / "LICENSE.txt").write_text(f"""\
{bundle_id}
kerneltab measurement table

License: CC BY 4.0  (https://creativecommons.org/licenses/by/4.0/)

인용할 때는 측정 조건을 함께 밝혀야 한다. 그것 없이는 재현이 불가능하다:

  GPU        {hw.name} ({hw.arch}, {hw.sm_count} SM)
  env_hash   {env["env_hash"]}
  SM clock   {env.get("locked_mhz")} MHz (locked={env.get("clock_locked")})
  MEM clock  {env.get("locked_mem_mhz")} MHz (locked={env.get("mem_clock_locked")})
  effective  {env.get("peak_tflops_f16_effective")} TFLOP/s f16, \
{env.get("bandwidth_gbps_effective")} GB/s

전체 조건은 env.json, 생성 도구 버전은 manifest.json 에 있다.

이 표를 만든 도구(kerneltab)는 Apache-2.0 이며 별개다.
CUTLASS (NVIDIA, BSD-3-Clause) 는 이 번들에 포함되지 않는다.
""")

    # --- 표 요약 (Phase 3 데이터를 분석하지 않고 개수만 센다) ---------------
    import pyarrow.parquet as pq
    t = pq.read_table(out / "table.parquet")
    cols = t.column_names
    n_rows = t.num_rows
    if "env_hash" in cols:
        eh = t.column("env_hash").to_pylist()
        n_rows = sum(1 for x in eh if str(x).startswith(env_hash))
    shapes = {(a, b, c) for a, b, c in zip(
        t.column("M").to_pylist(), t.column("N").to_pylist(),
        t.column("K").to_pylist())} if "M" in cols else set()
    n_kernels = len(set(t.column("kernel_id").to_pylist())) if "kernel_id" in cols else 0

    # 측정 시각 범위
    ts = sorted(x for x in t.column("timestamp").to_pylist() if x) \
        if "timestamp" in cols else []

    # --- 층별 형상 목록 (층 C 는 GPU 마다 다르다) ---------------------------
    layers = {name: [[p.M, p.N, p.K] for p in probs]
              for name, probs in all_layers(hw).items()}

    files = {}
    for f in sorted(out.iterdir()):
        if f.name == "BUNDLE.json" or f.is_dir():
            continue
        files[f.name] = {"bytes": f.stat().st_size, "sha256": sha256(f)}

    bundle = {
        "bundle_id": bundle_id,
        # 번들은 코드와 분리되어 유통되므로 파일이 라이선스를 들고 다닌다.
        # 표는 코드의 파생물이 아니다 — 도구는 Apache-2.0, 데이터는 CC BY 4.0.
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution_required": [
            "gpu_name", "env_hash", "locked_mhz", "locked_mem_mhz",
            "peak_tflops_f16_effective", "bandwidth_gbps_effective",
        ],
        "tool_license": "Apache-2.0",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "gpu_name": hw.name,
        "arch": hw.arch,
        "sm_count": hw.sm_count,
        "env_hash": env["env_hash"],
        "device_uuid": env.get("hardware_extra", {}).get("uuid"),
        # 규모
        "n_shapes": len(shapes),
        "n_kernels": n_kernels,
        "n_rows": n_rows,
        "measured_from_utc": ts[0] if ts else None,
        "measured_to_utc": ts[-1] if ts else None,
        # 측정 조건 — 이것이 없으면 표를 해석할 수 없다
        "clock_locked": env.get("clock_locked"),
        "locked_mhz": env.get("locked_mhz"),
        "mem_clock_locked": env.get("mem_clock_locked"),
        "locked_mem_mhz": env.get("locked_mem_mhz"),
        "peak_tflops_f16_effective": env.get("peak_tflops_f16_effective"),
        "bandwidth_gbps_effective": env.get("bandwidth_gbps_effective"),
        "ridge_point": round(hw.peak_tflops_f16 * 1e12 / (hw.bandwidth_gbps * 1e9), 3),
        "protocol": env.get("protocol"),
        "soak": env.get("soak"),
        # 층별 형상. 층 C 는 sm_count 에서 M 을 역산하므로 GPU 마다 다르다.
        # 여러 번들을 합칠 때 공통 형상을 가려내려면 이 정보가 필요하다.
        "shape_layers": layers,
        # 재현성
        "manifest": {k: man.get(k) for k in
                     ("cuda_version", "nvcc_version", "cutlass_commit",
                      "kerneltab_commit", "kerneltab_tree_hash",
                      "manifest_hash", "image_tag", "python_version")},
        "files": files,
        "schema_version": 1,
    }
    (out / "BUNDLE.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False) + "\n")

    # --- C-6: 크기 보고 -----------------------------------------------------
    print(f"\n번들 {bundle_id}")
    print(f"  {out}")
    print(f"  {'파일':22s} {'크기':>12s}")
    total = 0
    for name, meta in sorted(files.items(), key=lambda kv: -kv[1]["bytes"]):
        total += meta["bytes"]
        print(f"  {name:22s} {human(meta['bytes']):>12s}")
    print(f"  {'합계':22s} {human(total):>12s}")
    print(f"\n  형상 {bundle['n_shapes']}  커널 {bundle['n_kernels']:,}  "
          f"측정 {bundle['n_rows']:,}행")
    print(f"  ridge point {bundle['ridge_point']} FLOP/byte "
          f"({bundle['peak_tflops_f16_effective']} TFLOP/s / "
          f"{bundle['bandwidth_gbps_effective']} GB/s)")

    tsize = files.get("table.parquet", {}).get("bytes", 0)
    if tsize > 500 * 1024 * 1024:
        print(f"\n  !! table.parquet 이 {human(tsize)} 로 500MB 를 넘는다.")
        print("     배포 방법을 다시 논의해야 한다. 컬럼 dtype 최적화")
        print("     (문자열 -> dictionary, float64 -> float32) 로 크게 줄일 수 있다.")

    # --- C-3: 압축 ----------------------------------------------------------
    if args.archive:
        tar = Path(args.out) / f"{bundle_id}.tar.zst"
        print(f"\n압축 중 -> {tar}")
        zstd = shutil.which("zstd")
        if zstd:
            subprocess.run(["tar", "--zstd", "-cf", str(tar), "-C",
                            str(out.parent), bundle_id], check=True)
        else:
            tar = tar.with_suffix(".gz")
            subprocess.run(["tar", "-czf", str(tar), "-C",
                            str(out.parent), bundle_id], check=True)
            print("  (zstd 가 없어 gzip 으로 대체했다)")
        digest = sha256(tar)
        tar.with_suffix(tar.suffix + ".sha256").write_text(
            f"{digest}  {tar.name}\n")
        print(f"  {human(tar.stat().st_size)}  sha256={digest[:16]}...")

    # --- C-4: 원본 보존 -----------------------------------------------------
    if args.archive_raw:
        # 측정이 도는 중이면 results.jsonl 이 계속 자란다. zstd 는 stat 으로
        # 크기를 먼저 읽으므로 "Incomplete read" 로 죽고(코드 27), 죽지 않더라도
        # 마지막 줄이 잘린 스냅샷이 된다. 리허설에서 실제로 터졌다.
        if measurement_running():
            print("\n!! 측정이 진행 중이다. --archive-raw 는 건너뛴다.")
            print("   results.jsonl 이 자라는 중이라 잘린 스냅샷이 된다.")
            print("   측정이 끝난 뒤 다시 돌려라 (scripts/watch.py 로 확인).")
            args.archive_raw = False
            # 지난 시도가 남긴 파일이 있으면 지운다. 잘린 스냅샷이 정상
            # 아카이브처럼 남아 있는 것이 가장 위험하다.
            for ext in (".jsonl.zst", ".jsonl.gz"):
                stale = Path(args.out) / f"results-raw-{env_hash[:8]}{ext}"
                if stale.exists():
                    stale.unlink()
                    (stale.parent / (stale.name + ".sha256")).unlink(missing_ok=True)
                    print(f"   (잘렸을 수 있는 이전 파일 {stale.name} 삭제)")
    if args.archive_raw:
        raw = Path(args.out) / f"results-raw-{env_hash[:8]}.jsonl.zst"
        print(f"\n원본 압축 -> {raw}")
        # 파일 경로가 아니라 **stdin 으로** 넘긴다. 그러면 zstd 가 크기를
        # 미리 재지 않고 EOF 까지 읽는다.
        if shutil.which("zstd"):
            with RESULTS.open("rb") as f, raw.open("wb") as o:
                subprocess.run(["zstd", "-q", "-19", "-T0", "-"],
                               stdin=f, stdout=o, check=True)
        else:
            raw = raw.with_suffix(".gz")
            with RESULTS.open("rb") as f, raw.open("wb") as o:
                subprocess.run(["gzip", "-9", "-c"], stdin=f, stdout=o, check=True)
            print("  (zstd 가 없어 gzip 으로 대체했다)")
        d = sha256(raw)
        raw.with_suffix(raw.suffix + ".sha256").write_text(f"{d}  {raw.name}\n")
        print(f"  {human(RESULTS.stat().st_size)} -> {human(raw.stat().st_size)}"
              f"  sha256={d[:16]}...")
        print("  ※ results.jsonl 은 append-only 원본이다. table.parquet 은 "
              "파생물이므로 계산식이 바뀌면 이 원본에서 재생성한다.")

    print(f"\n소비: KERNELTAB_DATASETS={Path(args.out).resolve()} "
          f"python3 -c \"from core.bundle import load_bundle; "
          f"print(load_bundle('{bundle_id}').info['n_rows'])\"")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    raise SystemExit(main())
