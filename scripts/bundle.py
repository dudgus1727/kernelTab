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

from kerneltab.build import paths
from kerneltab.core.hardware import hardware_from_env
from kerneltab.core.shapes import all_layers

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


def _render_corrections(items: list) -> str:
    """정정 이력을 릴리즈 노트에 싣는다.

    이미 받아간 사람이 **자기 사본이 구버전인지** 알 수 있어야 한다.
    그래서 구 sha256 을 반드시 함께 적는다.
    """
    if not items:
        return ""
    out = ["", "---", ""]
    for c in items:
        out.append(f"## ⚠️ 정정 ({c.get('date', '')})")
        out.append("")
        out.append(c.get("what", ""))
        out.append("")
        aff = c.get("affected") or {}
        if aff:
            out.append("| env_hash | 행 수 |")
            out.append("|---|---:|")
            for k, v in sorted(aff.items(), key=lambda kv: -kv[1]):
                out.append(f"| `{k}` | {v:,} |")
            out.append("")
        if c.get("impact"):
            out.append(f"**영향:** {c['impact']}")
            out.append("")
        prev = c.get("prev_sha256") or {}
        if prev:
            out.append("체크섬으로 사본을 구분할 수 있다 (구 = 정정 전):")
            out.append("")
            out.append("| 파일 | 구 sha256 |")
            out.append("|---|---|")
            for k, v in prev.items():
                out.append(f"| `{k}` | `{v[:32]}...` |")
            out.append("")
        if c.get("fix"):
            out.append(f"**재발 방지:** {c['fix']}")
            out.append("")
    return "\n".join(out)


def _load_corrections() -> list:
    """`results/bundle_corrections.json` 이 있으면 그대로 싣는다."""
    f = paths.RESULTS_DIR / "bundle_corrections.json"
    if not f.exists():
        return []
    try:
        return json.loads(f.read_text())
    except Exception:
        return []


def assert_single_env(parquet_path: Path, expected: str) -> int:
    """번들에 들어갈 표가 **단일 측정 조건**인지 확인한다.

    ⚠️ 이 검사가 없어서 실제로 사고가 났다. `results/table.parquet` 은
    작업 파일이라 여러 조건을 담는데(조건 간 비교/드리프트 분석에 쓴다),
    `bundle.py` 가 그것을 **그대로 복사**해서 폐기한 드리프트 데이터
    226,145행이 공개 릴리즈에 실렸다. `BUNDLE.json` 의 `n_rows` 는
    필터한 값이라 980,915 로 맞았고, 그래서 아무도 못 알아챘다.

    앞의 다섯 사례(`docs/decisions.md` 13번)와 성격이 다르다 — 코드 안에서
    집계가 오염된 것이 아니라 **파일이 배포 경계를 넘어갔다.**
    `core.records` 로 강제한 층에서는 안 잡힌다.
    """
    import pyarrow.parquet as pq

    t = pq.read_table(parquet_path, columns=["env_hash"])
    seen: dict[str, int] = {}
    for x in t.column("env_hash").to_pylist():
        k = str(x)[:8]
        seen[k] = seen.get(k, 0) + 1
    bad = {k: v for k, v in seen.items() if not expected.startswith(k)}
    if bad:
        print(f"\n!! {parquet_path.name} 에 다른 측정 조건이 섞여 있다.")
        for k, v in sorted(bad.items(), key=lambda kv: -kv[1]):
            print(f"     {k}  {v:,}행")
        print("   배포되는 표는 단일 조건이어야 한다. 번들을 만들지 않는다.")
        return 6
    return 0


def _noise_coefficients() -> dict:
    """앵커에서 잰 노이즈 바닥 계수. 없으면 core.noise 의 기본값."""
    from kerneltab.core import noise
    return noise.coefficients()


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


#: GitHub Release 에셋 하나의 상한. 넘으면 업로드가 실패한다.
GH_ASSET_LIMIT = 2 * 1024 ** 3


def release_notes(b: dict) -> str:
    """BUNDLE.json 에서 릴리즈 노트를 만든다.

    측정 조건이 노트에 있어야 한다 — 릴리즈를 받은 사람이 저장소를 보지
    않고도 이 표가 **어떤 조건에서 측정된 것인지** 알 수 있어야 인용과
    재현이 가능하다.
    """
    m = b.get("manifest") or {}
    corr = _render_corrections(b.get("corrections") or [])
    return f"""\
# kerneltab 측정 표 — {b['gpu_name']} (`{b['bundle_id']}`)

CUTLASS GEMM 의 (형상 x config) -> 성능 표.

## 측정 조건 — **인용할 때 반드시 함께 밝힐 것**

| | |
|---|---|
| GPU | {b['gpu_name']} ({b['arch']}, {b['sm_count']} SM) |
| `env_hash` | `{b['env_hash']}` |
| SM 클럭 | {b.get('locked_mhz')} MHz (고정={b.get('clock_locked')}) |
| 메모리 클럭 | {b.get('locked_mem_mhz')} MHz (고정={b.get('mem_clock_locked')}) |
| 실효 피크 | {b.get('peak_tflops_f16_effective')} TFLOP/s (f16) |
| 실효 대역폭 | {b.get('bandwidth_gbps_effective')} GB/s |
| ridge point | {b.get('ridge_point')} FLOP/byte |
| 측정 구간 | {b.get('measured_from_utc')} ~ {b.get('measured_to_utc')} |

**`env_hash` 가 다르면 측정 조건이 다르다. 절대 시간을 직접 비교하지 마라.**

스펙 시트 값(154.8 TFLOP/s, 768 GB/s)은 이 데이터에 적용되지 않는다.
클럭을 고정해 측정했고, 컴퓨트 워크로드는 P2 상태로 동작해 메모리 클럭이
P0 최대치에 도달하지 못한다.

## 규모

- 형상 {b.get('n_shapes')}개 x 커널 {b.get('n_kernels'):,}개 = **{b.get('n_rows'):,}행**
- 층별 형상: {', '.join(f"{k} {len(v)}" for k, v in (b.get('shape_layers') or {{}}).items())}

## 파일

| 파일 | 내용 |
|---|---|
| `{b['bundle_id']}.tar.zst` | 번들 (table.parquet + env.json + kernels.jsonl + manifest + 체크섬) |
| `results-raw-*.jsonl.zst` | 측정 원본 JSONL. `table.parquet` 은 파생물이므로 계산식이 바뀌면 여기서 재생성한다 |
| `*.sha256` | 체크섬. 받은 뒤 반드시 대조할 것 |

```bash
sha256sum -c {b['bundle_id']}.tar.zst.sha256
tar --zstd -xf {b['bundle_id']}.tar.zst
```

## 쓰는 법

```python
from kerneltab.core.bundle import load_bundle
b = load_bundle("{b['bundle_id']}")   # sha256 자동 대조
X = b.ranking()    # 규칙 입력 (정답 컬럼 제거됨)
y = b.scoring()    # 채점용 (정답 포함)
```

`ranking()` 은 `time_ms` 등 정답과 그로부터 유도된 값(`difficulty` 포함)을
제거한다. 규칙 함수에 `scoring()` 을 넘기면 정답을 훔쳐보는 것이다.

## 재현

| | |
|---|---|
| CUDA | {m.get('cuda_version')} / nvcc {m.get('nvcc_version')} |
| CUTLASS | `{m.get('cutlass_commit')}` |
| kerneltab | `{m.get('kerneltab_commit')}` |

측정 방법과 주의점은 저장소의 `docs/measurement_drift.md` 를 볼 것.
**다중 시간 측정 드리프트**가 실재하며, 대책 없이 재현하면 데이터가
오염된다.

## 라이선스

측정 표: **CC BY 4.0** — 인용 시 위 측정 조건을 함께 밝힐 것.
생성 도구(kerneltab): Apache-2.0. CUTLASS(NVIDIA, BSD-3)는 포함되지 않는다.
{corr}"""


def github_release(bundle: dict, out: Path, dsdir: Path, bundle_id: str,
                   env_hash: str, publish: bool) -> int:
    """오프사이트 백업. 기본은 dry-run 이다.

    ⚠️ 공개 저장소면 실행하는 순간 데이터가 인터넷에 공개된다. 되돌리려면
    에셋을 지워야 하는데 그 사이 누가 받아갔는지는 알 수 없다. 그래서
    --publish 를 따로 요구한다.
    """
    gh = shutil.which("gh")
    if not gh:
        print("\n!! gh 가 없다. https://cli.github.com 에서 설치하라.")
        return 5

    notes = out / "RELEASE.md"
    if not notes.exists():          # 보통 main() 이 아카이브 전에 써 둔다
        notes.write_text(release_notes(bundle))

    assets = []
    for pat in (f"{bundle_id}.tar.zst", f"{bundle_id}.tar.gz",
                f"results-raw-{env_hash[:8]}.jsonl.zst",
                f"results-raw-{env_hash[:8]}.jsonl.gz"):
        f = dsdir / pat
        if f.exists():
            assets.append(f)
            chk = f.with_suffix(f.suffix + ".sha256")
            if chk.exists():
                assets.append(chk)

    print("\n--- GitHub Release ---")
    print(f"릴리즈 노트: {notes}")
    if not assets:
        print("!! 올릴 파일이 없다. --archive --archive-raw 를 먼저 돌려라.")
        return 5

    too_big = [f for f in assets if f.stat().st_size > GH_ASSET_LIMIT]
    for f in assets:
        mark = "  !! 2GB 초과" if f in too_big else ""
        print(f"  {f.name:56s} {human(f.stat().st_size):>10}{mark}")
    if too_big:
        print("\n!! GitHub Release 에셋은 파일당 2 GB 가 상한이다.")
        print("   split -b 1900M 으로 나눠 올리고 cat 으로 복원하라.")
        return 5

    # 태그에 env_hash 가 들어가는 것이 핵심이다. 같은 GPU 라도 측정 조건이
    # 다르면 다른 릴리즈여야 한다.
    tag = f"data-{bundle_id}"
    cmd = [gh, "release", "create", tag,
           "--title", f"kerneltab 측정 표 — {bundle['gpu_name']} ({bundle_id})",
           "--notes-file", str(notes)] + [str(f) for f in assets]

    if not publish:
        print("\n[dry-run] 실제로 올리지 않았다. 아래 명령을 확인한 뒤")
        print("          --github-release --publish 로 다시 돌려라.\n")
        print("  " + " \\\n    ".join(cmd))
        print("\n  ⚠️ 저장소가 public 이면 이 순간 데이터가 인터넷에 공개된다.")
        return 0

    print(f"\n업로드 중 (tag={tag}) ...")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("!! gh release create 실패")
        return r.returncode
    print(f"  https://github.com/<owner>/<repo>/releases/tag/{tag}")
    return 0


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
    ap.add_argument("--github-release", action="store_true",
                    help="GitHub Release 로 오프사이트 백업한다. RELEASE.md 를 "
                         "만들고 gh 명령을 **출력만** 한다 (기본 dry-run)")
    ap.add_argument("--publish", action="store_true",
                    help="--github-release 를 실제로 실행한다. 공개 저장소면 "
                         "이 순간 데이터가 인터넷에 공개된다")
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
    # table.parquet 은 **이 조건의 행만** 넣는다. 원본은 작업 파일이라
    # 여러 조건을 담는다 — 그대로 복사하면 폐기한 데이터가 배포된다.
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    _t = pq.read_table(TABLE)
    _keep = pc.starts_with(pc.cast(_t.column("env_hash"), "string"), env_hash[:8])
    _t = _t.filter(_keep)
    pq.write_table(_t, out / "table.parquet", compression="zstd")
    print(f"  table.parquet: {_t.num_rows:,}행 (원본에서 이 조건만 추림)")
    shutil.copy2(paths.ENV_JSON, out / "env.json")
    shutil.copy2(KERNELS, out / "kernels.jsonl")

    from manifest import build as build_manifest
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
    # ⚠️ **이 측정 조건의 줄만** 센다. table.parquet 에는 다른 env_hash 의
    #    줄도 들어 있다 (폐기된 드리프트 구간 등). 필터하지 않으면 형상 수,
    #    커널 수, 측정 구간이 전부 남의 데이터를 포함한 값이 되고, 그게
    #    릴리즈 노트에 실려 나간다. difficulty 가 22배로 나왔던 것과 같은
    #    함정이다 — env_hash 는 조인 키가 아니라 격리 경계다.
    ehs = ([str(x) for x in t.column("env_hash").to_pylist()]
           if "env_hash" in cols else [""] * t.num_rows)
    keep = [i for i, x in enumerate(ehs) if x.startswith(env_hash)]
    n_rows = len(keep)

    def col(name):
        if name not in cols:
            return []
        v = t.column(name).to_pylist()
        return [v[i] for i in keep]

    Ms, Ns, Ks = col("M"), col("N"), col("K")
    shapes = set(zip(Ms, Ns, Ks))
    n_kernels = len(set(col("kernel_id")))

    # 측정 시각 범위 (이 조건의 줄만)
    ts = sorted(x for x in col("timestamp") if x)

    # --- 층별 형상 목록 (층 C 는 GPU 마다 다르다) ---------------------------
    layers = {name: [[p.M, p.N, p.K] for p in probs]
              for name, probs in all_layers(hw).items()}

    files = {}
    for f in sorted(out.iterdir()):
        # BUNDLE.json 은 체크섬을 담는 파일이라 자기를 못 담는다.
        # RELEASE.md 는 BUNDLE.json 에서 결정적으로 생성되는 **사람용 노트**라
        # 데이터가 아니다 (내용은 BUNDLE.json 이 보증한다).
        if f.name in ("BUNDLE.json", "RELEASE.md") or f.is_dir():
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
        # 정정 이력. 번들이 코드와 분리되어 유통되므로 **파일 자체가
        # 이력을 들고 다녀야** 한다. 이미 받아간 사람이 자기 사본이
        # 구버전인지 확인할 수 있어야 한다.
        "corrections": _load_corrections(),
        # 측정 노이즈 바닥. 소비 쪽이 재계산 없이 정답 허용치를 정할 수
        # 있어야 한다. 형상마다 다르므로 고정 1% 를 쓰면 안 된다.
        "noise_floor": _noise_coefficients(),
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

    # RELEASE.md 는 **아카이브 전에** 쓴다. github_release() 안에서 쓰면
    # tar 안에는 낡은 사본이 들어간다.
    (out / "RELEASE.md").write_text(release_notes(bundle))

    # 배포 경계 게이트 — 아카이브 직전. validate_table --bundle 과 같은
    # 검사를 부른다 (구현이 둘로 갈리면 한쪽만 고치게 된다).
    rc = assert_single_env(out / "table.parquet", env_hash)
    if rc:
        return rc
    from validate_table import validate_bundle
    rc = validate_bundle(out)
    if rc:
        print("\n!! 번들 검사 실패 — 배포하지 않는다.")
        return rc

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

    # --- C-3: GitHub Release (오프사이트) ------------------------------------
    if args.github_release:
        rc = github_release(bundle, out, Path(args.out), bundle_id,
                            env_hash, args.publish)
        if rc:
            return rc

    print(f"\n소비: KERNELTAB_DATASETS={Path(args.out).resolve()} "
          f"python3 -c \"from kerneltab.core.bundle import load_bundle; "
          f"print(load_bundle('{bundle_id}').info['n_rows'])\"")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    raise SystemExit(main())
