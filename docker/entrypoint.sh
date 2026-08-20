#!/usr/bin/env bash
# kerneltab 컨테이너 진입점 — 동사를 scripts/*.py 로 넘긴다.
#
# 왜 Python CLI 가 아니라 셸 shim 인가: 통합 CLI(`docs/entrypoints.md`)는
# 패키지 구조를 건드리는 별개 작업이고, **resume 명령이 바뀌면 안 된다.**
# 이 shim 은 컨테이너 밖 사용법(`python3 scripts/xxx.py`)을 그대로 두면서
# 컨테이너 안에서만 짧은 이름을 준다.
#
# 모르는 인자는 그대로 실행한다 — `docker run ... IMG bash` 나
# `... python3 scripts/foo.py` 가 계속 동작해야 한다.
set -euo pipefail

# --- 산출물 경로 쓰기 검사 -------------------------------------------------
# 쓰는 동사에 대해서만, **시작하자마자** 확인한다. 나중에 nvcc 링크 단계에서
# "Permission denied" 로 죽으면 원인이 컴파일러 경고 수천 줄에 묻힌다
# (실제로 그랬다).
#
# 마운트를 잊었을 때도 여기서 잡는다 — 쓰기는 되지만 컨테이너와 함께
# 사라지는 상태이므로 경고한다.
need_write() {
  case "$1" in
    detect|build|drift|rehearse|sweep|export|bundle|probe) return 0 ;;
    *) return 1 ;;
  esac
}
if need_write "${1:-help}"; then
  for d in "${KERNELTAB_RESULTS_DIR:-/data/results}" \
           "${KERNELTAB_ARTIFACT_DIR:-/data/artifacts}"; do
    mkdir -p "$d" 2>/dev/null || true
    if ! [ -w "$d" ]; then
      cat >&2 <<ERR
⛔ $d 에 쓸 수 없다 (uid=$(id -u) gid=$(id -g)).

   호스트에서 디렉토리를 먼저 만들고 통째로 마운트하라:
     mkdir -p \$PWD/data/results \$PWD/data/artifacts
     docker run ... -v \$PWD/data:/data --user \$(id -u):\$(id -g) ...

   /data 아래 **하위 경로를 따로** 마운트하지 마라. 익명 볼륨이 끼면
   root 소유가 되고 --rm 과 함께 결과가 사라진다.
ERR
      exit 6
    fi
  done
  if ! grep -qE " (/data|/data/results) " /proc/self/mountinfo 2>/dev/null; then
    echo "⚠️  /data 가 바인드 마운트가 아닌 것 같다. 컨테이너를 지우면" >&2
    echo "    측정 결과가 사라진다. -v <호스트경로>:/data 를 붙여라." >&2
  fi
fi

case "${1:-help}" in
  detect)    shift; exec python3 /work/scripts/phase0_env.py       "$@" ;;
  build)     shift; exec python3 /work/scripts/build_kernels.py    "$@" ;;
  drift)     shift; exec python3 /work/scripts/measure_drift.py    "$@" ;;
  probe)     shift; exec python3 /work/scripts/probe_toolchain.py  "$@" ;;
  rehearse)  shift; exec python3 /work/scripts/rehearse.py         "$@" ;;
  sweep)     shift; exec python3 /work/scripts/sweep.py            "$@" ;;
  anchors)   shift; exec python3 /work/scripts/check_anchors.py    "$@" ;;
  gate)      shift; exec python3 /work/scripts/gate_g7.py           "$@" ;;
  overhead)  shift; exec python3 /work/scripts/report_overhead.py   "$@" ;;
  warmup)    shift; exec python3 /work/scripts/verify_warmup.py     "$@" ;;
  export)    shift; exec python3 /work/scripts/export.py           "$@" ;;
  bundle)    shift; exec python3 /work/scripts/bundle.py           "$@" ;;
  validate)  shift; exec python3 /work/scripts/validate_table.py   "$@" ;;
  manifest)  shift; exec python3 /work/scripts/manifest.py         "$@" ;;
  watch)     shift; exec python3 /work/scripts/watch.py            "$@" ;;
  status)    shift; exec python3 /work/scripts/sweep_status.py      "$@" ;;
  verify)
    shift
    case "${1:-}" in
      smem)   shift; exec python3 /work/scripts/check_smem.py   "$@" ;;
      splitk) shift; exec python3 /work/scripts/smoke_splitk.py "$@" ;;
      clock)  shift; exec python3 /work/scripts/verify_clock_lock.py "$@" ;;
      *) echo "verify: smem | splitk | clock" >&2; exit 2 ;;
    esac ;;
  test)      shift; exec python3 -m pytest /work/tests -q          "$@" ;;
  help|-h|--help)
    cat <<'USAGE'
kerneltab — CUTLASS GEMM (형상 x config) -> 성능 표 측정 하네스

  detect     GPU/환경 감지 -> results/env.json   (--gpu <UUID> 를 쓸 것)
  build      커널 생성 + nvcc 빌드 -> artifacts/
  drift      드리프트 3값 측정 (새 GPU 필수 게이트, G-4)
  probe      툴체인 표본 빌드 (nvcc/CUTLASS 회귀 점검)
  rehearse   측정 (--segment 로 세그먼트 하나)
  sweep      전수 측정 (세그먼트 라운드 로빈)
  anchors    앵커 판정 (세그먼트 간 계통 오차)
  gate       G-7 재현성 검증 5항목을 한 번에 (전수 전 필수)
  overhead   예비/워밍업 오버헤드 실측
  warmup     워밍업 시간 예산이 측정값을 바꾸지 않는지
  export     results/*.jsonl -> table.parquet
  bundle     배포 번들 + 체크섬
  validate   무결성 검사
  verify     smem | splitk | clock
  manifest   빌드 매니페스트 / 이미지 태그
  watch      슬라이스 하나의 하트비트 (0=진행중 3=끝남 5=죽음)
  status     스윕 전체 진행률 + 앵커 (읽기 전용, 측정 중 안전)
  test       테스트 (GPU 불필요)

⚠️ 클럭 고정은 **호스트에서** 한다. 컨테이너 안에서는 불가능하다
   (CAP_SYS_ADMIN + 드라이버 쓰기 접근이 필요하고 툴킷이 주지 않는다).
   호스트에서 고정한 뒤 detect --externally-locked-mhz/-mem-mhz 로 인정받는다.

⚠️ GPU 는 **UUID** 로 지정한다. --gpus 로 준 GPU 는 컨테이너 안에서
   인덱스 0 이 되므로 인덱스는 뜻이 없다.
USAGE
    ;;
  *) exec "$@" ;;
esac
