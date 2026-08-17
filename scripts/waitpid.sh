#!/bin/bash
# PID 기준으로 프로세스가 끝날 때까지 기다린다.
#
# ⛔ pgrep 을 쓰지 마라.
#
#     until ! pgrep -f "rehearse.py --all"; do sleep 60; done
#
# 이 감시 셸의 **명령줄 자체에 그 문자열이 들어 있어서** pgrep 이 자기
# 자신을 찾는다. 대상이 이미 끝났는데도 영원히 "아직 도는 중" 으로 판정한다.
# 이 버그로 13 시간을 날렸고, 문서만 고쳐 뒀더니 그 뒤에 또 밟았다.
# 그래서 헬퍼로 만든다 — 직접 짜지 마라.
#
#   scripts/waitpid.sh 12345                 # 끝날 때까지 대기
#   scripts/waitpid.sh 12345 -- echo done    # 끝난 뒤 명령 실행
#   scripts/waitpid.sh --file results/heartbeat.json   # 하트비트의 pid 로
#
# 종료 코드: 0 대상이 끝남 / 2 사용법 오류 / 3 그런 PID 없음(이미 끝남)
#
# 측정 진행 여부를 보는 것이라면 이것보다 scripts/watch.py 가 낫다 —
# 하트비트가 멈췄는지(=죽었는지)까지 구분한다.

set -u
INTERVAL="${WAITPID_INTERVAL:-30}"

usage() { echo "usage: $0 <pid> [-- cmd...]   |   $0 --file <json> [-- cmd...]" >&2; exit 2; }

[ $# -ge 1 ] || usage

if [ "$1" = "--file" ]; then
  [ $# -ge 2 ] || usage
  PID=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('pid',''))" "$2") || usage
  shift 2
else
  PID="$1"; shift
fi

case "$PID" in
  ''|*[!0-9]*) echo "PID 가 숫자가 아니다: '$PID'" >&2; exit 2 ;;
esac

if ! kill -0 "$PID" 2>/dev/null; then
  echo "PID $PID 는 이미 없다"
  exit 3
fi

echo "PID $PID 대기 중 (${INTERVAL}초 간격)"
while kill -0 "$PID" 2>/dev/null; do sleep "$INTERVAL"; done
echo "PID $PID 종료됨"

if [ $# -gt 0 ]; then
  [ "$1" = "--" ] && shift
  exec "$@"
fi
