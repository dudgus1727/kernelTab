#!/usr/bin/env bash
# kerneltab 이미지 빌드.
#
#   ./docker/build.sh                 # 빌드 + 이미지 자신의 매니페스트로 태그
#   ./docker/build.sh --no-cache
#
# ⚠️ **태그는 이미지 안에서 계산해야 한다.**
#    `python3 scripts/manifest.py --tag` 을 호스트에서 돌리면 호스트의 nvcc
#    (여기선 12.4)가 들어가 `kerneltab:cu124-...` 가 나온다. 이미지 안에는
#    CUDA 13.3 이 들어 있으므로 **거짓말이다.** 그래서 임시 태그로 빌드한 뒤
#    이미지 안에서 매니페스트를 뽑아 다시 태그한다.
set -euo pipefail
cd "$(dirname "$0")/.."

TMP_TAG="kerneltab:building-$$"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git status --porcelain 2>/dev/null || true)" ]; then
  echo "⚠️  worktree 가 dirty 하다. 커밋만으로는 이 이미지가 재현되지 않는다."
  echo "    (tree_hash 는 실제 파일 내용을 해싱하므로 태그에는 반영된다)"
fi

docker build -f docker/Dockerfile \
  --build-arg KERNELTAB_COMMIT="$COMMIT" \
  -t "$TMP_TAG" "$@" .

TAG="$(docker run --rm "$TMP_TAG" manifest --tag)"
echo "이미지 태그: $TAG"
docker tag "$TMP_TAG" "$TAG"
docker rmi "$TMP_TAG" >/dev/null

docker run --rm "$TAG" manifest | sed 's/^/  /'
echo
echo "다음:"
echo "  G=<GPU UUID>   # nvidia-smi --query-gpu=uuid --format=csv,noheader"
echo "  mkdir -p data/results data/artifacts"
echo "  docker run --rm --gpus \"\\\"device=\$G\\\"\" \\"
echo "      -v \$PWD/data:/data --user \$(id -u):\$(id -g) $TAG \\"
echo "      detect --gpu \$G --externally-locked-mhz <실측> --externally-locked-mem-mhz <실측>"
