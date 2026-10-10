#!/bin/sh
# Build the cgo shared library consumed by the Python client.
# Usage: sh python/build.sh [output-path]
set -eu
cd "$(dirname "$0")/.."
case "$(go env GOOS)" in
	darwin) LIB=libgomem.dylib ;;
	windows) LIB=gomem.dll ;;
	*) LIB=libgomem.so ;;
esac
OUT="${1:-python/$LIB}"
CGO_ENABLED=1 go build -buildmode=c-shared -ldflags="-s" -o "$OUT" ./python
echo "built $OUT"
