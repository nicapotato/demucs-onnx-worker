#!/usr/bin/env bash
# Download the bundled 6-stem graph. Nothing is fetched at separation time.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FILE="$(grep '^MODEL_FILE=' "$ROOT/project.conf" | cut -d= -f2-)"
URL="$(grep '^MODEL_URL=' "$ROOT/project.conf" | cut -d= -f2-)"
DEST="$ROOT/models/$FILE"

if [[ -f "$DEST" ]]; then
  bytes="$(wc -c < "$DEST" | tr -d ' ')"
  if [[ "$bytes" -gt 100000000 ]]; then
    echo "already present: $DEST ($bytes bytes)"
    exit 0
  fi
  echo "replacing incomplete $DEST ($bytes bytes)" >&2
  rm -f "$DEST"
fi

mkdir -p "$ROOT/models"
echo "downloading $URL"
curl -fL --retry 3 -o "$DEST.partial" "$URL"
mv -f "$DEST.partial" "$DEST"
bytes="$(wc -c < "$DEST" | tr -d ' ')"
echo "wrote $DEST ($bytes bytes)"
if [[ "$bytes" -lt 100000000 ]]; then
  echo "download looks truncated" >&2
  exit 1
fi
