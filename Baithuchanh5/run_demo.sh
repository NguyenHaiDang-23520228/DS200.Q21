#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="python3"
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
fi

mkdir -p output/results output/annotated logs
rm -f output/results/detections.ndjson output/results/summary.json output/results/frame_*.json
rm -f output/annotated/frame_*.jpg

echo "Starting storage server..."
"$PYTHON" storage_server.py > logs/storage.log 2>&1 &
STORAGE_PID=$!
sleep 1

echo "Starting processing server..."
"$PYTHON" processing_server.py > logs/processing.log 2>&1 &
PROCESSING_PID=$!
sleep 1

echo "Starting camera server with preview..."
"$PYTHON" camera_server.py --source video --video walkingstreet.mp4 --show --start-frame 1100 --max-frames 30 | tee logs/camera.log

echo "Generating annotated images..."
"$PYTHON" visualize_results.py

echo "Stopping background servers..."
kill "$PROCESSING_PID" "$STORAGE_PID" 2>/dev/null || true
wait "$PROCESSING_PID" "$STORAGE_PID" 2>/dev/null || true

echo ""
echo "Done."
echo "- JSON results: output/results/"
echo "- Annotated images: output/annotated/"
if [ -f output/results/summary.json ]; then
  cat output/results/summary.json
fi
