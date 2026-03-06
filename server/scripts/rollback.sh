#!/bin/bash
# Rollback production detector to baseline
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"

SRC="$MODELS_DIR/baseline_detector"
DST="$MODELS_DIR/production_detector"

if [ ! -d "$SRC" ]; then
    echo "ERROR: Baseline directory not found: $SRC"
    echo "If you never saved a baseline, copy the default PaddleOCR model there first."
    exit 1
fi

echo "Rolling back production_detector to baseline_detector..."
rm -rf "$DST"
cp -r "$SRC" "$DST"
echo "Done. Restart the server to use the baseline model."
