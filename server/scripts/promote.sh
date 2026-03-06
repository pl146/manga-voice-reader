#!/bin/bash
# Promote a training run to production detector
# Usage: ./promote.sh run_001
set -e

RUN="${1:?Usage: ./promote.sh <run_name>  (e.g. run_001)}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"

SRC="$MODELS_DIR/runs/$RUN"
DST="$MODELS_DIR/production_detector"

if [ ! -d "$SRC" ]; then
    echo "ERROR: Run directory not found: $SRC"
    exit 1
fi

# Backup current production if it exists and has model files
if [ -d "$DST" ] && ls "$DST"/*.pd* 1>/dev/null 2>&1; then
    BACKUP="$MODELS_DIR/production_backup_$(date +%Y%m%d_%H%M%S)"
    echo "Backing up current production to $BACKUP"
    cp -r "$DST" "$BACKUP"
fi

echo "Promoting $RUN to production_detector..."
rm -rf "$DST"
cp -r "$SRC" "$DST"
echo "Done. Restart the server to use the new model."
