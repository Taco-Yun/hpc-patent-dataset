#!/usr/bin/env bash
# Full run: steps 1-5 for filing dates 20150101-20251231, written to data/.
# This sends tens of thousands of requests to Google Patents and takes many hours (see README).
#
# Usage:
#   bash scripts/run_full.sh        # asks for confirmation first
#   bash scripts/run_full.sh -y     # no confirmation prompt
#
# Environment variables (PROXY, HEADLESS, CHROME_VERSION, DOWNLOAD_PDF, PYTHON) are passed on; see run_pipeline.sh.
# Both crawling steps skip files that already exist, so an interrupted run can be restarted with the same command.
set -euo pipefail

START="20150101"
END="20251231"

if [[ "${1:-}" != "-y" ]]; then
    echo "Full run: filing dates $START-$END into data/."
    echo "Step 1 opens Chrome and queries every filing date; step 3 downloads one page per patent ID."
    read -r -p "Continue? [y/N] " answer
    [[ "$answer" =~ ^[Yy]$ ]] || { echo "Cancelled."; exit 1; }
fi

bash "$(dirname "$0")/run_pipeline.sh" data "$START" "$END"
