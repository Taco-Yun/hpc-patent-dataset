#!/usr/bin/env bash
# Sample run: steps 1-5 on a small filing-date window, written to data/sample/.
# Use it to check that the whole pipeline works on your machine before the full run.
#
# Usage:
#   bash scripts/run_sample.sh                      # filing date 20150109 (5 patents)
#   bash scripts/run_sample.sh 20150118 20150119    # any other START END window
#   bash scripts/run_sample.sh --offline            # no crawling: steps 2, 4, 5 on the files already in data/sample/
#
# Environment variables (PROXY, HEADLESS, CHROME_VERSION, DOWNLOAD_PDF, PYTHON) are passed on; see run_pipeline.sh.
set -euo pipefail

if [[ "${1:-}" == "--offline" ]]; then
    export OFFLINE=1
    shift
fi
START="${1:-20150109}"
END="${2:-$START}"

bash "$(dirname "$0")/run_pipeline.sh" data/sample "$START" "$END"
