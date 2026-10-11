#!/usr/bin/env bash
# Run steps 1-5 of the pipeline for one filing-date window.
# Normally called through run_sample.sh or run_full.sh.
#
# Usage: bash scripts/run_pipeline.sh DATA_DIR START_DATE END_DATE
#
# Optional environment variables:
#   OFFLINE=1          skip the two crawling steps (1 and 3); only steps 2, 4, 5 run on files already on disk
#   PROXY=URL          proxy for steps 1 and 3, e.g. http://<ip>:<port>
#   HEADLESS=1         run Chrome without a window in step 1
#   CHROME_VERSION=N   installed Chrome major version for step 1 (default: auto-detect)
#   DOWNLOAD_PDF=1     also download patent PDFs in step 4
#   PYTHON="cmd"       Python command (default: "uv run python")
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: bash scripts/run_pipeline.sh DATA_DIR START_DATE END_DATE" >&2
    exit 2
fi
DATA_DIR="$1"
START="$2"
END="$3"

cd "$(dirname "$0")/.."
read -r -a PY <<< "${PYTHON:-uv run python}"

STEP1_ARGS=(--data-dir "$DATA_DIR" --start "$START" --end "$END")
STEP3_ARGS=(--data-dir "$DATA_DIR" --start "$START" --end "$END")
STEP4_ARGS=(--data-dir "$DATA_DIR")
if [[ -n "${PROXY:-}" ]]; then
    STEP1_ARGS+=(--proxy "$PROXY")
    STEP3_ARGS+=(--proxy "$PROXY")
fi
if [[ "${HEADLESS:-0}" == "1" ]]; then STEP1_ARGS+=(--headless); fi
if [[ -n "${CHROME_VERSION:-}" ]]; then STEP1_ARGS+=(--chrome-version "$CHROME_VERSION"); fi
if [[ "${DOWNLOAD_PDF:-0}" == "1" ]]; then STEP4_ARGS+=(--download-pdf); fi

echo "== Data directory: $DATA_DIR | filing dates: $START - $END =="

if [[ "${OFFLINE:-0}" == "1" ]]; then
    echo "== [1/5] Skipped (OFFLINE=1)"
else
    echo "== [1/5] Fetch search result pages"
    "${PY[@]}" src/step1_fetch_search_results.py "${STEP1_ARGS[@]}"
fi

echo "== [2/5] Extract patent IDs"
"${PY[@]}" src/step2_extract_patent_ids.py --data-dir "$DATA_DIR" --start "$START" --end "$END"

if [[ "${OFFLINE:-0}" == "1" ]]; then
    echo "== [3/5] Skipped (OFFLINE=1)"
else
    echo "== [3/5] Download patent pages"
    "${PY[@]}" src/step3_download_patent_pages.py "${STEP3_ARGS[@]}"
fi

echo "== [4/5] Parse patent pages"
"${PY[@]}" src/step4_parse_patent_pages.py "${STEP4_ARGS[@]}"

echo "== [5/5] Build the CSV dataset"
"${PY[@]}" src/step5_build_hpc_dataset.py --data-dir "$DATA_DIR"

echo "== Done. Final dataset: $DATA_DIR/processed/HPC_15to25_v2.csv"
