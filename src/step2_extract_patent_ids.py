"""
Step 2 — Extract the patent IDs from the search result pages saved by step 1.

Reads  <data-dir>/raw/id_list_json/{YYYYMMDD}_{page}.json  (only files whose date is within [--start, --end])
Writes <data-dir>/interim/patent_id_from_{start}_{end}.csv  with columns: number, patent_number
IDs are de-duplicated and sorted; "number" is a 1-based running index.
"""
import argparse
import os
import json
import csv

from utils import DEFAULT_DATA_DIR, DEFAULT_END_DATE, DEFAULT_START_DATE, display_path, id_list_filename


def parse_patent_ids(start_date, end_date, folder, output_file=None):
    if output_file is None:
        output_file = id_list_filename(start_date, end_date)

    result_ids = []
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".json"):
            continue
        date_prefix = filename.split("_")[0]
        if (date_prefix < start_date) or (date_prefix > end_date):
            continue

        file_path = os.path.join(folder, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            cluster = data.get("results", {}).get("cluster", [])
            for group in cluster:
                for item in group.get("result", []):
                    patent_id = item.get("id", "")
                    if "/US" in patent_id:
                        number = patent_id.split("patent/")[1].split("/")[0]
                        result_ids.append(number)

    # Remove duplicates and sort the list
    result_ids = sorted(set(result_ids))

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["number", "patent_number"])
        for idx, pid in enumerate(result_ids, start=1):
            writer.writerow([idx, pid])

    print(f"Successfully saved {len(result_ids)} patent IDs to {display_path(output_file)}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="Root data directory (default: <repo>/data).")
    parser.add_argument("--start", default=DEFAULT_START_DATE, help="First filing date, YYYYMMDD (default: %(default)s).")
    parser.add_argument("--end", default=DEFAULT_END_DATE, help="Last filing date, YYYYMMDD, inclusive (default: %(default)s).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    folder = os.path.join(args.data_dir, "raw", "id_list_json")
    output_dir = os.path.join(args.data_dir, "interim")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, id_list_filename(args.start, args.end))
    parse_patent_ids(args.start, args.end, folder, output_file)
