"""
Step 3 — Download the Google Patents page (HTML) of every patent ID in the list from step 2.

Reads  <data-dir>/interim/patent_id_from_{start}_{end}.csv  (second column; or the file given by --id-list)
Writes <data-dir>/raw/html/{patent_id}.html
       <data-dir>/raw/html/{patent_id}_404.html  when Google Patents returns 404 for that ID
Pages that already exist (including _404 markers) are skipped, so an interrupted run can simply be restarted.
"""
import argparse
import csv
import requests
import os
import time
from tqdm import tqdm  # Enhances readability by providing a visual progress bar.
from utils import DEFAULT_DATA_DIR, DEFAULT_END_DATE, DEFAULT_START_DATE, display_path, id_list_filename, read_second_column


def download_html(idx, base_url, paper_id, proxies, save_dir):
    """
    Download the HTML page corresponding to a specified patent ID and save it locally.

    - If the HTML file has already been downloaded, it will be skipped to avoid redundant requests.
    - If the request returns a 404 status code, a special `_404.html` file will be created.
    - If the download fails due to other reasons, an exception will be raised for the main function to handle retry logic.

    :param idx: int — The index of the current patent being processed (used for progress display).
    :param base_url: str — The fixed base URL of the patent site (e.g., "https://patents.google.com/patent").
    :param paper_id: str — The patent ID (e.g., "US20210012345A1").
    :param proxies: dict — Proxy server settings (e.g., {"http": "http://xxx.xxx.xxx.xxx:port"}).
    :param save_dir: str — The local directory to store HTML files (default is "<data-dir>/raw/html").
    :return: None — On success, the HTML content is saved to a local file.

    """
    # Ensure the directory for storing HTML files exists.
    os.makedirs(save_dir, exist_ok=True)

    # "Set the file path for saving the HTML file.
    file_path = os.path.join(save_dir, f"{paper_id}.html")
    error_file_path = os.path.join(save_dir, f"{paper_id}_404.html")

    # Check if the HTML file for the given patent already exists. If so, skip the download to avoid redundant processing.
    if os.path.exists(file_path) or os.path.exists(error_file_path):
        print(f"Existed, Skipped download: {file_path}")
        return

    # Construct the full URL by concatenating the patent ID with the base URL.
    full_url = f"{base_url}/{paper_id}"
    try:
        # Send an HTTP GET request to fetch the patent page, followed by a 5-second delay.
        response = requests.get(full_url, proxies=proxies, timeout=5)

        # Handle the case where the HTTP response status code is 404, indicating that the requested patent page does not exist.
        if response.status_code == 404:
            file_path = error_file_path  # If the download fails with a 404 error, save the result as  _404.html
        else:
            response.raise_for_status()  # For other HTTP status codes (e.g., 500, 403), an exception will be raised.

        # Save the downloaded HTML content.
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(response.text)

        print(f"Download successful: {full_url} -> {file_path}, Introduce a 1-second delay after each download, and a 30-second pause after every 30 downloads.")

        # Insert a one-second delay between requests to avoid being blocked by the server due to excessive frequency.
        time.sleep(1)

        # Pause the program for 30 seconds after every 30 requests to further reduce the risk of being blocked.
        if idx % 30 == 0:
            time.sleep(30)

    except requests.RequestException as e:
        # If an HTTP-related error occurs (e.g., timeout or connection error), raise the exception for the main function to handle the retry.
        raise RuntimeError(f"Failed to download the file: {full_url}, Error: {e}")
    except Exception as e:
        # If an error of a different type occurs, terminate the program immediately.
        print("An unexpected error occurred:", e)
        exit(e)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="Root data directory (default: <repo>/data). Output goes to <data-dir>/raw/html/.")
    parser.add_argument("--start", default=DEFAULT_START_DATE,
                        help="Start date used in the ID list file name (default: %(default)s).")
    parser.add_argument("--end", default=DEFAULT_END_DATE,
                        help="End date used in the ID list file name (default: %(default)s).")
    parser.add_argument("--id-list", default=None,
                        help="Path to the ID list CSV (default: <data-dir>/interim/patent_id_from_<start>_<end>.csv).")
    parser.add_argument("--proxy", default=None,
                        help="Proxy server, e.g. http://<ip>:<port> (default: no proxy). Recommended to avoid IP blocking.")
    return parser.parse_args()


if __name__ == "__main__":
    """
    This is the main function, which reads the CSV input, executes the download process,
    and oversees the retry mechanism."
    """
    args = parse_args()

    # Specify the path to the CSV file.
    csv_file_path = args.id_list or os.path.join(args.data_dir, "interim", id_list_filename(args.start, args.end))
    save_dir = os.path.join(args.data_dir, "raw", "html")
    print(f"Reading patent IDs from {display_path(csv_file_path)}; saving pages to {display_path(save_dir)}")

    # The fixed base URL for Google Patents, from which all patent pages are retrieved.
    fixed_base_url = "https://patents.google.com/patent"

    # Proxy server configuration (recommended to avoid IP blocking), {"http": "http://<ip>:<poat>", "https": "http://<ip>:<poat>"}
    proxies = {"http": args.proxy, "https": args.proxy} if args.proxy else None

    # Read patent IDs from the CSV file, skipping the first header row.
    paper_ids = read_second_column(csv_file_path)[1:]
    total = len(paper_ids)  # Count the total number of records to process

    # Use tqdm to display a progress bar, allowing users to monitor the current progress.
    with tqdm(total=total, desc="Processing pages") as pbar:
        for idx, key in enumerate(paper_ids, start=1):
            pbar.set_description(f"Processing {idx}/{total}: {key}")  # Update the progress bar display.
            pbar.update(1)
            # Implement a retry mechanism when downloading patent HTML pages, with a maximum of 3 attempts.
            retry_count = 3
            for attempt in range(retry_count):
                try:
                    download_html(idx, fixed_base_url, key, proxies, save_dir)
                    break  # Exit the retry loop upon successful download.
                except RuntimeError as e:
                    print(f"Retry ({attempt + 1}/{retry_count}), Error: {e}")
                    time.sleep(5)  # Wait 5 seconds between retries to avoid temporary server problems.
