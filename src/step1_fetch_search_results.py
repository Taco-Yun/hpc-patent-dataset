"""
Step 1 — Fetch Google Patents search results, one filing date at a time.

For every filing date in [--start, --end], query Google Patents and save each result page as JSON:
    <data-dir>/raw/id_list_json/{YYYYMMDD}_{page:03d}.json

A real Chrome browser is driven by undetected-chromedriver (see README: "Browser automation notes").
Files that already exist are skipped, so an interrupted run can simply be restarted.
"""
import argparse
import requests
import json
import os
import time
import re
import undetected_chromedriver as uc
from tqdm import tqdm
from datetime import datetime, timedelta

from utils import DEFAULT_DATA_DIR, DEFAULT_END_DATE, DEFAULT_START_DATE, display_path

# A fixed base URL for Google Patents, from which all search result pages are retrieved.
BASE_URL = "https://patents.google.com/xhr/query"


def launch_uc_chrome(headless=False, proxy=None, version_main=None):
    proxy = proxy or {}
    chrome_options = uc.ChromeOptions()
    chrome_options.add_argument('--disable-popup-blocking')
    chrome_options.add_argument("--enable-javascript")
    chrome_options.add_argument("--disable-extensions")

    if proxy:
        chrome_options.add_argument(f'--proxy-server={proxy.get("http") or proxy.get("https")}')

    # version_main=None lets undetected-chromedriver detect the installed Chrome major version.
    driver = uc.Chrome(options=chrome_options, headless=headless, enable_cdp_events=True, use_subprocess=True,
                       version_main=version_main)
    driver.requests = []
    driver.responses = []
    print("Initialization in progress, Please wait for 20 seconds.")
    time.sleep(20)  # Wait for the browser to initialise
    print("Initialization complete!")

    return driver

def generate_date_range(start_date, end_date):
    date_list = []
    current = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")

    while current <= end:
        date_list.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    return date_list

def extract_json_from_html(html_text):
    try:
        # Extract the JSON string enclosed in '<pre>' tags using regular expressions.
        match = re.search(r"<pre>(.*?)</pre>", html_text, re.DOTALL)
        if not match:
            raise ValueError("No <pre>...</pre> block found.")

        json_text = match.group(1)
        parsed_json = json.loads(json_text)
        return parsed_json
    except Exception as e:
        print(f"Failed to extract JSON: {e}")
        return None



def download_json(chrome, base_query, date, file_dir):
    page = 0
    while True:
        filename = os.path.join(file_dir, f"{date}_{str(page).zfill(3)}.json")

        # Skip the operation if the file already exists.
        if os.path.exists(filename):
            print(f"File already exists. Skipping!{filename}")
            page += 1
            continue

        # Include parameters for date and page number
        query = base_query.copy()
        query["after"] = f"filing:{date}"
        query["before"] = f"filing:{date}"
        if page != 0:
            query["page"] = str(page)

        # Encode the string as a URL
        url_params = "&".join([f"{k}={v}" for k, v in query.items()])
        full_url = f"{BASE_URL}?url={requests.utils.quote(url_params)}"

        print(f"Fetching: {full_url}")
        chrome.get(full_url)
        time.sleep(1)
        html = chrome.page_source
        data = extract_json_from_html(html)
        cluster = data.get("results", {}).get("cluster", [])

        if not cluster or not cluster[0].get("result"):
            print(f"No more results on {date} page {page}")
            break

        # Save the file.
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        total_pages = data["results"].get("total_num_pages", 0)
        if page >= total_pages:
            break
        print(f"Completed{filename}, Pausing for 1 second")
        time.sleep(1)

        page += 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="Root data directory (default: <repo>/data). Output goes to <data-dir>/raw/id_list_json/.")
    parser.add_argument("--start", default=DEFAULT_START_DATE, help="First filing date, YYYYMMDD (default: %(default)s).")
    parser.add_argument("--end", default=DEFAULT_END_DATE, help="Last filing date, YYYYMMDD, inclusive (default: %(default)s).")
    parser.add_argument("--query", default="(HPC)", help="Google Patents search query (default: %(default)s).")
    parser.add_argument("--country", default="US", help="Country filter (default: %(default)s).")
    parser.add_argument("--status", default="GRANT", help="Status filter (default: %(default)s).")
    parser.add_argument("--language", default="ENGLISH", help="Language filter (default: %(default)s).")
    parser.add_argument("--type", default="PATENT", help="Document type filter (default: %(default)s).")
    parser.add_argument("--proxy", default=None,
                        help="Proxy server, e.g. http://<ip>:<port> (default: no proxy). Recommended to avoid IP blocking.")
    parser.add_argument("--headless", action="store_true",
                        help="Run Chrome without a window (default: off, a Chrome window is opened).")
    parser.add_argument("--chrome-version", type=int, default=None,
                        help="Major version of the installed Chrome, i.e. the number before the first dot "
                             "in chrome://version (default: auto-detect). "
                             "Set it when the driver reports a Chrome version mismatch.")
    return parser.parse_args()


if __name__ == "__main__":
    """
    Handles the download logic and manages the retry mechanism.
    """
    args = parse_args()

    # Constant parameters
    file_dir = os.path.join(args.data_dir, "raw", "id_list_json")
    base_query = {
        "q": args.query,
        "country": args.country,
        "status": args.status,
        "language": args.language,
        "type": args.type
    }

    # Set the number of days for data processing.
    date_list = generate_date_range(args.start, args.end)
    total = len(date_list)

    os.makedirs(file_dir, exist_ok=True)
    print(f"Saving search result pages to {display_path(file_dir)}")

    # Proxy Server Configuration (Recommended to prevent IP address blocking), {"http": "http://<ip>:<poat>", "https": "http://<ip>:<poat>"}
    proxies = {"http": args.proxy, "https": args.proxy} if args.proxy else None

    chrome = launch_uc_chrome(headless=args.headless, proxy=proxies, version_main=args.chrome_version)

    # Use tqdm to display a progress bar, allowing users to track the current progress.
    with tqdm(total=total, desc="Processing pages") as pbar:
        for idx, date in enumerate(date_list, start=1):
            pbar.set_description(f"Processing {idx}/{total}: {date}")  # Update progress bar display
            pbar.update(1)
            retry_count = 3
            for attempt in range(retry_count):
                try:
                    download_json(chrome, base_query, date, file_dir)
                    break  # Exit the retry loop upon successful download.
                except RuntimeError as e:
                    print(f"Retry ({attempt + 1}/{retry_count}), Error: {e}")
                    time.sleep(5)  # Wait 5 seconds between each retry to avoid temporary server issues.
                except Exception as e:
                    print(f"Retry ({attempt + 1}/{retry_count}), Unexpected error on {date}: {e!r}")
                    time.sleep(5)
