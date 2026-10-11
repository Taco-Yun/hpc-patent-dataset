import csv
from pathlib import Path

# Repository root (the parent of src/). Default data paths are resolved against it,
# so the scripts behave the same no matter which directory they are run from.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"

# Default query settings (the values used to build the HPC dataset).
DEFAULT_START_DATE = "20150101"
DEFAULT_END_DATE = "20251231"


def id_list_filename(start_date, end_date):
    """
    File name of the patent ID list written by step 2 and read by step 3.

    :param start_date: str, YYYYMMDD
    :param end_date: str, YYYYMMDD
    :return: str, e.g. "patent_id_from_20150101_20251231.csv"
    """
    return f"patent_id_from_{start_date}_{end_date}.csv"


def display_path(path):
    """
    Return the path relative to the repository root when it lies inside the repository,
    otherwise the absolute path. Used for log messages and for paths stored in output files,
    so that local absolute paths (e.g. /Users/<name>/...) are not written into the dataset.

    :param path: str or Path
    :return: str
    """
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def read_second_column(csv_path):
    """
    Read the second column of a CSV file and return it as a list (the header row is included).

    :param csv_path: str, the path to the CSV file
    :return: list, all values from the second column
    """
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        return [row[1] for row in reader if row]  # Extract the second column from the CSV
