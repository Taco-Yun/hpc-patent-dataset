"""
Step 4 — Parse the downloaded patent pages into structured records.

Reads  <data-dir>/raw/html/{patent_id}.html   ({patent_id}_404.html markers are skipped)
Writes <data-dir>/interim/parsed_data.json    JSON Lines: one JSON object per line, one line per patent
With --download-pdf, also downloads each patent PDF to <data-dir>/raw/pdf/ (off by default; PDFs are large).
A page that fails to parse is reported and skipped; the remaining pages are still processed.
"""
import argparse
import os
import json
import re
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urlparse

from utils import DEFAULT_DATA_DIR, display_path


def get_html_filenames(directory):
    """
    List the HTML files in a directory and return their names without the .html extension.

    :param directory: str, the directory to scan
    :return: list[str], file names without the .html extension
    """
    filenames = []

    # Go through every file in the directory
    for file in os.listdir(directory):
        if file.endswith(".html"):  # HTML files only
            filenames.append(os.path.splitext(file)[0])  # Drop the extension

    return filenames


def extract_image_urls(soup):
    """
    Extract the thumbnail and full-size image URLs from the <li itemprop="images"> elements.

    :param soup: bs4.BeautifulSoup, the parsed HTML page
    :return: list[dict], one {"thumbnail": ..., "full_content": ...} per image
    """
    image_data = []

    # Go through every <li> that holds image information
    for li_tag in soup.find_all("li", itemprop="images"):
        thumbnail = li_tag.find("img", itemprop="thumbnail")["src"] if li_tag.find("img",
                                                                                   itemprop="thumbnail") else None
        full_content = li_tag.find("meta", itemprop="full")["content"] if li_tag.find("meta", itemprop="full") else None

        # Keep the image only when both URLs were found
        if thumbnail and full_content:
            image_data.append({"thumbnail": thumbnail, "full_content": full_content})

    return image_data


def download_images(image_data, target_directory):
    """
    Download the full-size images to a directory. Images that already exist are skipped.

    :param image_data: list[dict], output of extract_image_urls()
    :param target_directory: str, the directory to save the images in
    """
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)  # Create the target directory if it does not exist

    for image in image_data:
        full_content_url = image["full_content"]
        # File name = last part of the URL
        file_name = os.path.basename(full_content_url)
        save_path = os.path.join(target_directory, file_name)

        # Skip files that already exist
        if os.path.exists(save_path):
            print(f"↩️ File already exists, skipped: {file_name}")
            continue

        try:
            # Download the image
            response = requests.get(full_content_url, stream=True)
            if response.status_code == 200:
                with open(save_path, "wb") as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
                print(f"✅ Downloaded: {file_name}")
            else:
                print(f"⛔ Could not download: {file_name}, HTTP status code: {response.status_code}")
        except Exception as e:
            print(f"⛔ Error: {e}, URL: {full_content_url}")


def parse_classifications(soup):
    """
    Parse the nested <ul> structure of the "Classifications" section:
    - Level 1: the top-level <ul> -> several <li>, one per classification group
    - Level 2: the inner <ul> of each <li> -> several <li>, one per classification entry

    :param soup: bs4.BeautifulSoup, the parsed HTML page
    :return: list[list[dict]], one list of {"code": ..., "description": ...} per group
    """
    classifications = []

    # Find the top-level <ul> (return an empty list when the page has no "Classifications" heading)
    classifications_h2 = soup.find('h2', string="Classifications")
    if not classifications_h2:
        return classifications
    top_level_ul = classifications_h2.find_next('ul')

    if not top_level_ul:  # No top-level <ul>: return an empty list
        return classifications

    # Go through every <li> of the top-level <ul> (group level)
    for group_li in top_level_ul.find_all('li', recursive=False):
        classification_group = []

        # Find the inner <ul> of the current group
        sub_ul = group_li.find('ul')
        if not sub_ul:
            continue  # No inner <ul>: skip this group

        # Go through every <li> of the inner <ul> -> one classification entry each
        for inner_li in sub_ul.find_all('li', recursive=False):
            code = inner_li.find('span', itemprop='Code')
            description = inner_li.find('span', itemprop='Description')

            # Keep the entry when both code and description were found
            if code and description:
                classification_group.append({
                    'code': code.get_text(strip=True),
                    'description': description.get_text(strip=True)
                })

        # Keep the group when it is not empty
        if classification_group:
            classifications.append(classification_group)

    return classifications


def parse_pdf_download_info(soup, directory):
    """
    Get the PDF download link from the page and build the local path to save it to.

    :param soup: bs4.BeautifulSoup, the parsed HTML page
    :param directory: str, the directory to save the PDF in
    :return: dict, {"path": local path, "pdf_url": download link}; both are None when the page has no PDF link
    """

    pdf_info = {"path": None, "pdf_url": None}

    try:
        # Look for the <meta name="citation_pdf_url"> tag
        meta_tag = soup.find('meta', attrs={'name': 'citation_pdf_url'})
        if not meta_tag or 'content' not in meta_tag.attrs:
            print("Not found: <meta> tag citation_pdf_url")
            return pdf_info

        # PDF download link
        pdf_url = meta_tag['content']
        pdf_info["pdf_url"] = pdf_url

        # File name = last part of the URL path
        parsed_url = urlparse(pdf_url)
        file_name = os.path.basename(parsed_url.path)

        # Full local path
        output_path = os.path.join(directory, file_name)

        pdf_info["path"] = output_path

        return pdf_info
    except Exception as e:
        print(f"Failed to build the PDF download path: {e}")
        return pdf_info

def download_and_save_file(url, output_path):
    """
    Download a file and save it to the given path. Files that already exist are skipped.

    :param url: str, download link
    :param output_path: str, full path of the file to save (including the file name)
    :return: bool, True on success or when the file already exists, False on failure
    """
    # Skip files that already exist
    if os.path.exists(output_path):
        print(f"File already exists, skipped: {output_path}")
        return True

    try:
        print(f"Downloading: {url}")

        # Send the GET request
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Make sure the request succeeded

        # Write the file in binary mode
        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"Downloaded and saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def parse_inventors(soup):
    """
    Extract the names of all inventors.

    :param soup: bs4.BeautifulSoup, the parsed HTML page
    :return: list[str], inventor names
    """
    try:
        # Find the <dt> whose text is 'Inventor'
        inventor_dt = soup.find('dt', string='Inventor')
        if not inventor_dt:
            print("Not found: <dt> tag 'Inventor'")
            return []

        # All <dd itemprop="inventor"> tags after that <dt>
        inventor_dds = inventor_dt.find_all_next('dd', attrs={'itemprop': 'inventor'})

        # Text of each <dd>
        inventors = [dd.get_text(strip=True) for dd in inventor_dds]

        return inventors
    except Exception as e:
        print(f"Error while parsing inventors: {e}")
        return []

def parse_current_assignee(soup):
    """
    Extract the names of all current assignees.

    :param soup: bs4.BeautifulSoup, the parsed HTML page
    :return: list[str], current assignee names
    """
    try:
        # Find the <dt> whose text starts with "Current Assignee"
        current_assignee_dt = soup.find('dt', string=lambda text: text and text.startswith('Current Assignee'))
        if not current_assignee_dt:
            print("Not found: <dt> tag 'Current Assignee'")
            return []

        # All <dd itemprop="assigneeCurrent"> tags after that <dt>
        current_assignee_dds = current_assignee_dt.find_all_next('dd', attrs={'itemprop': 'assigneeCurrent'})

        # Text of each <dd>
        assignees = [dd.get_text(strip=True) for dd in current_assignee_dds]

        return assignees
    except Exception as e:
        print(f"Error while parsing current assignees: {e}")
        return []

def parse_events(soup):
    """
    Parse every <dd itemprop="events"> element and collect the itemprop attributes and texts of its child tags.

    :param soup: bs4.BeautifulSoup, the parsed HTML page
    :return: list[dict], one dict per event
    """
    try:
        # All <dd itemprop="events"> tags
        events_dds = soup.find_all('dd', attrs={'itemprop': 'events'})

        events = []

        # Go through every <dd>
        for dd in events_dds:
            event_data = {}  # Data of a single event

            # Go through every child tag of the <dd> and keep the ones with an itemprop
            for tag in dd.find_all(True):  # True = all child tags
                itemprop = tag.get('itemprop')
                if itemprop:
                    # When an itemprop appears more than once, store its values as a list
                    if itemprop in event_data:
                        if isinstance(event_data[itemprop], list):
                            event_data[itemprop].append(tag.get_text(strip=True))
                        else:
                            event_data[itemprop] = [event_data[itemprop], tag.get_text(strip=True)]
                    else:
                        event_data[itemprop] = tag.get_text(strip=True)

            events.append(event_data)  # Add the event to the result

        return events
    except Exception as e:
        print(f"Error while parsing events: {e}")
        return []

def get_ep_description(soup) -> json:
    description_section = soup.select_one('div[itemprop="content"]')
    output_text = []
    if description_section:
        # Go through the headings and the <li> paragraphs
        for element in description_section.find_all(['heading', 'li']):
            if element.name == 'heading':
                # Heading text
                output_text.append("\n" + element.get_text(strip=True).upper() + "\n")
            elif element.name == 'li':
                # Paragraph number from <para-num>
                para_num = element.select_one('para-num')
                para_text = para_num.get('num', '').strip() if para_num else ""

                # Paragraph text from <div class="description-line">
                description_text = element.select_one('.description-line')
                description_text = description_text.get_text(strip=True) if description_text else ""

                # Paragraph number + paragraph text
                output_text.append(f"{para_text} {description_text}")
    return {"CROSS-REFERENCE TO RELATED APPLICATION": "\n".join(output_text)}


def get_us_description_v2(soup) -> json:
    def normalize_text(node) -> str:
        """
        Get the text of a node with a space at tag boundaries, so that words from adjacent tags
        are not glued together (e.g. "204can", "104FIG"), and collapse consecutive whitespace into one space.
        """
        text = node.get_text(" ", strip=True)  # Key point: use a space as the separator
        text = re.sub(r"\s+", " ", text).strip()
        return text

    description_section = soup.select_one('section[itemprop="description"] div[itemprop="content"]')

    description_dict = {
        "CROSS-REFERENCE TO RELATED APPLICATION": "N/A",
        "BACKGROUND": "N/A",
        "SUMMARY": "N/A",
        "DETAILED DESCRIPTION": "N/A",
        "BRIEF DESCRIPTION OF THE DRAWINGS": "N/A"
    }

    current_heading = None

    if description_section:
        for element in description_section.find_all(['heading', 'div']):
            if element.name == 'heading':
                heading_text = normalize_text(element).upper()
                if heading_text in description_dict:
                    current_heading = heading_text
                    description_dict[current_heading] = ""
                else:
                    if current_heading:
                        # A non-standard heading is kept as part of the current section's text
                        description_dict[current_heading] += heading_text + "\n"

            elif element.name == 'div' and 'description-paragraph' in (element.get('class') or []):
                paragraph_text = normalize_text(element)
                if current_heading:
                    description_dict[current_heading] += paragraph_text + "\n"

    return {k: v.strip() for k, v in description_dict.items()}


def parse_claims(soup):
    """
    Parse the `itemprop="claims"` section and return the numbered claims.

    :param soup: bs4.BeautifulSoup, the parsed HTML page
    :return: list[str], the claims with their numbers; empty when the number of parsed claims
             differs from the claim count shown on the page
    """
    try:
        # The section with itemprop="claims"
        claims_section = soup.find('section', attrs={'itemprop': 'claims'})
        if not claims_section:
            raise ValueError("Section itemprop='claims' not found")

        # Claim count shown on the page, used to check that parsing is complete
        count_element = claims_section.find('h2').find('span', attrs={'itemprop': 'count'})
        count = int(count_element.get_text(strip=True)) if count_element else 0

        # All text of the section
        claims_content = claims_section.get_text(separator="\n", strip=True)

        # Split the text at the claim numbers `1. 2. ... count.` and keep the numbers
        import re
        matches = re.findall(r'(\d+\..*?)(?=\n\d+\.|\Z)', claims_content, re.S)

        # Check the result against the claim count
        if count and len(matches) != count:
            raise ValueError(f"Number of parsed claims ({len(matches)}) differs from the claim count ({count})")

        return [match.strip() for match in matches]
    except Exception as e:
        print(f"Parsing error: {e}")
        return []

def parse_citations(soup, h2_keyword):
    """
    Parse the table under the first <h2> heading that contains h2_keyword, with normalised column names as keys.

    :param soup: bs4.BeautifulSoup, the parsed HTML page
    :param h2_keyword: str, text to look for in the <h2> heading (e.g. "Non-Patent Citations" or "Patent Citations")
    :return: list[dict], one dict per table row
    """
    try:
        # The <h2> heading that contains the keyword
        section = soup.find('h2', string=lambda x: x and h2_keyword in x)
        if not section:
            raise ValueError(f"<h2> heading containing '{h2_keyword}' not found")

        # Try to read the row count from the heading, e.g. "Patent Citations (12)"
        total_count = None
        try:
            if "(" in section.text:
                total_count = int(section.text.split("(")[-1].rstrip(")"))
        except ValueError:
            print(f"Warning: could not read the count in '{h2_keyword}'; the row count check is skipped.")

        # The table that follows the heading
        table = section.find_next("table")
        if not table:
            raise ValueError(f"Table not found (under '{h2_keyword}')")

        # Keys from <thead>, lower-cased, spaces replaced by underscores
        thead = table.find("thead")
        if thead:
            keys = [
                th.text.strip().lower().replace(" ", "_")
                for th in thead.find_all("th")
            ]
        else:  # Tables without <thead> (probably a single column)
            keys = ["title"]

        # Rows of <tbody>
        tbody = table.find("tbody")
        rows = tbody.find_all("tr", recursive=False)
        if total_count is not None and len(rows) != total_count:
            print(f"Warning: number of rows ({len(rows)}) differs from the count in the heading ({total_count})")

        # One dict per <tr>
        data = []
        for row in rows:
            values = []
            for td in row.find_all("td", recursive=False):
                # Cells with a link: prefer the publication number
                if td.find("a"):
                    span = td.find("span", {"itemprop": "publicationNumber"})
                    if span:
                        values.append(span.text.strip())
                    else:
                        values.append(td.get_text(strip=True))
                else:
                    values.append(td.get_text(strip=True))

            # Pair the keys with the values
            entry = {keys[i]: values[i] for i in range(min(len(keys), len(values)))}
            data.append(entry)

        return data

    except Exception as e:
        print(f"Parsing error: {e}")
        return []



def parse_html_data(pdf_directory, html_content, filename, download_pdf=False):
    """
    Parse the HTML of a single patent page.

    :param pdf_directory: str, directory for the PDF (pdf_info.path points here)
    :param html_content: str, the HTML content
    :param filename: str, the file name without extension (= the patent ID)
    :param download_pdf: bool, whether to download the PDF (default: no)
    :return: dict, the parsed record, e.g. {"title": "Display panel and method of fabricating the same", ...}
    """
    soup = BeautifulSoup(html_content, "html.parser")

    ## title
    meta_title = soup.select_one('meta[name="DC.title"]')
    name = meta_title.get('content', '').strip() if meta_title else "N/A"

    ## abstract
    abstract_section = soup.select_one('section[itemprop="abstract"] div[itemprop="content"]')
    abstract_text = abstract_section.select_one('div.abstract').get_text(strip=True) if abstract_section else "N/A"

    ## image download
    image_urls = extract_image_urls(soup)
    # download_images(image_urls, directory)

    ## classifications
    classifications = parse_classifications(soup)

    ## blue table
    ### patent id
    patent_id = filename
    ### download pdf (off by default; enable with --download-pdf)
    pdf_info = parse_pdf_download_info(soup, pdf_directory)
    if download_pdf and pdf_info.get("pdf_url", None) is not None:
        os.makedirs(pdf_directory, exist_ok=True)
        download_and_save_file(pdf_info['pdf_url'], pdf_info['path'])
    # Store a repo-relative path in the output, never a local absolute path
    if pdf_info.get("path") is not None:
        pdf_info["path"] = display_path(pdf_info["path"])
    ### inventor
    inventors = parse_inventors(soup)
    ### current assignee
    current_assignee = parse_current_assignee(soup)
    ### worldwide application = Applications Claiming Priority
    ### event
    events = parse_events(soup)

    ## description (add or remove keywords here as needed)
    if "EP" in filename:
        description = get_ep_description(soup)
    elif "US" in filename:
        description = get_us_description_v2(soup)
    else:
        description = "N/A"

    ### claims(count)
    claims = parse_claims(soup)

    ### Patent citations (count)
    patent_citations = parse_citations(soup, "Patent Citations")
    ### Non-patent citations (count)
    non_patent_citations = parse_citations(soup, "Non-Patent Citations")
    ### Cited by (count) fix
    cited_by = parse_citations(soup, "Families Citing this family")
    ### Similar Documents fix
    similar_documents = parse_citations(soup, "Similar Documents")
    ### Parent Applications (count)
    parent_applications = parse_citations(soup, "Parent Applications")
    ### Applications Claiming Priority (count)
    applications_claiming_priority = parse_citations(soup, "Applications Claiming Priority")
    ### Legal events fix
    legal_events = parse_citations(soup, "Legal Events")

    return {
        "title": name,
        "abstract": abstract_text,
        "patent_id": patent_id,
        "image_urls": image_urls,
        "classifications": classifications,
        "pdf_info": pdf_info,
        "inventors": inventors,
        "current_assignee": current_assignee,
        "events": events,
        "description": description,
        "claims": claims,
        "patent_citations": patent_citations,
        "non_patent_citations": non_patent_citations,
        "cited_by": cited_by,
        "similar_documents": similar_documents,
        "parent_applications": parent_applications,
        "applications_claiming_priority": applications_claiming_priority,
        "legal_events": legal_events,
    }


def process_html_files(directory, json_output, pdf_directory, download_pdf=False):
    """
    Parse every HTML file in a directory and save the results as JSON Lines (JSON output only).

    - Written in append mode, one JSON object per line
    - Files are processed in file-name order, so the output order does not depend on the operating system
    - `{patent_id}_404.html` files (404 markers from step 3) are not parsed
    - When a page fails to parse, the error is printed and only that page is skipped
    - A tqdm progress bar shows the progress

    :param directory: str, directory that contains the HTML files
    :param json_output: str, path of the output JSON file
    :param pdf_directory: str, directory for the PDFs
    :param download_pdf: bool, whether to download the PDFs
    """

    # **Empty the output file**
    open(json_output, "w").close()

    # All HTML file names, without the 404 markers
    all_filenames = sorted(get_html_filenames(directory))
    filenames = [name for name in all_filenames if not name.endswith("_404")]
    skipped_404 = len(all_filenames) - len(filenames)
    total_files = len(filenames)
    written = 0
    failed = []

    # **Progress bar**
    with tqdm(total=total_files, desc="Processing HTML files") as pbar, \
            open(json_output, "a", encoding="utf-8") as json_file:

        for idx, filename in enumerate(filenames, start=1):
            pbar.set_description(f"Processing {idx}/{total_files}: {filename}")  # Update the progress bar
            pbar.update(1)
            file_path = os.path.join(directory, f"{filename}.html")
            try:
                # Read the HTML file
                with open(file_path, "r", encoding="utf-8") as file:
                    html_content = file.read()

                # Parse the HTML
                parsed_data = parse_html_data(pdf_directory, html_content, filename, download_pdf)
            except Exception as e:
                # Skip only this page; keep processing the others
                print(f"⛔ Failed to parse, skipped: {filename} -> {e!r}")
                failed.append(filename)
                continue

            # **Write one JSON object per line**
            json_file.write(json.dumps(parsed_data, ensure_ascii=False) + "\n")
            written += 1

            # For testing
            # if idx == 3:
            #     break

    print(f"✅ Done! Saved {written} records to {display_path(json_output)}.")
    if skipped_404:
        print(f"↩️ Skipped {skipped_404} 404 marker file(s) (_404.html).")
    if failed:
        print(f"⛔ {len(failed)} page(s) failed to parse: {', '.join(failed)}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="Root data directory (default: <repo>/data).")
    parser.add_argument("--download-pdf", action="store_true",
                        help="Also download each patent PDF to <data-dir>/raw/pdf/ (default: off).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Input HTML directory and PDF directory
    html_directory = os.path.join(args.data_dir, "raw", "html")
    pdf_directory = os.path.join(args.data_dir, "raw", "pdf")

    # Output JSON file
    output_dir = os.path.join(args.data_dir, "interim")
    os.makedirs(output_dir, exist_ok=True)
    json_output_file = os.path.join(output_dir, "parsed_data.json")

    # Parse the HTML files and save the results
    process_html_files(html_directory, json_output_file, pdf_directory, args.download_pdf)
